"""File-driven preprocessing workflow with dynamic pipeline execution."""

from __future__ import annotations

__all__ = [
    "DynamicPreprocessingWorkflow",
    "dynamic_workflow",
]

from typing import Any, Literal, cast
from collections.abc import Mapping, Sequence
from pathlib import Path

from niiflow.preproc.staging import StagedEntry
from niiflow.preproc.pipelines import dynamic_pipeline

from .mixins import SupportsFileDiscovery, SupportsStaging, InputData
from .plan import RunPlan
from .workflow import PlannableWorkflow
from .workflow_factory import create_workflow


class DynamicPreprocessingWorkflow(
    SupportsFileDiscovery,
    SupportsStaging,
    PlannableWorkflow,
):
    """File-driven preprocessing workflow with dynamic pipeline execution.

    This workflow discovers active files, stages per-entry pipeline parameters,
    and executes a dynamically constructed preprocessing pipeline for each
    staged entry.

    Each entry is organised around an **active file**: the canonical path that
    anchors one unit of work, such as a subject's T1w structural MRI. Stagers
    use this active file to resolve file-linked values in the per-entry
    parameter mapping before execution.

    ``pipeline_params`` define the preprocessing pipeline to run, including the
    stage order, stage parameters, and any file-linked fields that should be
    resolved during staging. In the current implementation, this dictionary is
    copied to every entry before stagers resolve per-entry values. Pass ``None``
    only when the instance will solely :meth:`run_plan` a previously staged plan
    (entry params then come from the plan).

    Active-file discovery is driven by :data:`~niiflow.preproc.workflows.types.InputData`
    passed to :meth:`plan`: an explicit file path, a ``search`` or ``from_file``
    mapping, or a sequence of those.

    The explicit API is:

    - :meth:`plan`: discover active files and build a staged run plan.
    - :meth:`run_plan`: execute an existing run plan.
    - :meth:`run`: execute staged entries (from :class:`~niiflow.preproc.workflows.workflow.ProcessingWorkflow`).

    For a single entry point that instantiates the workflow and selects
    plan / execute / from-plan modes, use :func:`dynamic_workflow`.

    Args:
        pipeline_params: Pipeline specification copied into entry parameters before
            staging. Usually contains ``steps`` plus any file-linked fields resolved
            by stagers. Required as an explicit argument; use ``None`` only for
            execute-from-plan usage where staging is not performed.
        staging_params: Optional stager specifications passed to
            :func:`~niiflow.preproc.staging.create_stager`. When omitted, entries
            are built from ``pipeline_params`` without running stagers.
        num_workers: Worker count (default ``1``, serial). Use ``"auto"`` for one
            worker per CPU core, or an integer ``> 1`` for a fixed pool size.
        logs_root: Directory for log files; ``None`` logs to console only.
        main_logs: Emit main workflow logs.
        status_logs: Emit per-entry status lines.
        worker_logs: Emit worker-process logs when ``num_workers > 1``.
        dev_mode: Enable debug-level worker logging.
        timeout: Soft per-entry limit in seconds.
    """

    def __init__(
        self,
        *,
        pipeline_params: dict[str, Any] | None,
        staging_params: dict[str, Any] | Sequence[dict[str, Any]] | None = None,
        num_workers: int | Literal["auto"] = 1,
        logs_root: Path | str | None = None,
        main_logs: bool = True,
        status_logs: bool = True,
        worker_logs: bool = True,
        dev_mode: bool = False,
        timeout: float | None = None,
    ) -> None:
        PlannableWorkflow.__init__(
            self,
            num_workers=num_workers,
            logs_root=logs_root,
            main_logs=main_logs,
            status_logs=status_logs,
            worker_logs=worker_logs,
            dev_mode=dev_mode,
            timeout=timeout,
        )
        if pipeline_params is not None and not isinstance(pipeline_params, dict):
            raise TypeError("`pipeline_params` must be a dictionary or None")
        self.configure_staging(
            staging_params=staging_params, entry_params=pipeline_params
        )

    def plan(
        self,
        inputs: InputData,
        *,
        save_filepaths_to: Path | str | None = None,
        save_plan_to: Path | str | None = None,
    ) -> RunPlan:
        """Build a staged preprocessing run plan from run inputs.

        Args:
            inputs: Run inputs accepted by
                :meth:`~niiflow.preproc.workflows.mixins.SupportsFileDiscovery.discover_active_files`.
            save_filepaths_to: Optional ``.txt`` path for discovered active file paths.
            save_plan_to: Optional ``.duckdb`` / ``.json`` path for the staged run plan.

        Returns:
            A run plan containing one staged entry per discovered active file.

        Notes:
            This method performs discovery and staging only. It does not execute
            the preprocessing pipeline.
        """
        active_files = self.discover_active_files(inputs, save_to=save_filepaths_to)
        return self.stage_active_files(active_files, save_to=save_plan_to)

    @staticmethod
    def process_single(entry: StagedEntry) -> None:
        """Run the configured preprocessing pipeline for one staged entry."""
        dynamic_pipeline(entry.params, run_id=str(entry.active))


def dynamic_workflow(
    *,
    settings: Mapping[str, Any] | None = None,
    inputs: InputData | None = None,
    save_filepaths_to: Path | str | None = None,
    save_plan_to: Path | str | None = None,
    from_plan: Path | str | None = None,
    plan_only: bool = False,
    dry_run: bool = False,
) -> RunPlan:
    """Instantiate and execute a :class:`DynamicPreprocessingWorkflow`.

    This is the orchestration entry point for scripts and higher-level drivers,
    analogous to :func:`~niiflow.preproc.pipelines.dynamic_pipeline`. The workflow
    class itself stays limited to ``plan`` / ``run_plan`` / ``run``.

    Modes (checked in order):

    * ``from_plan`` — load a saved plan and execute it (or print it when
      ``dry_run``). Planning is skipped; ``pipeline_params`` is forced to
      ``None``. ``settings`` may be omitted. Extra ``inputs`` /
      ``pipeline_params`` are ignored with a log message. Save paths and
      ``plan_only`` must not be set.
    * ``plan_only`` — build (and optionally save) a plan from ``inputs`` without
      executing. Requires ``settings`` with non-None ``pipeline_params``. With
      ``dry_run``, the plan is printed and not saved.
    * default — plan from ``inputs`` then execute. Requires ``settings`` with
      non-None ``pipeline_params``. With ``dry_run``, plan without saving,
      print, and skip execution.

    Args:
        settings: Keyword arguments forwarded to
            :class:`DynamicPreprocessingWorkflow`. Required unless ``from_plan``
            is set; must include non-None ``pipeline_params`` for planning modes.
            When ``from_plan`` is set, may be omitted (treated as ``{}``) and any
            ``pipeline_params`` entry is cleared to ``None``.
        inputs: Run inputs for planning modes. Required unless ``from_plan`` is set.
        save_filepaths_to: Optional ``.txt`` path for discovered active files.
        save_plan_to: Optional ``.duckdb`` / ``.json`` path for the staged plan.
        from_plan: Path to a saved plan to load and execute (skips planning).
        plan_only: When ``True``, stop after planning (do not execute).
        dry_run: When ``True``, print the plan via :meth:`RunPlan.view` and do not
            save or execute.

    Returns:
        The built or loaded :class:`~niiflow.preproc.workflows.plan.RunPlan`.

    Raises:
        ValueError: If mode arguments conflict or required ``settings`` / ``inputs``
            / ``pipeline_params`` are missing.
        TypeError: If ``settings`` is not a mapping or cannot bind to the workflow
            constructor.
    """
    if from_plan is not None:
        if plan_only:
            raise ValueError("`from_plan` cannot be combined with `plan_only`")
        if save_filepaths_to is not None or save_plan_to is not None:
            raise ValueError(
                "`from_plan` cannot be combined with `save_filepaths_to` or "
                "`save_plan_to`"
            )
        if settings is None:
            settings = {}
        elif not isinstance(settings, Mapping):
            raise TypeError(
                f"`settings` must be a mapping or None, got {type(settings).__name__}"
            )
        else:
            settings = dict(settings)

        workflow = cast(
            DynamicPreprocessingWorkflow,
            create_workflow(
                "DynamicPreprocessingWorkflow",
                {**settings, "pipeline_params": None},
            ),
        )
        if inputs is not None:
            workflow.log(
                "Ignoring `inputs` because `from_plan` is set; the loaded plan "
                "defines the worklist.",
                level="info",
            )
        if "pipeline_params" in settings:
            workflow.log(
                "Ignoring `pipeline_params` from settings because `from_plan` is "
                "set; entry params come from the loaded plan.",
                level="info",
            )

        run_plan = RunPlan.load(from_plan)
        if dry_run:
            print(run_plan.view())
        else:
            workflow.run_plan(run_plan)
        return run_plan

    if settings is None:
        raise ValueError("`settings` is required unless `from_plan` is set")
    if not isinstance(settings, Mapping):
        raise TypeError(
            f"`settings` must be a mapping or None, got {type(settings).__name__}"
        )
    settings = dict(settings)

    if inputs is None:
        raise ValueError("`inputs` is required unless `from_plan` is set")
    if settings.get("pipeline_params") is None:
        raise ValueError(
            "`settings` must include a non-None `pipeline_params` unless "
            "`from_plan` is set"
        )

    workflow = cast(
        DynamicPreprocessingWorkflow,
        create_workflow("DynamicPreprocessingWorkflow", settings),
    )
    run_plan = workflow.plan(
        inputs,
        save_filepaths_to=None if dry_run else save_filepaths_to,
        save_plan_to=None if dry_run else save_plan_to,
    )
    if dry_run:
        print(run_plan.view())
    elif not plan_only:
        workflow.run_plan(run_plan)
    return run_plan
