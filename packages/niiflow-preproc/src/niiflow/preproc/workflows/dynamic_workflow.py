"""File-driven preprocessing workflow with dynamic pipeline execution."""

from __future__ import annotations

__all__ = [
    "DynamicProcessingWorkflow",
    "dynamic_workflow",
]

from typing import Any, Literal, cast
from collections.abc import Collection, Mapping, Sequence
from pathlib import Path

from niiflow.preproc.staging import StagedEntry
from niiflow.preproc.pipelines import dynamic_pipeline

from .execution_state import ExecutionState, ExecutionStatus
from .mixins import SupportsInputDiscovery, SupportsStaging, InputData
from .plan import RunPlan
from .plannable_workflow import PlannableWorkflow
from .workflow_factory import create_workflow


class DynamicProcessingWorkflow(
    SupportsInputDiscovery,
    SupportsStaging,
    PlannableWorkflow,
):
    """File-centered processing workflow with dynamic pipeline execution.

    This workflow collects active files, stages per-entry pipeline parameters,
    and executes a dynamically constructed processing pipeline for each
    staged entry.

    Each entry is organised around an **active file**: the canonical path that
    anchors one unit of work, such as a subject's T1w structural MRI. Stagers
    use this active file to resolve file-linked values in the per-entry
    parameter mapping before execution.

    ``pipeline_params`` define the processing pipeline to run, including the
    stage order, stage parameters, and any file-linked fields that should be
    resolved during staging. A single dictionary is copied to every entry
    before stagers resolve per-entry values. A sequence of dictionaries is
    aligned one-to-one with collected active files (explicit paths or
    ``from_file`` listings; ``search`` is not allowed; see "Notes" below).
    Pass ``None`` only when the instance will solely :meth:`run_plan` a previously
    staged plan (entry params then come from the plan).

    Active-file collection is driven by :data:`~niiflow.preproc.workflows.types.InputData`
    passed to :meth:`plan`: an explicit file path, a ``search`` or ``from_file``
    mapping, or a sequence of those.

    The explicit API is:

    - :meth:`plan`: collect active files and build a staged run plan.
    - :meth:`run_plan`: execute an existing run plan.
    - :meth:`run`: execute staged entries (from :class:`~niiflow.preproc.workflows.workflow.ProcessingWorkflow`).

    For a single entry point that instantiates the workflow and selects
    plan / execute / from-plan modes, use :func:`dynamic_workflow`.

    Notes:
        ``from_file`` with ``strict: false`` can drop missing lines, which can
        break one-to-one alignment between collected active files and
        user-provided per-entry pipeline parameters.

    Args:
        pipeline_params: Pipeline specification attached to entries before
            staging. A mapping is copied to every entry. A sequence of mappings
            is aligned one-to-one with collected active files. Usually contains
            ``steps`` plus any file-linked fields resolved by stagers. Required
            as an explicit argument; use ``None`` only for execute-from-plan
            usage where staging is not performed.
        staging_params: Optional stager specifications passed to
            :func:`~niiflow.preproc.staging.create_stager`. When omitted, entries
            are built from ``pipeline_params`` without running stagers.
        num_workers: Worker count (default ``1``, serial). Use ``"auto"`` for one
            worker per CPU core, or an integer ``> 1`` for a fixed pool size.
        staging_workers: Thread count for staging (default ``1``, serial). Use an
            integer ``> 1`` to stage entries with a thread pool. Independent of
            ``num_workers``; do not set this to the HPC core count by default.
        logs_root: Directory for log files; ``None`` logs to console only.
        main_logs: Emit main workflow logs.
        status_logs: Emit per-entry status lines.
        worker_logs: Emit worker-process logs when ``num_workers > 1``.
        dev_mode: Enable debug-level worker logging.
        timeout: Hard per-entry execution limit in seconds. Exceeding it marks
            the entry ``TIMEOUT`` and replaces the worker pool. ``None`` applies
            no task timeout.
    """

    def __init__(
        self,
        *,
        pipeline_params: dict[str, Any] | Sequence[dict[str, Any]] | None,
        staging_params: dict[str, Any] | Sequence[dict[str, Any]] | None = None,
        num_workers: int | Literal["auto"] = 1,
        staging_workers: int = 1,
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
        if pipeline_params is None or isinstance(pipeline_params, dict):
            entry_params: dict[str, Any] | Sequence[dict[str, Any]] | None = (
                pipeline_params
            )
            self._per_entry_pipeline_params = False
        elif isinstance(pipeline_params, (list, tuple)):
            if not all(isinstance(item, dict) for item in pipeline_params):
                raise TypeError("Each `pipeline_params` item must be a dictionary")
            entry_params = pipeline_params
            self._per_entry_pipeline_params = True
        else:
            raise TypeError(
                f"`pipeline_params` must be a dictionary, sequence of dictionaries, "
                f"or None; got {type(pipeline_params).__name__}"
            )
        self.configure_staging(
            staging_params=staging_params,
            entry_params=entry_params,
            staging_workers=staging_workers,
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
                :meth:`~niiflow.preproc.workflows.mixins.SupportsInputDiscovery.collect_active_files`.
            save_filepaths_to: Optional ``.txt`` path for collected active file paths.
            save_plan_to: Optional ``.duckdb`` path for the staged run plan
                (``.json`` is deprecated until v0.5.0).

        Returns:
            A run plan containing one staged entry per collected active file.

        Notes:
            This method performs collection and staging only. It does not execute
            the preprocessing pipeline. When ``pipeline_params`` is a sequence,
            ``search`` inputs are rejected and the sequence must align
            one-to-one with the collected active files.
        """
        active_files = self.collect_active_files(
            inputs,
            save_to=save_filepaths_to,
            allow_search=not self._per_entry_pipeline_params,
        )
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
    save_execution_state_to: Path | str | None = None,
    from_plan: Path | str | None = None,
    from_execution_state: Path | str | None = None,
    run_statuses: Collection[ExecutionStatus | str] | None = None,
    plan_only: bool = False,
    dry_run: bool = False,
    start: int = 0,
    end: int | None = None,
) -> None:
    """Instantiate and execute a :class:`DynamicProcessingWorkflow`.

    This is the orchestration entry point for scripts and higher-level drivers,
    analogous to :func:`~niiflow.preproc.pipelines.dynamic_pipeline`. The workflow
    class itself stays limited to ``plan`` / ``run_plan`` / ``run``.

    Planning artifacts use separate read/write arguments: ``from_plan`` loads an
    existing plan, while ``save_plan_to`` persists a newly generated plan. In
    ``from_plan`` mode planning is skipped, so ``save_plan_to`` and
    ``save_filepaths_to`` are harmless no-ops.

    Execution state follows the same explicit object/persistence model through
    file-based orchestration: ``from_execution_state`` loads an existing state,
    while ``save_execution_state_to`` optionally persists the state used for this
    execution. Both may be supplied together.

    Modes (checked in order):

    * ``from_plan`` — load a saved plan and execute it (or print it when
      ``dry_run``). Planning is skipped; ``pipeline_params`` is forced to
      ``None``. ``settings`` may be omitted. Extra ``inputs`` /
      ``pipeline_params`` are ignored with a log message. Planning save paths
      are ignored. ``plan_only`` must not be set. Optional ``start`` / ``end``
      select a contiguous window while loading the plan.
    * ``plan_only`` — build (and optionally save) a plan from ``inputs`` without
      executing. Requires ``settings`` with non-None ``pipeline_params``. With
      ``dry_run``, the plan is printed and not saved. ``start`` / ``end`` are
      ignored with a warning (planning always covers the full worklist).
    * default — plan from ``inputs`` then execute. Requires ``settings`` with
      non-None ``pipeline_params``. With ``dry_run``, plan without saving,
      print the selected window, and skip execution. ``start`` / ``end`` are
      applied at execute time (or to the dry-run view) and do not affect a
      saved full plan.

    Args:
        settings: Keyword arguments forwarded to
            :class:`DynamicProcessingWorkflow`. Required unless ``from_plan``
            is set; must include non-None ``pipeline_params`` for generating the
            plan. When ``from_plan`` is set, may be omitted (treated as ``{}``)
            and any ``pipeline_params`` entry is cleared to ``None``.
        inputs: Run inputs for planning modes. Required unless ``from_plan`` is
            set.
        save_filepaths_to: Optional ``.txt`` path for collected active files.
            Ignored when ``from_plan`` is set.
        save_plan_to: Optional ``.duckdb`` path for a newly staged plan
            (``.json`` is deprecated until v0.5.0). Ignored when ``from_plan``
            is set.
        save_execution_state_to: Optional path at which to persist the execution
            state used by this run. May be combined with
            ``from_execution_state``.
        from_plan: Path to a saved plan to load and execute, skipping planning.
        from_execution_state: Optional path to a saved execution state to load
            before execution.
        run_statuses: Optional execution statuses eligible to run, supplied as
            :class:`ExecutionStatus` members or their string values. ``None`` selects
            all entries supplied to execution.
        plan_only: When ``True``, stop after planning and do not execute.
        dry_run: When ``True``, print the plan via :meth:`RunPlan.view` and do
            not execute. Newly generated plans are not saved.
        start: Inclusive plan entry index for execution / ranged load.
        end: Exclusive plan entry index for execution / ranged load.

    Raises:
        ValueError: If mode arguments conflict or required ``settings`` /
            ``inputs`` / ``pipeline_params`` are missing.
        TypeError: If ``settings`` is not a mapping or cannot bind to the
            workflow constructor.
    """
    slicing = start != 0 or end is not None
    if run_statuses is not None:
        raw_statuses = (
            (run_statuses,) if isinstance(run_statuses, str) else run_statuses
        )
        run_statuses = {ExecutionStatus(status) for status in raw_statuses}

    if from_plan is not None:
        if plan_only:
            raise ValueError("`from_plan` cannot be combined with `plan_only`")

        if settings is None:
            settings = {}
        elif not isinstance(settings, Mapping):
            raise TypeError(
                f"`settings` must be a mapping or None, got {type(settings).__name__}"
            )
        else:
            settings = dict(settings)

        workflow = cast(
            DynamicProcessingWorkflow,
            create_workflow(
                "DynamicProcessingWorkflow",
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

        if slicing:
            run_plan = RunPlan.load(from_plan, start=start, end=end)
            workflow.log(
                f"Selecting {len(run_plan.entries)} plan "
                f"entr{'y' if len(run_plan.entries) == 1 else 'ies'} "
                f"(start={start!r}, end={end!r})"
            )
        else:
            run_plan = RunPlan.load(from_plan)

        if dry_run:
            print(run_plan.view())
            return

        execution_state = (
            ExecutionState.load(from_execution_state)
            if from_execution_state is not None
            else None
        )
        try:
            workflow.run_plan(
                run_plan,
                execution_state=execution_state,
                save_execution_state_to=save_execution_state_to,
                run_statuses=run_statuses,
            )
        finally:
            if execution_state is not None:
                execution_state.close()
        return

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
        DynamicProcessingWorkflow,
        create_workflow("DynamicProcessingWorkflow", settings),
    )

    if plan_only and slicing:
        workflow.log(
            "Ignoring `start`/`end` because `plan_only` is set; the full plan "
            "will be built (and saved if requested).",
            level="warning",
        )

    run_plan = workflow.plan(
        inputs,
        save_filepaths_to=None if dry_run else save_filepaths_to,
        save_plan_to=None if dry_run else save_plan_to,
    )

    if dry_run:
        print(run_plan.slice(start=start, end=end).view())
        return

    if plan_only:
        return

    execution_state = (
        ExecutionState.load(from_execution_state)
        if from_execution_state is not None
        else None
    )
    try:
        workflow.run_plan(
            run_plan,
            start=start,
            end=end,
            execution_state=execution_state,
            save_execution_state_to=save_execution_state_to,
            run_statuses=run_statuses,
        )
    finally:
        if execution_state is not None:
            execution_state.close()
