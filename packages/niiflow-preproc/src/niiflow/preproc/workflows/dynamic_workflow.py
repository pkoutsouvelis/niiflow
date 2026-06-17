"""File-driven preprocessing workflow with dynamic pipeline execution."""

from __future__ import annotations

__all__ = [
    "DynamicPreprocessingWorkflow",
]

from typing import Any, Literal, cast
from collections.abc import Sequence
from pathlib import Path

from niiflow.preproc.staging import StagedEntry
from niiflow.preproc.pipelines import create_pipeline
from niiflow.preproc.pipelines.pipeline_stages import RuntimeContext

from .mixins import FileDiscoveryMixin, InputData, StagingMixin
from .plan import RunPlan
from .workflow import PlanningWorkflow


class DynamicPreprocessingWorkflow(
    FileDiscoveryMixin,
    StagingMixin,
    PlanningWorkflow,
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
    copied to every entry before stagers resolve per-entry values.

    The explicit API is:

    - :meth:`plan`: discover active files and build a staged run plan.
    - :meth:`run_plan`: execute an existing run plan.
    - :meth:`run_inputs`: build and execute a run plan from file inputs.
    - :meth:`run`: convenience method accepting either file inputs or a run plan.

    Args:
        pipeline_params: Pipeline specification copied into entry parameters before
            staging. In the current implementation, this usually contains ``steps``
            plus any file-linked fields resolved by stagers. Semantically, it defines
            the preprocessing pipeline; future implementations may keep it as an
            invariant workflow-level pipeline object instead of copying it into each
            entry.
        explorer_params: Optional data explorer configuration used to discover
            active files. When omitted, files are expected to be provided
            explicitly.
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
        pipeline_params: dict[str, Any],
        explorer_params: dict[str, Any] | None = None,
        staging_params: dict[str, Any] | Sequence[dict[str, Any]] | None = None,
        num_workers: int | Literal["auto"] = 1,
        logs_root: Path | str | None = None,
        main_logs: bool = True,
        status_logs: bool = True,
        worker_logs: bool = True,
        dev_mode: bool = False,
        timeout: float | None = None,
    ) -> None:
        PlanningWorkflow.__init__(
            self,
            num_workers=num_workers,
            logs_root=logs_root,
            main_logs=main_logs,
            status_logs=status_logs,
            worker_logs=worker_logs,
            dev_mode=dev_mode,
            timeout=timeout,
        )

        self.configure_explorer(explorer_params)
        self.configure_staging(staging_params)

        if not isinstance(pipeline_params, dict):
            raise TypeError("`pipeline_params` must be a dictionary")
        self.configure_entry_params(pipeline_params)

    @property
    def pipeline_params(self) -> dict[str, Any]:
        """Return a copy of the configured pipeline parameters."""
        return cast(dict[str, Any], self.entry_params)

    def plan(self, files: InputData) -> RunPlan:
        """Build a staged preprocessing run plan from file inputs.

        Args:
            files: File input specification. This may be a single path, a
                sequence of paths, or a finder configuration used to discover
                active files.

        Returns:
            A run plan containing one staged entry per discovered active file.

        Notes:
            This method performs discovery and staging only. It does not execute
            the preprocessing pipeline.
        """
        self.configure_files(files)
        active_files = self.discover_active_files()
        entries = self.stage_active_files(active_files)
        return RunPlan(entries=tuple(entries))

    def run_inputs(self, files: InputData) -> RunPlan:
        """Plan and execute preprocessing from file run inputs.

        This method resolves ``files`` into active files, stages one entry per
        active file, executes the resulting run plan, and returns that plan.

        Explicit file paths are used directly. Directory paths require a configured
        data explorer and are searched during planning.

        Args:
            files: File run inputs. This may be a single file path, a sequence of
                file paths, a directory path, or a sequence mixing files and
                directories. Directory inputs require ``explorer_params`` to have
                been provided when constructing the workflow.

        Returns:
            The run plan that was built and executed.

        Raises:
            FileNotFoundError: If any provided path does not exist.
            ValueError: If directory inputs are provided without a configured data
                explorer.
        """
        plan = self.plan(files)
        self.run_plan(plan)
        return plan

    def run(self, target: InputData | RunPlan) -> None:
        """Run preprocessing from file inputs or an existing run plan.

        If ``target`` is a :class:`RunPlan`, it is executed directly without file
        discovery or staging. If ``target`` is file input, it is passed to
        :meth:`run_inputs`, which builds and executes a new run plan.

        Args:
            target: Either a :class:`RunPlan` to execute, or file run inputs accepted
                by :meth:`run_inputs`.

        Notes:
            Use :meth:`plan` when you want to build a run plan without executing it.
            Use :meth:`run_plan` when you want to execute a previously built or
            loaded plan explicitly.
        """
        if isinstance(target, RunPlan):
            self.run_plan(target)
        else:
            self.run_inputs(target)

    @staticmethod
    def process_single(entry: StagedEntry) -> None:
        """Run the configured preprocessing pipeline for one staged entry."""
        pipeline = create_pipeline(entry.params)
        ctx = RuntimeContext(run_id=str(entry.active))
        pipeline.run(ctx)
