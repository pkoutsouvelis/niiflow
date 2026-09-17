"""Planning layer on top of
:class:`~niiflow.preproc.workflows.workflow.ProcessingWorkflow`."""

from __future__ import annotations

__all__ = [
    "PlannableWorkflow",
]

from abc import abstractmethod
from collections.abc import Collection
from pathlib import Path
from typing import Any

from .execution_state import ExecutionState, ExecutionStatus
from .plan import RunPlan
from .workflow import ProcessingWorkflow


class PlannableWorkflow(ProcessingWorkflow):
    """Base for workflows that support planning before execution.

    Plannable workflows discover or assemble inputs, resolve per-entry parameters, and
    produce a :class:`~niiflow.preproc.workflows.plan.RunPlan`.

    Subclasses implement :meth:`plan` with workflow-specific inputs. Use
    :meth:`run_plan` to execute a saved or freshly built plan. Use :meth:`run_entries`
    only when executing already staged entries directly.
    """

    @abstractmethod
    def plan(self, *args: Any, **kwargs: Any) -> RunPlan:
        """Build a run plan from workflow-specific inputs."""
        ...

    def run_plan(
        self,
        plan: RunPlan,
        *,
        start: int = 0,
        end: int | None = None,
        execution_state: ExecutionState | None = None,
        save_execution_state_to: Path | str | None = None,
        run_statuses: Collection[ExecutionStatus] | None = None,
    ) -> ExecutionState:
        """Execute a prepared run plan.

        Args:
            plan: Run plan whose entries should be executed.
            start: Inclusive plan entry index to begin at (supports negatives).
            end: Exclusive plan entry index to stop at. ``None`` runs through the
                last entry. Indices include staging-failed entries and use the same
                bound rules as :meth:`~niiflow.preproc.workflows.plan.RunPlan.slice`.
            execution_state: Existing state to validate and update. If omitted, a new
                state is initialized for the selected plan entries.
            save_execution_state_to: Optional path at which to persist the prepared
                execution state before execution.
            run_statuses: Execution statuses eligible to run. ``None`` selects all
                selected plan entries regardless of current execution status.

        Raises:
            TypeError: If ``plan`` is not a
                :class:`~niiflow.preproc.workflows.plan.RunPlan`.
            ValueError: If ``start`` / ``end`` are out of range for ``plan``.

        Notes:
            This method does not rebuild, modify, or restage the plan. It executes
            the selected window of entries as stored in ``plan``.
        """
        if not isinstance(plan, RunPlan):
            raise TypeError(f"`plan` must be a RunPlan, got {type(plan).__name__}")
        selected = plan.slice(start=start, end=end)
        total = len(plan.entries)
        if len(selected.entries) != total:
            self.log(
                f"Selecting {len(selected.entries)} of {total} plan "
                f"entr{'y' if total == 1 else 'ies'} "
                f"(start={start!r}, end={end!r})"
            )
        return self.run_entries(
            selected.entries,
            execution_state=execution_state,
            save_execution_state_to=save_execution_state_to,
            run_statuses=run_statuses,
        )
