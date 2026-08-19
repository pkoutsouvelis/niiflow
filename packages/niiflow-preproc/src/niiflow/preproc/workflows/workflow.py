"""Base workflow abstractions and multi-worker execution."""

from __future__ import annotations

__all__ = [
    "PlannableWorkflow",
    "ProcessingWorkflow",
]

from abc import ABC, abstractmethod
from typing import Any, Callable, Literal, Sequence
from pathlib import Path
import multiprocessing
import time
from concurrent.futures import (
    FIRST_COMPLETED,
    Future,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    as_completed,
    wait,
)
import traceback
from multiprocessing.managers import DictProxy

from tqdm.auto import tqdm

from niiflow.preproc.staging import StagedEntry
from .logging_manager import LoggingManager, ParallelLogging
from .logging_utils import (
    set_input_file_context,
    reset_input_file_context,
)
from .plan import RunPlan

EntryOutcome = Literal["SUCCESS", "TIMEOUT", "FAILURE", "STAGING_FAILURE"]


def _execute_one(
    process_fn: Callable[[StagedEntry], None],
    entry: StagedEntry,
    start_times: DictProxy | None = None,
) -> float:
    """Picklable single-entry executor with per-task input file context.

    Returns wall-clock processing seconds for this entry. When ``start_times`` is
    provided, records the worker-local monotonic start time under ``str(entry.active)``
    so the main process can apply per-entry timeouts without counting queue wait.
    """
    token = set_input_file_context(entry.active.name)
    started = time.monotonic()
    if start_times is not None:
        start_times[str(entry.active)] = started
    try:
        process_fn(entry)
    finally:
        reset_input_file_context(token)
    return time.monotonic() - started


class ProcessingWorkflow(ABC):
    """Base class for executing staged workflow entries.

    This class provides the common execution engine: logging, serial or
    parallel worker execution, per-entry status reporting, and optional soft
    per-entry time limits.

    Subclasses implement :meth:`process_single`, which defines how one
    :class:`~niiflow.preproc.staging.StagedEntry` is processed.

    The stable low-level execution API is :meth:`run_entries`. The public
    :meth:`run` method represents the default user-facing action for this class
    and may be specialized by higher-level workflow subclasses.
    """

    def __init__(
        self,
        *,
        num_workers: int | Literal["auto"] = 1,
        logs_root: Path | str | None = None,
        main_logs: bool = True,
        status_logs: bool = True,
        worker_logs: bool = True,
        dev_mode: bool = False,
        timeout: float | None = None,
    ) -> None:
        self._logging_manager = LoggingManager(
            logs_root=logs_root,
            main_logs=main_logs,
            status_logs=status_logs,
            worker_logs=worker_logs,
            dev_mode=dev_mode,
        )
        self._main_logger = self._logging_manager.setup_main_logging()
        self._status_logger = self._logging_manager.setup_status_logging()
        self.num_workers = num_workers
        self.timeout = timeout

    @property
    def num_workers(self) -> int:
        return self._num_workers

    @property
    def timeout(self) -> float | None:
        return self._timeout

    @num_workers.setter
    def num_workers(self, num_workers: int | Literal["auto"]) -> None:
        if isinstance(num_workers, str):
            if num_workers != "auto":
                raise ValueError(
                    f"`num_workers` must be 'auto' or an integer, got `{num_workers}`"
                )
            self._num_workers = multiprocessing.cpu_count()
        elif isinstance(num_workers, int):
            if num_workers < 1:
                raise ValueError(
                    f"`num_workers` must be at least 1, got `{num_workers}`"
                )
            self._num_workers = num_workers
        else:
            raise ValueError(
                f"`num_workers` must be 'auto' or an integer, got `{num_workers}`"
            )

    @timeout.setter
    def timeout(self, timeout: float | None) -> None:
        if timeout is not None:
            if not isinstance(timeout, (int, float)):
                raise ValueError(
                    f"`timeout` must be a number or None, got {type(timeout).__name__}"
                )
            if timeout <= 0:
                raise ValueError(f"`timeout` must be positive, got `{timeout}`")
            timeout = float(timeout)
        self._timeout = timeout

    def log(
        self,
        message: str,
        level: Literal["debug", "info", "warning", "error", "critical"] = "info",
    ) -> None:
        getattr(self._main_logger, level)(message)

    def status(self, message: str) -> None:
        self._status_logger.info(message)

    @staticmethod
    @abstractmethod
    def process_single(entry: StagedEntry) -> None:
        """Process one staged entry.

        Must be picklable for process pools.
        """
        ...

    def run_entries(self, entries: Sequence[StagedEntry]) -> None:
        """Execute a sequence of staged entries.

        This is the canonical low-level execution method. Each entry is processed
        independently by :meth:`process_single`, either serially or using the
        configured worker pool.

        Args:
            entries: Staged entries to process. Each entry should already contain
                all parameters required by :meth:`process_single`.

        Raises:
            TypeError: If any item in ``entries`` is not a
                :class:`~niiflow.preproc.staging.StagedEntry`.

        Notes:
            Entries with :attr:`~niiflow.preproc.staging.StagedEntry.errors` are
            reported as ``STAGING_FAILURE`` in the status log and are not passed to
            :meth:`process_single`.

            A configured timeout is a soft per-entry time limit. Exceeding it
            does not cancel the worker; ``status.log`` records ``TIMEOUT`` to
            mark the overrun. Completion is still ``SUCCESS`` or ``FAILURE``.
        """
        entry_list = list(entries)
        if not entry_list:
            self.log("No entries to process")
            return
        if not all(isinstance(entry, StagedEntry) for entry in entry_list):
            raise TypeError("`entries` must contain only `StagedEntry` objects")

        runnable, skipped = self._partition_entries(entry_list)
        if skipped:
            self.log(
                f"Skipping {len(skipped)} entr{'y' if len(skipped) == 1 else 'ies'} "
                "with staging errors"
            )
            for entry in skipped:
                self._status_entry(
                    entry,
                    "STAGING_FAILURE",
                    detail=" | ".join(error.message for error in entry.errors),
                )

        self.log(
            f"Starting workflow with {len(runnable)} runnable "
            f"entr{'y' if len(runnable) == 1 else 'ies'} "
            f"({len(entry_list)} total), {self._num_workers} workers"
        )

        if not runnable:
            self.log("No runnable entries to process")
            return

        with (
            self._logging_manager.setup_parallel_logging() as parallel,
            tqdm(
                total=len(runnable),
                desc="Processing entries",
                unit="entry",
                disable=None,
                leave=True,
            ) as progress,
        ):
            if self._num_workers <= 1:
                self._run_serial(runnable, progress)
            else:
                self._run_parallel(runnable, parallel, progress)

        self.log("Workflow complete")

    def run(self, entries: Sequence[StagedEntry]) -> None:
        """Run the default workflow action.

        For execution-only workflows, this is equivalent to
        :meth:`run_entries`.

        Args:
            entries: Staged entries to execute.
        """
        self.run_entries(entries)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Call :meth:`run`.

        This provides a compact shorthand for the class-specific default run interface.
        """
        return self.run(*args, **kwargs)

    @staticmethod
    def _partition_entries(
        entries: list[StagedEntry],
    ) -> tuple[list[StagedEntry], list[StagedEntry]]:
        runnable: list[StagedEntry] = []
        skipped: list[StagedEntry] = []
        for entry in entries:
            if entry.errors:
                skipped.append(entry)
            else:
                runnable.append(entry)
        return runnable, skipped

    def _status_entry(
        self,
        entry: StagedEntry,
        outcome: EntryOutcome,
        *,
        detail: str = "",
    ) -> None:
        """Append a ``status.log`` line for ``entry``.

        ``TIMEOUT`` marks an overrun and may be followed by ``SUCCESS`` or ``FAILURE``
        when the worker finishes.
        """
        if outcome == "SUCCESS":
            self.status(f"{entry.active} | SUCCESS")
        elif outcome == "TIMEOUT":
            self.status(f"{entry.active} | TIMEOUT | exceeded {self._timeout}s")
        elif outcome == "FAILURE":
            self.status(f"{entry.active} | FAILURE | {detail}")
        else:
            self.status(f"{entry.active} | STAGING_FAILURE | {detail}")

    def _status_future(
        self,
        fut: Future[float],
        entry: StagedEntry,
        *,
        timeout: float | None = None,
    ) -> bool:
        """Wait on ``fut`` and append a ``status.log`` line.

        Returns ``True`` when processing finished (``SUCCESS`` or ``FAILURE``). Returns
        ``False`` when ``timeout`` expired first (``TIMEOUT`` marks the overrun; the
        worker may still be running).
        """
        try:
            fut.result(timeout=timeout)
        except TimeoutError:
            self._status_entry(entry, "TIMEOUT")
            return False
        except Exception:
            tb = traceback.format_exc().strip().replace("\n", "\n  ")
            self._status_entry(entry, "FAILURE", detail=tb)
            return True
        else:
            self._status_entry(entry, "SUCCESS")
            return True

    def _run_serial(self, entries: list[StagedEntry], progress: Any) -> None:
        if self._timeout is None:
            for entry in entries:
                try:
                    _execute_one(self.process_single, entry)
                    self._status_entry(entry, "SUCCESS")
                except Exception:
                    tb = traceback.format_exc().strip().replace("\n", "\n  ")
                    self._status_entry(entry, "FAILURE", detail=tb)
                finally:
                    progress.update(1)
            return

        with ThreadPoolExecutor(max_workers=1) as executor:
            for entry in entries:
                fut = executor.submit(_execute_one, self.process_single, entry)
                if not self._status_future(fut, entry, timeout=self._timeout):
                    self._status_future(fut, entry)
                progress.update(1)

    def _run_parallel(
        self,
        entries: list[StagedEntry],
        parallel: ParallelLogging,
        progress: Any,
    ) -> None:
        manager = multiprocessing.Manager() if self._timeout is not None else None
        try:
            start_times = manager.dict() if manager is not None else None
            with ProcessPoolExecutor(
                max_workers=self._num_workers,
                initializer=parallel.worker_init_fn,
            ) as pool:
                futures = {
                    pool.submit(
                        _execute_one, self.process_single, entry, start_times
                    ): entry
                    for entry in entries
                }
                if self._timeout is None:
                    for fut in as_completed(futures):
                        self._status_future(fut, futures[fut])
                        progress.update(1)
                    return

                completed: set[Future[float]] = set()
                timed_out: set[Future[float]] = set()
                poll_interval = min(0.5, self._timeout / 10)

                while len(completed) < len(futures):
                    remaining = {
                        fut: entry
                        for fut, entry in futures.items()
                        if fut not in completed
                    }
                    done, not_done = wait(
                        remaining.keys(),
                        timeout=poll_interval,
                        return_when=FIRST_COMPLETED,
                    )
                    for fut in done:
                        if fut in completed:
                            continue
                        self._status_future(fut, remaining[fut])
                        completed.add(fut)
                        progress.update(1)

                    now = time.monotonic()
                    for fut in not_done:
                        if fut in timed_out or start_times is None:
                            continue
                        started = start_times.get(str(remaining[fut].active))
                        if started is not None and now - started > self._timeout:
                            self._status_entry(remaining[fut], "TIMEOUT")
                            timed_out.add(fut)
        finally:
            if manager is not None:
                manager.shutdown()


class PlannableWorkflow(ProcessingWorkflow):
    """Base for workflows that support planning before execution.

    Plannable workflows discover or assemble inputs, resolve per-entry parameters, and
    produce a :class:`~niiflow.preproc.workflows.plan.RunPlan`.

    Subclasses implement :meth:`plan` with workflow-specific inputs. Use
    :meth:`run_plan` to execute a saved or freshly built plan. Use :meth:`run_entries`
    only when executing already staged entries directly.
    """

    @abstractmethod
    def plan(self, source: Any) -> RunPlan:
        """Build a run plan from workflow-specific inputs."""
        ...

    def run_plan(
        self,
        plan: RunPlan,
        *,
        start: int = 0,
        end: int | None = None,
    ) -> None:
        """Execute a prepared run plan.

        Args:
            plan: Run plan whose entries should be executed.
            start: Inclusive plan entry index to begin at (supports negatives).
            end: Exclusive plan entry index to stop at. ``None`` runs through the
                last entry. Indices include staging-failed entries and use the same
                bound rules as :meth:`~niiflow.preproc.workflows.plan.RunPlan.slice`.

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
        self.run_entries(selected.entries)
