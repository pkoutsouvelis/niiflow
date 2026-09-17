"""Workflow execution with bounded scheduling, hard timeouts, and pool recovery."""

from __future__ import annotations

__all__ = [
    "ProcessingWorkflow",
]

from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Collection
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Literal, Sequence, assert_never
import multiprocessing
import time
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool

from tqdm.auto import tqdm

from niiflow.preproc.staging import StagedEntry
from .execution_state import ExecutionState, ExecutionStatus
from .logging_manager import LoggingManager, ParallelLogging
from .logging_utils import reset_input_file_context, set_input_file_context
from .validation import validate_staged_entries

_MAX_DIAGNOSIS_PRESTART_FAILURES = 2
_PROCESS_TERMINATE_GRACE = 2.0
_PROCESS_KILL_GRACE = 2.0
_STATUS_DETAIL_MAX_CHARS = 1_000
_MIN_POLL_INTERVAL = 0.05
_MAX_POLL_INTERVAL = 0.5


@dataclass
class _WorkerContext:
    start_queue: Any


_WORKER_CONTEXT: _WorkerContext | None = None


def _initialize_parallel_worker(
    worker_init_fn: Callable[[], None],
    start_queue: Any,
) -> None:
    global _WORKER_CONTEXT
    _WORKER_CONTEXT = _WorkerContext(start_queue=start_queue)
    worker_init_fn()


def _execute_one(
    process_fn: Callable[[StagedEntry], None],
    entry: StagedEntry,
) -> float:
    started = time.monotonic()

    context = _WORKER_CONTEXT
    if context is not None:
        context.start_queue.put((entry.id, started))

    log_token = set_input_file_context(entry.id)
    try:
        process_fn(entry)
    finally:
        reset_input_file_context(log_token)

    return time.monotonic() - started


class _PoolExit(Enum):
    COMPLETE = auto()
    TIMEOUT = auto()
    BROKEN = auto()


@dataclass(frozen=True)
class _PoolResult:
    """Outcome of driving one process-pool instance.

    ``casualties`` are submitted entries that did not reach a terminal status before the
    pool was terminated. ``started_casualties`` are casualties whose worker-start
    checkpoint was observed before the pool was fully terminated.
    """

    outcome: _PoolExit
    casualties: tuple[StagedEntry, ...] = ()
    started_casualties: frozenset[str] = frozenset()
    broken_pool: BrokenProcessPool | None = None
    made_progress: bool = False


class ProcessingWorkflow(ABC):
    """Base class for serial/process-pool execution of staged entries."""

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
        # Deliberately non-fork: the parallel logger owns a listener thread.
        self._mp_context = multiprocessing.get_context("spawn")
        self._logging_manager = LoggingManager(
            logs_root=logs_root,
            main_logs=main_logs,
            status_logs=status_logs,
            worker_logs=worker_logs,
            dev_mode=dev_mode,
            mp_context=self._mp_context,
        )
        self._main_logger = self._logging_manager.setup_main_logging()
        self._status_logger = self._logging_manager.setup_status_logging()
        self.num_workers = num_workers
        self.timeout = timeout

    @property
    def num_workers(self) -> int:
        return self._num_workers

    @num_workers.setter
    def num_workers(self, value: int | Literal["auto"]) -> None:
        if value == "auto":
            self._num_workers = multiprocessing.cpu_count()
        elif isinstance(value, int) and value >= 1:
            self._num_workers = value
        else:
            raise ValueError("`num_workers` must be 'auto' or an integer >= 1")

    @property
    def timeout(self) -> float | None:
        return self._timeout

    @timeout.setter
    def timeout(self, value: float | None) -> None:
        if value is None:
            self._timeout = None
        elif isinstance(value, (int, float)) and value > 0:
            self._timeout = float(value)
        else:
            raise ValueError("`timeout` must be a positive number or None")

    def log(
        self,
        message: str,
        level: Literal["debug", "info", "warning", "error", "critical"] = "info",
        **kwargs: Any,
    ) -> None:
        """Log through the workflow logger, forwarding standard logging kwargs."""
        getattr(self._main_logger, level)(message, **kwargs)

    def status(self, message: str) -> None:
        self._status_logger.info(message)

    @staticmethod
    @abstractmethod
    def process_single(entry: StagedEntry) -> None:
        """Process one entry.

        Must be picklable for process pools.
        """
        ...

    def run_entries(
        self,
        entries: Sequence[StagedEntry],
        *,
        execution_state: ExecutionState | None = None,
        save_execution_state_to: Path | str | None = None,
        run_statuses: Collection[ExecutionStatus] | None = None,
    ) -> ExecutionState:
        """Execute staged entries, optionally filtered by persistent execution state.

        Args:
            entries: Staged entries to consider for execution.
            execution_state: Existing state to validate and update. If omitted, a new
                in-memory state is initialized with all supplied entries ``PENDING``.
            save_execution_state_to: Optional path at which to persist the prepared
                execution state before execution. Subsequent status updates are written
                through to that state file.
            run_statuses: Execution statuses eligible to run. ``None`` selects all
                supplied entries regardless of their current execution status.

        Returns:
            The execution state used for this run.

        Notes:
            Entries with staging errors never enter execution and therefore retain their
            existing execution status (normally ``PENDING``). Staging failure is not an
            execution failure.
        """
        entry_list = list(validate_staged_entries(entries))
        state = self._prepare_execution_state(entry_list, execution_state)

        try:
            selected = self._filter_by_execution_status(entry_list, state, run_statuses)

            if save_execution_state_to is not None:
                state.save(save_execution_state_to)  # !Shouldn't we allow overwrite?!

            filtered_count = len(entry_list) - len(selected)
            if run_statuses is not None:
                names = ", ".join(sorted(status.value for status in run_statuses))
                self.log(
                    f"Execution-state filter selected {len(selected)} of {len(entry_list)} "
                    f"entr{'y' if len(entry_list) == 1 else 'ies'} "
                    f"for status{'es' if len(run_statuses) != 1 else ''}: {names or '<none>'}"
                )

            runnable: list[StagedEntry] = []
            staging_failed: list[StagedEntry] = []
            for entry in selected:
                (staging_failed if entry.errors else runnable).append(entry)

            if staging_failed:
                self.log(
                    f"Skipping {len(staging_failed)} "
                    f"entr{'y' if len(staging_failed) == 1 else 'ies'} with staging errors"
                )
                for entry in staging_failed:
                    self._status_staging_failure(
                        entry,
                        detail=" | ".join(error.message for error in entry.errors),
                    )

            self.log(
                f"Final workload: {len(runnable)} runnable "
                f"entr{'y' if len(runnable) == 1 else 'ies'} "
                f"from {len(entry_list)} supplied "
                f"({filtered_count} filtered by execution status, "
                f"{len(staging_failed)} staging-failed), {self._num_workers} workers"
            )
            if not runnable:
                self.log("Workflow complete")
                return state

            runnable_ids = [entry.id for entry in runnable]
            self.log(
                "Execution state for runnable entries before run: "
                f"{state.view(runnable_ids)}"
            )

            with tqdm(
                total=len(runnable), desc="Processing entries", unit="entry"
            ) as progress:
                # A finite timeout requires process isolation even with one worker.
                if self._num_workers == 1 and self._timeout is None:
                    self._run_serial(runnable, state, progress)
                else:
                    with self._logging_manager.setup_parallel_logging() as parallel:
                        self._run_parallel(runnable, state, parallel, progress)

            self.log(
                f"Final execution state for runnable entries: {state.view(runnable_ids)}"
            )
            self.log("Workflow complete")
            return state

        finally:
            state.close()

    def run(
        self,
        entries: Sequence[StagedEntry],
        *,
        execution_state: ExecutionState | None = None,
        save_execution_state_to: Path | str | None = None,
        run_statuses: Collection[ExecutionStatus] | None = None,
    ) -> ExecutionState:
        return self.run_entries(
            entries,
            execution_state=execution_state,
            save_execution_state_to=save_execution_state_to,
            run_statuses=run_statuses,
        )

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.run(*args, **kwargs)

    @staticmethod
    def _prepare_execution_state(
        entries: Sequence[StagedEntry],
        execution_state: ExecutionState | None,
    ) -> ExecutionState:
        if execution_state is None:
            return ExecutionState.from_entries(entries)

        if not isinstance(execution_state, ExecutionState):
            raise TypeError(
                "`execution_state` must be an ExecutionState or None, got "
                f"{type(execution_state).__name__}"
            )

        execution_state.add_entries(entries)
        return execution_state

    @staticmethod
    def _filter_by_execution_status(
        entries: Sequence[StagedEntry],
        execution_state: ExecutionState,
        run_statuses: Collection[ExecutionStatus] | None,
    ) -> list[StagedEntry]:
        if run_statuses is None:
            return list(entries)

        statuses = set(run_statuses)
        invalid = [
            status for status in statuses if not isinstance(status, ExecutionStatus)
        ]
        if invalid:
            raise TypeError(
                "`run_statuses` must contain only ExecutionStatus values, got "
                f"{type(invalid[0]).__name__}"
            )

        return [
            entry
            for entry in entries
            if execution_state.get_status(entry.id) in statuses
        ]

    def _record_execution_status(
        self,
        execution_state: ExecutionState,
        entry: StagedEntry,
        status: ExecutionStatus,
        *,
        detail: str = "",
    ) -> None:
        """Persist one execution transition and mirror terminal outcomes to
        status.log."""
        execution_state.update({entry.id: status})

        if status is ExecutionStatus.RUNNING:
            return
        if status is ExecutionStatus.SUCCESS:
            self.status(f"{entry.id} | SUCCESS")
        elif status is ExecutionStatus.TIMEOUT:
            if self._timeout is None:
                raise RuntimeError("Internal error: TIMEOUT with timeout=None")
            self.status(f"{entry.id} | TIMEOUT | exceeded {self._timeout:g}s")
        elif status is ExecutionStatus.FAILURE:
            self.status(f"{entry.id} | FAILURE | {self._compact(detail)}")
        elif status is ExecutionStatus.PENDING:
            raise RuntimeError("Internal error: PENDING is not an execution transition")
        else:
            assert_never(status)

    def _status_staging_failure(
        self,
        entry: StagedEntry,
        *,
        detail: str,
    ) -> None:
        """Record a staging failure for humans without changing execution state."""
        self.status(f"{entry.id} | STAGING_FAILURE | {self._compact(detail)}")

    @staticmethod
    def _compact(detail: str) -> str:
        text = " ".join(str(detail).split()) or "no detail"
        if len(text) > _STATUS_DETAIL_MAX_CHARS:
            text = text[: _STATUS_DETAIL_MAX_CHARS - 1] + "…"
        return text

    def _record_failure(
        self,
        execution_state: ExecutionState,
        entry: StagedEntry,
        exc: Exception,
    ) -> None:
        self.log(
            f"Processing failed for {entry.id}",
            level="error",
            exc_info=(type(exc), exc, exc.__traceback__),
        )
        self._record_execution_status(
            execution_state,
            entry,
            ExecutionStatus.FAILURE,
            detail=f"{type(exc).__name__}: {self._compact(str(exc))}",
        )

    def _run_serial(
        self,
        entries: Sequence[StagedEntry],
        execution_state: ExecutionState,
        progress: Any,
    ) -> None:
        """Run directly in the main process; only used when no timeout is configured."""
        for entry in entries:
            self._record_execution_status(
                execution_state, entry, ExecutionStatus.RUNNING
            )
            try:
                _execute_one(self.process_single, entry)
                self._record_execution_status(
                    execution_state, entry, ExecutionStatus.SUCCESS
                )
            except Exception as exc:
                self._record_failure(execution_state, entry, exc)
            finally:
                progress.update(1)

    def _run_parallel(
        self,
        entries: Sequence[StagedEntry],
        execution_state: ExecutionState,
        parallel: ParallelLogging,
        progress: Any,
    ) -> None:
        pending = deque(entries)

        while pending:
            result = self._drive_pool(
                pending,
                execution_state,
                parallel,
                progress,
                max_workers=self._num_workers,
            )

            if result.outcome is _PoolExit.COMPLETE:
                return

            if result.outcome is _PoolExit.TIMEOUT:
                # Timeout identifies its offending entries, which are not retried.
                # Any peer casualties are therefore safe to retry in a fresh pool.
                pending.extendleft(reversed(result.casualties))
                continue

            # BrokenProcessPool does not identify which entry killed a worker.
            # Diagnose the unresolved submitted entries in isolation.
            self._log_broken_pool(result.broken_pool)

            if result.casualties:
                suspects = result.casualties
            elif pending:
                # A pool can fail during submission before any Future becomes an
                # observable casualty. Probe the next still-pending entry in a fresh
                # one-worker pool; repeated pre-start failure is then treated as
                # infrastructure failure rather than blamed on that entry.
                suspects = (pending.popleft(),)
            else:
                raise RuntimeError(
                    "Worker process pool became unusable with no entry available "
                    "for diagnosis"
                ) from result.broken_pool

            self.log(
                f"Process pool failed; diagnosing {len(suspects)} "
                f"entr{'y' if len(suspects) == 1 else 'ies'} in isolation",
                level="warning",
            )
            self._diagnose_entries(
                suspects,
                execution_state,
                parallel,
                progress,
            )
            # Untouched work remains in `pending`; once diagnosis classifies the
            # ambiguous casualties, normal parallel execution resumes here.

    def _drive_pool(
        self,
        pending: deque[StagedEntry],
        execution_state: ExecutionState,
        parallel: ParallelLogging,
        progress: Any,
        *,
        max_workers: int,
        max_tasks_per_child: int | None = None,
    ) -> _PoolResult:
        """Drive one process-pool instance until completion or recovery.

        The same primitive powers normal execution and isolated diagnosis. Recovery
        policy lives in callers: this method only reports COMPLETE, TIMEOUT, or BROKEN
        together with unresolved casualties.
        """
        start_queue = self._mp_context.SimpleQueue()
        start_times: dict[str, float] = {}
        in_flight: dict[Future[float], StagedEntry] = {}
        made_progress = False

        try:
            pool = ProcessPoolExecutor(
                max_workers=max_workers,
                mp_context=self._mp_context,
                initializer=_initialize_parallel_worker,
                initargs=(parallel.worker_init_fn, start_queue),
                max_tasks_per_child=max_tasks_per_child,
            )
            try:
                while pending or in_flight:
                    # Keep at most one submitted Future per worker. Peek before submit so
                    # a failed submit cannot consume an entry from pending.
                    try:
                        while pending and len(in_flight) < max_workers:
                            entry = pending[0]
                            future = pool.submit(
                                _execute_one,
                                self.process_single,
                                entry,
                            )
                            pending.popleft()
                            in_flight[future] = entry
                    except BrokenProcessPool as exc:
                        # The executor was already/became unusable during submission. This
                        # is a pool-level signal; it does not identify a guilty entry.
                        return self._rescue_flight(
                            pool,
                            in_flight,
                            execution_state,
                            start_queue,
                            start_times,
                            progress,
                            _PoolExit.BROKEN,
                            exc,
                            made_progress,
                        )

                    # After a successful submit pass, empty in_flight with work still
                    # pending is an invariant violation (e.g. max_workers < 1 never opens
                    # the submit window). Fail fast instead of spinning forever.
                    if not in_flight:
                        if pending:
                            raise RuntimeError(
                                "Internal scheduler error: pending entries remain but no "
                                "futures were submitted"
                            )
                        break

                    wait(
                        tuple(in_flight),
                        timeout=self._poll_interval(),
                        return_when=FIRST_COMPLETED,
                    )

                    self._drain_start_events(
                        start_queue,
                        in_flight,
                        execution_state,
                        start_times,
                    )

                    broken, progressed = self._consume_done(
                        in_flight, execution_state, start_times, progress
                    )
                    made_progress |= progressed
                    if broken is not None:
                        return self._rescue_flight(
                            pool,
                            in_flight,
                            execution_state,
                            start_queue,
                            start_times,
                            progress,
                            _PoolExit.BROKEN,
                            broken,
                            made_progress,
                        )

                    if self._mark_timeouts(
                        in_flight, execution_state, start_times, progress
                    ):
                        return self._rescue_flight(
                            pool,
                            in_flight,
                            execution_state,
                            start_queue,
                            start_times,
                            progress,
                            _PoolExit.TIMEOUT,
                            None,
                            True,
                        )

                pool.shutdown(wait=True)
                self._drain_start_events(
                    start_queue,
                    in_flight,
                    execution_state,
                    start_times,
                    mark_running=False,
                )
                return _PoolResult(_PoolExit.COMPLETE, made_progress=made_progress)

            except BaseException:
                # Covers scheduler bugs, queue errors, KeyboardInterrupt/SystemExit, etc.;
                # never leave worker processes behind on an unexpected escape.
                self._terminate_pool(pool)
                self._drain_start_events(
                    start_queue,
                    in_flight,
                    execution_state,
                    start_times,
                    mark_running=False,
                )
                raise
        finally:
            start_queue.close()

    def _drain_start_events(
        self,
        start_queue: Any,
        in_flight: dict[Future[float], StagedEntry],
        execution_state: ExecutionState,
        start_times: dict[str, float],
        *,
        mark_running: bool = True,
    ) -> None:
        """Drain worker starts into local state and mark active entries RUNNING."""
        active = {entry.id: (future, entry) for future, entry in in_flight.items()}

        while not start_queue.empty():
            entry_id, started = start_queue.get()
            item = active.get(entry_id)
            if item is None:
                # The entry has been completed and popped from in_flight after the previous
                # drain and before the current one; its start time is no longer needed.
                continue

            future, entry = item
            if entry_id in start_times:
                raise RuntimeError(
                    f"Internal scheduler error: duplicate start event for entry `{entry_id}`"
                )
            start_times[entry_id] = started
            if (
                mark_running
                and not future.done()
                and execution_state.get_status(entry_id) is not ExecutionStatus.RUNNING
            ):
                self._record_execution_status(
                    execution_state,
                    entry,
                    ExecutionStatus.RUNNING,
                )

    def _consume_done(
        self,
        in_flight: dict[Future[float], StagedEntry],
        execution_state: ExecutionState,
        start_times: dict[str, float],
        progress: Any,
    ) -> tuple[BrokenProcessPool | None, bool]:
        """Consume every Future whose result is observable right now."""
        broken: BrokenProcessPool | None = None
        progressed = False

        for future, entry in list(in_flight.items()):
            if not future.done():
                continue
            try:
                future.result()
            except BrokenProcessPool as exc:
                # Keep one representative pool-level exception. Iteration/submission
                # order does not imply which entry actually caused the worker death.
                if broken is None:
                    broken = exc
                continue
            except Exception as exc:
                self._record_failure(execution_state, entry, exc)
            else:
                self._record_execution_status(
                    execution_state, entry, ExecutionStatus.SUCCESS
                )

            in_flight.pop(future)
            start_times.pop(entry.id, None)
            progress.update(1)
            progressed = True

        return broken, progressed

    def _mark_timeouts(
        self,
        in_flight: dict[Future[float], StagedEntry],
        execution_state: ExecutionState,
        start_times: dict[str, float],
        progress: Any,
    ) -> bool:
        if self._timeout is None:
            return False

        now = time.monotonic()
        overdue: list[tuple[Future[float], StagedEntry]] = []
        for future, entry in in_flight.items():
            if future.done():
                continue
            # Submitted does not necessarily mean started; a spawned worker may not yet
            # have entered _execute_one and therefore may not have recorded a start time.
            started = start_times.get(entry.id)
            if started is not None and now - started > self._timeout:
                overdue.append((future, entry))

        for future, entry in overdue:
            in_flight.pop(future)
            start_times.pop(entry.id, None)
            self._record_execution_status(
                execution_state, entry, ExecutionStatus.TIMEOUT
            )
            progress.update(1)

        return bool(overdue)

    def _rescue_flight(
        self,
        pool: ProcessPoolExecutor,
        in_flight: dict[Future[float], StagedEntry],
        execution_state: ExecutionState,
        start_queue: Any,
        start_times: dict[str, float],
        progress: Any,
        outcome: _PoolExit,
        broken: BrokenProcessPool | None,
        made_progress: bool,
    ) -> _PoolResult:
        """Reconcile completed work, kill the pool, and return unresolved casualties."""
        # Salvage start events already transmitted before forcefully terminating workers.
        # Killing a writer during queue I/O may make subsequent draining unreliable; the
        # post-reap drain below still captures events emitted in the meantime.
        self._drain_start_events(
            start_queue,
            in_flight,
            execution_state,
            start_times,
            mark_running=False,  # Will be terminated, so original status should be preserved.
        )

        # Best-effort final reconciliation reduces duplicate re-execution.
        final_broken, final_progress = self._consume_done(
            in_flight, execution_state, start_times, progress
        )
        made_progress |= final_progress

        # Timeout recovery can discover that the pool broke during final reconciliation;
        # this overrides the timeout outcome.
        if broken is None and final_broken is not None:
            broken = final_broken
            outcome = _PoolExit.BROKEN

        casualties = tuple(in_flight.values())

        self._terminate_pool(pool)
        try:
            self._drain_start_events(
                start_queue,
                in_flight,
                execution_state,
                start_times,
                mark_running=False,
            )
        except (EOFError, OSError) as exc:
            self.log(
                "Start-event channel became unreadable after pool termination; "
                "continuing with previously observed start checkpoints",
                level="warning",
                exc_info=(type(exc), exc, exc.__traceback__),
            )

        started_casualties = frozenset(
            entry.id for entry in casualties if start_times.get(entry.id) is not None
        )

        # Retain original status for started casualties not marked as RUNNING;
        # set the rest back to PENDING.
        running_casualties = {
            entry_id
            for entry_id in started_casualties
            if execution_state.get_status(entry_id) is ExecutionStatus.RUNNING
        }
        if running_casualties:
            execution_state.update(
                {entry_id: ExecutionStatus.PENDING for entry_id in running_casualties}
            )

        for entry_id in started_casualties:
            start_times.pop(entry_id, None)

        in_flight.clear()

        return _PoolResult(
            outcome=outcome,
            casualties=casualties,
            started_casualties=started_casualties,
            broken_pool=broken,
            made_progress=made_progress,
        )

    def _diagnose_entries(
        self,
        entries: Sequence[StagedEntry],
        execution_state: ExecutionState,
        parallel: ParallelLogging,
        progress: Any,
    ) -> None:
        """Diagnose ambiguous casualties using the normal pool driving engine.

        Diagnosis uses one worker and ``max_tasks_per_child=1``. Thus each entry runs in
        a fresh process, while timeout, ordinary failure, completion, and pool recovery
        remain exactly the same code paths as normal execution.
        """
        pending = deque(entries)
        consecutive_prestart_failures = 0

        while pending:
            result = self._drive_pool(
                pending,
                execution_state,
                parallel,
                progress,
                max_workers=1,
                max_tasks_per_child=1,
            )

            if result.outcome is _PoolExit.COMPLETE:
                return

            if result.outcome is _PoolExit.TIMEOUT:
                if result.casualties:
                    pending.extendleft(reversed(result.casualties))
                consecutive_prestart_failures = 0
                continue

            if len(result.casualties) > 1:
                raise RuntimeError(
                    "Internal scheduler error: isolated pool returned multiple casualties"
                )

            if result.made_progress:
                consecutive_prestart_failures = 0

            if result.casualties:
                entry = result.casualties[0]
                if entry.id in result.started_casualties:
                    # Exactly one entry was running and its worker died after its start
                    # signal, so the process death is attributable to this entry.
                    self._log_broken_pool(
                        result.broken_pool,
                        message=(
                            "Worker process terminated abruptly while processing "
                            f"isolated entry {entry.id}"
                        ),
                    )
                    self._record_execution_status(
                        execution_state,
                        entry,
                        ExecutionStatus.FAILURE,
                        detail=(
                            "worker process terminated abruptly during isolated execution: "
                            f"{self._compact(str(result.broken_pool))}"
                        ),
                    )
                    progress.update(1)
                    consecutive_prestart_failures = 0
                    continue

                # The worker died before this entry reached _execute_one; do not blame
                # the file. Put it back and retry in another fresh process.
                pending.extendleft(reversed(result.casualties))

            consecutive_prestart_failures += 1

            if consecutive_prestart_failures >= _MAX_DIAGNOSIS_PRESTART_FAILURES:
                self._log_broken_pool(
                    result.broken_pool,
                    message=(
                        "Isolated worker pool repeatedly failed before entry execution "
                        "began; treating failure as systemic"
                    ),
                )
                next_entry = pending[0].id if pending else "<unknown>"
                raise RuntimeError(
                    "Worker processes repeatedly failed before isolated execution "
                    f"could begin for `{next_entry}`; failure appears systemic"
                ) from result.broken_pool

            self._log_broken_pool(
                result.broken_pool,
                message=(
                    "Isolated worker pool failed before entry execution began; "
                    "retrying in a fresh pool"
                ),
            )

    def _log_broken_pool(
        self,
        exc: BrokenProcessPool | None,
        *,
        message: str = "Worker process pool became unusable",
    ) -> None:
        if exc is None:
            self.log(message, level="error")
            return
        self.log(
            message,
            level="error",
            exc_info=(type(exc), exc, exc.__traceback__),
        )

    def _poll_interval(self) -> float:
        if self._timeout is None:
            return _MAX_POLL_INTERVAL

        return min(
            _MAX_POLL_INTERVAL,
            max(_MIN_POLL_INTERVAL, self._timeout / 10),
        )

    @staticmethod
    def _terminate_pool(pool: ProcessPoolExecutor) -> None:
        """Force-stop and reap every worker; Python 3.13 needs a private fallback."""
        processes = dict(getattr(pool, "_processes", None) or {})
        terminate_workers = getattr(pool, "terminate_workers", None)

        if callable(terminate_workers):
            # Python 3.14+: public hard-termination API also performs executor shutdown.
            terminate_workers()
        else:
            # Python 3.13: shutdown cancels queued work but does not stop running calls.
            pool.shutdown(wait=False, cancel_futures=True)
            # Include a worker that may have been spawned between the first snapshot and
            # the executor observing shutdown.
            processes.update(dict(getattr(pool, "_processes", None) or {}))

            for process in processes.values():
                try:
                    if process.is_alive():
                        process.terminate()
                except (ProcessLookupError, ValueError):
                    pass

            # terminate() is asynchronous (SIGTERM on POSIX); briefly wait/reap before
            # escalating surviving processes to kill().
            deadline = time.monotonic() + _PROCESS_TERMINATE_GRACE
            for process in processes.values():
                try:
                    process.join(timeout=max(0.0, deadline - time.monotonic()))
                except (AssertionError, ValueError):
                    pass

        survivors = []
        for process in processes.values():
            try:
                if process.is_alive():
                    process.kill()
                    survivors.append(process)
            except (ProcessLookupError, ValueError):
                pass

        deadline = time.monotonic() + _PROCESS_KILL_GRACE
        for process in survivors:
            try:
                process.join(timeout=max(0.0, deadline - time.monotonic()))
            except (AssertionError, ValueError):
                pass

        unreaped = []
        for process in processes.values():
            try:
                if process.is_alive():
                    unreaped.append(process)
            except (ProcessLookupError, ValueError):
                pass

        if unreaped:
            raise RuntimeError(
                f"Failed to terminate and reap {len(unreaped)} process-pool "
                f"worker{'s' if len(unreaped) != 1 else ''}"
            )
