"""Tests for :class:`ProcessingWorkflow` and :class:`PlannableWorkflow`.

These exercise the base classes through minimal dummy subclasses so the execution
engine, constructor validation, and plan/run contract are covered without any
staging, discovery, or pipeline machinery. ``DynamicProcessingWorkflow`` is
tested separately in ``test_preproc_workflows_dynamic.py``.
"""

from __future__ import annotations

from collections import deque
from concurrent.futures import Future, ProcessPoolExecutor
from contextlib import nullcontext
import logging
import os
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from niiflow.preproc.staging import StagedEntry, StagingErrorRecord
from niiflow.preproc.workflows import (
    PlannableWorkflow,
    ProcessingWorkflow,
    RunPlan,
)
from niiflow.preproc.workflows.execution_state import (
    ExecutionState,
    ExecutionStatus,
)
from niiflow.preproc.workflows.workflow import (
    _PoolResult,
    _PoolExit,
    _execute_one,
)
from concurrent.futures.process import BrokenProcessPool

# Module-level so the dummy workflows stay picklable for process pools; a
# closure or method would not be.
_CALLS_FILENAME = "calls.log"


class _StartQueue:
    """Small in-process stand-in for multiprocessing.SimpleQueue."""

    def __init__(self) -> None:
        self.events: deque[tuple[str, float]] = deque()
        self.closed = False

    def put(self, event: tuple[str, float]) -> None:
        self.events.append(event)

    def empty(self) -> bool:
        return not self.events

    def get(self) -> tuple[str, float]:
        return self.events.popleft()

    def close(self) -> None:
        self.closed = True


def _record_call(entry: StagedEntry) -> None:
    out_dir = Path(entry.params["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / _CALLS_FILENAME).open("a", encoding="utf-8") as fh:
        fh.write(f"{entry.active.as_posix()}\n")


def _record_call_and_log(entry: StagedEntry) -> None:
    logging.getLogger().info("from worker")
    _record_call(entry)


def _boom(entry: StagedEntry) -> None:
    raise RuntimeError(f"boom for {entry.active}")


def _fail_selected(entry: StagedEntry) -> None:
    if entry.params.get("fail"):
        raise RuntimeError("boom")
    _record_call(entry)


def _sleep_then_fail(entry: StagedEntry) -> None:
    time.sleep(float(entry.params["sleep_seconds"]))
    raise RuntimeError("late boom")


def _crash_once_then_record(entry: StagedEntry) -> None:
    marker = Path(entry.params["crash_marker"])
    if entry.params.get("crash") and not marker.exists():
        marker.write_text("crashed", encoding="utf-8")
        os._exit(1)
    _record_call(entry)


def _always_crash_selected(entry: StagedEntry) -> None:
    if entry.params.get("crash"):
        os._exit(1)
    _record_call(entry)


def _sleep_then_record(entry: StagedEntry) -> None:
    """Sleep only for entries listed in ``slow_actives``, then record the call."""
    slow = {Path(item).resolve() for item in entry.params.get("slow_actives", ())}
    if entry.active in slow:
        time.sleep(float(entry.params.get("sleep_seconds", 2.0)))
    _record_call(entry)


class RecordingWorkflow(ProcessingWorkflow):
    """Execution-only workflow that appends each processed entry to a file."""

    @staticmethod
    def process_single(entry: StagedEntry) -> None:
        _record_call(entry)


class WorkerLoggingWorkflow(RecordingWorkflow):
    """Like :class:`RecordingWorkflow`, but also logs from the worker process."""

    @staticmethod
    def process_single(entry: StagedEntry) -> None:
        _record_call_and_log(entry)


class SleepingWorkflow(ProcessingWorkflow):
    """Execution-only workflow that can stall on selected entries."""

    @staticmethod
    def process_single(entry: StagedEntry) -> None:
        _sleep_then_record(entry)


class PlanningWorkflow(PlannableWorkflow):
    """Plannable workflow whose plan wraps each given path in one entry."""

    @staticmethod
    def process_single(entry: StagedEntry) -> None:
        _record_call(entry)

    def plan(self, source: list[Path]) -> RunPlan:
        return RunPlan(
            entries=tuple(
                StagedEntry(
                    active=Path(item).resolve(),
                    id=f"plan-{index}",
                    params={"out_dir": self.out_dir},  # type: ignore[attr-defined]
                )
                for index, item in enumerate(source)
            )
        )


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")
    return path


def _entry(
    active: Path,
    out_dir: Path,
    *,
    entry_id: str | None = None,
    errors: tuple = (),
    **params: object,
) -> StagedEntry:
    resolved = active.resolve()
    return StagedEntry(
        active=resolved,
        id=entry_id if entry_id is not None else str(resolved),
        params={"out_dir": str(out_dir), **params},
        errors=errors,
    )


def _recorded(out_dir: Path) -> list[str]:
    log = out_dir / _CALLS_FILENAME
    if not log.exists():
        return []
    return log.read_text(encoding="utf-8").split()


@pytest.fixture
def logs_dir(tmp_path: Path) -> Path:
    return tmp_path / "logs"


@pytest.fixture
def workflow(logs_dir: Path) -> RecordingWorkflow:
    return RecordingWorkflow(logs_root=logs_dir)


def _status_log(logs_dir: Path) -> str:
    return (logs_dir / "status.log").read_text(encoding="utf-8")


def _status_outcomes(status: str, entry_id: str) -> list[str]:
    prefix = f"{entry_id} |"
    outcomes: list[str] = []
    for line in status.splitlines():
        if line.startswith(prefix):
            outcomes.append(line.split("|", 1)[1].strip().split()[0])
    return outcomes


class _Progress:
    def __init__(self) -> None:
        self.updates: list[int] = []

    def update(self, value: int = 1) -> None:
        self.updates.append(value)


class TestAbstractContract:
    def test_processing_workflow_is_abstract(self) -> None:
        with pytest.raises(TypeError, match="process_single"):
            ProcessingWorkflow()  # type: ignore[abstract]

    def test_plannable_workflow_requires_plan(self, tmp_path: Path) -> None:
        class MissingPlan(PlannableWorkflow):
            @staticmethod
            def process_single(entry: StagedEntry) -> None:
                return None

        with pytest.raises(TypeError, match="plan"):
            MissingPlan()  # type: ignore[abstract]

    def test_plannable_extends_processing(self) -> None:
        assert issubclass(PlannableWorkflow, ProcessingWorkflow)

    def test_processing_workflow_does_not_define_plan(self) -> None:
        # Planning is deliberately not part of the execution-only base class.
        assert "plan" not in ProcessingWorkflow.__dict__


class TestNumWorkers:
    @pytest.mark.parametrize("value", [1, 2, 7])
    def test_accepts_positive_int(self, tmp_path: Path, value: int) -> None:
        wf = RecordingWorkflow(logs_root=tmp_path / "logs", num_workers=value)
        assert wf.num_workers == value

    def test_auto_resolves_to_cpu_count(self, tmp_path: Path) -> None:
        import multiprocessing

        wf = RecordingWorkflow(logs_root=tmp_path / "logs", num_workers="auto")
        assert wf.num_workers == multiprocessing.cpu_count()

    def test_defaults_to_one(self, workflow: RecordingWorkflow) -> None:
        assert workflow.num_workers == 1

    @pytest.mark.parametrize("value", [0, -3])
    def test_rejects_non_positive(self, tmp_path: Path, value: int) -> None:
        with pytest.raises(ValueError, match="integer >= 1"):
            RecordingWorkflow(logs_root=tmp_path / "logs", num_workers=value)

    @pytest.mark.parametrize("value", ["many", 1.5, None])
    def test_rejects_invalid_type(self, tmp_path: Path, value: object) -> None:
        with pytest.raises(ValueError, match="'auto' or an integer"):
            RecordingWorkflow(logs_root=tmp_path / "logs", num_workers=value)  # type: ignore[arg-type]

    def test_setter_revalidates(self, workflow: RecordingWorkflow) -> None:
        workflow.num_workers = 4
        assert workflow.num_workers == 4
        with pytest.raises(ValueError):
            workflow.num_workers = 0


class TestTimeout:
    def test_defaults_to_none(self, workflow: RecordingWorkflow) -> None:
        assert workflow.timeout is None

    @pytest.mark.parametrize("value", [1, 0.25, 30.0])
    def test_accepts_positive_number_as_float(
        self, tmp_path: Path, value: float
    ) -> None:
        wf = RecordingWorkflow(logs_root=tmp_path / "logs", timeout=value)
        assert wf.timeout == pytest.approx(float(value))
        assert isinstance(wf.timeout, float)

    @pytest.mark.parametrize("value", [0, -1.0])
    def test_rejects_non_positive(self, tmp_path: Path, value: float) -> None:
        with pytest.raises(ValueError, match="positive number"):
            RecordingWorkflow(logs_root=tmp_path / "logs", timeout=value)

    def test_rejects_non_number(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="number or None"):
            RecordingWorkflow(logs_root=tmp_path / "logs", timeout="10s")  # type: ignore[arg-type]

    def test_setter_accepts_none(self, workflow: RecordingWorkflow) -> None:
        workflow.timeout = 5
        workflow.timeout = None
        assert workflow.timeout is None


class TestRunEntries:
    def test_processes_every_entry_serially(
        self, workflow: RecordingWorkflow, tmp_path: Path
    ) -> None:
        out_dir = tmp_path / "out"
        actives = [tmp_path / f"img-{index}.nii.gz" for index in range(3)]
        for active in actives:
            active.write_bytes(b"")

        workflow.run_entries([_entry(active, out_dir) for active in actives])

        assert set(_recorded(out_dir)) == {a.resolve().as_posix() for a in actives}

    def test_empty_entries_is_a_noop(
        self, workflow: RecordingWorkflow, tmp_path: Path
    ) -> None:
        workflow.run_entries([])
        assert not (tmp_path / "out" / _CALLS_FILENAME).exists()

    def test_rejects_non_staged_entry(self, workflow: RecordingWorkflow) -> None:
        with pytest.raises(
            TypeError, match="`entries` must contain only `StagedEntry`"
        ):
            workflow.run_entries(["not-an-entry"])  # type: ignore[list-item]

    @pytest.mark.parametrize("entry_id", ["", None, 1])
    def test_rejects_invalid_entry_id_before_scheduling(
        self,
        workflow: RecordingWorkflow,
        tmp_path: Path,
        entry_id: object,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        processed: list[str] = []
        monkeypatch.setattr(
            workflow, "process_single", lambda entry: processed.append(entry.id)
        )
        entry = StagedEntry(
            active=(tmp_path / "a.nii.gz").resolve(),
            id=entry_id,  # type: ignore[arg-type]
            params={"out_dir": str(tmp_path / "out")},
        )

        with pytest.raises(ValueError, match="non-empty string `id`"):
            workflow.run_entries([entry])

        assert processed == []

    def test_rejects_duplicate_ids_before_scheduling(
        self,
        workflow: RecordingWorkflow,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        processed: list[str] = []
        monkeypatch.setattr(
            workflow, "process_single", lambda entry: processed.append(entry.id)
        )
        active = tmp_path / "same.nii.gz"
        entries = [
            _entry(active, tmp_path / "out", entry_id="duplicate"),
            _entry(active, tmp_path / "out", entry_id="duplicate"),
        ]

        with pytest.raises(ValueError, match=r"duplicate `duplicate`"):
            workflow.run_entries(entries)

        assert processed == []

    def test_skips_entries_with_staging_errors(
        self, workflow: RecordingWorkflow, tmp_path: Path, logs_dir: Path
    ) -> None:
        out_dir = tmp_path / "out"
        good = _touch(tmp_path / "good.nii.gz")
        bad = _touch(tmp_path / "bad.nii.gz")
        broken = _entry(
            bad,
            out_dir,
            errors=(
                StagingErrorRecord(active=bad.resolve(), message="missing companion"),
            ),
        )

        workflow.run_entries([_entry(good, out_dir), broken])

        assert _recorded(out_dir) == [good.resolve().as_posix()]
        assert f"{bad.resolve()} | STAGING_FAILURE | missing companion" in _status_log(
            logs_dir
        )

    def test_all_entries_failing_staging_is_a_noop(
        self, workflow: RecordingWorkflow, tmp_path: Path
    ) -> None:
        out_dir = tmp_path / "out"
        active = tmp_path / "bad.nii.gz"
        active.write_bytes(b"")
        broken = _entry(
            active,
            out_dir,
            errors=(StagingErrorRecord(active=active.resolve(), message="missing"),),
        )

        workflow.run_entries([broken])

        assert _recorded(out_dir) == []

    def test_continues_after_entry_failure(
        self,
        workflow: RecordingWorkflow,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        out_dir = tmp_path / "out"
        actives = [tmp_path / f"img-{index}.nii.gz" for index in range(3)]
        for active in actives:
            active.write_bytes(b"")

        def _fail_middle(entry: StagedEntry) -> None:
            if entry.active.name == "img-1.nii.gz":
                raise RuntimeError("synthetic failure")
            _record_call(entry)

        monkeypatch.setattr(workflow, "process_single", _fail_middle)
        workflow.run_entries([_entry(active, out_dir) for active in actives])

        assert set(_recorded(out_dir)) == {
            actives[0].resolve().as_posix(),
            actives[2].resolve().as_posix(),
        }

    def test_failures_are_reported_to_status_log(
        self,
        workflow: RecordingWorkflow,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        statuses: list[str] = []
        monkeypatch.setattr(workflow, "status", statuses.append)
        monkeypatch.setattr(workflow, "process_single", _boom)
        active = tmp_path / "a.nii.gz"
        active.write_bytes(b"")

        workflow.run_entries([_entry(active, tmp_path / "out")])

        assert len(statuses) == 1
        assert "FAILURE" in statuses[0]
        assert "RuntimeError" in statuses[0]

    def test_successes_are_reported_to_status_log(
        self,
        workflow: RecordingWorkflow,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        statuses: list[str] = []
        monkeypatch.setattr(workflow, "status", statuses.append)
        active = tmp_path / "a.nii.gz"
        active.write_bytes(b"")

        workflow.run_entries([_entry(active, tmp_path / "out")])

        assert statuses == [f"{active.resolve()} | SUCCESS"]

    def test_status_and_execution_state_use_id_not_duplicate_active(
        self,
        workflow: RecordingWorkflow,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        active = _touch(tmp_path / "same.nii.gz")
        called: list[str] = []
        statuses: list[str] = []
        monkeypatch.setattr(
            workflow, "process_single", lambda entry: called.append(entry.id)
        )
        monkeypatch.setattr(workflow, "status", statuses.append)
        entries = [
            _entry(active, tmp_path / "out", entry_id="logical"),
            _entry(active, tmp_path / "out", entry_id="logical#2"),
        ]

        state = workflow.run_entries(entries)

        assert called == ["logical", "logical#2"]
        assert statuses == ["logical | SUCCESS", "logical#2 | SUCCESS"]
        assert state.select([ExecutionStatus.SUCCESS]) == ("logical", "logical#2")

    def test_supplied_state_is_reused_and_may_contain_other_ids(
        self,
        workflow: RecordingWorkflow,
        tmp_path: Path,
    ) -> None:
        selected = _entry(
            tmp_path / "selected.nii.gz", tmp_path / "out", entry_id="selected"
        )
        outside = _entry(
            tmp_path / "outside.nii.gz", tmp_path / "out", entry_id="outside"
        )
        state = ExecutionState.from_entries([selected, outside])
        state.update({"outside": ExecutionStatus.TIMEOUT})

        returned = workflow.run_entries([selected], execution_state=state)

        assert returned is state
        assert state.get_status("selected") is ExecutionStatus.SUCCESS
        assert state.get_status("outside") is ExecutionStatus.TIMEOUT

    def test_supplied_state_adds_unseen_requested_ids_as_pending(
        self, workflow: RecordingWorkflow, tmp_path: Path
    ) -> None:
        unseen = _entry(tmp_path / "a.nii.gz", tmp_path / "out", entry_id="unseen")
        historical = _entry(
            tmp_path / "b.nii.gz", tmp_path / "out", entry_id="historical"
        )
        state = ExecutionState.from_entries([historical])
        state.update({"historical": ExecutionStatus.TIMEOUT})

        returned = workflow._prepare_execution_state([unseen], state)

        assert returned is state
        assert tuple(state.statuses) == ("historical", "unseen")
        assert state.get_status("historical") is ExecutionStatus.TIMEOUT
        assert state.get_status("unseen") is ExecutionStatus.PENDING

    def test_rejects_non_execution_state(
        self, workflow: RecordingWorkflow, tmp_path: Path
    ) -> None:
        entry = _entry(tmp_path / "a.nii.gz", tmp_path / "out")
        with pytest.raises(TypeError, match="must be an ExecutionState"):
            workflow.run_entries([entry], execution_state={})  # type: ignore[arg-type]

    def test_run_statuses_filter_uses_pre_run_state_and_preserves_excluded_entries(
        self,
        workflow: RecordingWorkflow,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        active = _touch(tmp_path / "same.nii.gz")
        entries = [
            _entry(active, tmp_path / "out", entry_id="success"),
            _entry(active, tmp_path / "out", entry_id="failure"),
            _entry(active, tmp_path / "out", entry_id="timeout"),
        ]
        state = ExecutionState.from_entries(entries)
        state.update(
            {
                "success": ExecutionStatus.SUCCESS,
                "failure": ExecutionStatus.FAILURE,
                "timeout": ExecutionStatus.TIMEOUT,
            }
        )
        called: list[str] = []
        monkeypatch.setattr(
            workflow, "process_single", lambda entry: called.append(entry.id)
        )

        returned = workflow.run_entries(
            entries,
            execution_state=state,
            run_statuses={ExecutionStatus.FAILURE},
        )

        assert returned is state
        assert called == ["failure"]
        assert state.statuses == {
            "success": ExecutionStatus.SUCCESS,
            "failure": ExecutionStatus.SUCCESS,
            "timeout": ExecutionStatus.TIMEOUT,
        }

    def test_run_statuses_none_runs_entries_regardless_of_existing_status(
        self,
        workflow: RecordingWorkflow,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        entry = _entry(tmp_path / "a.nii.gz", tmp_path / "out", entry_id="done")
        state = ExecutionState.from_entries([entry])
        state.update({"done": ExecutionStatus.SUCCESS})
        called: list[str] = []
        monkeypatch.setattr(
            workflow, "process_single", lambda item: called.append(item.id)
        )

        workflow.run_entries([entry], execution_state=state, run_statuses=None)

        assert called == ["done"]

    def test_rejects_invalid_run_status_without_mutating_state(
        self, workflow: RecordingWorkflow, tmp_path: Path
    ) -> None:
        entry = _entry(tmp_path / "a.nii.gz", tmp_path / "out", entry_id="entry")
        state = ExecutionState.from_entries([entry])

        with pytest.raises(TypeError, match="only ExecutionStatus"):
            workflow.run_entries(
                [entry],
                execution_state=state,
                run_statuses={"PENDING"},  # type: ignore[arg-type]
            )

        assert state.get_status("entry") is ExecutionStatus.PENDING

    def test_staging_failure_does_not_change_existing_execution_status(
        self, workflow: RecordingWorkflow, tmp_path: Path
    ) -> None:
        entry = _entry(
            tmp_path / "a.nii.gz",
            tmp_path / "out",
            entry_id="staging-failed",
            errors=(
                StagingErrorRecord(
                    active=tmp_path / "a.nii.gz",
                    message="missing companion",
                    entry_id="staging-failed",
                ),
            ),
        )
        state = ExecutionState.from_entries([entry])
        state.update({"staging-failed": ExecutionStatus.SUCCESS})

        workflow.run_entries([entry], execution_state=state)

        assert state.get_status("staging-failed") is ExecutionStatus.SUCCESS

    def test_running_and_terminal_transitions_are_incremental(
        self,
        workflow: RecordingWorkflow,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        entry = _entry(tmp_path / "a.nii.gz", tmp_path / "out", entry_id="entry")
        state = ExecutionState.from_entries([entry])
        transitions: list[ExecutionStatus] = []
        original_update = state.update

        def _record(updates: dict[str, ExecutionStatus]) -> None:
            transitions.extend(updates.values())
            original_update(updates)

        monkeypatch.setattr(state, "update", _record)

        workflow.run_entries([entry], execution_state=state)

        assert transitions == [ExecutionStatus.RUNNING, ExecutionStatus.SUCCESS]

    def test_state_is_saved_and_bound_before_processing(
        self,
        workflow: RecordingWorkflow,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        entry = _entry(tmp_path / "a.nii.gz", tmp_path / "out", entry_id="entry")
        state_path = tmp_path / "state.duckdb"
        observed: list[tuple[Path | None, bool]] = []
        monkeypatch.setattr(
            workflow,
            "process_single",
            lambda item: observed.append((state.path, state_path.is_file())),
        )
        state = ExecutionState.from_entries([entry])

        returned = workflow.run_entries(
            [entry],
            execution_state=state,
            save_execution_state_to=state_path,
        )

        assert returned is state
        assert observed == [(state_path.resolve(), True)]
        state.close()
        with ExecutionState.load(state_path) as restored:
            assert restored.get_status("entry") is ExecutionStatus.SUCCESS

    def test_empty_workload_logs_completion_without_empty_state_summary(
        self,
        workflow: RecordingWorkflow,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        messages: list[str] = []
        monkeypatch.setattr(workflow, "log", messages.append)

        workflow.run_entries([])

        assert messages[-1] == "Workflow complete"
        assert not any("state for runnable entries" in message for message in messages)

    def test_state_summaries_project_only_runnable_ids(
        self,
        workflow: RecordingWorkflow,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        runnable = _entry(tmp_path / "run.nii.gz", tmp_path / "out", entry_id="run")
        outside = _entry(
            tmp_path / "outside.nii.gz", tmp_path / "out", entry_id="outside"
        )
        state = ExecutionState.from_entries([runnable, outside])
        state.update({"outside": ExecutionStatus.FAILURE})
        messages: list[str] = []
        monkeypatch.setattr(workflow, "log", messages.append)

        workflow.run_entries([runnable], execution_state=state)

        summaries = [message for message in messages if "runnable entries" in message]
        assert len(summaries) == 2
        assert all("ExecutionState: 1 entries" in message for message in summaries)
        assert "pending=1" in summaries[0]
        assert "success=1" in summaries[1]
        assert all("failure=0" in message for message in summaries)

    def test_systemic_exception_does_not_log_completion(
        self,
        workflow: RecordingWorkflow,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        messages: list[str] = []
        monkeypatch.setattr(workflow, "log", messages.append)
        monkeypatch.setattr(
            workflow,
            "process_single",
            lambda entry: (_ for _ in ()).throw(KeyboardInterrupt()),
        )

        with pytest.raises(KeyboardInterrupt):
            workflow.run_entries([_entry(tmp_path / "a.nii.gz", tmp_path / "out")])

        assert "Workflow complete" not in messages

    def test_run_delegates_to_run_entries(
        self, workflow: RecordingWorkflow, tmp_path: Path
    ) -> None:
        out_dir = tmp_path / "out"
        active = tmp_path / "a.nii.gz"
        active.write_bytes(b"")

        workflow.run([_entry(active, out_dir)])

        assert _recorded(out_dir) == [active.resolve().as_posix()]

    def test_call_dunder_delegates_to_run(
        self, workflow: RecordingWorkflow, tmp_path: Path
    ) -> None:
        out_dir = tmp_path / "out"
        active = tmp_path / "a.nii.gz"
        active.write_bytes(b"")

        workflow([_entry(active, out_dir)])

        assert _recorded(out_dir) == [active.resolve().as_posix()]


class TestParallelExecution:
    def test_submission_window_is_bounded_by_worker_count(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        submitted: list[Future[float]] = []
        observed_windows: list[int] = []
        start_queue: object | None = None

        class _FakePool:
            def __init__(self, **kwargs: object) -> None:
                nonlocal start_queue
                assert kwargs["max_workers"] == 3
                start_queue = kwargs["initargs"][1]  # type: ignore[index]

            def submit(self, *args: object) -> Future[float]:
                future: Future[float] = Future()
                submitted.append(future)
                return future

            def shutdown(self, wait: bool = True, **kwargs: object) -> None:
                assert wait is True

        def _complete_one(futures: tuple[Future[float], ...], **kwargs: object) -> None:
            observed_windows.append(len(futures))
            future = futures[0]
            index = submitted.index(future)
            assert start_queue is not None
            start_queue.put((f"entry-{index}", time.monotonic()))  # type: ignore[union-attr]
            future.set_result(0.0)

        monkeypatch.setattr(
            "niiflow.preproc.workflows.workflow.ProcessPoolExecutor",
            _FakePool,
        )
        monkeypatch.setattr(
            "niiflow.preproc.workflows.workflow.wait",
            _complete_one,
        )

        wf = RecordingWorkflow(
            logs_root=tmp_path / "logs",
            num_workers=3,
        )
        entries = [
            _entry(
                tmp_path / f"img-{index}.nii.gz",
                tmp_path / "out",
                entry_id=f"entry-{index}",
            )
            for index in range(100)
        ]
        pending = deque(entries)
        state = ExecutionState.from_entries(entries)
        progress = _Progress()

        result = wf._drive_pool(
            pending,
            state,
            SimpleNamespace(worker_init_fn=lambda: None),  # type: ignore[arg-type]
            progress,
            max_workers=3,
        )

        assert result.outcome is _PoolExit.COMPLETE
        assert max(observed_windows) == 3
        assert all(window <= 3 for window in observed_windows)
        assert len(submitted) == len(entries)
        assert not pending
        assert set(state.statuses.values()) == {ExecutionStatus.SUCCESS}

    def test_submit_does_not_mark_running_before_worker_start_signal(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class _StopPoolRun(RuntimeError):
            pass

        class _FakePool:
            def __init__(self, **kwargs: object) -> None:
                pass

            def submit(self, *args: object) -> Future[float]:
                return Future()

        entry = _entry(tmp_path / "a.nii.gz", tmp_path / "out", entry_id="entry")
        state = ExecutionState.from_entries([entry])

        def _inspect_after_submit(
            futures: tuple[Future[float], ...],
            **kwargs: object,
        ) -> None:
            assert len(futures) == 1
            assert state.get_status("entry") is ExecutionStatus.PENDING
            raise _StopPoolRun

        monkeypatch.setattr(
            "niiflow.preproc.workflows.workflow.ProcessPoolExecutor",
            _FakePool,
        )
        monkeypatch.setattr(
            "niiflow.preproc.workflows.workflow.wait",
            _inspect_after_submit,
        )
        monkeypatch.setattr(
            RecordingWorkflow,
            "_terminate_pool",
            staticmethod(lambda pool: None),
        )
        wf = RecordingWorkflow(logs_root=tmp_path / "logs", num_workers=2)

        with pytest.raises(_StopPoolRun):
            wf._drive_pool(
                deque([entry]),
                state,
                SimpleNamespace(worker_init_fn=lambda: None),  # type: ignore[arg-type]
                _Progress(),
                max_workers=2,
            )

    def test_completed_task_is_not_marked_running_when_start_is_observed_late(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        start_queue: object | None = None

        class _FakePool:
            def __init__(self, **kwargs: object) -> None:
                nonlocal start_queue
                start_queue = kwargs["initargs"][1]  # type: ignore[index]

            def submit(self, *args: object) -> Future[float]:
                assert start_queue is not None
                start_queue.put(("entry", time.monotonic()))  # type: ignore[union-attr]
                future: Future[float] = Future()
                future.set_result(0.0)
                return future

            def shutdown(self, wait: bool = True, **kwargs: object) -> None:
                assert wait is True

        monkeypatch.setattr(
            "niiflow.preproc.workflows.workflow.ProcessPoolExecutor",
            _FakePool,
        )
        monkeypatch.setattr(
            "niiflow.preproc.workflows.workflow.wait",
            lambda *args, **kwargs: None,
        )
        entry = _entry(tmp_path / "a.nii.gz", tmp_path / "out", entry_id="entry")
        state = ExecutionState.from_entries([entry])
        transitions: list[ExecutionStatus] = []
        original_update = state.update

        def _record(updates: dict[str, ExecutionStatus]) -> None:
            transitions.extend(updates.values())
            original_update(updates)

        monkeypatch.setattr(state, "update", _record)
        wf = RecordingWorkflow(logs_root=tmp_path / "logs", num_workers=2)

        result = wf._drive_pool(
            deque([entry]),
            state,
            SimpleNamespace(worker_init_fn=lambda: None),  # type: ignore[arg-type]
            _Progress(),
            max_workers=2,
        )

        assert result.outcome is _PoolExit.COMPLETE
        assert transitions == [ExecutionStatus.SUCCESS]

    def test_scheduler_poll_interval_is_finite_without_timeout(
        self, tmp_path: Path
    ) -> None:
        # !Too low-level. It should check instead that a task lasting long
        # can still get marked as running without having a timeout. You can
        # patch _MAX_POLL_INTERVAL to be smaller than the task duration instead
        # of waiting full 0.5s.
        wf = RecordingWorkflow(logs_root=tmp_path / "logs", timeout=None)
        assert wf._poll_interval() == pytest.approx(0.5)

    @pytest.mark.parametrize("num_workers", [2, 4])
    def test_each_entry_is_processed_exactly_once(
        self, tmp_path: Path, num_workers: int
    ) -> None:
        out_dir = tmp_path / "out"
        wf = RecordingWorkflow(logs_root=tmp_path / "logs", num_workers=num_workers)
        actives = [_touch(tmp_path / f"img-{index}.nii.gz") for index in range(7)]

        wf.run_entries([_entry(active, out_dir) for active in actives])

        recorded = _recorded(out_dir)
        assert sorted(recorded) == sorted(a.resolve().as_posix() for a in actives)

    @pytest.mark.parametrize("num_workers", [2, 4])
    def test_spawns_num_workers_processes(
        self,
        tmp_path: Path,
        num_workers: int,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        spawned: list[int] = []
        contexts: list[object] = []

        class _SpyPool(ProcessPoolExecutor):
            def __init__(self, *args, **kwargs):
                contexts.append(kwargs.get("mp_context"))
                super().__init__(*args, **kwargs)

            def shutdown(self, wait=True, **kwargs):
                spawned.append(len(self._processes))
                return super().shutdown(wait=wait, **kwargs)

        monkeypatch.setattr(
            "niiflow.preproc.workflows.workflow.ProcessPoolExecutor",
            _SpyPool,
        )

        out_dir = tmp_path / "out"
        wf = RecordingWorkflow(logs_root=tmp_path / "logs", num_workers=num_workers)
        actives = [_touch(tmp_path / f"img-{index}.nii.gz") for index in range(7)]
        wf.run_entries([_entry(active, out_dir) for active in actives])

        assert spawned == [num_workers]
        assert len(contexts) == 1
        assert contexts[0] is wf._logging_manager.mp_context
        assert contexts[0].get_start_method() == "spawn"  # type: ignore[union-attr]

    def test_parallel_failure_is_compact_and_other_entries_continue(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        logs_root = tmp_path / "logs"
        out_dir = tmp_path / "out"
        wf = RecordingWorkflow(logs_root=logs_root, num_workers=2)
        actives = [_touch(tmp_path / f"img-{index}.nii.gz") for index in range(3)]
        entries = [
            _entry(active, out_dir, fail=index == 1)
            for index, active in enumerate(actives)
        ]
        wf.process_single = _fail_selected  # type: ignore[method-assign]
        diagnosis_calls: list[tuple[object, ...]] = []
        monkeypatch.setattr(
            wf,
            "_diagnose_entries",
            lambda *args: diagnosis_calls.append(args),
        )

        state = wf.run_entries(entries)

        status_lines = _status_log(logs_root).splitlines()
        failure_lines = [line for line in status_lines if " | FAILURE | " in line]
        assert failure_lines == [
            f"{actives[1].resolve()} | FAILURE | RuntimeError: boom"
        ]
        assert "Traceback (most recent call last)" not in _status_log(logs_root)
        assert "Traceback (most recent call last)" in (
            logs_root / "main.log"
        ).read_text(encoding="utf-8")
        assert set(_recorded(out_dir)) == {
            actives[0].resolve().as_posix(),
            actives[2].resolve().as_posix(),
        }
        assert state.get_status(entries[1].id) is ExecutionStatus.FAILURE
        assert all(
            state.get_status(entry.id) is ExecutionStatus.SUCCESS
            for entry in (entries[0], entries[2])
        )
        assert diagnosis_calls == []

    def test_broken_pool_requeues_without_status_amplification(
        self, tmp_path: Path
    ) -> None:
        logs_root = tmp_path / "logs"
        out_dir = tmp_path / "out"
        marker = tmp_path / "crash-marker"
        wf = RecordingWorkflow(logs_root=logs_root, num_workers=2)
        wf.process_single = _crash_once_then_record  # type: ignore[method-assign]
        actives = [_touch(tmp_path / f"img-{index}.nii.gz") for index in range(40)]
        entries = [
            _entry(
                active,
                out_dir,
                crash=index == 0,
                crash_marker=str(marker),
            )
            for index, active in enumerate(actives)
        ]

        wf.run_entries(entries)

        status = _status_log(logs_root)
        status_lines = status.splitlines()
        main_log = (logs_root / "main.log").read_text(encoding="utf-8")
        assert len(status_lines) == len(entries)
        assert all(" | SUCCESS" in line for line in status_lines)
        assert "Traceback (most recent call last)" not in status
        assert main_log.count("Worker process pool became unusable") == 1
        assert main_log.count("Traceback (most recent call last)") == 1
        assert set(_recorded(out_dir)) == {
            active.resolve().as_posix() for active in actives
        }

    @pytest.mark.serial
    def test_isolated_started_worker_death_is_attributed_to_that_entry(
        self, tmp_path: Path
    ) -> None:
        logs_root = tmp_path / "logs"
        out_dir = tmp_path / "out"
        wf = RecordingWorkflow(logs_root=logs_root, num_workers=2)
        wf.process_single = _always_crash_selected  # type: ignore[method-assign]
        actives = [_touch(tmp_path / f"img-{index}.nii.gz") for index in range(20)]
        entries = [
            _entry(active, out_dir, crash=index == 0)
            for index, active in enumerate(actives)
        ]

        state = wf.run_entries(entries)

        status = _status_log(logs_root)
        assert _status_outcomes(status, entries[0].id) == ["FAILURE"]
        assert state.get_status(entries[0].id) is ExecutionStatus.FAILURE
        assert all(
            state.get_status(entry.id) is ExecutionStatus.SUCCESS
            for entry in entries[1:]
        )


class TestPoolDrivingContracts:
    def test_execute_one_emits_start_before_processing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        entry = _entry(tmp_path / "a.nii.gz", tmp_path / "out", entry_id="entry")
        start_queue = _StartQueue()
        observed_starts: list[float] = []
        monkeypatch.setattr(
            "niiflow.preproc.workflows.workflow._WORKER_CONTEXT",
            SimpleNamespace(start_queue=start_queue),
        )

        def _process(item: StagedEntry) -> None:
            assert item is entry
            event_id, started = start_queue.events[0]
            assert event_id == item.id
            observed_starts.append(started)

        elapsed = _execute_one(_process, entry)

        assert observed_starts == [start_queue.events[0][1]]
        assert elapsed >= 0

    def test_drain_marks_only_currently_active_started_entry_running(
        self, tmp_path: Path
    ) -> None:
        wf = RecordingWorkflow(logs_root=tmp_path / "logs", num_workers=2)
        entry = _entry(tmp_path / "a.nii.gz", tmp_path / "out", entry_id="entry")
        state = ExecutionState.from_entries([entry])
        future: Future[float] = Future()
        start_queue = _StartQueue()
        start_queue.put(("entry", 10.0))
        start_times: dict[str, float] = {}

        wf._drain_start_events(
            start_queue,
            {future: entry},
            state,
            start_times,
        )

        assert state.get_status("entry") is ExecutionStatus.RUNNING
        assert start_times == {"entry": 10.0}

    def test_drain_start_events_records_start_without_marking_running(
        self, tmp_path: Path
    ) -> None:
        wf = RecordingWorkflow(logs_root=tmp_path / "logs", num_workers=2)
        entry = _entry(tmp_path / "a.nii.gz", tmp_path / "out", entry_id="entry")
        state = ExecutionState.from_entries([entry])
        future: Future[float] = Future()
        start_queue = _StartQueue()
        start_queue.put(("entry", 10.0))
        start_times: dict[str, float] = {}

        wf._drain_start_events(
            start_queue,
            {future: entry},
            state,
            start_times,
            mark_running=False,
        )

        assert state.get_status("entry") is ExecutionStatus.PENDING
        assert start_times == {"entry": 10.0}

    def test_timeout_clock_starts_only_after_worker_start_signal(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        wf = RecordingWorkflow(logs_root=tmp_path / "logs", num_workers=2, timeout=5)
        started = _entry(
            tmp_path / "started.nii.gz", tmp_path / "out", entry_id="started"
        )
        queued = _entry(tmp_path / "queued.nii.gz", tmp_path / "out", entry_id="queued")
        state = ExecutionState.from_entries([started, queued])
        state.update({"started": ExecutionStatus.RUNNING})
        started_future: Future[float] = Future()
        queued_future: Future[float] = Future()
        in_flight = {started_future: started, queued_future: queued}
        start_times = {"started": 10.0}
        monkeypatch.setattr(
            "niiflow.preproc.workflows.workflow.time.monotonic", lambda: 20.0
        )

        timed_out = wf._mark_timeouts(
            in_flight,
            state,
            start_times,
            _Progress(),
        )

        assert timed_out is True
        assert state.get_status("started") is ExecutionStatus.TIMEOUT
        assert state.get_status("queued") is ExecutionStatus.PENDING
        assert list(in_flight.values()) == [queued]
        assert start_times == {}

    def test_consume_done_records_ordinary_failure_without_broken_pool(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        wf = RecordingWorkflow(logs_root=tmp_path / "logs", num_workers=2)
        success = _entry(tmp_path / "ok.nii.gz", tmp_path / "out", entry_id="ok")
        failure = _entry(tmp_path / "bad.nii.gz", tmp_path / "out", entry_id="bad")
        state = ExecutionState.from_entries([success, failure])
        state.update(
            {
                "ok": ExecutionStatus.RUNNING,
                "bad": ExecutionStatus.RUNNING,
            }
        )
        success_future: Future[float] = Future()
        failure_future: Future[float] = Future()
        success_future.set_result(0.1)
        failure_future.set_exception(RuntimeError("ordinary"))
        in_flight = {success_future: success, failure_future: failure}
        progress = _Progress()
        diagnosis_calls: list[tuple[object, ...]] = []
        monkeypatch.setattr(
            wf,
            "_diagnose_entries",
            lambda *args: diagnosis_calls.append(args),
        )

        start_times = {
            "ok": time.monotonic(),
            "bad": time.monotonic(),
        }
        broken, made_progress = wf._consume_done(
            in_flight, state, start_times, progress
        )

        assert broken is None
        assert made_progress is True
        assert not in_flight
        assert state.get_status("ok") is ExecutionStatus.SUCCESS
        assert state.get_status("bad") is ExecutionStatus.FAILURE
        assert progress.updates == [1, 1]
        assert start_times == {}
        assert diagnosis_calls == []

    def test_recovery_reconciles_then_uses_post_termination_start_snapshot(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        wf = RecordingWorkflow(logs_root=tmp_path / "logs", num_workers=2)
        completed = _entry(tmp_path / "done.nii.gz", tmp_path / "out", entry_id="done")
        running = _entry(
            tmp_path / "running.nii.gz", tmp_path / "out", entry_id="running"
        )
        failed = _entry(tmp_path / "failed.nii.gz", tmp_path / "out", entry_id="failed")
        timed_out = _entry(
            tmp_path / "timed-out.nii.gz", tmp_path / "out", entry_id="timed-out"
        )
        untouched = _entry(
            tmp_path / "pending.nii.gz", tmp_path / "out", entry_id="pending"
        )
        state = ExecutionState.from_entries(
            [completed, running, failed, timed_out, untouched]
        )
        state.update(
            {
                "done": ExecutionStatus.RUNNING,
                "running": ExecutionStatus.RUNNING,
                "failed": ExecutionStatus.FAILURE,
                "timed-out": ExecutionStatus.TIMEOUT,
            }
        )
        done_future: Future[float] = Future()
        running_future: Future[float] = Future()
        failed_future: Future[float] = Future()
        timed_out_future: Future[float] = Future()
        done_future.set_result(0.1)
        in_flight = {
            done_future: completed,
            running_future: running,
            failed_future: failed,
            timed_out_future: timed_out,
        }
        start_queue = _StartQueue()
        start_queue.put(("done", time.monotonic()))
        start_times: dict[str, float] = {}
        terminated: list[object] = []
        pool = object()

        def _terminate(value: object) -> None:
            terminated.append(value)
            start_queue.put(("running", time.monotonic()))
            start_queue.put(("failed", time.monotonic()))

        monkeypatch.setattr(
            wf,
            "_terminate_pool",
            _terminate,
        )

        result = wf._rescue_flight(
            pool,  # type: ignore[arg-type]
            in_flight,
            state,
            start_queue,
            start_times,
            _Progress(),
            _PoolExit.TIMEOUT,
            None,
            False,
        )

        assert result == _PoolResult(
            outcome=_PoolExit.TIMEOUT,
            casualties=(running, failed, timed_out),
            started_casualties=frozenset({"running", "failed"}),
            made_progress=True,
        )
        assert state.get_status("done") is ExecutionStatus.SUCCESS
        assert state.get_status("running") is ExecutionStatus.PENDING
        assert state.get_status("failed") is ExecutionStatus.FAILURE
        assert state.get_status("timed-out") is ExecutionStatus.TIMEOUT
        assert state.get_status("pending") is ExecutionStatus.PENDING
        assert not in_flight
        assert start_times == {}
        assert terminated == [pool]

    def test_final_broken_future_overrides_timeout_outcome(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        wf = RecordingWorkflow(logs_root=tmp_path / "logs")
        entry = _entry(tmp_path / "a.nii.gz", tmp_path / "out", entry_id="entry")
        state = ExecutionState.from_entries([entry])
        future: Future[float] = Future()
        broken = BrokenProcessPool("pool died")
        future.set_exception(broken)
        start_queue = _StartQueue()
        start_queue.put(("entry", time.monotonic()))
        monkeypatch.setattr(wf, "_terminate_pool", lambda pool: None)

        result = wf._rescue_flight(
            object(),  # type: ignore[arg-type]
            {future: entry},
            state,
            start_queue,
            {},
            _Progress(),
            _PoolExit.TIMEOUT,
            None,
            False,
        )

        assert result.outcome is _PoolExit.BROKEN
        assert result.broken_pool is broken
        assert result.casualties == (entry,)
        assert result.started_casualties == frozenset({"entry"})

    def test_post_termination_channel_error_keeps_observed_starts(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class _CorruptStartQueue(_StartQueue):
            corrupted = False

            def empty(self) -> bool:
                if self.corrupted:
                    raise OSError("corrupt start channel")
                return super().empty()

        wf = RecordingWorkflow(logs_root=tmp_path / "logs")
        entry = _entry(tmp_path / "a.nii.gz", tmp_path / "out", entry_id="entry")
        state = ExecutionState.from_entries([entry])
        future: Future[float] = Future()
        start_queue = _CorruptStartQueue()
        start_queue.put(("entry", time.monotonic()))
        warnings: list[str] = []

        def _terminate(pool: object) -> None:
            start_queue.corrupted = True

        monkeypatch.setattr(wf, "_terminate_pool", _terminate)
        monkeypatch.setattr(
            wf,
            "log",
            lambda message, **kwargs: warnings.append(message),
        )

        result = wf._rescue_flight(
            object(),  # type: ignore[arg-type]
            {future: entry},
            state,
            start_queue,
            {},
            _Progress(),
            _PoolExit.BROKEN,
            BrokenProcessPool("pool died"),
            False,
        )

        assert result.started_casualties == frozenset({"entry"})
        assert state.get_status("entry") is ExecutionStatus.PENDING
        assert warnings == [
            "Start-event channel became unreadable after pool termination; "
            "continuing with previously observed start checkpoints"
        ]

    def test_timeout_requeues_collateral_but_not_overdue_entry(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # !To me this test seems quite obvious, as we manually remove the overdue
        # entry from the casualties, and consequently prevent requening. The "magic"
        # of timeout handling is that the _mark_timeouts method automatically removes
        # the overdue entry from the in_flight and hence the casualties.!
        wf = RecordingWorkflow(logs_root=tmp_path / "logs", num_workers=2)
        entries = [
            _entry(
                tmp_path / f"{name}.nii.gz",
                tmp_path / "out",
                entry_id=name,
            )
            for name in ("overdue", "collateral", "untouched")
        ]
        state = ExecutionState.from_entries(entries)
        pool_runs: list[tuple[str, ...]] = []

        def _drive_pool(
            pending: deque[StagedEntry],
            *args: object,
            **kwargs: object,
        ) -> _PoolResult:
            pool_runs.append(tuple(entry.id for entry in pending))
            if len(pool_runs) == 1:
                overdue = pending.popleft()
                collateral = pending.popleft()
                state.update({overdue.id: ExecutionStatus.TIMEOUT})
                return _PoolResult(
                    _PoolExit.TIMEOUT,
                    casualties=(collateral,),
                    made_progress=True,
                )
            while pending:
                state.update({pending.popleft().id: ExecutionStatus.SUCCESS})
            return _PoolResult(_PoolExit.COMPLETE, made_progress=True)

        monkeypatch.setattr(wf, "_drive_pool", _drive_pool)

        wf._run_parallel(
            entries,
            state,
            SimpleNamespace(),  # type: ignore[arg-type]
            _Progress(),
        )

        assert pool_runs == [
            ("overdue", "collateral", "untouched"),
            ("collateral", "untouched"),
        ]
        assert state.statuses == {
            "overdue": ExecutionStatus.TIMEOUT,
            "collateral": ExecutionStatus.SUCCESS,
            "untouched": ExecutionStatus.SUCCESS,
        }

    def test_broken_pool_run_diagnoses_only_casualties_then_resumes_pending(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        wf = RecordingWorkflow(logs_root=tmp_path / "logs", num_workers=2)
        entries = [
            _entry(tmp_path / f"{name}.nii.gz", tmp_path / "out", entry_id=name)
            for name in ("suspect-a", "suspect-b", "pending-a", "pending-b")
        ]
        state = ExecutionState.from_entries(entries)
        pool_runs: list[tuple[str, ...]] = []
        diagnosed: list[tuple[str, ...]] = []

        def _drive_pool(
            pending: deque[StagedEntry],
            *args: object,
            **kwargs: object,
        ) -> _PoolResult:
            pool_runs.append(tuple(entry.id for entry in pending))
            if len(pool_runs) == 1:
                casualties = (pending.popleft(), pending.popleft())
                return _PoolResult(
                    _PoolExit.BROKEN,
                    casualties=casualties,
                    broken_pool=BrokenProcessPool("pool died"),
                )
            while pending:
                state.update({pending.popleft().id: ExecutionStatus.SUCCESS})
            return _PoolResult(_PoolExit.COMPLETE, made_progress=True)

        def _diagnose(suspects: tuple[StagedEntry, ...], *args: object) -> None:
            diagnosed.append(tuple(entry.id for entry in suspects))
            state.update({entry.id: ExecutionStatus.SUCCESS for entry in suspects})

        monkeypatch.setattr(wf, "_drive_pool", _drive_pool)
        monkeypatch.setattr(wf, "_diagnose_entries", _diagnose)
        broken_logs: list[tuple[object, ...]] = []
        monkeypatch.setattr(
            wf,
            "_log_broken_pool",
            lambda *args, **kwargs: broken_logs.append((*args, kwargs)),
        )

        wf._run_parallel(
            entries,
            state,
            SimpleNamespace(),  # type: ignore[arg-type]
            _Progress(),
        )

        assert diagnosed == [("suspect-a", "suspect-b")]
        assert pool_runs == [
            ("suspect-a", "suspect-b", "pending-a", "pending-b"),
            ("pending-a", "pending-b"),
        ]
        assert set(state.statuses.values()) == {ExecutionStatus.SUCCESS}
        assert len(broken_logs) == 1


class TestIsolatedDiagnosis:
    def test_started_casualty_is_failure(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        wf = RecordingWorkflow(logs_root=tmp_path / "logs")
        entry = _entry(tmp_path / "a.nii.gz", tmp_path / "out", entry_id="entry")
        state = ExecutionState.from_entries([entry])
        broken = BrokenProcessPool("worker died")

        def _drive_pool(
            pending: deque[StagedEntry],
            *args: object,
            **kwargs: object,
        ) -> _PoolResult:
            assert kwargs == {"max_workers": 1, "max_tasks_per_child": 1}
            casualty = pending.popleft()
            return _PoolResult(
                _PoolExit.BROKEN,
                casualties=(casualty,),
                started_casualties=frozenset({casualty.id}),
                broken_pool=broken,
            )

        monkeypatch.setattr(wf, "_drive_pool", _drive_pool)
        progress = _Progress()

        wf._diagnose_entries(
            [entry],
            state,
            SimpleNamespace(),  # type: ignore[arg-type]
            progress,
        )

        assert state.get_status("entry") is ExecutionStatus.FAILURE
        assert progress.updates == [1]

    def test_timeout_in_diagnosis_uses_pool_driving_machinery_and_continues(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        wf = RecordingWorkflow(logs_root=tmp_path / "logs", timeout=1)
        timed_out = _entry(tmp_path / "slow.nii.gz", tmp_path / "out", entry_id="slow")
        success = _entry(tmp_path / "fast.nii.gz", tmp_path / "out", entry_id="fast")
        state = ExecutionState.from_entries([timed_out, success])
        calls: list[tuple[str, ...]] = []

        def _drive_pool(
            pending: deque[StagedEntry],
            *args: object,
            **kwargs: object,
        ) -> _PoolResult:
            assert kwargs == {"max_workers": 1, "max_tasks_per_child": 1}
            calls.append(tuple(entry.id for entry in pending))
            entry = pending.popleft()
            if entry.id == "slow":
                state.update({"slow": ExecutionStatus.TIMEOUT})
                return _PoolResult(_PoolExit.TIMEOUT, made_progress=True)
            state.update({"fast": ExecutionStatus.SUCCESS})
            return _PoolResult(_PoolExit.COMPLETE, made_progress=True)

        monkeypatch.setattr(wf, "_drive_pool", _drive_pool)

        wf._diagnose_entries(
            [timed_out, success],
            state,
            SimpleNamespace(),  # type: ignore[arg-type]
            _Progress(),
        )

        assert calls == [("slow", "fast"), ("fast",)]
        assert state.statuses == {
            "slow": ExecutionStatus.TIMEOUT,
            "fast": ExecutionStatus.SUCCESS,
        }

    def test_prestart_death_retries_same_entry_in_fresh_pool(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        wf = RecordingWorkflow(logs_root=tmp_path / "logs")
        entry = _entry(tmp_path / "a.nii.gz", tmp_path / "out", entry_id="entry")
        state = ExecutionState.from_entries([entry])
        attempts: list[str] = []

        def _drive_pool(
            pending: deque[StagedEntry],
            *args: object,
            **kwargs: object,
        ) -> _PoolResult:
            casualty = pending.popleft()
            attempts.append(casualty.id)
            if len(attempts) == 1:
                return _PoolResult(
                    _PoolExit.BROKEN,
                    casualties=(casualty,),
                    broken_pool=BrokenProcessPool("pre-start"),
                )
            state.update({casualty.id: ExecutionStatus.SUCCESS})
            return _PoolResult(_PoolExit.COMPLETE, made_progress=True)

        monkeypatch.setattr(wf, "_drive_pool", _drive_pool)
        broken_messages: list[str] = []

        def _record_broken(
            exc: BrokenProcessPool | None,
            *,
            message: str = "Worker process pool became unusable",
        ) -> None:
            broken_messages.append(message)

        monkeypatch.setattr(wf, "_log_broken_pool", _record_broken)

        wf._diagnose_entries(
            [entry],
            state,
            SimpleNamespace(),  # type: ignore[arg-type]
            _Progress(),
        )

        assert attempts == ["entry", "entry"]
        assert state.get_status("entry") is ExecutionStatus.SUCCESS
        assert len(broken_messages) == 1
        assert "failed before entry execution began" in broken_messages[0]
        assert "retrying in a fresh pool" in broken_messages[0]

    def test_repeated_prestart_death_is_systemic_and_does_not_blame_entry(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        wf = RecordingWorkflow(logs_root=tmp_path / "logs")
        entry = _entry(tmp_path / "a.nii.gz", tmp_path / "out", entry_id="entry")
        state = ExecutionState.from_entries([entry])
        attempts = 0

        def _drive_pool(
            pending: deque[StagedEntry],
            *args: object,
            **kwargs: object,
        ) -> _PoolResult:
            nonlocal attempts
            attempts += 1
            casualty = pending.popleft()
            return _PoolResult(
                _PoolExit.BROKEN,
                casualties=(casualty,),
                broken_pool=BrokenProcessPool("pre-start"),
            )

        monkeypatch.setattr(wf, "_drive_pool", _drive_pool)
        broken_messages: list[str] = []

        def _record_broken(
            exc: BrokenProcessPool | None,
            *,
            message: str = "Worker process pool became unusable",
        ) -> None:
            broken_messages.append(message)

        monkeypatch.setattr(wf, "_log_broken_pool", _record_broken)

        with pytest.raises(RuntimeError, match="failure appears systemic"):
            wf._diagnose_entries(
                [entry],
                state,
                SimpleNamespace(),  # type: ignore[arg-type]
                _Progress(),
            )

        assert attempts == 2
        assert state.get_status("entry") is ExecutionStatus.PENDING
        assert len(broken_messages) == 2
        assert "retrying in a fresh pool" in broken_messages[0]
        assert "treating failure as systemic" in broken_messages[1]


class TestPoolTermination:
    def test_uses_public_terminate_workers_when_available(self, tmp_path: Path) -> None:
        terminate_calls: list[None] = []
        shutdown_calls: list[tuple[bool, bool]] = []

        class _Process:
            def __init__(self) -> None:
                self.alive = True
                self.terminate_calls = 0
                self.kill_calls = 0
                self.join_calls = 0

            def terminate(self) -> None:
                self.terminate_calls += 1

            def kill(self) -> None:
                self.kill_calls += 1
                self.alive = False

            def join(self, timeout: float) -> None:
                self.join_calls += 1

            def is_alive(self) -> bool:
                return self.alive

        process = _Process()

        def _terminate_workers() -> None:
            terminate_calls.append(None)
            process.alive = False

        pool = SimpleNamespace(
            _processes={1: process},
            terminate_workers=_terminate_workers,
            shutdown=lambda *, wait, cancel_futures: shutdown_calls.append(
                (wait, cancel_futures)
            ),
        )

        RecordingWorkflow._terminate_pool(pool)  # type: ignore[arg-type]

        assert terminate_calls == [None]
        assert shutdown_calls == []
        assert process.terminate_calls == 0
        assert process.kill_calls == 0
        assert process.join_calls == 0

    def test_python_313_fallback_terminates_kills_and_reaps(
        self, tmp_path: Path
    ) -> None:
        class _Process:
            def __init__(self, *, survives_terminate: bool) -> None:
                self.alive = True
                self.survives_terminate = survives_terminate
                self.terminate_calls = 0
                self.kill_calls = 0
                self.join_timeouts: list[float] = []

            def is_alive(self) -> bool:
                return self.alive

            def terminate(self) -> None:
                self.terminate_calls += 1
                if not self.survives_terminate:
                    self.alive = False

            def kill(self) -> None:
                self.kill_calls += 1
                self.alive = False

            def join(self, timeout: float) -> None:
                self.join_timeouts.append(timeout)

        terminated = _Process(survives_terminate=False)
        survivor = _Process(survives_terminate=True)
        shutdown_calls: list[tuple[bool, bool]] = []
        pool = SimpleNamespace(
            _processes={1: terminated, 2: survivor},
            shutdown=lambda *, wait, cancel_futures: shutdown_calls.append(
                (wait, cancel_futures)
            ),
        )

        RecordingWorkflow._terminate_pool(pool)  # type: ignore[arg-type]

        assert shutdown_calls == [(False, True)]
        assert terminated.terminate_calls == survivor.terminate_calls == 1
        assert terminated.kill_calls == 0
        assert survivor.kill_calls == 1
        assert len(terminated.join_timeouts) == 1
        assert len(survivor.join_timeouts) == 2
        assert all(
            0 <= timeout <= 2.0
            for timeout in terminated.join_timeouts + survivor.join_timeouts
        )

    def test_raises_if_worker_survives_kill(self, tmp_path: Path) -> None:
        class _Process:
            def is_alive(self) -> bool:
                return True

            def terminate(self) -> None:
                pass

            def kill(self) -> None:
                pass

            def join(self, timeout: float) -> None:
                pass

        pool = SimpleNamespace(
            _processes={1: _Process()},
            shutdown=lambda *, wait, cancel_futures: None,
        )

        with pytest.raises(RuntimeError, match="Failed to terminate and reap 1"):
            RecordingWorkflow._terminate_pool(pool)  # type: ignore[arg-type]


class TestTimeoutExecution:
    def test_finite_timeout_uses_process_scheduler_with_one_worker(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        wf = RecordingWorkflow(logs_root=tmp_path / "logs", num_workers=1, timeout=1)
        serial_calls: list[tuple[object, ...]] = []
        parallel_calls: list[tuple[object, ...]] = []
        monkeypatch.setattr(wf, "_run_serial", lambda *args: serial_calls.append(args))
        monkeypatch.setattr(
            wf, "_run_parallel", lambda *args: parallel_calls.append(args)
        )
        monkeypatch.setattr(
            wf._logging_manager,
            "setup_parallel_logging",
            lambda: nullcontext(SimpleNamespace()),
        )

        wf.run_entries(
            [_entry(tmp_path / "a.nii.gz", tmp_path / "out", entry_id="entry")]
        )

        assert serial_calls == []
        assert len(parallel_calls) == 1

    def test_overdue_entry_is_terminated_and_the_run_continues(
        self, tmp_path: Path, logs_dir: Path
    ) -> None:
        fast = _touch(tmp_path / "fast.nii.gz")
        slow = _touch(tmp_path / "slow.nii.gz")
        out_dir = tmp_path / "out"
        wf = SleepingWorkflow(logs_root=logs_dir, num_workers=1, timeout=0.25)
        params = {
            "out_dir": str(out_dir),
            "slow_actives": (str(slow),),
            "sleep_seconds": 30.0,
        }

        started = time.monotonic()
        state = wf.run_entries(
            [
                StagedEntry(
                    active=slow.resolve(), id=str(slow.resolve()), params=params
                ),
                StagedEntry(
                    active=fast.resolve(), id=str(fast.resolve()), params=params
                ),
            ]
        )
        elapsed = time.monotonic() - started

        status = _status_log(logs_dir)
        assert _status_outcomes(status, str(fast.resolve())) == ["SUCCESS"]
        assert _status_outcomes(status, str(slow.resolve())) == ["TIMEOUT"]
        assert _recorded(out_dir) == [fast.resolve().as_posix()]
        assert state.get_status(str(slow.resolve())) is ExecutionStatus.TIMEOUT
        assert state.get_status(str(fast.resolve())) is ExecutionStatus.SUCCESS
        assert elapsed < 15.0

    @pytest.mark.parametrize("num_workers", [1, 2])
    def test_queue_wait_does_not_count_towards_the_timeout(
        self, tmp_path: Path, logs_dir: Path, num_workers: int
    ) -> None:
        """A fast entry must not TIMEOUT merely because it waited for a worker."""
        first = _touch(tmp_path / "first.nii.gz")
        second = _touch(tmp_path / "second.nii.gz")
        out_dir = tmp_path / "out"
        # !Why not shorter timeout/sleep times?!
        wf = SleepingWorkflow(logs_root=logs_dir, num_workers=num_workers, timeout=0.5)
        params = {
            "out_dir": str(out_dir),
            "slow_actives": (str(first),),
            "sleep_seconds": 1.5,
        }

        wf.run_entries(
            [
                StagedEntry(
                    active=first.resolve(), id=str(first.resolve()), params=params
                ),
                StagedEntry(
                    active=second.resolve(), id=str(second.resolve()), params=params
                ),
            ]
        )

        status = _status_log(logs_dir)
        assert _status_outcomes(status, str(first.resolve())) == ["TIMEOUT"]
        assert _status_outcomes(status, str(second.resolve())) == ["SUCCESS"]

    def test_processing_within_the_limit_is_a_success(
        self, tmp_path: Path, logs_dir: Path
    ) -> None:
        active = _touch(tmp_path / "entry.nii.gz")
        # !Why not shorter timeout/sleep times?!
        wf = SleepingWorkflow(logs_root=logs_dir, num_workers=1, timeout=0.5)
        params = {
            "out_dir": str(tmp_path / "out"),
            "slow_actives": (str(active),),
            "sleep_seconds": 0.2,
        }

        wf.run_entries(
            [
                StagedEntry(
                    active=active.resolve(), id=str(active.resolve()), params=params
                )
            ]
        )

        status = _status_log(logs_dir)
        assert _status_outcomes(status, str(active.resolve())) == ["SUCCESS"]
        assert "TIMEOUT" not in status

    def test_processing_past_the_limit_is_only_timeout(
        self, tmp_path: Path, logs_dir: Path
    ) -> None:
        active = _touch(tmp_path / "entry.nii.gz")
        out_dir = tmp_path / "out"
        wf = SleepingWorkflow(logs_root=logs_dir, num_workers=1, timeout=0.5)
        params = {
            "out_dir": str(out_dir),
            "slow_actives": (str(active),),
            "sleep_seconds": 0.8,
        }

        wf.run_entries(
            [
                StagedEntry(
                    active=active.resolve(), id=str(active.resolve()), params=params
                )
            ]
        )

        status = _status_log(logs_dir)
        assert _status_outcomes(status, str(active.resolve())) == ["TIMEOUT"]
        assert active.resolve().as_posix() not in _recorded(out_dir)

    def test_overdue_failure_is_terminated_before_it_can_fail(
        self,
        tmp_path: Path,
        logs_dir: Path,
    ) -> None:
        active = _touch(tmp_path / "entry.nii.gz")
        wf = RecordingWorkflow(logs_root=logs_dir, num_workers=1, timeout=0.5)
        wf.process_single = _sleep_then_fail  # type: ignore[method-assign]
        wf.run_entries([_entry(active, tmp_path / "out", sleep_seconds=0.8)])

        status = _status_log(logs_dir)
        assert _status_outcomes(status, str(active.resolve())) == ["TIMEOUT"]
        assert "late boom" not in status

    def test_timeout_none_does_not_apply_an_artificial_limit(
        self, tmp_path: Path, logs_dir: Path
    ) -> None:
        active = _touch(tmp_path / "entry.nii.gz")
        out_dir = tmp_path / "out"
        wf = SleepingWorkflow(logs_root=logs_dir, num_workers=2, timeout=None)
        # !Why not shorter sleep time?!
        params = {
            "out_dir": str(out_dir),
            "slow_actives": (str(active),),
            "sleep_seconds": 0.2,
        }

        wf.run_entries(
            [
                StagedEntry(
                    active=active.resolve(), id=str(active.resolve()), params=params
                )
            ]
        )

        assert _status_outcomes(_status_log(logs_dir), str(active.resolve())) == [
            "SUCCESS"
        ]
        assert _recorded(out_dir) == [active.resolve().as_posix()]


# !Move to dedicated test file!
class TestRunPlan:
    def test_executes_plan_entries(self, tmp_path: Path) -> None:
        out_dir = tmp_path / "out"
        wf = PlanningWorkflow(logs_root=tmp_path / "logs")
        wf.out_dir = str(out_dir)  # type: ignore[attr-defined]
        actives = [tmp_path / f"img-{index}.nii.gz" for index in range(2)]
        for active in actives:
            active.write_bytes(b"")

        wf.run_plan(wf.plan(actives))

        assert set(_recorded(out_dir)) == {a.resolve().as_posix() for a in actives}

    def test_run_plan_respects_start_end(self, tmp_path: Path) -> None:
        out_dir = tmp_path / "out"
        wf = PlanningWorkflow(logs_root=tmp_path / "logs")
        wf.out_dir = str(out_dir)  # type: ignore[attr-defined]
        actives = [tmp_path / f"img-{index}.nii.gz" for index in range(5)]
        for active in actives:
            active.write_bytes(b"")

        wf.run_plan(wf.plan(actives), start=1, end=4)

        assert _recorded(out_dir) == [a.resolve().as_posix() for a in actives[1:4]]

    def test_run_plan_negative_end(self, tmp_path: Path) -> None:
        out_dir = tmp_path / "out"
        wf = PlanningWorkflow(logs_root=tmp_path / "logs")
        wf.out_dir = str(out_dir)  # type: ignore[attr-defined]
        actives = [tmp_path / f"img-{index}.nii.gz" for index in range(3)]
        for active in actives:
            active.write_bytes(b"")

        wf.run_plan(wf.plan(actives), end=-1)

        assert _recorded(out_dir) == [a.resolve().as_posix() for a in actives[:2]]

    def test_rejects_non_run_plan(self, tmp_path: Path) -> None:
        wf = PlanningWorkflow(logs_root=tmp_path / "logs")
        with pytest.raises(TypeError, match="`plan` must be a RunPlan"):
            wf.run_plan([])  # type: ignore[arg-type]

    def test_empty_plan_is_a_noop(self, tmp_path: Path) -> None:
        wf = PlanningWorkflow(logs_root=tmp_path / "logs")
        wf.out_dir = str(tmp_path / "out")  # type: ignore[attr-defined]
        wf.run_plan(RunPlan(entries=()))
        assert _recorded(tmp_path / "out") == []

    def test_forwards_state_save_path_statuses_and_returns_used_state(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        wf = PlanningWorkflow(logs_root=tmp_path / "logs")
        wf.out_dir = str(tmp_path / "out")  # type: ignore[attr-defined]
        plan = wf.plan([tmp_path / f"img-{index}.nii.gz" for index in range(4)])
        state = ExecutionState.from_entries(plan.entries)
        state.update({"plan-1": ExecutionStatus.FAILURE})
        save_to = tmp_path / "state.duckdb"
        calls: list[tuple[tuple[str, ...], dict[str, object]]] = []

        def _run_entries(
            entries: tuple[StagedEntry, ...],
            **kwargs: object,
        ) -> ExecutionState:
            calls.append((tuple(entry.id for entry in entries), kwargs))
            return state

        monkeypatch.setattr(wf, "run_entries", _run_entries)

        returned = wf.run_plan(
            plan,
            start=-3,
            end=-1,
            execution_state=state,
            save_execution_state_to=save_to,
            run_statuses={ExecutionStatus.FAILURE},
        )

        assert returned is state
        assert calls == [
            (
                ("plan-1", "plan-2"),
                {
                    "execution_state": state,
                    "save_execution_state_to": save_to,
                    "run_statuses": {ExecutionStatus.FAILURE},
                },
            )
        ]

    def test_slicing_is_applied_once_and_full_plan_state_is_accepted(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        wf = PlanningWorkflow(logs_root=tmp_path / "logs")
        wf.out_dir = str(tmp_path / "out")  # type: ignore[attr-defined]
        plan = wf.plan([tmp_path / f"img-{index}.nii.gz" for index in range(5)])
        state = ExecutionState.from_entries(plan.entries)
        original_slice = RunPlan.slice
        slice_calls: list[tuple[int, int | None]] = []

        def _slice(
            instance: RunPlan, start: int = 0, end: int | None = None
        ) -> RunPlan:
            slice_calls.append((start, end))
            return original_slice(instance, start, end)

        monkeypatch.setattr(RunPlan, "slice", _slice)

        returned = wf.run_plan(
            plan,
            start=1,
            end=3,
            execution_state=state,
        )

        assert returned is state
        assert slice_calls == [(1, 3)]
        assert state.get_status("plan-0") is ExecutionStatus.PENDING
        assert state.get_status("plan-1") is ExecutionStatus.SUCCESS
        assert state.get_status("plan-2") is ExecutionStatus.SUCCESS
        assert state.get_status("plan-3") is ExecutionStatus.PENDING
        assert state.get_status("plan-4") is ExecutionStatus.PENDING


class TestLogging:
    def test_log_accepts_each_level(self, workflow: RecordingWorkflow) -> None:
        for level in ("debug", "info", "warning", "error", "critical"):
            workflow.log(f"message at {level}", level)  # type: ignore[arg-type]

    def test_logs_written_under_logs_root(self, tmp_path: Path) -> None:
        logs_root = tmp_path / "logs"
        wf = RecordingWorkflow(logs_root=logs_root)
        wf.log("hello")
        wf.status("a | SUCCESS")

        assert logs_root.exists()
        assert (logs_root / "main.log").exists()
        assert (logs_root / "status.log").exists()

        main_log = (logs_root / "main.log").read_text(encoding="utf-8")
        status_log = (logs_root / "status.log").read_text(encoding="utf-8")
        assert "hello" in main_log
        assert status_log.splitlines() == ["a | SUCCESS"]

    def test_status_detail_is_single_line_and_bounded(
        self, workflow: RecordingWorkflow
    ) -> None:
        compacted = workflow._compact("first\n\nsecond " + "x" * 2_000)

        assert "\n" not in compacted
        assert compacted.startswith("first second ")
        assert len(compacted) == 1_000
        assert compacted.endswith("…")

    def test_parallel_worker_logs_written_under_logs_root(self, tmp_path: Path) -> None:
        logs_root = tmp_path / "logs"
        out_dir = tmp_path / "out"
        wf = WorkerLoggingWorkflow(logs_root=logs_root, num_workers=2)
        active = _touch(tmp_path / "same.nii.gz")
        entries = [
            _entry(active, out_dir, entry_id="logical-entry"),
            _entry(active, out_dir, entry_id="logical-entry#2"),
        ]

        wf.run_entries(entries)

        assert (logs_root / "workers.log").exists()
        workers_log = (logs_root / "workers.log").read_text(encoding="utf-8")
        assert workers_log.count("from worker") == len(entries)
        for entry in entries:
            assert entry.id in workers_log
        status = _status_log(logs_root)
        for entry in entries:
            assert _status_outcomes(status, entry.id) == ["SUCCESS"]
