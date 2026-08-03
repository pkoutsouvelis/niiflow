"""Tests for :class:`ProcessingWorkflow` and :class:`PlannableWorkflow`.

These exercise the base classes through minimal dummy subclasses so the execution
engine, constructor validation, and plan/run contract are covered without any
staging, discovery, or pipeline machinery. ``DynamicPreprocessingWorkflow`` is
tested separately in ``test_preproc_workflows_dynamic.py``.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from niiflow.preproc.staging import StagedEntry, StagingErrorRecord
from niiflow.preproc.workflows import (
    PlannableWorkflow,
    ProcessingWorkflow,
    RunPlan,
)

# Module-level so the dummy workflows stay picklable for process pools; a
# closure or method would not be.
_CALLS_FILENAME = "calls.log"


def _record_call(entry: StagedEntry) -> None:
    out_dir = Path(entry.params["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / _CALLS_FILENAME).open("a", encoding="utf-8") as fh:
        fh.write(f"{entry.active.as_posix()}\n")


def _boom(entry: StagedEntry) -> None:
    raise RuntimeError(f"boom for {entry.active}")


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
                    active=Path(item).resolve(), params={"out_dir": self.out_dir}
                )
                for item in source
            )
        )


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")
    return path


def _entry(active: Path, out_dir: Path, *, errors: tuple = ()) -> StagedEntry:
    return StagedEntry(
        active=active.resolve(),
        params={"out_dir": str(out_dir)},
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
        with pytest.raises(ValueError, match="at least 1"):
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
        with pytest.raises(ValueError, match="must be positive"):
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


class TestTimeoutExecution:
    """A configured timeout is soft: overdue entries are reported, not cancelled."""

    def test_overdue_entry_is_reported_and_the_run_continues(
        self, tmp_path: Path, logs_dir: Path
    ) -> None:
        fast = _touch(tmp_path / "fast.nii.gz")
        slow = _touch(tmp_path / "slow.nii.gz")
        out_dir = tmp_path / "out"
        wf = SleepingWorkflow(logs_root=logs_dir, num_workers=1, timeout=0.5)
        params = {
            "out_dir": str(out_dir),
            "slow_actives": (str(slow),),
            "sleep_seconds": 2.0,
        }

        wf.run_entries(
            [
                StagedEntry(active=fast.resolve(), params=params),
                StagedEntry(active=slow.resolve(), params=params),
            ]
        )

        status = _status_log(logs_dir)
        assert f"{fast.resolve()} | SUCCESS" in status
        assert f"{slow.resolve()} | TIMEOUT" in status
        assert fast.resolve().as_posix() in _recorded(out_dir)

    @pytest.mark.parametrize("num_workers", [1, 2])
    def test_queue_wait_does_not_count_towards_the_timeout(
        self, tmp_path: Path, logs_dir: Path, num_workers: int
    ) -> None:
        """A fast entry must not TIMEOUT merely because it waited for a worker."""
        first = _touch(tmp_path / "first.nii.gz")
        second = _touch(tmp_path / "second.nii.gz")
        out_dir = tmp_path / "out"
        wf = SleepingWorkflow(logs_root=logs_dir, num_workers=num_workers, timeout=0.5)
        params = {
            "out_dir": str(out_dir),
            "slow_actives": (str(first),),
            "sleep_seconds": 1.5,
        }

        wf.run_entries(
            [
                StagedEntry(active=first.resolve(), params=params),
                StagedEntry(active=second.resolve(), params=params),
            ]
        )

        status = _status_log(logs_dir)
        assert f"{first.resolve()} | TIMEOUT" in status
        assert f"{second.resolve()} | SUCCESS" in status

    def test_processing_within_the_limit_is_a_success(
        self, tmp_path: Path, logs_dir: Path
    ) -> None:
        active = _touch(tmp_path / "entry.nii.gz")
        wf = SleepingWorkflow(logs_root=logs_dir, num_workers=1, timeout=0.5)
        params = {
            "out_dir": str(tmp_path / "out"),
            "slow_actives": (str(active),),
            "sleep_seconds": 0.2,
        }

        wf.run_entries([StagedEntry(active=active.resolve(), params=params)])

        status = _status_log(logs_dir)
        assert f"{active.resolve()} | SUCCESS" in status
        assert "TIMEOUT" not in status

    def test_processing_past_the_limit_is_a_timeout(
        self, tmp_path: Path, logs_dir: Path
    ) -> None:
        active = _touch(tmp_path / "entry.nii.gz")
        wf = SleepingWorkflow(logs_root=logs_dir, num_workers=1, timeout=0.5)
        params = {
            "out_dir": str(tmp_path / "out"),
            "slow_actives": (str(active),),
            "sleep_seconds": 0.8,
        }

        wf.run_entries([StagedEntry(active=active.resolve(), params=params)])

        status = _status_log(logs_dir)
        assert f"{active.resolve()} | TIMEOUT" in status
        assert "SUCCESS" not in status


class TestRunPlan:
    def test_executes_plan_entries(self, tmp_path: Path) -> None:
        out_dir = tmp_path / "out"
        wf = PlanningWorkflow(logs_root=tmp_path / "logs")
        wf.out_dir = str(out_dir)
        actives = [tmp_path / f"img-{index}.nii.gz" for index in range(2)]
        for active in actives:
            active.write_bytes(b"")

        wf.run_plan(wf.plan(actives))

        assert set(_recorded(out_dir)) == {a.resolve().as_posix() for a in actives}

    def test_rejects_non_run_plan(self, tmp_path: Path) -> None:
        wf = PlanningWorkflow(logs_root=tmp_path / "logs")
        with pytest.raises(TypeError, match="`plan` must be a RunPlan"):
            wf.run_plan([])  # type: ignore[arg-type]

    def test_empty_plan_is_a_noop(self, tmp_path: Path) -> None:
        wf = PlanningWorkflow(logs_root=tmp_path / "logs")
        wf.out_dir = str(tmp_path / "out")
        wf.run_plan(RunPlan(entries=()))
        assert _recorded(tmp_path / "out") == []


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
