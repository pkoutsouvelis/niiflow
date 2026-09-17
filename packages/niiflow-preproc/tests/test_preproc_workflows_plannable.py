"""Tests for :class:`~niiflow.preproc.workflows.PlannableWorkflow`."""

from __future__ import annotations

from pathlib import Path

import pytest

from niiflow.preproc.staging import StagedEntry
from niiflow.preproc.workflows import PlannableWorkflow, RunPlan
from niiflow.preproc.workflows.execution_state import ExecutionState, ExecutionStatus

_CALLS_FILENAME = "calls.log"


def _record_call(entry: StagedEntry) -> None:
    out_dir = Path(entry.params["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / _CALLS_FILENAME).open("a", encoding="utf-8") as fh:
        fh.write(f"{entry.active.as_posix()}\n")


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


def _recorded(out_dir: Path) -> list[str]:
    log = out_dir / _CALLS_FILENAME
    if not log.exists():
        return []
    return log.read_text(encoding="utf-8").split()


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

        assert returned is not state
        assert slice_calls == [(1, 3)]
        assert returned.get_status("plan-0") is ExecutionStatus.PENDING
        assert returned.get_status("plan-1") is ExecutionStatus.SUCCESS
        assert returned.get_status("plan-2") is ExecutionStatus.SUCCESS
        assert returned.get_status("plan-3") is ExecutionStatus.PENDING
        assert returned.get_status("plan-4") is ExecutionStatus.PENDING
        assert set(state.statuses.values()) == {ExecutionStatus.PENDING}
