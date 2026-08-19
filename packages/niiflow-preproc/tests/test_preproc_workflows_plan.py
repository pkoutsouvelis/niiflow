"""Tests for :class:`~niiflow.preproc.workflows.plan.RunPlan` persistence."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from niiflow.preproc.staging import StagedEntry, StagingErrorRecord
from niiflow.preproc.workflows import DynamicProcessingWorkflow, RunPlan


def _entry(
    active: Path,
    *,
    params: dict[str, Any] | None = None,
    errors: tuple[StagingErrorRecord, ...] = (),
) -> StagedEntry:
    return StagedEntry(
        active=active.resolve(),
        params=params or {"steps": []},
        errors=errors,
    )


def _sample_plan(tmp_path: Path) -> RunPlan:
    return RunPlan(
        entries=(
            _entry(
                tmp_path / "sub-01_T1w.nii.gz",
                params={
                    "steps": [],
                    "output_path": str(tmp_path / "out" / "result.txt"),
                    "nested": {"count": 2},
                },
            ),
            _entry(
                tmp_path / "sub-02_T1w.nii.gz",
                errors=(
                    StagingErrorRecord(
                        active=tmp_path / "sub-02_T1w.nii.gz",
                        message="missing pointer target",
                        stage="FileStager",
                        entry_index=1,
                        error_type="FileStagingError",
                    ),
                ),
            ),
        )
    )


@pytest.mark.parametrize("suffix", [".duckdb", ".json"])
class TestRunPlanRoundTrip:
    def test_save_load_preserves_entries(self, tmp_path: Path, suffix: str) -> None:
        plan = _sample_plan(tmp_path)
        plan_path = tmp_path / f"job{suffix}"

        plan.save(plan_path)
        loaded = RunPlan.load(plan_path)

        assert len(loaded.entries) == len(plan.entries)
        for original, restored in zip(plan.entries, loaded.entries, strict=True):
            assert restored.active == original.active.resolve()
            assert restored.params == original.params
            assert len(restored.errors) == len(original.errors)
            for orig_err, rest_err in zip(
                original.errors, restored.errors, strict=True
            ):
                assert rest_err.active == orig_err.active.resolve()
                assert rest_err.message == orig_err.message
                assert rest_err.stage == orig_err.stage
                assert rest_err.entry_index == orig_err.entry_index
                assert rest_err.error_type == orig_err.error_type

    def test_empty_plan_round_trip(self, tmp_path: Path, suffix: str) -> None:
        plan = RunPlan(entries=())
        plan_path = tmp_path / f"empty{suffix}"

        plan.save(plan_path)
        loaded = RunPlan.load(plan_path)

        assert loaded.entries == ()


class TestRunPlanView:
    def test_view_summarizes_entries(self, tmp_path: Path) -> None:
        plan = _sample_plan(tmp_path)
        text = plan.view()

        assert "RunPlan: 2 entries (1 runnable, 1 with staging error)" in text
        assert "Entry 0" in text
        assert str((tmp_path / "sub-01_T1w.nii.gz").resolve()) in text
        assert '"output_path"' in text
        assert "Entry 1" in text
        assert "[FileStager] missing pointer target" in text

    def test_view_empty_plan(self) -> None:
        assert RunPlan(entries=()).view() == "RunPlan: 0 entries"


class TestRunPlanValidation:
    def test_load_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            RunPlan.load(tmp_path / "missing.duckdb")

    def test_unsupported_extension_on_save(self, tmp_path: Path) -> None:
        plan = RunPlan(entries=())
        with pytest.raises(ValueError, match="Unsupported plan file extension"):
            plan.save(tmp_path / "plan.yaml")

    def test_unsupported_extension_on_load(self, tmp_path: Path) -> None:
        path = tmp_path / "plan.yaml"
        path.write_text("{}", encoding="utf-8")
        with pytest.raises(ValueError, match="Unsupported plan file extension"):
            RunPlan.load(path)

    def test_json_rejects_unsupported_version(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.json"
        path.write_text(
            '{"plan_version": 999, "entries": []}',
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="Unsupported plan version"):
            RunPlan.load(path)


class TestRunPlanScale:
    def test_duckdb_round_trip_many_entries(self, tmp_path: Path) -> None:
        entries = tuple(
            _entry(tmp_path / f"img-{index:04d}.nii.gz") for index in range(1000)
        )
        plan = RunPlan(entries=entries)
        plan_path = tmp_path / "large.duckdb"

        plan.save(plan_path)
        loaded = RunPlan.load(plan_path)

        assert len(loaded.entries) == 1000
        assert loaded.entries[0].active.name == "img-0000.nii.gz"
        assert loaded.entries[-1].active.name == "img-0999.nii.gz"


def _indexed_plan(tmp_path: Path, n: int = 5) -> RunPlan:
    return RunPlan(
        entries=tuple(
            _entry(tmp_path / f"img-{index:02d}.nii.gz") for index in range(n)
        )
    )


class TestRunPlanSlice:
    def test_selects_inclusive_exclusive_window(self, tmp_path: Path) -> None:
        plan = _indexed_plan(tmp_path)
        selected = plan.slice(start=1, end=4)

        assert [entry.active.name for entry in selected.entries] == [
            "img-01.nii.gz",
            "img-02.nii.gz",
            "img-03.nii.gz",
        ]

    def test_supports_negative_bounds(self, tmp_path: Path) -> None:
        plan = _indexed_plan(tmp_path)
        assert [e.active.name for e in plan.slice(start=-2).entries] == [
            "img-03.nii.gz",
            "img-04.nii.gz",
        ]
        assert [e.active.name for e in plan.slice(end=-1).entries] == [
            "img-00.nii.gz",
            "img-01.nii.gz",
            "img-02.nii.gz",
            "img-03.nii.gz",
        ]

    def test_full_range_returns_same_instance(self, tmp_path: Path) -> None:
        plan = _indexed_plan(tmp_path)
        assert plan.slice() is plan

    def test_includes_staging_failed_entries(self, tmp_path: Path) -> None:
        plan = _sample_plan(tmp_path)
        assert len(plan.slice(start=0, end=2).entries) == 2
        assert plan.slice(start=1, end=2).entries[0].errors

    def test_rejects_out_of_range(self, tmp_path: Path) -> None:
        plan = _indexed_plan(tmp_path)
        with pytest.raises(ValueError, match="`start`"):
            plan.slice(start=6)
        with pytest.raises(ValueError, match="`end`"):
            plan.slice(end=6)
        with pytest.raises(ValueError, match="must not precede"):
            plan.slice(start=3, end=1)

    def test_rejects_non_int_bounds(self, tmp_path: Path) -> None:
        plan = _indexed_plan(tmp_path)
        with pytest.raises(TypeError, match="`start`"):
            plan.slice(start=True)  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="`end`"):
            plan.slice(end=1.5)  # type: ignore[arg-type]

    def test_getitem_aliases_slice(self, tmp_path: Path) -> None:
        plan = _indexed_plan(tmp_path)
        assert [e.active.name for e in plan[1:4].entries] == [
            e.active.name for e in plan.slice(1, 4).entries
        ]
        assert [e.active.name for e in plan[-2:].entries] == [
            e.active.name for e in plan.slice(start=-2).entries
        ]
        assert [e.active.name for e in plan[:].entries] == [
            e.active.name for e in plan.entries
        ]
        assert plan[:] is plan

    def test_getitem_single_index_returns_one_entry_plan(self, tmp_path: Path) -> None:
        plan = _indexed_plan(tmp_path)
        assert [e.active.name for e in plan[2].entries] == ["img-02.nii.gz"]
        assert [e.active.name for e in plan[-1].entries] == ["img-04.nii.gz"]

    def test_getitem_rejects_step(self, tmp_path: Path) -> None:
        plan = _indexed_plan(tmp_path)
        with pytest.raises(ValueError, match="step"):
            _ = plan[0:4:2]


@pytest.mark.parametrize("suffix", [".duckdb", ".json"])
class TestRunPlanLoadRange:
    def test_load_range_matches_slice(self, tmp_path: Path, suffix: str) -> None:
        plan = _indexed_plan(tmp_path)
        plan_path = tmp_path / f"job{suffix}"
        plan.save(plan_path)

        loaded = RunPlan.load(plan_path, start=1, end=4)
        assert [entry.active.name for entry in loaded.entries] == [
            entry.active.name for entry in plan.slice(start=1, end=4).entries
        ]

    def test_load_range_negative_bounds(self, tmp_path: Path, suffix: str) -> None:
        plan = _indexed_plan(tmp_path)
        plan_path = tmp_path / f"job{suffix}"
        plan.save(plan_path)

        loaded = RunPlan.load(plan_path, start=-2, end=-1)
        assert [entry.active.name for entry in loaded.entries] == ["img-03.nii.gz"]

    def test_load_empty_range(self, tmp_path: Path, suffix: str) -> None:
        plan = _indexed_plan(tmp_path)
        plan_path = tmp_path / f"job{suffix}"
        plan.save(plan_path)

        loaded = RunPlan.load(plan_path, start=2, end=2)
        assert loaded.entries == ()

    def test_load_rejects_out_of_range(self, tmp_path: Path, suffix: str) -> None:
        plan = _indexed_plan(tmp_path)
        plan_path = tmp_path / f"job{suffix}"
        plan.save(plan_path)

        with pytest.raises(ValueError, match="`start`"):
            RunPlan.load(plan_path, start=10)


class TestRunPlanWorkflowIntegration:
    def test_plan_save_load_run(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        active = tmp_path / "input.nii.gz"
        active.write_bytes(b"nii")
        output = tmp_path / "out" / "result.txt"
        wf = DynamicProcessingWorkflow(
            staging_params={
                "stager_name": "FileStager",
                "params": {"pointers": {"output_path": "output"}},
            },
            pipeline_params={
                "steps": [],
                "output_path": str(output),
            },
            num_workers=1,
        )
        called: list[Path] = []

        def _record(entry: StagedEntry) -> None:
            called.append(entry.active)

        monkeypatch.setattr(wf, "process_single", _record)

        plan = wf.plan(active)
        plan_path = tmp_path / "job.duckdb"
        plan.save(plan_path)

        restored = RunPlan.load(plan_path)
        wf.run_plan(restored)

        assert called == [active.resolve()]
        assert restored.entries[0].params["output_path"] == str(output.resolve())
