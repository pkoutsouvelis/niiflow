"""Tests for :class:`~niiflow.preproc.workflows.plan.RunPlan` persistence."""

from __future__ import annotations

import json
import warnings
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import duckdb
import pytest

from niiflow.preproc.staging import StagedEntry, StagingErrorRecord
from niiflow.preproc.workflows import DynamicProcessingWorkflow, RunPlan


def _entry(
    active: Path,
    *,
    params: dict[str, Any] | None = None,
    errors: tuple[StagingErrorRecord, ...] = (),
    entry_id: str | None = None,
) -> StagedEntry:
    resolved = active.resolve()
    return StagedEntry(
        active=resolved,
        params=params or {"steps": []},
        id=entry_id if entry_id is not None else resolved.name,
        errors=errors,
    )


def _sample_plan(tmp_path: Path) -> RunPlan:
    return RunPlan(
        entries=(
            _entry(
                tmp_path / "sub-01_T1w.nii.gz",
                entry_id="primary-t1w",
                params={
                    "steps": [],
                    "output_path": str(tmp_path / "out" / "result.txt"),
                    "nested": {"count": 2},
                },
            ),
            _entry(
                tmp_path / "sub-02_T1w.nii.gz",
                entry_id="failed-t1w",
                errors=(
                    StagingErrorRecord(
                        active=tmp_path / "sub-02_T1w.nii.gz",
                        message="missing pointer target",
                        stage="FileStager",
                        entry_id="failed-t1w",
                        entry_index=1,
                        error_type="FileStagingError",
                    ),
                ),
            ),
        )
    )


def _json_plan_warning(suffix: str):
    if suffix == ".json":
        return pytest.warns(
            DeprecationWarning, match=r"deprecated and will be removed in v0\.5\.0"
        )
    return nullcontext()


def _persisted_plan_version(path: Path) -> int:
    if path.suffix == ".json":
        return int(json.loads(path.read_text(encoding="utf-8"))["plan_version"])

    conn = duckdb.connect(str(path), read_only=True)
    try:
        row = conn.execute(
            "SELECT value FROM meta WHERE key = 'plan_version'"
        ).fetchone()
    finally:
        conn.close()
    assert row is not None
    return int(row[0])


def _write_plan_fixture(
    path: Path,
    *,
    version: int | str | None,
    records: list[dict[str, Any]],
    include_ids: bool,
) -> None:
    if path.suffix == ".json":
        payload: dict[str, Any] = {"entries": records}
        if version is not None:
            payload["plan_version"] = version
        path.write_text(json.dumps(payload), encoding="utf-8")
        return

    conn = duckdb.connect(str(path))
    try:
        conn.execute("CREATE TABLE meta (key VARCHAR PRIMARY KEY, value VARCHAR)")
        if version is not None:
            conn.execute(
                "INSERT INTO meta VALUES ('plan_version', ?)",
                [str(version)],
            )
        if include_ids:
            conn.execute("""CREATE TABLE entries (
                    entry_index INTEGER PRIMARY KEY,
                    id VARCHAR,
                    active VARCHAR NOT NULL,
                    params VARCHAR NOT NULL,
                    errors VARCHAR NOT NULL
                )""")
            rows = [
                (
                    record["entry_index"],
                    record.get("id"),
                    record["active"],
                    json.dumps(record["params"]),
                    json.dumps(record["errors"]),
                )
                for record in records
            ]
            if rows:
                conn.executemany(
                    "INSERT INTO entries VALUES (?, ?, ?, ?, ?)",
                    rows,
                )
        else:
            conn.execute("""CREATE TABLE entries (
                    entry_index INTEGER PRIMARY KEY,
                    active VARCHAR NOT NULL,
                    params VARCHAR NOT NULL,
                    errors VARCHAR NOT NULL
                )""")
            rows = [
                (
                    record["entry_index"],
                    record["active"],
                    json.dumps(record["params"]),
                    json.dumps(record["errors"]),
                )
                for record in records
            ]
            if rows:
                conn.executemany(
                    "INSERT INTO entries VALUES (?, ?, ?, ?)",
                    rows,
                )
    finally:
        conn.close()


LegacyPlanFixture = tuple[Path, tuple[Path, ...], tuple[str, ...]]


@pytest.fixture(params=[".duckdb", ".json"], ids=["duckdb", "json"])
def v1_plan_fixture(
    tmp_path: Path,
    request: pytest.FixtureRequest,
) -> LegacyPlanFixture:
    duplicate = (tmp_path / "duplicate.nii.gz").resolve()
    suffix_collision = Path(f"{duplicate}#2")
    actives = (duplicate, duplicate, suffix_collision, duplicate)
    records = [
        {
            "entry_index": index,
            "active": str(active),
            "params": {"steps": [], "ordinal": index},
            "errors": [],
        }
        for index, active in enumerate(actives)
    ]
    path = tmp_path / f"v1-plan{request.param}"
    _write_plan_fixture(
        path,
        version=1,
        records=records,
        include_ids=False,
    )
    expected_ids = (
        str(duplicate),
        f"{duplicate}#3",
        str(suffix_collision),
        f"{duplicate}#4",
    )
    return path, actives, expected_ids


def _load_v1_with_warnings(
    path: Path,
    *,
    start: int = 0,
    end: int | None = None,
) -> RunPlan:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        loaded = RunPlan.load(path, start=start, end=end)

    assert any(
        warning.category is FutureWarning
        and "version 1 is deprecated" in str(warning.message)
        for warning in caught
    )
    if path.suffix == ".json":
        assert any(
            warning.category is DeprecationWarning
            and "deprecated and will be removed in v0.5.0" in str(warning.message)
            for warning in caught
        )
    return loaded


@pytest.mark.parametrize("suffix", [".duckdb", ".json"])
class TestRunPlanRoundTrip:
    def test_save_load_preserves_entries(self, tmp_path: Path, suffix: str) -> None:
        plan = _sample_plan(tmp_path)
        plan_path = tmp_path / f"job{suffix}"

        with _json_plan_warning(suffix):
            plan.save(plan_path)
            assert _persisted_plan_version(plan_path) == 2
            loaded = RunPlan.load(plan_path)

        assert len(loaded.entries) == len(plan.entries)
        for original, restored in zip(plan.entries, loaded.entries, strict=True):
            assert restored.id == original.id
            assert restored.active == original.active.resolve()
            assert restored.params == original.params
            assert len(restored.errors) == len(original.errors)
            for orig_err, rest_err in zip(
                original.errors, restored.errors, strict=True
            ):
                assert rest_err.active == orig_err.active.resolve()
                assert rest_err.message == orig_err.message
                assert rest_err.stage == orig_err.stage
                assert rest_err.entry_id == orig_err.entry_id
                assert rest_err.entry_index == orig_err.entry_index
                assert rest_err.error_type == orig_err.error_type

    def test_empty_plan_round_trip(self, tmp_path: Path, suffix: str) -> None:
        plan = RunPlan(entries=())
        plan_path = tmp_path / f"empty{suffix}"

        with _json_plan_warning(suffix):
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

    @pytest.mark.parametrize("suffix", [".duckdb", ".json"])
    @pytest.mark.parametrize(
        ("version", "match"),
        [
            pytest.param(None, "missing `plan_version`", id="missing"),
            pytest.param("unknown", "Invalid `plan_version`", id="unknown"),
            pytest.param(3, "Unsupported plan version", id="future"),
        ],
    )
    def test_rejects_invalid_plan_versions(
        self,
        tmp_path: Path,
        suffix: str,
        version: int | str | None,
        match: str,
    ) -> None:
        path = tmp_path / f"bad-version{suffix}"
        _write_plan_fixture(
            path,
            version=version,
            records=[],
            include_ids=True,
        )

        with _json_plan_warning(suffix):
            with pytest.raises(ValueError, match=match):
                RunPlan.load(path)

    @pytest.mark.parametrize("suffix", [".duckdb", ".json"])
    @pytest.mark.parametrize(
        ("ids", "match"),
        [
            pytest.param([None], "non-empty string", id="missing"),
            pytest.param([""], "non-empty string", id="empty"),
            pytest.param(
                ["duplicate", "duplicate"], "unique|duplicate", id="duplicate"
            ),
        ],
    )
    def test_v2_rejects_invalid_entry_ids(
        self,
        tmp_path: Path,
        suffix: str,
        ids: list[str | None],
        match: str,
    ) -> None:
        records = [
            {
                "entry_index": index,
                "active": str((tmp_path / f"entry-{index}.nii.gz").resolve()),
                "params": {"steps": [], "ordinal": index},
                "errors": [],
                **({} if entry_id is None else {"id": entry_id}),
            }
            for index, entry_id in enumerate(ids)
        ]
        path = tmp_path / f"bad-ids{suffix}"
        _write_plan_fixture(
            path,
            version=2,
            records=records,
            include_ids=True,
        )

        with _json_plan_warning(suffix):
            with pytest.raises(ValueError, match=match):
                RunPlan.load(path)

    def test_json_persistence_is_deprecated(self, tmp_path: Path) -> None:
        plan = RunPlan(entries=())
        path = tmp_path / "legacy.json"
        with pytest.warns(DeprecationWarning, match=r"`_save_json` is deprecated"):
            plan.save(path)
        with pytest.warns(DeprecationWarning, match=r"Use `\.duckdb` instead"):
            loaded = RunPlan.load(path)
        assert loaded.entries == ()


class TestRunPlanV1Compatibility:
    def test_full_load_synthesizes_deterministic_unique_ids(
        self,
        v1_plan_fixture: LegacyPlanFixture,
    ) -> None:
        path, actives, expected_ids = v1_plan_fixture

        loaded = _load_v1_with_warnings(path)

        assert tuple(entry.active for entry in loaded.entries) == actives
        assert tuple(entry.id for entry in loaded.entries) == expected_ids
        assert tuple(entry.params["ordinal"] for entry in loaded.entries) == (
            0,
            1,
            2,
            3,
        )

    def test_ranged_load_matches_ids_from_full_plan(
        self,
        v1_plan_fixture: LegacyPlanFixture,
    ) -> None:
        path, actives, expected_ids = v1_plan_fixture

        loaded = _load_v1_with_warnings(path, start=1, end=3)

        assert tuple(entry.active for entry in loaded.entries) == actives[1:3]
        assert tuple(entry.id for entry in loaded.entries) == expected_ids[1:3]
        assert tuple(entry.params["ordinal"] for entry in loaded.entries) == (1, 2)

    def test_range_reconciles_full_plan_before_slice_on_suffix_collision(
        self,
        v1_plan_fixture: LegacyPlanFixture,
    ) -> None:
        path, _, expected_ids = v1_plan_fixture

        loaded = _load_v1_with_warnings(path, start=1, end=2)

        assert [entry.id for entry in loaded.entries] == [expected_ids[1]]
        assert loaded.entries[0].id.endswith("#3")


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
        with _json_plan_warning(suffix):
            plan.save(plan_path)
            loaded = RunPlan.load(plan_path, start=1, end=4)
        assert [entry.active.name for entry in loaded.entries] == [
            entry.active.name for entry in plan.slice(start=1, end=4).entries
        ]

    def test_load_range_negative_bounds(self, tmp_path: Path, suffix: str) -> None:
        plan = _indexed_plan(tmp_path)
        plan_path = tmp_path / f"job{suffix}"
        with _json_plan_warning(suffix):
            plan.save(plan_path)
            loaded = RunPlan.load(plan_path, start=-2, end=-1)
        assert [entry.active.name for entry in loaded.entries] == ["img-03.nii.gz"]

    def test_load_empty_range(self, tmp_path: Path, suffix: str) -> None:
        plan = _indexed_plan(tmp_path)
        plan_path = tmp_path / f"job{suffix}"
        with _json_plan_warning(suffix):
            plan.save(plan_path)
            loaded = RunPlan.load(plan_path, start=2, end=2)
        assert loaded.entries == ()

    def test_load_rejects_out_of_range(self, tmp_path: Path, suffix: str) -> None:
        plan = _indexed_plan(tmp_path)
        plan_path = tmp_path / f"job{suffix}"
        with _json_plan_warning(suffix):
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
