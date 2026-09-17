"""Tests for core staging abstractions.

These exercise :class:`Stager`, :class:`StagedEntry`, :class:`StagingContext`,
and :func:`make_entries` through minimal dummy stagers so batch behaviour and
entry construction are covered without FileStager or dynamic-reference machinery.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from niiflow.preproc.staging import (
    StagedEntry,
    StagingErrorRecord,
    make_entries,
)
from niiflow.preproc.staging.stager import (
    StagingContext,
    Stager,
    _reconcile_entry_ids,
)


def _touch(path: Path, content: str = "nii") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path.resolve()


@pytest.fixture
def bids_tree(tmp_path: Path) -> dict[str, Path]:
    """Minimal BIDS-like tree with an active file."""
    root = tmp_path / "dataset"
    active = _touch(
        root / "sub-01" / "ses-pre" / "func" / "sub-01_ses-pre_task-rest_bold.nii.gz"
    )
    return {"root": root.resolve(), "active": active}


class RecordingStager(Stager):
    """Records which actives were staged; optionally fails on a marker param."""

    def __init__(self, *, allow_failed_entries: bool = False) -> None:
        self.allow_failed_entries = allow_failed_entries
        self.seen: list[Path] = []

    def stage_single(self, entry: StagedEntry) -> StagedEntry:
        if entry.errors:
            return entry
        if entry.params.get("boom"):
            raise RuntimeError(f"boom for {entry.active}")
        self.seen.append(entry.active)
        return StagedEntry(
            active=entry.active,
            id=entry.id,
            params={**entry.params, "staged": True},
            errors=entry.errors,
        )


class TestStagingContext:
    def test_holds_active_path(self, tmp_path: Path) -> None:
        active = _touch(tmp_path / "a.nii.gz")
        ctx = StagingContext(active=active)
        assert ctx.active == active


class TestMakeEntries:
    def test_shared_params_dict_is_copied_for_each_active_file(
        self, bids_tree: dict[str, Path]
    ) -> None:
        other = _touch(bids_tree["root"] / "sub-02" / "func" / "run2.nii.gz")
        shared = {"mask": "to-be-resolved", "nested": {"steps": []}}
        entries = make_entries([bids_tree["active"], other], shared)

        assert len(entries) == 2
        assert entries[0].active == bids_tree["active"]
        assert entries[1].active == other
        assert entries[0].params == shared
        assert entries[1].params == shared
        assert entries[0].params is not entries[1].params
        assert entries[0].params is not shared

        entries[0].params["mask"] = "mutated"
        entries[0].params["nested"]["steps"].append("first")
        assert entries[1].params["mask"] == "to-be-resolved"
        assert entries[1].params["nested"]["steps"] == []
        assert shared["nested"]["steps"] == []

    def test_per_entry_params_must_align_with_active_files(
        self, bids_tree: dict[str, Path]
    ) -> None:
        other = _touch(bids_tree["root"] / "sub-02" / "func" / "run2.nii.gz")
        params = [{"input": None, "nested": []}, {"input": str(other), "nested": []}]
        entries = make_entries(
            [bids_tree["active"], other],
            params,
        )

        assert entries[0].params["input"] is None
        assert entries[1].params["input"] == str(other)
        entries[0].params["nested"].append("mutated")
        assert entries[1].params["nested"] == []
        assert params[0]["nested"] == []

    def test_duplicate_active_ids_are_deterministic(
        self, bids_tree: dict[str, Path]
    ) -> None:
        active = bids_tree["active"]
        entries = make_entries([active, active, active], {})

        assert [entry.active for entry in entries] == [active, active, active]
        assert [entry.id for entry in entries] == [
            str(active),
            f"{active}#2",
            f"{active}#3",
        ]

    def test_duplicate_ids_are_stable_in_active_order(self, tmp_path: Path) -> None:
        first = _touch(tmp_path / "first.nii.gz")
        second = _touch(tmp_path / "second.nii.gz")

        entries = make_entries([first, second, first, first, second], {})

        assert [entry.active for entry in entries] == [
            first,
            second,
            first,
            first,
            second,
        ]
        assert [entry.id for entry in entries] == [
            str(first),
            str(second),
            f"{first}#2",
            f"{first}#3",
            f"{second}#2",
        ]

    def test_reconciliation_skips_reserved_generated_id(self, tmp_path: Path) -> None:
        active = (tmp_path / "active.nii.gz").resolve()
        entries = [
            StagedEntry(active=active, id="duplicate", params={}),
            StagedEntry(active=active, id="duplicate", params={}),
            StagedEntry(active=active, id=f"{active}#2", params={}),
        ]

        reconciled = _reconcile_entry_ids(entries)

        assert [entry.id for entry in reconciled] == [
            "duplicate",
            f"{active}#3",
            f"{active}#2",
        ]

    def test_reconciliation_leaves_unique_ids_unchanged(self, tmp_path: Path) -> None:
        entries = [
            StagedEntry(
                active=(tmp_path / f"{entry_id}.nii.gz").resolve(),
                id=entry_id,
                params={},
            )
            for entry_id in ("first", "second", "third")
        ]

        reconciled = _reconcile_entry_ids(entries)

        assert [entry.id for entry in reconciled] == ["first", "second", "third"]
        assert all(actual is original for actual, original in zip(reconciled, entries))

    def test_rejects_empty_active_files(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            make_entries([], {"input": None})

    def test_allows_missing_active_file(self, tmp_path: Path) -> None:
        missing = tmp_path / "missing.nii.gz"
        entries = make_entries([missing], {"label": "x"})
        assert len(entries) == 1
        assert entries[0].active == missing.resolve()
        assert entries[0].id == str(missing.resolve())
        assert entries[0].params == {"label": "x"}

    def test_allows_directory_active_path(self, tmp_path: Path) -> None:
        directory = tmp_path / "folder"
        directory.mkdir()
        entry = make_entries([directory], {"input": None})[0]

        assert entry.active == directory.resolve()
        assert entry.id == str(directory.resolve())

    def test_resolve_actives_controls_symlink_resolution(self, tmp_path: Path) -> None:
        target = _touch(tmp_path / "target.nii.gz")
        symlink = tmp_path / "alias.nii.gz"
        symlink.symlink_to(target)

        resolved = make_entries([symlink], {}, resolve_actives=True)[0]
        unresolved = make_entries([symlink], {}, resolve_actives=False)[0]

        assert resolved.active == target
        assert resolved.id == str(target)
        assert unresolved.active == symlink.absolute()
        assert unresolved.id == str(symlink.absolute())

    def test_does_not_call_exists_or_is_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        active = tmp_path / "unchecked.nii.gz"

        def _unexpected_check(path: Path) -> bool:
            raise AssertionError(f"Unexpected filesystem check for {path}")

        monkeypatch.setattr(Path, "exists", _unexpected_check)
        monkeypatch.setattr(Path, "is_file", _unexpected_check)

        entry = make_entries([active], {})[0]
        assert entry.active == active.resolve()

    def test_rejects_mismatched_per_entry_param_lengths(
        self, bids_tree: dict[str, Path]
    ) -> None:
        with pytest.raises(ValueError, match="same length"):
            make_entries([bids_tree["active"]], [{}, {}])

    def test_rejects_non_dict_per_entry_params(
        self, bids_tree: dict[str, Path]
    ) -> None:
        with pytest.raises(TypeError, match="dictionary"):
            make_entries([bids_tree["active"]], ["not-a-dict"])  # type: ignore[arg-type]


class TestStagerBatch:
    def test_stage_transforms_each_entry(self, bids_tree: dict[str, Path]) -> None:
        other = _touch(bids_tree["root"] / "sub-02" / "func" / "run2.nii.gz")
        entries = make_entries([bids_tree["active"], other], {"label": "x"})
        stager = RecordingStager()
        staged = stager.stage(entries)

        assert [e.active for e in staged] == [bids_tree["active"], other]
        assert all(e.params["staged"] is True for e in staged)
        assert stager.seen == [bids_tree["active"], other]

    def test_entries_with_existing_errors_are_skipped(
        self, bids_tree: dict[str, Path]
    ) -> None:
        prior = StagingErrorRecord(
            active=bids_tree["active"],
            message="previous failure",
        )
        entry = StagedEntry(
            active=bids_tree["active"],
            id=str(bids_tree["active"]),
            params={"label": "x"},
            errors=(prior,),
        )
        stager = RecordingStager()
        staged = stager.stage([entry])[0]

        assert staged.errors == (prior,)
        assert staged.params == entry.params
        assert stager.seen == []

    def test_allow_failed_entries_records_error_and_continues(
        self, bids_tree: dict[str, Path]
    ) -> None:
        bad = _touch(bids_tree["root"] / "sub-02" / "func" / "bad.nii.gz")
        good = _touch(bids_tree["root"] / "sub-03" / "func" / "good.nii.gz")
        entries = make_entries(
            [bad, good],
            [{"boom": True}, {"label": "ok"}],
        )
        stager = RecordingStager(allow_failed_entries=True)
        staged = stager.stage(entries)

        assert len(staged) == 2
        assert staged[0].errors
        assert staged[0].errors[0].error_type == "RuntimeError"
        assert staged[0].errors[0].entry_id == entries[0].id
        assert staged[0].errors[0].entry_index == 0
        assert staged[0].errors[0].stage == "RecordingStager"
        assert staged[1].params["staged"] is True
        assert not staged[1].errors

    def test_first_failure_aborts_by_default(self, bids_tree: dict[str, Path]) -> None:
        bad = _touch(bids_tree["root"] / "sub-02" / "func" / "bad.nii.gz")
        good = _touch(bids_tree["root"] / "sub-03" / "func" / "good.nii.gz")
        entries = make_entries(
            [bad, good],
            [{"boom": True}, {"label": "ok"}],
        )
        stager = RecordingStager()
        with pytest.raises(RuntimeError, match="boom"):
            stager.stage(entries)

    def test_rejects_non_sequence_entries(self, bids_tree: dict[str, Path]) -> None:
        entry = make_entries([bids_tree["active"]], {"label": "x"})[0]
        stager = RecordingStager()
        with pytest.raises(TypeError, match="must be a sequence of `StagedEntry`"):
            stager.stage(entry)  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="must be a sequence of `StagedEntry`"):
            stager.stage(None)  # type: ignore[arg-type]

    def test_rejects_non_staged_entry_before_stage_single(
        self, bids_tree: dict[str, Path]
    ) -> None:
        stager = RecordingStager(allow_failed_entries=True)
        with pytest.raises(TypeError, match="must contain only `StagedEntry`"):
            stager.stage([bids_tree["active"]])  # type: ignore[list-item]
        assert stager.seen == []

    def test_rejects_non_staged_entry_return_from_stage_single(
        self, bids_tree: dict[str, Path]
    ) -> None:
        class BrokenStager(Stager):
            allow_failed_entries = True

            def stage_single(self, entry: StagedEntry) -> StagedEntry:
                return entry.params  # type: ignore[return-value]

        entry = make_entries([bids_tree["active"]], {"label": "x"})[0]
        with pytest.raises(TypeError, match="must return a `StagedEntry`"):
            BrokenStager().stage([entry])

    def test_rejects_invalid_num_workers(self, bids_tree: dict[str, Path]) -> None:
        entry = make_entries([bids_tree["active"]], {"label": "x"})[0]
        stager = RecordingStager()
        with pytest.raises(ValueError, match="integer >= 1"):
            stager.stage([entry], num_workers=0)
        with pytest.raises(ValueError, match="integer >= 1"):
            stager.stage([entry], num_workers=True)  # type: ignore[arg-type]

    def test_progress_bar_tracks_entries(
        self, bids_tree: dict[str, Path], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        other = _touch(bids_tree["root"] / "sub-02" / "func" / "run2.nii.gz")
        entries = make_entries([bids_tree["active"], other], {"label": "x"})
        inits: list[dict[str, object]] = []
        updates = {"n": 0}

        class _FakeTqdm:
            def __init__(self, *args: object, **kwargs: object) -> None:
                inits.append(dict(kwargs))

            def __enter__(self) -> _FakeTqdm:
                return self

            def __exit__(self, *args: object) -> None:
                return None

            def update(self, n: int = 1) -> None:
                updates["n"] += n

        monkeypatch.setattr("niiflow.preproc.staging.stager.tqdm", _FakeTqdm)
        staged = RecordingStager().stage(entries)
        assert len(staged) == 2
        assert inits[0]["total"] == 2
        assert inits[0]["desc"] == "RecordingStager"
        assert updates["n"] == 2

    def test_threaded_stage_preserves_order(self, bids_tree: dict[str, Path]) -> None:
        other = _touch(bids_tree["root"] / "sub-02" / "func" / "run2.nii.gz")
        entries = make_entries([bids_tree["active"], other], {"label": "x"})
        staged = RecordingStager().stage(entries, num_workers=2)

        assert [e.active for e in staged] == [bids_tree["active"], other]
        assert [e.id for e in staged] == [e.id for e in entries]
        assert all(e.params["staged"] is True for e in staged)

    @pytest.mark.parametrize("num_workers", [1, 3])
    def test_stage_reconciles_stager_ids_in_input_order(
        self, tmp_path: Path, num_workers: int
    ) -> None:
        actives = [_touch(tmp_path / f"{index}.nii.gz") for index in range(3)]
        entries = make_entries(actives, [{"index": index} for index in range(3)])

        class DuplicateIdStager(Stager):
            def stage_single(self, entry: StagedEntry) -> StagedEntry:
                return StagedEntry(
                    active=entry.active,
                    id="duplicate",
                    params=entry.params,
                    errors=entry.errors,
                )

        staged = DuplicateIdStager().stage(entries, num_workers=num_workers)

        assert [entry.active for entry in staged] == actives
        assert [entry.params["index"] for entry in staged] == [0, 1, 2]
        assert [entry.id for entry in staged] == [
            "duplicate",
            f"{actives[1]}#2",
            f"{actives[2]}#2",
        ]

    def test_threaded_runs_entries_concurrently(
        self, bids_tree: dict[str, Path]
    ) -> None:
        import threading

        other = _touch(bids_tree["root"] / "sub-02" / "func" / "run2.nii.gz")
        entries = make_entries([bids_tree["active"], other], {"label": "x"})

        class BarrierStager(Stager):
            def __init__(self) -> None:
                self.barrier = threading.Barrier(2, timeout=5)
                self.max_concurrent = 0
                self._current = 0
                self._lock = threading.Lock()

            def stage_single(self, entry: StagedEntry) -> StagedEntry:
                with self._lock:
                    self._current += 1
                    self.max_concurrent = max(self.max_concurrent, self._current)
                self.barrier.wait()
                with self._lock:
                    self._current -= 1
                return entry

        stager = BarrierStager()
        staged = stager.stage(entries, num_workers=2)
        assert len(staged) == 2
        assert stager.max_concurrent == 2

    def test_threaded_allow_failed_entries_records_error_and_continues(
        self, bids_tree: dict[str, Path]
    ) -> None:
        bad = _touch(bids_tree["root"] / "sub-02" / "func" / "bad.nii.gz")
        good = _touch(bids_tree["root"] / "sub-03" / "func" / "good.nii.gz")
        entries = make_entries(
            [bad, good],
            [{"boom": True}, {"label": "ok"}],
        )
        stager = RecordingStager(allow_failed_entries=True)
        staged = stager.stage(entries, num_workers=2)

        assert len(staged) == 2
        assert staged[0].errors
        assert staged[0].errors[0].entry_id == entries[0].id
        assert staged[0].errors[0].entry_index == 0
        assert staged[1].params["staged"] is True
        assert not staged[1].errors

    def test_threaded_first_failure_aborts(self, bids_tree: dict[str, Path]) -> None:
        bad = _touch(bids_tree["root"] / "sub-02" / "func" / "bad.nii.gz")
        good = _touch(bids_tree["root"] / "sub-03" / "func" / "good.nii.gz")
        entries = make_entries(
            [bad, good],
            [{"boom": True}, {"label": "ok"}],
        )
        with pytest.raises(RuntimeError, match="boom"):
            RecordingStager().stage(entries, num_workers=2)

    def test_make_entries_classmethod_delegates(
        self, bids_tree: dict[str, Path]
    ) -> None:
        entries = RecordingStager.make_entries([bids_tree["active"]], {"x": 1})
        assert len(entries) == 1
        assert entries[0].params["x"] == 1
