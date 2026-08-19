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
from niiflow.preproc.staging.stager import StagingContext, Stager


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
        shared = {"mask": "to-be-resolved"}
        entries = make_entries([bids_tree["active"], other], shared)

        assert len(entries) == 2
        assert entries[0].active == bids_tree["active"]
        assert entries[1].active == other
        assert entries[0].params == shared
        assert entries[1].params == shared
        assert entries[0].params is not entries[1].params
        assert entries[0].params is not shared

        entries[0].params["mask"] = "mutated"
        assert entries[1].params["mask"] == "to-be-resolved"

    def test_per_entry_params_must_align_with_active_files(
        self, bids_tree: dict[str, Path]
    ) -> None:
        other = _touch(bids_tree["root"] / "sub-02" / "func" / "run2.nii.gz")
        entries = make_entries(
            [bids_tree["active"], other],
            [{"input": None}, {"input": str(other)}],
        )

        assert entries[0].params["input"] is None
        assert entries[1].params["input"] == str(other)

    def test_rejects_empty_active_files(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            make_entries([], {"input": None})

    def test_allows_missing_active_file(self, tmp_path: Path) -> None:
        missing = tmp_path / "missing.nii.gz"
        entries = make_entries([missing], {"label": "x"})
        assert len(entries) == 1
        assert entries[0].active == missing.resolve()
        assert entries[0].params == {"label": "x"}

    def test_rejects_directory_active_path(self, tmp_path: Path) -> None:
        directory = tmp_path / "folder"
        directory.mkdir()
        with pytest.raises(ValueError, match="must be a file"):
            make_entries([directory], {"input": None})

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

    def test_make_entries_classmethod_delegates(
        self, bids_tree: dict[str, Path]
    ) -> None:
        entries = RecordingStager.make_entries([bids_tree["active"]], {"x": 1})
        assert len(entries) == 1
        assert entries[0].params["x"] == 1
