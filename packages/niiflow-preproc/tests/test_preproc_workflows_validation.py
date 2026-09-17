"""Tests for shared workflow-entry validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from niiflow.preproc.staging import StagedEntry
from niiflow.preproc.workflows.validation import validate_staged_entries


def _entry(tmp_path: Path, entry_id: Any) -> StagedEntry:
    return StagedEntry(
        active=(tmp_path / f"entry-{entry_id}.nii.gz").resolve(),
        params={},
        id=entry_id,
    )


class TestValidateStagedEntries:
    def test_returns_tuple_in_input_order(self, tmp_path: Path) -> None:
        entries = [
            _entry(tmp_path, "third"),
            _entry(tmp_path, "first"),
            _entry(tmp_path, "second"),
        ]

        validated = validate_staged_entries(entries)

        assert validated == tuple(entries)
        assert all(
            restored is original
            for restored, original in zip(validated, entries, strict=True)
        )

    def test_accepts_empty_sequence(self) -> None:
        assert validate_staged_entries([]) == ()

    @pytest.mark.parametrize(
        ("entries", "bad_index"),
        [
            (["not-an-entry"], 0),
            ([object(), "not-an-entry"], 0),
        ],
    )
    def test_rejects_non_staged_entries(
        self,
        entries: list[object],
        bad_index: int,
    ) -> None:
        with pytest.raises(
            TypeError,
            match=rf"only `StagedEntry` objects.*index {bad_index}",
        ):
            validate_staged_entries(entries)  # type: ignore[arg-type]

    def test_reports_non_staged_entry_position(self, tmp_path: Path) -> None:
        entries = [_entry(tmp_path, "valid"), object()]

        with pytest.raises(TypeError, match=r"object at index 1"):
            validate_staged_entries(entries)  # type: ignore[arg-type]

    @pytest.mark.parametrize("entry_id", ["", None, 7, Path("id")])
    def test_rejects_invalid_entry_ids(
        self,
        tmp_path: Path,
        entry_id: object,
    ) -> None:
        entry = _entry(tmp_path, entry_id)

        with pytest.raises(ValueError, match="non-empty string `id`"):
            validate_staged_entries([entry])

    def test_rejects_duplicate_ids_and_reports_position(
        self,
        tmp_path: Path,
    ) -> None:
        entries = [
            _entry(tmp_path, "same"),
            _entry(tmp_path, "other"),
            _entry(tmp_path, "same"),
        ]

        with pytest.raises(
            ValueError,
            match=r"duplicate `same` at index 2",
        ):
            validate_staged_entries(entries)
