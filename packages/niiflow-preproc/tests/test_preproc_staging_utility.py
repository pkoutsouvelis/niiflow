"""Tests for :mod:`niiflow.preproc.staging.utility`."""

from __future__ import annotations

from pathlib import Path

import pytest

from niiflow.preproc.staging import EnsureActiveExists, make_entries


def _touch(path: Path, content: str = "nii") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path.resolve()


class TestEnsureActiveExists:
    def test_passes_existing_active(self, tmp_path: Path) -> None:
        active = _touch(tmp_path / "a.nii.gz")
        entry = make_entries([active], {})[0]
        staged = EnsureActiveExists().stage([entry])[0]
        assert staged.errors == ()
        assert staged.active == active

    def test_raises_on_missing_active(self, tmp_path: Path) -> None:
        missing = tmp_path / "missing.nii.gz"
        entry = make_entries([missing], {})[0]
        with pytest.raises(FileNotFoundError, match="does not exist"):
            EnsureActiveExists().stage([entry])

    def test_allow_failed_entries_records_error(self, tmp_path: Path) -> None:
        missing = tmp_path / "missing.nii.gz"
        existing = _touch(tmp_path / "ok.nii.gz")
        entries = make_entries([missing, existing], [{}, {}])
        staged = EnsureActiveExists(allow_failed_entries=True).stage(entries)

        assert staged[0].errors
        assert staged[0].errors[0].stage == "EnsureActiveExists"
        assert staged[1].errors == ()
        assert staged[1].active == existing
