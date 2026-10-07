"""Tests for :mod:`niiflow.preproc.staging.utility`."""

from __future__ import annotations

from pathlib import Path

import pytest

from niiflow.preproc.staging import ActiveStager, EnsureActiveExists, make_entries


def _touch(path: Path, content: str = "nii") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path.resolve()


class TestActiveStager:
    def test_passes_existing_active(self, tmp_path: Path) -> None:
        active = _touch(tmp_path / "a.nii.gz")
        entry = make_entries([active], {})[0]
        staged = ActiveStager().stage([entry])[0]
        assert staged.errors == ()
        assert staged.active == active
        assert staged.id == entry.id

    def test_raises_on_missing_active(self, tmp_path: Path) -> None:
        missing = tmp_path / "missing.nii.gz"
        entry = make_entries([missing], {}, must_exist=False)[0]
        with pytest.raises(FileNotFoundError, match="does not exist"):
            ActiveStager().stage([entry])

    def test_must_exist_false_keeps_missing(self, tmp_path: Path) -> None:
        missing = tmp_path / "missing.nii.gz"
        entry = make_entries([missing], {}, must_exist=False)[0]
        staged = ActiveStager(must_exist=False, resolve=False).stage([entry])[0]
        assert staged.errors == ()
        assert staged.active == missing.absolute()

    def test_resolve_false_keeps_symlink(self, tmp_path: Path) -> None:
        target = _touch(tmp_path / "target.nii.gz")
        link = tmp_path / "alias.nii.gz"
        link.symlink_to(target)
        entry = make_entries([link], {}, resolve_actives=False)[0]
        staged = ActiveStager(resolve=False).stage([entry])[0]
        assert staged.active == link.absolute()
        assert staged.active != target

    def test_allow_failed_entries_records_error(self, tmp_path: Path) -> None:
        missing = tmp_path / "missing.nii.gz"
        existing = _touch(tmp_path / "ok.nii.gz")
        entries = make_entries([missing, existing], [{}, {}], must_exist=False)
        staged = ActiveStager(allow_failed_entries=True).stage(entries)

        assert staged[0].errors
        assert staged[0].errors[0].stage == "ActiveStager"
        assert staged[0].errors[0].entry_id == entries[0].id
        assert staged[1].errors == ()
        assert staged[1].id == entries[1].id

    def test_ensure_active_exists_is_deprecated_alias(self, tmp_path: Path) -> None:
        active = _touch(tmp_path / "a.nii.gz")
        entry = make_entries([active], {})[0]
        with pytest.warns(DeprecationWarning, match="Use `ActiveStager` instead"):
            staged = EnsureActiveExists().stage([entry])[0]
        assert staged.errors == ()
        assert issubclass(EnsureActiveExists, ActiveStager)
