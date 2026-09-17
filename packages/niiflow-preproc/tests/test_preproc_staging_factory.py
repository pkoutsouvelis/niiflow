"""Tests for :mod:`niiflow.preproc.staging.stager_factory`."""

from __future__ import annotations

from pathlib import Path

import pytest

from niiflow.preproc.staging import (
    EnsureActiveExists,
    EnsureActivesExist,
    FileStager,
    ResolveActiveReferences,
    ResolveParamReferences,
    StagedEntry,
    Stager,
    add_reference_staging_bookends,
    create_stager,
    discover_stager_classes,
    make_entries,
)


def _touch(path: Path, content: str = "nii") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path.resolve()


def _stage(stager: Stager, entries: list[StagedEntry]) -> list[StagedEntry]:
    staged = list(entries)
    for step in add_reference_staging_bookends([stager]):
        staged = step.stage(staged)
    return staged


@pytest.fixture
def bids_tree(tmp_path: Path) -> dict[str, Path]:
    root = tmp_path / "dataset"
    active = _touch(
        root / "sub-01" / "ses-pre" / "func" / "sub-01_ses-pre_task-rest_bold.nii.gz"
    )
    return {"root": root.resolve(), "active": active}


class TestCreateStager:
    def test_discover_stager_classes_lists_file_stager(self) -> None:
        registry = discover_stager_classes()

        assert "FileStager" in registry
        assert registry["FileStager"] is FileStager

    def test_creates_file_stager_by_class_name(self) -> None:
        stager = create_stager(
            "FileStager",
            {"pointers": {"input": "input"}},
        )

        assert isinstance(stager, FileStager)
        assert isinstance(stager, Stager)
        assert stager.pointers == {"input": "input"}

    def test_forwards_constructor_kwargs(self) -> None:
        stager = create_stager(
            "FileStager",
            {"pointers": {"output": "output"}, "allow_overwrite": True},
        )

        assert stager.allow_overwrite is True  # type: ignore[attr-defined]

    def test_created_stager_can_stage_entries(self, bids_tree: dict[str, Path]) -> None:
        stager = create_stager(
            "FileStager",
            {"pointers": {"input": "input"}},
        )
        entry = make_entries([bids_tree["active"]], {"input": None})[0]

        staged = _stage(stager, [entry])[0]

        assert staged.params["input"] == bids_tree["active"]

    def test_accepts_custom_registry(self) -> None:
        class DummyStager(Stager):
            def stage_single(self, entry: StagedEntry) -> StagedEntry:
                return entry

        stager = create_stager("DummyStager", registry={"DummyStager": DummyStager})

        assert isinstance(stager, DummyStager)

    def test_rejects_empty_stager_name(self) -> None:
        with pytest.raises(ValueError, match="non-empty string"):
            create_stager("   ")

    def test_rejects_unknown_stager_name(self) -> None:
        with pytest.raises(ValueError, match="Unknown stager"):
            create_stager(
                "NoSuchStager",
                {"pointers": {}},
            )

    def test_rejects_non_dict_stager_kwargs(self) -> None:
        with pytest.raises(TypeError, match="dictionary or None"):
            create_stager("FileStager", ["not", "a", "dict"])  # type: ignore[arg-type]

    def test_rejects_invalid_constructor_kwargs(self) -> None:
        with pytest.raises(TypeError, match="Failed to instantiate stager"):
            create_stager(
                "FileStager",
                {"pointers": {"input": "input"}, "not_a_flag": True},
            )

    def test_discover_includes_reference_stagers(self) -> None:
        registry = discover_stager_classes()
        assert "ResolveActiveReferences" in registry
        assert "ResolveParamReferences" in registry
        assert isinstance(
            create_stager("ResolveActiveReferences"), ResolveActiveReferences
        )
        assert isinstance(
            create_stager("ResolveParamReferences"), ResolveParamReferences
        )

    def test_discover_includes_ensure_actives_exist(self) -> None:
        registry = discover_stager_classes()
        assert "EnsureActivesExist" in registry
        stager = create_stager("EnsureActivesExist", {"allow_failed_entries": True})
        assert isinstance(stager, EnsureActivesExist)
        assert stager.allow_failed_entries is True

    def test_ensure_active_exists_alias_still_constructible(self) -> None:
        registry = discover_stager_classes()
        assert "EnsureActiveExists" in registry
        with pytest.warns(DeprecationWarning, match="Use `EnsureActivesExist` instead"):
            stager = create_stager("EnsureActiveExists", {"allow_failed_entries": True})
        assert isinstance(stager, EnsureActiveExists)
        assert isinstance(stager, EnsureActivesExist)
        assert stager.allow_failed_entries is True
