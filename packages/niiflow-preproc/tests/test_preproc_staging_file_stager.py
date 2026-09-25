"""Tests for :class:`FileStager`.

These exercise input/output pointer resolution, root specs, search policies, and
FileStager-specific error handling. Dynamic-reference helpers and bookend
stagers live in ``test_preproc_staging_dynamic_referencing.py``; ``make_entries`` / base
``Stager`` behaviour live in ``test_preproc_staging_base.py``.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from niiflow.preproc.staging import (
    FileStager,
    FileStagingError,
    StagedEntry,
    add_reference_staging_bookends,
    make_entries,
)
from niiflow.preproc.staging.stager import StagingContext, Stager


def _stage(stager: Stager, entries: Sequence[StagedEntry]) -> list[StagedEntry]:
    """Run FileStager between active/param reference bookends."""
    staged = list(entries)
    for step in add_reference_staging_bookends([stager]):
        staged = step.stage(staged)
    return staged


def _touch(path: Path, content: str = "nii") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path.resolve()


def _file_stem(path: Path) -> str:
    return path.stem


@pytest.fixture
def bids_tree(tmp_path: Path) -> dict[str, Path]:
    """Minimal BIDS-like tree with an active file and a sibling derivative root."""
    root = tmp_path / "dataset"
    active = _touch(
        root / "sub-01" / "ses-pre" / "func" / "sub-01_ses-pre_task-rest_bold.nii.gz"
    )
    derivative = root / "derivatives" / "niiflow"
    _touch(derivative / "sub-01" / "ses-pre" / "placeholder.txt")
    return {
        "root": root.resolve(),
        "active": active,
        "derivative": derivative.resolve(),
    }


class _StaticExplorer:
    """Explorer double that returns predetermined paths per search root."""

    def __init__(self, files_by_root: dict[Path, Sequence[Path | str]]) -> None:
        self.files_by_root = {
            Path(root).resolve(): [Path(path).resolve() for path in paths]
            for root, paths in files_by_root.items()
        }

    def list(
        self, root: Path | str, *, sort: bool = True, unique: bool = True
    ) -> list[str]:
        paths = list(self.files_by_root.get(Path(root).resolve(), []))
        if unique:
            paths = list(dict.fromkeys(paths))
        if sort:
            paths.sort()
        return [str(path) for path in paths]


def _patch_explorer(
    monkeypatch: pytest.MonkeyPatch,
    files_by_root: dict[Path, Sequence[Path | str]],
) -> _StaticExplorer:
    explorer = _StaticExplorer(files_by_root)

    def _factory(**search_spec: Any) -> _StaticExplorer:
        return explorer

    monkeypatch.setattr(
        "niiflow.preproc.staging.file_stager.get_data_explorer",
        _factory,
    )
    return explorer


# ---------------------------------------------------------------------------
# Pointer validation
# ---------------------------------------------------------------------------


class TestPointerValidation:
    def test_rejects_unknown_pointer_kind(self) -> None:
        with pytest.raises(ValueError, match="must be 'input' or 'output'"):
            FileStager({"mask": "inpuit"})  # type: ignore[arg-type]

    def test_rejects_non_string_pointer_keys(self) -> None:
        with pytest.raises(TypeError, match="Pointer keys must be strings"):
            FileStager({1: "input"})  # type: ignore[arg-type]

    @pytest.mark.parametrize("pointers", [None, {}])
    def test_rejects_empty_or_missing_pointers(
        self, pointers: dict[str, str] | None
    ) -> None:
        with pytest.raises((TypeError, ValueError)):
            FileStager(pointers)  # type: ignore[arg-type]

    def test_rejects_non_mapping_pointers(self) -> None:
        with pytest.raises(TypeError, match="pointers must be a dictionary"):
            FileStager(["mask"])  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Direct path resolution
# ---------------------------------------------------------------------------


class TestDirectPaths:
    def test_none_input_defaults_to_active_file(
        self, bids_tree: dict[str, Path]
    ) -> None:
        stager = FileStager({"input": "input"})
        entry = make_entries([bids_tree["active"]], {"input": None})[0]
        staged = _stage(stager, [entry])[0]

        assert staged.params["input"] == bids_tree["active"]

    def test_string_and_path_inputs_are_resolved(
        self, bids_tree: dict[str, Path]
    ) -> None:
        mask = _touch(bids_tree["root"] / "masks" / "brain.nii.gz")
        stager = FileStager({"mask": "input"})
        entry = make_entries([bids_tree["active"]], {"mask": str(mask)})[0]
        staged = _stage(stager, [entry])[0]

        assert staged.params["mask"] == mask

    def test_relative_string_input_is_anchored_to_active_parent(
        self, bids_tree: dict[str, Path]
    ) -> None:
        mask = _touch(bids_tree["active"].parent / "brain_mask.nii.gz")
        stager = FileStager({"mask": "input"})
        entry = make_entries([bids_tree["active"]], {"mask": "brain_mask.nii.gz"})[0]
        staged = _stage(stager, [entry])[0]

        assert staged.params["mask"] == mask

    def test_none_output_is_rejected(self, bids_tree: dict[str, Path]) -> None:
        stager = FileStager({"output": "output"})
        entry = make_entries([bids_tree["active"]], {"output": None})[0]

        with pytest.raises(FileStagingError, match="cannot be None"):
            _stage(stager, [entry])

    def test_output_dict_builds_path_under_active_parent(
        self, bids_tree: dict[str, Path]
    ) -> None:
        stager = FileStager({"output": "output"})
        entry = make_entries(
            [bids_tree["active"]],
            {"output": {"name": "denoised.nii.gz"}},
        )[0]
        staged = _stage(stager, [entry])[0]

        expected = bids_tree["active"].parent / "denoised.nii.gz"
        assert staged.params["output"] == expected

    def test_existing_output_file_is_allowed_by_default(
        self, bids_tree: dict[str, Path]
    ) -> None:
        existing = _touch(bids_tree["active"].parent / "denoised.nii.gz")
        stager = FileStager({"output": "output"})
        entry = make_entries([bids_tree["active"]], {"output": str(existing)})[0]
        staged = _stage(stager, [entry])[0]

        assert staged.params["output"] == existing

    def test_allow_overwrite_false_blocks_existing_output(
        self, bids_tree: dict[str, Path]
    ) -> None:
        existing = _touch(bids_tree["active"].parent / "denoised.nii.gz")
        stager = FileStager({"output": "output"}, allow_overwrite=False)
        entry = make_entries([bids_tree["active"]], {"output": str(existing)})[0]

        with pytest.raises(FileStagingError, match="already exists"):
            _stage(stager, [entry])

    def test_output_cannot_point_to_existing_directory(
        self, bids_tree: dict[str, Path]
    ) -> None:
        out_dir = bids_tree["active"].parent / "denoised.nii.gz"
        out_dir.mkdir()
        stager = FileStager({"output": "output"}, allow_overwrite=True)
        entry = make_entries([bids_tree["active"]], {"output": str(out_dir)})[0]

        with pytest.raises(FileStagingError, match="existing directory"):
            _stage(stager, [entry])

    def test_ensure_inputs_exist_false_allows_missing_direct_input(
        self, bids_tree: dict[str, Path], tmp_path: Path
    ) -> None:
        missing = tmp_path / "missing_mask.nii.gz"
        stager = FileStager({"mask": "input"}, ensure_inputs_exist=False)
        entry = make_entries([bids_tree["active"]], {"mask": str(missing)})[0]
        staged = _stage(stager, [entry])[0]

        assert staged.params["mask"] == missing.resolve()


# ---------------------------------------------------------------------------
# Root resolution
# ---------------------------------------------------------------------------


class TestRootResolution:
    def test_omitted_mode_uses_active_parent(self, bids_tree: dict[str, Path]) -> None:
        stager = FileStager({"input": "input"})
        ctx = StagingContext(active=bids_tree["active"])
        root = stager.get_root({}, ctx=ctx, must_exist=True)

        assert root == bids_tree["active"].parent
        assert stager.get_root(None, ctx=ctx, must_exist=True) == root

    def test_path_root_resolves_explicit_directory(
        self, bids_tree: dict[str, Path]
    ) -> None:
        stager = FileStager({"input": "input"})
        ctx = StagingContext(active=bids_tree["active"])
        root = stager.get_root(
            {"mode": "path", "value": bids_tree["derivative"]},
            ctx=ctx,
            must_exist=True,
        )

        assert root == bids_tree["derivative"]

    def test_parent_up_moves_from_active_parent(
        self, bids_tree: dict[str, Path]
    ) -> None:
        stager = FileStager({"input": "input"})
        ctx = StagingContext(active=bids_tree["active"])
        root = stager.get_root(
            {"mode": "parent_up", "value": 2}, ctx=ctx, must_exist=True
        )

        assert root == bids_tree["active"].parents[2]

    def test_parent_match_finds_nearest_matching_ancestor(
        self, bids_tree: dict[str, Path]
    ) -> None:
        stager = FileStager({"input": "input"})
        ctx = StagingContext(active=bids_tree["active"])
        root = stager.get_root(
            {"mode": "parent_match", "value": "sub-*/ses-*"},
            ctx=ctx,
            must_exist=True,
        )

        assert root == bids_tree["active"].parent.parent

    def test_parent_match_most_global_selects_dataset_root(
        self, bids_tree: dict[str, Path]
    ) -> None:
        stager = FileStager({"input": "input"})
        ctx = StagingContext(active=bids_tree["active"])
        root = stager.get_root(
            {
                "mode": "parent_match",
                "value": "sub-*",
                "selection": "most_global",
            },
            ctx=ctx,
            must_exist=True,
        )

        assert root == bids_tree["root"] / "sub-01"

    def test_mirror_root_maps_derivative_tail(self, bids_tree: dict[str, Path]) -> None:
        stager = FileStager({"input": "input"})
        ctx = StagingContext(active=bids_tree["active"])
        mirrored = stager.get_root(
            {
                "mirror": {
                    "source": bids_tree["root"],
                    "target": bids_tree["derivative"],
                },
            },
            ctx=ctx,
            must_exist=False,
        )

        expected = bids_tree["derivative"] / "sub-01" / "ses-pre" / "func"
        assert mirrored == expected

    def test_mirror_tries_sources_in_order(self, bids_tree: dict[str, Path]) -> None:
        stager = FileStager({"input": "input"})
        ctx = StagingContext(active=bids_tree["active"])
        missing = bids_tree["root"] / "does-not-exist"
        mirrored = stager.get_root(
            {
                "mirror": {
                    "source": [missing, bids_tree["root"]],
                    "target": bids_tree["derivative"],
                },
            },
            ctx=ctx,
            must_exist=False,
        )

        expected = bids_tree["derivative"] / "sub-01" / "ses-pre" / "func"
        assert mirrored == expected

    def test_mirror_allow_missing_source_keeps_root(
        self, bids_tree: dict[str, Path]
    ) -> None:
        stager = FileStager({"input": "input"})
        ctx = StagingContext(active=bids_tree["active"])
        missing = bids_tree["root"] / "does-not-exist"
        mirrored = stager.get_root(
            {
                "mirror": {
                    "source": missing,
                    "target": bids_tree["derivative"],
                    "allow_missing_source": True,
                },
            },
            ctx=ctx,
            must_exist=False,
        )

        assert mirrored == bids_tree["active"].parent

    def test_mirror_missing_source_raises_by_default(
        self, bids_tree: dict[str, Path]
    ) -> None:
        stager = FileStager({"input": "input"})
        ctx = StagingContext(active=bids_tree["active"])
        missing = bids_tree["root"] / "does-not-exist"
        with pytest.raises(FileNotFoundError, match="Cannot mirror root"):
            stager.get_root(
                {
                    "mirror": {
                        "source": [missing],
                        "target": bids_tree["derivative"],
                    },
                },
                ctx=ctx,
                must_exist=False,
            )

    def test_get_roots_accepts_list_of_root_specs(
        self, bids_tree: dict[str, Path]
    ) -> None:
        stager = FileStager({"input": "input"})
        ctx = StagingContext(active=bids_tree["active"])
        roots = stager.get_roots(
            [
                None,
                {"mode": "path", "value": bids_tree["derivative"]},
            ],
            ctx=ctx,
            must_exist=True,
        )

        assert roots == [bids_tree["active"].parent, bids_tree["derivative"]]

    def test_rejects_value_without_mode(self, bids_tree: dict[str, Path]) -> None:
        stager = FileStager({"input": "input"})
        ctx = StagingContext(active=bids_tree["active"])

        with pytest.raises(ValueError, match="must not define a value"):
            stager.get_root({"value": "/tmp"}, ctx=ctx, must_exist=True)

    def test_rejects_empty_root_list(self, bids_tree: dict[str, Path]) -> None:
        stager = FileStager({"input": "input"})
        ctx = StagingContext(active=bids_tree["active"])

        with pytest.raises(ValueError, match="cannot be empty"):
            stager.get_roots([], ctx=ctx, must_exist=True)


# ---------------------------------------------------------------------------
# Search-based inputs
# ---------------------------------------------------------------------------


class TestSearchInputs:
    def test_search_first_returns_lexicographically_first_match(
        self,
        bids_tree: dict[str, Path],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        search_root = bids_tree["active"].parent
        later = _touch(search_root / "z_last.nii.gz")
        earlier = _touch(search_root / "a_first.nii.gz")
        _patch_explorer(
            monkeypatch,
            {search_root: [later, earlier]},
        )

        stager = FileStager({"mask": "input"})
        entry = make_entries(
            [bids_tree["active"]],
            {
                "mask": {
                    "search": {"patterns": "*.nii.gz"},
                    "resolve_results": "first",
                }
            },
        )[0]
        staged = _stage(stager, [entry])[0]

        assert staged.params["mask"] == earlier

    def test_search_all_returns_sorted_paths(
        self,
        bids_tree: dict[str, Path],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        search_root = bids_tree["active"].parent
        b = _touch(search_root / "b.nii.gz")
        a = _touch(search_root / "a.nii.gz")
        _patch_explorer(monkeypatch, {search_root: [b, a]})

        stager = FileStager({"mask": "input"})
        entry = make_entries(
            [bids_tree["active"]],
            {
                "mask": {
                    "search": {"patterns": "*.nii.gz"},
                    "resolve_results": "all",
                }
            },
        )[0]
        staged = _stage(stager, [entry])[0]

        assert staged.params["mask"] == [a, b]

    def test_search_single_requires_exactly_one_match(
        self,
        bids_tree: dict[str, Path],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        search_root = bids_tree["active"].parent
        only = _touch(search_root / "only.nii.gz")
        _patch_explorer(monkeypatch, {search_root: [only]})

        stager = FileStager({"mask": "input"})
        entry = make_entries(
            [bids_tree["active"]],
            {
                "mask": {
                    "search": {"patterns": "*.nii.gz"},
                    "resolve_results": "single",
                }
            },
        )[0]
        staged = _stage(stager, [entry])[0]

        assert staged.params["mask"] == only

    def test_search_single_raises_when_multiple_matches(
        self,
        bids_tree: dict[str, Path],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        search_root = bids_tree["active"].parent
        _patch_explorer(
            monkeypatch,
            {
                search_root: [
                    _touch(search_root / "a.nii.gz"),
                    _touch(search_root / "b.nii.gz"),
                ]
            },
        )

        stager = FileStager({"mask": "input"})
        entry = make_entries(
            [bids_tree["active"]],
            {
                "mask": {
                    "search": {"patterns": "*.nii.gz"},
                    "resolve_results": "single",
                }
            },
        )[0]

        with pytest.raises(FileStagingError, match="single file"):
            _stage(stager, [entry])

    def test_search_with_multiple_roots_concatenates_results(
        self,
        bids_tree: dict[str, Path],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        root_a = bids_tree["active"].parent
        root_b = bids_tree["derivative"] / "sub-01" / "ses-pre"
        file_a = _touch(root_a / "a.nii.gz")
        file_b = _touch(root_b / "b.nii.gz")
        _patch_explorer(monkeypatch, {root_a: [file_a], root_b: [file_b]})

        stager = FileStager({"mask": "input"})
        entry = make_entries(
            [bids_tree["active"]],
            {
                "mask": {
                    "root": [
                        None,
                        {
                            "mode": "path",
                            "value": root_b,
                        },
                    ],
                    "search": {"patterns": "*.nii.gz"},
                    "resolve_results": "all",
                }
            },
        )[0]
        staged = _stage(stager, [entry])[0]

        assert staged.params["mask"] == sorted([file_a, file_b])

    def test_search_requires_existing_roots_even_when_inputs_not_checked(
        self,
        bids_tree: dict[str, Path],
        tmp_path: Path,
    ) -> None:
        missing_root = tmp_path / "missing"
        stager = FileStager({"mask": "input"}, ensure_inputs_exist=False)
        entry = make_entries(
            [bids_tree["active"]],
            {
                "mask": {
                    "root": {"mode": "path", "value": missing_root},
                    "search": {"patterns": "*.nii.gz"},
                }
            },
        )[0]

        with pytest.raises(FileStagingError, match="Directory does not exist"):
            _stage(stager, [entry])

    def test_does_not_create_new_explorer_for_identical_search_specs(
        self,
        bids_tree: dict[str, Path],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        search_root = bids_tree["active"].parent
        only = _touch(search_root / "cached.nii.gz")
        calls: list[dict[str, Any]] = []

        def _factory(**search_spec: Any) -> _StaticExplorer:
            calls.append(search_spec)
            return _StaticExplorer({search_root: [only]})

        monkeypatch.setattr(
            "niiflow.preproc.staging.file_stager.get_data_explorer",
            _factory,
        )

        stager = FileStager({"mask": "input"})
        search = {"patterns": "*.nii.gz"}
        ctx = StagingContext(active=bids_tree["active"])
        spec = {"search": search}

        stager.get_input_file(spec, ctx=ctx)  # type: ignore[arg-type]
        stager.get_input_file(spec, ctx=ctx)  # type: ignore[arg-type]

        assert len(calls) == 1

    def test_reuses_same_explorer_instance_for_identical_search_specs(
        self,
        bids_tree: dict[str, Path],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        search_root = bids_tree["active"].parent
        only = _touch(search_root / "cached.nii.gz")

        monkeypatch.setattr(
            "niiflow.preproc.staging.file_stager.get_data_explorer",
            lambda **search_spec: _StaticExplorer({search_root: [only]}),
        )

        stager = FileStager({"mask": "input"})
        search = {"patterns": "*.nii.gz"}
        first = stager.get_explorer(search)
        second = stager.get_explorer(search)
        assert first is second


# ---------------------------------------------------------------------------
# Batch staging / error handling
# ---------------------------------------------------------------------------


class TestBatchStaging:
    def test_stage_deep_copies_shared_params_per_entry(
        self, bids_tree: dict[str, Path]
    ) -> None:
        other = _touch(bids_tree["root"] / "sub-02" / "func" / "run2.nii.gz")
        shared = {"output": "{active.stem}_out.nii.gz"}
        entries = make_entries([bids_tree["active"], other], shared)
        stager = FileStager({"output": "output"})
        staged = _stage(stager, entries)

        assert staged[0].params["output"] != staged[1].params["output"]
        assert staged[0].params is not shared
        assert staged[1].params is not shared
        assert [entry.id for entry in staged] == [entry.id for entry in entries]

    def test_active_file_is_never_modified(self, bids_tree: dict[str, Path]) -> None:
        stager = FileStager({"output": "output"})
        entry = make_entries(
            [bids_tree["active"]],
            {"output": "{active.stem}_out.nii.gz"},
        )[0]
        staged = _stage(stager, [entry])[0]

        assert staged.active == bids_tree["active"]

    def test_allow_failed_entries_records_error_and_continues(
        self, bids_tree: dict[str, Path]
    ) -> None:
        bad = _touch(bids_tree["root"] / "sub-02" / "func" / "bad.nii.gz")
        good = _touch(bids_tree["root"] / "sub-03" / "func" / "good.nii.gz")
        existing = _touch(bad.parent / "blocked.nii.gz")

        entries = make_entries(
            [bad, good],
            [
                {"output": str(existing)},
                {
                    "output": {
                        "name": "{active.stem}_ok.nii.gz",
                    }
                },
            ],
        )
        stager = FileStager(
            {"output": "output"}, allow_failed_entries=True, allow_overwrite=False
        )
        staged = _stage(stager, entries)

        assert len(staged) == 2
        assert staged[0].errors
        assert staged[0].errors[0].error_type == "FileStagingError"
        assert staged[0].errors[0].entry_id == entries[0].id
        assert staged[0].errors[0].entry_index == 0
        assert (
            staged[1].params["output"] == good.parent / f"{_file_stem(good)}_ok.nii.gz"
        )
        assert not staged[1].errors

    def test_pointer_context_is_included_in_staging_error(
        self,
        bids_tree: dict[str, Path],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        search_root = bids_tree["active"].parent
        _patch_explorer(monkeypatch, {search_root: []})

        stager = FileStager({"mask": "input"})
        entry = make_entries(
            [bids_tree["active"]],
            {
                "mask": {
                    "search": {"patterns": "*.nii.gz"},
                }
            },
        )[0]

        with pytest.raises(FileStagingError, match="input pointer 'mask'"):
            _stage(stager, [entry])

    def test_missing_active_as_input_surfaces_as_staging_error(
        self, tmp_path: Path
    ) -> None:
        missing = tmp_path / "missing.nii.gz"
        stager = FileStager({"input": "input"})
        entry = StagedEntry(
            active=missing,
            id="missing-active",
            params={"input": None},
        )

        with pytest.raises(FileStagingError, match="does not exist"):
            stager.stage_single(entry)

    def test_missing_active_allowed_when_not_used_as_input(
        self, tmp_path: Path
    ) -> None:
        missing = tmp_path / "planned" / "missing.nii.gz"
        existing = _touch(tmp_path / "mask.nii.gz")
        stager = FileStager({"mask": "input", "output": "output"})
        entry = StagedEntry(
            active=missing,
            id="planned-active",
            params={"mask": str(existing), "output": "out.nii.gz"},
        )

        staged = stager.stage_single(entry)
        assert staged.id == entry.id
        assert staged.params["mask"] == existing.resolve()
        assert staged.params["output"] == (missing.parent / "out.nii.gz").resolve()


# ---------------------------------------------------------------------------
# Name-based inputs
# ---------------------------------------------------------------------------


class TestNameInputs:
    def test_name_joins_default_root_and_filename(
        self, bids_tree: dict[str, Path]
    ) -> None:
        companion = _touch(bids_tree["active"].parent / "mask.nii.gz")
        stager = FileStager({"mask": "input"})
        entry = make_entries(
            [bids_tree["active"]],
            {"mask": {"name": "mask.nii.gz"}},
        )[0]
        staged = _stage(stager, [entry])[0]
        assert staged.params["mask"] == companion

    def test_name_uses_active_name_reference(self, bids_tree: dict[str, Path]) -> None:
        stager = FileStager({"mask": "input"})
        entry = make_entries(
            [bids_tree["active"]],
            {"mask": {"name": "{active.name}"}},
        )[0]
        staged = _stage(stager, [entry])[0]
        assert staged.params["mask"] == bids_tree["active"]

    def test_name_with_mirrored_root(self, bids_tree: dict[str, Path]) -> None:
        mirrored = _touch(
            bids_tree["derivative"] / "sub-01" / "ses-pre" / "func" / "mask.nii.gz"
        )
        stager = FileStager({"mask": "input"})
        entry = make_entries(
            [bids_tree["active"]],
            {
                "mask": {
                    "root": {
                        "mirror": {
                            "source": bids_tree["root"],
                            "target": bids_tree["derivative"],
                        }
                    },
                    "name": "mask.nii.gz",
                }
            },
        )[0]
        staged = _stage(stager, [entry])[0]
        assert staged.params["mask"] == mirrored

    def test_name_rejects_list_root(self, bids_tree: dict[str, Path]) -> None:
        stager = FileStager({"mask": "input"})
        ctx = StagingContext(active=bids_tree["active"])
        with pytest.raises(TypeError, match="cannot be a list"):
            stager.get_input_file(
                {"root": [None], "name": "mask.nii.gz"},
                ctx=ctx,
            )


# ---------------------------------------------------------------------------
# Spec validation
# ---------------------------------------------------------------------------


class TestSpecValidation:
    def test_input_spec_requires_search_or_name(
        self, bids_tree: dict[str, Path]
    ) -> None:
        stager = FileStager({"mask": "input"})
        ctx = StagingContext(active=bids_tree["active"])

        with pytest.raises(ValueError, match="exactly one of 'search' or 'name'"):
            stager.get_input_file({}, ctx=ctx)  # type: ignore[arg-type]

    def test_input_spec_rejects_search_and_name(
        self, bids_tree: dict[str, Path]
    ) -> None:
        stager = FileStager({"mask": "input"})
        ctx = StagingContext(active=bids_tree["active"])

        with pytest.raises(ValueError, match="exactly one of 'search' or 'name'"):
            stager.get_input_file(
                {"search": {"patterns": "*.nii.gz"}, "name": "a.nii.gz"},  # type: ignore[arg-type]
                ctx=ctx,
            )

    def test_input_spec_rejects_resolve_results_with_name(
        self, bids_tree: dict[str, Path]
    ) -> None:
        stager = FileStager({"mask": "input"})
        ctx = StagingContext(active=bids_tree["active"])

        with pytest.raises(ValueError, match="only valid with 'search'"):
            stager.get_input_file(
                {"name": "a.nii.gz", "resolve_results": "first"},  # type: ignore[arg-type]
                ctx=ctx,
            )

    def test_output_root_cannot_be_a_list(self, bids_tree: dict[str, Path]) -> None:
        stager = FileStager({"output": "output"})
        ctx = StagingContext(active=bids_tree["active"])

        with pytest.raises(TypeError, match="cannot be a list"):
            stager.get_output_path(
                {"root": [None], "name": "out.nii.gz"},
                ctx=ctx,
            )

    def test_unknown_root_mode_is_rejected(self, bids_tree: dict[str, Path]) -> None:
        stager = FileStager({"input": "input"})
        ctx = StagingContext(active=bids_tree["active"])

        with pytest.raises(ValueError, match="Unknown root mode"):
            stager.get_root({"mode": "unknown"}, ctx=ctx, must_exist=False)  # type: ignore[typeddict-item]
        with pytest.raises(ValueError, match="Unknown root mode"):
            stager.get_root({"mode": "active"}, ctx=ctx, must_exist=False)  # type: ignore[typeddict-item]

    def test_unsupported_spec_keys_are_rejected(
        self, bids_tree: dict[str, Path]
    ) -> None:
        stager = FileStager({"mask": "input"})
        ctx = StagingContext(active=bids_tree["active"])

        with pytest.raises(ValueError, match="unsupported key"):
            stager.get_input_file(
                {"search": {"patterns": "*.nii.gz"}, "extra": True},  # type: ignore[arg-type]
                ctx=ctx,
            )
