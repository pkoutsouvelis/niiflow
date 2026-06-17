"""Tests for :class:`FileStager` and the staging entry-point API.

These tests exercise user-facing resolution of input/output pointers, dynamic
references (including ``|strip:<suffix>`` modifiers), root specs, search
policies, the ``allow_overwrite`` / ``ensure_inputs_exist`` /
``allow_failed_entries`` knobs, and
:func:`~niiflow.preproc.staging.stager_factory.create_stager`. Search-based inputs use a
lightweight explorer double so the suite does not depend on real
``nifti_finder`` traversal.
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
    create_stager,
    make_entries,
)
from niiflow.preproc.staging.stager import (
    StageContext,
    Stager,
    StagedEntry,
    StagingErrorRecord,
)
from niiflow.preproc.staging.stager_factory import discover_stager_classes
from niiflow.preproc.utils.file import get_ext


def _touch(path: Path, content: str = "nii") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path.resolve()


def _file_stem(path: Path) -> str:
    ext = get_ext(path)
    return path.name[: -len(ext)] if ext else path.stem


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
# make_entries
# ---------------------------------------------------------------------------


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

    def test_rejects_missing_active_file(self, tmp_path: Path) -> None:
        missing = tmp_path / "missing.nii.gz"
        with pytest.raises(FileNotFoundError, match="does not exist"):
            make_entries([missing], {"input": None})

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
            make_entries([bids_tree["active"]], ["not-a-dict"])


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


# ---------------------------------------------------------------------------
# Direct input / output resolution
# ---------------------------------------------------------------------------


class TestDirectPaths:
    def test_none_input_defaults_to_active_file(
        self, bids_tree: dict[str, Path]
    ) -> None:
        stager = FileStager({"input": "input"})
        entry = make_entries([bids_tree["active"]], {"input": None})[0]
        staged = stager.stage([entry])[0]

        assert staged.params["input"] == bids_tree["active"]

    def test_string_and_path_inputs_are_resolved(
        self, bids_tree: dict[str, Path]
    ) -> None:
        mask = _touch(bids_tree["root"] / "masks" / "brain.nii.gz")
        stager = FileStager({"mask": "input"})
        entry = make_entries([bids_tree["active"]], {"mask": str(mask)})[0]
        staged = stager.stage([entry])[0]

        assert staged.params["mask"] == mask

    def test_none_output_is_rejected(self, bids_tree: dict[str, Path]) -> None:
        stager = FileStager({"output": "output"})
        entry = make_entries([bids_tree["active"]], {"output": None})[0]

        with pytest.raises(FileStagingError, match="cannot be None"):
            stager.stage([entry])

    def test_output_dict_builds_path_under_active_parent(
        self, bids_tree: dict[str, Path]
    ) -> None:
        stager = FileStager({"output": "output"})
        entry = make_entries(
            [bids_tree["active"]],
            {"output": {"root": {"mode": "active"}, "name": "denoised.nii.gz"}},
        )[0]
        staged = stager.stage([entry])[0]

        expected = bids_tree["active"].parent / "denoised.nii.gz"
        assert staged.params["output"] == expected

    def test_existing_output_file_is_allowed_by_default(
        self, bids_tree: dict[str, Path]
    ) -> None:
        existing = _touch(bids_tree["active"].parent / "denoised.nii.gz")
        stager = FileStager({"output": "output"})
        entry = make_entries([bids_tree["active"]], {"output": str(existing)})[0]
        staged = stager.stage([entry])[0]

        assert staged.params["output"] == existing

    def test_allow_overwrite_false_blocks_existing_output(
        self, bids_tree: dict[str, Path]
    ) -> None:
        existing = _touch(bids_tree["active"].parent / "denoised.nii.gz")
        stager = FileStager({"output": "output"}, allow_overwrite=False)
        entry = make_entries([bids_tree["active"]], {"output": str(existing)})[0]

        with pytest.raises(FileStagingError, match="already exists"):
            stager.stage([entry])

    def test_output_cannot_point_to_existing_directory(
        self, bids_tree: dict[str, Path]
    ) -> None:
        out_dir = bids_tree["active"].parent / "denoised.nii.gz"
        out_dir.mkdir()
        stager = FileStager({"output": "output"}, allow_overwrite=True)
        entry = make_entries([bids_tree["active"]], {"output": str(out_dir)})[0]

        with pytest.raises(FileStagingError, match="existing directory"):
            stager.stage([entry])

    def test_ensure_inputs_exist_false_allows_missing_direct_input(
        self, bids_tree: dict[str, Path], tmp_path: Path
    ) -> None:
        missing = tmp_path / "missing_mask.nii.gz"
        stager = FileStager({"mask": "input"}, ensure_inputs_exist=False)
        entry = make_entries([bids_tree["active"]], {"mask": str(missing)})[0]
        staged = stager.stage([entry])[0]

        assert staged.params["mask"] == missing.resolve()


# ---------------------------------------------------------------------------
# Root resolution
# ---------------------------------------------------------------------------


class TestRootResolution:
    def test_active_root_is_parent_of_active_file(
        self, bids_tree: dict[str, Path]
    ) -> None:
        stager = FileStager({})
        ctx = StageContext(active=bids_tree["active"])
        root = stager.get_root({"mode": "active"}, ctx=ctx, must_exist=True)

        assert root == bids_tree["active"].parent

    def test_path_root_resolves_explicit_directory(
        self, bids_tree: dict[str, Path]
    ) -> None:
        stager = FileStager({})
        ctx = StageContext(active=bids_tree["active"])
        root = stager.get_root(
            {"mode": "path", "value": bids_tree["derivative"]},
            ctx=ctx,
            must_exist=True,
        )

        assert root == bids_tree["derivative"]

    def test_parent_up_moves_from_active_parent(
        self, bids_tree: dict[str, Path]
    ) -> None:
        stager = FileStager({})
        ctx = StageContext(active=bids_tree["active"])
        root = stager.get_root(
            {"mode": "parent_up", "value": 2}, ctx=ctx, must_exist=True
        )

        assert root == bids_tree["active"].parents[2]

    def test_parent_match_finds_nearest_matching_ancestor(
        self, bids_tree: dict[str, Path]
    ) -> None:
        stager = FileStager({})
        ctx = StageContext(active=bids_tree["active"])
        root = stager.get_root(
            {"mode": "parent_match", "value": "sub-*/ses-*"},
            ctx=ctx,
            must_exist=True,
        )

        assert root == bids_tree["active"].parent.parent

    def test_parent_match_most_global_selects_dataset_root(
        self, bids_tree: dict[str, Path]
    ) -> None:
        stager = FileStager({})
        ctx = StageContext(active=bids_tree["active"])
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
        stager = FileStager({})
        ctx = StageContext(active=bids_tree["active"])
        mirrored = stager.get_root(
            {
                "mode": "active",
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

    def test_get_roots_accepts_list_of_root_specs(
        self, bids_tree: dict[str, Path]
    ) -> None:
        stager = FileStager({})
        ctx = StageContext(active=bids_tree["active"])
        roots = stager.get_roots(
            [
                {"mode": "active"},
                {"mode": "path", "value": bids_tree["derivative"]},
            ],
            ctx=ctx,
            must_exist=True,
        )

        assert roots == [bids_tree["active"].parent, bids_tree["derivative"]]

    def test_rejects_active_mode_with_value(self, bids_tree: dict[str, Path]) -> None:
        stager = FileStager({})
        ctx = StageContext(active=bids_tree["active"])

        with pytest.raises(ValueError, match="must not define a value"):
            stager.get_root(
                {"mode": "active", "value": "/tmp"}, ctx=ctx, must_exist=True
            )

    def test_rejects_empty_root_list(self, bids_tree: dict[str, Path]) -> None:
        stager = FileStager({})
        ctx = StageContext(active=bids_tree["active"])

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
                    "root": {"mode": "active"},
                    "search": {"pattern": "*.nii.gz"},
                    "resolve_results": "first",
                }
            },
        )[0]
        staged = stager.stage([entry])[0]

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
                    "search": {"pattern": "*.nii.gz"},
                    "resolve_results": "all",
                }
            },
        )[0]
        staged = stager.stage([entry])[0]

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
                    "search": {"pattern": "*.nii.gz"},
                    "resolve_results": "single",
                }
            },
        )[0]
        staged = stager.stage([entry])[0]

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
                    "search": {"pattern": "*.nii.gz"},
                    "resolve_results": "single",
                }
            },
        )[0]

        with pytest.raises(FileStagingError, match="single file"):
            stager.stage([entry])

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
                        {"mode": "active"},
                        {
                            "mode": "path",
                            "value": root_b,
                        },
                    ],
                    "search": {"pattern": "*.nii.gz"},
                    "resolve_results": "all",
                }
            },
        )[0]
        staged = stager.stage([entry])[0]

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
                    "search": {"pattern": "*.nii.gz"},
                }
            },
        )[0]

        with pytest.raises(FileStagingError, match="Directory does not exist"):
            stager.stage([entry])

    def test_reuses_cached_explorer_for_identical_search_specs(
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
        search = {"pattern": "*.nii.gz"}
        ctx = StageContext(active=bids_tree["active"])
        spec = {"root": {"mode": "active"}, "search": search}

        stager.get_input_file(spec, ctx=ctx)
        stager.get_input_file(spec, ctx=ctx)

        assert len(calls) == 1


# ---------------------------------------------------------------------------
# Dynamic references
# ---------------------------------------------------------------------------


class TestDynamicReferences:
    def test_active_stem_is_available_before_input_resolution(
        self, bids_tree: dict[str, Path]
    ) -> None:
        stager = FileStager({"output": "output"})
        entry = make_entries(
            [bids_tree["active"]],
            {
                "output": {
                    "root": {"mode": "active"},
                    "name": "{active.stem}_denoised.nii.gz",
                }
            },
        )[0]
        staged = stager.stage([entry])[0]

        expected = (
            bids_tree["active"].parent
            / f"{_file_stem(bids_tree['active'])}_denoised.nii.gz"
        )
        assert staged.params["output"] == expected

    def test_bare_string_output_is_resolved_relative_to_cwd(
        self, bids_tree: dict[str, Path], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(bids_tree["root"])
        stager = FileStager({"output": "output"})
        entry = make_entries(
            [bids_tree["active"]],
            {"output": "outputs/{active.stem}.nii.gz"},
        )[0]
        staged = stager.stage([entry])[0]

        assert (
            staged.params["output"]
            == (
                bids_tree["root"] / f"outputs/{_file_stem(bids_tree['active'])}.nii.gz"
            ).resolve()
        )

    def test_output_name_can_reference_resolved_input_path(
        self, bids_tree: dict[str, Path]
    ) -> None:
        mask = _touch(bids_tree["active"].parent / "brain_mask.nii.gz")
        stager = FileStager({"mask": "input", "output": "output"})
        entry = make_entries(
            [bids_tree["active"]],
            {
                "mask": str(mask),
                "output": {
                    "root": {"mode": "active"},
                    "name": "{params.mask.stem}_applied.nii.gz",
                },
            },
        )[0]
        staged = stager.stage([entry])[0]

        assert (
            staged.params["output"]
            == bids_tree["active"].parent / "brain_mask_applied.nii.gz"
        )

    def test_nested_params_path_with_list_index(
        self, bids_tree: dict[str, Path]
    ) -> None:
        stager = FileStager({"output": "output"})
        entry = make_entries(
            [bids_tree["active"]],
            {
                "runs": [{"label": "rest"}],
                "output": {
                    "root": {"mode": "active"},
                    "name": "{params.runs.[0].label}_out.nii.gz",
                },
            },
        )[0]
        staged = stager.stage([entry])[0]

        assert staged.params["output"] == bids_tree["active"].parent / "rest_out.nii.gz"

    def test_missing_params_reference_raises_during_staging(
        self, bids_tree: dict[str, Path]
    ) -> None:
        stager = FileStager({"output": "output"})
        entry = make_entries(
            [bids_tree["active"]],
            {
                "output": {
                    "root": {"mode": "active"},
                    "name": "{params.missing}.nii.gz",
                }
            },
        )[0]

        with pytest.raises(FileStagingError, match="output pointer 'output'"):
            stager.stage([entry])

    def test_unrecognized_reference_prefix_raises(
        self, bids_tree: dict[str, Path]
    ) -> None:
        stager = FileStager({"output": "output"})
        entry = make_entries(
            [bids_tree["active"]],
            {"output": "{unknown.path}/out.nii.gz"},
        )[0]

        with pytest.raises(FileStagingError, match="Unknown dynamic reference"):
            stager.stage([entry])

    def test_interpolated_none_reference_is_rejected(
        self, bids_tree: dict[str, Path]
    ) -> None:
        stager = FileStager({"output": "output"})
        entry = make_entries(
            [bids_tree["active"]],
            {
                "mask": None,
                "output": "prefix_{params.mask.name}_suffix.nii.gz",
            },
        )[0]

        with pytest.raises(FileStagingError, match="resolved to None"):
            stager.stage([entry])

    def test_output_name_can_strip_active_stem_suffix(
        self, bids_tree: dict[str, Path]
    ) -> None:
        active = _touch(bids_tree["root"] / "sub-01" / "anat" / "sub-01_T1w_bet.nii.gz")
        stager = FileStager({"output": "output"})
        entry = make_entries(
            [active],
            {
                "output": {
                    "root": {"mode": "active"},
                    "name": "{active.stem|strip:_bet}_denoised.nii.gz",
                }
            },
        )[0]
        staged = stager.stage([entry])[0]

        assert staged.params["output"] == active.parent / "sub-01_T1w_denoised.nii.gz"

    def test_output_name_can_strip_resolved_input_stem(
        self, bids_tree: dict[str, Path]
    ) -> None:
        mask = _touch(bids_tree["active"].parent / "brain_mask_bet.nii.gz")
        stager = FileStager({"mask": "input", "output": "output"})
        entry = make_entries(
            [bids_tree["active"]],
            {
                "mask": str(mask),
                "output": {
                    "root": {"mode": "active"},
                    "name": "{params.mask.stem|strip:_bet}_applied.nii.gz",
                },
            },
        )[0]
        staged = stager.stage([entry])[0]

        assert (
            staged.params["output"]
            == bids_tree["active"].parent / "brain_mask_applied.nii.gz"
        )

    def test_bare_string_output_can_strip_active_stem(
        self, bids_tree: dict[str, Path], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        active = _touch(bids_tree["root"] / "sub-01" / "anat" / "sub-01_T1w_bet.nii.gz")
        monkeypatch.chdir(bids_tree["root"])
        stager = FileStager({"output": "output"})
        entry = make_entries(
            [active],
            {"output": "outputs/{active.stem|strip:_bet}_denoised.nii.gz"},
        )[0]
        staged = stager.stage([entry])[0]

        assert (
            staged.params["output"]
            == (bids_tree["root"] / "outputs/sub-01_T1w_denoised.nii.gz").resolve()
        )

    def test_invalid_strip_modifier_raises_during_staging(
        self, bids_tree: dict[str, Path]
    ) -> None:
        stager = FileStager({"output": "output"})
        entry = make_entries(
            [bids_tree["active"]],
            {
                "output": {
                    "root": {"mode": "active"},
                    "name": "{active.stem|lower}.nii.gz",
                }
            },
        )[0]

        with pytest.raises(
            FileStagingError, match="Unsupported dynamic-reference modifier"
        ):
            stager.stage([entry])


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
        staged = stager.stage(entries)

        assert staged[0].params["output"] != staged[1].params["output"]
        assert staged[0].params is not shared
        assert staged[1].params is not shared

    def test_active_file_is_never_modified(self, bids_tree: dict[str, Path]) -> None:
        stager = FileStager({"output": "output"})
        entry = make_entries(
            [bids_tree["active"]],
            {"output": "{active.stem}_out.nii.gz"},
        )[0]
        staged = stager.stage([entry])[0]

        assert staged.active == bids_tree["active"]

    def test_entries_with_existing_errors_are_skipped(
        self, bids_tree: dict[str, Path]
    ) -> None:
        prior = StagingErrorRecord(
            active=bids_tree["active"],
            message="previous failure",
        )
        entry = StagedEntry(
            active=bids_tree["active"],
            params={"output": "out.nii.gz"},
            errors=(prior,),
        )
        stager = FileStager({"output": "output"})
        staged = stager.stage([entry])[0]

        assert staged.errors == (prior,)
        assert staged.params == entry.params

    def test_allow_failed_entries_records_error_and_continues(
        self, bids_tree: dict[str, Path]
    ) -> None:
        good = _touch(bids_tree["root"] / "sub-02" / "func" / "good.nii.gz")
        bad = _touch(bids_tree["root"] / "sub-03" / "func" / "bad.nii.gz")
        existing = _touch(good.parent / "blocked.nii.gz")

        entries = make_entries(
            [good, bad],
            [
                {"output": str(existing)},
                {
                    "output": {
                        "root": {"mode": "active"},
                        "name": "{active.stem}_ok.nii.gz",
                    }
                },
            ],
        )
        stager = FileStager(
            {"output": "output"}, allow_failed_entries=True, allow_overwrite=False
        )
        staged = stager.stage(entries)

        assert len(staged) == 2
        assert staged[0].errors
        assert staged[0].errors[0].error_type == "FileStagingError"
        assert staged[0].errors[0].entry_index == 0
        assert staged[1].params["output"] == bad.parent / f"{_file_stem(bad)}_ok.nii.gz"
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
                    "root": {"mode": "active"},
                    "search": {"pattern": "*.nii.gz"},
                }
            },
        )[0]

        with pytest.raises(FileStagingError, match="input pointer 'mask'"):
            stager.stage([entry])

    def test_invalid_active_file_surfaces_as_staging_error(
        self, tmp_path: Path
    ) -> None:
        missing = tmp_path / "missing.nii.gz"
        stager = FileStager({"input": "input"})
        entry = StagedEntry(active=missing, params={"input": None})

        with pytest.raises(FileStagingError, match="does not exist"):
            stager.stage_single(entry)


# ---------------------------------------------------------------------------
# Spec validation
# ---------------------------------------------------------------------------


class TestSpecValidation:
    def test_input_spec_requires_search_mapping(
        self, bids_tree: dict[str, Path]
    ) -> None:
        stager = FileStager({"mask": "input"})
        ctx = StageContext(active=bids_tree["active"])

        with pytest.raises(ValueError, match="missing required key"):
            stager.get_input_file({"root": {"mode": "active"}}, ctx=ctx)

    def test_output_root_cannot_be_a_list(self, bids_tree: dict[str, Path]) -> None:
        stager = FileStager({"output": "output"})
        ctx = StageContext(active=bids_tree["active"])

        with pytest.raises(TypeError, match="cannot be a list"):
            stager.get_output_path(
                {"root": [{"mode": "active"}], "name": "out.nii.gz"},
                ctx=ctx,
            )

    def test_unknown_root_mode_is_rejected(self, bids_tree: dict[str, Path]) -> None:
        stager = FileStager({})
        ctx = StageContext(active=bids_tree["active"])

        with pytest.raises(ValueError, match="Unknown root mode"):
            stager.get_root({"mode": "unknown"}, ctx=ctx, must_exist=False)  # type: ignore[typeddict-item]

    def test_unsupported_spec_keys_are_rejected(
        self, bids_tree: dict[str, Path]
    ) -> None:
        stager = FileStager({"mask": "input"})
        ctx = StageContext(active=bids_tree["active"])

        with pytest.raises(ValueError, match="unsupported key"):
            stager.get_input_file(
                {"search": {"pattern": "*.nii.gz"}, "extra": True},
                ctx=ctx,
            )


# ---------------------------------------------------------------------------
# create_stager
# ---------------------------------------------------------------------------


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

        assert stager.allow_overwrite is True

    def test_created_stager_can_stage_entries(self, bids_tree: dict[str, Path]) -> None:
        stager = create_stager(
            "FileStager",
            {"pointers": {"input": "input"}},
        )
        entry = make_entries([bids_tree["active"]], {"input": None})[0]

        staged = stager.stage([entry])[0]

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
