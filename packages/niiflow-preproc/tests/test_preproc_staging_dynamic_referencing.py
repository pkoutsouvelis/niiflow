"""Tests for dynamic reference resolution.

Helpers (:func:`resolve_dynamic_refs`, modifiers) are covered first. Reference
stagers and FileStager integration follow, since those stagers are meant to be
chained rather than used in isolation.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from niiflow.preproc.staging import (
    DynamicReferenceError,
    FileStager,
    FileStagingError,
    ResolveActiveReferences,
    ResolveParamReferences,
    StagedEntry,
    Stager,
    StagingContext,
    add_reference_staging_bookends,
    get_modifiers,
    make_entries,
    resolve_dynamic_refs,
    strip,
)


def _touch(path: Path, content: str = "nii") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path.resolve()


def _file_stem(path: Path) -> str:
    return path.stem


def _stage(stager: Stager, entries: Sequence[StagedEntry]) -> list[StagedEntry]:
    """Run a stager between active/param reference bookends."""
    staged = list(entries)
    for step in add_reference_staging_bookends([stager]):
        staged = step.stage(staged)
    return staged


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


@dataclass
class _FakeImage:
    """Stand-in for objects with non-path attributes (e.g. ANTsImage)."""

    spacing: tuple[float, float, float]
    orientation: str = "RPI"


def _ctx(active: Path) -> StagingContext:
    return StagingContext(active=active)


def _resolve(
    value: Any,
    *,
    active: Path,
    params: dict[str, Any] | None = None,
    resolve_params: bool = True,
) -> Any:
    return resolve_dynamic_refs(
        value,
        params={} if params is None else params,
        ctx=_ctx(active),
        resolve_params=resolve_params,
    )


class TestActiveReferences:
    def test_active_path_attrs(self, tmp_path: Path) -> None:
        active = tmp_path / "sub-01_T1w.nii.gz"
        active.write_bytes(b"x")
        assert _resolve("{active}", active=active) == active
        assert _resolve("{active.name}", active=active) == active.name
        assert _resolve("{active.stem}", active=active) == active.stem
        assert _resolve("{active.parent}", active=active) == active.parent

    def test_active_rejects_nested_attr_path(self, tmp_path: Path) -> None:
        active = tmp_path / "a.nii.gz"
        active.write_bytes(b"x")
        with pytest.raises(DynamicReferenceError, match=r"active\.?<"):
            _resolve("{active.parent.name}", active=active)


class TestParamsDictWalk:
    def test_nested_dict_and_list_index(self, tmp_path: Path) -> None:
        active = tmp_path / "a.nii.gz"
        active.write_bytes(b"x")
        params = {"runs": [{"label": "rest"}]}
        assert (
            _resolve("{params.runs.[0].label}", active=active, params=params) == "rest"
        )

    def test_bare_params_returns_native_value(self, tmp_path: Path) -> None:
        active = tmp_path / "a.nii.gz"
        active.write_bytes(b"x")
        params = {"count": 3}
        assert _resolve("{params.count}", active=active, params=params) == 3


class TestParamsObjectAttributes:
    def test_getattr_on_custom_object(self, tmp_path: Path) -> None:
        active = tmp_path / "a.nii.gz"
        active.write_bytes(b"x")
        image = _FakeImage(spacing=(1.0, 1.0, 1.2))
        params = {"image": image}
        assert _resolve("{params.image.spacing}", active=active, params=params) == (
            1.0,
            1.0,
            1.2,
        )
        assert (
            _resolve("{params.image.orientation}", active=active, params=params)
            == "RPI"
        )

    def test_getattr_on_sequence_is_allowed(self, tmp_path: Path) -> None:
        active = tmp_path / "a.nii.gz"
        active.write_bytes(b"x")
        params = {"spacing": (1.0, 1.0, 1.2)}
        count = _resolve("{params.spacing.count}", active=active, params=params)
        assert callable(count)
        assert count(1.0) == 2

    def test_pathlib_stem_keeps_inner_suffix(self, tmp_path: Path) -> None:
        active = tmp_path / "a.nii.gz"
        active.write_bytes(b"x")
        mask = tmp_path / "sub-01_T1w_bet.nii.gz"
        params = {"mask": mask}
        # pathlib stem drops only the last suffix (``.gz``).
        assert (
            _resolve("{params.mask.stem}", active=active, params=params)
            == "sub-01_T1w_bet.nii"
        )
        assert (
            _resolve("{params.mask.name|strip:.nii.gz}", active=active, params=params)
            == "sub-01_T1w_bet"
        )

    def test_dict_key_preferred_over_getattr(self, tmp_path: Path) -> None:
        active = tmp_path / "a.nii.gz"
        active.write_bytes(b"x")
        params = {"image": {"spacing": "from-dict"}}
        assert (
            _resolve("{params.image.spacing}", active=active, params=params)
            == "from-dict"
        )

    def test_path_attr_on_non_path_raises(self, tmp_path: Path) -> None:
        active = tmp_path / "a.nii.gz"
        active.write_bytes(b"x")
        params = {"count": 3}
        with pytest.raises(DynamicReferenceError, match="has no attribute"):
            _resolve("{params.count.stem}", active=active, params=params)

    def test_missing_getattr_raises(self, tmp_path: Path) -> None:
        active = tmp_path / "a.nii.gz"
        active.write_bytes(b"x")
        params = {"image": _FakeImage(spacing=(1.0, 1.0, 1.0))}
        with pytest.raises(DynamicReferenceError, match="has no attribute"):
            _resolve("{params.image.missing_attr}", active=active, params=params)

    def test_circular_unresolved_path_attr_raises(self, tmp_path: Path) -> None:
        active = tmp_path / "a.nii.gz"
        active.write_bytes(b"x")
        params = {"left": "{params.right}", "right": "{params.left}"}
        with pytest.raises(DynamicReferenceError, match="unresolved dynamic reference"):
            _resolve("{params.left.stem}", active=active, params=params)

    def test_params_refs_skipped_when_disabled(self, tmp_path: Path) -> None:
        active = tmp_path / "a.nii.gz"
        active.write_bytes(b"x")
        assert (
            _resolve(
                "{params.x}",
                active=active,
                params={"x": 1},
                resolve_params=False,
            )
            == "{params.x}"
        )


class TestModifiers:
    def test_strip_on_str_and_path(self, tmp_path: Path) -> None:
        assert strip("sub-01_bet", arg="_bet", ref="x") == "sub-01"
        path = tmp_path / "sub-01_bet.nii.gz"
        assert strip(path, arg="_bet.nii.gz", ref="x") == str(tmp_path / "sub-01")

    def test_strip_rejects_non_str_path(self) -> None:
        with pytest.raises(ValueError, match="requires str or Path"):
            strip(3, arg="_x", ref="params.count|strip:_x")

    def test_strip_rejects_empty_arg(self) -> None:
        with pytest.raises(ValueError, match="Empty strip suffix"):
            strip("abc", arg="", ref="x|strip:")

    def test_strip_via_resolve(self, tmp_path: Path) -> None:
        active = tmp_path / "sub-01_T1w_bet.nii.gz"
        active.write_bytes(b"x")
        assert (
            _resolve("{active.name|strip:_bet.nii.gz}", active=active) == "sub-01_T1w"
        )

    def test_strip_on_non_path_attr_raises(self, tmp_path: Path) -> None:
        active = tmp_path / "a.nii.gz"
        active.write_bytes(b"x")
        params = {"image": _FakeImage(spacing=(1.0, 1.0, 1.0))}
        with pytest.raises(DynamicReferenceError, match="requires str or Path"):
            _resolve("{params.image.spacing|strip:_x}", active=active, params=params)

    def test_unknown_modifier_raises(self, tmp_path: Path) -> None:
        active = tmp_path / "a.nii.gz"
        active.write_bytes(b"x")
        with pytest.raises(
            DynamicReferenceError, match="Unsupported dynamic-reference modifier"
        ):
            _resolve("{active.stem|lower}", active=active)

    def test_get_modifiers_discovers_registered_callables(self) -> None:
        modifiers = get_modifiers()
        assert modifiers == {"strip": strip}


# ---------------------------------------------------------------------------
# Reference-resolution stagers
# ---------------------------------------------------------------------------


class TestReferenceStagers:
    def test_active_stager_expands_active_leaves_params(self, tmp_path: Path) -> None:
        active = _touch(tmp_path / "sub-01_T1w.nii.gz")
        entry = make_entries(
            [active],
            {"subject": "{active.stem}", "other": "{params.missing}"},
        )[0]
        staged = ResolveActiveReferences().stage([entry])[0]
        stem = active.stem
        assert staged.params["subject"] == stem
        assert staged.params["other"] == "{params.missing}"

    def test_active_stager_allows_missing_active(self, tmp_path: Path) -> None:
        missing = tmp_path / "planned" / "sub-01_T1w.nii.gz"
        entry = make_entries(
            [missing],
            {"subject": "{active.stem}", "dir": "{active.parent}"},
        )[0]
        staged = ResolveActiveReferences().stage([entry])[0]
        assert staged.params["subject"] == "sub-01_T1w.nii"
        assert staged.params["dir"] == missing.parent.resolve()

    def test_params_stager_expands_params_refs(self, tmp_path: Path) -> None:
        active = _touch(tmp_path / "a.nii.gz")
        entry = make_entries(
            [active],
            {"label": "rest", "out": "{params.label}_x"},
        )[0]
        staged = ResolveParamReferences().stage([entry])[0]
        assert staged.params["out"] == "rest_x"

    def test_bookends_around_file_stager(self, tmp_path: Path) -> None:
        active = _touch(tmp_path / "sub-01_T1w.nii.gz")
        stager = FileStager({"output": "output"})
        entry = make_entries(
            [active],
            {
                "note": "{active.stem}",
                "output": {
                    "name": "{active.stem}_out.nii.gz",
                },
            },
        )[0]
        staged = _stage(stager, [entry])[0]
        stem = active.stem
        assert staged.params["note"] == stem
        assert staged.params["output"] == active.parent / f"{stem}_out.nii.gz"

    def test_file_stager_resolves_params_on_input_pointers(
        self, tmp_path: Path
    ) -> None:
        active = _touch(tmp_path / "anat" / "t1.nii.gz")
        mask = _touch(tmp_path / "anat" / "brain_mask.nii.gz")
        stager = FileStager({"mask": "input"})
        entry = make_entries(
            [active],
            {"mask_name": "brain_mask.nii.gz", "mask": "{params.mask_name}"},
        )[0]
        # Active bookend not required here; relative bare path anchors to active.parent.
        staged = stager.stage([entry])[0]
        assert staged.params["mask"] == mask

    def test_params_stager_raises_on_missing(self, tmp_path: Path) -> None:
        active = _touch(tmp_path / "a.nii.gz")
        entry = make_entries([active], {"label": "{params.missing}"})[0]
        with pytest.raises(DynamicReferenceError):
            ResolveParamReferences().stage([entry])

    def test_bookends_are_idempotent(self) -> None:
        middle = [FileStager()]
        once = add_reference_staging_bookends(middle)
        twice = add_reference_staging_bookends(once)
        assert [type(s).__name__ for s in once] == [
            "ResolveActiveReferences",
            "FileStager",
            "ResolveParamReferences",
        ]
        assert [type(s).__name__ for s in twice] == [
            "ResolveActiveReferences",
            "FileStager",
            "ResolveParamReferences",
        ]


# ---------------------------------------------------------------------------
# Dynamic references through FileStager + bookends
# ---------------------------------------------------------------------------


class TestDynamicReferences:

    def test_stages_dynamic_references_without_pointers(
        self, bids_tree: dict[str, Path]
    ) -> None:
        """Dynamic-reference-only staging needs no pointers at all."""
        stager = FileStager()
        entry = make_entries(
            [bids_tree["active"]],
            {"subject": "{active.stem}", "where": "{active.parent}"},
        )[0]
        staged = _stage(stager, [entry])[0]

        active = bids_tree["active"]
        expected_stem = active.stem
        assert staged.params["subject"] == expected_stem
        assert staged.params["where"] == active.parent

    def test_active_stem_is_available_before_input_resolution(
        self, bids_tree: dict[str, Path]
    ) -> None:
        stager = FileStager({"output": "output"})
        entry = make_entries(
            [bids_tree["active"]],
            {
                "output": {
                    "name": "{active.stem}_denoised.nii.gz",
                }
            },
        )[0]
        staged = _stage(stager, [entry])[0]

        expected = (
            bids_tree["active"].parent
            / f"{_file_stem(bids_tree['active'])}_denoised.nii.gz"
        )
        assert staged.params["output"] == expected

    def test_bare_string_output_is_anchored_to_active_parent(
        self, bids_tree: dict[str, Path]
    ) -> None:
        stager = FileStager({"output": "output"})
        entry = make_entries(
            [bids_tree["active"]],
            {"output": "outputs/{active.stem}.nii.gz"},
        )[0]
        staged = _stage(stager, [entry])[0]

        assert (
            staged.params["output"]
            == (
                bids_tree["active"].parent
                / f"outputs/{_file_stem(bids_tree['active'])}.nii.gz"
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
                    "name": "{params.mask.name|strip:.nii.gz}_applied.nii.gz",
                },
            },
        )[0]
        staged = _stage(stager, [entry])[0]

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
                    "name": "{params.runs.[0].label}_out.nii.gz",
                },
            },
        )[0]
        staged = _stage(stager, [entry])[0]

        assert staged.params["output"] == bids_tree["active"].parent / "rest_out.nii.gz"

    def test_missing_params_reference_raises_during_staging(
        self, bids_tree: dict[str, Path]
    ) -> None:
        stager = FileStager({"output": "output"})
        entry = make_entries(
            [bids_tree["active"]],
            {
                "output": {
                    "name": "{params.missing}.nii.gz",
                }
            },
        )[0]

        with pytest.raises(FileStagingError, match="output pointer 'output'"):
            _stage(stager, [entry])

    def test_missing_params_reference_raises_outside_pointers(
        self, bids_tree: dict[str, Path]
    ) -> None:
        # Dynamic refs are expanded for all params, not only pointer specs.
        stager = FileStager()
        entry = make_entries(
            [bids_tree["active"]],
            {"label": "{params.missing}"},
        )[0]

        with pytest.raises(
            DynamicReferenceError, match="Failed to resolve params reference"
        ):
            _stage(stager, [entry])

    def test_path_attribute_on_non_filepath_param_raises(
        self, bids_tree: dict[str, Path]
    ) -> None:
        stager = FileStager({"output": "output"})
        entry = make_entries(
            [bids_tree["active"]],
            {
                "count": 3,
                "output": {
                    "name": "{params.count.stem}_out.nii.gz",
                },
            },
        )[0]

        with pytest.raises(FileStagingError, match="has no attribute"):
            _stage(stager, [entry])

    def test_circular_params_path_attribute_raises(
        self, bids_tree: dict[str, Path]
    ) -> None:
        stager = FileStager({"output": "output"})
        entry = make_entries(
            [bids_tree["active"]],
            {
                "left": "{params.right}",
                "right": "{params.left}",
                "output": {
                    "name": "{params.left.stem}_out.nii.gz",
                },
            },
        )[0]

        with pytest.raises(FileStagingError, match="unresolved dynamic reference"):
            _stage(stager, [entry])

    def test_unrecognized_reference_prefix_raises(
        self, bids_tree: dict[str, Path]
    ) -> None:
        stager = FileStager({"output": "output"})
        entry = make_entries(
            [bids_tree["active"]],
            {"output": "{unknown.path}/out.nii.gz"},
        )[0]

        with pytest.raises(DynamicReferenceError, match="Unknown dynamic reference"):
            _stage(stager, [entry])

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
            _stage(stager, [entry])

    def test_output_name_can_strip_active_stem_suffix(
        self, bids_tree: dict[str, Path]
    ) -> None:
        active = _touch(bids_tree["root"] / "sub-01" / "anat" / "sub-01_T1w_bet.nii.gz")
        stager = FileStager({"output": "output"})
        entry = make_entries(
            [active],
            {
                "output": {
                    "name": "{active.name|strip:_bet.nii.gz}_denoised.nii.gz",
                }
            },
        )[0]
        staged = _stage(stager, [entry])[0]

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
                    "name": "{params.mask.name|strip:_bet.nii.gz}_applied.nii.gz",
                },
            },
        )[0]
        staged = _stage(stager, [entry])[0]

        assert (
            staged.params["output"]
            == bids_tree["active"].parent / "brain_mask_applied.nii.gz"
        )

    def test_bare_string_output_can_strip_active_stem(
        self, bids_tree: dict[str, Path]
    ) -> None:
        active = _touch(bids_tree["root"] / "sub-01" / "anat" / "sub-01_T1w_bet.nii.gz")
        stager = FileStager({"output": "output"})
        entry = make_entries(
            [active],
            {"output": "outputs/{active.name|strip:_bet.nii.gz}_denoised.nii.gz"},
        )[0]
        staged = _stage(stager, [entry])[0]

        assert (
            staged.params["output"]
            == (active.parent / "outputs/sub-01_T1w_denoised.nii.gz").resolve()
        )

    def test_invalid_strip_modifier_raises_during_staging(
        self, bids_tree: dict[str, Path]
    ) -> None:
        stager = FileStager({"output": "output"})
        entry = make_entries(
            [bids_tree["active"]],
            {
                "output": {
                    "name": "{active.stem|lower}.nii.gz",
                }
            },
        )[0]

        with pytest.raises(
            DynamicReferenceError, match="Unsupported dynamic-reference modifier"
        ):
            _stage(stager, [entry])
