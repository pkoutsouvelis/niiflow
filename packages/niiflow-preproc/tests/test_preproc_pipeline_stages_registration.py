"""Tests for registration pipeline stages."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from niiflow.preproc.pipelines.pipeline_stages.registration import (
    ANTsApplyTransforms,
    _resolve_transformlist,
    _save_transformlist,
)


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("stub", encoding="utf-8")
    return path


def _save_chain(
    sources: list[Path],
    output_path: Path,
    *,
    inverse: bool = False,
) -> Path:
    key = "invtransforms" if inverse else "fwdtransforms"
    record = _save_transformlist(key, [str(path) for path in sources], output_path)
    return record.path


def _apply_stage(**params: object) -> ANTsApplyTransforms:
    base = {
        "image": "moving.nii.gz",
        "target": "fixed.nii.gz",
    }
    return ANTsApplyTransforms(params={**base, **params}, save_outputs={})


class TestResolveTransformlist:
    def test_single_manifest_matches_one_item_list(self, tmp_path: Path) -> None:
        affine = _touch(tmp_path / "affine.mat")
        warp = _touch(tmp_path / "warp.nii.gz")
        manifest = _save_chain(
            [warp, affine],
            tmp_path / "fwd.json",
        )

        single = _resolve_transformlist(manifest)
        listed = _resolve_transformlist([manifest])

        assert single == listed
        paths, flags = single
        assert paths == [
            str(tmp_path / "fwd_00.nii.gz"),
            str(tmp_path / "fwd_01.mat"),
        ]
        assert flags == [False, False]

    def test_composes_manifests_in_list_order(self, tmp_path: Path) -> None:
        native = tmp_path / "native"
        template = tmp_path / "template"
        native_warp = _touch(native / "warp.nii.gz")
        native_affine = _touch(native / "affine.mat")
        template_affine = _touch(template / "affine.mat")
        template_inv = _touch(template / "inverse.mat")

        native_manifest = _save_chain(
            [native_warp, native_affine],
            native / "fwd.json",
        )
        template_manifest = _save_chain(
            [template_affine, template_inv],
            template / "inv.json",
            inverse=True,
        )

        paths, flags = _resolve_transformlist([template_manifest, native_manifest])

        assert paths == [
            str(template / "inv_00.mat"),
            str(template / "inv_01.mat"),
            str(native / "fwd_00.nii.gz"),
            str(native / "fwd_01.mat"),
        ]
        assert flags == [True, True, False, False]

    def test_mixed_list_leaves_bare_file_inversion_unspecified(
        self, tmp_path: Path
    ) -> None:
        chain_dir = tmp_path / "chain"
        affine = _touch(chain_dir / "affine.mat")
        manifest = _save_chain([affine], chain_dir / "inv.json", inverse=True)
        extra = _touch(tmp_path / "extra.nii.gz")

        paths, flags = _resolve_transformlist([manifest, extra, Path(manifest)])

        assert paths == [
            str(chain_dir / "inv_00.mat"),
            str(extra),
            str(chain_dir / "inv_00.mat"),
        ]
        assert flags == [True, None, True]

    def test_bare_list_returns_unspecified_flags(self, tmp_path: Path) -> None:
        first = _touch(tmp_path / "a.mat")
        second = _touch(tmp_path / "b.nii.gz")

        paths, flags = _resolve_transformlist((first, second))

        assert paths == [str(first), str(second)]
        assert flags == [None, None]

    def test_rejects_single_transform_path(self, tmp_path: Path) -> None:
        transform = _touch(tmp_path / "a.mat")
        with pytest.raises(ValueError, match="single path must be a .json manifest"):
            _resolve_transformlist(transform)

    def test_rejects_empty_list(self) -> None:
        with pytest.raises(ValueError, match="non-empty list"):
            _resolve_transformlist([])

    def test_rejects_missing_transform_in_list(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="Transform file not found"):
            _resolve_transformlist([tmp_path / "missing.mat"])

    def test_rejects_unknown_transform_extension_in_list(self, tmp_path: Path) -> None:
        transform = _touch(tmp_path / "transform.txt")
        with pytest.raises(ValueError, match="Unknown transform extension '.txt'"):
            _resolve_transformlist([transform])

    def test_rejects_unknown_transform_extension_in_manifest(
        self, tmp_path: Path
    ) -> None:
        transform = _touch(tmp_path / "transform.txt")
        manifest = tmp_path / "chain.json"
        manifest.write_text(
            json.dumps(
                {
                    "format": "ants_transform_chain",
                    "transforms": [
                        {
                            "path": transform.name,
                            "invert": False,
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )

        with pytest.raises(ValueError, match="Unknown transform extension '.txt'"):
            _resolve_transformlist(manifest)

    def test_rejects_non_manifest_json_inside_list(self, tmp_path: Path) -> None:
        manifest = tmp_path / "not-a-chain.json"
        manifest.write_text(json.dumps({"format": "other"}), encoding="utf-8")
        with pytest.raises(ValueError, match="not an ANTs transform-chain manifest"):
            _resolve_transformlist([manifest])


class TestANTsApplyTransformsComposition:
    def test_check_params_allows_whichtoinvert_when_list_includes_manifest(
        self, tmp_path: Path
    ) -> None:
        manifest = tmp_path / "fwd.json"
        manifest.write_text("{}", encoding="utf-8")
        transform = _touch(tmp_path / "extra.mat")

        stage = _apply_stage(
            transformlist=[str(manifest), str(transform)],
            whichtoinvert=[False, True],
        )

        assert stage.params["whichtoinvert"] == [False, True]

    def test_check_params_allows_whichtoinvert_for_bare_transforms(
        self, tmp_path: Path
    ) -> None:
        transform = _touch(tmp_path / "extra.mat")
        stage = _apply_stage(
            transformlist=[str(transform)],
            whichtoinvert=[True],
        )
        assert stage.params["whichtoinvert"] == [True]

    def test_forward_uses_composed_inversion_flags(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        affine = _touch(tmp_path / "affine.mat")
        manifest = _save_chain([affine], tmp_path / "inv.json", inverse=True)
        extra = _touch(tmp_path / "extra.mat")
        captured: dict[str, object] = {}

        def fake_apply(**kwargs: object) -> str:
            captured.update(kwargs)
            return "warped"

        monkeypatch.setattr(
            "niiflow.preproc.pipelines.pipeline_stages.registration.ants_apply_transforms",
            fake_apply,
        )

        stage = _apply_stage(transformlist=[str(manifest), str(extra)])
        loaded = stage.load_param("transformlist", [str(manifest), str(extra)])
        result = stage.forward(
            image="moving",
            target="fixed",
            transformlist=loaded,
        )

        assert result == {"out_image": "warped"}
        assert captured["transformlist"] == [
            str(tmp_path / "inv_00.mat"),
            str(extra),
        ]
        assert captured["whichtoinvert"] == [True, None]

    def test_forward_merges_non_conflicting_whichtoinvert(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        affine = _touch(tmp_path / "affine.mat")
        manifest = _save_chain([affine], tmp_path / "inv.json", inverse=True)
        extra = _touch(tmp_path / "extra.mat")
        captured: dict[str, object] = {}

        def fake_apply(**kwargs: object) -> str:
            captured.update(kwargs)
            return "warped"

        monkeypatch.setattr(
            "niiflow.preproc.pipelines.pipeline_stages.registration.ants_apply_transforms",
            fake_apply,
        )

        stage = _apply_stage(transformlist=[manifest, extra])
        loaded = stage.load_param("transformlist", [manifest, extra])
        stage.forward(
            image="moving",
            target="fixed",
            transformlist=loaded,
            whichtoinvert=[True, True],
        )

        assert captured["whichtoinvert"] == [True, True]

    def test_forward_rejects_conflicting_manifest_flag(self, tmp_path: Path) -> None:
        affine = _touch(tmp_path / "affine.mat")
        manifest = _save_chain([affine], tmp_path / "inv.json", inverse=True)
        stage = _apply_stage(transformlist=[manifest])
        loaded = stage.load_param("transformlist", [manifest])

        with pytest.raises(ValueError, match="entry 0 conflicts"):
            stage.forward(
                image="moving",
                target="fixed",
                transformlist=loaded,
                whichtoinvert=[False],
            )

    def test_forward_merges_none_and_bool_for_bare_transforms(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        transform = _touch(tmp_path / "a.mat")
        captured: dict[str, object] = {}

        def fake_apply(**kwargs: object) -> str:
            captured.update(kwargs)
            return "warped"

        monkeypatch.setattr(
            "niiflow.preproc.pipelines.pipeline_stages.registration.ants_apply_transforms",
            fake_apply,
        )

        stage = _apply_stage(
            transformlist=[str(transform)],
            whichtoinvert=[True],
        )
        loaded = stage.load_param("transformlist", [str(transform)])
        stage.forward(
            image="moving",
            target="fixed",
            transformlist=loaded,
            whichtoinvert=[True],
        )

        assert captured["transformlist"] == [str(transform)]
        assert captured["whichtoinvert"] == [True]

    def test_forward_preserves_none_plus_none(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        transform = _touch(tmp_path / "a.mat")
        captured: dict[str, object] = {}

        def fake_apply(**kwargs: object) -> str:
            captured.update(kwargs)
            return "warped"

        monkeypatch.setattr(
            "niiflow.preproc.pipelines.pipeline_stages.registration.ants_apply_transforms",
            fake_apply,
        )

        stage = _apply_stage(transformlist=[transform])
        loaded = stage.load_param("transformlist", [transform])
        stage.forward(
            image="moving",
            target="fixed",
            transformlist=loaded,
            whichtoinvert=[None],
        )

        assert captured["whichtoinvert"] == [None]

    def test_forward_rejects_wrong_whichtoinvert_length(self, tmp_path: Path) -> None:
        transform = _touch(tmp_path / "a.mat")
        stage = _apply_stage(transformlist=[transform])
        loaded = stage.load_param("transformlist", [transform])

        with pytest.raises(ValueError, match="one entry per flattened transform"):
            stage.forward(
                image="moving",
                target="fixed",
                transformlist=loaded,
                whichtoinvert=[],
            )

    def test_forward_rejects_invalid_whichtoinvert_entry(self, tmp_path: Path) -> None:
        transform = _touch(tmp_path / "a.mat")
        stage = _apply_stage(transformlist=[transform])
        loaded = stage.load_param("transformlist", [transform])

        with pytest.raises(TypeError, match="entry 0"):
            stage.forward(
                image="moving",
                target="fixed",
                transformlist=loaded,
                whichtoinvert=[1],
            )
