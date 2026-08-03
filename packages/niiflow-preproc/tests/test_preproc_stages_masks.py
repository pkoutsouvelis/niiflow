"""Concrete tests for mask pipeline stages."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from niiflow.preproc.pipelines.pipeline_stages import (
    ApplyMask,
    RelabelMask,
    RuntimeContext,
    SmoothMask,
)
from stage_helpers import make_stage, step_ctx


@pytest.fixture
def ants_mod():
    return pytest.importorskip("ants")


def _write_nifti(ants_mod, path: Path, array: np.ndarray) -> Path:
    image = ants_mod.from_numpy(array.astype(np.float64))
    ants_mod.image_write(image, str(path))
    return path


class TestApplyMask:
    def test_applies_mask_and_saves(
        self, tmp_path: Path, ants_mod, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        image_path = _write_nifti(
            ants_mod, tmp_path / "image.nii.gz", np.ones((4, 4, 4))
        )
        mask_path = _write_nifti(ants_mod, tmp_path / "mask.nii.gz", np.ones((4, 4, 4)))
        out_path = tmp_path / "masked.nii.gz"

        stage = ApplyMask(
            params={"image": str(image_path), "mask": str(mask_path)},
            save_outputs={"out_image": str(out_path)},
        )

        def fake_apply_mask(**kwargs):
            return kwargs["image"]

        monkeypatch.setattr(
            "niiflow.preproc.pipelines.pipeline_stages.masks.ants_apply_mask",
            fake_apply_mask,
        )
        ctx = stage.run(step_ctx("apply"))
        assert "out_image" in ctx.outputs["apply"]
        assert out_path.is_file()


class TestSmoothMask:
    def test_smooths_and_saves(self, tmp_path: Path, ants_mod) -> None:
        mask = np.zeros((9, 9, 9), dtype=np.float64)
        mask[3:6, 3:6, 3:6] = 1.0
        mask_path = _write_nifti(ants_mod, tmp_path / "mask.nii.gz", mask)
        out_path = tmp_path / "smooth.nii.gz"

        stage = SmoothMask(
            params={"mask": str(mask_path), "sigma": 1.0},
            save_outputs={"out_mask": str(out_path)},
        )
        ctx = stage.run(step_ctx("smooth"))
        out = ctx.outputs["smooth"]["out_mask"]
        assert float(out.numpy().max()) == pytest.approx(1.0)
        assert out_path.is_file()

    def test_construction_via_helpers(self, tmp_path: Path) -> None:
        stage = make_stage(SmoothMask, tmp_path)
        assert isinstance(stage, SmoothMask)
        assert "mask" in stage.params


class TestRelabelMask:
    def test_relabels_and_saves(self, tmp_path: Path, ants_mod) -> None:
        data = np.array([[[0, 1], [2, 1]]], dtype=np.float64)
        mask_path = _write_nifti(ants_mod, tmp_path / "labels.nii.gz", data)
        out_path = tmp_path / "relabeled.nii.gz"

        stage = RelabelMask(
            params={
                "mask": str(mask_path),
                "mapping": {0: 0, 1: 10, 2: 20},
                "dtype": "float64",
            },
            save_outputs={"out_mask": str(out_path)},
        )
        ctx = stage.run(step_ctx("relabel"))
        out = ctx.outputs["relabel"]["out_mask"].numpy()
        # NIfTI I/O may permute axes; assert label remapping by value counts.
        assert sorted(np.unique(out).tolist()) == [0.0, 10.0, 20.0]
        assert int(np.sum(out == 10.0)) == 2
        assert int(np.sum(out == 20.0)) == 1
        assert int(np.sum(out == 0.0)) == 1
        assert out_path.is_file()

    def test_int_dtype_float_mapping_warns(self, tmp_path: Path, ants_mod) -> None:
        data = np.array([[[0, 1]]], dtype=np.float64)
        mask_path = _write_nifti(ants_mod, tmp_path / "labels.nii.gz", data)
        stage = RelabelMask(
            params={
                "mask": str(mask_path),
                "mapping": {0: 0.0, 1: 1.7},
                "dtype": "int32",
            }
        )
        with pytest.warns(UserWarning, match="make sure this is intentional"):
            stage.run(RuntimeContext(step_id="relabel"))

    def test_check_params_rejects_non_mapping(self, tmp_path: Path) -> None:
        with pytest.raises(TypeError, match="mapping"):
            RelabelMask(params={"mask": str(tmp_path / "m.nii.gz"), "mapping": [1, 2]})
