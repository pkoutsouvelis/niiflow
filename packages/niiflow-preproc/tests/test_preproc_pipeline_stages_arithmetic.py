"""Concrete tests for arithmetic pipeline stages."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from niiflow.preproc.pipelines.pipeline_stages import PointwiseArithmetic
from test_preproc_pipeline_stages_base import step_ctx
from test_preproc_pipeline_stages_shipped_shared import make_stage


@pytest.fixture
def ants_mod():
    return pytest.importorskip("ants")


def _write_nifti(ants_mod, path: Path, array: np.ndarray) -> Path:
    image = ants_mod.from_numpy(array.astype(np.float64))
    ants_mod.image_write(image, str(path))
    return path


class TestPointwiseArithmetic:
    def test_scalar_ops_and_saves(self, tmp_path: Path, ants_mod) -> None:
        image = ants_mod.from_numpy(np.array([[[1.0, 2.0], [3.0, 4.0]]]))
        out_path = tmp_path / "out.nii.gz"
        stage = PointwiseArithmetic(
            params={
                "image": image,
                "operations": [{"mul": 2.0}, {"sub": 1.0}],
            },
            save_outputs={"out_image": str(out_path)},
        )
        ctx = stage.run(step_ctx("arith"))
        out = ctx.outputs["arith"]["out_image"].numpy()
        np.testing.assert_allclose(out, np.array([[[1.0, 3.0], [5.0, 7.0]]]))
        assert out_path.is_file()

    def test_image_operand(self, ants_mod) -> None:
        field = ants_mod.from_numpy(np.array([[[0.0, 1.0], [2.0, 3.0]]]))
        mask = ants_mod.from_numpy(np.array([[[0.0, 1.0], [0.5, 1.0]]]))
        stage = PointwiseArithmetic(
            params={
                "image": field,
                "operations": [
                    {"sub": 1},
                    {"mul": mask},
                    {"add": 1},
                ],
            }
        )
        ctx = stage.run(step_ctx("blend"))
        np.testing.assert_allclose(
            ctx.outputs["blend"]["out_image"].numpy(),
            np.array([[[1.0, 1.0], [1.5, 3.0]]]),
        )

    def test_loads_operands_from_paths(self, tmp_path: Path, ants_mod) -> None:
        field_path = _write_nifti(
            ants_mod,
            tmp_path / "field.nii.gz",
            np.ones((4, 4, 4)),
        )
        mask_path = _write_nifti(
            ants_mod,
            tmp_path / "mask.nii.gz",
            np.full((4, 4, 4), 0.5),
        )
        stage = PointwiseArithmetic(
            params={
                "image": str(field_path),
                "operations": [{"mul": str(mask_path)}, {"add": 1.0}],
            }
        )
        ctx = stage.run(step_ctx("paths"))
        out = ctx.outputs["paths"]["out_image"].numpy()
        np.testing.assert_allclose(out, np.full(out.shape, 1.5))

    def test_construction_via_helpers(self, tmp_path: Path) -> None:
        stage = make_stage(PointwiseArithmetic, tmp_path)
        assert isinstance(stage, PointwiseArithmetic)
        assert "operations" in stage.params

    def test_construction_rejects_missing_operations(self, tmp_path: Path) -> None:
        with pytest.raises(
            ValueError,
            match=r"Missing required parameter\(s\) for PointwiseArithmetic: "
            r"\['operations'\]",
        ):
            PointwiseArithmetic(
                params={
                    "image": str(tmp_path / "image.nii.gz"),
                }
            )

    def test_check_params_rejects_empty_operations(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="at least one operation"):
            PointwiseArithmetic(
                params={
                    "image": str(tmp_path / "image.nii.gz"),
                    "operations": [],
                }
            )

    def test_check_params_rejects_unsupported_key(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="unsupported operation"):
            PointwiseArithmetic(
                params={
                    "image": str(tmp_path / "image.nii.gz"),
                    "operations": [{"pow": 2.0}],
                }
            )

    def test_check_params_rejects_non_sequence(self, tmp_path: Path) -> None:
        with pytest.raises(TypeError, match="sequence"):
            PointwiseArithmetic(
                params={
                    "image": str(tmp_path / "image.nii.gz"),
                    "operations": {"mul": 2.0},
                }
            )
