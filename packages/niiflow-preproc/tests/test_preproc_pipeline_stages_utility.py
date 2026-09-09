"""Tests for the utility stages in
:mod:`niiflow.preproc.pipelines.pipeline_stages.utility`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from niiflow.preproc.pipelines.pipeline_stages import (
    GetImage,
    Rename,
    RuntimeContext,
    SyncMetadata,
)
from test_preproc_pipeline_stages_base import STEP_ID, step_ctx, touch


class TestRename:
    def test_moves_file_to_new_location(self, tmp_path: Path) -> None:
        src = touch(tmp_path / "src.txt")
        dest = tmp_path / "nested" / "dest.txt"
        ctx = Rename(params={"source": str(src), "dest": str(dest)}).run(step_ctx())
        assert dest.exists()
        assert not src.exists()
        assert ctx.outputs[STEP_ID]["out_path"] == dest.resolve()

    def test_copy_flag_preserves_source(self, tmp_path: Path) -> None:
        src = touch(tmp_path / "src.txt")
        dest = tmp_path / "dest.txt"
        Rename(params={"source": str(src), "dest": str(dest), "copy": True}).run(
            step_ctx()
        )
        assert dest.exists()
        assert src.exists()

    def test_forward_raises_when_source_missing(self, tmp_path: Path) -> None:
        stage = Rename(
            params={
                "source": str(tmp_path / "missing.txt"),
                "dest": str(tmp_path / "dest.txt"),
            }
        )
        with pytest.raises(FileNotFoundError):
            stage.run(step_ctx())

    def test_rejects_legacy_path_param(self, tmp_path: Path) -> None:
        src = touch(tmp_path / "src.txt")
        with pytest.raises(ValueError, match=r"Missing required parameter\(s\)"):
            Rename(params={"path": str(src), "dest": str(tmp_path / "d.txt")})

    def test_check_params_rejects_non_bool_copy(self, tmp_path: Path) -> None:
        src = touch(tmp_path / "src.txt")
        with pytest.raises(TypeError, match="copy"):
            Rename(
                params={
                    "source": str(src),
                    "dest": str(tmp_path / "d.txt"),
                    "copy": "yes",
                }
            )

    def test_save_output_records_new_location_as_json(self, tmp_path: Path) -> None:
        src = touch(tmp_path / "src.txt")
        dest = tmp_path / "dest.txt"
        record_path = tmp_path / "rename.json"
        Rename(
            params={"source": str(src), "dest": str(dest)},
            save_outputs={"out_path": str(record_path)},
        ).run(step_ctx())
        assert json.loads(record_path.read_text(encoding="utf-8")) == {
            "path": str(dest.resolve())
        }

    def test_rejects_legacy_path_save_option(self, tmp_path: Path) -> None:
        src = touch(tmp_path / "src.txt")
        stage = Rename(
            params={"source": str(src), "dest": str(tmp_path / "dest.txt")},
            save_outputs={"path": str(tmp_path / "rename.json")},
        )
        with pytest.raises(ValueError, match="did not return this output"):
            stage.run(step_ctx())


# ---------------------------------------------------------------------------
# GetImage — materialise an image into the context (optionally QC-gated).
# ---------------------------------------------------------------------------


class TestGetImage:
    @staticmethod
    def _image() -> tuple[Any, Any]:
        ants = pytest.importorskip("ants")
        import numpy as np

        return ants, ants.from_numpy(np.zeros((4, 5, 6), dtype="float32"))

    def test_publishes_in_memory_image_to_context(self, tmp_path: Path) -> None:
        _, image = self._image()
        ctx = GetImage(params={"image": image}).run(step_ctx())
        out_image = ctx.outputs[STEP_ID]["out_image"]
        assert isinstance(out_image, type(image))
        assert tuple(out_image.shape) == (4, 5, 6)

    def test_loads_image_from_path(self, tmp_path: Path) -> None:
        ants, image = self._image()
        src = tmp_path / "src.nii.gz"
        ants.image_write(image, str(src))
        ctx = GetImage(params={"image": str(src)}).run(step_ctx())
        assert tuple(ctx.outputs[STEP_ID]["out_image"].shape) == (4, 5, 6)

    def test_saves_via_save_outputs_out_image(self, tmp_path: Path) -> None:
        ants, image = self._image()
        dest = tmp_path / "out" / "saved.nii.gz"
        GetImage(
            params={"image": image},
            save_outputs={"out_image": str(dest)},
        ).run(step_ctx())
        assert dest.is_file()
        ants.image_read(str(dest))  # round-trips as a readable image

    def test_disabled_via_qc_output_skips_save(self, tmp_path: Path) -> None:
        _, image = self._image()
        dest = tmp_path / "out" / "saved.nii.gz"
        ctx = RuntimeContext(step_id="save", outputs={"qc": {"passed": False}})
        out = GetImage(
            params={"image": image, "enable": "ctx.outputs.qc.passed"},
            save_outputs={"out_image": str(dest)},
        ).run(ctx)
        assert not dest.exists()
        assert "save" not in out.outputs
        assert out.steps_completed == []

    def test_enabled_via_qc_output_saves(self, tmp_path: Path) -> None:
        _, image = self._image()
        dest = tmp_path / "out" / "saved.nii.gz"
        ctx = RuntimeContext(step_id="save", outputs={"qc": {"passed": True}})
        out = GetImage(
            params={"image": image, "enable": "ctx.outputs.qc.passed"},
            save_outputs={"out_image": str(dest)},
        ).run(ctx)
        assert dest.is_file()
        assert out.steps_completed == ["save"]

    def test_save_output_rejects_non_nifti_path(self, tmp_path: Path) -> None:
        _, image = self._image()
        stage = GetImage(params={"image": image})
        with pytest.raises(ValueError, match=r"\.nii"):
            stage.save_output("out_image", image, tmp_path / "out.txt")


# ---------------------------------------------------------------------------
# SyncMetadata — copy reference header onto image when grids match.
# ---------------------------------------------------------------------------


class TestSyncMetadata:
    @staticmethod
    def _volume(ants, data, *, origin, spacing, direction=None):
        import numpy as np

        kwargs: dict = {"origin": origin, "spacing": spacing}
        if direction is not None:
            kwargs["direction"] = direction
        return ants.from_numpy(np.asarray(data, dtype="float32"), **kwargs)

    def test_publishes_image_with_reference_metadata(self) -> None:
        ants = pytest.importorskip("ants")
        import numpy as np

        voxels = np.arange(8, dtype="float32").reshape(2, 2, 2)
        image = self._volume(
            ants,
            voxels,
            origin=(1.0 + 1e-8, 2.0, 3.0),
            spacing=(1.0, 1.0, 1.0),
        )
        reference = self._volume(
            ants,
            np.zeros_like(voxels),
            origin=(1.0, 2.0, 3.0),
            spacing=(1.0, 1.0, 1.0),
        )
        ctx = SyncMetadata(params={"image": image, "reference": reference}).run(
            step_ctx()
        )
        out_image = ctx.outputs[STEP_ID]["out_image"]
        np.testing.assert_array_equal(out_image.numpy(), voxels)
        assert out_image.origin == reference.origin
        assert out_image.spacing == reference.spacing

    def test_loads_images_from_paths(self, tmp_path: Path) -> None:
        ants = pytest.importorskip("ants")
        import numpy as np

        voxels = np.ones((3, 3, 3), dtype="float32")
        image = self._volume(
            ants, voxels, origin=(4.0, 5.0, 6.0), spacing=(1.0, 1.0, 1.0)
        )
        reference = self._volume(
            ants, np.zeros_like(voxels), origin=(4.0, 5.0, 6.0), spacing=(1.0, 1.0, 1.0)
        )
        image_path = tmp_path / "image.nii.gz"
        reference_path = tmp_path / "reference.nii.gz"
        ants.image_write(image, str(image_path))
        ants.image_write(reference, str(reference_path))
        ctx = SyncMetadata(
            params={"image": str(image_path), "reference": str(reference_path)}
        ).run(step_ctx())
        np.testing.assert_array_equal(ctx.outputs[STEP_ID]["out_image"].numpy(), voxels)

    def test_saves_via_save_outputs_out_image(self, tmp_path: Path) -> None:
        ants = pytest.importorskip("ants")
        import numpy as np

        image = self._volume(
            ants, np.ones((2, 2, 2)), origin=(0.0, 0.0, 0.0), spacing=(1.0, 1.0, 1.0)
        )
        dest = tmp_path / "out" / "synced.nii.gz"
        SyncMetadata(
            params={"image": image, "reference": image.clone()},
            save_outputs={"out_image": str(dest)},
        ).run(step_ctx())
        assert dest.is_file()
        ants.image_read(str(dest))

    def test_forwards_custom_origin_tolerance(self) -> None:
        ants = pytest.importorskip("ants")
        import numpy as np

        image = self._volume(
            ants, np.ones((2, 2, 2)), origin=(1.1, 2.0, 3.0), spacing=(1.0, 1.0, 1.0)
        )
        reference = self._volume(
            ants, np.zeros((2, 2, 2)), origin=(1.0, 2.0, 3.0), spacing=(1.0, 1.0, 1.0)
        )
        ctx = SyncMetadata(
            params={"image": image, "reference": reference, "origin_atol": 0.2}
        ).run(step_ctx())
        assert ctx.outputs[STEP_ID]["out_image"].origin == reference.origin

    def test_forwards_custom_rtol(self) -> None:
        ants = pytest.importorskip("ants")
        import numpy as np

        image = self._volume(
            ants, np.ones((2, 2, 2)), origin=(1.1, 2.0, 3.0), spacing=(1.0, 1.0, 1.0)
        )
        reference = self._volume(
            ants, np.zeros((2, 2, 2)), origin=(1.0, 2.0, 3.0), spacing=(1.0, 1.0, 1.0)
        )
        ctx = SyncMetadata(
            params={"image": image, "reference": reference, "rtol": 0.15}
        ).run(step_ctx())
        assert ctx.outputs[STEP_ID]["out_image"].origin == reference.origin

    def test_raises_when_grids_differ_beyond_tolerance(self) -> None:
        ants = pytest.importorskip("ants")
        import numpy as np

        image = self._volume(
            ants, np.ones((2, 2, 2)), origin=(5.0, 2.0, 3.0), spacing=(1.0, 1.0, 1.0)
        )
        reference = self._volume(
            ants, np.zeros((2, 2, 2)), origin=(1.0, 2.0, 3.0), spacing=(1.0, 1.0, 1.0)
        )
        stage = SyncMetadata(params={"image": image, "reference": reference})
        with pytest.raises(ValueError, match="origin"):
            stage.run(step_ctx())

    def test_rejects_missing_reference(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match=r"Missing required parameter\(s\)"):
            SyncMetadata(params={"image": str(tmp_path / "image.nii.gz")})

    def test_save_output_rejects_non_nifti_path(self) -> None:
        ants = pytest.importorskip("ants")
        import numpy as np

        image = self._volume(
            ants, np.ones((2, 2, 2)), origin=(0.0, 0.0, 0.0), spacing=(1.0, 1.0, 1.0)
        )
        stage = SyncMetadata(params={"image": image, "reference": image.clone()})
        with pytest.raises(ValueError, match=r"\.nii"):
            stage.save_output("out_image", image, Path("out.txt"))
