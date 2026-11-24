"""Tests for metric pipeline stages."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from niiflow.preproc.pipelines.pipeline_stages import (
    ANTsSimilarityMetric,
    DiceSimilarityCoefficient,
)
from niiflow.preproc.utils.file import ants_image_write
from test_preproc_pipeline_stages_base import step_ctx


@pytest.fixture
def ants_mod():
    return pytest.importorskip("ants")


def _write_mask(ants_mod, path: Path, array: np.ndarray) -> Path:
    image = ants_mod.from_numpy(array)
    return ants_image_write(image, path)


class TestANTsSimilarityMetric:
    def test_scores_one_metric_and_saves(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured: dict[str, object] = {}

        def fake_metrics(image, target, metric, mask=None, **kwargs):
            captured["metric"] = metric
            captured["mask"] = mask
            captured["sampling_strategy"] = kwargs.get("sampling_strategy")
            return 0.75

        monkeypatch.setattr(
            "niiflow.preproc.pipelines.pipeline_stages.metrics.ants_similarity_metrics",
            fake_metrics,
        )
        value_path = tmp_path / "value.txt"
        report_path = tmp_path / "report.json"
        stage = ANTsSimilarityMetric(
            params={
                "image": object(),
                "target": object(),
                "metric": "MattesMutualInformation",
                "sampling_strategy": "regular",
            },
            save_outputs={
                "value": str(value_path),
                "report": str(report_path),
            },
        )
        # Skip disk loads: the objects above are not paths or ANTs images.
        stage.load_param = lambda key, value: value  # type: ignore[method-assign]
        out = stage.run(step_ctx("similarity")).outputs["similarity"]
        assert out["value"] == 0.75
        assert out["report"] == {
            "metric": "ANTsSimilarityMetric",
            "value": 0.75,
        }
        assert "check" not in out["report"]
        assert "passed" not in out["report"]
        assert captured["metric"] == "MattesMutualInformation"
        assert captured["mask"] is None
        assert captured["sampling_strategy"] == "regular"
        assert value_path.read_text(encoding="utf-8") == "0.75\n"
        assert json.loads(report_path.read_text(encoding="utf-8")) == out["report"]

    def test_construction_requires_metric(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="metric"):
            ANTsSimilarityMetric(
                params={"image": "image.nii.gz", "target": "target.nii.gz"},
                save_outputs={},
            )

    def test_construction_rejects_empty_metric(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            ANTsSimilarityMetric(
                params={
                    "image": "image.nii.gz",
                    "target": "target.nii.gz",
                    "metric": "",
                },
                save_outputs={},
            )

    def test_construction_rejects_non_string_metric(self) -> None:
        with pytest.raises(TypeError, match="metric"):
            ANTsSimilarityMetric(
                params={
                    "image": "image.nii.gz",
                    "target": "target.nii.gz",
                    "metric": ["Correlation"],
                },
                save_outputs={},
            )

    def test_load_param_accepts_image_mask_and_extra_kwargs(
        self, tmp_path: Path, ants_mod
    ) -> None:
        image = ants_mod.from_numpy(np.zeros((2, 2), dtype=np.float32))
        mask_path = _write_mask(
            ants_mod, tmp_path / "mask.nii.gz", np.ones((2, 2), dtype=np.float32)
        )
        stage = ANTsSimilarityMetric(
            params={
                "image": image,
                "target": image,
                "metric": "Correlation",
                "mask": None,
            },
            save_outputs={},
        )
        assert stage.load_param("image", image) is image
        assert stage.load_param("mask", None) is None
        loaded_mask = stage.load_param("mask", mask_path)
        assert loaded_mask.shape == image.shape
        assert stage.load_param("metric", "Correlation") == "Correlation"
        assert stage.load_param("sampling_percentage", 1.0) == 1.0

    def test_load_param_rejects_missing_file(self, tmp_path: Path) -> None:
        stage = ANTsSimilarityMetric(
            params={
                "image": "image.nii.gz",
                "target": "target.nii.gz",
                "metric": "Correlation",
            },
            save_outputs={},
        )
        missing = tmp_path / "missing.nii.gz"
        with pytest.raises(ValueError, match="Failed to read image"):
            stage.load_param("image", missing)

    def test_save_output_rejects_unknown_key(self, tmp_path: Path) -> None:
        stage = ANTsSimilarityMetric(
            params={
                "image": "image.nii.gz",
                "target": "target.nii.gz",
                "metric": "Correlation",
            },
            save_outputs={},
        )
        with pytest.raises(KeyError, match="Unknown output key"):
            stage.save_output("score", 1.0, tmp_path / "score.txt")

    def test_save_output_rejects_non_txt_value(self, tmp_path: Path) -> None:
        stage = ANTsSimilarityMetric(
            params={
                "image": "image.nii.gz",
                "target": "target.nii.gz",
                "metric": "Correlation",
            },
            save_outputs={},
        )
        with pytest.raises(ValueError, match=r"\.txt"):
            stage.save_output("value", 1.0, tmp_path / "value.json")

    def test_save_output_writes_report_json(self, tmp_path: Path) -> None:
        stage = ANTsSimilarityMetric(
            params={
                "image": "image.nii.gz",
                "target": "target.nii.gz",
                "metric": "Correlation",
            },
            save_outputs={},
        )
        report_path = tmp_path / "report.json"
        report = {
            "metric": "ANTsSimilarityMetric",
            "value": 1.0,
        }
        record = stage.save_output("report", report, report_path)
        assert record == report_path.resolve()
        assert json.loads(report_path.read_text(encoding="utf-8")) == report


class TestDiceSimilarityCoefficient:
    def test_scores_loaded_masks_and_saves(self, tmp_path: Path, ants_mod) -> None:
        image_path = _write_mask(
            ants_mod,
            tmp_path / "image.nii.gz",
            np.array([[1.0, 1.0], [0.0, 0.0]]),
        )
        target_path = _write_mask(
            ants_mod,
            tmp_path / "target.nii.gz",
            np.array([[1.0, 0.0], [0.0, 0.0]]),
        )
        value_path = tmp_path / "value.txt"
        report_path = tmp_path / "report.json"
        stage = DiceSimilarityCoefficient(
            params={"image": str(image_path), "target": str(target_path)},
            save_outputs={
                "value": str(value_path),
                "report": str(report_path),
            },
        )
        out = stage.run(step_ctx("dice")).outputs["dice"]
        assert out["value"] == pytest.approx(2.0 / 3.0)
        assert out["report"]["metric"] == "DiceSimilarityCoefficient"
        assert out["report"]["value"] == pytest.approx(2.0 / 3.0)
        assert "check" not in out["report"]
        assert "passed" not in out["report"]
        assert value_path.read_text(encoding="utf-8") == f"{out['value']}\n"
        saved = json.loads(report_path.read_text(encoding="utf-8"))
        assert saved["metric"] == "DiceSimilarityCoefficient"
        assert saved["value"] == pytest.approx(2.0 / 3.0)

    def test_multi_label_masks_are_rejected(self, tmp_path: Path, ants_mod) -> None:
        image_path = _write_mask(
            ants_mod,
            tmp_path / "image.nii.gz",
            np.array([[0.0, 1.0], [2.0, 1.0]]),
        )
        target_path = _write_mask(
            ants_mod,
            tmp_path / "target.nii.gz",
            np.array([[0.0, 1.0], [1.0, 1.0]]),
        )
        stage = DiceSimilarityCoefficient(
            params={"image": str(image_path), "target": str(target_path)},
            save_outputs={},
        )
        with pytest.raises(ValueError, match="only 0s and 1s"):
            stage.run(step_ctx("dice"))

    def test_load_param_passes_unknown_keys_through(self) -> None:
        stage = DiceSimilarityCoefficient(
            params={"image": "image.nii.gz", "target": "target.nii.gz"},
            save_outputs={},
        )
        assert stage.load_param("note", "kept") == "kept"

    def test_save_output_rejects_unknown_key(self, tmp_path: Path) -> None:
        stage = DiceSimilarityCoefficient(
            params={"image": "image.nii.gz", "target": "target.nii.gz"},
            save_outputs={},
        )
        with pytest.raises(KeyError, match="Unknown output key"):
            stage.save_output("score", 1.0, tmp_path / "score.txt")
