"""Tests for array and ANTs image similarity metrics."""

from __future__ import annotations

import numpy as np
import pytest

from niiflow.preproc.functional.array.metrics import (
    dice_similarity_coefficient as array_dice,
)
from niiflow.preproc.functional.image.metrics import (
    ants_similarity_metrics,
    dice_similarity_coefficient as image_dice,
)


class TestDiceSimilarityCoefficient:
    def test_identical_masks_score_one(self) -> None:
        mask = np.array([[0, 1], [1, 0]], dtype=np.uint8)
        assert array_dice(mask, mask.copy()) == 1.0

    def test_partial_overlap(self) -> None:
        image = np.array([[1, 1], [0, 0]], dtype=np.int32)
        target = np.array([[1, 0], [0, 0]], dtype=np.int32)
        assert array_dice(image, target) == pytest.approx(2.0 / 3.0)

    def test_disjoint_masks_score_zero(self) -> None:
        image = np.array([1, 0, 0], dtype=np.uint8)
        target = np.array([0, 1, 0], dtype=np.uint8)
        assert array_dice(image, target) == 0.0

    def test_both_empty_score_one(self) -> None:
        image = np.zeros((2, 2), dtype=np.uint8)
        target = np.zeros((2, 2), dtype=bool)
        assert array_dice(image, target) == 1.0

    def test_one_empty_scores_zero(self) -> None:
        image = np.array([1, 0], dtype=bool)
        target = np.array([False, False])
        assert array_dice(image, target) == 0.0

    def test_accepts_boolean_masks(self) -> None:
        image = np.array([True, True, False])
        target = np.array([True, False, False])
        assert array_dice(image, target) == pytest.approx(2.0 / 3.0)

    def test_does_not_mutate_inputs(self) -> None:
        image = np.array([[0, 1], [1, 1]], dtype=np.uint8)
        target = np.array([[1, 1], [0, 0]], dtype=np.uint8)
        original_image = image.copy()
        original_target = target.copy()
        array_dice(image, target)
        np.testing.assert_array_equal(image, original_image)
        np.testing.assert_array_equal(target, original_target)

    def test_accepts_exact_float_binary_masks(self) -> None:
        image = np.array([0.0, 1.0, 1.0], dtype=np.float64)
        target = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        assert array_dice(image, target) == pytest.approx(2.0 / 3.0)

    def test_rejects_non_binary_float(self) -> None:
        image = np.array([0.0, 0.5], dtype=np.float64)
        target = np.array([0.0, 1.0], dtype=np.float64)
        with pytest.raises(ValueError, match="only 0s and 1s"):
            array_dice(image, target)

    def test_rejects_values_other_than_zero_and_one(self) -> None:
        image = np.array([0, 1, 2], dtype=np.int32)
        target = np.array([0, 1, 1], dtype=np.int32)
        with pytest.raises(ValueError, match="only 0s and 1s"):
            array_dice(image, target)

    def test_rejects_shape_mismatch(self) -> None:
        image = np.ones((2, 2), dtype=np.uint8)
        target = np.ones((2,), dtype=np.uint8)
        with pytest.raises(ValueError, match="shape"):
            array_dice(image, target)

    def test_rejects_non_array(self) -> None:
        with pytest.raises(ValueError, match="numpy array"):
            array_dice([0, 1], np.array([0, 1], dtype=np.uint8))  # type: ignore[arg-type]


@pytest.fixture
def ants_mod():
    return pytest.importorskip("ants")


@pytest.fixture
def ants_image(ants_mod):
    return ants_mod.from_numpy(np.zeros((4, 5), dtype=np.float64))


class TestImageDiceSimilarityCoefficient:
    def test_scores_integer_masks(self, ants_mod) -> None:
        image = ants_mod.from_numpy(np.array([[1, 1], [0, 0]], dtype=np.uint8))
        target = ants_mod.from_numpy(np.array([[1, 0], [0, 0]], dtype=np.uint8))
        assert image_dice(image, target) == pytest.approx(2.0 / 3.0)

    def test_scores_exact_float_binary_masks(self, ants_mod) -> None:
        image = ants_mod.from_numpy(np.array([[1.0, 1.0], [0.0, 0.0]]))
        target = ants_mod.from_numpy(np.array([[1.0, 0.0], [0.0, 0.0]]))
        assert image_dice(image, target) == pytest.approx(2.0 / 3.0)

    def test_rejects_non_binary_float_mask(self, ants_mod) -> None:
        image = ants_mod.from_numpy(np.array([[0.0, 0.5], [1.0, 0.0]]))
        target = ants_mod.from_numpy(np.array([[0.0, 1.0], [1.0, 0.0]]))
        with pytest.raises(ValueError, match="only 0s and 1s"):
            image_dice(image, target)

    def test_rejects_multi_label_integer_mask(self, ants_mod) -> None:
        image = ants_mod.from_numpy(np.array([[0, 1], [2, 1]], dtype=np.uint8))
        target = ants_mod.from_numpy(np.array([[0, 1], [1, 1]], dtype=np.uint8))
        with pytest.raises(ValueError, match="only 0s and 1s"):
            image_dice(image, target)

    def test_rejects_physical_space_mismatch(
        self, ants_mod, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        image = ants_mod.from_numpy(np.zeros((2, 2), dtype=np.uint8))
        target = ants_mod.from_numpy(np.zeros((2, 2), dtype=np.uint8))
        monkeypatch.setattr(
            "niiflow.preproc.functional.image.metrics.image_physical_space_consistency",
            lambda *_a, **_k: False,
        )
        with pytest.raises(ValueError, match="same physical space"):
            image_dice(image, target)

    def test_rejects_shape_mismatch(self, ants_mod) -> None:
        image = ants_mod.from_numpy(np.zeros((2, 2), dtype=np.uint8))
        target = ants_mod.from_numpy(np.zeros((3, 3), dtype=np.uint8))
        with pytest.raises(ValueError, match="shape"):
            image_dice(image, target)


class TestAntsSimilarityMetrics:
    @pytest.fixture
    def mask(self, ants_mod, ants_image):
        return ants_mod.from_numpy(np.ones(ants_image.shape, dtype=np.float64))

    def test_negates_ants_cost_and_passes_metric_through(
        self, ants_image, mask, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        target = ants_image.clone()
        calls: list[dict] = []

        def fake_similarity(fixed, moving, **kwargs):
            calls.append({"fixed": fixed, "moving": moving, **kwargs})
            return -1.5

        monkeypatch.setattr(
            "niiflow.preproc.functional.image.metrics.image_similarity",
            fake_similarity,
        )
        score = ants_similarity_metrics(
            ants_image,
            target,
            "MattesMutualInformation",
            mask,
            sampling_strategy="regular",
            sampling_percentage=0.5,
        )
        assert score == pytest.approx(1.5)
        assert calls == [
            {
                "fixed": target,
                "moving": ants_image,
                "metric_type": "MattesMutualInformation",
                "fixed_mask": mask,
                "moving_mask": mask,
                "sampling_strategy": "regular",
                "sampling_percentage": 0.5,
            }
        ]

    def test_forwards_any_metric_name(
        self, ants_image, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        target = ants_image.clone()
        calls: list[dict] = []

        def fake_similarity(fixed, moving, **kwargs):
            calls.append(kwargs)
            return -0.25

        monkeypatch.setattr(
            "niiflow.preproc.functional.image.metrics.image_similarity",
            fake_similarity,
        )
        score = ants_similarity_metrics(ants_image, target, "MeanSquares")
        assert score == pytest.approx(0.25)
        assert calls == [
            {
                "metric_type": "MeanSquares",
                "fixed_mask": None,
                "moving_mask": None,
            }
        ]

    def test_rejects_non_string_and_empty_metric(self, ants_image, mask) -> None:
        target = ants_image.clone()
        with pytest.raises(TypeError, match="single ANTs metric name"):
            ants_similarity_metrics(
                ants_image,
                target,
                ("Correlation",),  # type: ignore[arg-type]
                mask,
            )
        with pytest.raises(ValueError, match="non-empty"):
            ants_similarity_metrics(ants_image, target, "", mask)

    def test_rejects_reserved_kwargs(self, ants_image, mask) -> None:
        target = ants_image.clone()
        with pytest.raises(TypeError, match="does not accept .* via `kwargs`"):
            ants_similarity_metrics(
                ants_image,
                target,
                "Correlation",
                mask,
                metric_type="MeanSquares",
            )

    def test_rejects_physical_space_mismatch(
        self, ants_mod, ants_image, mask, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        target = ants_mod.from_numpy(np.zeros(ants_image.shape, dtype=np.float64))
        monkeypatch.setattr(
            "niiflow.preproc.functional.image.metrics.image_physical_space_consistency",
            lambda *_a, **_k: False,
        )
        with pytest.raises(ValueError, match="same physical space"):
            ants_similarity_metrics(ants_image, target, "Correlation", mask)

    def test_rejects_shape_mismatch(self, ants_mod, ants_image, mask) -> None:
        target = ants_mod.from_numpy(np.zeros((3, 3), dtype=np.float64))
        with pytest.raises(ValueError, match="shape"):
            ants_similarity_metrics(ants_image, target, "Correlation", mask)
