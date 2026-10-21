"""Tests for :mod:`niiflow.preproc.functional.array.masks` and image wrappers."""

from __future__ import annotations

import numpy as np
import pytest

from niiflow.preproc.functional.array.masks import (
    compose_masks,
    relabel_mask,
    smooth_mask,
)


@pytest.fixture
def binary_cube() -> np.ndarray:
    mask = np.zeros((9, 9, 9), dtype=np.float64)
    mask[3:6, 3:6, 3:6] = 1.0
    return mask


@pytest.fixture
def ants_mod():
    return pytest.importorskip("ants")


class TestSmoothMask:
    def test_preserves_shape_and_range(self, binary_cube: np.ndarray) -> None:
        original = binary_cube.copy()
        out = smooth_mask(binary_cube, sigma=1.0)
        assert out.shape == binary_cube.shape
        assert out.dtype == np.float64
        assert float(out.min()) >= 0.0
        assert float(out.max()) == pytest.approx(1.0)
        np.testing.assert_array_equal(binary_cube, original)

    def test_threshold_rebinarizes(self, binary_cube: np.ndarray) -> None:
        out = smooth_mask(binary_cube, sigma=1.0, threshold=0.5)
        assert set(np.unique(out).tolist()) <= {0.0, 1.0}

    def test_all_zero_mask_returns_zeros(self) -> None:
        mask = np.zeros((4, 4), dtype=np.float64)
        out = smooth_mask(mask, sigma=0.5)
        np.testing.assert_array_equal(out, mask)

    def test_rejects_non_binary_values(self) -> None:
        mask = np.array([[0.0, 0.25], [0.75, 1.0]], dtype=np.float64)
        with pytest.raises(ValueError, match="only 0s and 1s"):
            smooth_mask(mask, sigma=1.0)

    def test_rejects_multi_label_mask(self) -> None:
        mask = np.array([[0, 1], [2, 1]], dtype=np.int32)
        with pytest.raises(ValueError, match="only 0s and 1s"):
            smooth_mask(mask, sigma=1.0)

    def test_rejects_negative_sigma(self, binary_cube: np.ndarray) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            smooth_mask(binary_cube, sigma=-1.0)

    def test_rejects_sigma_tuple_wrong_length(self, binary_cube: np.ndarray) -> None:
        with pytest.raises(ValueError, match="tuple length"):
            smooth_mask(binary_cube, sigma=(1.0, 1.0))


class TestRelabelMask:
    def test_remaps_labels_exactly(self) -> None:
        mask = np.array([[0, 1], [2, 1]], dtype=np.int32)
        out = relabel_mask(mask, {0: 0, 1: 10, 2: 20})
        np.testing.assert_array_equal(out, np.array([[0, 10], [20, 10]]))
        assert out.dtype == np.int32

    def test_unmapped_keep(self) -> None:
        mask = np.array([0, 1, 2], dtype=np.int32)
        out = relabel_mask(mask, {1: 9}, unmapped="keep")
        np.testing.assert_array_equal(out, np.array([0, 9, 2]))

    def test_unmapped_zero(self) -> None:
        mask = np.array([0, 1, 2], dtype=np.int32)
        out = relabel_mask(mask, {1: 9}, unmapped="zero")
        np.testing.assert_array_equal(out, np.array([0, 9, 0]))

    def test_unmapped_raise(self) -> None:
        mask = np.array([0, 1, 2], dtype=np.int32)
        with pytest.raises(ValueError, match="unmapped label"):
            relabel_mask(mask, {1: 9}, unmapped="raise")

    def test_rejects_non_integral_mask(self) -> None:
        mask = np.array([0.0, 1.5], dtype=np.float64)
        with pytest.raises(ValueError, match="integer-valued"):
            relabel_mask(mask, {0: 0, 1: 2})

    def test_rejects_collisions_by_default(self) -> None:
        mask = np.array([1, 2], dtype=np.int32)
        with pytest.raises(ValueError, match="colliding"):
            relabel_mask(mask, {1: 9, 2: 9})

    def test_allow_collisions(self) -> None:
        mask = np.array([1, 2], dtype=np.int32)
        out = relabel_mask(mask, {1: 9, 2: 9}, allow_collisions=True)
        np.testing.assert_array_equal(out, np.array([9, 9]))

    def test_dtype_cast(self) -> None:
        mask = np.array([[0, 1], [1, 0]], dtype=np.int64)
        out = relabel_mask(mask, {0: 0, 1: 2}, dtype="uint8")
        assert out.dtype == np.uint8
        np.testing.assert_array_equal(out, np.array([[0, 2], [2, 0]]))

    def test_int_dtype_with_float_mapping_warns(self) -> None:
        mask = np.array([0, 1], dtype=np.int32)
        with pytest.warns(UserWarning, match="make sure this is intentional"):
            out = relabel_mask(mask, {0: 0, 1: 1.7}, dtype="int32")
        assert out.dtype == np.int32
        assert int(out[1]) == 1

    def test_empty_mapping_rejected(self) -> None:
        mask = np.array([0, 1], dtype=np.int32)
        with pytest.raises(ValueError, match="at least one"):
            relabel_mask(mask, {})


class TestComposeMasks:
    def test_or_preserves_non_conflicting_labels(self) -> None:
        first = np.array([[0, 2], [0, 3]], dtype=np.int16)
        second = np.array([[1, 0], [0, 3]], dtype=np.uint8)
        out = compose_masks([first, second], operator="OR")
        np.testing.assert_array_equal(out, np.array([[1, 2], [0, 3]]))
        assert out.dtype == np.result_type(first.dtype, second.dtype)

    def test_and_keeps_only_shared_non_zero_indices(self) -> None:
        first = np.array([[1, 2], [0, 3]], dtype=np.int16)
        second = np.array([[1, 0], [4, 3]], dtype=np.int16)
        out = compose_masks([first, second], operator="and")
        np.testing.assert_array_equal(out, np.array([[1, 0], [0, 3]]))

    def test_rejects_conflicting_non_zero_labels(self) -> None:
        first = np.array([0, 1, 2], dtype=np.int16)
        second = np.array([3, 1, 4], dtype=np.int16)
        with pytest.raises(ValueError, match="conflicting non-zero values"):
            compose_masks([first, second], operator="OR")

    def test_rejects_different_shapes(self) -> None:
        with pytest.raises(ValueError, match="same shape"):
            compose_masks([np.zeros((2, 2), dtype=int), np.zeros((2, 3), dtype=int)])

    def test_rejects_non_integral_values(self) -> None:
        with pytest.raises(ValueError, match="integer-valued"):
            compose_masks(
                [
                    np.array([0.0, 0.5]),
                    np.array([0.0, 1.0]),
                ]
            )

    def test_rejects_unknown_operator(self) -> None:
        masks = [np.array([0, 1]), np.array([1, 1])]
        with pytest.raises(ValueError, match="operator"):
            compose_masks(masks, operator="XOR")  # type: ignore[arg-type]


class TestImageMasks:
    def test_ants_threshold_forwards_parameters_and_adds_epsilon(
        self, ants_mod, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from niiflow.preproc.functional.image.masks import ants_threshold

        image = ants_mod.from_numpy(np.array([[0.0, 1.0]], dtype=np.float32))
        calls: list[dict] = []

        def fake_threshold_image(*args, **kwargs):
            calls.append({"args": args, "kwargs": kwargs})
            return image

        monkeypatch.setattr(
            "niiflow.preproc.functional.image.masks.threshold_image",
            fake_threshold_image,
        )
        out = ants_threshold(
            image,
            low_thresh=0.0,
            high_thresh=2.0,
            inval=5,
            outval=-1,
            binary=False,
            inclusive=True,
        )
        assert out is image
        assert calls[0]["args"] == (image,)
        assert calls[0]["kwargs"]["low_thresh"] > 0.0
        assert calls[0]["kwargs"]["high_thresh"] == 2.0
        assert calls[0]["kwargs"]["inval"] == 5
        assert calls[0]["kwargs"]["outval"] == -1
        assert calls[0]["kwargs"]["binary"] is False

    def test_ants_threshold_non_zero_values(self, ants_mod) -> None:
        from niiflow.preproc.functional.image.masks import ants_threshold

        image = ants_mod.from_numpy(
            np.array([[0.0, 0.5], [2.0, 0.0]], dtype=np.float32)
        )
        out = ants_threshold(image, low_thresh=0.0, inclusive=True)
        np.testing.assert_array_equal(out.numpy(), np.array([[0.0, 1.0], [1.0, 0.0]]))

    def test_ants_threshold_inclusive_requires_low_threshold(self, ants_mod) -> None:
        from niiflow.preproc.functional.image.masks import ants_threshold

        image = ants_mod.from_numpy(np.zeros((2, 2), dtype=np.float32))
        with pytest.raises(ValueError, match="requires `low_thresh`"):
            ants_threshold(image, inclusive=True)

    def test_compose_masks_preserves_geometry(self, ants_mod) -> None:
        from niiflow.preproc.functional.image.masks import (
            compose_masks as image_compose_masks,
        )

        kwargs = {"origin": (4.0, 5.0), "spacing": (1.0, 2.0)}
        first = ants_mod.from_numpy(
            np.array([[0, 2], [0, 3]], dtype=np.float32), **kwargs
        )
        second = ants_mod.from_numpy(
            np.array([[1, 0], [0, 3]], dtype=np.float32), **kwargs
        )
        out = image_compose_masks([first, second], operator="OR")
        np.testing.assert_array_equal(out.numpy(), np.array([[1, 2], [0, 3]]))
        assert out.origin == first.origin
        assert out.spacing == first.spacing
        np.testing.assert_array_equal(out.direction, first.direction)

    def test_compose_masks_rejects_grid_mismatch(self, ants_mod) -> None:
        from niiflow.preproc.functional.image.masks import (
            compose_masks as image_compose_masks,
        )

        first = ants_mod.from_numpy(
            np.zeros((2, 2), dtype=np.float32), origin=(0.0, 0.0)
        )
        second = ants_mod.from_numpy(
            np.zeros((2, 2), dtype=np.float32), origin=(1.0, 0.0)
        )
        with pytest.raises(ValueError, match="voxel grid"):
            image_compose_masks([first, second])

    def test_smooth_mask_preserves_geometry(
        self, ants_mod, binary_cube: np.ndarray
    ) -> None:
        from niiflow.preproc.functional.image.masks import (
            smooth_mask as image_smooth_mask,
        )

        image = ants_mod.from_numpy(
            binary_cube,
            origin=(1.0, 2.0, 3.0),
            spacing=(0.5, 0.5, 0.5),
        )
        out = image_smooth_mask(image, sigma=1.0)
        assert out.origin == image.origin
        assert out.spacing == image.spacing
        np.testing.assert_array_equal(out.direction, image.direction)
        assert float(out.numpy().max()) == pytest.approx(1.0)

    def test_smooth_mask_use_mm_false_uses_sigma_in_voxels(
        self, ants_mod, binary_cube: np.ndarray
    ) -> None:
        from niiflow.preproc.functional.array.masks import (
            smooth_mask as array_smooth_mask,
        )
        from niiflow.preproc.functional.image.masks import (
            smooth_mask as image_smooth_mask,
        )

        image = ants_mod.from_numpy(binary_cube, spacing=(2.0, 2.0, 2.0))
        out = image_smooth_mask(image, sigma=1.0, use_mm=False)
        expected = array_smooth_mask(binary_cube, sigma=1.0)
        np.testing.assert_allclose(out.numpy(), expected)

    def test_smooth_mask_use_mm_rescales_sigma_by_spacing(
        self, ants_mod, binary_cube: np.ndarray
    ) -> None:
        from niiflow.preproc.functional.array.masks import (
            smooth_mask as array_smooth_mask,
        )
        from niiflow.preproc.functional.image.masks import (
            smooth_mask as image_smooth_mask,
        )

        image = ants_mod.from_numpy(binary_cube, spacing=(2.0, 1.0, 0.5))
        out = image_smooth_mask(image, sigma=2.0, use_mm=True)
        expected = array_smooth_mask(binary_cube, sigma=(1.0, 2.0, 4.0))
        np.testing.assert_allclose(out.numpy(), expected)

    def test_smooth_mask_use_mm_rescales_sigma_tuple(
        self, ants_mod, binary_cube: np.ndarray
    ) -> None:
        from niiflow.preproc.functional.array.masks import (
            smooth_mask as array_smooth_mask,
        )
        from niiflow.preproc.functional.image.masks import (
            smooth_mask as image_smooth_mask,
        )

        image = ants_mod.from_numpy(binary_cube, spacing=(2.0, 1.0, 0.5))
        out = image_smooth_mask(image, sigma=(2.0, 1.0, 1.0), use_mm=True)
        expected = array_smooth_mask(binary_cube, sigma=(1.0, 1.0, 2.0))
        np.testing.assert_allclose(out.numpy(), expected)

    def test_smooth_mask_rejects_non_bool_use_mm(
        self, ants_mod, binary_cube: np.ndarray
    ) -> None:
        from niiflow.preproc.functional.image.masks import (
            smooth_mask as image_smooth_mask,
        )

        image = ants_mod.from_numpy(binary_cube)
        with pytest.raises(TypeError, match="use_mm"):
            image_smooth_mask(image, sigma=1.0, use_mm=1)  # type: ignore[arg-type]

    def test_relabel_mask_preserves_geometry(self, ants_mod) -> None:
        from niiflow.preproc.functional.image.masks import (
            relabel_mask as image_relabel_mask,
        )

        data = np.array([[[0, 1], [2, 1]]], dtype=np.float64)
        image = ants_mod.from_numpy(
            data,
            origin=(4.0, 5.0, 6.0),
            spacing=(1.0, 1.0, 2.0),
        )
        out = image_relabel_mask(image, {0: 0, 1: 10, 2: 20}, dtype="float64")
        assert out.origin == image.origin
        assert out.spacing == image.spacing
        np.testing.assert_array_equal(
            out.numpy(), np.array([[[0, 10], [20, 10]]], dtype=np.float64)
        )

    def test_ants_apply_mask_forwards_kwargs(
        self, ants_mod, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from niiflow.preproc.functional.image.masks import ants_apply_mask

        ants_image = ants_mod.from_numpy(np.ones((4, 5), dtype=np.float64))
        mask = ants_image.clone()
        calls: list[dict] = []

        def fake_apply_mask(**kwargs):
            calls.append(kwargs)
            return ants_image

        monkeypatch.setattr(
            "niiflow.preproc.functional.image.masks.mask_image",
            fake_apply_mask,
        )
        out = ants_apply_mask(image=ants_image, mask=mask, level=1, binarize=True)

        assert calls == [
            {
                "image": ants_image,
                "mask": mask,
                "level": 1,
                "binarize": True,
            }
        ]
        assert out is ants_image
