"""Tests for array and image cropping/padding functions.

These tests pin the public contract of the cropping and padding helpers
in :mod:`niiflow.preproc.functional.array.croppad` and their ANTsImage
wrappers in :mod:`niiflow.preproc.functional.image.croppad` (same names as the
array layer):

* :func:`bbox_from_mask` -- bounding box of a binary mask, optionally
  padded and always clipped to image bounds.
* :func:`crop_to_range` -- per-axis ``(start, stop)`` cropping with
  Python-slice semantics.
* :func:`crop_to_mask` -- crop to the bounding box of a mask, or to the
  array's own non-zero region.
* :func:`center_crop` -- crop centered on the array or on a mask.
* :func:`pad_to_range` -- per-axis ``(before, after)`` padding.
* :func:`center_pad` -- pad centered on the array or on a mask.
Planner functions return the operation plan in the same vocabulary they
accept: ``(result, crop_ranges)`` from crop helpers (replayable via
:func:`crop_to_range`) and ``(result, pad_ranges)`` from :func:`center_pad`
(replayable via :func:`pad_to_range`). To reach an arbitrary target shape,
chain :func:`center_crop` then :func:`center_pad` (each skips axes that
already satisfy its part of the target).

The behaviors exercised here are:

1. Numerical correctness on simple, hand-checkable 2D / 3D inputs.
2. Input non-mutation: callers' arrays are never modified in place.
3. Equivalence of the three accepted mask encodings: ``bool``,
   ``int`` 0/1, and ``float`` 0.0/1.0.
4. Shape and dtype preservation contracts.
5. Comprehensive input validation -- every documented contract failure
   raises :class:`ValueError` with a useful message.
6. Image-layer croppad mirrors the array behavior and keeps the image
   spacing and direction, while shifting the origin to remain spatially
   consistent with the cropping or padding performed.
"""

from __future__ import annotations

import numpy as np
import pytest

from niiflow.preproc.functional.array.croppad import (
    bbox_from_mask,
    center_crop,
    center_pad,
    crop_to_mask,
    crop_to_range,
    pad_to_range,
)
from niiflow.preproc.functional.image import croppad as image_croppad

# ---------------------------------------------------------------------------
# Shared fixtures.
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_volume() -> np.ndarray:
    """A deterministic 3D float volume of shape ``(4, 5, 6)``."""
    arr = np.arange(4 * 5 * 6, dtype=np.float64).reshape(4, 5, 6)
    return arr


@pytest.fixture
def cube_mask() -> np.ndarray:
    """A 3D boolean mask of shape ``(10, 10, 10)`` enclosing voxels in [3, 7)."""
    mask = np.zeros((10, 10, 10), dtype=bool)
    mask[3:7, 3:7, 3:7] = True
    return mask


@pytest.fixture
def cube_volume() -> np.ndarray:
    """A 3D float volume of shape ``(10, 10, 10)`` with values equal to the index."""
    return np.arange(10 * 10 * 10, dtype=np.float64).reshape(10, 10, 10)


# ---------------------------------------------------------------------------
# bbox_from_mask
# ---------------------------------------------------------------------------


class TestBboxFromMask:
    """Bounding box of a mask, with optional padding clipped to image bounds."""

    def test_compact_cube_yields_expected_slices(self, cube_mask: np.ndarray) -> None:
        bbox = bbox_from_mask(cube_mask)
        assert bbox == (slice(3, 7), slice(3, 7), slice(3, 7))

    def test_zero_pad_does_not_expand(self, cube_mask: np.ndarray) -> None:
        assert bbox_from_mask(cube_mask, pad=0) == bbox_from_mask(cube_mask)

    def test_pad_expands_and_clips_to_bounds(self, cube_mask: np.ndarray) -> None:
        bbox = bbox_from_mask(cube_mask, pad=2)
        # Original bbox [3, 7) +/- 2 -> [1, 9), clipped to [0, 10).
        assert bbox == (slice(1, 9), slice(1, 9), slice(1, 9))

    def test_huge_pad_saturates_to_image_bounds(self, cube_mask: np.ndarray) -> None:
        bbox = bbox_from_mask(cube_mask, pad=100)
        assert bbox == (slice(0, 10), slice(0, 10), slice(0, 10))

    def test_single_voxel_mask(self) -> None:
        mask = np.zeros((5, 5, 5), dtype=bool)
        mask[2, 3, 1] = True
        assert bbox_from_mask(mask) == (slice(2, 3), slice(3, 4), slice(1, 2))

    def test_2d_mask(self) -> None:
        mask = np.zeros((8, 8), dtype=bool)
        mask[2:5, 1:6] = True
        assert bbox_from_mask(mask) == (slice(2, 5), slice(1, 6))

    @pytest.mark.parametrize(
        "mask_factory",
        [
            lambda m: m.astype(bool),
            lambda m: m.astype(np.int32),
            lambda m: m.astype(np.float64),
        ],
        ids=["bool", "int32", "float64"],
    )
    def test_accepts_bool_int_and_float_encodings(
        self, cube_mask: np.ndarray, mask_factory
    ) -> None:
        reference = bbox_from_mask(cube_mask)
        assert bbox_from_mask(mask_factory(cube_mask)) == reference

    def test_slices_actually_index_the_mask(self, cube_mask: np.ndarray) -> None:
        bbox = bbox_from_mask(cube_mask, pad=1)
        sub = cube_mask[bbox]
        # All true voxels of the mask must still be in the sliced region.
        assert sub.sum() == cube_mask.sum()


class TestBboxFromMaskValidation:
    """Input validation contract for ``bbox_from_mask``."""

    def test_rejects_non_ndarray(self) -> None:
        with pytest.raises(ValueError, match="must be a numpy array"):
            bbox_from_mask([[1, 0], [0, 1]])  # type: ignore[arg-type]

    def test_rejects_non_binary_values(self) -> None:
        mask = np.array([[0, 1, 2], [1, 0, 1]])
        with pytest.raises(ValueError, match="only 0s and 1s"):
            bbox_from_mask(mask)

    def test_rejects_non_numeric_dtype(self) -> None:
        with pytest.raises(ValueError, match="boolean or numeric"):
            bbox_from_mask(np.array([["a", "b"], ["c", "d"]]))

    def test_rejects_empty_mask(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            bbox_from_mask(np.zeros((4, 4), dtype=bool))

    def test_rejects_negative_pad(self, cube_mask: np.ndarray) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            bbox_from_mask(cube_mask, pad=-1)


# ---------------------------------------------------------------------------
# crop_to_range
# ---------------------------------------------------------------------------


class TestCropToRange:
    """Per-axis ``(start, stop)`` cropping with Python-slice semantics."""

    def test_explicit_ranges_match_numpy_slicing(
        self, simple_volume: np.ndarray
    ) -> None:
        out = crop_to_range(simple_volume, [(1, 3), (0, 4), (2, 5)])
        np.testing.assert_array_equal(out, simple_volume[1:3, 0:4, 2:5])

    def test_none_start_means_from_beginning(self, simple_volume: np.ndarray) -> None:
        out = crop_to_range(simple_volume, [(None, 2), (None, 3), (None, 4)])
        np.testing.assert_array_equal(out, simple_volume[:2, :3, :4])

    def test_none_stop_means_to_end(self, simple_volume: np.ndarray) -> None:
        out = crop_to_range(simple_volume, [(1, None), (2, None), (3, None)])
        np.testing.assert_array_equal(out, simple_volume[1:, 2:, 3:])

    def test_all_none_returns_full_copy(self, simple_volume: np.ndarray) -> None:
        out = crop_to_range(simple_volume, [(None, None), (None, None), (None, None)])
        np.testing.assert_array_equal(out, simple_volume)
        assert out is not simple_volume

    def test_output_is_a_copy(self, simple_volume: np.ndarray) -> None:
        out = crop_to_range(simple_volume, [(1, 3), (0, 4), (2, 5)])
        out[0, 0, 0] = 99.0
        # Source must be untouched.
        assert simple_volume[1, 0, 2] != 99.0

    def test_preserves_dtype(self) -> None:
        arr = np.arange(60, dtype=np.int16).reshape(3, 4, 5)
        out = crop_to_range(arr, [(0, 2), (1, 3), (None, None)])
        assert out.dtype == np.int16

    def test_supports_2d(self) -> None:
        arr = np.arange(16, dtype=np.float64).reshape(4, 4)
        out = crop_to_range(arr, [(1, 3), (0, 2)])
        np.testing.assert_array_equal(out, arr[1:3, 0:2])


class TestCropToRangeValidation:
    """Input validation contract for ``crop_to_range``."""

    def test_rejects_non_ndarray(self) -> None:
        with pytest.raises(ValueError, match="must be a numpy array"):
            crop_to_range([1, 2, 3], [(0, 2)])  # type: ignore[arg-type]

    def test_rejects_non_numeric_dtype(self) -> None:
        with pytest.raises(ValueError, match="numeric dtype"):
            crop_to_range(np.array([["a", "b"]]), [(0, 1), (0, 1)])

    def test_rejects_wrong_ndim_ranges(self, simple_volume: np.ndarray) -> None:
        with pytest.raises(ValueError, match="length 3"):
            crop_to_range(simple_volume, [(0, 1), (0, 1)])

    def test_rejects_non_pair_entry(self, simple_volume: np.ndarray) -> None:
        with pytest.raises(ValueError, match="length-2"):
            crop_to_range(simple_volume, [(0, 1), (0, 1, 2), (0, 1)])

    def test_rejects_negative_start(self, simple_volume: np.ndarray) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            crop_to_range(simple_volume, [(-1, 1), (0, 1), (0, 1)])

    def test_rejects_stop_beyond_axis(self, simple_volume: np.ndarray) -> None:
        with pytest.raises(ValueError, match="exceeds axis length"):
            crop_to_range(
                simple_volume,
                [(0, 100), (None, None), (None, None)],
            )

    def test_rejects_empty_or_inverted_range(self, simple_volume: np.ndarray) -> None:
        with pytest.raises(ValueError, match="greater than start"):
            crop_to_range(simple_volume, [(2, 2), (0, 1), (0, 1)])


# ---------------------------------------------------------------------------
# crop_to_mask
# ---------------------------------------------------------------------------


class TestCropToMask:
    """Crop to a mask's bounding box, or to the array's non-zero region."""

    def test_crop_to_external_mask(
        self, cube_volume: np.ndarray, cube_mask: np.ndarray
    ) -> None:
        out, crop_ranges = crop_to_mask(cube_volume, mask=cube_mask)
        assert out.shape == (4, 4, 4)
        assert crop_ranges == ((3, 7), (3, 7), (3, 7))
        np.testing.assert_array_equal(out, crop_to_range(cube_volume, crop_ranges))

    def test_crop_to_self_uses_nonzero_voxels(self) -> None:
        arr = np.zeros((8, 8, 8), dtype=np.float64)
        arr[2:5, 1:4, 3:6] = 7.0
        out, crop_ranges = crop_to_mask(arr)
        assert out.shape == (3, 3, 3)
        assert crop_ranges == ((2, 5), (1, 4), (3, 6))
        np.testing.assert_array_equal(out, np.full((3, 3, 3), 7.0))

    def test_pad_expands_crop(
        self, cube_volume: np.ndarray, cube_mask: np.ndarray
    ) -> None:
        out, crop_ranges = crop_to_mask(cube_volume, mask=cube_mask, pad=2)
        # Bbox [3, 7) +/- 2 -> [1, 9).
        assert out.shape == (8, 8, 8)
        assert crop_ranges == ((1, 9), (1, 9), (1, 9))
        np.testing.assert_array_equal(out, crop_to_range(cube_volume, crop_ranges))

    def test_pad_is_clipped_to_image_bounds(
        self, cube_volume: np.ndarray, cube_mask: np.ndarray
    ) -> None:
        out, crop_ranges = crop_to_mask(cube_volume, mask=cube_mask, pad=100)
        assert out.shape == cube_volume.shape
        assert crop_ranges == ((0, 10), (0, 10), (0, 10))
        np.testing.assert_array_equal(out, cube_volume)

    def test_output_is_a_copy(
        self, cube_volume: np.ndarray, cube_mask: np.ndarray
    ) -> None:
        out, _ = crop_to_mask(cube_volume, mask=cube_mask)
        out[0, 0, 0] = -1.0
        assert cube_volume[3, 3, 3] != -1.0


class TestCropToMaskValidation:
    """Input validation contract for ``crop_to_mask``."""

    def test_rejects_non_ndarray(self) -> None:
        with pytest.raises(ValueError, match="must be a numpy array"):
            crop_to_mask([1, 2, 3])  # type: ignore[arg-type]

    def test_rejects_shape_mismatched_mask(self, cube_volume: np.ndarray) -> None:
        with pytest.raises(ValueError, match="must have shape"):
            crop_to_mask(cube_volume, mask=np.ones((5, 5, 5), dtype=bool))

    def test_rejects_non_ndarray_mask(self, cube_volume: np.ndarray) -> None:
        with pytest.raises(ValueError, match="must be a numpy array"):
            crop_to_mask(cube_volume, mask=[1, 0, 1])  # type: ignore[arg-type]

    def test_rejects_empty_mask(self, cube_volume: np.ndarray) -> None:
        with pytest.raises(ValueError, match="empty"):
            crop_to_mask(cube_volume, mask=np.zeros((10, 10, 10), dtype=bool))

    def test_rejects_negative_pad(
        self, cube_volume: np.ndarray, cube_mask: np.ndarray
    ) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            crop_to_mask(cube_volume, mask=cube_mask, pad=-1)

    def test_rejects_self_crop_when_all_zero(self) -> None:
        with pytest.raises(ValueError, match="all voxels are zero"):
            crop_to_mask(np.zeros((4, 4, 4), dtype=np.float64))


# ---------------------------------------------------------------------------
# center_crop
# ---------------------------------------------------------------------------


class TestCenterCrop:
    """Center cropping, optionally centered on a mask bbox."""

    def test_centered_on_array_center(self, cube_volume: np.ndarray) -> None:
        out, crop_ranges = center_crop(cube_volume, shape=(4, 4, 4))
        # (10 - 4) / 2 = 3 per axis.
        assert crop_ranges == ((3, 7), (3, 7), (3, 7))
        np.testing.assert_array_equal(out, crop_to_range(cube_volume, crop_ranges))

    def test_same_shape_returns_copy(self, cube_volume: np.ndarray) -> None:
        out, crop_ranges = center_crop(cube_volume, shape=cube_volume.shape)
        assert crop_ranges == ((0, 10), (0, 10), (0, 10))
        np.testing.assert_array_equal(out, cube_volume)
        assert out is not cube_volume

    def test_odd_target_size_in_even_axis(self) -> None:
        arr = np.arange(10, dtype=np.float64)
        out, crop_ranges = center_crop(arr, shape=(3,))
        np.testing.assert_array_equal(out, crop_to_range(arr, crop_ranges))

    def test_mask_centers_window_on_mask(self) -> None:
        arr = np.arange(20, dtype=np.float64)
        mask = np.zeros(20, dtype=bool)
        mask[14:18] = True
        out, crop_ranges = center_crop(arr, shape=(6,), mask=mask)
        assert crop_ranges == ((13, 19),)
        np.testing.assert_array_equal(out, crop_to_range(arr, crop_ranges))

    def test_mask_near_edge_is_clamped(self) -> None:
        arr = np.arange(20, dtype=np.float64)
        mask = np.zeros(20, dtype=bool)
        mask[17:20] = True
        out, crop_ranges = center_crop(arr, shape=(8,), mask=mask)
        assert crop_ranges == ((12, 20),)
        np.testing.assert_array_equal(out, crop_to_range(arr, crop_ranges))

    def test_output_is_a_copy(self, cube_volume: np.ndarray) -> None:
        out, _ = center_crop(cube_volume, shape=(4, 4, 4))
        out[0, 0, 0] = -1.0
        assert cube_volume[3, 3, 3] != -1.0

    def test_preserves_dtype(self) -> None:
        arr = np.arange(60, dtype=np.int16).reshape(3, 4, 5)
        out, _ = center_crop(arr, shape=(2, 2, 2))
        assert out.dtype == np.int16


class TestCenterCropValidation:
    """Input validation contract for ``center_crop``."""

    def test_rejects_non_ndarray(self) -> None:
        with pytest.raises(ValueError, match="must be a numpy array"):
            center_crop([1, 2, 3], shape=(2,))  # type: ignore[arg-type]

    def test_rejects_wrong_shape_length(self, cube_volume: np.ndarray) -> None:
        with pytest.raises(ValueError, match="length 3"):
            center_crop(cube_volume, shape=(4, 4))

    def test_rejects_non_positive_shape(self, cube_volume: np.ndarray) -> None:
        with pytest.raises(ValueError, match="positive integers"):
            center_crop(cube_volume, shape=(0, 4, 4))

    def test_leaves_axes_where_target_is_not_smaller(
        self, cube_volume: np.ndarray
    ) -> None:
        out, crop_ranges = center_crop(cube_volume, shape=(12, 4, 4))
        # Axis 0 target 12 >= 10: left intact; axes 1-2 cropped to 4.
        assert out.shape == (10, 4, 4)
        assert crop_ranges == ((0, 10), (3, 7), (3, 7))

    def test_rejects_shape_mismatched_mask(self, cube_volume: np.ndarray) -> None:
        with pytest.raises(ValueError, match="must have shape"):
            center_crop(
                cube_volume, shape=(4, 4, 4), mask=np.ones((5, 5, 5), dtype=bool)
            )


# ---------------------------------------------------------------------------
# pad_to_range
# ---------------------------------------------------------------------------


class TestPadToRange:
    """Per-axis ``(before, after)`` padding."""

    def test_constant_zero_padding_default(self, simple_volume: np.ndarray) -> None:
        out = pad_to_range(simple_volume, [(1, 2), (0, 0), (3, 0)])
        # Shape adds (3, 0, 3).
        assert out.shape == (
            simple_volume.shape[0] + 3,
            simple_volume.shape[1],
            simple_volume.shape[2] + 3,
        )
        # Original sits at [1:5, :, 3:9].
        np.testing.assert_array_equal(out[1:5, :, 3:9], simple_volume)
        # Padding voxels are zero.
        assert out[0].sum() == 0.0
        assert out[5:].sum() == 0.0
        assert out[:, :, :3].sum() == 0.0

    def test_constant_value_other_than_zero(self, simple_volume: np.ndarray) -> None:
        out = pad_to_range(
            simple_volume,
            [(1, 0), (0, 0), (0, 0)],
            mode="constant",
            constant_values=-7,
        )
        np.testing.assert_array_equal(out[0], np.full((5, 6), -7.0))
        np.testing.assert_array_equal(out[1:], simple_volume)

    def test_no_padding_returns_same_shape(self, simple_volume: np.ndarray) -> None:
        out = pad_to_range(simple_volume, [(0, 0), (0, 0), (0, 0)])
        np.testing.assert_array_equal(out, simple_volume)
        assert out.shape == simple_volume.shape

    def test_reflect_mode(self) -> None:
        arr = np.arange(4, dtype=np.float64)
        out = pad_to_range(arr, [(2, 2)], mode="reflect")
        # numpy.pad with reflect: [2,1,0,1,2,3,2,1] for arr=[0,1,2,3], pad=(2,2)
        np.testing.assert_array_equal(out, np.array([2, 1, 0, 1, 2, 3, 2, 1]))

    def test_preserves_dtype_for_constant(self) -> None:
        arr = np.arange(6, dtype=np.int32).reshape(2, 3)
        out = pad_to_range(arr, [(1, 1), (0, 1)], constant_values=0)
        assert out.dtype == np.int32


class TestPadToRangeValidation:
    """Input validation contract for ``pad_to_range``."""

    def test_rejects_non_ndarray(self) -> None:
        with pytest.raises(ValueError, match="must be a numpy array"):
            pad_to_range([1, 2, 3], [(0, 1)])  # type: ignore[arg-type]

    def test_rejects_wrong_ndim_ranges(self, simple_volume: np.ndarray) -> None:
        with pytest.raises(ValueError, match="length 3"):
            pad_to_range(simple_volume, [(0, 1), (0, 1)])

    def test_rejects_none_widths(self, simple_volume: np.ndarray) -> None:
        with pytest.raises(ValueError, match="may not contain"):
            pad_to_range(simple_volume, [(None, 0), (0, 0), (0, 0)])  # type: ignore[list-item]

    def test_rejects_negative_width(self, simple_volume: np.ndarray) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            pad_to_range(simple_volume, [(-1, 0), (0, 0), (0, 0)])


# ---------------------------------------------------------------------------
# center_pad
# ---------------------------------------------------------------------------


class TestCenterPad:
    """Center padding, optionally centered on a mask bbox."""

    def test_symmetric_padding_around_array_center(self) -> None:
        arr = np.ones((4, 4, 4), dtype=np.float64)
        out, pad_ranges = center_pad(arr, shape=(6, 6, 6))
        assert out.shape == (6, 6, 6)
        assert pad_ranges == ((1, 1), (1, 1), (1, 1))
        np.testing.assert_array_equal(out, pad_to_range(arr, pad_ranges))

    def test_asymmetric_target_picks_consistent_side(self) -> None:
        arr = np.array([1.0, 2.0, 3.0, 4.0])
        out, pad_ranges = center_pad(arr, shape=(7,))
        expected_offset = int(np.rint(2.0 - 3.5))
        pad_before = -expected_offset
        pad_after = 3 - pad_before
        expected = np.pad(arr, (pad_before, pad_after), constant_values=0)
        np.testing.assert_array_equal(out, expected)
        assert pad_ranges == ((pad_before, pad_after),)

    def test_no_padding_when_shape_matches(self, cube_volume: np.ndarray) -> None:
        out, pad_ranges = center_pad(cube_volume, shape=cube_volume.shape)
        assert pad_ranges == ((0, 0), (0, 0), (0, 0))
        np.testing.assert_array_equal(out, cube_volume)

    def test_constant_value_other_than_zero(self) -> None:
        arr = np.ones((2, 2), dtype=np.float64)
        out, pad_ranges = center_pad(
            arr, shape=(4, 4), mode="constant", constant_values=9.0
        )
        expected = pad_to_range(arr, pad_ranges, mode="constant", constant_values=9.0)
        np.testing.assert_array_equal(out, expected)

    def test_mask_centers_array_on_mask_center(self) -> None:
        arr = np.arange(4, dtype=np.float64)
        mask = np.zeros(4, dtype=bool)
        mask[2:4] = True
        out, pad_ranges = center_pad(arr, shape=(8,), mask=mask)
        expected = np.pad(arr, (1, 3), constant_values=0)
        np.testing.assert_array_equal(out, expected)
        assert pad_ranges == ((1, 3),)

    def test_mask_at_array_start_clamps(self) -> None:
        arr = np.arange(4, dtype=np.float64)
        mask = np.zeros(4, dtype=bool)
        mask[0:1] = True
        out, pad_ranges = center_pad(arr, shape=(8,), mask=mask)
        expected = np.pad(arr, (4, 0), constant_values=0)
        np.testing.assert_array_equal(out, expected)
        assert pad_ranges == ((4, 0),)

    def test_mask_at_array_end_clamps(self) -> None:
        arr = np.arange(4, dtype=np.float64)
        mask = np.zeros(4, dtype=bool)
        mask[3:4] = True
        out, pad_ranges = center_pad(arr, shape=(8,), mask=mask)
        expected_offset = int(np.rint(3.5 - 4.0))
        expected_pad_before = -expected_offset
        expected_pad_after = 4 - expected_pad_before
        expected = np.pad(
            arr, (expected_pad_before, expected_pad_after), constant_values=0
        )
        np.testing.assert_array_equal(out, expected)
        assert pad_ranges == ((expected_pad_before, expected_pad_after),)


class TestCenterPadValidation:
    """Input validation contract for ``center_pad``."""

    def test_rejects_non_ndarray(self) -> None:
        with pytest.raises(ValueError, match="must be a numpy array"):
            center_pad([1, 2, 3], shape=(5,))  # type: ignore[arg-type]

    def test_leaves_axes_where_target_is_not_larger(
        self, cube_volume: np.ndarray
    ) -> None:
        out, pad_ranges = center_pad(cube_volume, shape=(8, 12, 12))
        # Axis 0 target 8 <= 10: left intact; axes 1-2 padded to 12.
        assert out.shape == (10, 12, 12)
        assert pad_ranges == ((0, 0), (1, 1), (1, 1))

    def test_rejects_shape_mismatched_mask(self, cube_volume: np.ndarray) -> None:
        with pytest.raises(ValueError, match="must have shape"):
            center_pad(
                cube_volume,
                shape=(12, 12, 12),
                mask=np.ones((5, 5, 5), dtype=bool),
            )


# ---------------------------------------------------------------------------
# Returned range round-trips and chained crop/pad.
# ---------------------------------------------------------------------------


class TestReturnedRanges:
    """Returned ranges replay directly via crop_to_range / pad_to_range."""

    def test_crop_to_mask_ranges_round_trip(
        self, cube_volume: np.ndarray, cube_mask: np.ndarray
    ) -> None:
        out, crop_ranges = crop_to_mask(cube_volume, mask=cube_mask)
        replay = crop_to_range(cube_volume, crop_ranges)
        np.testing.assert_array_equal(replay, out)

    def test_center_crop_ranges_round_trip(self, cube_volume: np.ndarray) -> None:
        out, crop_ranges = center_crop(cube_volume, shape=(4, 4, 4))
        replay = crop_to_range(cube_volume, crop_ranges)
        np.testing.assert_array_equal(replay, out)

    def test_center_pad_ranges_round_trip(self) -> None:
        arr = np.ones((4, 4, 4), dtype=np.float64)
        out, pad_ranges = center_pad(arr, shape=(6, 6, 6))
        replay = pad_to_range(arr, pad_ranges, constant_values=0)
        np.testing.assert_array_equal(replay, out)

    def test_bbox_ranges_match_crop_to_mask(
        self, cube_volume: np.ndarray, cube_mask: np.ndarray
    ) -> None:
        bbox = bbox_from_mask(cube_mask)
        crop_ranges_from_bbox = tuple((sl.start, sl.stop) for sl in bbox)
        out, crop_ranges = crop_to_mask(cube_volume, mask=cube_mask)
        assert crop_ranges_from_bbox == crop_ranges
        replay = crop_to_range(cube_volume, crop_ranges)
        np.testing.assert_array_equal(replay, out)


class TestChainedCropPad:
    """``center_crop`` then ``center_pad`` reaches mixed targets."""

    def test_mixed_per_axis(self) -> None:
        arr = np.ones((6, 4), dtype=np.float64)
        cropped, crop_ranges = center_crop(arr, shape=(4, 4))
        out, pad_ranges = center_pad(cropped, shape=(4, 6))
        assert out.shape == (4, 6)
        assert crop_ranges == ((1, 5), (0, 4))
        assert pad_ranges == ((0, 0), (1, 1))
        replay = pad_to_range(
            crop_to_range(arr, crop_ranges), pad_ranges, constant_values=0
        )
        np.testing.assert_array_equal(out, replay)

    def test_pure_crop_then_pad(self) -> None:
        arr = np.ones((4, 4, 4), dtype=np.float64)
        cropped, _ = center_crop(arr, shape=(4, 4, 4))
        out, pad_ranges = center_pad(cropped, shape=(8, 8, 8))
        expected, expected_pad_ranges = center_pad(arr, shape=(8, 8, 8))
        np.testing.assert_array_equal(out, expected)
        assert pad_ranges == expected_pad_ranges

    def test_mask_centers_per_axis(self) -> None:
        arr = np.arange(40, dtype=np.float64).reshape(10, 4)
        mask = np.zeros((10, 4), dtype=bool)
        mask[6:8, 1:3] = True
        cropped, crop_ranges = center_crop(arr, shape=(4, 4), mask=mask)
        out, pad_ranges = center_pad(cropped, shape=(4, 6))
        assert out.shape == (4, 6)
        assert crop_ranges == ((5, 9), (0, 4))
        assert pad_ranges == ((0, 0), (1, 1))
        replay = pad_to_range(
            crop_to_range(arr, crop_ranges), pad_ranges, constant_values=0
        )
        np.testing.assert_array_equal(out, replay)


# ---------------------------------------------------------------------------
# Image-layer croppad (ANTsImage).
# ---------------------------------------------------------------------------


@pytest.fixture
def ants_mod():
    """Import ants only for image-specific tests."""
    return pytest.importorskip("ants")


@pytest.fixture
def ants_image_3d(ants_mod):
    """A 3D ANTsImage with non-trivial metadata."""
    arr = np.arange(8 * 8 * 8, dtype=np.float64).reshape(8, 8, 8)
    return ants_mod.from_numpy(
        arr,
        origin=(10.0, 20.0, 30.0),
        spacing=(0.7, 1.1, 1.3),
        direction=np.eye(3),
    )


@pytest.fixture
def ants_mask_3d(ants_mod, ants_image_3d):
    """A 3D ANTsImage mask compatible with `ants_image_3d`."""
    mask_arr = np.zeros((8, 8, 8), dtype=np.uint8)
    mask_arr[2:6, 2:6, 2:6] = 1
    return ants_mod.from_numpy(
        mask_arr,
        origin=ants_image_3d.origin,
        spacing=ants_image_3d.spacing,
        direction=np.asarray(ants_image_3d.direction),
    )


def _expected_origin(image, voxel_offset: tuple[int, ...]) -> np.ndarray:
    """Compute the spatially-consistent origin after a voxel-index offset."""
    spacing = np.asarray(image.spacing, dtype=np.float64)
    direction = np.asarray(image.direction, dtype=np.float64)
    origin = np.asarray(image.origin, dtype=np.float64)
    return origin + direction @ (spacing * np.asarray(voxel_offset))


class TestImageCropPad:
    """Image-layer croppad must mirror array behavior and update metadata."""

    def test_bbox_from_mask_matches_array(self, ants_mask_3d) -> None:
        assert image_croppad.bbox_from_mask(ants_mask_3d) == bbox_from_mask(
            ants_mask_3d.numpy()
        )

    def test_crop_to_range_matches_array(self, ants_image_3d) -> None:
        out = image_croppad.crop_to_range(ants_image_3d, [(1, 5), (2, 7), (0, 6)])
        expected = crop_to_range(ants_image_3d.numpy(), [(1, 5), (2, 7), (0, 6)])
        np.testing.assert_allclose(out.numpy(), expected)
        np.testing.assert_allclose(
            np.asarray(out.origin),
            _expected_origin(ants_image_3d, (1, 2, 0)),
        )
        assert tuple(out.spacing) == tuple(ants_image_3d.spacing)

    def test_crop_to_range_handles_none_starts(self, ants_image_3d) -> None:
        out = image_croppad.crop_to_range(
            ants_image_3d, [(None, 5), (None, None), (3, None)]
        )
        # Voxel offset = (0, 0, 3).
        np.testing.assert_allclose(
            np.asarray(out.origin),
            _expected_origin(ants_image_3d, (0, 0, 3)),
        )

    def test_crop_to_mask_matches_array(self, ants_image_3d, ants_mask_3d) -> None:
        out_img, crop_ranges = image_croppad.crop_to_mask(
            ants_image_3d, mask=ants_mask_3d, pad=1
        )
        expected_arr, expected_crop_ranges = crop_to_mask(
            ants_image_3d.numpy(), mask=ants_mask_3d.numpy(), pad=1
        )
        np.testing.assert_allclose(out_img.numpy(), expected_arr)
        assert crop_ranges == expected_crop_ranges
        np.testing.assert_allclose(
            np.asarray(out_img.origin),
            _expected_origin(
                ants_image_3d,
                tuple(start for start, _ in crop_ranges),
            ),
        )

    def test_crop_to_mask_without_mask(self, ants_mod) -> None:
        arr = np.zeros((6, 6, 6), dtype=np.float64)
        arr[1:4, 2:5, 3:6] = 1.0
        img = ants_mod.from_numpy(
            arr,
            origin=(0.0, 0.0, 0.0),
            spacing=(1.0, 1.0, 1.0),
            direction=np.eye(3),
        )
        out_img, crop_ranges = image_croppad.crop_to_mask(img)
        assert out_img.shape == (3, 3, 3)
        assert crop_ranges == ((1, 4), (2, 5), (3, 6))
        np.testing.assert_allclose(out_img.numpy(), np.ones((3, 3, 3)))
        np.testing.assert_allclose(np.asarray(out_img.origin), [1.0, 2.0, 3.0])

    def test_center_crop_matches_array(self, ants_image_3d) -> None:
        out_img, crop_ranges = image_croppad.center_crop(ants_image_3d, shape=(4, 4, 4))
        expected, expected_crop_ranges = center_crop(
            ants_image_3d.numpy(), shape=(4, 4, 4)
        )
        np.testing.assert_allclose(out_img.numpy(), expected)
        assert crop_ranges == expected_crop_ranges
        np.testing.assert_allclose(
            np.asarray(out_img.origin),
            _expected_origin(ants_image_3d, (2, 2, 2)),
        )

    def test_pad_to_range_matches_array(self, ants_image_3d) -> None:
        out = image_croppad.pad_to_range(
            ants_image_3d, [(2, 1), (0, 3), (1, 1)], constant_values=0
        )
        expected = pad_to_range(
            ants_image_3d.numpy(), [(2, 1), (0, 3), (1, 1)], constant_values=0
        )
        np.testing.assert_allclose(out.numpy(), expected)
        # Voxel offset = (-2, 0, -1).
        np.testing.assert_allclose(
            np.asarray(out.origin),
            _expected_origin(ants_image_3d, (-2, 0, -1)),
        )

    def test_center_pad_matches_array(self, ants_image_3d) -> None:
        out_img, pad_ranges = image_croppad.center_pad(
            ants_image_3d, shape=(12, 12, 12)
        )
        expected, expected_pad_ranges = center_pad(
            ants_image_3d.numpy(), shape=(12, 12, 12)
        )
        np.testing.assert_allclose(out_img.numpy(), expected)
        assert pad_ranges == expected_pad_ranges
        np.testing.assert_allclose(
            np.asarray(out_img.origin),
            _expected_origin(ants_image_3d, (-2, -2, -2)),
        )

    def test_chained_crop_pad_mixed(self, ants_image_3d) -> None:
        cropped_img, _ = image_croppad.center_crop(ants_image_3d, shape=(4, 8, 8))
        out_img, pad_ranges = image_croppad.center_pad(cropped_img, shape=(4, 10, 8))
        arr = ants_image_3d.numpy()
        cropped, crop_ranges = center_crop(arr, shape=(4, 8, 8))
        expected, expected_pad_ranges = center_pad(cropped, shape=(4, 10, 8))
        np.testing.assert_allclose(out_img.numpy(), expected)
        assert pad_ranges == expected_pad_ranges
        crop_offset = tuple(start for start, _ in crop_ranges)
        pad_offset = tuple(-before for before, _ in pad_ranges)
        np.testing.assert_allclose(
            np.asarray(out_img.origin),
            _expected_origin(
                ants_image_3d, tuple(c + p for c, p in zip(crop_offset, pad_offset))
            ),
        )

    def test_wrappers_accept_mask_image(self, ants_image_3d, ants_mask_3d) -> None:
        cropped_img, cropped_ranges = image_croppad.center_crop(
            ants_image_3d, shape=(4, 4, 4), mask=ants_mask_3d
        )
        padded_img, padded_ranges = image_croppad.center_pad(
            ants_image_3d, shape=(10, 10, 10), mask=ants_mask_3d
        )

        expected_crop, expected_crop_ranges = center_crop(
            ants_image_3d.numpy(),
            shape=(4, 4, 4),
            mask=ants_mask_3d.numpy(),
        )
        expected_pad, expected_pad_ranges = center_pad(
            ants_image_3d.numpy(),
            shape=(10, 10, 10),
            mask=ants_mask_3d.numpy(),
        )

        np.testing.assert_allclose(cropped_img.numpy(), expected_crop)
        assert cropped_ranges == expected_crop_ranges
        np.testing.assert_allclose(padded_img.numpy(), expected_pad)
        assert padded_ranges == expected_pad_ranges

    def test_spacing_and_direction_preserved(self, ants_image_3d) -> None:
        cropped, _ = image_croppad.center_crop(ants_image_3d, shape=(4, 8, 8))
        out_img, _ = image_croppad.center_pad(cropped, shape=(4, 10, 8))
        assert tuple(out_img.spacing) == tuple(ants_image_3d.spacing)
        np.testing.assert_allclose(
            np.asarray(out_img.direction), np.asarray(ants_image_3d.direction)
        )

    def test_image_wrappers_return_ranges(self, ants_image_3d) -> None:
        _, crop_ranges = image_croppad.crop_to_mask(ants_image_3d)
        _, center_ranges = image_croppad.center_crop(ants_image_3d, shape=(4, 4, 4))
        assert len(crop_ranges) == 3
        assert all(len(pair) == 2 for pair in crop_ranges)
        assert len(center_ranges) == 3

    def test_non_identity_direction_origin_update(self, ants_mod) -> None:
        arr = np.arange(4 * 4 * 4, dtype=np.float64).reshape(4, 4, 4)
        # 90-degree rotation in the xy-plane.
        direction = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        img = ants_mod.from_numpy(
            arr,
            origin=(0.0, 0.0, 0.0),
            spacing=(1.0, 1.0, 1.0),
            direction=direction,
        )
        out = image_croppad.crop_to_range(img, [(1, 4), (0, 4), (0, 4)])
        # voxel offset = (1, 0, 0). new origin = direction @ (1, 0, 0) = (0, 1, 0).
        np.testing.assert_allclose(np.asarray(out.origin), [0.0, 1.0, 0.0])

    def test_wrappers_reject_non_image_inputs(self, ants_mod) -> None:
        _ = ants_mod
        with pytest.raises(ValueError, match="ANTsImage"):
            image_croppad.bbox_from_mask(np.zeros((4, 4), dtype=bool))  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="ANTsImage"):
            image_croppad.crop_to_range(
                np.zeros((4, 4, 4)),  # type: ignore[arg-type]
                [(0, 1), (0, 1), (0, 1)],
            )
