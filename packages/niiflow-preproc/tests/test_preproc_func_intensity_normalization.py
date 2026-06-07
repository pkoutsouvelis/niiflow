"""Tests for array and image intensity normalization functions.

These tests pin the public contract of the three intensity normalization
functions in the module:

* :func:`clamp_intensities` -- percentile clipping, optionally restricted
  to a region of interest.
* :func:`z_transform_norm` -- z-score normalization driven by mean/std
  computed over the full array or a masked region.
* :func:`minmax_norm` -- min-max normalization to ``[0, 1]`` driven by
  min/max over the full array or a masked region.

The behaviors exercised here are:

1. Numerical correctness on simple, hand-checkable inputs (no random
   tolerances unless explicitly noted).
2. Input non-mutation: callers' arrays are never modified in place.
3. Equivalence of the three accepted `limit_to` mask encodings:
   ``bool`` arrays, ``int`` 0/1 arrays, and ``float`` 0.0/1.0 arrays.
4. Restricted vs. unrestricted statistics: a mask that excludes outliers
   must yield different results than no mask, and yields the same results
   as running the function on the masked subset alone.
5. Comprehensive input validation -- every documented contract failure
   raises :class:`ValueError` with a useful message.
"""

from __future__ import annotations

import numpy as np
import pytest

from niiflow.preproc.functional.array.intensity_normalization import (
    clamp_intensities,
    minmax_norm,
    z_transform_norm,
)
from niiflow.preproc.functional.image import (
    intensity_normalization as image_intensity_normalization,
)

# ---------------------------------------------------------------------------
# Shared fixtures.
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_array() -> np.ndarray:
    """A deterministic 1-D float array spanning ``[0.0, 9.0]``."""
    return np.arange(10, dtype=np.float64)


@pytest.fixture
def array_with_outliers() -> np.ndarray:
    """A 1-D array dominated by 1..8 with extreme tails at 0 and 9."""
    return np.array([-1000.0, 1, 2, 3, 4, 5, 6, 7, 8, 1000.0], dtype=np.float64)


# ---------------------------------------------------------------------------
# clamp_intensities
# ---------------------------------------------------------------------------


class TestClampIntensities:
    """Percentile clipping with and without a region-of-interest mask."""

    def test_default_percentiles_clip_extreme_tails(
        self, array_with_outliers: np.ndarray
    ) -> None:
        out = clamp_intensities(array_with_outliers)

        lower, upper = np.percentile(array_with_outliers, [1.0, 99.0])
        assert out.min() == pytest.approx(lower)
        assert out.max() == pytest.approx(upper)
        assert out.shape == array_with_outliers.shape

    def test_zero_to_hundred_percentiles_is_identity(
        self, simple_array: np.ndarray
    ) -> None:
        out = clamp_intensities(simple_array, lower_pct=0.0, upper_pct=100.0)
        np.testing.assert_array_equal(out, simple_array)

    def test_custom_percentiles_match_numpy(self, simple_array: np.ndarray) -> None:
        out = clamp_intensities(simple_array, lower_pct=25.0, upper_pct=75.0)
        lower, upper = np.percentile(simple_array, [25.0, 75.0])
        np.testing.assert_array_equal(out, np.clip(simple_array, lower, upper))

    def test_does_not_modify_input(self, array_with_outliers: np.ndarray) -> None:
        original = array_with_outliers.copy()
        _ = clamp_intensities(array_with_outliers)
        np.testing.assert_array_equal(array_with_outliers, original)

    def test_preserves_dtype(self) -> None:
        arr = np.arange(10, dtype=np.float32)
        out = clamp_intensities(arr, lower_pct=10.0, upper_pct=90.0)
        assert out.dtype == np.float32

    def test_2d_array(self) -> None:
        rng = np.random.default_rng(0)
        arr = rng.normal(size=(8, 8)).astype(np.float64)
        out = clamp_intensities(arr, lower_pct=5.0, upper_pct=95.0)

        lower, upper = np.percentile(arr, [5.0, 95.0])
        np.testing.assert_array_equal(out, np.clip(arr, lower, upper))

    def test_limit_to_restricts_percentile_computation(self) -> None:
        arr = np.array([0.0, 1, 2, 3, 4, 5, 6, 7, 8, 100.0])
        mask = np.array([0, 1, 1, 1, 1, 1, 1, 1, 1, 0], dtype=bool)

        out = clamp_intensities(arr, lower_pct=0.0, upper_pct=100.0, limit_to=mask)

        # Within the mask the data already spans [1, 8] -> no clipping.
        # Outside the mask the original values are untouched.
        np.testing.assert_array_equal(out, arr)

    def test_limit_to_only_clips_masked_region(self) -> None:
        arr = np.array([-50.0, 1, 2, 3, 4, 5, 6, 7, 8, 50.0])
        mask = np.array([0, 1, 1, 1, 1, 1, 1, 1, 1, 0], dtype=bool)

        out = clamp_intensities(arr, lower_pct=25.0, upper_pct=75.0, limit_to=mask)

        # Outside-mask voxels are preserved exactly.
        assert out[0] == -50.0
        assert out[-1] == 50.0

        # Inside-mask values are clipped to the masked percentiles.
        inside = arr[mask]
        lower, upper = np.percentile(inside, [25.0, 75.0])
        np.testing.assert_array_equal(out[mask], np.clip(inside, lower, upper))

    @pytest.mark.parametrize(
        "mask_factory",
        [
            lambda m: m.astype(bool),
            lambda m: m.astype(np.int32),
            lambda m: m.astype(np.float64),
        ],
        ids=["bool", "int32", "float64"],
    )
    def test_limit_to_accepts_bool_int_and_float_encodings(self, mask_factory) -> None:
        arr = np.array([-50.0, 1, 2, 3, 4, 5, 6, 7, 8, 50.0])
        bool_mask = np.array([0, 1, 1, 1, 1, 1, 1, 1, 1, 0], dtype=bool)
        mask = mask_factory(bool_mask)

        out = clamp_intensities(arr, lower_pct=25.0, upper_pct=75.0, limit_to=mask)

        reference = clamp_intensities(
            arr, lower_pct=25.0, upper_pct=75.0, limit_to=bool_mask
        )
        np.testing.assert_array_equal(out, reference)


class TestClampIntensitiesValidation:
    """Every documented contract violation must raise ``ValueError``."""

    def test_rejects_non_ndarray(self) -> None:
        with pytest.raises(ValueError, match="must be a numpy array"):
            clamp_intensities([1, 2, 3])  # type: ignore[arg-type]

    def test_rejects_non_numeric_dtype(self) -> None:
        with pytest.raises(ValueError, match="numeric dtype"):
            clamp_intensities(np.array(["a", "b", "c"]))

    def test_rejects_inverted_percentiles(self, simple_array: np.ndarray) -> None:
        with pytest.raises(ValueError, match="lower_pct <= upper_pct"):
            clamp_intensities(simple_array, lower_pct=80.0, upper_pct=20.0)

    @pytest.mark.parametrize(
        "lower_pct,upper_pct",
        [(-1.0, 99.0), (1.0, 101.0), (-5.0, 105.0)],
    )
    def test_rejects_out_of_range_percentiles(
        self, simple_array: np.ndarray, lower_pct: float, upper_pct: float
    ) -> None:
        with pytest.raises(ValueError, match="0 <= lower_pct <= upper_pct <= 100"):
            clamp_intensities(simple_array, lower_pct=lower_pct, upper_pct=upper_pct)

    def test_rejects_non_ndarray_limit_to(self, simple_array: np.ndarray) -> None:
        with pytest.raises(ValueError, match="`limit_to` must be a numpy array"):
            clamp_intensities(simple_array, limit_to=[1, 0, 1, 0, 1, 0, 1, 0, 1, 0])  # type: ignore[arg-type]

    def test_rejects_shape_mismatched_limit_to(self, simple_array: np.ndarray) -> None:
        with pytest.raises(ValueError, match="must have shape"):
            clamp_intensities(simple_array, limit_to=np.ones(5, dtype=bool))

    def test_rejects_limit_to_with_non_binary_values(
        self, simple_array: np.ndarray
    ) -> None:
        mask = np.array([0, 1, 2, 0, 1, 0, 1, 0, 1, 0])
        with pytest.raises(ValueError, match="only 0s and 1s"):
            clamp_intensities(simple_array, limit_to=mask)

    def test_rejects_limit_to_with_non_numeric_dtype(
        self, simple_array: np.ndarray
    ) -> None:
        mask = np.array(["1"] * 10)
        with pytest.raises(ValueError, match="boolean or numeric"):
            clamp_intensities(simple_array, limit_to=mask)

    def test_rejects_empty_limit_to_mask(self, simple_array: np.ndarray) -> None:
        with pytest.raises(ValueError, match="empty"):
            clamp_intensities(simple_array, limit_to=np.zeros(10, dtype=bool))


# ---------------------------------------------------------------------------
# z_transform_norm
# ---------------------------------------------------------------------------


class TestZTransformNorm:
    """Mean/std-driven normalization across full or masked region."""

    def test_full_array_normalization_is_unit_variance_zero_mean(
        self, simple_array: np.ndarray
    ) -> None:
        out = z_transform_norm(simple_array)

        assert out.mean() == pytest.approx(0.0, abs=1e-12)
        assert out.std() == pytest.approx(1.0, abs=1e-12)
        assert out.shape == simple_array.shape
        assert out.dtype == np.float64

    def test_matches_manual_formula(self, simple_array: np.ndarray) -> None:
        mean = simple_array.mean()
        std = simple_array.std()
        expected = (simple_array - mean) / std

        np.testing.assert_allclose(z_transform_norm(simple_array), expected)

    def test_does_not_modify_input(self, simple_array: np.ndarray) -> None:
        original = simple_array.copy()
        _ = z_transform_norm(simple_array)
        np.testing.assert_array_equal(simple_array, original)

    def test_returns_float64_for_integer_input(self) -> None:
        arr = np.arange(10, dtype=np.int32)
        out = z_transform_norm(arr)
        assert out.dtype == np.float64

    def test_limit_to_uses_masked_statistics_for_all_voxels(self) -> None:
        arr = np.array([-100.0, 1, 2, 3, 4, 5, 6, 7, 8, 100.0])
        mask = np.array([0, 1, 1, 1, 1, 1, 1, 1, 1, 0], dtype=bool)

        out = z_transform_norm(arr, limit_to=mask)

        inside = arr[mask]
        expected_mean = inside.mean()
        expected_std = inside.std()
        expected = (arr - expected_mean) / expected_std

        np.testing.assert_allclose(out, expected)
        # Masked region itself has unit variance / zero mean.
        assert out[mask].mean() == pytest.approx(0.0, abs=1e-12)
        assert out[mask].std() == pytest.approx(1.0, abs=1e-12)

    def test_limit_to_differs_from_unmasked_when_outliers_present(self) -> None:
        arr = np.array([-100.0, 1, 2, 3, 4, 5, 6, 7, 8, 100.0])
        mask = np.array([0, 1, 1, 1, 1, 1, 1, 1, 1, 0], dtype=bool)

        unmasked = z_transform_norm(arr)
        masked = z_transform_norm(arr, limit_to=mask)

        assert not np.allclose(unmasked, masked)

    @pytest.mark.parametrize(
        "mask_factory",
        [
            lambda m: m.astype(bool),
            lambda m: m.astype(np.int32),
            lambda m: m.astype(np.float64),
        ],
        ids=["bool", "int32", "float64"],
    )
    def test_limit_to_accepts_bool_int_and_float_encodings(self, mask_factory) -> None:
        arr = np.array([-100.0, 1, 2, 3, 4, 5, 6, 7, 8, 100.0])
        bool_mask = np.array([0, 1, 1, 1, 1, 1, 1, 1, 1, 0], dtype=bool)

        out = z_transform_norm(arr, limit_to=mask_factory(bool_mask))
        reference = z_transform_norm(arr, limit_to=bool_mask)

        np.testing.assert_array_equal(out, reference)

    def test_raises_on_constant_full_array(self) -> None:
        arr = np.full(10, 3.14, dtype=np.float64)
        with pytest.raises(ValueError, match="Standard deviation"):
            z_transform_norm(arr)

    def test_raises_on_constant_masked_region(self) -> None:
        arr = np.array([0.0, 7, 7, 7, 7, 7, 7, 7, 7, 0.0])
        mask = np.array([0, 1, 1, 1, 1, 1, 1, 1, 1, 0], dtype=bool)
        with pytest.raises(ValueError, match="Standard deviation"):
            z_transform_norm(arr, limit_to=mask)


class TestZTransformNormValidation:
    """Input validation contract for ``z_transform_norm``."""

    def test_rejects_non_ndarray(self) -> None:
        with pytest.raises(ValueError, match="must be a numpy array"):
            z_transform_norm([1, 2, 3])  # type: ignore[arg-type]

    def test_rejects_non_numeric_dtype(self) -> None:
        with pytest.raises(ValueError, match="numeric dtype"):
            z_transform_norm(np.array(["a", "b", "c"]))

    def test_rejects_shape_mismatched_limit_to(self, simple_array: np.ndarray) -> None:
        with pytest.raises(ValueError, match="must have shape"):
            z_transform_norm(simple_array, limit_to=np.ones(5, dtype=bool))

    def test_rejects_limit_to_with_non_binary_values(
        self, simple_array: np.ndarray
    ) -> None:
        mask = np.array([0, 1, 2, 0, 1, 0, 1, 0, 1, 0])
        with pytest.raises(ValueError, match="only 0s and 1s"):
            z_transform_norm(simple_array, limit_to=mask)

    def test_rejects_empty_limit_to_mask(self, simple_array: np.ndarray) -> None:
        with pytest.raises(ValueError, match="empty"):
            z_transform_norm(simple_array, limit_to=np.zeros(10, dtype=bool))


# ---------------------------------------------------------------------------
# minmax_norm
# ---------------------------------------------------------------------------


class TestMinmaxNorm:
    """Min/max-driven normalization across full or masked region."""

    def test_full_array_normalization_maps_to_unit_interval(
        self, simple_array: np.ndarray
    ) -> None:
        out = minmax_norm(simple_array)

        assert out.min() == pytest.approx(0.0)
        assert out.max() == pytest.approx(1.0)
        assert out.shape == simple_array.shape
        assert out.dtype == np.float64

    def test_matches_manual_formula(self, simple_array: np.ndarray) -> None:
        expected = (simple_array - simple_array.min()) / (
            simple_array.max() - simple_array.min()
        )
        np.testing.assert_allclose(minmax_norm(simple_array), expected)

    def test_does_not_modify_input(self, simple_array: np.ndarray) -> None:
        original = simple_array.copy()
        _ = minmax_norm(simple_array)
        np.testing.assert_array_equal(simple_array, original)

    def test_returns_float64_for_integer_input(self) -> None:
        arr = np.arange(10, dtype=np.int32)
        out = minmax_norm(arr)
        assert out.dtype == np.float64

    def test_limit_to_uses_masked_statistics_for_all_voxels(self) -> None:
        arr = np.array([-100.0, 1, 2, 3, 4, 5, 6, 7, 8, 100.0])
        mask = np.array([0, 1, 1, 1, 1, 1, 1, 1, 1, 0], dtype=bool)

        out = minmax_norm(arr, limit_to=mask)

        inside = arr[mask]
        expected = (arr - inside.min()) / (inside.max() - inside.min())
        np.testing.assert_allclose(out, expected)

        # Inside-mask values land in [0, 1].
        assert out[mask].min() == pytest.approx(0.0)
        assert out[mask].max() == pytest.approx(1.0)
        # Outside-mask values can fall outside [0, 1] (here they do).
        assert out[0] < 0.0
        assert out[-1] > 1.0

    def test_limit_to_differs_from_unmasked_when_outliers_present(self) -> None:
        arr = np.array([-100.0, 1, 2, 3, 4, 5, 6, 7, 8, 100.0])
        mask = np.array([0, 1, 1, 1, 1, 1, 1, 1, 1, 0], dtype=bool)

        unmasked = minmax_norm(arr)
        masked = minmax_norm(arr, limit_to=mask)

        assert not np.allclose(unmasked, masked)

    @pytest.mark.parametrize(
        "mask_factory",
        [
            lambda m: m.astype(bool),
            lambda m: m.astype(np.int32),
            lambda m: m.astype(np.float64),
        ],
        ids=["bool", "int32", "float64"],
    )
    def test_limit_to_accepts_bool_int_and_float_encodings(self, mask_factory) -> None:
        arr = np.array([-100.0, 1, 2, 3, 4, 5, 6, 7, 8, 100.0])
        bool_mask = np.array([0, 1, 1, 1, 1, 1, 1, 1, 1, 0], dtype=bool)

        out = minmax_norm(arr, limit_to=mask_factory(bool_mask))
        reference = minmax_norm(arr, limit_to=bool_mask)

        np.testing.assert_array_equal(out, reference)

    def test_raises_on_constant_full_array(self) -> None:
        arr = np.full(10, 3.14, dtype=np.float64)
        with pytest.raises(ValueError, match="Maximum equals minimum"):
            minmax_norm(arr)

    def test_raises_on_constant_masked_region(self) -> None:
        arr = np.array([0.0, 7, 7, 7, 7, 7, 7, 7, 7, 0.0])
        mask = np.array([0, 1, 1, 1, 1, 1, 1, 1, 1, 0], dtype=bool)
        with pytest.raises(ValueError, match="Maximum equals minimum"):
            minmax_norm(arr, limit_to=mask)


class TestMinmaxNormValidation:
    """Input validation contract for ``minmax_norm``."""

    def test_rejects_non_ndarray(self) -> None:
        with pytest.raises(ValueError, match="must be a numpy array"):
            minmax_norm([1, 2, 3])  # type: ignore[arg-type]

    def test_rejects_non_numeric_dtype(self) -> None:
        with pytest.raises(ValueError, match="numeric dtype"):
            minmax_norm(np.array(["a", "b", "c"]))

    def test_rejects_shape_mismatched_limit_to(self, simple_array: np.ndarray) -> None:
        with pytest.raises(ValueError, match="must have shape"):
            minmax_norm(simple_array, limit_to=np.ones(5, dtype=bool))

    def test_rejects_limit_to_with_non_binary_values(
        self, simple_array: np.ndarray
    ) -> None:
        mask = np.array([0, 1, 2, 0, 1, 0, 1, 0, 1, 0])
        with pytest.raises(ValueError, match="only 0s and 1s"):
            minmax_norm(simple_array, limit_to=mask)

    def test_rejects_empty_limit_to_mask(self, simple_array: np.ndarray) -> None:
        with pytest.raises(ValueError, match="empty"):
            minmax_norm(simple_array, limit_to=np.zeros(10, dtype=bool))


@pytest.fixture
def ants_mod():
    """Import ants only for image-specific tests."""
    return pytest.importorskip("ants")


class TestImageIntensityNormalization:
    """Image-layer normalization must mirror array behavior and preserve metadata."""

    def test_clamp_intensities_matches_array_function(self, ants_mod) -> None:
        arr = np.array(
            [[-100.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 100.0]],
            dtype=np.float64,
        )
        img = ants_mod.from_numpy(arr)

        out_img = image_intensity_normalization.clamp_intensities(
            img, lower_pct=10.0, upper_pct=90.0
        )
        expected = clamp_intensities(arr, lower_pct=10.0, upper_pct=90.0)

        np.testing.assert_allclose(out_img.numpy(), expected)
        assert out_img.shape == img.shape

    def test_z_transform_norm_matches_array_function(self, ants_mod) -> None:
        arr = np.array(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
            dtype=np.float64,
        )
        img = ants_mod.from_numpy(arr)

        out_img = image_intensity_normalization.z_transform_norm(img)
        expected = z_transform_norm(arr)

        np.testing.assert_allclose(out_img.numpy(), expected)

    def test_minmax_norm_matches_array_function(self, ants_mod) -> None:
        arr = np.array(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
            dtype=np.float64,
        )
        img = ants_mod.from_numpy(arr)

        out_img = image_intensity_normalization.minmax_norm(img)
        expected = minmax_norm(arr)

        np.testing.assert_allclose(out_img.numpy(), expected)

    def test_image_wrappers_accept_limit_to_mask(self, ants_mod) -> None:
        arr = np.array(
            [[-100.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 100.0]],
            dtype=np.float64,
        )
        mask_arr = np.array(
            [[0, 1, 1], [1, 1, 1], [1, 1, 0]],
            dtype=np.uint8,
        )
        img = ants_mod.from_numpy(arr)
        mask_img = ants_mod.from_numpy(mask_arr)

        clamped = image_intensity_normalization.clamp_intensities(
            img, lower_pct=25.0, upper_pct=75.0, limit_to=mask_img
        )
        zed = image_intensity_normalization.z_transform_norm(img, limit_to=mask_img)
        minmaxed = image_intensity_normalization.minmax_norm(img, limit_to=mask_img)

        np.testing.assert_allclose(
            clamped.numpy(),
            clamp_intensities(arr, lower_pct=25.0, upper_pct=75.0, limit_to=mask_arr),
        )
        np.testing.assert_allclose(
            zed.numpy(), z_transform_norm(arr, limit_to=mask_arr)
        )
        np.testing.assert_allclose(
            minmaxed.numpy(), minmax_norm(arr, limit_to=mask_arr)
        )

    def test_image_wrappers_preserve_metadata(self, ants_mod) -> None:
        arr = np.array(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
            dtype=np.float64,
        )
        direction = np.eye(2)
        img = ants_mod.from_numpy(
            arr,
            origin=(10.0, 20.0),
            spacing=(0.7, 1.1),
            direction=direction,
        )

        out_img = image_intensity_normalization.minmax_norm(img)

        assert tuple(out_img.origin) == tuple(img.origin)
        assert tuple(out_img.spacing) == tuple(img.spacing)
        np.testing.assert_allclose(
            np.asarray(out_img.direction), np.asarray(img.direction)
        )

    def test_image_wrappers_reject_non_image_inputs(self, ants_mod) -> None:
        _ = ants_mod  # keep fixture usage explicit for skip behavior
        with pytest.raises(ValueError, match="ANTsImage"):
            image_intensity_normalization.clamp_intensities(
                np.array([1, 2, 3])  # type: ignore[arg-type]
            )
