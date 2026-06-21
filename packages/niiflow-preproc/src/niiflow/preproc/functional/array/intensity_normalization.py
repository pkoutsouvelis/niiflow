"""Intensity normalization functions.

All functions in this module accept an optional ``limit_to`` mask that restricts the
statistics used to drive the transformation (percentiles, mean/std, min/max) to a region
of interest. :func:`clamp_intensities`, :func:`z_transform_norm`, and
:func:`minmax_norm` additionally accept a ``non_zero`` flag that restricts both the
statistics and the transformed region to the non-zero voxels of the input. Outputs are
always fresh arrays; inputs are never modified in place.
"""

from __future__ import annotations

__all__ = [
    "clamp_intensities",
    "z_transform_norm",
    "minmax_norm",
]

import numpy as np

from .utils import resolve_norm_mask, validate_numeric_array


def clamp_intensities(
    array: np.ndarray,
    lower_pct: float = 1.0,
    upper_pct: float = 99.0,
    limit_to: np.ndarray | None = None,
    non_zero: bool = False,
) -> np.ndarray:
    """Clamp array values to a percentile range.

    Values below the ``lower_pct``-th percentile are pulled up to the lower
    bound, and values above the ``upper_pct``-th percentile are pulled down
    to the upper bound. When `limit_to` is provided, both the percentile
    computation and the clipping are restricted to the masked region;
    voxels outside the mask are left untouched.

    Args:
        array:
            Numeric input array.
        lower_pct:
            Lower percentile, in ``[0, 100]``. Defaults to ``1.0``.
        upper_pct:
            Upper percentile, in ``[0, 100]`` and ``>= lower_pct``.
            Defaults to ``99.0``.
        limit_to:
            Optional mask of the same shape as `array`. Accepts a boolean
            array or a numeric array containing only ``0`` and ``1``. When
            given, statistics are computed from the masked region and
            clipping is applied only to the masked region. Defaults to
            ``None``.
        non_zero:
            When ``True``, restrict both the percentile computation and the
            clipping to the non-zero voxels of `array`; zero voxels are left
            untouched. Combined with `limit_to` by intersection. Defaults to
            ``False``.

    Returns:
        A clipped copy of `array` with the same shape and dtype.
    """
    validate_numeric_array(array)
    if not 0 <= lower_pct <= upper_pct <= 100:
        raise ValueError(
            "Require `0 <= lower_pct <= upper_pct <= 100`, got "
            f"lower_pct={lower_pct}, upper_pct={upper_pct}"
        )

    result = array.copy()
    mask = resolve_norm_mask(array, limit_to, non_zero)
    if mask is None:
        lower, upper = np.percentile(result, [lower_pct, upper_pct])
        np.clip(result, lower, upper, out=result)
        return result

    lower, upper = np.percentile(result[mask], [lower_pct, upper_pct])
    result[mask] = np.clip(result[mask], lower, upper)
    return result


def z_transform_norm(
    array: np.ndarray,
    limit_to: np.ndarray | None = None,
    non_zero: bool = False,
) -> np.ndarray:
    """Z-score normalize an array: ``(array - mean) / std``.

    The mean and standard deviation are computed over the full array
    (or over the masked region when `limit_to` is given) and applied to
    every element of the output.

    Args:
        array:
            Numeric input array.
        limit_to:
            Optional mask of the same shape as `array`. Accepts a boolean
            array or a numeric array containing only ``0`` and ``1``. When
            given, mean and standard deviation are computed from the
            masked region only. Defaults to ``None``.
        non_zero:
            When ``True``, compute the mean and standard deviation from the
            non-zero voxels of `array` and apply the transform only to those
            voxels; zero voxels remain zero. Combined with `limit_to` by
            intersection for the statistics. Defaults to ``False``.

    Returns:
        A z-transformed copy of `array` as ``float64`` with the same shape.

    Raises:
        ValueError: If the standard deviation over the selected region is
            zero (constant input).
    """
    validate_numeric_array(array)

    mask = resolve_norm_mask(array, limit_to, non_zero)
    sample = array if mask is None else array[mask]

    mean = float(np.mean(sample))
    std = float(np.std(sample))
    if std == 0.0:
        raise ValueError(
            "Standard deviation over the selected region is zero; "
            "cannot z-transform a constant input."
        )

    result = array.astype(np.float64, copy=True)
    if non_zero:
        apply_mask = array != 0
        result[apply_mask] = (result[apply_mask] - mean) / std
        return result
    return (result - mean) / std


def minmax_norm(
    array: np.ndarray,
    limit_to: np.ndarray | None = None,
    non_zero: bool = False,
) -> np.ndarray:
    """Min-max normalize an array to the ``[0, 1]`` range.

    The min and max are computed over the full array (or over the masked
    region when `limit_to` is given) and applied to every element of the
    output. When a mask is given, values outside the mask may fall outside
    ``[0, 1]`` (they are rescaled with the same affine transform).

    Args:
        array:
            Numeric input array.
        limit_to:
            Optional mask of the same shape as `array`. Accepts a boolean
            array or a numeric array containing only ``0`` and ``1``. When
            given, min and max are computed from the masked region only.
            Defaults to ``None``.
        non_zero:
            When ``True``, compute min and max from the non-zero voxels of
            `array` and apply the transform only to those voxels; zero voxels
            remain zero. Combined with `limit_to` by intersection for the
            statistics. Defaults to ``False``.

    Returns:
        A min-max-normalized copy of `array` as ``float64`` with the same
        shape.

    Raises:
        ValueError: If the maximum equals the minimum over the selected
            region (constant input).
    """
    validate_numeric_array(array)

    mask = resolve_norm_mask(array, limit_to, non_zero)
    sample = array if mask is None else array[mask]

    minimum = float(np.min(sample))
    maximum = float(np.max(sample))
    if maximum == minimum:
        raise ValueError(
            "Maximum equals minimum over the selected region; "
            "cannot min-max normalize a constant input."
        )

    result = array.astype(np.float64, copy=True)
    if non_zero:
        apply_mask = array != 0
        result[apply_mask] = (result[apply_mask] - minimum) / (maximum - minimum)
        return result
    return (result - minimum) / (maximum - minimum)
