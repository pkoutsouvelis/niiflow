"""Mask utilities for numpy arrays."""

from __future__ import annotations

__all__ = [
    "relabel_mask",
    "smooth_mask",
]

import warnings
from collections.abc import Mapping
from typing import Any, Literal

import numpy as np
from scipy.ndimage import gaussian_filter

from niiflow.preproc.functional.array.utils import (
    validate_binary_values,
    validate_numeric_array,
)
from niiflow.preproc.utils.misc import resolve_numpy_dtype


def smooth_mask(
    mask: np.ndarray,
    sigma: float | tuple[float, ...] = 1.0,
    *,
    threshold: float | None = None,
) -> np.ndarray:
    """Gaussian-smooth a binary mask and renormalize to ``[0, 1]``.

    The mask is cast to ``float64``, smoothed with
    :func:`scipy.ndimage.gaussian_filter`, then divided by its post-smoothing
    maximum (when positive) so the result lies in ``[0, 1]``. Input values
    must be binary (``0`` / ``1``).

    Args:
        mask:
            Numeric binary mask array (``0`` / ``1``).
        sigma:
            Standard deviation for the Gaussian kernel, forwarded to
            :func:`scipy.ndimage.gaussian_filter`. Defaults to ``1.0``.
        threshold:
            When set, re-binarize the renormalized result at this cutoff
            (values ``>= threshold`` become ``1.0``, others ``0.0``).
            Defaults to ``None`` (keep soft probabilities).

    Returns:
        A ``float64`` array of the same shape as `mask`.

    Raises:
        ValueError: If `mask` is invalid or non-binary, or `threshold` /
            `sigma` are invalid.
    """
    validate_numeric_array(mask, name="mask")
    validate_binary_values(mask, name="mask")

    if isinstance(sigma, (int, float)):
        if float(sigma) < 0:
            raise ValueError(f"`sigma` must be non-negative, got {sigma!r}")
    elif isinstance(sigma, tuple):
        if len(sigma) != mask.ndim:
            raise ValueError(
                f"`sigma` tuple length ({len(sigma)}) must match mask ndim "
                f"({mask.ndim})"
            )
        if any(float(s) < 0 for s in sigma):
            raise ValueError(f"`sigma` values must be non-negative, got {sigma!r}")
    else:
        raise TypeError(
            f"`sigma` must be a float or a tuple of floats, got {type(sigma).__name__}"
        )

    if threshold is not None and (
        not isinstance(threshold, (int, float)) or isinstance(threshold, bool)
    ):
        raise TypeError(
            f"`threshold` must be a number or None, got {type(threshold).__name__}"
        )

    smoothed = gaussian_filter(mask.astype(np.float64, copy=False), sigma=sigma)
    max_val = float(smoothed.max())
    if max_val > 0.0:
        smoothed = smoothed / max_val

    if threshold is not None:
        smoothed = (smoothed >= float(threshold)).astype(np.float64)

    return smoothed


def relabel_mask(
    mask: np.ndarray,
    mapping: Mapping[Any, Any],
    *,
    dtype: str | np.dtype[Any] | type[Any] | None = None,
    unmapped: Literal["keep", "zero", "raise"] = "keep",
    allow_collisions: bool = False,
) -> np.ndarray:
    """Remap integer label values in `mask` according to `mapping`.

    Args:
        mask:
            Label mask whose voxels are integer-valued (any numeric dtype).
        mapping:
            Source-label → target-label mapping. Keys and values may be ints
            or floats; source keys are matched by equality against mask
            voxels.
        dtype:
            Optional output dtype (string alias or NumPy dtype). When omitted,
            the input dtype is preserved. If an integer dtype is requested
            while any mapping value is non-integral, a :class:`UserWarning`
            is emitted including ``make sure this is intentional``.
        unmapped:
            Policy for labels present in `mask` but absent from `mapping`:

            * ``"keep"`` (default) — leave them unchanged.
            * ``"zero"`` — set them to ``0``.
            * ``"raise"`` — raise :class:`ValueError`.
        allow_collisions:
            When ``False`` (default), reject mappings where two distinct
            sources share the same target.

    Returns:
        Relabeled array with the requested (or original) dtype.

    Raises:
        ValueError: If `mask` is not integer-valued, `mapping`/`unmapped` are
            invalid, collisions are disallowed, or unmapped labels are found
            under ``unmapped="raise"``.
        TypeError: If `mapping` is not a mapping or `allow_collisions` is not
            a bool.
    """
    validate_numeric_array(mask, name="mask")
    if not bool(np.all(np.equal(np.mod(mask, 1), 0))):
        raise ValueError(
            "`mask` must be a label mask with integer-valued voxels; got "
            "non-integral values."
        )

    if not isinstance(mapping, Mapping):
        raise TypeError(
            f"`mapping` must be a mapping of source→target labels, got "
            f"{type(mapping).__name__}"
        )
    if not mapping:
        raise ValueError("`mapping` must contain at least one source→target pair")
    if not isinstance(allow_collisions, bool):
        raise TypeError(
            f"`allow_collisions` must be a boolean, got "
            f"{type(allow_collisions).__name__}"
        )
    if unmapped not in {"keep", "zero", "raise"}:
        raise ValueError(
            f"`unmapped` must be one of 'keep', 'zero', 'raise', got {unmapped!r}"
        )
    targets = list(mapping.values())
    if not allow_collisions and len(targets) != len(set(targets)):
        raise ValueError(
            "`mapping` has colliding target labels; pass allow_collisions=True "
            "if this is intentional."
        )

    present = {
        label.item() if isinstance(label, np.generic) else label
        for label in np.unique(mask)
    }
    mapped_sources = set(mapping.keys())
    missing = sorted(present - mapped_sources, key=lambda x: (str(type(x)), x))
    if unmapped == "raise" and missing:
        raise ValueError(
            f"`mask` contains unmapped label(s) {missing}; add them to "
            f"`mapping` or choose unmapped='keep'/'zero'."
        )

    out_dtype = resolve_numpy_dtype(dtype) if dtype is not None else mask.dtype
    if np.issubdtype(out_dtype, np.integer):
        non_integral_targets = [
            value
            for value in mapping.values()
            if isinstance(value, (float, np.floating)) and not float(value).is_integer()
        ]
        if non_integral_targets:
            warnings.warn(
                f"Integer output dtype {out_dtype} requested with non-integral "
                f"mapping target(s) {non_integral_targets}; values will be "
                f"truncated on cast — make sure this is intentional.",
                UserWarning,
                stacklevel=2,
            )

    work_dtype = np.result_type(mask.dtype, np.float64, out_dtype)
    if unmapped == "zero":
        out = np.zeros(mask.shape, dtype=work_dtype)
    else:
        out = mask.astype(work_dtype, copy=True)

    for source, target in mapping.items():
        out[mask == source] = target

    return out.astype(out_dtype, copy=False)
