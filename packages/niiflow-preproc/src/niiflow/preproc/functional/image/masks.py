"""Mask utilities for ANTsImage objects.

Mirrors the array helpers in :mod:`niiflow.preproc.functional.array.masks` and wraps
:func:`ants.mask_image` for applying (possibly multi-label) masks.
"""

from __future__ import annotations

__all__ = [
    "ants_apply_mask",
    "ants_threshold",
    "compose_masks",
    "relabel_mask",
    "smooth_mask",
]

from collections.abc import Mapping, Sequence
from typing import Any, Literal

import numpy as np
from ants.core import ANTsImage
from ants.ops import mask_image, threshold_image

from niiflow.preproc.functional.array import masks as _array_masks

from .metadata import sync_ants_metadata
from .utils import (
    ants_to_numpy_with_metadata,
    ensure_ants_image,
    mm_to_voxels,
    numpy_to_ants_with_metadata,
)


def ants_apply_mask(**kwargs: Any) -> ANTsImage:
    """Apply a (possibly multi-label) mask to an ANTsImage.

    Wraps :func:`ants.mask_image`. All arguments are forwarded as keyword
    arguments.

    Args:
        **kwargs:
            Keyword arguments forwarded to :func:`ants.mask_image`. Common
            ones include ``image`` (the :class:`ants.core.ANTsImage` to mask),
            ``mask`` (the mask or label image), ``level`` (label value(s) to
            keep), and ``binarize``.

    Returns:
        The masked :class:`ants.core.ANTsImage`.
    """
    return mask_image(**kwargs)


def ants_threshold(
    image: ANTsImage,
    low_thresh: int | float | None = None,
    high_thresh: int | float | None = None,
    inval: int = 1,
    outval: int = 0,
    binary: bool = True,
    *,
    inclusive: bool = False,
) -> ANTsImage:
    """Threshold an ANTs image, with optional strict lower-bound handling.

    Wraps :func:`ants.threshold_image`, preserving its parameters and defaults.
    When `inclusive` is enabled, a tiny epsilon is added to `low_thresh`. This
    makes ``low_thresh=0`` a convenient non-zero binarization: zero-valued
    voxels remain zero while positive voxels become `inval`.

    Args:
        image:
            Image to threshold.
        low_thresh:
            Lower edge of the threshold window.
        high_thresh:
            Upper edge of the threshold window.
        inval:
            Value assigned inside the threshold window.
        outval:
            Value assigned outside the threshold window.
        binary:
            Forwarded to :func:`ants.threshold_image`. When ``False``, ANTs
            multiplies the thresholded result by the input image.
        inclusive:
            Add a floating-point epsilon to `low_thresh` before thresholding.
            Requires `low_thresh` to be set.

    Returns:
        The thresholded ANTs image.
    """
    ensure_ants_image(image)
    if not isinstance(inclusive, bool):
        raise TypeError(
            f"`inclusive` must be a boolean, got {type(inclusive).__name__}"
        )

    if inclusive:
        if low_thresh is None:
            raise ValueError("`inclusive=True` requires `low_thresh` to be set")
        if not isinstance(low_thresh, (int, float)) or isinstance(low_thresh, bool):
            raise TypeError(
                f"`low_thresh` must be a number, got {type(low_thresh).__name__}"
            )
        if not np.isfinite(low_thresh):
            raise ValueError(f"`low_thresh` must be finite, got {low_thresh!r}")
        low_thresh = float(low_thresh) + np.finfo(np.float64).eps

    return threshold_image(
        image,
        low_thresh=low_thresh,
        high_thresh=high_thresh,
        inval=inval,
        outval=outval,
        binary=binary,
    )


def compose_masks(
    masks: Sequence[ANTsImage],
    operator: Literal["OR", "AND", "or", "and"] = "OR",
) -> ANTsImage:
    """Compose same-grid ANTs masks with logical OR or AND.

    Wraps :func:`~niiflow.preproc.functional.array.masks.compose_masks` and
    preserves the first mask's origin, spacing, and direction.
    """
    if not isinstance(masks, Sequence) or isinstance(masks, (str, bytes)):
        raise TypeError(
            f"`masks` must be a sequence of ANTsImage objects, got "
            f"{type(masks).__name__}"
        )
    if len(masks) < 2:
        raise ValueError("`masks` must contain at least two masks")

    reference = masks[0]
    ensure_ants_image(reference, name="masks[0]")
    reference_array, metadata = ants_to_numpy_with_metadata(reference)
    arrays = [reference_array]
    for index, mask in enumerate(masks[1:], start=1):
        ensure_ants_image(mask, name=f"masks[{index}]")
        try:
            sync_ants_metadata(mask, reference)
        except ValueError as e:
            raise ValueError(
                f"`masks[{index}]` must share a voxel grid with `masks[0]`: {e}"
            ) from e
        arrays.append(mask.numpy())

    out_array = _array_masks.compose_masks(arrays, operator=operator)
    return numpy_to_ants_with_metadata(out_array, metadata)


def smooth_mask(
    mask: ANTsImage,
    sigma: float | tuple[float, ...] = 1.0,
    *,
    threshold: float | None = None,
    use_mm: bool = False,
) -> ANTsImage:
    """Gaussian-smooth a binary ANTs mask and renormalize to ``[0, 1]``.

    Wraps :func:`~niiflow.preproc.functional.array.masks.smooth_mask`,
    preserving origin, spacing, and direction.

    Args:
        mask:
            Binary ANTs mask (``0`` / ``1``).
        sigma:
            Gaussian standard deviation. When `use_mm` is ``False`` (default),
            this is in voxels and is forwarded unchanged. When `use_mm` is
            ``True``, this is in millimetres and is divided by the image
            spacing per axis before smoothing.
        threshold:
            Optional re-binarization cutoff after renormalization.
        use_mm:
            Interpret `sigma` in millimetres using `mask` spacing. Defaults
            to ``False`` (voxel units).

    Returns:
        The smoothed (and optionally thresholded) mask.
    """
    ensure_ants_image(mask, name="mask")
    if not isinstance(use_mm, bool):
        raise TypeError(f"`use_mm` must be a boolean, got {type(use_mm).__name__}")

    mask_array, metadata = ants_to_numpy_with_metadata(mask)
    voxel_sigma = mm_to_voxels(sigma, metadata["spacing"]) if use_mm else sigma
    out_array = _array_masks.smooth_mask(
        mask_array,
        sigma=voxel_sigma,
        threshold=threshold,
    )
    return numpy_to_ants_with_metadata(out_array, metadata)


def relabel_mask(
    mask: ANTsImage,
    mapping: Mapping[Any, Any],
    *,
    dtype: str | np.dtype[Any] | type[Any] | None = None,
    unmapped: Literal["keep", "zero", "raise"] = "keep",
    allow_collisions: bool = False,
) -> ANTsImage:
    """Remap integer labels in an ANTs mask according to `mapping`.

    Wraps :func:`~niiflow.preproc.functional.array.masks.relabel_mask`,
    preserving origin, spacing, and direction.
    """
    ensure_ants_image(mask, name="mask")
    mask_array, metadata = ants_to_numpy_with_metadata(mask)
    out_array = _array_masks.relabel_mask(
        mask_array,
        mapping,
        dtype=dtype,
        unmapped=unmapped,
        allow_collisions=allow_collisions,
    )
    return numpy_to_ants_with_metadata(out_array, metadata)
