"""Mask utilities for ANTsImage objects.

Mirrors the array helpers in :mod:`niiflow.preproc.functional.array.masks` and wraps
:func:`ants.mask_image` for applying (possibly multi-label) masks.
"""

from __future__ import annotations

__all__ = [
    "ants_apply_mask",
    "relabel_mask",
    "smooth_mask",
]

from collections.abc import Mapping
from typing import Any, Literal

import numpy as np
from ants.core import ANTsImage
from ants.ops import mask_image

from niiflow.preproc.functional.array import masks as _array_masks

from .utils import (
    ants_to_numpy_with_metadata,
    ensure_ants_image,
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


def smooth_mask(
    mask: ANTsImage,
    sigma: float | tuple[float, ...] = 1.0,
    *,
    threshold: float | None = None,
) -> ANTsImage:
    """Gaussian-smooth a binary ANTs mask and renormalize to ``[0, 1]``.

    Wraps :func:`~niiflow.preproc.functional.array.masks.smooth_mask`,
    preserving origin, spacing, and direction.
    """
    ensure_ants_image(mask, name="mask")
    mask_array, metadata = ants_to_numpy_with_metadata(mask)
    out_array = _array_masks.smooth_mask(
        mask_array,
        sigma=sigma,
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
