"""Intensity normalization functions for ANTsImage objects.

Each function mirrors its array-based counterpart in
:mod:`niiflow.preproc.functional.array.intensity_normalization`, preserving image
metadata while operating on voxel data in numpy.
"""

from __future__ import annotations

__all__ = [
    "clamp_intensities",
    "z_transform_norm",
    "minmax_norm",
]

from ants.core import ANTsImage

from niiflow.preproc.functional.array import (
    intensity_normalization as _array_intensity_normalization,
)

from .utils import (
    ants_to_numpy_with_metadata,
    ensure_ants_image,
    numpy_to_ants_with_metadata,
)


def clamp_intensities(
    image: ANTsImage,
    lower_pct: float = 1.0,
    upper_pct: float = 99.0,
    limit_to: ANTsImage | None = None,
) -> ANTsImage:
    """Clamp ANTsImage intensities to a percentile range."""
    image_array, metadata = ants_to_numpy_with_metadata(image)
    mask_array = None
    if limit_to is not None:
        ensure_ants_image(limit_to, name="limit_to")
        mask_array = limit_to.numpy()

    out_array = _array_intensity_normalization.clamp_intensities(
        array=image_array,
        lower_pct=lower_pct,
        upper_pct=upper_pct,
        limit_to=mask_array,
    )
    return numpy_to_ants_with_metadata(out_array, metadata)


def z_transform_norm(
    image: ANTsImage,
    limit_to: ANTsImage | None = None,
) -> ANTsImage:
    """Apply z-transform normalization to an ANTsImage."""
    image_array, metadata = ants_to_numpy_with_metadata(image)
    mask_array = None
    if limit_to is not None:
        ensure_ants_image(limit_to, name="limit_to")
        mask_array = limit_to.numpy()

    out_array = _array_intensity_normalization.z_transform_norm(
        array=image_array, limit_to=mask_array
    )
    return numpy_to_ants_with_metadata(out_array, metadata)


def minmax_norm(
    image: ANTsImage,
    limit_to: ANTsImage | None = None,
) -> ANTsImage:
    """Apply min-max normalization to an ANTsImage."""
    image_array, metadata = ants_to_numpy_with_metadata(image)
    mask_array = None
    if limit_to is not None:
        ensure_ants_image(limit_to, name="limit_to")
        mask_array = limit_to.numpy()

    out_array = _array_intensity_normalization.minmax_norm(
        array=image_array, limit_to=mask_array
    )
    return numpy_to_ants_with_metadata(out_array, metadata)
