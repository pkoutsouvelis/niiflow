"""Intensity normalization functions for ANTsImage objects."""

from __future__ import annotations

__all__ = [
    "clamp_intensities_ants",
    "z_transform_norm_ants",
    "minmax_norm_ants",
]

from ants.core import ANTsImage

from niiflow.preproc.functional.array import (
    clamp_intensities,
    minmax_norm,
    z_transform_norm,
)

from .utils import (
    ants_to_numpy_with_metadata,
    ensure_ants_image,
    numpy_to_ants_with_metadata,
)


def clamp_intensities_ants(
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

    out_array = clamp_intensities(
        array=image_array,
        lower_pct=lower_pct,
        upper_pct=upper_pct,
        limit_to=mask_array,
    )
    return numpy_to_ants_with_metadata(out_array, metadata)


def z_transform_norm_ants(
    image: ANTsImage,
    limit_to: ANTsImage | None = None,
) -> ANTsImage:
    """Apply z-transform normalization to an ANTsImage."""
    image_array, metadata = ants_to_numpy_with_metadata(image)
    mask_array = None
    if limit_to is not None:
        ensure_ants_image(limit_to, name="limit_to")
        mask_array = limit_to.numpy()

    out_array = z_transform_norm(array=image_array, limit_to=mask_array)
    return numpy_to_ants_with_metadata(out_array, metadata)


def minmax_norm_ants(
    image: ANTsImage,
    limit_to: ANTsImage | None = None,
) -> ANTsImage:
    """Apply min-max normalization to an ANTsImage."""
    image_array, metadata = ants_to_numpy_with_metadata(image)
    mask_array = None
    if limit_to is not None:
        ensure_ants_image(limit_to, name="limit_to")
        mask_array = limit_to.numpy()

    out_array = minmax_norm(array=image_array, limit_to=mask_array)
    return numpy_to_ants_with_metadata(out_array, metadata)
