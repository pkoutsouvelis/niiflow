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
    non_zero: bool = False,
) -> ANTsImage:
    """Clamp ANTsImage intensities to a percentile range.

    When ``non_zero`` is ``True``, both the percentile computation and the clipping are
    restricted to the non-zero voxels of the image.
    """
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
        non_zero=non_zero,
    )
    return numpy_to_ants_with_metadata(out_array, metadata)


def z_transform_norm(
    image: ANTsImage,
    limit_to: ANTsImage | None = None,
    non_zero: bool = False,
) -> ANTsImage:
    """Apply z-transform normalization to an ANTsImage.

    When ``non_zero`` is ``True``, the mean and standard deviation are computed from the
    non-zero voxels and the transform is applied only to those voxels; zero voxels
    remain zero.
    """
    image_array, metadata = ants_to_numpy_with_metadata(image)
    mask_array = None
    if limit_to is not None:
        ensure_ants_image(limit_to, name="limit_to")
        mask_array = limit_to.numpy()

    out_array = _array_intensity_normalization.z_transform_norm(
        array=image_array, limit_to=mask_array, non_zero=non_zero
    )
    return numpy_to_ants_with_metadata(out_array, metadata)


def minmax_norm(
    image: ANTsImage,
    limit_to: ANTsImage | None = None,
    non_zero: bool = False,
) -> ANTsImage:
    """Apply min-max normalization to an ANTsImage.

    When ``non_zero`` is ``True``, min and max are computed from the non-zero voxels and
    the transform is applied only to those voxels; zero voxels remain zero.
    """
    image_array, metadata = ants_to_numpy_with_metadata(image)
    mask_array = None
    if limit_to is not None:
        ensure_ants_image(limit_to, name="limit_to")
        mask_array = limit_to.numpy()

    out_array = _array_intensity_normalization.minmax_norm(
        array=image_array, limit_to=mask_array, non_zero=non_zero
    )
    return numpy_to_ants_with_metadata(out_array, metadata)
