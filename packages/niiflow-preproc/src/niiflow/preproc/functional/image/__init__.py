"""Functional operations on ANTsImage objects."""

from .intensity_normalization import (
    clamp_intensities_ants,
    minmax_norm_ants,
    z_transform_norm_ants,
)

__all__ = [
    "clamp_intensities_ants",
    "z_transform_norm_ants",
    "minmax_norm_ants",
]
