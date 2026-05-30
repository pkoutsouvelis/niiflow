"""Functional operations on ANTsImage objects."""

from .croppad import (
    bbox_from_mask_ants,
    center_crop_ants,
    center_pad_ants,
    crop_to_mask_ants,
    crop_to_range_ants,
    pad_to_range_ants,
)
from .intensity_normalization import (
    clamp_intensities_ants,
    minmax_norm_ants,
    z_transform_norm_ants,
)

__all__ = [
    "clamp_intensities_ants",
    "z_transform_norm_ants",
    "minmax_norm_ants",
    "bbox_from_mask_ants",
    "crop_to_range_ants",
    "crop_to_mask_ants",
    "center_crop_ants",
    "pad_to_range_ants",
    "center_pad_ants",
]
