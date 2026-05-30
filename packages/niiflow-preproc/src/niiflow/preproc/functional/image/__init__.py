"""Functional operations on ANTsImage objects."""

from .bias_field import bias_field_correction_ants
from .croppad import (
    bbox_from_mask_ants,
    center_crop_ants,
    center_pad_ants,
    crop_to_mask_ants,
    crop_to_range_ants,
    pad_to_range_ants,
)
from .denoising import denoise_ants
from .intensity_normalization import (
    clamp_intensities_ants,
    minmax_norm_ants,
    z_transform_norm_ants,
)
from .pipelines import preprocessing_pipeline_ants
from .registration import apply_transforms_ants, registration_ants
from .resampling import resample_ants, resample_to_target_ants
from .skull_stripping import brain_extraction_ants, mask_image_ants

__all__ = [
    "apply_transforms_ants",
    "bias_field_correction_ants",
    "brain_extraction_ants",
    "clamp_intensities_ants",
    "z_transform_norm_ants",
    "minmax_norm_ants",
    "bbox_from_mask_ants",
    "crop_to_range_ants",
    "crop_to_mask_ants",
    "center_crop_ants",
    "pad_to_range_ants",
    "center_pad_ants",
    "denoise_ants",
    "mask_image_ants",
    "preprocessing_pipeline_ants",
    "registration_ants",
    "resample_ants",
    "resample_to_target_ants",
]
