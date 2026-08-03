"""Functional operations on ANTsImage objects."""

from .bias_field import ants_bias_field_correction
from .croppad import (
    bbox_from_mask,
    center_crop,
    center_pad,
    crop_to_mask,
    crop_to_range,
    pad_to_range,
)
from .denoising import ants_denoise
from .intensity_normalization import (
    clamp_intensities,
    minmax_norm,
    z_transform_norm,
)
from .masks import ants_apply_mask, relabel_mask, smooth_mask
from .pipelines import ants_preprocess_brain_image
from .registration import ants_apply_transforms, ants_registration
from .reorientation import ants_reorient
from .resampling import ants_resample, ants_resample_to_target
from .skull_stripping import ants_brain_extraction

__all__ = [
    "ants_apply_transforms",
    "ants_bias_field_correction",
    "ants_brain_extraction",
    "clamp_intensities",
    "z_transform_norm",
    "minmax_norm",
    "bbox_from_mask",
    "crop_to_range",
    "crop_to_mask",
    "center_crop",
    "pad_to_range",
    "center_pad",
    "ants_denoise",
    "ants_apply_mask",
    "smooth_mask",
    "relabel_mask",
    "ants_preprocess_brain_image",
    "ants_registration",
    "ants_reorient",
    "ants_resample",
    "ants_resample_to_target",
]
