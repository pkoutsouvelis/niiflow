"""Pipeline stages for preprocessing pipelines."""

__all__ = [
    "ANTsApplyTransforms",
    "ANTsBiasFieldCorrection",
    "ANTsBrainExtraction",
    "ANTsDenoise",
    "ANTsPreprocessBrainImage",
    "ANTsRegistration",
    "ANTsResample",
    "ANTsResampleToTarget",
    "ApplyMask",
    "CenterCrop",
    "CenterPad",
    "CheckDimensions",
    "CheckVoxelSpacing",
    "ClampIntensities",
    "Compose",
    "CropToMask",
    "CropToRange",
    "Delete",
    "GetImage",
    "MinmaxNorm",
    "PadToRange",
    "PipelineStage",
    "RelabelMask",
    "RuntimeContext",
    "Rename",
    "Reorient",
    "SmoothMask",
    "ToNumpy",
    "ZTransformNorm",
    "create_stage",
    "discover_stage_classes",
]

from .bias_field import ANTsBiasFieldCorrection
from .compose import Compose
from .croppad import (
    CenterCrop,
    CenterPad,
    CropToMask,
    CropToRange,
    PadToRange,
)
from .denoising import ANTsDenoise
from .intensity_normalization import ClampIntensities, MinmaxNorm, ZTransformNorm
from .masks import ApplyMask, RelabelMask, SmoothMask
from .pipelines import ANTsPreprocessBrainImage
from .qc import CheckDimensions, CheckVoxelSpacing
from .registration import ANTsApplyTransforms, ANTsRegistration
from .resampling import ANTsResample, ANTsResampleToTarget
from .pipeline_stage import PipelineStage, RuntimeContext
from .skull_stripping import ANTsBrainExtraction
from .stage_factory import create_stage, discover_stage_classes
from .utility import Delete, GetImage, Rename, Reorient, ToNumpy
