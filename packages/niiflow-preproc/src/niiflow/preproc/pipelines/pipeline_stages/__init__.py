"""Pipeline stages for preprocessing pipelines."""

__all__ = [
    "AggregateChecks",
    "ANTsApplyTransforms",
    "ANTsBiasFieldCorrection",
    "ANTsBrainExtraction",
    "ANTsDenoise",
    "ANTsPreprocessBrainImage",
    "ANTsRegistration",
    "ANTsResample",
    "ANTsResampleToTarget",
    "ANTsSimilarityMetric",
    "ANTsThreshold",
    "ApplyMask",
    "CenterCrop",
    "CenterPad",
    "CheckDimensions",
    "CheckMetric",
    "CheckVoxelSpacing",
    "ClampIntensities",
    "Compose",
    "ComposeMasks",
    "CropToMask",
    "CropToRange",
    "Delete",
    "DiceSimilarityCoefficient",
    "GetImage",
    "MinmaxNorm",
    "PadToRange",
    "PipelineStage",
    "PointwiseArithmetic",
    "RelabelMask",
    "RuntimeContext",
    "Rename",
    "Reorient",
    "SmoothMask",
    "SyncMetadata",
    "ToANTs",
    "ToNumpy",
    "ZTransformNorm",
    "create_pipeline_stage",
    "discover_pipeline_stage_classes",
]

from .arithmetic import PointwiseArithmetic
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
from .masks import ANTsThreshold, ApplyMask, ComposeMasks, RelabelMask, SmoothMask
from .metrics import ANTsSimilarityMetric, DiceSimilarityCoefficient
from .pipelines import ANTsPreprocessBrainImage
from .qc import AggregateChecks, CheckDimensions, CheckMetric, CheckVoxelSpacing
from .registration import ANTsApplyTransforms, ANTsRegistration
from .resampling import ANTsResample, ANTsResampleToTarget
from .pipeline_stage import PipelineStage, RuntimeContext
from .skull_stripping import ANTsBrainExtraction
from .pipeline_stage_factory import (
    create_pipeline_stage,
    discover_pipeline_stage_classes,
)
from .utility import Delete, GetImage, Rename, Reorient, SyncMetadata, ToANTs, ToNumpy
