"""Pipeline stages wrapping
:mod:`niiflow.preproc.functional.image.intensity_normalization`."""

from __future__ import annotations

__all__ = [
    "ClampIntensities",
    "MinmaxNorm",
    "ZTransformNorm",
]

from pathlib import Path
from typing import Any

from ants.core import ANTsImage

from niiflow.preproc.functional.image.intensity_normalization import (
    clamp_intensities,
    minmax_norm,
    z_transform_norm,
)
from niiflow.preproc.pipelines.pipeline_stages.pipeline_stage import PipelineStage
from niiflow.preproc.utils.file import ants_image_read, ants_image_write


class ClampIntensities(PipelineStage):
    """Clamp image intensities to a percentile range.

    Wraps
    :func:`~niiflow.preproc.functional.image.intensity_normalization.clamp_intensities`.

    **Parameters** (``params``):

    * ``image`` — :class:`ants.core.ANTsImage` or path to load.
    * ``lower_pct`` — lower percentile bound in ``[0, 100]`` (default ``1.0``).
    * ``upper_pct`` — upper percentile bound in ``[0, 100]`` (default ``99.0``).
    * ``limit_to`` — optional mask restricting statistics and clipping
      (:class:`ants.core.ANTsImage`, path, or ``None``).
    * ``non_zero`` — when ``True``, restrict both the percentile computation and
      the clipping to the non-zero voxels of the image; zero voxels are left
      untouched. Combined with ``limit_to`` by intersection (default ``False``).

    **Outputs** (from :meth:`forward`):

    * ``out_image`` — intensity-clamped image.

    **Persistence** (``save_outputs``):

    * ``out_image`` — NIfTI path (``.nii`` or ``.nii.gz``).
    """

    REQUIRED_PARAMS = frozenset({"image"})

    def load_param(self, key: str, value: Any) -> Any:
        if key == "image":
            if isinstance(value, ANTsImage):
                return value
            try:
                return ants_image_read(value, reorient=True)
            except Exception as e:
                raise ValueError(f"Failed to read image from {value}") from e

        if key == "limit_to":
            if value is None:
                return None
            if isinstance(value, ANTsImage):
                return value
            try:
                return ants_image_read(value, reorient=True)
            except Exception as e:
                raise ValueError(f"Failed to read limit_to mask from {value}") from e

        return value

    def forward(self, **params: Any) -> dict[str, Any]:
        return {"out_image": clamp_intensities(**params)}

    def save_output(self, key: str, value: Any, output_path: Path) -> Path:
        if key == "out_image":
            return ants_image_write(value, output_path)
        raise KeyError(f"Unknown output key {key!r} for {type(self).__name__}")


class ZTransformNorm(PipelineStage):
    """Z-score normalize an image.

    Wraps
    :func:`~niiflow.preproc.functional.image.intensity_normalization.z_transform_norm`.

    **Parameters** (``params``):

    * ``image`` — :class:`ants.core.ANTsImage` or path to load.
    * ``limit_to`` — optional mask restricting mean/std computation
      (:class:`ants.core.ANTsImage`, path, or ``None``).
    * ``non_zero`` — when ``True``, compute mean/std from the non-zero voxels and
      apply the transform only to those voxels; zero voxels remain zero. Combined
      with ``limit_to`` by intersection for the statistics (default ``False``).

    **Outputs** (from :meth:`forward`):

    * ``out_image`` — z-transformed image.

    **Persistence** (``save_outputs``):

    * ``out_image`` — NIfTI path (``.nii`` or ``.nii.gz``).
    """

    REQUIRED_PARAMS = frozenset({"image"})

    def load_param(self, key: str, value: Any) -> Any:
        if key == "image":
            if isinstance(value, ANTsImage):
                return value
            try:
                return ants_image_read(value, reorient=True)
            except Exception as e:
                raise ValueError(f"Failed to read image from {value}") from e

        if key == "limit_to":
            if value is None:
                return None
            if isinstance(value, ANTsImage):
                return value
            try:
                return ants_image_read(value, reorient=True)
            except Exception as e:
                raise ValueError(f"Failed to read limit_to mask from {value}") from e

        return value

    def forward(self, **params: Any) -> dict[str, Any]:
        return {"out_image": z_transform_norm(**params)}

    def save_output(self, key: str, value: Any, output_path: Path) -> Path:
        if key == "out_image":
            return ants_image_write(value, output_path)
        raise KeyError(f"Unknown output key {key!r} for {type(self).__name__}")


class MinmaxNorm(PipelineStage):
    """Min-max normalize an image to ``[0, 1]``.

    Wraps
    :func:`~niiflow.preproc.functional.image.intensity_normalization.minmax_norm`.

    **Parameters** (``params``):

    * ``image`` — :class:`ants.core.ANTsImage` or path to load.
    * ``limit_to`` — optional mask restricting min/max computation
      (:class:`ants.core.ANTsImage`, path, or ``None``).
    * ``non_zero`` — when ``True``, compute min/max from the non-zero voxels and
      apply the transform only to those voxels; zero voxels remain zero. Combined
      with ``limit_to`` by intersection for the statistics (default ``False``).

    **Outputs** (from :meth:`forward`):

    * ``out_image`` — min-max normalized image.

    **Persistence** (``save_outputs``):

    * ``out_image`` — NIfTI path (``.nii`` or ``.nii.gz``).
    """

    REQUIRED_PARAMS = frozenset({"image"})

    def load_param(self, key: str, value: Any) -> Any:
        if key == "image":
            if isinstance(value, ANTsImage):
                return value
            try:
                return ants_image_read(value, reorient=True)
            except Exception as e:
                raise ValueError(f"Failed to read image from {value}") from e

        if key == "limit_to":
            if value is None:
                return None
            if isinstance(value, ANTsImage):
                return value
            try:
                return ants_image_read(value, reorient=True)
            except Exception as e:
                raise ValueError(f"Failed to read limit_to mask from {value}") from e

        return value

    def forward(self, **params: Any) -> dict[str, Any]:
        return {"out_image": minmax_norm(**params)}

    def save_output(self, key: str, value: Any, output_path: Path) -> Path:
        if key == "out_image":
            return ants_image_write(value, output_path)
        raise KeyError(f"Unknown output key {key!r} for {type(self).__name__}")
