"""Pipeline stages wrapping :mod:`niiflow.preproc.functional.image.resampling`."""

from __future__ import annotations

__all__ = [
    "ANTsResample",
    "ANTsResampleToTarget",
]

from pathlib import Path
from typing import Any

from ants.core import ANTsImage

from niiflow.preproc.functional.image.resampling import (
    ants_resample,
    ants_resample_to_target,
)
from niiflow.preproc.pipelines.pipeline_stages.pipeline_stage import PipelineStage
from niiflow.preproc.utils.file import ants_image_read, ants_image_write


class ANTsResample(PipelineStage):
    """Resample an image by target spacing or voxel count.

    Wraps :func:`~niiflow.preproc.functional.image.resampling.ants_resample`.

    **Parameters** (``params``):

    * ``image`` — :class:`ants.core.ANTsImage` or path to load.
    * ``resample_params`` — one value per spatial dimension; target spacing
      (default) or voxel counts when ``use_voxels`` is true.
    * ``use_voxels`` — interpret ``resample_params`` as voxel counts.
    * ``interpolation`` — interpolation mode forwarded to ``ants_resample``.

    **Outputs** (from :meth:`forward`):

    * ``out_image`` — resampled image.

    **Persistence** (``save_outputs``):

    * ``out_image`` — NIfTI path (``.nii`` or ``.nii.gz``).
    """

    REQUIRED_PARAMS = frozenset({"image", "resample_params"})

    def load_param(self, key: str, value: Any) -> Any:
        if key == "image":
            if isinstance(value, ANTsImage):
                return value
            try:
                return ants_image_read(value, reorient=True)
            except Exception as e:
                raise ValueError(f"Failed to read image from {value}") from e
        return value

    def forward(self, **params: Any) -> dict[str, Any]:
        return {"out_image": ants_resample(**params)}

    def save_output(self, key: str, value: Any, output_path: Path) -> Path:
        """Persist the resampled image to ``output_path``.

        Only ``out_image`` is supported. ``output_path`` must end with ``.nii`` or
        ``.nii.gz``; returns the resolved image path.
        """
        if key == "out_image":
            return ants_image_write(value, output_path)
        raise KeyError(f"Unknown output key {key!r} for {type(self).__name__}")


class ANTsResampleToTarget(PipelineStage):
    """Resample an image into the grid of a target image.

    Wraps
    :func:`~niiflow.preproc.functional.image.resampling.ants_resample_to_target`.
    The output is defined on ``target``'s grid (origin, spacing, direction and
    shape) and inherits its pixel type.

    **Parameters** (``params``):

    * ``image`` — :class:`ants.core.ANTsImage` or path to load.
    * ``target`` — :class:`ants.core.ANTsImage` or path whose grid defines the
      output space.
    * ``interpolation`` — interpolation mode forwarded to
      ``ants_resample_to_target``.
    * Additional kwargs are forwarded to ``ants_resample_to_target`` (e.g.
      ``imagetype``, ``verbose``).

    **Outputs** (from :meth:`forward`):

    * ``out_image`` — image resampled onto ``target``'s grid.

    **Persistence** (``save_outputs``):

    * ``out_image`` — NIfTI path (``.nii`` or ``.nii.gz``).
    """

    REQUIRED_PARAMS = frozenset({"image", "target"})

    def load_param(self, key: str, value: Any) -> Any:
        if key == "image":
            if isinstance(value, ANTsImage):
                return value
            try:
                return ants_image_read(value, reorient=True)
            except Exception as e:
                raise ValueError(f"Failed to read image from {value}") from e

        if key == "target":
            if isinstance(value, ANTsImage):
                return value
            try:
                return ants_image_read(value, reorient=True)
            except Exception as e:
                raise ValueError(f"Failed to read target image from {value}") from e
        return value

    def forward(self, **params: Any) -> dict[str, Any]:
        return {"out_image": ants_resample_to_target(**params)}

    def save_output(self, key: str, value: Any, output_path: Path) -> Path:
        """Persist the resampled image to ``output_path``.

        Only ``out_image`` is supported. ``output_path`` must end with ``.nii`` or
        ``.nii.gz``; returns the resolved image path.
        """
        if key == "out_image":
            return ants_image_write(value, output_path)
        raise KeyError(f"Unknown output key {key!r} for {type(self).__name__}")
