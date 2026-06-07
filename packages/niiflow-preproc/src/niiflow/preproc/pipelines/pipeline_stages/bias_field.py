"""Pipeline stage for bias field correction of ANTsImage objects."""

from __future__ import annotations

__all__ = [
    "ANTsBiasFieldCorrection",
]

from typing import Any
from pathlib import Path

from ants.core import ANTsImage

from niiflow.preproc.functional.image.bias_field import ants_bias_field_correction
from niiflow.preproc.pipelines.pipeline_stages.pipeline_stage import PipelineStage
from niiflow.preproc.utils.file import ants_image_read, ants_image_write, get_ext


class ANTsBiasFieldCorrection(PipelineStage):
    """Correct intensity inhomogeneity with ANTs N4 bias field correction.

    Wraps
    :func:`~niiflow.preproc.functional.image.bias_field.ants_bias_field_correction`.

    **Parameters** (``params``):

    * ``image`` — :class:`ants.core.ANTsImage` or path to load.
    * ``mask`` — optional brain/foreground mask (:class:`ants.core.ANTsImage`
      or path).
    * ``weight_mask`` — optional per-voxel weights (:class:`ants.core.ANTsImage`
      or path).
    * Additional kwargs are forwarded to ``ants_bias_field_correction``.

    **Outputs** (from :meth:`forward`):

    * ``out_image`` — bias-corrected image.

    **Persistence** (``save_options``):

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

        if key == "mask":
            if isinstance(value, ANTsImage):
                return value
            try:
                return ants_image_read(value, reorient=True)
            except Exception as e:
                raise ValueError(f"Failed to read mask from {value}") from e

        if key == "weight_mask":
            if isinstance(value, ANTsImage):
                return value
            try:
                return ants_image_read(value, reorient=True)
            except Exception as e:
                raise ValueError(f"Failed to read weight mask from {value}") from e
        return value

    def forward(self, **params: Any) -> dict[str, Any]:
        return {"out_image": ants_bias_field_correction(**params)}

    def save_output(self, key: str, value: Any, output_path: Path) -> Path:
        """Persist the bias-corrected image to ``output_path``.

        Only ``out_image`` is supported. ``output_path`` must end with ``.nii`` or
        ``.nii.gz``; returns the resolved image path.
        """
        if key == "out_image":
            if get_ext(output_path) not in (".nii.gz", ".nii"):
                raise ValueError(
                    f"Output path for {key!r} must end with .nii.gz or .nii, got {output_path!s}"
                )
            return ants_image_write(value, output_path)
        raise KeyError(f"Unknown output key {key!r} for {type(self).__name__}")
