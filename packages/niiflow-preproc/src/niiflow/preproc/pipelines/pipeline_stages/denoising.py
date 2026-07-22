"""Pipeline stage for denoising ANTsImage objects."""

from __future__ import annotations

__all__ = [
    "ANTsDenoise",
]

from pathlib import Path
from typing import Any

from ants.core import ANTsImage

from niiflow.preproc.functional.image.denoising import ants_denoise
from niiflow.preproc.pipelines.pipeline_stages.pipeline_stage import PipelineStage
from niiflow.preproc.utils.file import ants_image_read, ants_image_write


class ANTsDenoise(PipelineStage):
    """Denoise an image with ANTs adaptive non-local means.

    Wraps :func:`~niiflow.preproc.functional.image.denoising.ants_denoise`.

    **Parameters** (``params``):

    * ``image`` — :class:`ants.core.ANTsImage` or path to load.
    * ``mask`` — optional mask limiting the denoising region
      (:class:`ants.core.ANTsImage` or path).
    * Additional kwargs are forwarded to ``ants_denoise``.

    **Outputs** (from :meth:`forward`):

    * ``out_image`` — denoised image.

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
        return value

    def forward(self, **params: Any) -> dict[str, Any]:
        return {"out_image": ants_denoise(**params)}

    def save_output(self, key: str, value: Any, output_path: Path) -> Path:
        """Persist the denoised image to ``output_path``.

        Only ``out_image`` is supported. ``output_path`` must end with ``.nii`` or
        ``.nii.gz``; returns the resolved image path.
        """
        if key == "out_image":
            return ants_image_write(value, output_path)
        raise KeyError(f"Unknown output key {key!r} for {type(self).__name__}")
