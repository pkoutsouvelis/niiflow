"""Pipeline stage for skull-stripping."""

from __future__ import annotations

__all__ = [
    "ANTsBrainExtraction",
]

from typing import Any
from pathlib import Path

from ants.core import ANTsImage

from niiflow.preproc.functional.image.skull_stripping import ants_brain_extraction
from niiflow.preproc.pipelines.pipeline_stages.pipeline_stage import PipelineStage
from niiflow.preproc.utils.file import ants_image_read, ants_image_write


class ANTsBrainExtraction(PipelineStage):
    """Skull-strip a brain image with ANTsPyNet brain extraction.

    Wraps
    :func:`~niiflow.preproc.functional.image.skull_stripping.ants_brain_extraction`.

    **Parameters** (``params``):

    * ``image`` — :class:`ants.core.ANTsImage` or path to load.
    * Additional kwargs are forwarded to ``ants_brain_extraction`` (e.g.
      ``modality``, ``apply_mask``, ``verbose``).

    **Outputs** (from :meth:`forward`):

    * ``out_image`` — skull-stripped (or masked) brain image.
    * ``brain_mask`` — brain mask when the functional call returns a tuple.

    **Persistence** (``save_outputs``):

    * ``out_image`` / ``brain_mask`` — NIfTI paths (``.nii`` or ``.nii.gz``).
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

    def forward(self, **params: Any) -> dict[str, Any]:
        result = ants_brain_extraction(**params)
        if isinstance(result, ANTsImage):
            return {"out_image": result}
        out_image, mask = result
        return {"out_image": out_image, "brain_mask": mask}

    def save_output(self, key: str, value: Any, output_path: Path) -> Path:
        """Persist ``out_image`` or ``brain_mask`` to ``output_path``.

        Both keys require ``output_path`` to end with ``.nii`` or ``.nii.gz``. Returns
        the resolved image path.
        """
        if key in ("out_image", "brain_mask"):
            return ants_image_write(value, output_path)
        raise KeyError(f"Unknown output key {key!r} for {type(self).__name__}")
