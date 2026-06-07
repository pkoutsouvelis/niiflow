"""Pipeline stages wrapping :mod:`niiflow.preproc.functional.image.pipelines`."""

from __future__ import annotations

__all__ = [
    "ANTsPreprocessBrainImage",
]

from pathlib import Path
from typing import Any

from ants.core import ANTsImage

from niiflow.preproc.functional.image.pipelines import ants_preprocess_brain_image
from niiflow.preproc.pipelines.pipeline_stages.pipeline_stage import PipelineStage
from niiflow.preproc.utils.file import ants_image_read, ants_image_write, get_ext


class ANTsPreprocessBrainImage(PipelineStage):
    """Run the ANTsPyNet brain preprocessing pipeline as a single stage.

    Wraps
    :func:`~niiflow.preproc.functional.image.pipelines.ants_preprocess_brain_image`,
    which chains optional steps such as intensity truncation, skull-stripping,
    template registration, N4 bias correction, denoising, intensity matching,
    and normalization.

    **Parameters** (``params``):

    * ``image`` — :class:`ants.core.ANTsImage` or path to load.
    * ``reference_image`` — optional :class:`ants.core.ANTsImage` or path used
      for intensity matching.
    * Additional kwargs are forwarded to ``ants_preprocess_brain_image`` (e.g.
      ``truncate_intensity``, ``brain_extraction_modality``,
      ``template_transform_type``, ``template``, ``do_bias_correction``,
      ``do_denoising``, ``intensity_matching_type``,
      ``intensity_normalization_type``, ``verbose``). The ``template`` option
      may be a named template string and is passed through unchanged.

    **Outputs** (from :meth:`forward`):

    * ``out_image`` — preprocessed image.
    * Auxiliary outputs produced by the pipeline are flattened into the output
      mapping, e.g. ``brain_mask`` and ``bias_field`` (when the corresponding
      steps are enabled).

    **Persistence** (``save_options``):

    * ``out_image`` / ``brain_mask`` / ``bias_field`` — NIfTI paths (``.nii`` or
      ``.nii.gz``).
    """

    REQUIRED_PARAMS = frozenset({"image"})

    def check_params(self, params: dict[str, Any]) -> None:
        if "return_metadata" in params:
            raise ValueError(
                "`return_metadata` is internally set to True; cannot be set in `params`"
            )

    def load_param(self, key: str, value: Any) -> Any:
        if key == "image":
            if isinstance(value, ANTsImage):
                return value
            try:
                return ants_image_read(value, reorient=True)
            except Exception as e:
                raise ValueError(f"Failed to read image from {value}") from e

        if key == "reference_image":
            if value is None or isinstance(value, ANTsImage):
                return value
            try:
                return ants_image_read(value, reorient=True)
            except Exception as e:
                raise ValueError(f"Failed to read reference image from {value}") from e
        return value

    def forward(self, **params: Any) -> dict[str, Any]:
        out_image, metadata = ants_preprocess_brain_image(
            return_metadata=True, **params
        )
        return {"out_image": out_image, **metadata}

    def save_output(self, key: str, value: Any, output_path: Path) -> Path:
        """Persist an image output to ``output_path``.

        ``out_image``, ``brain_mask`` and ``bias_field`` are written as NIfTI;
        ``output_path`` must end with ``.nii`` or ``.nii.gz``.
        """
        if key in ("out_image", "brain_mask", "bias_field"):
            if get_ext(output_path) not in (".nii.gz", ".nii"):
                raise ValueError(
                    f"Output path for {key!r} must end with .nii.gz or .nii, "
                    f"got {output_path!s}"
                )
            return ants_image_write(value, output_path)
        raise KeyError(f"Unknown output key {key!r} for {type(self).__name__}")
