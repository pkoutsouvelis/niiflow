"""Pipeline stages for mask operations."""

from __future__ import annotations

__all__ = [
    "ApplyMask",
    "RelabelMask",
    "SmoothMask",
]

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ants.core import ANTsImage

from niiflow.preproc.functional.image.masks import (
    ants_apply_mask,
    relabel_mask,
    smooth_mask,
)
from niiflow.preproc.pipelines.pipeline_stages.pipeline_stage import PipelineStage
from niiflow.preproc.utils.file import ants_image_read, ants_image_write
from niiflow.preproc.utils.misc import resolve_numpy_dtype


class ApplyMask(PipelineStage):
    """Apply a mask to an image.

    Wraps :func:`~niiflow.preproc.functional.image.masks.ants_apply_mask`.

    **Parameters** (``params``):

    * ``image`` — :class:`ants.core.ANTsImage` or path to load.
    * ``mask`` — mask or label image (:class:`ants.core.ANTsImage` or path).
    * Additional kwargs are forwarded to ``ants_apply_mask`` (e.g. ``level``,
      ``binarize``).

    **Outputs** (from :meth:`forward`):

    * ``out_image`` — masked image.

    **Persistence** (``save_outputs``):

    * ``out_image`` — NIfTI path (``.nii`` or ``.nii.gz``).
    """

    REQUIRED_PARAMS = frozenset({"image", "mask"})

    def load_param(self, key: str, value: Any) -> Any:
        if key in {"image", "mask"}:
            if isinstance(value, ANTsImage):
                return value
            try:
                return ants_image_read(value, reorient=True)
            except Exception as e:
                raise ValueError(f"Failed to read {key} from {value}") from e
        return value

    def forward(self, **params: Any) -> dict[str, Any]:
        return {"out_image": ants_apply_mask(**params)}

    def save_output(self, key: str, value: Any, output_path: Path) -> Path:
        if key == "out_image":
            return ants_image_write(value, output_path)
        raise KeyError(f"Unknown output key {key!r} for {type(self).__name__}")


class SmoothMask(PipelineStage):
    """Gaussian-smooth a binary mask and renormalize to ``[0, 1]``.

    Wraps :func:`~niiflow.preproc.functional.image.masks.smooth_mask`.

    **Parameters** (``params``):

    * ``mask`` — binary mask (:class:`ants.core.ANTsImage` or path).
    * ``sigma`` — Gaussian standard deviation (default ``1.0``).
    * ``threshold`` — optional re-binarization cutoff after renormalization.

    **Outputs** (from :meth:`forward`):

    * ``out_mask`` — smoothed (and optionally thresholded) mask.

    **Persistence** (``save_outputs``):

    * ``out_mask`` — NIfTI path (``.nii`` or ``.nii.gz``).
    """

    REQUIRED_PARAMS = frozenset({"mask"})

    def load_param(self, key: str, value: Any) -> Any:
        if key == "mask":
            if isinstance(value, ANTsImage):
                return value
            try:
                return ants_image_read(value, reorient=True)
            except Exception as e:
                raise ValueError(f"Failed to read mask from {value}") from e
        return value

    def forward(self, **params: Any) -> dict[str, Any]:
        return {"out_mask": smooth_mask(**params)}

    def save_output(self, key: str, value: Any, output_path: Path) -> Path:
        if key == "out_mask":
            return ants_image_write(value, output_path)
        raise KeyError(f"Unknown output key {key!r} for {type(self).__name__}")


class RelabelMask(PipelineStage):
    """Remap integer labels in a mask according to a source→target mapping.

    Wraps :func:`~niiflow.preproc.functional.image.masks.relabel_mask`.

    **Parameters** (``params``):

    * ``mask`` — label mask (:class:`ants.core.ANTsImage` or path).
    * ``mapping`` — dict of source label → target label.
    * ``dtype`` — optional output dtype string (e.g. ``"uint8"``, ``"int32"``).
    * ``unmapped`` — ``"keep"`` (default), ``"zero"``, or ``"raise"``.
    * ``allow_collisions`` — allow many-to-one mappings (default ``False``).

    **Outputs** (from :meth:`forward`):

    * ``out_mask`` — relabeled mask.

    **Persistence** (``save_outputs``):

    * ``out_mask`` — NIfTI path (``.nii`` or ``.nii.gz``).
    """

    REQUIRED_PARAMS = frozenset({"mask", "mapping"})

    def check_params(self, params: dict[str, Any]) -> None:
        mapping = params["mapping"]
        if not isinstance(mapping, Mapping):
            raise TypeError(
                f"`mapping` must be a mapping, got {type(mapping).__name__}"
            )
        if "unmapped" in params and params["unmapped"] not in {
            "keep",
            "zero",
            "raise",
        }:
            raise ValueError(
                f"`unmapped` must be one of 'keep', 'zero', 'raise', got "
                f"{params['unmapped']!r}"
            )
        if "allow_collisions" in params and not isinstance(
            params["allow_collisions"], bool
        ):
            raise TypeError(
                f"`allow_collisions` must be a boolean, got "
                f"{type(params['allow_collisions']).__name__}"
            )

    def load_param(self, key: str, value: Any) -> Any:
        if key == "mask":
            if isinstance(value, ANTsImage):
                return value
            try:
                return ants_image_read(value, reorient=True)
            except Exception as e:
                raise ValueError(f"Failed to read mask from {value}") from e
        if key == "dtype" and value is not None:
            return resolve_numpy_dtype(value)
        return value

    def forward(self, **params: Any) -> dict[str, Any]:
        return {"out_mask": relabel_mask(**params)}

    def save_output(self, key: str, value: Any, output_path: Path) -> Path:
        if key == "out_mask":
            return ants_image_write(value, output_path)
        raise KeyError(f"Unknown output key {key!r} for {type(self).__name__}")
