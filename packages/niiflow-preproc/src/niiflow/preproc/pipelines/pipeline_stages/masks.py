"""Pipeline stages for mask operations."""

from __future__ import annotations

__all__ = [
    "ANTsThreshold",
    "ApplyMask",
    "ComposeMasks",
    "RelabelMask",
    "SmoothMask",
]

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ants.core import ANTsImage

from niiflow.preproc.functional.image.masks import (
    ants_apply_mask,
    ants_threshold,
    compose_masks,
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


class ANTsThreshold(PipelineStage):
    """Threshold an image with ANTs.

    Wraps :func:`~niiflow.preproc.functional.image.masks.ants_threshold`, which
    forwards to :func:`ants.threshold_image` with an additional ``inclusive``
    flag. When enabled, a tiny epsilon is added to ``low_thresh`` so the lower
    bound is excluded; for example, ``low_thresh=0`` selects positive,
    non-zero voxels.

    **Parameters** (``params``):

    * ``image`` — :class:`ants.core.ANTsImage` or path to load.
    * ``inclusive`` — whether to exclude the exact lower-bound value (default
      ``False``).
    * Additional kwargs are forwarded to ``ants_threshold`` (e.g.
      ``low_thresh``, ``high_thresh``, ``inval``, ``outval``, ``binary``).

    **Outputs** (from :meth:`forward`):

    * ``out_mask`` — thresholded image.

    **Persistence** (``save_outputs``):

    * ``out_mask`` — NIfTI path (``.nii`` or ``.nii.gz``).
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
        return value

    def forward(self, **params: Any) -> dict[str, Any]:
        return {"out_mask": ants_threshold(**params)}

    def save_output(self, key: str, value: Any, output_path: Path) -> Path:
        if key == "out_mask":
            return ants_image_write(value, output_path)
        raise KeyError(f"Unknown output key {key!r} for {type(self).__name__}")


class ComposeMasks(PipelineStage):
    """Compose same-grid integer masks using logical OR or AND.

    Wraps :func:`~niiflow.preproc.functional.image.masks.compose_masks`.
    Overlapping non-zero labels must agree.

    **Parameters** (``params``):

    * ``masks`` — sequence of :class:`ants.core.ANTsImage` objects or paths.
    * ``operator`` — ``"OR"`` (default) or ``"AND"``.

    **Outputs** (from :meth:`forward`):

    * ``out_mask`` — composed mask.

    **Persistence** (``save_outputs``):

    * ``out_mask`` — NIfTI path (``.nii`` or ``.nii.gz``).
    """

    REQUIRED_PARAMS = frozenset({"masks"})

    def load_param(self, key: str, value: Any) -> Any:
        if key == "masks":
            if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
                raise TypeError(
                    f"`masks` must be a sequence, got {type(value).__name__}"
                )
            loaded: list[ANTsImage] = []
            for index, mask in enumerate(value):
                if isinstance(mask, ANTsImage):
                    loaded.append(mask)
                    continue
                try:
                    loaded.append(ants_image_read(mask, reorient=True))
                except Exception as e:
                    raise ValueError(
                        f"Failed to read masks[{index}] from {mask}"
                    ) from e
            return loaded
        return value

    def forward(self, **params: Any) -> dict[str, Any]:
        return {"out_mask": compose_masks(**params)}

    def save_output(self, key: str, value: Any, output_path: Path) -> Path:
        if key == "out_mask":
            return ants_image_write(value, output_path)
        raise KeyError(f"Unknown output key {key!r} for {type(self).__name__}")


class SmoothMask(PipelineStage):
    """Gaussian-smooth a binary mask and renormalize to ``[0, 1]``.

    Wraps :func:`~niiflow.preproc.functional.image.masks.smooth_mask`.

    **Parameters** (``params``):

    * ``mask`` — binary mask (:class:`ants.core.ANTsImage` or path).
    * ``sigma`` — Gaussian standard deviation (default ``1.0``). Interpreted
      in voxels unless ``use_mm`` is ``True``.
    * ``threshold`` — optional re-binarization cutoff after renormalization.
    * ``use_mm`` — when ``True``, treat ``sigma`` as millimetres and divide
      by the mask spacing per axis before smoothing (default ``False``).

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
