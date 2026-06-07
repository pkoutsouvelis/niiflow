"""Pipeline stages for utility operations."""

from __future__ import annotations

__all__ = [
    "Delete",
    "Reorient",
    "ToNumpy",
]

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ants.core import ANTsImage

from niiflow.preproc.functional.image.reorientation import ants_reorient

from niiflow.preproc.functional.image.utils import ants_to_numpy_with_metadata
from niiflow.preproc.pipelines.pipeline_stages.pipeline_stage import PipelineStage
from niiflow.preproc.utils.file import (
    ants_image_read,
    ants_image_write,
    delete_paths,
    get_ext,
    resolve_path,
    write_json,
    write_npy,
)
from niiflow.preproc.utils.misc import resolve_numpy_dtype


class Delete(PipelineStage):
    """Delete one or more files from disk.

    **Parameters** (``params``):

    * ``paths`` — a single file path, a list of paths, or a ``ctx.`` reference
      resolving to either form.
    * ``missing_ok`` — when ``True``, skip paths that do not exist instead of
      raising :class:`FileNotFoundError` (default ``False``).

    **Outputs** (from :meth:`forward`):

    * ``deleted`` — list of resolved :class:`pathlib.Path` objects removed.

    **Persistence** (``save_options``):

    * ``deleted`` — optional JSON path recording the deleted paths as strings.
    """

    REQUIRED_PARAMS = frozenset({"paths"})

    def check_params(self, params: dict[str, Any]) -> None:
        paths = params["paths"]
        if isinstance(paths, (str, Path)):
            return
        if isinstance(paths, Sequence) and not isinstance(paths, (str, bytes)):
            if not paths:
                raise ValueError("`paths` must not be an empty sequence")
            return
        raise TypeError(
            f"`paths` must be a path or sequence of paths, got {type(paths).__name__}"
        )

    def load_param(self, key: str, value: Any) -> Any:
        if key == "paths":
            if isinstance(value, (str, Path)):
                return [resolve_path(value)]
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                return [resolve_path(path) for path in value]
            raise TypeError(
                f"`paths` must be a path or sequence of paths, "
                f"got {type(value).__name__}"
            )
        if key == "missing_ok":
            if not isinstance(value, bool):
                raise TypeError(
                    f"`missing_ok` must be a boolean, got {type(value).__name__}"
                )
            return value
        return value

    def forward(self, **params: Any) -> dict[str, Any]:
        return {
            "deleted": delete_paths(
                params["paths"],
                missing_ok=params.get("missing_ok", False),
            )
        }

    def save_output(self, key: str, value: Any, output_path: Path) -> Path:
        if key == "deleted":
            if not isinstance(value, list):
                raise TypeError(
                    f"`deleted` must be a list of paths, got {type(value).__name__}"
                )
            return write_json({"paths": [str(path) for path in value]}, output_path)
        raise KeyError(f"Unknown output key {key!r} for {type(self).__name__}")


class Reorient(PipelineStage):
    """Reorient a 3D image to a target orientation code.

    Wraps :func:`~niiflow.preproc.functional.image.reorientation.ants_reorient`.
    Images loaded from disk via
    :func:`~niiflow.preproc.utils.file.ants_image_read` (with ``reorient=True``)
    are already canonicalized to ITK **RPI**; use this stage when a different
    target orientation is required (for example before writing NIfTI, or for
    in-memory images passed through ``ctx.artifacts``).

    **Parameters** (``params``):

    * ``image`` — :class:`ants.core.ANTsImage` or path to load.
    * ``orientation`` — three-letter code such as ``"RPI"`` (ITK) or ``"RAS"``
      (nibabel); default ``"RPI"``.
    * ``convention`` — ``"itk"`` (default) or ``"nibabel"``.

    **Outputs** (from :meth:`forward`):

    * ``out_image`` — reoriented image.

    **Persistence** (``save_options``):

    * ``out_image`` — NIfTI path (``.nii`` or ``.nii.gz``).
    """

    REQUIRED_PARAMS = frozenset({"image", "orientation"})

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
        return {"out_image": ants_reorient(**params)}

    def save_output(self, key: str, value: Any, output_path: Path) -> Path:
        if key == "out_image":
            if get_ext(output_path) not in (".nii.gz", ".nii"):
                raise ValueError(
                    f"Output path for {key!r} must end with .nii.gz or .nii, "
                    f"got {output_path!s}"
                )
            return ants_image_write(value, output_path)
        raise KeyError(f"Unknown output key {key!r} for {type(self).__name__}")


class ToNumpy(PipelineStage):
    """Convert an ANTs image to a NumPy array.

    **Parameters** (``params``):

    * ``image`` — path to an image readable by ANTs, or an in-memory
      :class:`ants.core.ANTsImage`.
    * ``dtype`` — target array dtype as a string alias (e.g. ``"float32"``) or
      any string accepted by :func:`numpy.dtype`.

    **Outputs** (from :meth:`forward`):

    * ``array`` — voxel data cast to ``dtype``.
    * ``metadata`` — orientation metadata from
      :func:`~niiflow.preproc.functional.image.utils.ants_to_numpy_with_metadata`.

    **Persistence** (``save_options``):

    * ``array`` — ``.npy`` path.
    * ``metadata`` — JSON path.
    """

    REQUIRED_PARAMS = frozenset({"image", "dtype"})

    def load_param(self, key: str, value: Any) -> Any:
        if key == "image":
            if isinstance(value, ANTsImage):
                return value
            return ants_image_read(value, reorient=True)
        if key == "dtype":
            return resolve_numpy_dtype(value)
        return value

    def forward(self, **params: Any) -> dict[str, Any]:
        array, meta = ants_to_numpy_with_metadata(params["image"])
        return {
            "array": array.astype(params["dtype"]),
            "metadata": meta,
        }

    def save_output(self, key: str, value: Any, output_path: Path) -> Path:
        if key == "array":
            return write_npy(value, output_path)
        if key == "metadata":
            return write_json(value, output_path)
        raise KeyError(f"Unknown output key {key!r} for {type(self).__name__}")
