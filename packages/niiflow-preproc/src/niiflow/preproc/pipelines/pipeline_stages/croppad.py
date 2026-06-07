"""Pipeline stages wrapping :mod:`niiflow.preproc.functional.image.croppad`."""

from __future__ import annotations

__all__ = [
    "CenterCrop",
    "CenterPad",
    "CropToMask",
    "CropToRange",
    "PadToRange",
]

from pathlib import Path
from typing import Any

from ants.core import ANTsImage

from niiflow.preproc.functional.image.croppad import (
    center_crop,
    center_pad,
    crop_to_mask,
    crop_to_range,
    pad_to_range,
)
from niiflow.preproc.pipelines.pipeline_stages.pipeline_stage import PipelineStage
from niiflow.preproc.utils.file import (
    ants_image_read,
    ants_image_write,
    get_ext,
    read_json,
    write_json,
)


class CropToRange(PipelineStage):
    """Crop an image to explicit per-axis voxel ranges.

    Wraps :func:`~niiflow.preproc.functional.image.croppad.crop_to_range`.

    **Parameters** (``params``):

    * ``image`` — :class:`ants.core.ANTsImage` or path to load.
    * ``ranges`` — per-axis ``(start, stop)`` pairs, or a path to a JSON file
      containing the range specification.

    **Outputs** (from :meth:`forward`):

    * ``out_image`` — cropped image.

    **Persistence** (``save_options``):

    * ``out_image`` — NIfTI path (``.nii`` or ``.nii.gz``).
    """

    REQUIRED_PARAMS = frozenset({"image", "ranges"})

    def load_param(self, key: str, value: Any) -> Any:
        if key == "image":
            if isinstance(value, ANTsImage):
                return value
            try:
                return ants_image_read(value, reorient=True)
            except Exception as e:
                raise ValueError(f"Failed to read image from {value}") from e

        if key == "ranges":
            if isinstance(value, (Path, str)):
                try:
                    return read_json(value)
                except Exception as e:
                    raise ValueError(f"Failed to read ranges from {value}") from e
        return value

    def forward(self, **params: Any) -> dict[str, Any]:
        return {"out_image": crop_to_range(**params)}

    def save_output(self, key: str, value: Any, output_path: Path) -> Path:
        """Persist the cropped image to ``output_path``.

        Only ``out_image`` is supported; returns the resolved image path.
        """
        if key == "out_image":
            return ants_image_write(value, output_path)
        raise KeyError(f"Unknown output key {key!r} for {type(self).__name__}")


class CropToMask(PipelineStage):
    """Crop an image to the bounding box of a mask.

    Wraps :func:`~niiflow.preproc.functional.image.croppad.crop_to_mask`.

    **Parameters** (``params``):

    * ``image`` — :class:`ants.core.ANTsImage` or path to load.
    * ``mask`` — optional mask (:class:`ants.core.ANTsImage`, path, or
      ``None`` to use the image itself).
    * ``pad`` — optional voxel padding around the bounding box.

    **Outputs** (from :meth:`forward`):

    * ``out_image`` — cropped image.
    * ``ranges`` — per-axis ``(start, stop)`` tuple applied by the crop.

    **Persistence** (``save_options``):

    * ``out_image`` — NIfTI path (``.nii`` or ``.nii.gz``).
    * ``ranges`` — JSON path for the crop specification.
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
            if value is None:
                return None
            if isinstance(value, ANTsImage):
                return value
            try:
                return ants_image_read(value, reorient=True)
            except Exception as e:
                raise ValueError(f"Failed to read mask from {value}") from e

        return value

    def forward(self, **params: Any) -> dict[str, Any]:
        out_image, ranges = crop_to_mask(**params)
        return {"out_image": out_image, "ranges": ranges}

    def save_output(self, key: str, value: Any, output_path: Path) -> Path:
        """Persist ``out_image`` or ``ranges`` to ``output_path``.

        ``out_image`` requires a NIfTI extension; ``ranges`` requires ``.json``.
        """
        if key == "out_image":
            if get_ext(output_path) not in (".nii.gz", ".nii"):
                raise ValueError(
                    f"Output path for {key!r} must end with .nii.gz or .nii, got {output_path!s}"
                )
            return ants_image_write(value, output_path)
        if key == "ranges":
            if get_ext(output_path) != ".json":
                raise ValueError(
                    f"Output path for {key!r} must end with .json, got {output_path!s}"
                )
            return write_json(value, output_path)
        raise KeyError(f"Unknown output key {key!r} for {type(self).__name__}")


class CenterCrop(PipelineStage):
    """Center-crop an image to a target shape.

    Wraps :func:`~niiflow.preproc.functional.image.croppad.center_crop`.

    **Parameters** (``params``):

    * ``image`` — :class:`ants.core.ANTsImage` or path to load.
    * ``shape`` — target spatial shape.
    * ``mask`` — optional mask used to centre the crop (:class:`ants.core.ANTsImage`,
      path, or ``None``).

    **Outputs** (from :meth:`forward`):

    * ``out_image`` — center-cropped image.
    * ``ranges`` — per-axis ``(start, stop)`` tuple applied by the crop.

    **Persistence** (``save_options``):

    * ``out_image`` — NIfTI path (``.nii`` or ``.nii.gz``).
    * ``ranges`` — JSON path for the crop specification.
    """

    REQUIRED_PARAMS = frozenset({"image", "shape"})

    def load_param(self, key: str, value: Any) -> Any:
        if key == "image":
            if isinstance(value, ANTsImage):
                return value
            try:
                return ants_image_read(value, reorient=True)
            except Exception as e:
                raise ValueError(f"Failed to read image from {value}") from e

        if key == "mask":
            if value is None:
                return None
            if isinstance(value, ANTsImage):
                return value
            try:
                return ants_image_read(value, reorient=True)
            except Exception as e:
                raise ValueError(f"Failed to read mask from {value}") from e
        return value

    def forward(self, **params: Any) -> dict[str, Any]:
        out_image, ranges = center_crop(**params)
        return {"out_image": out_image, "ranges": ranges}

    def save_output(self, key: str, value: Any, output_path: Path) -> Path:
        """Persist ``out_image`` or ``ranges`` to ``output_path``.

        ``out_image`` requires a NIfTI extension; ``ranges`` requires ``.json``.
        """
        if key == "out_image":
            if get_ext(output_path) not in (".nii.gz", ".nii"):
                raise ValueError(
                    f"Output path for {key!r} must end with .nii.gz or .nii, got {output_path!s}"
                )
            return ants_image_write(value, output_path)
        if key == "ranges":
            if get_ext(output_path) != ".json":
                raise ValueError(
                    f"Output path for {key!r} must end with .json, got {output_path!s}"
                )
            return write_json(value, output_path)
        raise KeyError(f"Unknown output key {key!r} for {type(self).__name__}")


class PadToRange(PipelineStage):
    """Pad an image by explicit per-axis before/after widths.

    Wraps :func:`~niiflow.preproc.functional.image.croppad.pad_to_range`.

    **Parameters** (``params``):

    * ``image`` — :class:`ants.core.ANTsImage` or path to load.
    * ``ranges`` — per-axis ``(before, after)`` padding widths, or a path to a
      JSON file containing the specification.
    * ``mode``, ``constant_values`` — padding mode forwarded to the functional.

    **Outputs** (from :meth:`forward`):

    * ``out_image`` — padded image.

    **Persistence** (``save_options``):

    * ``out_image`` — NIfTI path (``.nii`` or ``.nii.gz``).
    """

    REQUIRED_PARAMS = frozenset({"image", "ranges"})

    def load_param(self, key: str, value: Any) -> Any:
        if key == "image":
            if isinstance(value, ANTsImage):
                return value
            try:
                return ants_image_read(value, reorient=True)
            except Exception as e:
                raise ValueError(f"Failed to read image from {value}") from e

        if key == "ranges":
            if isinstance(value, (Path, str)):
                try:
                    return read_json(value)
                except Exception as e:
                    raise ValueError(f"Failed to read ranges from {value}") from e
        return value

    def forward(self, **params: Any) -> dict[str, Any]:
        return {"out_image": pad_to_range(**params)}

    def save_output(self, key: str, value: Any, output_path: Path) -> Path:
        """Persist the padded image to ``output_path``.

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


class CenterPad(PipelineStage):
    """Center-pad an image to a target shape.

    Wraps :func:`~niiflow.preproc.functional.image.croppad.center_pad`.

    **Parameters** (``params``):

    * ``image`` — :class:`ants.core.ANTsImage` or path to load.
    * ``shape`` — target spatial shape.
    * ``mask`` — optional mask used to centre the padding (:class:`ants.core.ANTsImage`,
      path, or ``None``).
    * ``mode``, ``constant_values`` — padding mode forwarded to the functional.

    **Outputs** (from :meth:`forward`):

    * ``out_image`` — center-padded image.
    * ``ranges`` — per-axis ``(before, after)`` padding applied.

    **Persistence** (``save_options``):

    * ``out_image`` — NIfTI path (``.nii`` or ``.nii.gz``).
    * ``ranges`` — JSON path for the padding specification.
    """

    REQUIRED_PARAMS = frozenset({"image", "shape"})

    def load_param(self, key: str, value: Any) -> Any:
        if key == "image":
            if isinstance(value, ANTsImage):
                return value
            try:
                return ants_image_read(value, reorient=True)
            except Exception as e:
                raise ValueError(f"Failed to read image from {value}") from e

        if key == "mask":
            if value is None:
                return None
            if isinstance(value, ANTsImage):
                return value
            try:
                return ants_image_read(value, reorient=True)
            except Exception as e:
                raise ValueError(f"Failed to read mask from {value}") from e
        return value

    def forward(self, **params: Any) -> dict[str, Any]:
        out_image, ranges = center_pad(**params)
        return {"out_image": out_image, "ranges": ranges}

    def save_output(self, key: str, value: Any, output_path: Path) -> Path:
        """Persist ``out_image`` or ``ranges`` to ``output_path``.

        ``out_image`` requires a NIfTI extension; ``ranges`` requires ``.json``.
        """
        if key == "out_image":
            if get_ext(output_path) not in (".nii.gz", ".nii"):
                raise ValueError(
                    f"Output path for {key!r} must end with .nii.gz or .nii, got {output_path!s}"
                )
            return ants_image_write(value, output_path)
        if key == "ranges":
            if get_ext(output_path) != ".json":
                raise ValueError(
                    f"Output path for {key!r} must end with .json, got {output_path!s}"
                )
            return write_json(value, output_path)
        raise KeyError(f"Unknown output key {key!r} for {type(self).__name__}")
