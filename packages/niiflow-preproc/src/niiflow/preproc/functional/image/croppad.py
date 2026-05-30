"""Cropping and padding functions for ANTsImage objects.

Each function mirrors its array-based counterpart in
:mod:`niiflow.preproc.functional.array.croppad`, with extra bookkeeping to keep the
resulting :class:`ants.core.ANTsImage`'s origin consistent with the cropping or padding
performed in voxel space. Spacing and direction are inherited from the input image
unchanged.
"""

from __future__ import annotations

__all__ = [
    "bbox_from_mask_ants",
    "crop_to_range_ants",
    "crop_to_mask_ants",
    "center_crop_ants",
    "pad_to_range_ants",
    "center_pad_ants",
]

from collections.abc import Sequence

from ants.core import ANTsImage

from niiflow.preproc.functional.array import (
    bbox_from_mask,
    center_crop,
    center_pad,
    crop_to_mask,
    crop_to_range,
    pad_to_range,
)

from .utils import (
    ants_to_numpy_with_metadata,
    ensure_ants_image,
    reconstruct_ants_image,
)


def _voxel_offset_from_ranges(
    ranges: Sequence[Sequence[int | None]],
    *,
    pad: bool = False,
) -> tuple[int, ...]:
    """Derive the per-axis voxel offset used to keep an image origin consistent.

    The ranges are taken straight from the array-layer crop/pad helpers (or
    from a user-supplied, already-validated range spec), so no further
    validation is performed here.

    * For crop ranges (``pad=False``), the offset is the ``start`` of each
      ``(start, stop)`` pair (a ``None`` start means ``0``).
    * For pad ranges (``pad=True``), the offset is ``-before`` for each
      ``(before, after)`` pair.
    """
    if pad:
        return tuple(-int(before) for before, _ in ranges)  # type: ignore[arg-type]
    return tuple(0 if start is None else int(start) for start, _ in ranges)


def bbox_from_mask_ants(mask: ANTsImage, pad: int = 0) -> tuple[slice, ...]:
    """Bounding box of an ANTsImage mask, as a tuple of voxel-index slices."""
    ensure_ants_image(mask, name="mask")
    return bbox_from_mask(mask.numpy(), pad=pad)


def crop_to_range_ants(
    image: ANTsImage,
    ranges: Sequence[Sequence[int | None]],
) -> ANTsImage:
    """Crop an ANTsImage to per-axis ``(start, stop)`` ranges."""
    image_array, _ = ants_to_numpy_with_metadata(image)
    out_array = crop_to_range(image_array, ranges)
    voxel_offset = _voxel_offset_from_ranges(ranges)
    return reconstruct_ants_image(out_array, image, voxel_offset=voxel_offset)


def crop_to_mask_ants(
    image: ANTsImage,
    mask: ANTsImage | None = None,
    pad: int = 0,
) -> tuple[ANTsImage, tuple[tuple[int, int], ...]]:
    """Crop an ANTsImage to the bounding box of `mask` (or of itself)."""
    image_array, _ = ants_to_numpy_with_metadata(image)
    mask_array = None
    if mask is not None:
        ensure_ants_image(mask, name="mask")
        mask_array = mask.numpy()

    out_array, crop_ranges = crop_to_mask(image_array, mask=mask_array, pad=pad)
    voxel_offset = _voxel_offset_from_ranges(crop_ranges)
    out_image = reconstruct_ants_image(out_array, image, voxel_offset=voxel_offset)
    return out_image, crop_ranges


def center_crop_ants(
    image: ANTsImage,
    shape: Sequence[int],
    mask: ANTsImage | None = None,
) -> tuple[ANTsImage, tuple[tuple[int, int], ...]]:
    """Center-crop an ANTsImage to `shape` (centered on image or mask)."""
    image_array, _ = ants_to_numpy_with_metadata(image)
    mask_array = None
    if mask is not None:
        ensure_ants_image(mask, name="mask")
        mask_array = mask.numpy()

    out_array, crop_ranges = center_crop(image_array, shape=shape, mask=mask_array)
    voxel_offset = _voxel_offset_from_ranges(crop_ranges)
    out_image = reconstruct_ants_image(out_array, image, voxel_offset=voxel_offset)
    return out_image, crop_ranges


def pad_to_range_ants(
    image: ANTsImage,
    ranges: Sequence[Sequence[int]],
    mode: str = "constant",
    constant_values: float | int = 0,
) -> ANTsImage:
    """Pad an ANTsImage by per-axis ``(before, after)`` widths."""
    image_array, _ = ants_to_numpy_with_metadata(image)
    out_array = pad_to_range(
        image_array,
        ranges,
        mode=mode,
        constant_values=constant_values,
    )
    voxel_offset = _voxel_offset_from_ranges(ranges, pad=True)
    return reconstruct_ants_image(out_array, image, voxel_offset=voxel_offset)


def center_pad_ants(
    image: ANTsImage,
    shape: Sequence[int],
    mask: ANTsImage | None = None,
    mode: str = "constant",
    constant_values: float | int = 0,
) -> tuple[ANTsImage, tuple[tuple[int, int], ...]]:
    """Center-pad an ANTsImage to `shape` (centered on image or mask)."""
    image_array, _ = ants_to_numpy_with_metadata(image)
    mask_array = None
    if mask is not None:
        ensure_ants_image(mask, name="mask")
        mask_array = mask.numpy()

    out_array, pad_ranges = center_pad(
        image_array,
        shape=shape,
        mask=mask_array,
        mode=mode,
        constant_values=constant_values,
    )
    voxel_offset = _voxel_offset_from_ranges(pad_ranges, pad=True)
    out_image = reconstruct_ants_image(out_array, image, voxel_offset=voxel_offset)
    return out_image, pad_ranges
