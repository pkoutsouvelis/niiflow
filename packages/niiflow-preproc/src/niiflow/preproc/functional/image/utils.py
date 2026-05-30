"""Shared utility helpers for ANTsImage functional modules."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import ants
import numpy as np
from ants.core import ANTsImage

from niiflow.preproc.functional.array.utils import validate_numeric_array


def ensure_ants_image(image: ANTsImage, name: str = "image") -> None:
    """Ensure `image` is an :class:`ants.core.ANTsImage`."""
    if not isinstance(image, ANTsImage):
        raise ValueError(f"`{name}` must be an ANTsImage, got {type(image).__name__}")


def reject_reserved_kwargs(
    kwargs: dict[str, Any],
    reserved: Sequence[str],
    *,
    func_name: str,
) -> None:
    """Reject `kwargs` keys that the wrapper already sets explicitly.

    Several wrappers forward ``**kwargs`` to an ANTs function while also
    passing a few arguments explicitly, often under a more intuitive name
    (e.g. ``image``/``target`` instead of ``moving``/``fixed``). Letting the
    same underlying argument come through ``kwargs`` as well would either
    raise an opaque :class:`TypeError` ("got multiple values for ...") or
    silently override the explicit value, so we reject the clashing keys up
    front with an actionable message.

    Args:
        kwargs:
            The keyword arguments forwarded by the caller.
        reserved:
            Names that the wrapper controls and must not be overridden.
        func_name:
            Name of the calling wrapper, used in the error message.

    Raises:
        TypeError: If any reserved name is present in `kwargs`.
    """
    clashing = sorted(set(reserved) & set(kwargs))
    if clashing:
        raise TypeError(
            f"{func_name}() does not accept {clashing} via `kwargs`; these are "
            "set through dedicated parameters."
        )


def ants_to_numpy_with_metadata(image: ANTsImage) -> tuple[np.ndarray, dict[str, Any]]:
    """Convert `image` to numpy and capture enough metadata to reconstruct it."""
    ensure_ants_image(image)
    data = image.numpy()
    validate_numeric_array(data, name="image.numpy()")
    metadata = {"reference_image": image}
    return data, metadata


def numpy_to_ants_with_metadata(
    array: np.ndarray, metadata: dict[str, Any]
) -> ANTsImage:
    """Convert a numpy array back to ANTsImage using captured metadata."""
    validate_numeric_array(array)
    if not isinstance(metadata, dict):
        raise ValueError(
            f"`metadata` must be a dictionary, got {type(metadata).__name__}"
        )
    if "reference_image" not in metadata:
        raise ValueError("`metadata` must include a `reference_image` key")

    reference_image = metadata["reference_image"]
    ensure_ants_image(reference_image, name="metadata['reference_image']")
    if array.shape != reference_image.shape:
        raise ValueError(
            "`array` must have the same shape as metadata reference image, got "
            f"{array.shape} and {reference_image.shape}"
        )
    return reference_image.new_image_like(array)


def reconstruct_ants_image(
    array: np.ndarray,
    reference_image: ANTsImage,
    voxel_offset: Sequence[int] | np.ndarray | None = None,
) -> ANTsImage:
    """Build a new ANTsImage from `array`, preserving `reference_image`'s frame.

    The new image inherits spacing and direction from `reference_image`.
    The origin is shifted to remain spatially consistent with the cropping
    or padding implied by `voxel_offset`, which is the index (in
    `reference_image`'s voxel coordinates) that becomes the new array's
    ``(0, ..., 0)`` voxel:

    * Positive values correspond to cropping forward along that axis.
    * Negative values correspond to padding before along that axis.
    * Zero (the default for every axis) leaves the origin unchanged.

    Args:
        array:
            The reshaped numeric array to wrap.
        reference_image:
            The ANTsImage whose frame (spacing, direction, origin) is
            inherited.
        voxel_offset:
            Per-axis offset, of length ``reference_image.dimension``.
            When ``None`` (default) a zero offset is used, in which case
            this is equivalent to :func:`numpy_to_ants_with_metadata`
            when shapes match.

    Returns:
        An :class:`ants.core.ANTsImage` wrapping `array` with consistent
        spatial metadata.
    """
    validate_numeric_array(array)
    ensure_ants_image(reference_image, name="reference_image")

    ndim = reference_image.dimension
    if voxel_offset is None:
        offset = np.zeros(ndim, dtype=np.float64)
    else:
        offset = np.asarray(voxel_offset, dtype=np.float64)
        if offset.shape != (ndim,):
            raise ValueError(
                f"`voxel_offset` must have length {ndim}, got shape "
                f"{tuple(offset.shape)}"
            )

    spacing = np.asarray(reference_image.spacing, dtype=np.float64)
    direction = np.asarray(reference_image.direction, dtype=np.float64)
    origin = np.asarray(reference_image.origin, dtype=np.float64)
    new_origin = origin + direction @ (spacing * offset)

    return ants.from_numpy(
        array,
        origin=tuple(new_origin.tolist()),
        spacing=tuple(float(s) for s in reference_image.spacing),
        direction=direction,
    )
