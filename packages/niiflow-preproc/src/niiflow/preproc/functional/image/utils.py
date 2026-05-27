"""Shared utility helpers for ANTsImage functional modules."""

from __future__ import annotations

from typing import Any

import numpy as np
from ants.core import ANTsImage

from niiflow.preproc.functional.array.utils import validate_numeric_array


def ensure_ants_image(image: ANTsImage, name: str = "image") -> None:
    """Ensure `image` is an :class:`ants.core.ANTsImage`."""
    if not isinstance(image, ANTsImage):
        raise ValueError(f"`{name}` must be an ANTsImage, got {type(image).__name__}")


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
