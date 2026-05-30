"""Shared utility helpers for numpy-array functional modules."""

from __future__ import annotations

import numpy as np


def validate_numeric_array(array: np.ndarray, name: str = "array") -> None:
    """Ensure `array` is a numeric :class:`numpy.ndarray`."""
    if not isinstance(array, np.ndarray):
        raise ValueError(f"`{name}` must be a numpy array, got {type(array).__name__}")
    if not np.issubdtype(array.dtype, np.number):
        raise ValueError(f"`{name}` must have a numeric dtype, got {array.dtype}")


def resolve_mask(mask: np.ndarray, name: str = "mask") -> np.ndarray:
    """Validate `mask` and coerce it to a boolean :class:`numpy.ndarray`.

    Accepts either a boolean array, or a numeric array whose only values
    are ``0`` and ``1``. The returned mask is guaranteed to contain at
    least one ``True`` voxel.

    Raises:
        ValueError: If `mask` is not a numpy array, has a non-numeric and
            non-boolean dtype, contains values other than ``0`` and ``1``,
            or is entirely empty (no ``True`` / ``1`` voxels).
    """
    if not isinstance(mask, np.ndarray):
        raise ValueError(f"`{name}` must be a numpy array, got {type(mask).__name__}")

    if mask.dtype == bool:
        bool_mask = mask
    elif np.issubdtype(mask.dtype, np.number):
        unique_vals = np.unique(mask)
        if not np.all(np.isin(unique_vals, (0, 1))):
            raise ValueError(
                f"`{name}` must contain only 0s and 1s, got unique values "
                f"{unique_vals.tolist()}"
            )
        bool_mask = mask.astype(bool)
    else:
        raise ValueError(
            f"`{name}` must be a boolean or numeric array, got dtype {mask.dtype}"
        )

    if not bool_mask.any():
        raise ValueError(f"`{name}` mask is empty (contains no True / 1 values)")

    return bool_mask


def resolve_limit_to_mask(
    limit_to: np.ndarray, array_shape: tuple[int, ...]
) -> np.ndarray:
    """Validate `limit_to` and return it as a boolean mask."""
    if not isinstance(limit_to, np.ndarray):
        raise ValueError(
            f"`limit_to` must be a numpy array, got {type(limit_to).__name__}"
        )
    if limit_to.shape != array_shape:
        raise ValueError(
            f"`limit_to` must have shape {array_shape}, got {limit_to.shape}"
        )
    return resolve_mask(limit_to, name="limit_to")
