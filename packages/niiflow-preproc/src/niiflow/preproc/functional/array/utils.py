"""Shared utility helpers for numpy-array functional modules."""

from __future__ import annotations

import numpy as np


def validate_numeric_array(array: np.ndarray, name: str = "array") -> None:
    """Ensure `array` is a numeric :class:`numpy.ndarray`."""
    if not isinstance(array, np.ndarray):
        raise ValueError(f"`{name}` must be a numpy array, got {type(array).__name__}")
    if not np.issubdtype(array.dtype, np.number):
        raise ValueError(f"`{name}` must have a numeric dtype, got {array.dtype}")


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

    if limit_to.dtype == bool:
        mask = limit_to
    elif np.issubdtype(limit_to.dtype, np.number):
        unique_vals = np.unique(limit_to)
        if not np.all(np.isin(unique_vals, (0, 1))):
            raise ValueError(
                "`limit_to` must contain only 0s and 1s, got unique values "
                f"{unique_vals.tolist()}"
            )
        mask = limit_to.astype(bool)
    else:
        raise ValueError(
            f"`limit_to` must be a boolean or numeric array, got dtype {limit_to.dtype}"
        )

    if not mask.any():
        raise ValueError("`limit_to` mask is empty (contains no True / 1 values)")

    return mask
