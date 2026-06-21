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


def resolve_norm_mask(
    array: np.ndarray,
    limit_to: np.ndarray | None,
    non_zero: bool,
) -> np.ndarray | None:
    """Combine a `limit_to` mask with an optional non-zero restriction.

    Returns a boolean mask (same shape as `array`) selecting the voxels that
    statistics and the transform should be restricted to, or ``None`` when
    neither restriction is requested (operate on the full array).

    When `non_zero` is ``True``, the non-zero voxels of `array` (``array != 0``)
    are intersected with `limit_to` (when provided).

    Raises:
        ValueError: If `limit_to` is invalid, or if the combined region selects
            no voxels (e.g. `non_zero=True` on an all-zero array).
    """
    mask: np.ndarray | None = None
    if limit_to is not None:
        mask = resolve_limit_to_mask(limit_to, array.shape)
    if non_zero:
        non_zero_mask = array != 0
        mask = non_zero_mask if mask is None else (mask & non_zero_mask)
    if mask is not None and not mask.any():
        raise ValueError(
            "The selected region is empty (no voxels remain after applying "
            "`limit_to` / `non_zero`); nothing to compute statistics from."
        )
    return mask
