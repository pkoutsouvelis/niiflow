"""Miscellaneous utility helpers."""

from __future__ import annotations

from typing import Any

import numpy as np

NUMPY_DTYPE_ALIASES: dict[str, np.dtype[Any]] = {
    "bool": np.dtype(np.bool_),
    "float16": np.dtype(np.float16),
    "float32": np.dtype(np.float32),
    "float64": np.dtype(np.float64),
    "int8": np.dtype(np.int8),
    "int16": np.dtype(np.int16),
    "int32": np.dtype(np.int32),
    "int64": np.dtype(np.int64),
    "uint8": np.dtype(np.uint8),
    "uint16": np.dtype(np.uint16),
    "uint32": np.dtype(np.uint32),
    "uint64": np.dtype(np.uint64),
}


def resolve_numpy_dtype(value: str | np.dtype[Any] | type[Any]) -> np.dtype[Any]:
    """Resolve a user-provided dtype string (or pass-through) to :class:`numpy.dtype`.

    Known aliases are listed in :data:`NUMPY_DTYPE_ALIASES`; other valid NumPy dtype
    strings (e.g. ``"f4"``, ``"<f8"``) are accepted via :func:`numpy.dtype`.
    """
    if isinstance(value, np.dtype):
        return value
    if isinstance(value, str):
        key = value.strip().lower()
        if key in NUMPY_DTYPE_ALIASES:
            return NUMPY_DTYPE_ALIASES[key]
        try:
            return np.dtype(key)
        except TypeError as e:
            raise ValueError(
                f"Unknown numpy dtype {value!r}; expected one of "
                f"{sorted(NUMPY_DTYPE_ALIASES)} or a valid NumPy dtype string"
            ) from e
    if isinstance(value, type) and issubclass(value, (np.generic, bool, int, float)):
        return np.dtype(value)
    raise TypeError(
        f"`dtype` must be a string, numpy dtype, or numpy scalar type, "
        f"got {type(value).__name__}"
    )
