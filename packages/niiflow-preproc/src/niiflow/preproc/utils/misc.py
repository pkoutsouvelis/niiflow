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


def split_dotted_path(path: str) -> list[str | int]:
    """Split a dotted path and parse list indices written as ``[idx]``."""
    parts: list[str | int] = []
    for token in path.split("."):
        token = token.strip()
        if not token:
            raise ValueError(f"Invalid empty token in dotted path {path!r}.")
        if token.startswith("[") and token.endswith("]"):
            raw = token[1:-1]
            try:
                parts.append(int(raw))
            except ValueError as exc:
                raise ValueError(
                    f"Invalid list index {token!r} in dotted path {path!r}."
                ) from exc
        else:
            parts.append(token)
    return parts


def get_by_dotted_path(data: Any, path: str) -> Any:
    """Read a nested value from dictionaries/lists using a dotted path."""
    current = data
    for part in split_dotted_path(path):
        if isinstance(part, int):
            if not isinstance(current, list):
                raise TypeError(
                    f"Expected list before index [{part}] in {path!r}, got "
                    f"{type(current).__name__}."
                )
            current = current[part]
        else:
            if not isinstance(current, dict):
                raise TypeError(
                    f"Expected dict before key {part!r} in {path!r}, got "
                    f"{type(current).__name__}."
                )
            current = current[part]
    return current


def set_by_dotted_path(data: Any, path: str, value: Any) -> None:
    """Set a nested value in dictionaries/lists using a dotted path."""
    parts = split_dotted_path(path)
    if not parts:
        raise ValueError("Cannot set an empty dotted path.")

    current = data
    for part in parts[:-1]:
        if isinstance(part, int):
            if not isinstance(current, list):
                raise TypeError(
                    f"Expected list before index [{part}] in {path!r}, got "
                    f"{type(current).__name__}."
                )
            current = current[part]
        else:
            if not isinstance(current, dict):
                raise TypeError(
                    f"Expected dict before key {part!r} in {path!r}, got "
                    f"{type(current).__name__}."
                )
            current = current[part]

    last = parts[-1]
    if isinstance(last, int):
        if not isinstance(current, list):
            raise TypeError(
                f"Expected list before index [{last}] in {path!r}, got "
                f"{type(current).__name__}."
            )
        current[last] = value
    else:
        if not isinstance(current, dict):
            raise TypeError(
                f"Expected dict before key {last!r} in {path!r}, got "
                f"{type(current).__name__}."
            )
        current[last] = value
