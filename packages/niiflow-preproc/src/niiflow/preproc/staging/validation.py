"""Small validation helpers."""

from __future__ import annotations

from typing import Any, Mapping
from pathlib import Path

from .types import PointerKind
from niiflow.preproc.utils.file import resolve_path


def ensure_file(path: Path | str, *, must_exist: bool = True) -> Path:
    """Ensure that a path exists and is a file."""
    path = resolve_path(path)
    if must_exist and not path.exists():
        raise FileNotFoundError(f"File does not exist: {path}")
    if must_exist and not path.is_file():
        raise ValueError(f"Path is not a file: {path}")
    return path


def ensure_directory(path: Path | str, *, must_exist: bool = True) -> Path:
    """Ensure that a path exists and is a directory."""
    path = resolve_path(path)
    if must_exist and not path.exists():
        raise FileNotFoundError(f"Directory does not exist: {path}")
    if must_exist and not path.is_dir():
        raise ValueError(f"Path is not a directory: {path}")
    return path


def ensure_no_overwrite(path: Path | str, *, allow_overwrite: bool = False) -> Path:
    """Ensure that a path does not already exist."""
    path = resolve_path(path)
    if path.exists() and path.is_dir():
        raise ValueError(f"Output path points to an existing directory: {path}")
    if path.exists() and not allow_overwrite:
        raise FileExistsError(f"Path already exists and allow_overwrite=False: {path}")
    return path


def require_mapping(value: Any, label: str) -> dict[str, Any]:
    """Return ``value`` as a dict-like mapping or raise a clear TypeError."""
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping, got {type(value).__name__}.")
    return dict(value)


def require_keys(mapping: Mapping[str, Any], required: list[str], label: str) -> None:
    missing = [key for key in required if key not in mapping]
    if missing:
        raise ValueError(f"{label} missing required key(s): {missing}.")


def check_allowed_keys(
    mapping: Mapping[str, Any], allowed: set[str], label: str
) -> None:
    unknown = [key for key in mapping if key not in allowed]
    if unknown:
        raise ValueError(
            f"{label} contains unsupported key(s): {unknown}. Allowed keys: {sorted(allowed)}."
        )


def validate_pointers(
    pointers: dict[str, PointerKind],
) -> dict[str, PointerKind]:
    """Validate and return a copy of the pointer mapping for :class:`FileStager`.

    Each key must be a dotted parameter path string; each value must be ``"input"`` or
    ``"output"``. The mapping must be non-empty.
    """
    if not isinstance(pointers, dict):
        raise TypeError(
            f"pointers must be a dictionary, got {type(pointers).__name__}."
        )
    if not pointers:
        raise ValueError("pointers must be a non-empty dictionary.")

    out: dict[str, PointerKind] = {}
    for key, value in pointers.items():
        if not isinstance(key, str):
            raise TypeError(f"Pointer keys must be strings, got {type(key).__name__}.")
        if value not in {"input", "output"}:
            raise ValueError(
                f"Pointer {key!r} must be 'input' or 'output', got {value!r}."
            )
        out[key] = value
    return out
