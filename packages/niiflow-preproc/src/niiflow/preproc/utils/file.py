"""Utility functions for file I/O operations."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from ants.core import ANTsImage, ANTsTransform
from ants.core import image_read, image_write, read_transform, write_transform


def resolve_path(path: Path | str) -> Path:
    """Expand user and resolve path."""
    if not isinstance(path, (Path, str)):
        raise ValueError(f"`path` must be a Path or str, got {type(path).__name__}")
    return Path(path).expanduser().resolve()


def get_ext(path: Path | str) -> str:
    """Get file extension from filepath with leading dot."""
    p = Path(path)
    suffixes = p.suffixes
    full_ext = "".join(suffixes)
    return full_ext


def ants_image_read(path: Path | str, *, reorient: bool | str = True) -> ANTsImage:
    """Read an image from disk into an :class:`ants.core.ANTsImage`.

    ``path`` is expanded and resolved before reading. By default (``reorient=True``),
    three-dimensional images are reoriented to **RPI** on load. Pass ``False`` to
    preserve the on-disk orientation, or a three-letter code (e.g. ``"RAS"``) for a
    specific target orientation.
    """
    resolved = resolve_path(path)
    try:
        return image_read(str(resolved), reorient=reorient)  # type: ignore[arg-type]
    except Exception as e:
        raise ValueError(f"Failed to read ANTs image from {resolved}") from e


def ants_image_write(image: ANTsImage, path: Path | str) -> Path:
    """Write an :class:`ants.core.ANTsImage` to disk.

    Parent directories are created when missing. Returns the resolved output path.
    """
    resolved = resolve_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    try:
        image_write(image, str(resolved))
    except Exception as e:
        raise ValueError(f"Failed to write ANTs image to {resolved}") from e
    return resolved


def ants_read_transform(path: Path | str) -> ANTsTransform:
    """Read a transform from disk into an :class:`ants.core.ANTsTransform`.

    ``path`` is expanded and resolved before reading.
    """
    resolved = resolve_path(path)
    try:
        return read_transform(str(resolved))
    except Exception as e:
        raise ValueError(f"Failed to read ANTs transform from {resolved}") from e


def ants_write_transform(transform: ANTsTransform, path: Path | str) -> Path:
    """Write an :class:`ants.core.ANTsTransform` to disk.

    Parent directories are created when missing. Returns the resolved output path.
    """
    resolved = resolve_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    try:
        write_transform(transform, str(resolved))
    except Exception as e:
        raise ValueError(f"Failed to write ANTs transform to {resolved}") from e
    return resolved


def write_npy(array: np.ndarray, path: Path | str) -> Path:
    """Write a NumPy array to a ``.npy`` file.

    Parent directories are created when missing. Returns the resolved output path.
    """
    if not isinstance(array, np.ndarray):
        raise TypeError(f"`array` must be a numpy.ndarray, got {type(array).__name__}")
    resolved = resolve_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(resolved), array)
    return resolved


def json_safe(value: Any) -> Any:
    """Convert a value to a JSON-safe representation.

    Mappings, sequences (including tuples), and scalars are supported. Paths are written
    as strings.
    """
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def read_json(path: Path | str) -> Any:
    """Load a JSON file and return its parsed root value.

    The root may be a mapping, list, or scalar depending on the file contents.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
    """
    resolved = resolve_path(path)
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    text = resolved.read_text(encoding="utf-8")
    return json.loads(text)


def write_json(data: Any, path: Path | str) -> Path:
    """Write a JSON-serializable value to a file.

    Mappings, sequences (including tuples), and scalars are supported. Paths are written
    as strings.

    Parent directories are created when missing. Returns the resolved output path.
    """
    resolved = resolve_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(
        json.dumps(json_safe(data), indent=2) + "\n",
        encoding="utf-8",
    )
    return resolved


def read_txt(path: Path | str) -> str:
    """Load a text file and return its contents.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
    """
    resolved = resolve_path(path)
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved.read_text(encoding="utf-8")


def write_txt(text: str, path: Path | str) -> Path:
    """Write ``text`` to a text file.

    Parent directories are created when missing. Returns the resolved output path. A
    single trailing newline is always written.
    """
    if not isinstance(text, str):
        raise TypeError(f"`text` must be a str, got {type(text).__name__}")
    resolved = resolve_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(text.rstrip("\n") + "\n", encoding="utf-8")
    return resolved


def read_yaml(path: Path | str) -> dict[str, Any]:
    """Load a YAML file and return its root mapping.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        TypeError: If the document root is not a mapping.
    """
    resolved = resolve_path(path)
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    text = resolved.read_text(encoding="utf-8")
    data = yaml.safe_load(text)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise TypeError(
            f"YAML root at {resolved} must be a mapping, got {type(data).__name__}"
        )
    return data


def write_yaml(data: dict[str, Any], path: Path | str) -> Path:
    """Write ``data`` to a YAML file.

    Parent directories are created when missing. Returns the resolved output path.
    """
    if not isinstance(data, dict):
        raise TypeError(f"`data` must be a dict, got {type(data).__name__}")
    resolved = resolve_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(
        yaml.safe_dump(data, sort_keys=False, default_flow_style=False),
        encoding="utf-8",
    )
    return resolved


def delete_paths(
    paths: Path | str | Sequence[Path | str],
    *,
    missing_ok: bool = False,
) -> list[Path]:
    """Delete one or more files.

    Each path is expanded and resolved before deletion. Returns the resolved paths that
    were removed. Directories are never deleted.
    """
    if isinstance(paths, (str, Path)):
        items: Sequence[Path | str] = [paths]
    elif isinstance(paths, Sequence):
        items = paths
    else:
        raise TypeError(
            f"`paths` must be a path or sequence of paths, got {type(paths).__name__}"
        )

    deleted: list[Path] = []
    for path in items:
        resolved = resolve_path(path)
        if not resolved.exists():
            if missing_ok:
                continue
            raise FileNotFoundError(resolved)
        if resolved.is_dir():
            raise ValueError(f"Refusing to delete directory {resolved}")
        resolved.unlink()
        deleted.append(resolved)
    return deleted
