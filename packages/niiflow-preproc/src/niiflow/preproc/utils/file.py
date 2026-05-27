"""Utility functions for file I/O operations."""

from pathlib import Path


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
