"""Load active file paths from a text list."""

from __future__ import annotations

__all__ = [
    "read_paths_from_file",
]

from pathlib import Path

from niiflow.preproc.utils.file import get_ext, resolve_path


def read_paths_from_file(
    path: Path | str,
    *,
    strict: bool = True,
) -> list[Path]:
    """Read a ``.txt`` file listing file paths (one per line).

    Blank lines are ignored. Each non-blank line is expanded and resolved to an
    absolute path. Only paths that exist and are files are returned.

    Args:
        path: Path to a ``.txt`` file containing one filesystem path per line.
        strict: When ``True`` (default), raise if a listed path does not exist
            or is not a file. When ``False``, skip such entries.

    Returns:
        Resolved :class:`~pathlib.Path` objects for every accepted file, in file
        order.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If ``path`` is not a ``.txt`` file, or (when ``strict``) a
            listed entry is missing or not a file.
        TypeError: If ``strict`` is not a boolean.
    """
    if not isinstance(strict, bool):
        raise TypeError(f"`strict` must be a boolean, got {type(strict).__name__}")

    list_path = resolve_path(path)
    if not list_path.is_file():
        raise FileNotFoundError(list_path)
    if get_ext(list_path) != ".txt":
        raise ValueError(f"`path` must be a .txt file, got {list_path}")

    lines = list_path.read_text(encoding="utf-8").splitlines()
    found: list[Path] = []
    for line_no, raw in enumerate(lines, start=1):
        entry = raw.strip()
        if not entry:
            continue

        resolved = resolve_path(entry)
        if resolved.is_file():
            found.append(resolved)
            continue

        if strict:
            kind = "directory" if resolved.exists() else "missing path"
            raise ValueError(
                f"Line {line_no} of {list_path} is not an existing file "
                f"({kind}): {resolved}"
            )

    return found
