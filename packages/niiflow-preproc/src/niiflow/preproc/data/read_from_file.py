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
    skip_resolve_filepaths: bool = False,
) -> list[Path]:
    """Read a ``.txt`` file listing file paths (one path per line).

    Blank lines are ignored. Each non-blank line is turned into a
    :class:`~pathlib.Path`. By default, listed paths are expanded and resolved
    to absolute paths. When ``skip_resolve_filepaths`` is ``True``, listed paths
    are not expanded or resolved — they are parsed as written. The ``.txt``
    list path itself is always resolved.

    When ``strict`` is ``True``, each listed path must exist and be a file.
    When ``False``, missing paths and directories are skipped.

    Args:
        path: Path to a ``.txt`` file containing one filesystem path per line.
        strict: When ``True`` (default), raise if a listed path does not exist
            or is not a file. When ``False``, skip such entries.
        skip_resolve_filepaths: When ``False`` (default), expand and resolve
            each extracted filepath. When ``True``, do not expand or resolve
            listed paths (parsed as they are). Does not affect resolution of
            ``path``.

    Returns:
        :class:`~pathlib.Path` objects for every accepted file, in file order.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If ``path`` is not a ``.txt`` file, or (when ``strict``) a
            listed entry is missing or not a file.
        TypeError: If ``strict`` or ``skip_resolve_filepaths`` is not a boolean.
    """
    if not isinstance(strict, bool):
        raise TypeError(f"`strict` must be a boolean, got {type(strict).__name__}")
    if not isinstance(skip_resolve_filepaths, bool):
        raise TypeError(
            f"`skip_resolve_filepaths` must be a boolean, "
            f"got {type(skip_resolve_filepaths).__name__}"
        )

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

        candidate = Path(entry) if skip_resolve_filepaths else resolve_path(entry)
        if candidate.is_file():
            found.append(candidate)
            continue

        if strict:
            kind = "directory" if candidate.exists() else "missing path"
            raise ValueError(
                f"Line {line_no} of {list_path} is not an existing file "
                f"({kind}): {candidate}"
            )

    return found
