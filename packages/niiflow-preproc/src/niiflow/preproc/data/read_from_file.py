"""Load active file paths from a text list."""

from __future__ import annotations

__all__ = [
    "read_paths_from_file",
]

import warnings
from pathlib import Path

from niiflow.preproc.utils.file import get_ext, resolve_path


def read_paths_from_file(
    path: Path | str,
    *,
    resolve: bool | None = None,
    must_exist: bool | None = None,
    strict: bool | None = None,
    skip_resolve_filepaths: bool | None = None,
) -> list[Path]:
    """Read a ``.txt`` file listing file paths (one path per line).

    Blank lines are ignored. Each non-blank line is turned into a
    :class:`~pathlib.Path`. By default, listed paths are expanded and resolved
    to absolute paths. When ``resolve`` is ``False``, listed paths are parsed
    as written. The ``.txt`` list path itself is always resolved.

    When ``must_exist`` is ``True``, each listed path must exist and be a file.
    When ``False``, every non-blank line is kept with no existence check.

    ``strict`` and ``skip_resolve_filepaths`` are deprecated aliases (removed in
    v0.5.0): ``strict`` maps to ``must_exist``, and
    ``skip_resolve_filepaths=True`` maps to ``resolve=False``.

    Args:
        path: Path to a ``.txt`` file containing one filesystem path per line.
        resolve: When ``True`` (default), expand and resolve each listed path.
            When ``False``, keep the path as written. ``None`` means not passed.
        must_exist: When ``True`` (default), raise if a listed path does not
            exist or is not a file. When ``False``, keep listed paths without
            checking that they exist. ``None`` means not passed.
        strict: Deprecated alias of ``must_exist``. ``None`` means it was not
            passed. Cannot be combined with ``must_exist``.
        skip_resolve_filepaths: Deprecated inverse of ``resolve``. ``None``
            means it was not passed. ``True`` sets ``resolve=False``. Cannot be
            combined with ``resolve``.

    Returns:
        :class:`~pathlib.Path` objects for every accepted line, in file order.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If ``path`` is not a ``.txt`` file, or (when ``must_exist``)
            a listed entry is missing or not a file.
        TypeError: If a flag is not a boolean, or a deprecated alias is combined
            with its replacement.
    """
    if strict is not None:
        warnings.warn(
            "`strict` is deprecated and will be removed in v0.5.0. "
            "Use `must_exist` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if must_exist is not None:
            raise TypeError("Pass only one of `must_exist` and `strict`.")
        must_exist = strict
    if must_exist is None:
        must_exist = True
    if not isinstance(must_exist, bool):
        raise TypeError(
            f"`must_exist` must be a boolean, got {type(must_exist).__name__}"
        )

    if skip_resolve_filepaths is not None:
        warnings.warn(
            "`skip_resolve_filepaths` is deprecated and will be removed in "
            "v0.5.0. Use `resolve` instead (`skip_resolve_filepaths=True` is "
            "`resolve=False`).",
            DeprecationWarning,
            stacklevel=2,
        )
        if resolve is not None:
            raise TypeError("Pass only one of `resolve` and `skip_resolve_filepaths`.")
        if not isinstance(skip_resolve_filepaths, bool):
            raise TypeError(
                "`skip_resolve_filepaths` must be a boolean, got "
                f"{type(skip_resolve_filepaths).__name__}"
            )
        resolve = not skip_resolve_filepaths
    if resolve is None:
        resolve = True
    if not isinstance(resolve, bool):
        raise TypeError(f"`resolve` must be a boolean, got {type(resolve).__name__}")

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

        candidate = resolve_path(entry) if resolve else Path(entry)
        if must_exist and not candidate.is_file():
            kind = "directory" if candidate.exists() else "missing path"
            raise ValueError(
                f"Line {line_no} of {list_path} is not an existing file "
                f"({kind}): {candidate}"
            )
        found.append(candidate)

    return found
