"""Small input-search helpers."""

from __future__ import annotations

import fnmatch
from pathlib import Path
from typing import Any

from niiflow.preproc.utils.file import resolve_path
from .types import ResolveResults, RootSelection


def parent_up(active: Path, levels: int) -> Path:
    """Move ``levels`` up from the active file's parent directory."""
    if levels < 0:
        raise ValueError("parent_up levels must be >= 0.")
    root = active.parent
    for _ in range(levels):
        root = root.parent
    return root


def match_parent(
    path: Path,
    pattern: str,
    *,
    selection: RootSelection = "most_local",
) -> Path:
    """Find a parent path matching slash-separated shell-style patterns.

    ``path`` is treated as a directory/root candidate. The path itself is considered
    before its parents, so ``most_local`` means the nearest match.
    """
    pattern_parts = tuple(part for part in pattern.split("/") if part)
    if not pattern_parts:
        raise ValueError("parent_match pattern cannot be empty.")

    start = Path(path).expanduser().resolve()
    candidates: list[Path] = []

    for candidate in (start, *start.parents):
        parts = candidate.parts
        if len(parts) < len(pattern_parts):
            continue
        tail = parts[-len(pattern_parts) :]
        if all(
            fnmatch.fnmatchcase(part, patt) for part, patt in zip(tail, pattern_parts)
        ):
            candidates.append(candidate)

    if not candidates:
        raise FileNotFoundError(
            f"Could not resolve parent_match pattern {pattern!r} from path {start}."
        )

    if len(candidates) == 1:
        return candidates[0]
    if selection == "most_local":
        return candidates[0]
    if selection == "most_global":
        return candidates[-1]
    if selection == "raise":
        raise RuntimeError(
            f"parent_match pattern {pattern!r} matched multiple roots for {start}: "
            + ", ".join(str(item) for item in candidates)
        )
    raise ValueError(
        f"Unknown parent_match selection {selection!r}. Expected 'most_local', "
        "'most_global', or 'raise'."
    )


def mirror_root(root: Path, *, source: str | Path, target: str | Path) -> Path:
    """Mirror ``root`` from a source hierarchy into a target hierarchy.

    If ``source`` is absolute, ``root`` must be below it. If ``source`` is relative, it
    is interpreted as a parent-match pattern and matched with
    ``selection='most_global'``.
    """
    root = resolve_path(root)
    target_root = resolve_path(target)
    source_path = resolve_path(source)

    if source_path.is_absolute():
        source_root = source_path.resolve()
    else:
        source_root = match_parent(root, str(source_path), selection="most_global")

    try:
        relative_tail = root.relative_to(source_root)
    except ValueError as exc:
        raise FileNotFoundError(
            f"Cannot mirror root {root} because it is not under source {source_root}."
        ) from exc

    return target_root / relative_tail


def resolve_search_result(
    paths: list[Path],
    *,
    policy: ResolveResults = "first",
    pointer: str | None = None,
) -> Path | list[Path]:
    """Resolve search results according to the multiplicity policy."""
    if not isinstance(paths, list):
        raise TypeError(f"`paths` must be a list, got {type(paths).__name__}")

    paths = sorted(paths)
    label = f" for pointer {pointer!r}" if pointer else ""

    if policy == "all":
        return paths

    if not paths:
        raise FileNotFoundError(f"Search returned no files{label}.")

    if policy == "first":
        return paths[0]

    if policy == "single":
        if len(paths) != 1:
            raise RuntimeError(
                f"Search expected a single file{label}, but returned {len(paths)} files: "
                + ", ".join(str(path) for path in paths)
            )
        return paths[0]

    raise ValueError(f"Unknown resolve_results policy {policy!r}.")


def explorer_cache_key(search: dict[str, Any]) -> tuple[Any, ...] | None:
    """Return a hashable cache key, or None if the search spec cannot be cached."""

    def freeze(value: Any) -> Any:
        if isinstance(value, dict):
            return tuple(sorted((key, freeze(item)) for key, item in value.items()))
        if isinstance(value, list):
            return tuple(freeze(item) for item in value)
        if isinstance(value, tuple):
            return tuple(freeze(item) for item in value)
        if isinstance(value, set):
            return tuple(sorted(freeze(item) for item in value))
        hash(value)
        return value

    try:
        return freeze(search)
    except TypeError:
        return None
