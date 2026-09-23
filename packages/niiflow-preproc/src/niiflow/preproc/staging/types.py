"""Shared staging type aliases and TypedDict specs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, NotRequired, Required, TypeAlias, TypedDict

PointerKind: TypeAlias = Literal["input", "output"]
RootMode: TypeAlias = Literal["path", "parent_up", "parent_match"]
RootSelection: TypeAlias = Literal["most_local", "most_global", "raise"]
ResolveResults: TypeAlias = Literal["first", "all", "single"]


class MirrorSpec(TypedDict):
    """Mirror one hierarchy into another hierarchy.

    ``source`` may be a single path/pattern or a list tried in order (first match
    wins). When no source matches, raise unless ``allow_missing_source`` is true,
    in which case the unresolved root is left unchanged.
    """

    source: str | Path | list[str | Path]
    target: str | Path
    allow_missing_source: NotRequired[bool]


class RootSearchSpec(TypedDict, total=False):
    """Root resolution specification.

    Omit ``mode`` to start from the active file's parent (then apply ``mirror`` if
    present). ``mode`` is required for ``path``, ``parent_up``, and ``parent_match``.
    """

    mode: NotRequired[RootMode]
    value: NotRequired[str | Path | int | None]
    mirror: NotRequired[MirrorSpec | None]
    selection: NotRequired[RootSelection]


SingleRootSpec: TypeAlias = RootSearchSpec | str | Path | None
RootSpec: TypeAlias = SingleRootSpec | list[SingleRootSpec]


class InputSearchSpec(TypedDict, total=False):
    """Input file search specification.

    Locate files under ``root`` with an explorer. ``root`` is optional and defaults to
    the active file's parent. Mutually exclusive with :class:`InputNameSpec`.
    """

    root: RootSpec
    search: Required[dict[str, Any]]
    resolve_results: NotRequired[ResolveResults]


class InputNameSpec(TypedDict, total=False):
    """Input file path-construction specification.

    Join ``root`` with ``name``, the same shape as an output pointer. ``root`` is
    optional and defaults to the active file's parent. Mutually exclusive with
    :class:`InputSearchSpec`.
    """

    root: RootSpec
    name: Required[str]


InputSpec: TypeAlias = InputSearchSpec | InputNameSpec | str | Path | None


class OutputSearchSpec(TypedDict, total=False):
    """Output path construction specification."""

    root: RootSpec
    name: Required[str]


OutputSpec: TypeAlias = OutputSearchSpec | str | Path | None
