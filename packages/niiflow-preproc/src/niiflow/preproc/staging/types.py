"""Shared staging type aliases and TypedDict specs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, NotRequired, Required, TypeAlias, TypedDict

PointerKind: TypeAlias = Literal["input", "output"]
RootMode: TypeAlias = Literal["path", "parent_up", "parent_match"]
RootSelection: TypeAlias = Literal["most_local", "most_global", "raise"]
ResolveResults: TypeAlias = Literal["first", "all", "single"]


class MirrorSpec(TypedDict):
    """Mirror one hierarchy into another hierarchy."""

    source: str | Path
    target: str | Path


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
    """Input file search specification."""

    root: RootSpec
    search: Required[dict[str, Any]]
    resolve_results: NotRequired[ResolveResults]


InputSpec: TypeAlias = InputSearchSpec | str | Path | None


class OutputSearchSpec(TypedDict, total=False):
    """Output path construction specification."""

    root: RootSpec
    name: Required[str]


OutputSpec: TypeAlias = OutputSearchSpec | str | Path | None
