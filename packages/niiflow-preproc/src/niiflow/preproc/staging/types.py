"""Shared staging type aliases and TypedDict specs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, NotRequired, Required, TypeAlias, TypedDict

PointerKind: TypeAlias = Literal["input", "output"]
RootMode: TypeAlias = Literal["active", "path", "parent_up", "parent_match"]
RootSelection: TypeAlias = Literal["most_local", "most_global", "raise"]
ResolveResults: TypeAlias = Literal["first", "all", "single"]


class MirrorSpec(TypedDict):
    """Mirror one hierarchy into another hierarchy."""

    source: str | Path
    target: str | Path


class RootSearchSpec(TypedDict, total=False):
    """Root resolution specification."""

    mode: Required[RootMode]
    value: NotRequired[str | Path | int | None]
    mirror: NotRequired[MirrorSpec | None]
    selection: NotRequired[RootSelection]


RootSpec: TypeAlias = RootSearchSpec | list[RootSearchSpec] | str | Path | None


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
