"""Type aliases for the preprocessing package."""

from __future__ import annotations

from typing import Any, Literal, Sequence, TypedDict, TypeAlias
from nifti_finder.explorers import FileFinder


class SimpleFilterConfig(TypedDict):
    name: str
    kwargs: dict[str, Any]


class ComposeFilterKwargs(TypedDict):
    filters: FilterConfig | list[FilterConfig | None]
    logic: Literal["AND", "OR"]


class ComposeFilterConfig(TypedDict):
    name: Literal["ComposeFilter"]
    kwargs: ComposeFilterKwargs


FilterConfig: TypeAlias = SimpleFilterConfig | ComposeFilterConfig


class NiftiFinderConfig(TypedDict, total=False):
    patterns: str | Sequence[str]
    levels: dict[str, str | Sequence[str]] | None
    filters: FilterConfig | None


DataExplorer: TypeAlias = FileFinder
