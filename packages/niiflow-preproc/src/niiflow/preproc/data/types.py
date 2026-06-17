"""Type aliases for the preprocessing package."""

from __future__ import annotations

from typing import Any, Literal, Sequence, TypedDict, TypeAlias
from nifti_finder.explorers import AllPurposeFileExplorer


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
    pattern: str | Sequence[str]
    filters: FilterConfig | None


DataExplorer: TypeAlias = AllPurposeFileExplorer
