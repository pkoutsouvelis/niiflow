"""Typed shapes for workflow run inputs (active-file discovery).

``InputData`` is what plannable workflows accept as ``run_inputs``: either one input
specification or a sequence of them. Each specification is a discriminated union on
``mode`` (when a mapping), or a bare path string/Path for an explicit active file.
"""

from __future__ import annotations

__all__ = [
    "FromFileInput",
    "InputData",
    "RunInputSpec",
    "SearchInput",
]

from collections.abc import Sequence
from pathlib import Path
from typing import Literal, TypeAlias, TypedDict

from niiflow.preproc.data.types import NiftiFinderConfig


class SearchInput(TypedDict):
    """Discover active files under one or more roots with a FileFinder config.

    ``explorer_params`` is forwarded to :func:`~niiflow.preproc.data.get_data_explorer`
    (``patterns``, optional ``levels`` / ``filters``).
    """

    mode: Literal["search"]
    roots: Path | str | Sequence[Path | str]
    explorer_params: NiftiFinderConfig


class FromFileInput(TypedDict):
    """Load active file paths from a text file (one path per line)."""

    mode: Literal["from_file"]
    path: Path | str


RunInputSpec: TypeAlias = Path | str | SearchInput | FromFileInput

InputData: TypeAlias = RunInputSpec | Sequence[RunInputSpec]
