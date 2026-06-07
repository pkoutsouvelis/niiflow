"""Type aliases for the preprocessing package."""

from __future__ import annotations

from typing import Any, Literal, NotRequired, Sequence, TypedDict, TypeAlias
from pathlib import Path

# ---------------------------------------------------------------------------
# Data Explorer: nifti-finder discovery types
# ---------------------------------------------------------------------------


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


class NiftiFinderConfig(TypedDict):
    root: str | Path
    pattern: str | Sequence[str]
    filters: NotRequired[FilterConfig | None]


InputData: TypeAlias = Path | str | Sequence[Path | str] | NiftiFinderConfig


# ---------------------------------------------------------------------------
# Pipeline Stage: Relative file discovery types
# ---------------------------------------------------------------------------


class ActiveRootSource(TypedDict):
    """Use the directory containing the active file.

    Given:
        /data/bids/sub-001/ses-01/anat/sub-001_ses-01_T1w.nii.gz

    Resolves to:
        /data/bids/sub-001/ses-01/anat
    """

    kind: Literal["active"]


class ParentByLevelsRootSource(TypedDict):
    """Move upward from active.parent by a fixed number of levels.

    levels=0:     same as active

    levels=1:     parent of active.parent

    levels=2:     grandparent of active.parent
    """

    kind: Literal["parent"]
    levels: int


class ParentMatchSpec(TypedDict, total=False):
    """Rule for selecting an ancestor directory.

    Exactly one field should be provided.
    """

    name: str
    name_startswith: str
    name_endswith: str
    glob: str


class ParentByMatchRootSource(TypedDict):
    """Search upward from active.parent until an ancestor directory matches a rule."""

    kind: Literal["parent"]
    match: ParentMatchSpec


ParentRootSource: TypeAlias = ParentByLevelsRootSource | ParentByMatchRootSource


class ContextRootSource(TypedDict):
    """Resolve root from a path stored in PipelineContext.

    Examples:
        {"kind": "context", "value": "ctx.run_id"}
        {"kind": "context", "value": "ctx.artifacts.skullstrip_dir"}
        {"kind": "context", "value": "ctx.metadata.template_root"}
    """

    kind: Literal["context"]
    value: str


class PathRootSource(TypedDict):
    """Resolve root from an explicit filesystem path."""

    kind: Literal["path"]
    value: str | Path


RootSource: TypeAlias = (
    ActiveRootSource | ParentRootSource | ContextRootSource | PathRootSource
)


class MirrorSpec(TypedDict):
    """
    Mirror the resolved root from one tree into another.

    Example:
        resolved root = /data/bids/sub-001/ses-01/anat
        source        = /data/bids
        target        = /data/derivatives/skullstrip

        mirrored root = /data/derivatives/skullstrip/sub-001/ses-01/anat
    """

    source: str | Path
    target: str | Path


class RootSpec(TypedDict, total=False):
    """Resolve one directory root for globbing / nifti-finder exploration.

    source:
        How to derive the initial root directory.

    mirror:
        Optional mapping from one root tree to another.
    """

    source: RootSource
    mirror: MirrorSpec


class PathInputSpec(TypedDict, total=False):
    """Use an explicit filepath.

    The path may be absolute or relative. If relative, your resolver can interpret it
    relative to the current working directory or reject it, depending on your policy.
    """

    kind: Literal["path"]
    path: str | Path


class ContextPathInputSpec(TypedDict, total=False):
    """Use a filepath stored in PipelineContext.

    Example values:
        ctx.artifacts.brain_mask
        ctx.metadata.template_path
        ctx.run_id
    """

    kind: Literal["context"]
    value: str


class SuffixSearchSpec(TypedDict):
    """Search for a file by replacing or deriving a suffix from the active filename.

    Example:
        active:
            sub-001_ses-01_T1w.nii.gz

        suffix:
            _FLAIR.nii.gz

        candidate:
            sub-001_ses-01_FLAIR.nii.gz
    """

    mode: Literal["suffix"]
    suffix: str


class FinderSearchSpec(TypedDict, total=False):
    """Search using nifti-finder inside the resolved root.

    The root is specified in the surrounding SearchInputSpec.
    """

    mode: Literal["nifti_finder"]
    patterns: str | Sequence[str]
    filters: FilterConfig | None


SearchSpec: TypeAlias = SuffixSearchSpec | FinderSearchSpec


class SearchInputSpec(TypedDict, total=False):
    """Resolve a root, then search for file(s) inside that root.

    root:
        Directory root for searching.
        Defaults to the parent of the path identified by ``ctx.run_id`` when
        that value is a filesystem path; otherwise omitted.

    search:
        Either suffix-based discovery or nifti-finder-based discovery.

    return_all:
        If False, expect exactly one result.
        If True, return all matches.
    """

    kind: Literal["search"]
    root: RootSpec
    search: SearchSpec


FileInput: TypeAlias = PathInputSpec | ContextPathInputSpec | SearchInputSpec
