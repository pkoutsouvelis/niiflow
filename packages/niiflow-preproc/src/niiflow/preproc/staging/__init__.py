"""Staging utilities."""

from .file_stager import FileStager
from .table_stager import TableStager
from .utility import EnsureActivesExist, EnsureActiveExists
from .dynamic_referencing import (
    DynamicReferenceError,
    ResolveActiveReferences,
    ResolveParamReferences,
    add_reference_staging_bookends,
    resolve_dynamic_refs,
)
from .modifiers import (
    get_modifiers,
    lstrip,
    replace,
    rstrip,
    strip,
)
from .stager import (
    FileStagingError,
    StagingContext,
    StagedEntry,
    Stager,
    StagingErrorRecord,
    make_entries,
)
from .stager_factory import create_stager, discover_stager_classes
from .types import (
    InputNameSpec,
    InputSearchSpec,
    InputSpec,
    MirrorSpec,
    OutputSearchSpec,
    OutputSpec,
    PointerKind,
    ResolveResults,
    RootMode,
    RootSearchSpec,
    RootSelection,
    RootSpec,
    SingleRootSpec,
)

__all__ = [
    "FileStager",
    "TableStager",
    "EnsureActivesExist",
    "EnsureActiveExists",
    "DynamicReferenceError",
    "ResolveActiveReferences",
    "ResolveParamReferences",
    "add_reference_staging_bookends",
    "resolve_dynamic_refs",
    "get_modifiers",
    "lstrip",
    "replace",
    "rstrip",
    "strip",
    "StagingContext",
    "StagedEntry",
    "Stager",
    "FileStagingError",
    "StagingErrorRecord",
    "create_stager",
    "discover_stager_classes",
    "make_entries",
    "InputNameSpec",
    "InputSearchSpec",
    "InputSpec",
    "MirrorSpec",
    "OutputSearchSpec",
    "OutputSpec",
    "PointerKind",
    "ResolveResults",
    "RootMode",
    "RootSearchSpec",
    "RootSelection",
    "RootSpec",
    "SingleRootSpec",
]
