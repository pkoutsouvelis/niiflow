"""Staging utilities."""

from .file_stager import FileStager
from .stager import (
    FileStagingError,
    StageContext,
    StagedEntry,
    Stager,
    StagingErrorRecord,
    make_entries,
)
from .stager_factory import create_stager, discover_stager_classes
from .types import (
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
)

__all__ = [
    "FileStager",
    "StageContext",
    "StagedEntry",
    "Stager",
    "FileStagingError",
    "StagingErrorRecord",
    "create_stager",
    "discover_stager_classes",
    "make_entries",
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
]
