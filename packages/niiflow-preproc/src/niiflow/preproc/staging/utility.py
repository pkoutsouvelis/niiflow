"""Utility staging operations."""

from __future__ import annotations

__all__ = [
    "EnsureActiveExists",
]

from .stager import StagedEntry, Stager
from .validation import ensure_file


class EnsureActiveExists(Stager):
    """Require :attr:`~StagedEntry.active` to exist and be a file.

    Place this stager wherever existence must be enforced in a staging chain (for
    example after a stager that invents new actives). With
    ``allow_failed_entries=False`` (default), the first missing or non-file active
    aborts staging. With ``True``, the failure is recorded on the entry and later
    stagers may skip it.
    """

    def __init__(self, *, allow_failed_entries: bool = False) -> None:
        self.allow_failed_entries = bool(allow_failed_entries)

    def stage_single(self, entry: StagedEntry) -> StagedEntry:
        if entry.errors:
            return entry
        ensure_file(entry.active, must_exist=True)
        return entry
