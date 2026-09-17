"""Utility staging operations."""

from __future__ import annotations

__all__ = [
    "EnsureActivesExist",
    "EnsureActiveExists",
]

from niiflow.preproc.utils.decorators import deprecate

from .stager import StagedEntry, Stager
from .validation import ensure_file


class EnsureActivesExist(Stager):
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


class EnsureActiveExists(EnsureActivesExist):
    """Deprecated alias of :class:`EnsureActivesExist`.

    Will be removed in v0.5.0.
    """

    @deprecate(remove_in="0.5.0", alternative="EnsureActivesExist")
    def __init__(self, *, allow_failed_entries: bool = False) -> None:
        super().__init__(allow_failed_entries=allow_failed_entries)
