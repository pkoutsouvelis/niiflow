"""Utility staging operations."""

from __future__ import annotations

__all__ = [
    "ActiveStager",
    "EnsureActiveExists",
]

from dataclasses import replace

from niiflow.preproc.utils.decorators import deprecate

from .stager import StagedEntry, Stager
from .validation import ensure_file


class ActiveStager(Stager):
    """Normalize :attr:`~StagedEntry.active` paths (resolve and/or require a file).

    Place this wherever actives should be standardized — typically early in a chain, and
    again after a stager that invents new actives. Other stagers should not realpath or
    existence-check the active; they operate on params or expand dynamic references.

    With ``allow_failed_entries=False`` (default), the first failing active aborts
    staging. With ``True``, the failure is recorded on the entry and later stagers may
    skip it.
    """

    def __init__(
        self,
        *,
        allow_failed_entries: bool = False,
        resolve: bool = True,
        must_exist: bool = True,
    ) -> None:
        super().__init__(allow_failed_entries=allow_failed_entries)
        self.resolve = bool(resolve)
        self.must_exist = bool(must_exist)

    def stage_single(self, entry: StagedEntry) -> StagedEntry:
        if entry.errors:
            return entry
        active = ensure_file(
            entry.active, must_exist=self.must_exist, resolve=self.resolve
        )
        if active == entry.active:
            return entry
        return replace(entry, active=active, id=str(active))


class EnsureActiveExists(ActiveStager):
    """Deprecated alias of :class:`ActiveStager`.

    Will be removed in v0.5.0. ``resolve_actives`` maps to ``resolve``.
    """

    @deprecate(remove_in="0.5.0", alternative="ActiveStager")
    def __init__(
        self,
        *,
        allow_failed_entries: bool = False,
        resolve_actives: bool = True,
        resolve: bool | None = None,
        must_exist: bool = True,
    ) -> None:
        if resolve is None:
            resolve = resolve_actives
        super().__init__(
            allow_failed_entries=allow_failed_entries,
            resolve=resolve,
            must_exist=must_exist,
        )
