"""Core staging abstractions."""

from __future__ import annotations

__all__ = [
    "StagingContext",
    "StagingErrorRecord",
    "StagedEntry",
    "Stager",
    "FileStagingError",
    "make_entries",
]

from collections.abc import Sequence
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class StagingContext:
    """Context available while staging one entry.

    Attributes:
        active: Stable file anchor for this run. Preserved across a staging
            chain; stagers transform ``params``, not the active path.
    """

    active: Path


@dataclass(frozen=True)
class StagingErrorRecord:
    """Structured staging error recorded on a :class:`StagedEntry`.

    Attributes:
        active: Active file anchor for the entry that failed.
        message: Human-readable error description.
        stage: Name of the stager implementation that reported the error.
        entry_index: Index of the entry within the batch passed to
            :meth:`Stager.stage`.
        error_type: Exception class name (e.g. ``"FileStagingError"``).
    """

    active: Path
    message: str
    stage: str | None = None
    entry_index: int | None = None
    error_type: str | None = None


@dataclass(frozen=True)
class StagedEntry:
    """One staging unit passed through a :class:`Stager`.

    Attributes:
        active: Stable file anchor for this run.
        params: Current parameter dictionary. Stagers deep-copy and transform
            this mapping; the active path is never rewritten.
        errors: Errors accumulated during staging. A stager may skip entries
            that already contain errors.
    """

    active: Path
    params: dict[str, Any]
    errors: tuple[StagingErrorRecord, ...] = field(default_factory=tuple)


class FileStagingError(RuntimeError):
    """Raised when :class:`FileStager` fails while resolving a pointer."""


class Stager(ABC):
    """Base class for staging transformations applied to :class:`StagedEntry` batches.

    Subclasses implement :meth:`stage_single` to transform each entry's ``params`` while
    preserving :attr:`StagedEntry.active`. The base :meth:`stage` implementation
    sequences entries, optionally tolerating per-entry failures when
    :attr:`allow_failed_entries` is ``True``.
    """

    allow_failed_entries: bool = False

    def stage(self, entries: Sequence[StagedEntry]) -> list[StagedEntry]:
        """Stage a sequence of entries.

        Active files and any errors already recorded on an entry are preserved so
        multiple stagers can be chained.

        Raises:
            TypeError: If ``entries`` is not a sequence of :class:`StagedEntry`
                objects, or :meth:`stage_single` returns a non-entry.
        """
        if isinstance(entries, (str, bytes)) or not isinstance(entries, Sequence):
            raise TypeError(
                f"`entries` must be a sequence of `StagedEntry` objects, got "
                f"{type(entries).__name__}"
            )

        staged: list[StagedEntry] = []

        for index, entry in enumerate(entries):
            if not isinstance(entry, StagedEntry):
                raise TypeError(
                    "`entries` must contain only `StagedEntry` objects, got "
                    f"{type(entry).__name__} at index {index}"
                )
            try:
                staged_entry = self.stage_single(entry)
            except Exception as exc:
                error = self.make_error(exc, entry, index=index)

                if not self.allow_failed_entries:
                    raise

                staged.append(
                    StagedEntry(
                        active=entry.active,
                        params=entry.params,
                        errors=entry.errors + (error,),
                    )
                )
                continue

            if not isinstance(staged_entry, StagedEntry):
                raise TypeError(
                    f"{type(self).__name__}.stage_single() must return a "
                    f"`StagedEntry`, got {type(staged_entry).__name__} at index "
                    f"{index}"
                )
            staged.append(staged_entry)

        return staged

    @staticmethod
    def make_entries(
        active_files: Sequence[str | Path],
        params: dict[str, Any] | Sequence[dict[str, Any]],
    ) -> list[StagedEntry]:
        """Create entries from active files and shared/per-entry params."""
        return make_entries(active_files, params)

    @abstractmethod
    def stage_single(self, entry: StagedEntry) -> StagedEntry:
        """Stage a single entry."""
        ...

    def make_error(
        self,
        exc: Exception,
        entry: StagedEntry,
        *,
        index: int,
    ) -> StagingErrorRecord:
        """Create a structured error from an exception for a given entry.

        Override for custom error handling.
        """
        return StagingErrorRecord(
            active=entry.active,
            stage=type(self).__name__,
            entry_index=index,
            error_type=type(exc).__name__,
            message=str(exc),
        )


def make_entries(
    active_files: Sequence[str | Path],
    params: dict[str, Any] | Sequence[dict[str, Any]],
) -> list[StagedEntry]:
    """Create :class:`StagedEntry` objects from active files and parameter specs.

    ``active_files`` are stable anchors expanded to absolute paths. Existence is
    not checked here; use :class:`~niiflow.preproc.staging.utility.EnsureActiveExists`
    in a staging chain when actives must exist on disk. Paths that already exist
    as directories are rejected. If ``params`` is a dictionary, a deep copy is
    attached to each entry. If ``params`` is a sequence, it must align one-to-one
    with ``active_files`` and each mapping is deep-copied.
    """
    if not active_files:
        raise ValueError("active_files must be a non-empty sequence.")

    actives = [Path(item).expanduser().resolve() for item in active_files]

    for active in actives:
        if active.exists() and active.is_dir():
            raise ValueError(f"Active path must be a file, got directory: {active}")

    if isinstance(params, dict):
        return [
            StagedEntry(active=active, params=deepcopy(params)) for active in actives
        ]

    if not isinstance(params, Sequence):
        raise TypeError(
            "params must be either a dictionary or a sequence of dictionaries, "
            f"got {type(params).__name__}."
        )

    if len(params) != len(actives):
        raise ValueError(
            "When params is a sequence, it must have the same length as "
            f"active_files. Got {len(params)} params for {len(actives)} files."
        )

    entries: list[StagedEntry] = []
    for active, entry_params in zip(actives, params):
        if not isinstance(entry_params, dict):
            raise TypeError(
                "Each per-entry params object must be a dictionary, got "
                f"{type(entry_params).__name__}."
            )
        entries.append(StagedEntry(active=active, params=deepcopy(entry_params)))

    return entries
