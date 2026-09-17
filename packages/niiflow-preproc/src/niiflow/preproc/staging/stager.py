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
from concurrent.futures import CancelledError, ThreadPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm


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
        entry_id: Entry identity at the time the error was recorded.
        entry_index: Index of the entry within the batch passed to
            :meth:`Stager.stage`.
        error_type: Exception class name (e.g. ``"FileStagingError"``).
    """

    active: Path
    message: str
    stage: str | None = None
    entry_id: str | None = None
    entry_index: int | None = None
    error_type: str | None = None


@dataclass(frozen=True)
class StagedEntry:
    """One staging unit passed through a :class:`Stager`.

    Attributes:
        active: Stable file anchor for this run.
        params: Current parameter dictionary. Stagers deep-copy and transform
            this mapping.
        id: Logical entry identity. ``None`` is allowed transiently inside a
            custom ``stage_single`` implementation, but :meth:`Stager.stage`
            always returns entries with non-empty IDs unique within that batch.
        errors: Errors accumulated during staging. A stager may skip entries
            that already contain errors.
    """

    active: Path
    params: dict[str, Any]
    id: str
    errors: tuple[StagingErrorRecord, ...] = field(default_factory=tuple)


class FileStagingError(RuntimeError):
    """Raised when :class:`FileStager` fails while resolving a pointer."""


def _reconcile_entry_ids(
    entries: Sequence[StagedEntry],
) -> list[StagedEntry]:
    """Return entries with IDs unique within the collection.

    Existing IDs are reserved before duplicate IDs are repaired, so generated suffixes
    cannot steal an ID already owned elsewhere in the collection.
    """
    reserved: set[str] = set()

    for index, entry in enumerate(entries):
        if not isinstance(entry.id, str) or not entry.id:
            raise ValueError(
                f"`StagedEntry.id` must be a non-empty string at index {index}"
            )
        reserved.add(entry.id)

    used: set[str] = set()
    next_occurrence: dict[str, int] = {}
    reconciled: list[StagedEntry] = []

    for entry in entries:
        if entry.id not in used:
            entry_id = entry.id
        else:
            base = str(entry.active)
            occurrence = next_occurrence.get(base, 2)

            while True:
                candidate = f"{base}#{occurrence}"
                occurrence += 1

                if candidate not in reserved and candidate not in used:
                    break

            next_occurrence[base] = occurrence
            entry_id = candidate

        used.add(entry_id)
        reconciled.append(
            entry if entry.id == entry_id else replace(entry, id=entry_id)
        )

    return reconciled


class Stager(ABC):
    """Base class for staging transformations applied to :class:`StagedEntry` batches.

    Subclasses implement :meth:`stage_single` to transform one entry. The public
    :meth:`stage` method sequences entries (optionally on a thread pool), tolerates per-
    entry failures when :attr:`allow_failed_entries` is ``True``, and reconciles output
    IDs before returning.

    A ``stage_single`` implementation may preserve, replace, or omit an entry ID. After
    :meth:`stage` returns, every entry has a non-empty ID unique within the returned
    collection.
    """

    allow_failed_entries: bool = False

    def stage(
        self, entries: Sequence[StagedEntry], *, num_workers: int = 1
    ) -> list[StagedEntry]:
        """Stage a sequence of entries.

        ``num_workers=1`` (default) runs serially; ``num_workers > 1`` uses a
        :class:`~concurrent.futures.ThreadPoolExecutor`. Output order matches input
        order. Before returning, IDs are reconciled so every output entry has a
        non-empty ID unique within the returned collection.

        Raises:
            TypeError: If ``entries`` is not a sequence of :class:`StagedEntry`
                objects, or :meth:`stage_single` returns a non-entry.
            ValueError: If ``num_workers`` is not an integer ``>= 1``.
        """
        if isinstance(entries, (str, bytes)) or not isinstance(entries, Sequence):
            raise TypeError(
                f"`entries` must be a sequence of `StagedEntry` objects, got "
                f"{type(entries).__name__}"
            )
        if isinstance(num_workers, bool) or not isinstance(num_workers, int):
            raise ValueError(
                f"`num_workers` must be an integer >= 1, got {num_workers!r}"
            )
        if num_workers < 1:
            raise ValueError(
                f"`num_workers` must be an integer >= 1, got {num_workers!r}"
            )

        prepared: list[StagedEntry] = []
        for index, entry in enumerate(entries):
            if not isinstance(entry, StagedEntry):
                raise TypeError(
                    "`entries` must contain only `StagedEntry` objects, got "
                    f"{type(entry).__name__} at index {index}"
                )
            prepared.append(entry)

        if not prepared:
            return []

        progress_kw: dict[str, Any] = {
            "total": len(prepared),
            "desc": type(self).__name__,
            "unit": "entry",
            "disable": None,
            "leave": True,
        }

        if num_workers <= 1:
            staged = self._stage_serial(prepared, progress_kw)
        else:
            staged = self._stage_threaded(prepared, num_workers, progress_kw)

        return _reconcile_entry_ids(staged)

    def _stage_single(self, entry: StagedEntry, *, index: int) -> StagedEntry:
        """Internal wrapper around :meth:`stage_single` with framework handling."""
        try:
            staged_entry = self.stage_single(entry)
        except Exception as exc:
            error = self.make_error(exc, entry, index=index)
            if not self.allow_failed_entries:
                raise
            return replace(entry, errors=entry.errors + (error,))

        if not isinstance(staged_entry, StagedEntry):
            raise TypeError(
                f"{type(self).__name__}.stage_single() must return a "
                f"`StagedEntry`, got {type(staged_entry).__name__} at index "
                f"{index}"
            )
        return staged_entry

    def _stage_serial(
        self, entries: list[StagedEntry], progress_kw: dict[str, Any]
    ) -> list[StagedEntry]:
        staged: list[StagedEntry] = []
        with tqdm(**progress_kw) as progress:
            for index, entry in enumerate(entries):
                staged.append(self._stage_single(entry, index=index))
                progress.update(1)
        return staged

    def _stage_threaded(
        self,
        entries: list[StagedEntry],
        num_workers: int,
        progress_kw: dict[str, Any],
    ) -> list[StagedEntry]:
        staged: list[StagedEntry | None] = [None] * len(entries)
        errors: dict[int, BaseException] = {}
        workers = min(num_workers, len(entries))

        with (
            tqdm(**progress_kw) as progress,
            ThreadPoolExecutor(max_workers=workers) as pool,
        ):
            futures = {
                pool.submit(self._stage_single, entry, index=index): index
                for index, entry in enumerate(entries)
            }
            for fut in as_completed(futures):
                index = futures[fut]
                try:
                    staged[index] = fut.result()
                except CancelledError:
                    continue
                except Exception as exc:
                    errors[index] = exc
                    if not self.allow_failed_entries:
                        for other in futures:
                            other.cancel()
                progress.update(1)

        if errors:
            raise errors[min(errors)]

        result: list[StagedEntry] = []
        for entry in staged:
            if entry is None:
                raise RuntimeError(
                    "Internal staging error: a threaded result was missing."
                )
            result.append(entry)
        return result

    @staticmethod
    def make_entries(
        active_files: Sequence[str | Path],
        params: dict[str, Any] | Sequence[dict[str, Any]],
        *,
        resolve_actives: bool = True,
    ) -> list[StagedEntry]:
        """Create entries from active files and shared/per-entry params."""
        return make_entries(
            active_files,
            params,
            resolve_actives=resolve_actives,
        )

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
            entry_id=entry.id,
            entry_index=index,
            error_type=type(exc).__name__,
            message=str(exc),
        )


def make_entries(
    active_files: Sequence[str | Path],
    params: dict[str, Any] | Sequence[dict[str, Any]],
    *,
    resolve_actives: bool = True,
) -> list[StagedEntry]:
    """Create :class:`StagedEntry` objects from active files and parameter specs.

    Duplicate active paths are valid: each ``(active, params)`` pair is a distinct
    staged unit and IDs are disambiguated deterministically.

    When ``resolve_actives`` is ``True`` (default), active paths are expanded and
    resolved to absolute paths. When ``False``, paths are expanded and converted to
    absolute paths without filesystem resolution.

    No filesystem existence/type checks are performed here. Use staging utilities
    such as :class:`EnsureActivesExist` when physical-path validation is required.

    If ``params`` is a dictionary, it is deep-copied for each entry. If ``params`` is
    a sequence, it must align one-to-one with ``active_files`` and each mapping is
    deep-copied.
    """
    if not active_files:
        raise ValueError("active_files must be a non-empty sequence.")

    if not isinstance(resolve_actives, bool):
        raise TypeError(
            f"`resolve_actives` must be a boolean, got "
            f"{type(resolve_actives).__name__}"
        )

    actives = []
    for item in active_files:
        active = Path(item).expanduser()
        active = active.resolve() if resolve_actives else active.absolute()
        actives.append(active)

    if isinstance(params, dict):
        entry_params: Sequence[dict[str, Any]] = [params] * len(actives)

    elif isinstance(params, Sequence):
        if len(params) != len(actives):
            raise ValueError(
                "When params is a sequence, it must have the same length as "
                f"active_files. Got {len(params)} params for {len(actives)} files."
            )

        if not all(isinstance(item, dict) for item in params):
            raise TypeError("Each per-entry params object must be a dictionary.")

        entry_params = params

    else:
        raise TypeError(
            "`params` must be either a dictionary or a sequence of dictionaries, "
            f"got {type(params).__name__}."
        )

    entries = [
        StagedEntry(
            active=active,
            id=str(active),
            params=deepcopy(entry_params),
        )
        for active, entry_params in zip(actives, entry_params)
    ]

    return _reconcile_entry_ids(entries)
