"""Persistent execution state for staged workflow entries."""

from __future__ import annotations

__all__ = [
    "EXECUTION_STATE_VERSION",
    "ExecutionState",
    "ExecutionStatus",
]

from collections import Counter
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any
from uuid import uuid4

import duckdb

from niiflow.preproc.staging import StagedEntry
from niiflow.preproc.utils.file import resolve_path

from .validation import validate_staged_entries

EXECUTION_STATE_VERSION = 1


class ExecutionStatus(StrEnum):
    """Persistent execution status of one staged entry."""

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
    TIMEOUT = "TIMEOUT"


def _read_execution_state_version(meta: Mapping[str, str]) -> int:
    raw = meta.get("execution_state_version")
    if raw is None:
        raise ValueError(
            "Execution-state file is missing `execution_state_version` metadata"
        )

    try:
        version = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid `execution_state_version` metadata: {raw!r}"
        ) from exc

    if version != EXECUTION_STATE_VERSION:
        raise ValueError(
            f"Unsupported execution-state version {version}; "
            f"expected {EXECUTION_STATE_VERSION}"
        )

    return version


def _coerce_status(status: ExecutionStatus) -> ExecutionStatus:
    if not isinstance(status, ExecutionStatus):
        raise TypeError(
            f"`status` must be an ExecutionStatus, got {type(status).__name__}"
        )
    return status


@dataclass
class ExecutionState:
    """Current execution status keyed by stable :class:`StagedEntry` IDs.

    Use :meth:`from_entries` to initialize an in-memory state with every entry
    ``PENDING``. Calling :meth:`save` persists and binds the state to a DuckDB file. A
    state returned by :meth:`load` is likewise bound to its source file.

    Once bound, :meth:`set_status` and :meth:`set_many` synchronously write through to
    DuckDB before updating the in-memory view. The workflow parent process should be the
    sole writer.
    """

    _statuses: dict[str, ExecutionStatus]
    _path: Path | None = field(default=None, init=False, repr=False)
    _conn: duckdb.DuckDBPyConnection | None = field(
        default=None, init=False, repr=False
    )

    @classmethod
    def from_entries(
        cls,
        entries: Sequence[StagedEntry],
    ) -> ExecutionState:
        """Initialize an in-memory state with every entry ``PENDING``."""
        entries = validate_staged_entries(entries)
        return cls(_statuses={entry.id: ExecutionStatus.PENDING for entry in entries})

    @classmethod
    def load(cls, path: Path | str) -> ExecutionState:
        """Load and bind an existing DuckDB execution-state file."""
        resolved = resolve_path(path)
        if resolved.suffix.lower() != ".duckdb":
            raise ValueError("Execution state must use a '.duckdb' file")
        if not resolved.is_file():
            raise FileNotFoundError(resolved)

        conn = duckdb.connect(str(resolved))
        try:
            meta_rows = conn.execute("SELECT key, value FROM meta").fetchall()
            meta = {str(key): str(value) for key, value in meta_rows}
            _read_execution_state_version(meta)

            rows = conn.execute("""SELECT entry_index, id, status
                   FROM entries
                   ORDER BY entry_index""").fetchall()

            statuses: dict[str, ExecutionStatus] = {}
            for index, entry_id, raw_status in rows:
                if not isinstance(entry_id, str) or not entry_id:
                    raise ValueError(
                        f"Execution-state entry at index {index} has an invalid `id`"
                    )
                if entry_id in statuses:
                    raise ValueError(
                        f"Execution-state file contains duplicate ID `{entry_id}`"
                    )
                try:
                    status = ExecutionStatus(raw_status)
                except ValueError as exc:
                    raise ValueError(
                        f"Execution-state entry `{entry_id}` has unknown status "
                        f"{raw_status!r}"
                    ) from exc
                statuses[entry_id] = status

            state = cls(_statuses=statuses)
            state._path = resolved
            state._conn = conn
            return state
        except BaseException:
            conn.close()
            raise

    @classmethod
    def concat(
        cls,
        states: Sequence[ExecutionState],
    ) -> ExecutionState:
        """Concatenate multiple execution states into a single state.

        Entries with the same ID are deduplicated, as long as they share the same
        status, otherwise an error is raised.
        """
        statuses: dict[str, ExecutionStatus] = {}

        for state in states:
            for entry_id, status in state.statuses.items():
                existing = statuses.get(entry_id)

                if existing is None:
                    statuses[entry_id] = status
                elif existing is not status:
                    raise ValueError(
                        f"Conflicting execution status for entry {entry_id}: "
                        f"{existing.value} != {status.value}"
                    )

        return cls(_statuses=statuses)

    @property
    def path(self) -> Path | None:
        """Bound DuckDB path, or ``None`` while the state is in memory only."""
        return self._path

    @property
    def statuses(self) -> Mapping[str, ExecutionStatus]:
        """Read-only snapshot-like view of entry statuses."""
        return self._statuses.copy()

    def get_status(self, entry_id: str) -> ExecutionStatus:
        """Return the current status for ``entry_id``."""
        try:
            return self._statuses[entry_id]
        except KeyError as exc:
            raise KeyError(f"Unknown execution-state entry ID `{entry_id}`") from exc

    def add_entries(
        self,
        entries: Sequence[StagedEntry],
    ) -> None:
        """Add new entries to the execution state.

        New entries are marked with status ``PENDING``; if their ID is already present
        in the state, they are ignored.
        """
        entries = validate_staged_entries(entries)

        new_entries = [entry for entry in entries if entry.id not in self._statuses]
        if not new_entries:
            return

        start_index = len(self._statuses)

        conn = self._connection()
        if conn is not None:
            conn.execute("BEGIN TRANSACTION")
            try:
                conn.executemany(
                    """INSERT INTO entries (entry_index, id, status) VALUES (?, ?, ?)""",
                    [
                        (
                            start_index + offset,
                            entry.id,
                            ExecutionStatus.PENDING.value,
                        )
                        for offset, entry in enumerate(new_entries)
                    ],
                )
                conn.execute("COMMIT")
            except BaseException:
                conn.execute("ROLLBACK")
                raise

        for entry in new_entries:
            self._statuses[entry.id] = ExecutionStatus.PENDING

    def save(
        self,
        path: Path | str,
        *,
        overwrite: bool = False,
    ) -> Path:
        """Persist this state to DuckDB and bind subsequent updates to that file.

        By default an existing file is never overwritten; resume existing state with
        :meth:`load` instead. Once saved, status updates are committed dynamically.
        """
        resolved = resolve_path(path)
        if resolved.suffix.lower() != ".duckdb":
            raise ValueError("Execution state must use a '.duckdb' file")
        resolved.parent.mkdir(parents=True, exist_ok=True)

        if resolved.exists() and not overwrite:
            raise FileExistsError(
                f"Execution-state file already exists: {resolved}. "
                "Use `ExecutionState.load()` to resume it."
            )

        self.close()

        temporary = resolved.with_name(f".{resolved.name}.{uuid4().hex}.tmp")
        conn: duckdb.DuckDBPyConnection | None = None

        try:
            conn = duckdb.connect(str(temporary))
            conn.execute("""CREATE TABLE meta ( key VARCHAR PRIMARY KEY, value VARCHAR
                         NOT NULL )""")
            allowed = ", ".join(f"'{status.value}'" for status in ExecutionStatus)
            conn.execute(f"""CREATE TABLE entries (
                        entry_index INTEGER PRIMARY KEY,
                        id VARCHAR NOT NULL UNIQUE,
                        status VARCHAR NOT NULL
                            CHECK (status IN ({allowed}))
                    )""")
            conn.execute(
                "INSERT INTO meta VALUES (?, ?)",
                [
                    "execution_state_version",
                    str(EXECUTION_STATE_VERSION),
                ],
            )
            conn.execute(
                "INSERT INTO meta VALUES (?, ?)",
                ["created_at", datetime.now(UTC).isoformat()],
            )

            if self._statuses:
                conn.executemany(
                    """INSERT INTO entries (entry_index, id, status) VALUES (?, ?, ?)""",
                    [
                        (index, entry_id, status.value)
                        for index, (entry_id, status) in enumerate(
                            self._statuses.items()
                        )
                    ],
                )

            conn.close()
            conn = None

            temporary.replace(resolved)

            self._path = resolved
            self._conn = duckdb.connect(str(resolved))
            return resolved

        except BaseException:
            if conn is not None:
                conn.close()
            temporary.unlink(missing_ok=True)
            raise

    def update(
        self,
        updates: Mapping[str, ExecutionStatus] | Sequence[tuple[str, ExecutionStatus]],
    ) -> None:
        """Atomically apply several status updates to a persisted state."""
        if isinstance(updates, Mapping):
            items = list(updates.items())
        else:
            items = list(updates)
        if not items:
            return

        normalized: list[tuple[str, ExecutionStatus]] = []
        seen: set[str] = set()

        for entry_id, status in items:
            if entry_id in seen:
                raise ValueError(f"Duplicate execution-state update for `{entry_id}`")
            seen.add(entry_id)

            if entry_id not in self._statuses:
                raise KeyError(f"Unknown execution-state entry ID `{entry_id}`")
            normalized.append((entry_id, _coerce_status(status)))

        changed = [
            (entry_id, status)
            for entry_id, status in normalized
            if self._statuses[entry_id] is not status
        ]
        if not changed:
            return

        conn = self._connection()
        if conn is not None:
            conn.execute("BEGIN TRANSACTION")
            try:
                conn.executemany(
                    "UPDATE entries SET status = ? WHERE id = ?",
                    [(status.value, entry_id) for entry_id, status in changed],
                )
                conn.execute("COMMIT")
            except BaseException:
                conn.execute("ROLLBACK")
                raise

        for entry_id, status in changed:
            self._statuses[entry_id] = status

    def select(
        self,
        statuses: Iterable[ExecutionStatus] | None = None,
    ) -> tuple[str, ...]:
        """Return IDs in original order, optionally filtered by status."""
        if statuses is None:
            return tuple(self._statuses)

        selected = {_coerce_status(status) for status in statuses}
        return tuple(
            entry_id
            for entry_id, status in self._statuses.items()
            if status in selected
        )

    def view(self, entry_ids: Collection[str] | None = None) -> str:
        """Return a compact human-readable state summary."""
        if entry_ids is None:
            statuses = self._statuses.values()
            total = len(self._statuses)
        else:
            statuses = (self._statuses[entry_id] for entry_id in entry_ids)
            total = len(entry_ids)

        counts = Counter(statuses)

        parts = [
            f"{status.value.lower()}={counts.get(status, 0)}"
            for status in ExecutionStatus
        ]

        return f"ExecutionState: {total} entries ({', '.join(parts)})"

    def _connection(self) -> duckdb.DuckDBPyConnection | None:
        """Return the live bound connection, reopening it when necessary."""
        # Use existing connection if already loaded/saved, otherwise open a new one.
        if self._path is None:
            return None
        if self._conn is None:
            if not self._path.is_file():
                raise FileNotFoundError(
                    f"Execution-state file no longer exists: {self._path}"
                )
            self._conn = duckdb.connect(str(self._path))
        return self._conn

    def close(self) -> None:
        """Close the bound DuckDB connection, if any."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> ExecutionState:
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()
