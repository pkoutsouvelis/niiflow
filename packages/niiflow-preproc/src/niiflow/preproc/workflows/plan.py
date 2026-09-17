"""Run plan persistence and in-memory representation."""

from __future__ import annotations

__all__ = [
    "PLAN_VERSION",
    "RunPlan",
]

import json
import warnings
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb

from niiflow.preproc.staging import StagedEntry, StagingErrorRecord
from niiflow.preproc.staging.stager import _reconcile_entry_ids
from niiflow.preproc.utils.decorators import deprecate
from niiflow.preproc.utils.file import (
    json_safe,
    read_json,
    resolve_path,
    write_json,
)
from .validation import validate_staged_entries

PLAN_VERSION = 2


def _error_to_record(error: StagingErrorRecord) -> dict[str, Any]:
    return {
        "active": str(error.active),
        "message": error.message,
        "stage": error.stage,
        "entry_id": error.entry_id,
        "entry_index": error.entry_index,
        "error_type": error.error_type,
    }


def _error_from_record(data: dict[str, Any]) -> StagingErrorRecord:
    return StagingErrorRecord(
        active=Path(data["active"]),
        message=str(data["message"]),
        stage=data.get("stage"),
        entry_id=data.get("entry_id"),
        entry_index=data.get("entry_index"),
        error_type=data.get("error_type"),
    )


def _entry_to_record(index: int, entry: StagedEntry) -> dict[str, Any]:
    return {
        "entry_index": index,
        "id": entry.id,
        "active": str(entry.active),
        "params": json_safe(entry.params),
        "errors": [_error_to_record(error) for error in entry.errors],
    }


def _entry_from_record(
    record: dict[str, Any],
    *,
    plan_version: int,
) -> StagedEntry:
    entry_id = record.get("id")
    if plan_version == 1 and not entry_id:
        entry_id = str(Path(record["active"]))
    if not isinstance(entry_id, str) or not entry_id:
        raise ValueError("Plan entry `id` must be a non-empty string")

    params = record["params"]
    if isinstance(params, str):
        params = json.loads(params)
    if not isinstance(params, dict):
        raise TypeError("`params` must deserialize to a mapping")

    errors = record.get("errors", [])
    if isinstance(errors, str):
        errors = json.loads(errors)
    if not isinstance(errors, list):
        raise TypeError("`errors` must deserialize to a list")

    return StagedEntry(
        active=Path(record["active"]),
        id=entry_id,
        params=params,
        errors=tuple(_error_from_record(item) for item in errors),
    )


def _read_plan_version(meta: dict[str, str]) -> int:
    raw = meta.get("plan_version")
    if raw is None:
        raise ValueError("Plan file is missing `plan_version` metadata")
    try:
        version = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid `plan_version` metadata: {raw!r}") from exc

    if version == 1:
        warnings.warn(
            "RunPlan version 1 is deprecated and will be removed in a version "
            "0.5.0. Re-save the plan to upgrade it.",
            FutureWarning,
            stacklevel=3,
        )
    elif version != PLAN_VERSION:
        raise ValueError(
            f"Unsupported plan version {version}; expected 1 or {PLAN_VERSION}"
        )

    return version


def _resolve_entry_bounds(start: int, end: int | None, count: int) -> tuple[int, int]:
    """Normalize negative ``start`` / ``end`` and validate the resolved range.

    Bounds use Python-slice semantics (``start`` inclusive, ``end`` exclusive) over plan
    entry indices, including entries that carry staging errors. Out-of-range bounds
    raise instead of silently clamping.
    """
    if not isinstance(start, int) or isinstance(start, bool):
        raise TypeError(f"`start` must be an int, got {type(start).__name__}")
    if end is not None and (not isinstance(end, int) or isinstance(end, bool)):
        raise TypeError(f"`end` must be an int or None, got {type(end).__name__}")

    start_index = start + count if start < 0 else start
    end_index = count if end is None else (end + count if end < 0 else end)

    if not 0 <= start_index <= count:
        raise ValueError(
            f"`start` {start!r} is out of range for {count} entr"
            f"{'y' if count == 1 else 'ies'}; resolved to index {start_index}, "
            f"expected 0..{count}"
        )
    if not 0 <= end_index <= count:
        raise ValueError(
            f"`end` {end!r} is out of range for {count} entr"
            f"{'y' if count == 1 else 'ies'}; resolved to index {end_index}, "
            f"expected 0..{count}"
        )
    if end_index < start_index:
        raise ValueError(
            f"`end` {end!r} (resolved index {end_index}) must not precede "
            f"`start` {start!r} (resolved index {start_index})"
        )
    return start_index, end_index


def _plan_suffix(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix not in {".json", ".duckdb"}:
        raise ValueError(
            f"Unsupported plan file extension {suffix!r}; use '.duckdb' or '.json'"
        )
    return suffix


@dataclass(frozen=True)
class RunPlan:
    """Staged entries ready for execution.

    Inspect with :meth:`view`, persist with :meth:`save` (``.duckdb`` recommended;
    ``.json`` is deprecated and will be removed in v0.5.0), and reload with
    :meth:`load`. Use :meth:`slice` or ranged :meth:`load` to select a contiguous window
    of plan entry indices.
    """

    entries: tuple[StagedEntry, ...]

    def view(self) -> str:
        """Return a human-readable summary of staged entries."""
        total = len(self.entries)
        failed = sum(1 for entry in self.entries if entry.errors)
        runnable = total - failed

        lines = [f"RunPlan: {total} entr{'y' if total == 1 else 'ies'}"]
        if total:
            summary = f" ({runnable} runnable"
            if failed:
                summary += f", {failed} with staging error{'s' if failed != 1 else ''}"
            summary += ")"
            lines[0] += summary

        for index, entry in enumerate(self.entries):
            lines.append("")
            lines.append(f"Entry {index}")
            lines.append(f"  id: {entry.id}")
            lines.append(f"  active: {entry.active}")
            if entry.errors:
                lines.append("  staging errors:")
                for error in entry.errors:
                    stage = error.stage or "unknown"
                    lines.append(f"    - [{stage}] {error.message}")
            lines.append("  params:")
            params_text = json.dumps(
                json_safe(entry.params),
                indent=2,
                sort_keys=True,
            )
            for line in params_text.splitlines():
                lines.append(f"    {line}")

        return "\n".join(lines)

    def slice(self, start: int = 0, end: int | None = None) -> RunPlan:
        """Return a plan containing entries ``[start:end]`` by plan index.

        Indices include staging-failed entries and use the same bound rules as
        :meth:`load` (inclusive ``start``, exclusive ``end``, negative indexing, strict
        out-of-range errors).

        ``plan[start:end]`` is an alias for ``plan.slice(start, end)``.
        """
        start_index, end_index = _resolve_entry_bounds(start, end, len(self.entries))
        if start_index == 0 and end_index == len(self.entries):
            return self
        return RunPlan(entries=self.entries[start_index:end_index])

    def __getitem__(self, key: int | slice) -> RunPlan:
        """Alias for :meth:`slice` using standard indexing / slicing syntax.

        ``plan[i:j]`` is equivalent to ``plan.slice(i, j)``. A single index ``plan[i]``
        returns a one-entry plan (not a bare :class:`StagedEntry`). Step values other
        than ``1`` are not supported.
        """
        if isinstance(key, slice):
            if key.step not in (None, 1):
                raise ValueError(
                    f"RunPlan slicing does not support step={key.step!r}; use step 1"
                )
            start = 0 if key.start is None else key.start
            return self.slice(start=start, end=key.stop)
        if isinstance(key, bool) or not isinstance(key, int):
            raise TypeError(
                f"RunPlan indices must be integers or slices, got {type(key).__name__}"
            )
        if key < 0:
            return self.slice(start=key, end=None if key == -1 else key + 1)
        return self.slice(start=key, end=key + 1)

    def save(self, path: Path | str) -> Path:
        """Write this plan to ``path``.

        Uses DuckDB for ``.duckdb`` files. JSON (``.json``) remains available until
        v0.5.0 but emits :class:`DeprecationWarning`.
        """
        resolved = resolve_path(path)
        resolved.parent.mkdir(parents=True, exist_ok=True)
        suffix = _plan_suffix(resolved)
        if suffix == ".json":
            return self._save_json(resolved)
        return self._save_duckdb(resolved)

    @classmethod
    def load(
        cls,
        path: Path | str,
        *,
        start: int = 0,
        end: int | None = None,
    ) -> RunPlan:
        """Load a plan previously written with :meth:`save`.

        Optional ``start`` / ``end`` select a contiguous window of plan entry indices
        (including staging-failed entries). Current-version ``.duckdb`` plans
        materialize only the selected rows; deprecated version-1 plans are fully loaded
        first so synthesized IDs are reconciled consistently. JSON (``.json``) plans are
        parsed then sliced; that format is deprecated and will be removed in v0.5.0.
        """
        resolved = resolve_path(path)
        if not resolved.is_file():
            raise FileNotFoundError(resolved)
        suffix = _plan_suffix(resolved)
        if suffix == ".json":
            return cls._load_json(resolved, start=start, end=end)
        return cls._load_duckdb(resolved, start=start, end=end)

    @deprecate(remove_in="0.5.0", alternative=".duckdb")
    def _save_json(self, path: Path) -> Path:
        return write_json(
            {
                "plan_version": PLAN_VERSION,
                "created_at": datetime.now(UTC).isoformat(),
                "entries": [
                    _entry_to_record(index, entry)
                    for index, entry in enumerate(self.entries)
                ],
            },
            path,
        )

    @classmethod
    @deprecate(remove_in="0.5.0", alternative=".duckdb")
    def _load_json(
        cls,
        path: Path,
        *,
        start: int = 0,
        end: int | None = None,
    ) -> RunPlan:
        payload = read_json(path)
        meta = (
            {"plan_version": str(payload["plan_version"])}
            if "plan_version" in payload
            else {}
        )
        version = _read_plan_version(meta)
        records = payload.get("entries")
        if not isinstance(records, list):
            raise TypeError("Plan document must contain an `entries` list")

        start_index, end_index = _resolve_entry_bounds(start, end, len(records))

        if version == 1:  # Remove in 0.5.0
            entries = tuple(
                _reconcile_entry_ids(
                    [_entry_from_record(item, plan_version=version) for item in records]
                )
            )
            return cls(entries=entries[start_index:end_index])

        selected = records[start_index:end_index]
        return cls(
            entries=validate_staged_entries(
                tuple(
                    _entry_from_record(item, plan_version=version) for item in selected
                )
            )
        )

    def _save_duckdb(self, path: Path) -> Path:
        if path.exists():
            path.unlink()
        conn = duckdb.connect(str(path))
        try:
            conn.execute("""CREATE TABLE meta ( key VARCHAR PRIMARY KEY,

                         value VARCHAR NOT NULL )
                         """)
            conn.execute("""CREATE TABLE entries ( entry_index INTEGER PRIMARY KEY, id
                         VARCHAR NOT NULL UNIQUE, active VARCHAR NOT NULL, params
                         VARCHAR NOT NULL, errors VARCHAR NOT NULL )""")
            conn.execute(
                "INSERT INTO meta VALUES (?, ?)",
                ["plan_version", str(PLAN_VERSION)],
            )
            conn.execute(
                "INSERT INTO meta VALUES (?, ?)",
                ["created_at", datetime.now(UTC).isoformat()],
            )
            if self.entries:
                rows = []
                for index, entry in enumerate(self.entries):
                    record = _entry_to_record(index, entry)
                    rows.append(
                        (
                            index,
                            entry.id,
                            str(entry.active),
                            json.dumps(record["params"]),
                            json.dumps(record["errors"]),
                        )
                    )
                conn.executemany(
                    """INSERT INTO entries (entry_index, id, active, params, errors)
                    VALUES (?, ?, ?, ?, ?)""",
                    rows,
                )
        finally:
            conn.close()
        return path

    @classmethod
    def _load_duckdb(
        cls,
        path: Path,
        *,
        start: int = 0,
        end: int | None = None,
    ) -> RunPlan:
        conn = duckdb.connect(str(path), read_only=True)
        try:
            meta_rows = conn.execute("SELECT key, value FROM meta").fetchall()
            meta = {row[0]: row[1] for row in meta_rows}
            version = _read_plan_version(meta)

            count_row = conn.execute("SELECT COUNT(*) FROM entries").fetchone()
            count = int(count_row[0]) if count_row is not None else 0
            start_index, end_index = _resolve_entry_bounds(start, end, count)

            if version == 1:  # Remove in 0.5.0
                rows = conn.execute(
                    """SELECT entry_index, active, params, errors FROM entries ORDER BY
                    entry_index"""
                ).fetchall()
            else:
                rows = conn.execute(
                    """SELECT entry_index, id, active, params, errors FROM entries WHERE
                    entry_index >= ?

                    AND entry_index < ? ORDER BY entry_index
                    """,
                    [start_index, end_index],
                ).fetchall()
        finally:
            conn.close()

        if version == 1:  # Remove in 0.5.0
            entries = tuple(
                _reconcile_entry_ids(
                    [
                        _entry_from_record(
                            {
                                "entry_index": row[0],
                                "active": row[1],
                                "params": row[2],
                                "errors": row[3],
                            },
                            plan_version=version,
                        )
                        for row in rows
                    ]
                )
            )
            return cls(entries=entries[start_index:end_index])

        entries = validate_staged_entries(
            tuple(
                _entry_from_record(
                    {
                        "entry_index": row[0],
                        "id": row[1],
                        "active": row[2],
                        "params": row[3],
                        "errors": row[4],
                    },
                    plan_version=version,
                )
                for row in rows
            )
        )
        return cls(entries=entries)
