"""Run plan persistence and in-memory representation."""

from __future__ import annotations

__all__ = [
    "PLAN_VERSION",
    "RunPlan",
]

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb

from niiflow.preproc.staging import StagedEntry, StagingErrorRecord
from niiflow.preproc.utils.file import (
    json_safe,
    read_json,
    resolve_path,
    write_json,
)

PLAN_VERSION = 1


def _error_to_record(error: StagingErrorRecord) -> dict[str, Any]:
    return {
        "active": str(error.active),
        "message": error.message,
        "stage": error.stage,
        "entry_index": error.entry_index,
        "error_type": error.error_type,
    }


def _error_from_record(data: dict[str, Any]) -> StagingErrorRecord:
    return StagingErrorRecord(
        active=Path(data["active"]),
        message=str(data["message"]),
        stage=data.get("stage"),
        entry_index=data.get("entry_index"),
        error_type=data.get("error_type"),
    )


def _entry_to_record(index: int, entry: StagedEntry) -> dict[str, Any]:
    return {
        "entry_index": index,
        "active": str(entry.active),
        "params": json_safe(entry.params),
        "errors": [_error_to_record(error) for error in entry.errors],
    }


def _entry_from_record(record: dict[str, Any]) -> StagedEntry:
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
        params=params,
        errors=tuple(_error_from_record(item) for item in errors),
    )


def _read_plan_version(meta: dict[str, str]) -> None:
    raw = meta.get("plan_version")
    if raw is None:
        raise ValueError("Plan file is missing `plan_version` metadata")
    try:
        version = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid `plan_version` metadata: {raw!r}") from exc
    if version != PLAN_VERSION:
        raise ValueError(f"Unsupported plan version {version}; expected {PLAN_VERSION}")


@dataclass(frozen=True)
class RunPlan:
    """Staged entries ready for execution.

    Inspect with :meth:`view`, persist with :meth:`save` (``.duckdb`` for production
    scale, ``.json`` for debug), and reload with :meth:`load`.
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

    def save(self, path: Path | str) -> Path:
        """Write this plan to ``path``.

        Uses DuckDB for ``.duckdb`` files and JSON for ``.json`` files.
        """
        resolved = resolve_path(path)
        resolved.parent.mkdir(parents=True, exist_ok=True)
        suffix = resolved.suffix.lower()
        if suffix == ".json":
            return self._save_json(resolved)
        if suffix == ".duckdb":
            return self._save_duckdb(resolved)
        raise ValueError(
            f"Unsupported plan file extension {suffix!r}; use '.duckdb' or '.json'"
        )

    @classmethod
    def load(cls, path: Path | str) -> RunPlan:
        """Load a plan previously written with :meth:`save`."""
        resolved = resolve_path(path)
        if not resolved.is_file():
            raise FileNotFoundError(resolved)
        suffix = resolved.suffix.lower()
        if suffix == ".json":
            return cls._load_json(resolved)
        if suffix == ".duckdb":
            return cls._load_duckdb(resolved)
        raise ValueError(
            f"Unsupported plan file extension {suffix!r}; use '.duckdb' or '.json'"
        )

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
    def _load_json(cls, path: Path) -> RunPlan:
        payload = read_json(path)
        _read_plan_version({"plan_version": str(payload.get("plan_version", ""))})
        entries = payload.get("entries")
        if not isinstance(entries, list):
            raise TypeError("Plan document must contain an `entries` list")
        return cls(entries=tuple(_entry_from_record(item) for item in entries))

    def _save_duckdb(self, path: Path) -> Path:
        if path.exists():
            path.unlink()
        conn = duckdb.connect(str(path))
        try:
            conn.execute("""CREATE TABLE meta ( key VARCHAR PRIMARY KEY,

                         value VARCHAR NOT NULL )
                         """
                            )
            conn.execute(\
                         """CREATE TABLE entries ( entry_index INTEGER PRIMARY KEY,
                         active VARCHAR NOT NULL, params VARCHAR NOT NULL,

                         errors VARCHAR NOT NULL )
                         """
                            )
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
                            str(entry.active),
                            json.dumps(record["params"]),
                            json.dumps(record["errors"]),
                        )
                    )
                conn.executemany(
                    """INSERT INTO entries (entry_index, active, params, errors) VALUES
                    (?, ?, ?, ?)"""
                                   ,
                    rows,
                )
        finally:
            conn.close()
        return path

    @classmethod
    def _load_duckdb(cls, path: Path) -> RunPlan:
        conn = duckdb.connect(str(path), read_only=True)
        try:
            meta_rows = conn.execute("SELECT key, value FROM meta").fetchall()
            meta = {row[0]: row[1] for row in meta_rows}
            _read_plan_version(meta)

            rows = conn.execute(\
                                """
                SELECT entry_index, active, params, errors
                FROM entries
                ORDER BY entry_index
                """).fetchall()
        finally:
            conn.close()

        entries = tuple(
            _entry_from_record(
                {
                    "entry_index": row[0],
                    "active": row[1],
                    "params": row[2],
                    "errors": row[3],
                }
            )
            for row in rows
        )
        return cls(entries=entries)
