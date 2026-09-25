"""Table-backed parameter staging."""

from __future__ import annotations

__all__ = [
    "TableStager",
]

import csv
import warnings
from collections.abc import Sequence
from copy import deepcopy
from pathlib import Path

from niiflow.preproc.utils.file import get_ext, resolve_path
from niiflow.preproc.utils.misc import get_by_dotted_path, set_by_dotted_path
from .stager import FileStagingError, StagedEntry, Stager
from .validation import ensure_file


class TableStager(Stager):
    """Replace selected params with values looked up from a table row.

    The table is read once at construction. ``pointers`` lists dotted ``params``
    paths whose current values are treated as table **column names**. For each
    entry, the active file selects a row via ``id_column`` (same match rules as
    nifti-finder's ``IncludeFromTable``), and each pointer is overwritten with
    that row's cell for the named column.

    Matching:

    - No ``id_pattern``: exact membership of the resolved active path, then its
      basename, then its stem (full suffix via :func:`get_ext`).
    - ``id_pattern``: must contain exactly one ``{id}`` placeholder. A row
      matches when ``pattern`` with ``{id}`` replaced by the row id occurs as a
      substring of the active path. The first matching row in table order is
      used; further matches emit a :class:`UserWarning`.

    The active path is not modified. Rows with an empty id are ignored.
    Duplicate ids raise at construction. A missing row, missing/invalid column
    name at a pointer, or missing column in the table raises
    :class:`FileStagingError`.

    Args:
        table_path: CSV or TSV path. ``.tsv`` uses tabs; any other suffix uses
            commas.
        id_column: Column whose values identify rows.
        pointers: Non-empty sequence of dotted params paths. Each path's value
            must be a non-empty column name string before staging.
        id_pattern: Optional ``{id}`` pattern. ``None`` selects exact matching.
        allow_failed_entries: When ``False``, the first staging error aborts
            :meth:`stage`. When ``True``, the error is recorded and staging
            continues.
    """

    def __init__(
        self,
        table_path: Path | str,
        id_column: str,
        pointers: Sequence[str],
        *,
        id_pattern: str | None = None,
        allow_failed_entries: bool = False,
    ) -> None:
        if not isinstance(id_column, str) or not id_column:
            raise ValueError("`id_column` must be a non-empty string.")
        if id_pattern is not None and (
            not isinstance(id_pattern, str) or id_pattern.count("{id}") != 1
        ):
            raise ValueError("id_pattern must contain exactly one {id} placeholder.")

        self.table_path = resolve_path(table_path)
        self.id_column = id_column
        self.pointers = _validate_pointers(pointers)
        self.id_pattern = id_pattern
        self.allow_failed_entries = bool(allow_failed_entries)

        if self.id_pattern is None:
            self._id_before = ""
            self._id_after = ""
        else:
            self._id_before, self._id_after = self.id_pattern.split("{id}")

        self._rows, self._columns = self._load_rows()

    def stage_single(self, entry: StagedEntry) -> StagedEntry:
        """Replace each pointer's column name with the matched row's cell."""
        if entry.errors:
            return entry

        out_params = deepcopy(entry.params)
        current_pointer: str | None = None
        try:
            active = ensure_file(entry.active, must_exist=False)
            row = self._match_row(active)
            for pointer in self.pointers:
                current_pointer = pointer
                column = get_by_dotted_path(out_params, pointer)
                if not isinstance(column, str) or not column:
                    raise FileStagingError(
                        f"Table pointer {pointer!r} must hold a non-empty column "
                        f"name string, got {column!r}."
                    )
                if column not in self._columns:
                    raise FileStagingError(
                        f"Table pointer {pointer!r} names unknown column "
                        f"{column!r} in {self.table_path}. Found columns: "
                        f"{sorted(self._columns)}."
                    )
                set_by_dotted_path(out_params, pointer, row[column])
            return StagedEntry(
                active=entry.active,
                id=entry.id,
                params=out_params,
                errors=entry.errors,
            )
        except FileStagingError:
            raise
        except Exception as exc:
            if current_pointer is not None:
                message = (
                    f"Failed while resolving table pointer {current_pointer!r}: {exc}"
                )
            else:
                message = f"Failed while staging active file {entry.active}: {exc}"
            raise FileStagingError(message) from exc

    def _load_rows(self) -> tuple[dict[str, dict[str, str]], frozenset[str]]:
        if not self.table_path.exists():
            raise FileNotFoundError(f"Table file {self.table_path} does not exist")

        delim = "\t" if self.table_path.suffix.lower() == ".tsv" else ","
        rows: dict[str, dict[str, str]] = {}

        with self.table_path.open("r", newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle, delimiter=delim)
            fieldnames = reader.fieldnames
            if not fieldnames:
                raise ValueError(f"No header row found in table: {self.table_path}")

            if self.id_column not in fieldnames:
                raise ValueError(
                    f"Missing required column(s) {[self.id_column]} in table: "
                    f"{self.table_path}. Found columns: {fieldnames}"
                )

            columns = frozenset(fieldnames)
            for row in reader:
                row_id = str(row.get(self.id_column, "")).strip()
                if not row_id:
                    continue
                if row_id in rows:
                    raise ValueError(
                        f"Duplicate id {row_id!r} in column {self.id_column!r} "
                        f"of table {self.table_path}."
                    )
                rows[row_id] = {
                    column: str(row.get(column, "")).strip() for column in fieldnames
                }
        return rows, columns

    def _match_row(self, active: Path) -> dict[str, str]:
        active_str = str(active)
        if self.id_pattern is None:
            stem = active.name.replace(get_ext(active), "")
            for candidate in (active_str, active.name, stem):
                row = self._rows.get(candidate)
                if row is not None:
                    return row
            raise FileStagingError(
                f"No table row in {self.table_path} matches active file {active}."
            )

        matched = [
            row_id
            for row_id in self._rows
            if f"{self._id_before}{row_id}{self._id_after}" in active_str
        ]
        if not matched:
            raise FileStagingError(
                f"No table row in {self.table_path} matches active file {active}."
            )
        if len(matched) > 1:
            warnings.warn(
                f"Multiple table ids matched {active_str}; "
                f"using the first ({matched[0]}).",
                UserWarning,
                stacklevel=2,
            )
        return self._rows[matched[0]]


def _validate_pointers(pointers: Sequence[str]) -> list[str]:
    """Validate a non-empty sequence of dotted params paths."""
    if isinstance(pointers, (str, bytes)) or not isinstance(pointers, Sequence):
        raise TypeError(
            f"pointers must be a sequence of strings, got {type(pointers).__name__}."
        )
    if not pointers:
        raise ValueError("pointers must be a non-empty sequence.")

    out: list[str] = []
    seen: set[str] = set()
    for index, pointer in enumerate(pointers):
        if not isinstance(pointer, str) or not pointer:
            raise TypeError(
                f"Pointer entries must be non-empty strings, got {pointer!r} "
                f"at index {index}."
            )
        if pointer in seen:
            raise ValueError(f"Duplicate table pointer {pointer!r}.")
        seen.add(pointer)
        out.append(pointer)
    return out
