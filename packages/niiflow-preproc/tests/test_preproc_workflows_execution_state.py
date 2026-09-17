"""Tests for persistent workflow execution state."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import duckdb
import pytest

from niiflow.preproc.staging import StagedEntry
from niiflow.preproc.workflows.execution_state import (
    EXECUTION_STATE_VERSION,
    ExecutionState,
    ExecutionStatus,
)


def _entry(tmp_path: Path, entry_id: Any) -> StagedEntry:
    return StagedEntry(
        active=(tmp_path / f"{entry_id}.nii.gz").resolve(),
        params={},
        id=entry_id,
    )


def _state(
    tmp_path: Path,
    *entry_ids: str,
) -> ExecutionState:
    return ExecutionState.from_entries(
        [_entry(tmp_path, entry_id) for entry_id in entry_ids]
    )


def _read_rows(path: Path) -> list[tuple[int, str, str]]:
    conn = duckdb.connect(str(path), read_only=True)
    try:
        return conn.execute(
            "SELECT entry_index, id, status FROM entries ORDER BY entry_index"
        ).fetchall()
    finally:
        conn.close()


def _write_raw_state(
    path: Path,
    *,
    metadata: tuple[tuple[str, str], ...] = (
        ("execution_state_version", str(EXECUTION_STATE_VERSION)),
    ),
    rows: tuple[tuple[int, object, object], ...] = (),
    id_type: str = "VARCHAR",
) -> None:
    conn = duckdb.connect(str(path))
    try:
        conn.execute("CREATE TABLE meta (key VARCHAR, value VARCHAR)")
        if metadata:
            conn.executemany("INSERT INTO meta VALUES (?, ?)", metadata)
        conn.execute(f"""CREATE TABLE entries (
                    entry_index INTEGER,
                    id {id_type},
                    status VARCHAR
                )""")
        if rows:
            conn.executemany("INSERT INTO entries VALUES (?, ?, ?)", rows)
    finally:
        conn.close()


class TestExecutionStateInitialization:
    def test_from_entries_preserves_order_and_starts_pending(
        self,
        tmp_path: Path,
    ) -> None:
        entries = [
            _entry(tmp_path, "third"),
            _entry(tmp_path, "first"),
            _entry(tmp_path, "second"),
        ]

        state = ExecutionState.from_entries(entries)

        assert tuple(state.statuses) == ("third", "first", "second")
        assert state.statuses == {
            "third": ExecutionStatus.PENDING,
            "first": ExecutionStatus.PENDING,
            "second": ExecutionStatus.PENDING,
        }
        assert state.path is None

    def test_from_entries_accepts_empty_sequence(self) -> None:
        state = ExecutionState.from_entries([])
        assert state.statuses == {}
        assert state.select() == ()

    @pytest.mark.parametrize("entry_id", ["", None, 1])
    def test_from_entries_validates_id_type_and_value(
        self,
        tmp_path: Path,
        entry_id: object,
    ) -> None:
        with pytest.raises(ValueError, match="non-empty string `id`"):
            ExecutionState.from_entries([_entry(tmp_path, entry_id)])

    def test_from_entries_rejects_duplicate_ids(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match=r"duplicate `same`"):
            ExecutionState.from_entries(
                [_entry(tmp_path, "same"), _entry(tmp_path, "same")]
            )

    def test_from_entries_rejects_non_entries(self) -> None:
        with pytest.raises(TypeError, match="only `StagedEntry`"):
            ExecutionState.from_entries(["bad"])  # type: ignore[list-item]


class TestExecutionStateQueries:
    def test_statuses_is_an_independent_snapshot(self, tmp_path: Path) -> None:
        state = _state(tmp_path, "one")
        snapshot = state.statuses

        snapshot["one"] = ExecutionStatus.FAILURE  # type: ignore[index]
        state.update({"one": ExecutionStatus.SUCCESS})

        assert snapshot["one"] is ExecutionStatus.FAILURE
        assert state.statuses["one"] is ExecutionStatus.SUCCESS

    def test_get_status_returns_current_status(self, tmp_path: Path) -> None:
        state = _state(tmp_path, "one")
        state.update({"one": ExecutionStatus.RUNNING})
        assert state.get_status("one") is ExecutionStatus.RUNNING

    def test_get_status_rejects_unknown_id(self, tmp_path: Path) -> None:
        state = _state(tmp_path, "one")
        with pytest.raises(
            KeyError, match="Unknown execution-state entry ID `missing`"
        ):
            state.get_status("missing")

    @pytest.mark.parametrize("entry_id", [1, None, ("one",)])
    def test_get_status_rejects_non_string_id(
        self,
        tmp_path: Path,
        entry_id: object,
    ) -> None:
        state = _state(tmp_path, "one")
        with pytest.raises(KeyError, match="Unknown execution-state entry ID"):
            state.get_status(entry_id)  # type: ignore[arg-type]

    def test_select_returns_original_order(self, tmp_path: Path) -> None:
        state = _state(tmp_path, "one", "two", "three", "four")
        state.update(
            {
                "one": ExecutionStatus.SUCCESS,
                "two": ExecutionStatus.FAILURE,
                "three": ExecutionStatus.SUCCESS,
            }
        )

        assert state.select() == ("one", "two", "three", "four")
        assert state.select([ExecutionStatus.SUCCESS]) == ("one", "three")
        assert state.select(
            (status for status in (ExecutionStatus.PENDING, ExecutionStatus.FAILURE))
        ) == ("two", "four")
        assert state.select([]) == ()

    @pytest.mark.parametrize("status", ["SUCCESS", 1, None])
    def test_select_validates_status_type(
        self,
        tmp_path: Path,
        status: object,
    ) -> None:
        state = _state(tmp_path, "one")
        with pytest.raises(TypeError, match="must be an ExecutionStatus"):
            state.select([status])  # type: ignore[list-item]

    def test_view_summarizes_every_status(self, tmp_path: Path) -> None:
        state = _state(tmp_path, "pending", "running", "success", "failure", "timeout")
        state.update(
            {
                "running": ExecutionStatus.RUNNING,
                "success": ExecutionStatus.SUCCESS,
                "failure": ExecutionStatus.FAILURE,
                "timeout": ExecutionStatus.TIMEOUT,
            }
        )

        assert state.view() == (
            "ExecutionState: 5 entries "
            "(pending=1, running=1, success=1, failure=1, timeout=1)"
        )

    def test_projected_view_uses_only_requested_ids(self, tmp_path: Path) -> None:
        state = _state(tmp_path, "one", "two", "three")
        state.update(
            {
                "one": ExecutionStatus.SUCCESS,
                "two": ExecutionStatus.FAILURE,
            }
        )

        assert state.view(["three", "one"]) == (
            "ExecutionState: 2 entries "
            "(pending=1, running=0, success=1, failure=0, timeout=0)"
        )
        assert state.view([]) == (
            "ExecutionState: 0 entries "
            "(pending=0, running=0, success=0, failure=0, timeout=0)"
        )

    def test_projected_view_rejects_unknown_id(self, tmp_path: Path) -> None:
        state = _state(tmp_path, "one")
        with pytest.raises(KeyError):
            state.view(["one", "missing"])


class TestExecutionStateUpdates:
    @pytest.mark.parametrize("initial", list(ExecutionStatus))
    @pytest.mark.parametrize("updated", list(ExecutionStatus))
    def test_supports_every_enum_transition(
        self,
        tmp_path: Path,
        initial: ExecutionStatus,
        updated: ExecutionStatus,
    ) -> None:
        state = _state(tmp_path, "one")
        state.update({"one": initial})

        state.update({"one": updated})

        assert state.get_status("one") is updated

    def test_sequence_updates_preserve_entry_order(self, tmp_path: Path) -> None:
        state = _state(tmp_path, "one", "two", "three")
        state.update(
            [
                ("three", ExecutionStatus.FAILURE),
                ("one", ExecutionStatus.SUCCESS),
            ]
        )

        assert tuple(state.statuses) == ("one", "two", "three")
        assert state.statuses == {
            "one": ExecutionStatus.SUCCESS,
            "two": ExecutionStatus.PENDING,
            "three": ExecutionStatus.FAILURE,
        }

    def test_empty_and_unchanged_updates_are_noops(self, tmp_path: Path) -> None:
        state = _state(tmp_path, "one")
        state.update({})
        state.update({"one": ExecutionStatus.PENDING})
        assert state.statuses == {"one": ExecutionStatus.PENDING}

    @pytest.mark.parametrize("status", ["SUCCESS", 1, None])
    def test_update_validates_status_type(
        self,
        tmp_path: Path,
        status: object,
    ) -> None:
        state = _state(tmp_path, "one")
        with pytest.raises(TypeError, match="must be an ExecutionStatus"):
            state.update({"one": status})  # type: ignore[dict-item]
        assert state.get_status("one") is ExecutionStatus.PENDING

    def test_update_rejects_unknown_id(self, tmp_path: Path) -> None:
        state = _state(tmp_path, "one")
        with pytest.raises(
            KeyError, match="Unknown execution-state entry ID `missing`"
        ):
            state.update({"missing": ExecutionStatus.SUCCESS})

    @pytest.mark.parametrize("entry_id", [1, None, ("one",)])
    def test_update_rejects_non_string_id(
        self,
        tmp_path: Path,
        entry_id: object,
    ) -> None:
        state = _state(tmp_path, "one")
        with pytest.raises(KeyError, match="Unknown execution-state entry ID"):
            state.update([(entry_id, ExecutionStatus.SUCCESS)])  # type: ignore[list-item]

    def test_rejects_duplicate_sequence_updates(self, tmp_path: Path) -> None:
        state = _state(tmp_path, "one")
        with pytest.raises(
            ValueError,
            match="Duplicate execution-state update for `one`",
        ):
            state.update(
                [
                    ("one", ExecutionStatus.RUNNING),
                    ("one", ExecutionStatus.SUCCESS),
                ]
            )
        assert state.get_status("one") is ExecutionStatus.PENDING

    @pytest.mark.parametrize(
        "updates",
        [
            [
                ("one", ExecutionStatus.SUCCESS),
                ("missing", ExecutionStatus.FAILURE),
            ],
            [
                ("one", ExecutionStatus.SUCCESS),
                ("two", "FAILURE"),
            ],
        ],
    )
    def test_invalid_batch_does_not_partially_mutate_memory(
        self,
        tmp_path: Path,
        updates: list[tuple[str, object]],
    ) -> None:
        state = _state(tmp_path, "one", "two")

        with pytest.raises((KeyError, TypeError)):
            state.update(updates)  # type: ignore[arg-type]

        assert state.statuses == {
            "one": ExecutionStatus.PENDING,
            "two": ExecutionStatus.PENDING,
        }

    def test_invalid_batch_does_not_partially_mutate_database(
        self,
        tmp_path: Path,
    ) -> None:
        state = _state(tmp_path, "one", "two")
        path = state.save(tmp_path / "state.duckdb")

        with pytest.raises(TypeError, match="must be an ExecutionStatus"):
            state.update(
                [
                    ("one", ExecutionStatus.SUCCESS),
                    ("two", "FAILURE"),  # type: ignore[list-item]
                ]
            )
        state.close()

        assert _read_rows(path) == [
            (0, "one", "PENDING"),
            (1, "two", "PENDING"),
        ]


class TestExecutionStateAddEntries:
    def test_appends_new_entries_as_pending_without_changing_existing_statuses(
        self,
        tmp_path: Path,
    ) -> None:
        state = _state(tmp_path, "one", "two")
        state.update({"one": ExecutionStatus.SUCCESS})

        state.add_entries(
            [
                _entry(tmp_path, "three"),
                _entry(tmp_path, "four"),
            ]
        )

        assert tuple(state.statuses) == ("one", "two", "three", "four")
        assert state.statuses == {
            "one": ExecutionStatus.SUCCESS,
            "two": ExecutionStatus.PENDING,
            "three": ExecutionStatus.PENDING,
            "four": ExecutionStatus.PENDING,
        }

    def test_existing_entries_are_ignored_without_resetting_status(
        self,
        tmp_path: Path,
    ) -> None:
        state = _state(tmp_path, "one", "two")
        state.update(
            {
                "one": ExecutionStatus.FAILURE,
                "two": ExecutionStatus.TIMEOUT,
            }
        )

        state.add_entries(
            [
                _entry(tmp_path, "two"),
                _entry(tmp_path, "three"),
                _entry(tmp_path, "one"),
            ]
        )

        assert tuple(state.statuses) == ("one", "two", "three")
        assert state.statuses == {
            "one": ExecutionStatus.FAILURE,
            "two": ExecutionStatus.TIMEOUT,
            "three": ExecutionStatus.PENDING,
        }

    def test_empty_or_entirely_existing_collection_is_a_noop(
        self,
        tmp_path: Path,
    ) -> None:
        state = _state(tmp_path, "one")
        state.update({"one": ExecutionStatus.RUNNING})

        state.add_entries([])
        state.add_entries([_entry(tmp_path, "one")])

        assert state.statuses == {"one": ExecutionStatus.RUNNING}
        assert state.path is None

    @pytest.mark.parametrize(
        ("entries", "error"),
        [
            pytest.param(["bad"], TypeError, id="non-entry"),
            pytest.param(
                [StagedEntry(active=Path("bad"), id="", params={})],
                ValueError,
                id="empty-id",
            ),
            pytest.param(
                [
                    StagedEntry(active=Path("first"), id="duplicate", params={}),
                    StagedEntry(active=Path("second"), id="duplicate", params={}),
                ],
                ValueError,
                id="duplicate-id",
            ),
        ],
    )
    def test_validates_the_complete_collection_before_mutating(
        self,
        tmp_path: Path,
        entries: list[object],
        error: type[Exception],
    ) -> None:
        state = _state(tmp_path, "one")
        state.update({"one": ExecutionStatus.SUCCESS})

        with pytest.raises(error):
            state.add_entries(entries)  # type: ignore[arg-type]

        assert state.statuses == {"one": ExecutionStatus.SUCCESS}

    def test_bound_state_persists_new_entries_with_contiguous_indices(
        self,
        tmp_path: Path,
    ) -> None:
        state = _state(tmp_path, "one", "two")
        state.update({"one": ExecutionStatus.SUCCESS})
        path = state.save(tmp_path / "state.duckdb")

        state.add_entries(
            [
                _entry(tmp_path, "three"),
                _entry(tmp_path, "four"),
            ]
        )
        state.close()

        assert _read_rows(path) == [
            (0, "one", "SUCCESS"),
            (1, "two", "PENDING"),
            (2, "three", "PENDING"),
            (3, "four", "PENDING"),
        ]

    def test_closed_bound_state_reopens_for_persisted_addition(
        self,
        tmp_path: Path,
    ) -> None:
        state = _state(tmp_path, "one")
        path = state.save(tmp_path / "state.duckdb")
        state.close()

        state.add_entries([_entry(tmp_path, "two")])

        assert state._conn is not None
        state.close()
        with ExecutionState.load(path) as restored:
            assert restored.statuses == {
                "one": ExecutionStatus.PENDING,
                "two": ExecutionStatus.PENDING,
            }

    def test_database_failure_rolls_back_without_mutating_memory(
        self,
        tmp_path: Path,
    ) -> None:
        state = _state(tmp_path, "one", "two")
        path = state.save(tmp_path / "state.duckdb")
        state.close()

        conn = duckdb.connect(str(path))
        try:
            conn.execute("UPDATE entries SET entry_index = 2 WHERE id = 'two'")
        finally:
            conn.close()

        state = ExecutionState.load(path)
        try:
            with pytest.raises(duckdb.ConstraintException):
                state.add_entries([_entry(tmp_path, "three")])

            assert state.statuses == {
                "one": ExecutionStatus.PENDING,
                "two": ExecutionStatus.PENDING,
            }
        finally:
            state.close()

        assert _read_rows(path) == [
            (0, "one", "PENDING"),
            (2, "two", "PENDING"),
        ]


class TestExecutionStateConcat:
    def test_concatenates_disjoint_states_in_state_and_entry_order(
        self,
        tmp_path: Path,
    ) -> None:
        first = _state(tmp_path, "one", "two")
        first.update({"one": ExecutionStatus.SUCCESS})
        second = _state(tmp_path, "three", "four")
        second.update(
            {
                "three": ExecutionStatus.FAILURE,
                "four": ExecutionStatus.TIMEOUT,
            }
        )

        combined = ExecutionState.concat([first, second])

        assert tuple(combined.statuses) == ("one", "two", "three", "four")
        assert combined.statuses == {
            "one": ExecutionStatus.SUCCESS,
            "two": ExecutionStatus.PENDING,
            "three": ExecutionStatus.FAILURE,
            "four": ExecutionStatus.TIMEOUT,
        }
        assert combined.path is None

    def test_empty_collection_returns_empty_unbound_state(self) -> None:
        combined = ExecutionState.concat([])

        assert combined.statuses == {}
        assert combined.path is None

    def test_overlapping_id_with_same_status_is_included_once(
        self,
        tmp_path: Path,
    ) -> None:
        first = _state(tmp_path, "one", "shared")
        second = _state(tmp_path, "shared", "two")
        first.update({"shared": ExecutionStatus.RUNNING})
        second.update({"shared": ExecutionStatus.RUNNING})

        combined = ExecutionState.concat([first, second])

        assert tuple(combined.statuses) == ("one", "shared", "two")
        assert combined.get_status("shared") is ExecutionStatus.RUNNING

    def test_conflicting_status_for_overlapping_id_is_rejected(
        self,
        tmp_path: Path,
    ) -> None:
        first = _state(tmp_path, "shared")
        second = _state(tmp_path, "shared")
        first.update({"shared": ExecutionStatus.SUCCESS})
        second.update({"shared": ExecutionStatus.FAILURE})

        with pytest.raises(
            ValueError,
            match="Conflicting execution status for entry shared: SUCCESS != FAILURE",
        ):
            ExecutionState.concat([first, second])

    def test_result_and_sources_are_independent(
        self,
        tmp_path: Path,
    ) -> None:
        first = _state(tmp_path, "one")
        second = _state(tmp_path, "two")
        combined = ExecutionState.concat([first, second])

        first.update({"one": ExecutionStatus.SUCCESS})
        combined.update({"two": ExecutionStatus.FAILURE})

        assert combined.get_status("one") is ExecutionStatus.PENDING
        assert second.get_status("two") is ExecutionStatus.PENDING

    def test_accepts_bound_states_but_does_not_bind_result(
        self,
        tmp_path: Path,
    ) -> None:
        first = _state(tmp_path, "one")
        first.update({"one": ExecutionStatus.SUCCESS})
        first_path = first.save(tmp_path / "first.duckdb")
        first.close()
        second = _state(tmp_path, "two")
        second.update({"two": ExecutionStatus.TIMEOUT})
        second_path = second.save(tmp_path / "second.duckdb")
        second.close()

        with (
            ExecutionState.load(first_path) as loaded_first,
            ExecutionState.load(second_path) as loaded_second,
        ):
            combined = ExecutionState.concat([loaded_first, loaded_second])

        assert combined.statuses == {
            "one": ExecutionStatus.SUCCESS,
            "two": ExecutionStatus.TIMEOUT,
        }
        assert combined.path is None


class TestExecutionStateFork:
    def test_returns_independent_unbound_snapshot(self, tmp_path: Path) -> None:
        state = _state(tmp_path, "one", "two")
        state.update({"one": ExecutionStatus.SUCCESS})
        path = state.save(tmp_path / "state.duckdb")

        forked = state.fork()
        forked.update({"two": ExecutionStatus.FAILURE})

        assert forked is not state
        assert forked.path is None
        assert forked.statuses == {
            "one": ExecutionStatus.SUCCESS,
            "two": ExecutionStatus.FAILURE,
        }
        assert state.path == path
        assert state.statuses == {
            "one": ExecutionStatus.SUCCESS,
            "two": ExecutionStatus.PENDING,
        }
        state.close()


class TestExecutionStatePersistence:
    def test_in_memory_updates_do_not_bind_state(self, tmp_path: Path) -> None:
        state = _state(tmp_path, "one")
        state.update({"one": ExecutionStatus.SUCCESS})
        assert state.path is None
        assert list(tmp_path.glob("*.duckdb")) == []

    def test_save_binds_state_to_resolved_path(self, tmp_path: Path) -> None:
        state = _state(tmp_path, "one")
        relative = tmp_path / "nested" / ".." / "state.duckdb"

        saved = state.save(relative)

        assert saved == (tmp_path / "state.duckdb").resolve()
        assert state.path == saved
        assert saved.is_file()
        state.close()

    def test_save_and_load_round_trip_ids_statuses_and_order(
        self,
        tmp_path: Path,
    ) -> None:
        state = _state(tmp_path, "third", "first", "second")
        state.update(
            {
                "third": ExecutionStatus.TIMEOUT,
                "first": ExecutionStatus.RUNNING,
                "second": ExecutionStatus.FAILURE,
            }
        )
        path = state.save(tmp_path / "state.duckdb")
        state.close()

        loaded = ExecutionState.load(path)
        try:
            assert tuple(loaded.statuses) == ("third", "first", "second")
            assert loaded.statuses == {
                "third": ExecutionStatus.TIMEOUT,
                "first": ExecutionStatus.RUNNING,
                "second": ExecutionStatus.FAILURE,
            }
            assert loaded.path == path.resolve()
        finally:
            loaded.close()

    def test_empty_state_round_trip(self, tmp_path: Path) -> None:
        path = ExecutionState.from_entries([]).save(tmp_path / "empty.duckdb")
        loaded = ExecutionState.load(path)
        try:
            assert loaded.statuses == {}
            assert loaded.select() == ()
        finally:
            loaded.close()

    def test_bound_updates_write_through(self, tmp_path: Path) -> None:
        state = _state(tmp_path, "one", "two")
        path = state.save(tmp_path / "state.duckdb")
        state.update(
            {
                "one": ExecutionStatus.SUCCESS,
                "two": ExecutionStatus.TIMEOUT,
            }
        )
        state.close()

        assert _read_rows(path) == [
            (0, "one", "SUCCESS"),
            (1, "two", "TIMEOUT"),
        ]

    def test_loaded_updates_write_through(self, tmp_path: Path) -> None:
        path = _state(tmp_path, "one").save(tmp_path / "state.duckdb")
        loaded = ExecutionState.load(path)
        loaded.update({"one": ExecutionStatus.FAILURE})
        loaded.close()

        reopened = ExecutionState.load(path)
        try:
            assert reopened.get_status("one") is ExecutionStatus.FAILURE
        finally:
            reopened.close()

    def test_save_does_not_overwrite_by_default(self, tmp_path: Path) -> None:
        path = tmp_path / "state.duckdb"
        original = _state(tmp_path, "original")
        original.save(path)
        original.close()
        replacement = _state(tmp_path, "replacement")

        with pytest.raises(FileExistsError, match="already exists"):
            replacement.save(path)

        loaded = ExecutionState.load(path)
        try:
            assert tuple(loaded.statuses) == ("original",)
        finally:
            loaded.close()

    def test_save_overwrites_when_requested(self, tmp_path: Path) -> None:
        path = tmp_path / "state.duckdb"
        original = _state(tmp_path, "original")
        original.save(path)
        original.close()
        replacement = _state(tmp_path, "replacement")
        replacement.update({"replacement": ExecutionStatus.SUCCESS})

        replacement.save(path, overwrite=True)
        replacement.close()

        loaded = ExecutionState.load(path)
        try:
            assert loaded.statuses == {
                "replacement": ExecutionStatus.SUCCESS,
            }
        finally:
            loaded.close()

    def test_failed_initial_save_is_atomic_and_removes_temporary_file(
        self,
        tmp_path: Path,
    ) -> None:
        path = tmp_path / "state.duckdb"
        invalid = ExecutionState({"one": "PENDING"})  # type: ignore[dict-item]

        with pytest.raises(AttributeError):
            invalid.save(path)

        assert not path.exists()
        assert list(tmp_path.glob(f".{path.name}.*.tmp")) == []

    def test_failed_overwrite_preserves_existing_file(
        self,
        tmp_path: Path,
    ) -> None:
        path = _state(tmp_path, "original").save(tmp_path / "state.duckdb")
        invalid = ExecutionState({"replacement": "SUCCESS"})  # type: ignore[dict-item]

        with pytest.raises(AttributeError):
            invalid.save(path, overwrite=True)

        assert list(tmp_path.glob(f".{path.name}.*.tmp")) == []
        loaded = ExecutionState.load(path)
        try:
            assert loaded.statuses == {
                "original": ExecutionStatus.PENDING,
            }
        finally:
            loaded.close()

    def test_close_then_update_reopens_connection(self, tmp_path: Path) -> None:
        state = _state(tmp_path, "one")
        path = state.save(tmp_path / "state.duckdb")
        state.close()
        assert state._conn is None

        state.update({"one": ExecutionStatus.SUCCESS})

        assert state._conn is not None
        state.close()
        assert _read_rows(path) == [(0, "one", "SUCCESS")]

    def test_context_manager_closes_and_state_can_reopen(
        self,
        tmp_path: Path,
    ) -> None:
        path = _state(tmp_path, "one").save(tmp_path / "state.duckdb")

        with ExecutionState.load(path) as loaded:
            assert loaded._conn is not None
            loaded.update({"one": ExecutionStatus.RUNNING})

        assert loaded._conn is None
        loaded.update({"one": ExecutionStatus.SUCCESS})
        loaded.close()
        assert _read_rows(path) == [(0, "one", "SUCCESS")]

    def test_missing_bound_file_is_reported_when_reopening(
        self,
        tmp_path: Path,
    ) -> None:
        state = _state(tmp_path, "one")
        path = state.save(tmp_path / "state.duckdb")
        state.close()
        path.unlink()

        with pytest.raises(FileNotFoundError, match="no longer exists"):
            state.update({"one": ExecutionStatus.SUCCESS})
        assert state.get_status("one") is ExecutionStatus.PENDING


class TestExecutionStateSchema:
    def test_database_contains_version_and_creation_metadata(
        self,
        tmp_path: Path,
    ) -> None:
        state = _state(tmp_path, "one")
        path = state.save(tmp_path / "state.duckdb")
        state.close()
        conn = duckdb.connect(str(path), read_only=True)
        try:
            metadata = dict(conn.execute("SELECT key, value FROM meta").fetchall())
        finally:
            conn.close()

        assert metadata["execution_state_version"] == str(EXECUTION_STATE_VERSION)
        created_at = datetime.fromisoformat(metadata["created_at"])
        assert created_at.tzinfo is not None

    def test_entries_schema_stores_only_index_id_and_status(
        self,
        tmp_path: Path,
    ) -> None:
        state = _state(tmp_path, "one")
        path = state.save(tmp_path / "state.duckdb")
        state.close()
        conn = duckdb.connect(str(path), read_only=True)
        try:
            columns = conn.execute("PRAGMA table_info('entries')").fetchall()
        finally:
            conn.close()

        assert [column[1] for column in columns] == [
            "entry_index",
            "id",
            "status",
        ]
        assert all(
            "exception" not in column[1].lower()
            and "error" not in column[1].lower()
            and "detail" not in column[1].lower()
            for column in columns
        )

    def test_entries_schema_enforces_id_and_status_constraints(
        self,
        tmp_path: Path,
    ) -> None:
        state = _state(tmp_path, "one")
        path = state.save(tmp_path / "state.duckdb")
        state.close()
        conn = duckdb.connect(str(path))
        try:
            with pytest.raises(duckdb.ConstraintException):
                conn.execute(
                    "INSERT INTO entries VALUES (?, ?, ?)",
                    [1, "one", "SUCCESS"],
                )
            with pytest.raises(duckdb.ConstraintException):
                conn.execute(
                    "INSERT INTO entries VALUES (?, ?, ?)",
                    [1, "two", "UNKNOWN"],
                )
        finally:
            conn.close()


class TestExecutionStateLoadValidation:
    def test_load_rejects_non_duckdb_extension(self, tmp_path: Path) -> None:
        path = tmp_path / "state.db"
        path.write_bytes(b"")
        with pytest.raises(ValueError, match=r"must use a '\.duckdb' file"):
            ExecutionState.load(path)

    def test_save_rejects_non_duckdb_extension(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match=r"must use a '\.duckdb' file"):
            _state(tmp_path, "one").save(tmp_path / "state.db")

    def test_load_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            ExecutionState.load(tmp_path / "missing.duckdb")

    def test_load_corrupt_database(self, tmp_path: Path) -> None:
        path = tmp_path / "corrupt.duckdb"
        path.write_bytes(b"not a DuckDB database")
        with pytest.raises(duckdb.Error):
            ExecutionState.load(path)

    @pytest.mark.parametrize(
        ("metadata", "message"),
        [
            ((), "missing `execution_state_version`"),
            (
                (("execution_state_version", "not-an-int"),),
                "Invalid `execution_state_version`",
            ),
            (
                (("execution_state_version", "999"),),
                "Unsupported execution-state version 999",
            ),
        ],
    )
    def test_load_validates_version_metadata(
        self,
        tmp_path: Path,
        metadata: tuple[tuple[str, str], ...],
        message: str,
    ) -> None:
        path = tmp_path / "state.duckdb"
        _write_raw_state(path, metadata=metadata)

        with pytest.raises(ValueError, match=message):
            ExecutionState.load(path)

    @pytest.mark.parametrize(
        "rows",
        [
            ((0, "", "PENDING"),),
            ((0, None, "PENDING"),),
        ],
    )
    def test_load_rejects_invalid_id_rows(
        self,
        tmp_path: Path,
        rows: tuple[tuple[int, object, object], ...],
    ) -> None:
        path = tmp_path / "state.duckdb"
        _write_raw_state(path, rows=rows)

        with pytest.raises(ValueError, match="invalid `id`"):
            ExecutionState.load(path)

    def test_load_rejects_non_string_id_row(self, tmp_path: Path) -> None:
        path = tmp_path / "state.duckdb"
        _write_raw_state(
            path,
            rows=((0, 7, "PENDING"),),
            id_type="INTEGER",
        )

        with pytest.raises(ValueError, match="invalid `id`"):
            ExecutionState.load(path)

    def test_load_rejects_duplicate_id_rows(self, tmp_path: Path) -> None:
        path = tmp_path / "state.duckdb"
        _write_raw_state(
            path,
            rows=((0, "same", "PENDING"), (1, "same", "SUCCESS")),
        )

        with pytest.raises(ValueError, match="duplicate ID `same`"):
            ExecutionState.load(path)

    @pytest.mark.parametrize("status", ["UNKNOWN", "", None])
    def test_load_rejects_invalid_status_rows(
        self,
        tmp_path: Path,
        status: object,
    ) -> None:
        path = tmp_path / "state.duckdb"
        _write_raw_state(path, rows=((0, "one", status),))

        with pytest.raises(ValueError, match="unknown status"):
            ExecutionState.load(path)
