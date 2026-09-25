"""Tests for :class:`TableStager`."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest

from niiflow.preproc.staging import (
    FileStagingError,
    StagedEntry,
    StagingErrorRecord,
    TableStager,
    create_stager,
    discover_stager_classes,
    make_entries,
)
from niiflow.preproc.staging.stager import Stager


def _stage(stager: Stager, entries: Sequence[StagedEntry]) -> list[StagedEntry]:
    return stager.stage(list(entries))


def _touch(path: Path, content: str = "nii") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path.resolve()


def _write_table(path: Path, header: str, rows: list[str]) -> Path:
    path.write_text(
        "\n".join([header, *rows]) + "\n",
        encoding="utf-8",
    )
    return path


@pytest.fixture
def bids_tree(tmp_path: Path) -> dict[str, Path]:
    root = tmp_path / "dataset"
    active = _touch(
        root / "sub-01" / "ses-pre" / "func" / "sub-01_ses-pre_task-rest_bold.nii.gz"
    )
    return {"root": root.resolve(), "active": active}


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_rejects_non_sequence_pointers(self, tmp_path: Path) -> None:
        table = _write_table(tmp_path / "plan.csv", "id,value", ["a,1"])
        with pytest.raises(TypeError, match="pointers must be a sequence"):
            TableStager(table, "id", {"image": "value"})  # type: ignore[arg-type]

    def test_rejects_string_pointers(self, tmp_path: Path) -> None:
        table = _write_table(tmp_path / "plan.csv", "id,value", ["a,1"])
        with pytest.raises(TypeError, match="pointers must be a sequence"):
            TableStager(table, "id", "image")  # type: ignore[arg-type]

    def test_rejects_non_string_pointer_entries(self, tmp_path: Path) -> None:
        table = _write_table(tmp_path / "plan.csv", "id,value", ["a,1"])
        with pytest.raises(TypeError, match="non-empty strings"):
            TableStager(table, "id", [1])  # type: ignore[list-item]

    def test_rejects_empty_pointer_entry(self, tmp_path: Path) -> None:
        table = _write_table(tmp_path / "plan.csv", "id,value", ["a,1"])
        with pytest.raises(TypeError, match="non-empty strings"):
            TableStager(table, "id", [""])

    def test_rejects_duplicate_pointers(self, tmp_path: Path) -> None:
        table = _write_table(tmp_path / "plan.csv", "id,value", ["a,1"])
        with pytest.raises(ValueError, match="Duplicate table pointer"):
            TableStager(table, "id", ["image", "image"])

    @pytest.mark.parametrize("pointers", [None, []])
    def test_rejects_empty_or_missing_pointers(
        self, tmp_path: Path, pointers: list[str] | None
    ) -> None:
        table = _write_table(tmp_path / "plan.csv", "id,value", ["a,1"])
        with pytest.raises((TypeError, ValueError)):
            TableStager(table, "id", pointers)  # type: ignore[arg-type]

    def test_rejects_empty_id_column(self, tmp_path: Path) -> None:
        table = _write_table(tmp_path / "plan.csv", "id,value", ["a,1"])
        with pytest.raises(ValueError, match="id_column"):
            TableStager(table, "", ["value"])

    def test_rejects_id_pattern_without_placeholder(self, tmp_path: Path) -> None:
        table = _write_table(tmp_path / "plan.csv", "id,value", ["a,1"])
        with pytest.raises(ValueError, match="exactly one \\{id\\}"):
            TableStager(table, "id", ["value"], id_pattern="sub-01")

    def test_missing_table_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="does not exist"):
            TableStager(tmp_path / "missing.csv", "id", ["image"])

    def test_missing_id_column_raises(self, tmp_path: Path) -> None:
        table = _write_table(tmp_path / "plan.csv", "path,value", ["a,1"])
        with pytest.raises(ValueError, match="Missing required column"):
            TableStager(table, "id", ["value"])

    def test_headerless_table_raises(self, tmp_path: Path) -> None:
        table = tmp_path / "plan.csv"
        table.write_text("", encoding="utf-8")
        with pytest.raises(ValueError, match="No header row"):
            TableStager(table, "id", ["image"])

    def test_blank_ids_are_skipped(
        self, tmp_path: Path, bids_tree: dict[str, Path]
    ) -> None:
        active = bids_tree["active"]
        table = _write_table(
            tmp_path / "plan.csv",
            "id,value",
            [",ignored", f"{active},kept"],
        )
        stager = TableStager(table, "id", ["value"])
        entry = make_entries([active], {"value": "value"})[0]

        assert _stage(stager, [entry])[0].params["value"] == "kept"

    def test_duplicate_ids_raise(self, tmp_path: Path) -> None:
        table = _write_table(tmp_path / "plan.csv", "id,value", ["a,1", "a,2"])
        with pytest.raises(ValueError, match="Duplicate id"):
            TableStager(table, "id", ["image"])

    def test_tsv_delimiter(self, tmp_path: Path, bids_tree: dict[str, Path]) -> None:
        active = bids_tree["active"]
        table = _write_table(
            tmp_path / "plan.tsv",
            "id\tload_n4",
            [f"{active}\t/data/n4.nii.gz"],
        )
        stager = TableStager(table, "id", ["steps.find.params.image"])
        entry = make_entries(
            [active],
            {"steps": {"find": {"params": {"image": "load_n4"}}}},
        )[0]

        staged = _stage(stager, [entry])[0]

        assert staged.params["steps"]["find"]["params"]["image"] == "/data/n4.nii.gz"


# ---------------------------------------------------------------------------
# Row matching
# ---------------------------------------------------------------------------


class TestMatching:
    def test_exact_full_path_writes_columns(
        self, tmp_path: Path, bids_tree: dict[str, Path]
    ) -> None:
        active = bids_tree["active"]
        table = _write_table(
            tmp_path / "plan.csv",
            "new_path,load_n4,sitk_compatible",
            [f"{active},/n4/source.nii.gz,True"],
        )
        stager = TableStager(
            table,
            "new_path",
            ["steps.find.params.image", "steps.find.params.compatible"],
        )
        entry = make_entries(
            [active],
            {
                "steps": {
                    "find": {
                        "params": {
                            "image": "load_n4",
                            "compatible": "sitk_compatible",
                        }
                    }
                },
                "kept": "yes",
            },
        )[0]

        staged = _stage(stager, [entry])[0]

        assert staged.active == active
        assert staged.params["steps"]["find"]["params"]["image"] == "/n4/source.nii.gz"
        assert staged.params["steps"]["find"]["params"]["compatible"] == "True"
        assert staged.params["kept"] == "yes"

    def test_exact_match_falls_back_to_name_then_stem(
        self, tmp_path: Path, bids_tree: dict[str, Path]
    ) -> None:
        active = bids_tree["active"]
        table = _write_table(
            tmp_path / "plan.csv",
            "id,value",
            [f"{active.name.removesuffix(''.join(active.suffixes))},from-stem"],
        )
        stager = TableStager(table, "id", ["value"])
        entry = make_entries([active], {"value": "value"})[0]

        staged = _stage(stager, [entry])[0]

        assert staged.params["value"] == "from-stem"

    def test_exact_name_wins_over_stem(
        self, tmp_path: Path, bids_tree: dict[str, Path]
    ) -> None:
        active = bids_tree["active"]
        table = _write_table(
            tmp_path / "plan.csv",
            "id,value",
            [
                f"{active.name},from-name",
                f"{active.name.removesuffix(''.join(active.suffixes))},from-stem",
            ],
        )
        stager = TableStager(table, "id", ["value"])
        entry = make_entries([active], {"value": "value"})[0]

        assert _stage(stager, [entry])[0].params["value"] == "from-name"

    def test_id_pattern_matches_substring_not_exact_path(
        self, tmp_path: Path, bids_tree: dict[str, Path]
    ) -> None:
        active = bids_tree["active"]
        relative = "sub-01/ses-pre/func/sub-01_ses-pre_task-rest_bold.nii.gz"
        table = _write_table(
            tmp_path / "plan.csv",
            "new_path,load_n4",
            [f"{relative},/n4/source.nii.gz"],
        )
        stager = TableStager(
            table,
            "new_path",
            ["image"],
            id_pattern="{id}",
        )
        entry = make_entries([active], {"image": "load_n4"})[0]

        assert _stage(stager, [entry])[0].params["image"] == "/n4/source.nii.gz"

    def test_id_pattern_does_not_match_a_longer_id(
        self, tmp_path: Path, bids_tree: dict[str, Path]
    ) -> None:
        active = bids_tree["active"]
        table = _write_table(
            tmp_path / "plan.csv",
            "id,value",
            ["01,wrong", "011,also-wrong"],
        )
        stager = TableStager(
            table,
            "id",
            ["value"],
            id_pattern="sub-{id}_",
        )
        entry = make_entries([active], {"value": "value"})[0]

        assert _stage(stager, [entry])[0].params["value"] == "wrong"

    def test_multiple_pattern_matches_warn_and_use_table_order(
        self, tmp_path: Path, bids_tree: dict[str, Path]
    ) -> None:
        active = bids_tree["active"]
        table = _write_table(
            tmp_path / "plan.csv",
            "id,value",
            ["sub-01,first", "ses-pre,second"],
        )
        stager = TableStager(
            table,
            "id",
            ["value"],
            id_pattern="{id}",
        )
        entry = make_entries([active], {"value": "value"})[0]

        with pytest.warns(UserWarning, match="using the first"):
            staged = _stage(stager, [entry])[0]

        assert staged.params["value"] == "first"

    def test_id_pattern_miss_raises(
        self, tmp_path: Path, bids_tree: dict[str, Path]
    ) -> None:
        table = _write_table(tmp_path / "plan.csv", "id,value", ["zz,1"])
        stager = TableStager(table, "id", ["value"], id_pattern="{id}")
        entry = make_entries([bids_tree["active"]], {"value": "value"})[0]

        with pytest.raises(FileStagingError, match="No table row"):
            _stage(stager, [entry])

    def test_missing_row_raises(
        self, tmp_path: Path, bids_tree: dict[str, Path]
    ) -> None:
        table = _write_table(tmp_path / "plan.csv", "id,value", ["other,1"])
        stager = TableStager(table, "id", ["value"])
        entry = make_entries([bids_tree["active"]], {"value": "value"})[0]

        with pytest.raises(FileStagingError, match="No table row"):
            _stage(stager, [entry])

    def test_missing_params_path_names_the_pointer(
        self, tmp_path: Path, bids_tree: dict[str, Path]
    ) -> None:
        active = bids_tree["active"]
        table = _write_table(tmp_path / "plan.csv", "id,value", [f"{active},1"])
        stager = TableStager(table, "id", ["steps.find.params.image"])
        entry = make_entries([active], {})[0]

        with pytest.raises(
            FileStagingError, match="table pointer 'steps.find.params.image'"
        ):
            _stage(stager, [entry])

    def test_rejects_non_string_column_name_at_pointer(
        self, tmp_path: Path, bids_tree: dict[str, Path]
    ) -> None:
        active = bids_tree["active"]
        table = _write_table(tmp_path / "plan.csv", "id,value", [f"{active},kept"])
        stager = TableStager(table, "id", ["value"])
        entry = make_entries([active], {"value": None})[0]

        with pytest.raises(FileStagingError, match="non-empty column name"):
            _stage(stager, [entry])

    def test_unknown_column_names_the_pointer(
        self, tmp_path: Path, bids_tree: dict[str, Path]
    ) -> None:
        active = bids_tree["active"]
        table = _write_table(tmp_path / "plan.csv", "id,value", [f"{active},kept"])
        stager = TableStager(table, "id", ["value"])
        entry = make_entries([active], {"value": "load_n4"})[0]

        with pytest.raises(FileStagingError, match="unknown column 'load_n4'"):
            _stage(stager, [entry])

    def test_allow_failed_entries_records_error_and_continues(
        self, tmp_path: Path, bids_tree: dict[str, Path]
    ) -> None:
        good = bids_tree["active"]
        bad = _touch(bids_tree["root"] / "sub-02" / "func" / "missing.nii.gz")
        table = _write_table(tmp_path / "plan.csv", "id,value", [f"{good},kept"])
        entries = make_entries([bad, good], [{"value": "value"}, {"value": "value"}])
        stager = TableStager(table, "id", ["value"], allow_failed_entries=True)

        staged = _stage(stager, entries)

        assert staged[0].errors
        assert staged[0].errors[0].error_type == "FileStagingError"
        assert staged[0].errors[0].entry_index == 0
        assert staged[1].params["value"] == "kept"
        assert not staged[1].errors

    def test_entries_with_errors_are_left_unchanged(
        self, tmp_path: Path, bids_tree: dict[str, Path]
    ) -> None:
        active = bids_tree["active"]
        table = _write_table(tmp_path / "plan.csv", "id,value", [f"{active},kept"])
        entry = make_entries([active], {"value": "value"})[0]
        failed = StagedEntry(
            active=entry.active,
            id=entry.id,
            params=entry.params,
            errors=entry.errors + (StagingErrorRecord(active=active, message="prior"),),
        )

        staged = _stage(
            TableStager(table, "id", ["value"]),
            [failed],
        )[0]

        assert staged.params["value"] == "value"
        assert staged.errors == failed.errors

    def test_discovered_by_stager_factory(
        self, tmp_path: Path, bids_tree: dict[str, Path]
    ) -> None:
        active = bids_tree["active"]
        table = _write_table(tmp_path / "plan.csv", "id,value", [f"{active},kept"])
        assert discover_stager_classes()["TableStager"] is TableStager

        stager = create_stager(
            "TableStager",
            {
                "table_path": table,
                "id_column": "id",
                "pointers": ["value"],
            },
        )
        entry = make_entries([active], {"value": "value"})[0]

        assert _stage(stager, [entry])[0].params["value"] == "kept"
