"""Tests for :class:`SupportsFileDiscovery` and :class:`SupportsStaging`.

The mixins are exercised on minimal dummy workflows so discovery and staging are
covered independently of ``DynamicPreprocessingWorkflow``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from niiflow.preproc.staging import FileStager, StagedEntry, Stager
from niiflow.preproc.workflows import PlannableWorkflow, RunPlan
from niiflow.preproc.workflows.mixins import SupportsFileDiscovery, SupportsStaging


class DiscoveringWorkflow(SupportsFileDiscovery, PlannableWorkflow):
    """Plannable workflow that only discovers active files."""

    @staticmethod
    def process_single(entry: StagedEntry) -> None:
        return None

    def plan(self, source: Any) -> RunPlan:
        return RunPlan(
            entries=tuple(
                StagedEntry(active=active, params={})
                for active in self.discover_active_files(source)
            )
        )


class StagingWorkflow(SupportsStaging, PlannableWorkflow):
    """Plannable workflow that only stages already-known active files."""

    def __init__(
        self,
        *,
        staging_params: Any = None,
        entry_params: Any = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.configure_staging(staging_params=staging_params, entry_params=entry_params)

    @staticmethod
    def process_single(entry: StagedEntry) -> None:
        return None

    def plan(self, source: Any) -> RunPlan:
        return self.stage_active_files(list(source))


class BareDiscovery(SupportsFileDiscovery):
    """Mixin used without :class:`ProcessingWorkflow`, so ``log()`` is missing."""


class BareStaging(SupportsStaging):
    """Mixin used without :class:`ProcessingWorkflow`, so ``log()`` is missing."""


class RecordingStager(Stager):
    """Stager that appends its own name to every entry's ``params['seen']``."""

    def __init__(self, name: str = "recording") -> None:
        self.name = name

    def stage_single(self, entry: StagedEntry) -> StagedEntry:
        seen = [*entry.params.get("seen", []), self.name]
        return StagedEntry(
            active=entry.active,
            params={**entry.params, "seen": seen},
            errors=entry.errors,
        )


@pytest.fixture
def discovering(tmp_path: Path) -> DiscoveringWorkflow:
    return DiscoveringWorkflow(logs_root=tmp_path / "logs")


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")
    return path


class TestDiscoverActiveFiles:
    def test_single_path_is_resolved(
        self, discovering: DiscoveringWorkflow, tmp_path: Path
    ) -> None:
        active = _touch(tmp_path / "a.nii.gz")
        assert discovering.discover_active_files(active) == [active.resolve()]

    def test_string_path_is_accepted(
        self, discovering: DiscoveringWorkflow, tmp_path: Path
    ) -> None:
        active = _touch(tmp_path / "a.nii.gz")
        assert discovering.discover_active_files(str(active)) == [active.resolve()]

    def test_sequence_preserves_first_seen_order_and_dedups(
        self, discovering: DiscoveringWorkflow, tmp_path: Path
    ) -> None:
        first = _touch(tmp_path / "b.nii.gz")
        second = _touch(tmp_path / "a.nii.gz")

        found = discovering.discover_active_files([first, second, first])

        assert found == [first.resolve(), second.resolve()]

    def test_rejects_missing_path(
        self, discovering: DiscoveringWorkflow, tmp_path: Path
    ) -> None:
        with pytest.raises(FileNotFoundError):
            discovering.discover_active_files(tmp_path / "nope.nii.gz")

    def test_rejects_directory(
        self, discovering: DiscoveringWorkflow, tmp_path: Path
    ) -> None:
        with pytest.raises(ValueError, match="must be a file"):
            discovering.discover_active_files(tmp_path)

    def test_rejects_empty_sequence(self, discovering: DiscoveringWorkflow) -> None:
        with pytest.raises(ValueError, match="must be non-empty"):
            discovering.discover_active_files([])

    def test_rejects_unsupported_input_type(
        self, discovering: DiscoveringWorkflow
    ) -> None:
        with pytest.raises(TypeError, match="`files` must be"):
            discovering.discover_active_files(42)  # type: ignore[arg-type]

    def test_rejects_unsupported_sequence_item(
        self, discovering: DiscoveringWorkflow
    ) -> None:
        with pytest.raises(TypeError, match="Unsupported run-input entry type"):
            discovering.discover_active_files([42])  # type: ignore[list-item]

    def test_rejects_unknown_mapping_mode(
        self, discovering: DiscoveringWorkflow
    ) -> None:
        with pytest.raises(ValueError, match="Unknown/missing run-input mode"):
            discovering.discover_active_files({"mode": "telepathy"})

    def test_from_file_mode_reads_paths(
        self, discovering: DiscoveringWorkflow, tmp_path: Path
    ) -> None:
        actives = [_touch(tmp_path / f"img-{index}.nii.gz") for index in range(2)]
        listing = tmp_path / "files.txt"
        listing.write_text(
            "\n".join(str(active) for active in actives), encoding="utf-8"
        )

        found = discovering.discover_active_files(
            {"mode": "from_file", "path": listing}
        )

        assert found == [active.resolve() for active in actives]

    def test_from_file_mode_requires_path(
        self, discovering: DiscoveringWorkflow
    ) -> None:
        with pytest.raises(ValueError, match="`from_file` input requires `path`"):
            discovering.discover_active_files({"mode": "from_file"})

    def test_search_mode_requires_roots(self, discovering: DiscoveringWorkflow) -> None:
        with pytest.raises(ValueError, match="`search` input requires `roots`"):
            discovering.discover_active_files({"mode": "search"})

    def test_search_mode_requires_explorer_params(
        self, discovering: DiscoveringWorkflow, tmp_path: Path
    ) -> None:
        with pytest.raises(TypeError, match="mapping `explorer_params`"):
            discovering.discover_active_files({"mode": "search", "roots": tmp_path})

    def test_save_to_writes_one_path_per_line(
        self, discovering: DiscoveringWorkflow, tmp_path: Path
    ) -> None:
        actives = [_touch(tmp_path / f"img-{index}.nii.gz") for index in range(2)]
        listing = tmp_path / "found.txt"

        discovering.discover_active_files(actives, save_to=listing)

        assert listing.read_text(encoding="utf-8").splitlines() == [
            str(active.resolve()) for active in actives
        ]

    def test_save_to_rejects_non_txt_path(
        self, discovering: DiscoveringWorkflow, tmp_path: Path
    ) -> None:
        active = _touch(tmp_path / "a.nii.gz")
        with pytest.raises(ValueError, match="must be a .txt path"):
            discovering.discover_active_files(active, save_to=tmp_path / "found.json")

    def test_requires_processing_workflow_for_log(self, tmp_path: Path) -> None:
        active = _touch(tmp_path / "a.nii.gz")
        with pytest.raises(
            AttributeError, match="must inherit from ProcessingWorkflow"
        ):
            BareDiscovery().discover_active_files(active)


class TestConfigureStaging:
    def test_none_staging_params_means_no_stagers(self, tmp_path: Path) -> None:
        wf = StagingWorkflow(logs_root=tmp_path / "logs")
        assert wf._stagers == []
        assert wf._entry_params == {}

    def test_single_mapping_builds_one_stager(self, tmp_path: Path) -> None:
        wf = StagingWorkflow(
            logs_root=tmp_path / "logs",
            staging_params={"stager_name": "FileStager", "params": {"pointers": {}}},
        )
        assert len(wf._stagers) == 1
        assert isinstance(wf._stagers[0], FileStager)

    def test_sequence_builds_a_stager_chain(self, tmp_path: Path) -> None:
        wf = StagingWorkflow(
            logs_root=tmp_path / "logs",
            staging_params=[
                {"stager_name": "FileStager"},
                {"stager_name": "FileStager", "params": {"allow_overwrite": False}},
            ],
        )
        assert len(wf._stagers) == 2
        assert wf._stagers[1].allow_overwrite is False

    def test_rejects_non_mapping_staging_params(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="`staging_params` must be"):
            StagingWorkflow(logs_root=tmp_path / "logs", staging_params="FileStager")

    def test_rejects_non_mapping_staging_params_item(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="item must be a dictionary"):
            StagingWorkflow(logs_root=tmp_path / "logs", staging_params=["FileStager"])

    def test_entry_params_mapping_is_copied(self, tmp_path: Path) -> None:
        params = {"steps": [], "nested": {"key": "value"}}
        wf = StagingWorkflow(logs_root=tmp_path / "logs", entry_params=params)

        params["nested"]["key"] = "mutated"

        assert wf._entry_params == {"steps": [], "nested": {"key": "value"}}

    def test_entry_params_sequence_is_accepted(self, tmp_path: Path) -> None:
        wf = StagingWorkflow(
            logs_root=tmp_path / "logs",
            entry_params=[{"index": 0}, {"index": 1}],
        )
        assert wf._entry_params == [{"index": 0}, {"index": 1}]

    def test_rejects_non_mapping_entry_params(self, tmp_path: Path) -> None:
        with pytest.raises(TypeError, match="`entry_params` must be"):
            StagingWorkflow(logs_root=tmp_path / "logs", entry_params="steps")

    def test_rejects_non_mapping_entry_params_item(self, tmp_path: Path) -> None:
        with pytest.raises(TypeError, match="item must be a dictionary"):
            StagingWorkflow(logs_root=tmp_path / "logs", entry_params=["steps"])


class TestStageActiveFiles:
    def test_builds_one_entry_per_active_file(self, tmp_path: Path) -> None:
        wf = StagingWorkflow(logs_root=tmp_path / "logs")
        actives = [_touch(tmp_path / f"img-{index}.nii.gz") for index in range(3)]

        plan = wf.stage_active_files(actives)

        assert isinstance(plan, RunPlan)
        assert [entry.active for entry in plan.entries] == actives

    def test_entry_params_are_attached_to_every_entry(self, tmp_path: Path) -> None:
        wf = StagingWorkflow(
            logs_root=tmp_path / "logs",
            entry_params={"steps": []},
        )
        actives = [_touch(tmp_path / f"img-{index}.nii.gz") for index in range(2)]

        plan = wf.stage_active_files(actives)

        assert all(entry.params == {"steps": []} for entry in plan.entries)

    def test_stagers_run_in_configured_order(self, tmp_path: Path) -> None:
        wf = StagingWorkflow(logs_root=tmp_path / "logs")
        wf._stagers = [RecordingStager("first"), RecordingStager("second")]
        active = _touch(tmp_path / "a.nii.gz")

        plan = wf.stage_active_files([active])

        assert plan.entries[0].params["seen"] == ["first", "second"]

    def test_save_to_writes_the_plan(self, tmp_path: Path) -> None:
        wf = StagingWorkflow(logs_root=tmp_path / "logs")
        active = _touch(tmp_path / "a.nii.gz")
        plan_path = tmp_path / "plan.json"

        wf.stage_active_files([active], save_to=plan_path)

        assert plan_path.is_file()

    def test_requires_processing_workflow_for_log(self, tmp_path: Path) -> None:
        bare = BareStaging()
        bare.configure_staging()
        with pytest.raises(
            AttributeError, match="must inherit from ProcessingWorkflow"
        ):
            bare.stage_active_files([_touch(tmp_path / "a.nii.gz")])
