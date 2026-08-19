"""Tests for :class:`SupportsInputDiscovery` and :class:`SupportsStaging`.

The mixins are exercised on minimal dummy workflows so collection and staging are
covered independently of ``DynamicProcessingWorkflow``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from niiflow.preproc.staging import FileStager, StagedEntry, Stager
from niiflow.preproc.workflows import PlannableWorkflow, RunPlan
from niiflow.preproc.workflows.mixins import SupportsInputDiscovery, SupportsStaging


class DiscoveringWorkflow(SupportsInputDiscovery, PlannableWorkflow):
    """Plannable workflow that only collects active files."""

    @staticmethod
    def process_single(entry: StagedEntry) -> None:
        return None

    def plan(self, source: Any) -> RunPlan:
        return RunPlan(
            entries=tuple(
                StagedEntry(active=active, params={})
                for active in self.collect_active_files(source)
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
        self.configure_staging(
            staging_params=staging_params,
            entry_params=entry_params,
        )

    @staticmethod
    def process_single(entry: StagedEntry) -> None:
        return None

    def plan(self, source: Any) -> RunPlan:
        return self.stage_active_files(list(source))


class BareDiscovery(SupportsInputDiscovery):
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


class TestCollectActiveFiles:
    def test_single_path_is_resolved(
        self, discovering: DiscoveringWorkflow, tmp_path: Path
    ) -> None:
        active = _touch(tmp_path / "a.nii.gz")
        assert discovering.collect_active_files(active) == [active.resolve()]

    def test_string_path_is_accepted(
        self, discovering: DiscoveringWorkflow, tmp_path: Path
    ) -> None:
        active = _touch(tmp_path / "a.nii.gz")
        assert discovering.collect_active_files(str(active)) == [active.resolve()]

    def test_sequence_preserves_first_seen_order_and_dedups(
        self, discovering: DiscoveringWorkflow, tmp_path: Path
    ) -> None:
        first = _touch(tmp_path / "b.nii.gz")
        second = _touch(tmp_path / "a.nii.gz")

        found = discovering.collect_active_files([first, second, first])

        assert found == [first.resolve(), second.resolve()]

    def test_rejects_missing_path(
        self, discovering: DiscoveringWorkflow, tmp_path: Path
    ) -> None:
        with pytest.raises(FileNotFoundError):
            discovering.collect_active_files(tmp_path / "nope.nii.gz")

    def test_from_file_strict_false_skips_missing(
        self, discovering: DiscoveringWorkflow, tmp_path: Path
    ) -> None:
        existing = _touch(tmp_path / "a.nii.gz")
        listing = tmp_path / "files.txt"
        listing.write_text(
            f"{existing}\n{tmp_path / 'missing.nii.gz'}\n", encoding="utf-8"
        )

        found = discovering.collect_active_files(
            {"mode": "from_file", "path": listing, "strict": False}
        )

        assert found == [existing.resolve()]

    def test_from_file_strict_true_raises_on_missing(
        self, discovering: DiscoveringWorkflow, tmp_path: Path
    ) -> None:
        existing = _touch(tmp_path / "a.nii.gz")
        listing = tmp_path / "files.txt"
        listing.write_text(
            f"{existing}\n{tmp_path / 'missing.nii.gz'}\n", encoding="utf-8"
        )

        with pytest.raises(ValueError, match="not an existing file"):
            discovering.collect_active_files({"mode": "from_file", "path": listing})

    def test_from_file_rejects_non_bool_strict(
        self, discovering: DiscoveringWorkflow, tmp_path: Path
    ) -> None:
        listing = tmp_path / "files.txt"
        listing.write_text("", encoding="utf-8")
        with pytest.raises(TypeError, match="`strict` must be a boolean"):
            discovering.collect_active_files(
                {"mode": "from_file", "path": listing, "strict": "yes"}  # type: ignore[arg-type]
            )

    def test_rejects_directory(
        self, discovering: DiscoveringWorkflow, tmp_path: Path
    ) -> None:
        with pytest.raises(ValueError, match="must be a file"):
            discovering.collect_active_files(tmp_path)

    def test_rejects_empty_sequence(self, discovering: DiscoveringWorkflow) -> None:
        with pytest.raises(ValueError, match="must be non-empty"):
            discovering.collect_active_files([])

    def test_rejects_unsupported_input_type(
        self, discovering: DiscoveringWorkflow
    ) -> None:
        with pytest.raises(TypeError, match="`inputs` must be"):
            discovering.collect_active_files(42)  # type: ignore[arg-type]

    def test_rejects_unsupported_sequence_item(
        self, discovering: DiscoveringWorkflow
    ) -> None:
        with pytest.raises(TypeError, match="Unsupported run-input entry type"):
            discovering.collect_active_files([42])  # type: ignore[list-item]

    def test_rejects_unknown_mapping_mode(
        self, discovering: DiscoveringWorkflow
    ) -> None:
        with pytest.raises(ValueError, match="Unknown/missing run-input mode"):
            discovering.collect_active_files({"mode": "unknown"})  # type: ignore[arg-type]

    def test_from_file_mode_reads_paths(
        self, discovering: DiscoveringWorkflow, tmp_path: Path
    ) -> None:
        actives = [_touch(tmp_path / f"img-{index}.nii.gz") for index in range(2)]
        listing = tmp_path / "files.txt"
        listing.write_text(
            "\n".join(str(active) for active in actives), encoding="utf-8"
        )

        found = discovering.collect_active_files({"mode": "from_file", "path": listing})

        assert found == [active.resolve() for active in actives]

    def test_from_file_mode_requires_path(
        self, discovering: DiscoveringWorkflow
    ) -> None:
        with pytest.raises(ValueError, match="`from_file` input requires `path`"):
            discovering.collect_active_files({"mode": "from_file"})  # type: ignore[arg-type]

    def test_from_file_skip_resolve_filepaths(
        self, discovering: DiscoveringWorkflow, tmp_path: Path
    ) -> None:
        active = _touch(tmp_path / "a.nii.gz")
        listing = tmp_path / "files.txt"
        listing.write_text(str(active) + "\n", encoding="utf-8")

        found = discovering.collect_active_files(
            {
                "mode": "from_file",
                "path": listing,
                "skip_resolve_filepaths": True,
            }
        )

        assert found == [Path(str(active))]

    def test_from_file_rejects_non_bool_skip_resolve_filepaths(
        self, discovering: DiscoveringWorkflow, tmp_path: Path
    ) -> None:
        listing = tmp_path / "files.txt"
        listing.write_text("", encoding="utf-8")
        with pytest.raises(TypeError, match="skip_resolve_filepaths"):
            discovering.collect_active_files(
                {
                    "mode": "from_file",
                    "path": listing,
                    "skip_resolve_filepaths": "yes",
                }  # type: ignore[arg-type]
            )

    def test_search_mode_requires_roots(self, discovering: DiscoveringWorkflow) -> None:
        with pytest.raises(ValueError, match="`search` input requires `roots`"):
            discovering.collect_active_files({"mode": "search"})  # type: ignore[arg-type]

    def test_search_mode_requires_explorer_params(
        self, discovering: DiscoveringWorkflow, tmp_path: Path
    ) -> None:
        with pytest.raises(TypeError, match="mapping `explorer_params`"):
            discovering.collect_active_files({"mode": "search", "roots": tmp_path})  # type: ignore[arg-type]

    def test_save_to_writes_one_path_per_line(
        self, discovering: DiscoveringWorkflow, tmp_path: Path
    ) -> None:
        actives = [_touch(tmp_path / f"img-{index}.nii.gz") for index in range(2)]
        listing = tmp_path / "found.txt"

        discovering.collect_active_files(actives, save_to=listing)

        assert listing.read_text(encoding="utf-8").splitlines() == [
            str(active.resolve()) for active in actives
        ]

    def test_save_to_rejects_non_txt_path(
        self, discovering: DiscoveringWorkflow, tmp_path: Path
    ) -> None:
        active = _touch(tmp_path / "a.nii.gz")
        with pytest.raises(ValueError, match="must be a .txt path"):
            discovering.collect_active_files(active, save_to=tmp_path / "found.json")

    def test_requires_processing_workflow_for_log(self, tmp_path: Path) -> None:
        active = _touch(tmp_path / "a.nii.gz")
        with pytest.raises(
            AttributeError, match="must inherit from ProcessingWorkflow"
        ):
            BareDiscovery().collect_active_files(active)

    def test_rejects_when_all_allow_flags_are_false(
        self, discovering: DiscoveringWorkflow, tmp_path: Path
    ) -> None:
        active = _touch(tmp_path / "a.nii.gz")
        with pytest.raises(ValueError, match="At least one of"):
            discovering.collect_active_files(
                active,
                allow_explicit=False,
                allow_from_file=False,
                allow_search=False,
            )

    def test_allow_explicit_false_rejects_path(
        self, discovering: DiscoveringWorkflow, tmp_path: Path
    ) -> None:
        active = _touch(tmp_path / "a.nii.gz")
        with pytest.raises(ValueError, match="Explicit path inputs are not allowed"):
            discovering.collect_active_files(active, allow_explicit=False)

    def test_allow_from_file_false_rejects_listing(
        self, discovering: DiscoveringWorkflow, tmp_path: Path
    ) -> None:
        active = _touch(tmp_path / "a.nii.gz")
        listing = tmp_path / "files.txt"
        listing.write_text(f"{active}\n", encoding="utf-8")
        with pytest.raises(ValueError, match="from_file inputs are not allowed"):
            discovering.collect_active_files(
                {"mode": "from_file", "path": listing},
                allow_from_file=False,
            )

    def test_allow_search_false_rejects_search(
        self, discovering: DiscoveringWorkflow, tmp_path: Path
    ) -> None:
        with pytest.raises(ValueError, match="search inputs are not allowed"):
            discovering.collect_active_files(
                {
                    "mode": "search",
                    "roots": tmp_path,
                    "explorer_params": {"patterns": "*.nii*"},
                },
                allow_search=False,
            )

    def test_allow_search_false_still_accepts_explicit_and_from_file(
        self, discovering: DiscoveringWorkflow, tmp_path: Path
    ) -> None:
        first = _touch(tmp_path / "a.nii.gz")
        second = _touch(tmp_path / "b.nii.gz")
        listing = tmp_path / "files.txt"
        listing.write_text(f"{second}\n", encoding="utf-8")

        found = discovering.collect_active_files(
            [first, {"mode": "from_file", "path": listing}],
            allow_search=False,
        )

        assert found == [first.resolve(), second.resolve()]

    def test_rejects_non_bool_allow_flag(
        self, discovering: DiscoveringWorkflow, tmp_path: Path
    ) -> None:
        active = _touch(tmp_path / "a.nii.gz")
        with pytest.raises(TypeError, match="`allow_search` must be a boolean"):
            discovering.collect_active_files(active, allow_search="no")  # type: ignore[arg-type]

    def test_collect_explicit_active_file(
        self, discovering: DiscoveringWorkflow, tmp_path: Path
    ) -> None:
        active = _touch(tmp_path / "a.nii.gz")
        assert discovering.collect_explicit_active_file(active) == active.resolve()

    def test_collect_active_files_from_file(
        self, discovering: DiscoveringWorkflow, tmp_path: Path
    ) -> None:
        actives = [_touch(tmp_path / f"img-{index}.nii.gz") for index in range(2)]
        listing = tmp_path / "files.txt"
        listing.write_text(
            "\n".join(str(active) for active in actives), encoding="utf-8"
        )

        found = discovering.collect_active_files_from_file(listing)

        assert found == [active.resolve() for active in actives]

    def test_search_active_files(
        self, discovering: DiscoveringWorkflow, tmp_path: Path
    ) -> None:
        first = _touch(tmp_path / "a.nii.gz")
        second = _touch(tmp_path / "b.nii.gz")
        _touch(tmp_path / "notes.txt")

        found = discovering.search_active_files(tmp_path, {"patterns": "*.nii*"})

        assert found == [first.resolve(), second.resolve()]


class TestConfigureStaging:
    def test_none_staging_params_still_bookends_references(
        self, tmp_path: Path
    ) -> None:
        from niiflow.preproc.staging import (
            ResolveActiveReferences,
            ResolveParamReferences,
        )

        wf = StagingWorkflow(logs_root=tmp_path / "logs")
        assert len(wf._stagers) == 2
        assert isinstance(wf._stagers[0], ResolveActiveReferences)
        assert isinstance(wf._stagers[1], ResolveParamReferences)
        assert wf._staging_params == []
        assert wf._entry_params == {}

    def test_single_mapping_is_bookended(self, tmp_path: Path) -> None:
        from niiflow.preproc.staging import (
            ResolveActiveReferences,
            ResolveParamReferences,
        )

        wf = StagingWorkflow(
            logs_root=tmp_path / "logs",
            staging_params={"stager_name": "FileStager", "params": {"pointers": {}}},
        )
        assert len(wf._stagers) == 3
        assert isinstance(wf._stagers[0], ResolveActiveReferences)
        assert isinstance(wf._stagers[1], FileStager)
        assert isinstance(wf._stagers[2], ResolveParamReferences)

    def test_sequence_builds_a_stager_chain(self, tmp_path: Path) -> None:
        wf = StagingWorkflow(
            logs_root=tmp_path / "logs",
            staging_params=[
                {"stager_name": "FileStager"},
                {"stager_name": "FileStager", "params": {"allow_overwrite": False}},
            ],
        )
        assert len(wf._stagers) == 4
        assert wf._stagers[2].allow_overwrite is False  # type: ignore[attr-defined]

    def test_bookends_are_idempotent_when_user_lists_them(self, tmp_path: Path) -> None:
        from niiflow.preproc.staging import (
            ResolveActiveReferences,
            ResolveParamReferences,
        )

        wf = StagingWorkflow(
            logs_root=tmp_path / "logs",
            staging_params=[
                {"stager_name": "ResolveActiveReferences"},
                {"stager_name": "FileStager", "params": {"pointers": {}}},
                {"stager_name": "ResolveParamReferences"},
            ],
        )
        assert len(wf._stagers) == 3
        assert isinstance(wf._stagers[0], ResolveActiveReferences)
        assert isinstance(wf._stagers[1], FileStager)
        assert isinstance(wf._stagers[2], ResolveParamReferences)

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

    def test_allows_missing_active_without_ensure_stager(self, tmp_path: Path) -> None:
        wf = StagingWorkflow(
            logs_root=tmp_path / "logs",
            entry_params={"subject": "{active.stem}"},
        )
        missing = tmp_path / "planned" / "img.nii.gz"

        plan = wf.stage_active_files([missing])

        assert len(plan.entries) == 1
        assert plan.entries[0].active == missing.resolve()
        assert plan.entries[0].params["subject"] == "img.nii"

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
        wf._stagers = [RecordingStager("first"), RecordingStager("second")]  # type: ignore[assignment]
        active = _touch(tmp_path / "a.nii.gz")

        plan = wf.stage_active_files([active])

        assert plan.entries[0].params["seen"] == ["first", "second"]

    def test_save_to_writes_the_plan(self, tmp_path: Path) -> None:
        wf = StagingWorkflow(logs_root=tmp_path / "logs")
        active = _touch(tmp_path / "a.nii.gz")
        plan_path = tmp_path / "plan.json"

        plan = wf.stage_active_files([active], save_to=plan_path)
        loaded = RunPlan.load(plan_path)

        assert plan_path.is_file()
        assert [entry.active for entry in loaded.entries] == [
            entry.active for entry in plan.entries
        ]

    def test_requires_processing_workflow_for_log(self, tmp_path: Path) -> None:
        bare = BareStaging()
        bare.configure_staging()
        with pytest.raises(
            AttributeError, match="must inherit from ProcessingWorkflow"
        ):
            bare.stage_active_files([_touch(tmp_path / "a.nii.gz")])
