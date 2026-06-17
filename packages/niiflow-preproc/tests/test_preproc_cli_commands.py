"""Integration tests for :mod:`niiflow.preproc.cli.commands`."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from niiflow.preproc.cli.commands import execute, execute_plan, plan
from niiflow.preproc.config import load_preproc_config
from niiflow.preproc.staging import StagedEntry
from niiflow.preproc.workflows import DynamicPreprocessingWorkflow, RunPlan
from niiflow.preproc.workflows.workflow import ProcessingWorkflow
from niiflow.preproc.workflows.workflow_factory import discover_workflow_classes


def _touch(entry: StagedEntry) -> None:
    out_dir = Path(entry.params["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{entry.active.name}.done").write_text("ok", encoding="utf-8")


def _config(
    tmp_path: Path,
    run_inputs: Path | str,
    *,
    plan_path: Path | str | None = None,
    workflow_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    kwargs = {
        "num_workers": 1,
        "staging_params": {"stager_name": "FileStager", "params": {"pointers": {}}},
        "pipeline_params": {"steps": [], "out_dir": str(tmp_path / "sentinels")},
    }
    if workflow_kwargs:
        kwargs.update(workflow_kwargs)

    config: dict[str, Any] = {
        "workflow": {
            "name": "DynamicPreprocessingWorkflow",
            "kwargs": kwargs,
        },
        "run_inputs": str(run_inputs),
    }
    if plan_path is not None:
        config["artifacts"] = {"plan_path": str(plan_path)}
    return config


def _write_config(path: Path, config: dict[str, Any]) -> Path:
    path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    return path


class TestConfigValidation:
    def test_missing_workflow_raises(self) -> None:
        with pytest.raises(
            ValueError, match="Missing required config section `workflow`"
        ):
            plan({"run_inputs": "data"})

    def test_workflow_must_be_mapping(self) -> None:
        with pytest.raises(TypeError, match="`workflow` must be a mapping"):
            plan({"workflow": "DynamicPreprocessingWorkflow", "run_inputs": "data"})

    def test_missing_workflow_name_raises(self) -> None:
        with pytest.raises(
            ValueError, match="`workflow.name` must be a non-empty string"
        ):
            plan({"workflow": {"kwargs": {}}, "run_inputs": "data"})

    def test_empty_workflow_name_raises(self) -> None:
        with pytest.raises(
            ValueError, match="`workflow.name` must be a non-empty string"
        ):
            plan({"workflow": {"name": "   "}, "run_inputs": "data"})

    def test_invalid_workflow_kwargs_type_raises(self) -> None:
        with pytest.raises(TypeError, match="`workflow.kwargs` must be a mapping"):
            plan(
                {
                    "workflow": {"name": "DynamicPreprocessingWorkflow", "kwargs": []},
                    "run_inputs": "data",
                }
            )

    def test_missing_run_inputs_raises(self, tmp_path: Path) -> None:
        file_path = tmp_path / "img.nii.gz"
        file_path.write_bytes(b"")
        config = _config(tmp_path, file_path)
        del config["run_inputs"]

        with pytest.raises(
            ValueError, match="Missing required config section `run_inputs`"
        ):
            plan(config)

    def test_plan_requires_output_path(self, tmp_path: Path) -> None:
        file_path = tmp_path / "img.nii.gz"
        file_path.write_bytes(b"")
        config = _config(tmp_path, file_path)

        with pytest.raises(ValueError, match="A run-plan output path is required"):
            plan(config)


class TestPlanCommand:
    def test_plan_saves_to_explicit_output(self, tmp_path: Path) -> None:
        file_path = tmp_path / "img.nii.gz"
        file_path.write_bytes(b"")
        plan_path = tmp_path / "plans" / "job.duckdb"
        config = _config(tmp_path, file_path)

        result = plan(config, output=plan_path)

        assert isinstance(result, RunPlan)
        assert len(result.entries) == 1
        assert result.entries[0].active == file_path.resolve()
        assert plan_path.exists()
        assert RunPlan.load(plan_path).entries == result.entries

    def test_plan_saves_to_artifacts_plan_path(self, tmp_path: Path) -> None:
        file_path = tmp_path / "img.nii.gz"
        file_path.write_bytes(b"")
        plan_path = tmp_path / "artifacts" / "job.duckdb"
        config = _config(tmp_path, file_path, plan_path=plan_path)

        result = plan(config)

        assert plan_path.exists()
        assert RunPlan.load(plan_path).entries == result.entries

    def test_plan_does_not_execute_entries(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        file_path = tmp_path / "img.nii.gz"
        file_path.write_bytes(b"")
        plan_path = tmp_path / "job.duckdb"
        config = _config(tmp_path, file_path)
        monkeypatch.setattr(
            DynamicPreprocessingWorkflow, "process_single", staticmethod(_touch)
        )

        plan(config, output=plan_path)

        assert not (tmp_path / "sentinels").exists()

    def test_plan_dry_run_does_not_require_output(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        file_path = tmp_path / "img.nii.gz"
        file_path.write_bytes(b"")
        config = _config(tmp_path, file_path)

        result = plan(config, dry_run=True)

        assert isinstance(result, RunPlan)
        assert len(result.entries) == 1
        assert result.view() in capsys.readouterr().out

    def test_plan_from_loaded_config_file(self, tmp_path: Path) -> None:
        file_path = tmp_path / "img.nii.gz"
        file_path.write_bytes(b"")
        plan_path = tmp_path / "job.duckdb"
        config_path = _write_config(
            tmp_path / "job.json",
            _config(tmp_path, file_path, plan_path=plan_path),
        )

        loaded = load_preproc_config(config_path)
        result = plan(loaded)

        assert plan_path.exists()
        assert len(result.entries) == 1


class TestDryRunFlag:
    def test_plan_dry_run_prints_without_saving(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
    ) -> None:
        file_path = tmp_path / "img.nii.gz"
        file_path.write_bytes(b"")
        plan_path = tmp_path / "artifacts" / "job.duckdb"
        config = _config(tmp_path, file_path, plan_path=plan_path)
        monkeypatch.setattr(
            DynamicPreprocessingWorkflow, "process_single", staticmethod(_touch)
        )

        result = plan(config, dry_run=True)

        assert isinstance(result, RunPlan)
        assert len(result.entries) == 1
        assert result.view() in capsys.readouterr().out
        assert not plan_path.exists()
        assert not (tmp_path / "sentinels").exists()

    def test_execute_dry_run_prints_without_saving_or_executing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
    ) -> None:
        file_path = tmp_path / "img.nii.gz"
        file_path.write_bytes(b"")
        plan_path = tmp_path / "artifacts" / "job.duckdb"
        config = _config(tmp_path, file_path, plan_path=plan_path)
        monkeypatch.setattr(
            DynamicPreprocessingWorkflow, "process_single", staticmethod(_touch)
        )

        result = execute(config, dry_run=True)

        assert isinstance(result, RunPlan)
        assert result.view() in capsys.readouterr().out
        assert not plan_path.exists()
        assert not (tmp_path / "sentinels").exists()

    def test_execute_plan_dry_run_prints_without_executing(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys,
    ) -> None:
        file_path = tmp_path / "img.nii.gz"
        file_path.write_bytes(b"")
        plan_path = tmp_path / "job.duckdb"
        config = _config(tmp_path, file_path)
        monkeypatch.setattr(
            DynamicPreprocessingWorkflow, "process_single", staticmethod(_touch)
        )
        plan(config, output=plan_path)

        result = execute_plan(config, plan_path=plan_path, dry_run=True)

        assert isinstance(result, RunPlan)
        assert result.view() in capsys.readouterr().out
        assert not (tmp_path / "sentinels").exists()


class TestExecuteCommand:
    def test_execute_runs_and_returns_plan(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        file_path = tmp_path / "img.nii.gz"
        file_path.write_bytes(b"")
        config = _config(tmp_path, file_path)
        monkeypatch.setattr(
            DynamicPreprocessingWorkflow, "process_single", staticmethod(_touch)
        )

        result = execute(config)

        assert isinstance(result, RunPlan)
        assert (tmp_path / "sentinels" / "img.nii.gz.done").exists()

    def test_execute_saves_plan_via_cli_option(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        file_path = tmp_path / "img.nii.gz"
        file_path.write_bytes(b"")
        plan_path = tmp_path / "saved.duckdb"
        config = _config(tmp_path, file_path)
        monkeypatch.setattr(
            DynamicPreprocessingWorkflow, "process_single", staticmethod(_touch)
        )

        result = execute(config, save_plan=plan_path)

        assert result is not None
        assert plan_path.exists()
        assert RunPlan.load(plan_path).entries == result.entries

    def test_execute_saves_plan_via_artifacts(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        file_path = tmp_path / "img.nii.gz"
        file_path.write_bytes(b"")
        plan_path = tmp_path / "artifacts" / "job.duckdb"
        config = _config(tmp_path, file_path, plan_path=plan_path)
        monkeypatch.setattr(
            DynamicPreprocessingWorkflow, "process_single", staticmethod(_touch)
        )

        execute(config)

        assert plan_path.exists()

    def test_execute_without_save_path_still_runs(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        file_path = tmp_path / "img.nii.gz"
        file_path.write_bytes(b"")
        config = _config(tmp_path, file_path)
        monkeypatch.setattr(
            DynamicPreprocessingWorkflow, "process_single", staticmethod(_touch)
        )

        result = execute(config)

        assert result is not None
        assert not (tmp_path / "job.duckdb").exists()
        assert (tmp_path / "sentinels" / "img.nii.gz.done").exists()


class TestExecutePlanCommand:
    def test_execute_plan_runs_saved_plan(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        file_path = tmp_path / "img.nii.gz"
        file_path.write_bytes(b"")
        plan_path = tmp_path / "job.duckdb"
        config = _config(tmp_path, file_path)
        monkeypatch.setattr(
            DynamicPreprocessingWorkflow, "process_single", staticmethod(_touch)
        )
        plan(config, output=plan_path)
        assert not (tmp_path / "sentinels").exists()

        result = execute_plan(config, plan_path=plan_path)

        assert isinstance(result, RunPlan)
        assert (tmp_path / "sentinels" / "img.nii.gz.done").exists()

    def test_execute_plan_does_not_require_run_inputs(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        file_path = tmp_path / "img.nii.gz"
        file_path.write_bytes(b"")
        plan_path = tmp_path / "job.duckdb"
        config = _config(tmp_path, file_path)
        monkeypatch.setattr(
            DynamicPreprocessingWorkflow, "process_single", staticmethod(_touch)
        )
        plan(config, output=plan_path)
        del config["run_inputs"]

        execute_plan(config, plan_path=plan_path)

        assert (tmp_path / "sentinels" / "img.nii.gz.done").exists()


class TestNonPlanningWorkflow:
    @pytest.fixture
    def entry_list_workflow(self, monkeypatch: pytest.MonkeyPatch):
        class EntryListWorkflow(ProcessingWorkflow):
            last_run_inputs: Any = None

            def __init__(self, **kwargs: Any) -> None:
                super().__init__(num_workers=1, **kwargs)

            @staticmethod
            def process_single(entry: StagedEntry) -> None:
                return None

            def run(self, entries: Any) -> None:
                EntryListWorkflow.last_run_inputs = entries
                if isinstance(entries, (list, tuple)):
                    self.run_entries(entries)

        registry = discover_workflow_classes()
        registry = {**registry, "EntryListWorkflow": EntryListWorkflow}
        monkeypatch.setattr(
            "niiflow.preproc.workflows.workflow_factory.discover_workflow_classes",
            lambda: registry,
        )
        return EntryListWorkflow

    def test_execute_runs_non_planning_workflow(
        self, entry_list_workflow, tmp_path: Path
    ) -> None:
        entry = StagedEntry(
            active=(tmp_path / "img.nii.gz").resolve(),
            params={"steps": []},
        )
        config = {
            "workflow": {"name": "EntryListWorkflow", "kwargs": {}},
            "run_inputs": [entry],
        }

        result = execute(config)

        assert result is None
        assert entry_list_workflow.last_run_inputs == [entry]

    def test_plan_rejects_non_planning_workflow(
        self, entry_list_workflow, tmp_path: Path
    ) -> None:
        config = {
            "workflow": {"name": "EntryListWorkflow", "kwargs": {}},
            "run_inputs": [],
        }

        with pytest.raises(TypeError, match="does not support planning"):
            plan(config, output=tmp_path / "job.duckdb")

    def test_execute_plan_rejects_non_planning_workflow(
        self, entry_list_workflow, tmp_path: Path
    ) -> None:
        config = {
            "workflow": {"name": "EntryListWorkflow", "kwargs": {}},
        }

        with pytest.raises(TypeError, match="does not support run plans"):
            execute_plan(config, plan_path=tmp_path / "job.duckdb")
