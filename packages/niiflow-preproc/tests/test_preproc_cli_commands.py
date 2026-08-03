"""Tests for :mod:`niiflow.preproc.cli.commands` registry."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from niiflow.preproc.cli.commands import COMMANDS, run
from niiflow.preproc.config import load_config
from niiflow.preproc.staging import StagedEntry
from niiflow.preproc.workflows import DynamicPreprocessingWorkflow, RunPlan
from niiflow.preproc.workflows.dynamic_workflow import dynamic_workflow


def _touch(entry: StagedEntry) -> None:
    out_dir = Path(entry.params["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{entry.active.name}.done").write_text("ok", encoding="utf-8")


def _config(
    tmp_path: Path,
    inputs: Path | str,
    *,
    save_plan_to: Path | str | None = None,
    **extra: Any,
) -> dict[str, Any]:
    config: dict[str, Any] = {
        "settings": {
            "num_workers": 1,
            "staging_params": {"stager_name": "FileStager", "params": {"pointers": {}}},
            "pipeline_params": {"steps": [], "out_dir": str(tmp_path / "sentinels")},
        },
        "inputs": str(inputs),
        **extra,
    }
    if save_plan_to is not None:
        config["save_plan_to"] = str(save_plan_to)
    return config


def _write_config(path: Path, config: dict[str, Any]) -> Path:
    path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    return path


class TestRegistry:
    def test_dynamic_workflow_is_registered(self) -> None:
        assert COMMANDS["dynamic_workflow"] is dynamic_workflow

    def test_unknown_command_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown command 'missing'"):
            run("missing", {})


class TestRunDynamicWorkflow:
    def test_plan_and_execute(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        file_path = tmp_path / "img.nii.gz"
        file_path.write_bytes(b"")
        monkeypatch.setattr(
            DynamicPreprocessingWorkflow, "process_single", staticmethod(_touch)
        )

        run("dynamic_workflow", _config(tmp_path, file_path))
        assert (tmp_path / "sentinels" / "img.nii.gz.done").exists()

    def test_plan_only_saves_without_executing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        file_path = tmp_path / "img.nii.gz"
        file_path.write_bytes(b"")
        plan_path = tmp_path / "plans" / "job.duckdb"
        monkeypatch.setattr(
            DynamicPreprocessingWorkflow, "process_single", staticmethod(_touch)
        )

        run(
            "dynamic_workflow",
            _config(tmp_path, file_path, save_plan_to=plan_path, plan_only=True),
        )

        assert plan_path.exists()
        loaded = RunPlan.load(plan_path)
        assert len(loaded.entries) == 1
        assert loaded.entries[0].active == file_path.resolve()
        assert not (tmp_path / "sentinels").exists()

    def test_dry_run_prints_without_saving_or_executing(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        file_path = tmp_path / "img.nii.gz"
        file_path.write_bytes(b"")
        plan_path = tmp_path / "artifacts" / "job.duckdb"
        monkeypatch.setattr(
            DynamicPreprocessingWorkflow, "process_single", staticmethod(_touch)
        )

        run(
            "dynamic_workflow",
            _config(tmp_path, file_path, save_plan_to=plan_path, dry_run=True),
        )

        captured = capsys.readouterr()
        assert "RunPlan:" in captured.out
        assert str(file_path.resolve()) in captured.out
        assert not plan_path.exists()
        assert not (tmp_path / "sentinels").exists()

    def test_from_plan_executes_saved_plan(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        file_path = tmp_path / "img.nii.gz"
        file_path.write_bytes(b"")
        plan_path = tmp_path / "job.duckdb"
        monkeypatch.setattr(
            DynamicPreprocessingWorkflow, "process_single", staticmethod(_touch)
        )
        run(
            "dynamic_workflow",
            _config(tmp_path, file_path, save_plan_to=plan_path, plan_only=True),
        )
        assert not (tmp_path / "sentinels").exists()

        run(
            "dynamic_workflow",
            {
                "settings": {
                    "num_workers": 1,
                    "pipeline_params": {
                        "steps": [],
                        "out_dir": str(tmp_path / "sentinels"),
                    },
                },
                "from_plan": str(plan_path),
            },
        )

        assert (tmp_path / "sentinels" / "img.nii.gz.done").exists()

    def test_run_from_loaded_config_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        file_path = tmp_path / "img.nii.gz"
        file_path.write_bytes(b"")
        plan_path = tmp_path / "job.duckdb"
        config_path = _write_config(
            tmp_path / "job.json",
            _config(tmp_path, file_path, save_plan_to=plan_path, plan_only=True),
        )
        monkeypatch.setattr(
            DynamicPreprocessingWorkflow, "process_single", staticmethod(_touch)
        )

        run("dynamic_workflow", load_config(config_path))

        assert plan_path.exists()
        assert len(RunPlan.load(plan_path).entries) == 1
        assert not (tmp_path / "sentinels").exists()
