"""Tests for :mod:`niiflow.preproc.cli.main` parser and dispatch."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from niiflow.preproc.cli import commands
from niiflow.preproc.cli.main import build_parser, main
from niiflow.preproc.config import load_config
from niiflow.preproc.staging import StagedEntry
from niiflow.preproc.workflows import DynamicProcessingWorkflow, RunPlan


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
            "pipeline_params": {
                "steps": [],
                "out_dir": str(tmp_path / "sentinels"),
            },
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


def _touch(entry: StagedEntry) -> None:
    out_dir = Path(entry.params["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{entry.active.name}.done").write_text("ok", encoding="utf-8")


class TestBuildParser:
    def test_dynamic_workflow_parses_config(self, tmp_path: Path) -> None:
        config_path = tmp_path / "job.json"

        args = build_parser().parse_args(["dynamic_workflow", str(config_path)])

        assert args.command == "dynamic_workflow"
        assert args.config == config_path
        assert args.debug is False

    def test_debug_flag(self, tmp_path: Path) -> None:
        config_path = tmp_path / "job.json"

        args = build_parser().parse_args(
            ["--debug", "dynamic_workflow", str(config_path)]
        )

        assert args.debug is True
        assert args.command == "dynamic_workflow"

    def test_help_lists_registered_commands(self) -> None:
        help_text = build_parser().format_help()
        assert "dynamic_workflow" in help_text

    def test_missing_subcommand_exits(self) -> None:
        with pytest.raises(SystemExit):
            build_parser().parse_args([])

    def test_unknown_subcommand_exits(self) -> None:
        with pytest.raises(SystemExit):
            build_parser().parse_args(["execute", "job.json"])


class TestMainDispatch:
    @pytest.fixture
    def job_paths(self, tmp_path: Path) -> dict[str, Path]:
        file_path = tmp_path / "img.nii.gz"
        file_path.write_bytes(b"")
        config_path = _write_config(
            tmp_path / "job.json",
            _config(tmp_path, file_path, plan_only=True),
        )
        return {
            "config_path": config_path,
            "file_path": file_path,
            "tmp_path": tmp_path,
        }

    def test_main_passes_loaded_config_to_run(
        self, job_paths: dict[str, Path], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        recorded: list[tuple[str, dict[str, Any]]] = []

        def _record(name: str, config: dict[str, Any]) -> None:
            recorded.append((name, dict(config)))

        monkeypatch.setattr(commands, "run", _record)

        main(["dynamic_workflow", str(job_paths["config_path"])])

        expected = dict(load_config(job_paths["config_path"]))
        assert recorded == [("dynamic_workflow", expected)]


class TestMainEquivalence:
    def test_main_matches_direct_run(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        file_path = tmp_path / "img.nii.gz"
        file_path.write_bytes(b"")
        plan_path = tmp_path / "plan.duckdb"
        config = _config(tmp_path, file_path, save_plan_to=plan_path, plan_only=True)
        config_path = _write_config(tmp_path / "job.json", config)
        monkeypatch.setattr(
            DynamicProcessingWorkflow, "process_single", staticmethod(_touch)
        )

        commands.run("dynamic_workflow", load_config(config_path))
        assert not (tmp_path / "sentinels").exists()
        direct_plan = RunPlan.load(plan_path)

        cli_plan_path = tmp_path / "cli-plan.duckdb"
        cli_config = _config(
            tmp_path, file_path, save_plan_to=cli_plan_path, plan_only=True
        )
        cli_config_path = _write_config(tmp_path / "cli-job.json", cli_config)

        main(["dynamic_workflow", str(cli_config_path)])

        assert cli_plan_path.exists()
        assert RunPlan.load(cli_plan_path).entries == direct_plan.entries
        assert not (tmp_path / "sentinels").exists()

    def test_main_reports_errors_without_debug(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        config_path = tmp_path / "empty.json"
        config_path.write_text("{}", encoding="utf-8")

        with pytest.raises(SystemExit) as exc:
            main(["dynamic_workflow", str(config_path)])

        assert exc.value.code == 1
        captured = capsys.readouterr()
        assert "Error:" in captured.err
        assert "`settings` is required" in captured.err
