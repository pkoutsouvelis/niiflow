"""Tests for :mod:`niiflow.preproc.cli.main` parser and dispatch."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from niiflow.preproc.cli import commands
from niiflow.preproc.cli.main import build_parser, main
from niiflow.preproc.config import load_preproc_config
from niiflow.preproc.staging import StagedEntry
from niiflow.preproc.workflows import DynamicPreprocessingWorkflow, RunPlan


def _config(
    tmp_path: Path,
    run_inputs: Path | str,
    *,
    plan_path: Path | str | None = None,
) -> dict[str, Any]:
    config: dict[str, Any] = {
        "workflow": {
            "name": "DynamicPreprocessingWorkflow",
            "kwargs": {
                "num_workers": 1,
                "staging_params": {
                    "stager_name": "FileStager",
                    "params": {"pointers": {}},
                },
                "pipeline_params": {
                    "steps": [],
                    "out_dir": str(tmp_path / "sentinels"),
                },
            },
        },
        "run_inputs": str(run_inputs),
    }
    if plan_path is not None:
        config["artifacts"] = {"plan_path": str(plan_path)}
    return config


def _write_config(path: Path, config: dict[str, Any]) -> Path:
    path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    return path


def _touch(entry: StagedEntry) -> None:
    out_dir = Path(entry.params["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{entry.active.name}.done").write_text("ok", encoding="utf-8")


class TestBuildParser:
    def test_execute_parses_config_and_save_plan(self, tmp_path: Path) -> None:
        config_path = tmp_path / "job.json"
        plan_path = tmp_path / "plan.duckdb"

        args = build_parser().parse_args(
            ["execute", str(config_path), "--save-plan", str(plan_path)]
        )

        assert args.command == "execute"
        assert args.config == config_path
        assert args.save_plan == plan_path
        assert args.debug is False

    def test_plan_parses_output_flag(self, tmp_path: Path) -> None:
        config_path = tmp_path / "job.json"
        plan_path = tmp_path / "plan.duckdb"

        args = build_parser().parse_args(
            ["plan", str(config_path), "-o", str(plan_path)]
        )

        assert args.command == "plan"
        assert args.config == config_path
        assert args.output == plan_path

    def test_plan_parses_long_output_flag(self, tmp_path: Path) -> None:
        config_path = tmp_path / "job.json"
        plan_path = tmp_path / "plan.duckdb"

        args = build_parser().parse_args(
            ["plan", str(config_path), "--output", str(plan_path)]
        )

        assert args.output == plan_path

    def test_dry_run_flag_parses_on_plan(self, tmp_path: Path) -> None:
        config_path = tmp_path / "job.json"

        args = build_parser().parse_args(["plan", str(config_path), "--dry-run"])

        assert args.command == "plan"
        assert args.config == config_path
        assert args.dry_run is True

    def test_dry_run_flag_parses_on_execute(self, tmp_path: Path) -> None:
        config_path = tmp_path / "job.json"

        args = build_parser().parse_args(["execute", str(config_path), "--dry-run"])

        assert args.command == "execute"
        assert args.dry_run is True

    def test_execute_plan_parses_config_and_plan_path(self, tmp_path: Path) -> None:
        config_path = tmp_path / "job.json"
        plan_path = tmp_path / "saved.duckdb"

        args = build_parser().parse_args(
            ["execute-plan", str(config_path), str(plan_path)]
        )

        assert args.command == "execute-plan"
        assert args.config == config_path
        assert args.plan_path == plan_path

    def test_debug_flag(self, tmp_path: Path) -> None:
        config_path = tmp_path / "job.json"

        args = build_parser().parse_args(["--debug", "plan", str(config_path)])

        assert args.debug is True
        assert args.command == "plan"

    def test_missing_subcommand_exits(self) -> None:
        with pytest.raises(SystemExit):
            build_parser().parse_args([])


class TestMainDispatch:
    @pytest.fixture
    def job_paths(self, tmp_path: Path) -> dict[str, Path]:
        file_path = tmp_path / "img.nii.gz"
        file_path.write_bytes(b"")
        config_path = _write_config(
            tmp_path / "job.json",
            _config(tmp_path, file_path),
        )
        plan_path = tmp_path / "plan.duckdb"
        return {
            "config_path": config_path,
            "plan_path": plan_path,
            "file_path": file_path,
        }

    def test_main_execute_passes_same_arguments_as_direct_call(
        self, job_paths: dict[str, Path], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        recorded: list[tuple[dict[str, Any], Path | None, bool]] = []

        def _record(
            config: dict[str, Any],
            *,
            save_plan: Path | str | None = None,
            dry_run: bool = False,
        ):
            recorded.append(
                (dict(config), Path(save_plan) if save_plan else None, dry_run)
            )
            return None

        monkeypatch.setattr(commands, "execute", _record)

        main(
            [
                "execute",
                str(job_paths["config_path"]),
                "--save-plan",
                str(job_paths["plan_path"]),
            ]
        )

        expected_config = dict(load_preproc_config(job_paths["config_path"]))
        assert recorded == [(expected_config, job_paths["plan_path"], False)]

    def test_main_plan_passes_same_arguments_as_direct_call(
        self, job_paths: dict[str, Path], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        recorded: list[tuple[dict[str, Any], Path | None, bool]] = []

        def _record(
            config: dict[str, Any],
            *,
            output: Path | str | None = None,
            dry_run: bool = False,
        ):
            recorded.append((dict(config), Path(output) if output else None, dry_run))
            return RunPlan(entries=())

        monkeypatch.setattr(commands, "plan", _record)

        main(["plan", str(job_paths["config_path"]), "-o", str(job_paths["plan_path"])])

        expected_config = dict(load_preproc_config(job_paths["config_path"]))
        assert recorded == [(expected_config, job_paths["plan_path"], False)]

    def test_main_plan_dry_run_passes_flag(
        self, job_paths: dict[str, Path], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        recorded: list[bool] = []

        def _record(
            config: dict[str, Any],
            *,
            output: Path | str | None = None,
            dry_run: bool = False,
        ):
            recorded.append(dry_run)
            return RunPlan(entries=())

        monkeypatch.setattr(commands, "plan", _record)

        main(["plan", str(job_paths["config_path"]), "--dry-run"])

        assert recorded == [True]

    def test_main_execute_plan_passes_same_arguments_as_direct_call(
        self, job_paths: dict[str, Path], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        recorded: list[tuple[dict[str, Any], Path, bool]] = []

        def _record(
            config: dict[str, Any],
            *,
            plan_path: Path | str,
            dry_run: bool = False,
        ):
            recorded.append((dict(config), Path(plan_path), dry_run))
            return RunPlan(entries=())

        monkeypatch.setattr(commands, "execute_plan", _record)

        main(
            [
                "execute-plan",
                str(job_paths["config_path"]),
                str(job_paths["plan_path"]),
            ]
        )

        expected_config = dict(load_preproc_config(job_paths["config_path"]))
        assert recorded == [(expected_config, job_paths["plan_path"], False)]


class TestMainEquivalence:
    def test_main_plan_matches_direct_command(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        file_path = tmp_path / "img.nii.gz"
        file_path.write_bytes(b"")
        config_path = _write_config(tmp_path / "job.json", _config(tmp_path, file_path))
        plan_path = tmp_path / "plan.duckdb"
        monkeypatch.setattr(
            DynamicPreprocessingWorkflow, "process_single", staticmethod(_touch)
        )

        direct_plan = commands.plan(
            load_preproc_config(config_path),
            output=plan_path,
        )
        assert not (tmp_path / "sentinels").exists()

        file_path.write_bytes(b"")
        config_path = _write_config(tmp_path / "job.json", _config(tmp_path, file_path))
        cli_plan_path = tmp_path / "cli-plan.duckdb"

        main(["plan", str(config_path), "-o", str(cli_plan_path)])

        assert cli_plan_path.exists()
        assert RunPlan.load(cli_plan_path).entries == direct_plan.entries
        assert not (tmp_path / "sentinels").exists()

    def test_main_execute_matches_direct_command(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        file_path = tmp_path / "img.nii.gz"
        file_path.write_bytes(b"")
        config_path = _write_config(tmp_path / "job.json", _config(tmp_path, file_path))
        plan_path = tmp_path / "plan.duckdb"
        monkeypatch.setattr(
            DynamicPreprocessingWorkflow, "process_single", staticmethod(_touch)
        )
        loaded = load_preproc_config(config_path)

        direct_plan = commands.execute(loaded, save_plan=plan_path)

        file_path.write_bytes(b"")
        config_path = _write_config(tmp_path / "job.json", _config(tmp_path, file_path))
        cli_plan_path = tmp_path / "cli-plan.duckdb"
        sentinels = tmp_path / "sentinels"
        if sentinels.exists():
            for path in sentinels.iterdir():
                path.unlink()

        main(
            [
                "execute",
                str(config_path),
                "--save-plan",
                str(cli_plan_path),
            ]
        )

        assert direct_plan is not None
        assert cli_plan_path.exists()
        assert RunPlan.load(cli_plan_path).entries == direct_plan.entries
        assert (sentinels / "img.nii.gz.done").exists()

    def test_main_reports_errors_without_debug(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        config_path = tmp_path / "missing-sections.json"
        config_path.write_text("{}", encoding="utf-8")

        with pytest.raises(SystemExit) as exc:
            main(["plan", str(config_path), "-o", str(tmp_path / "plan.duckdb")])

        assert exc.value.code == 1
        captured = capsys.readouterr()
        assert "Error:" in captured.err
        assert "Missing required config section `workflow`" in captured.err
