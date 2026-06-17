"""Tests for :func:`~niiflow.preproc.config.load_preproc_config`."""

from __future__ import annotations

from pathlib import Path

import pytest

from niiflow.preproc.config import load_preproc_config


def _sample_config() -> dict:
    return {
        "workflow": {
            "name": "DynamicPreprocessingWorkflow",
            "kwargs": {"pipeline_params": {"steps": []}},
        },
        "run_inputs": "data/root",
        "artifacts": {"plan_path": "out/plan.duckdb"},
    }


def test_load_preproc_config_from_json(tmp_path: Path) -> None:
    config_path = tmp_path / "job.json"
    config_path.write_text(
        """{
        "workflow": {
            "name": "DynamicPreprocessingWorkflow",
            "kwargs": {"pipeline_params": {"steps": []}}
        },
        "run_inputs": "data/root"
        }""",
        encoding="utf-8",
    )

    config = load_preproc_config(config_path)

    assert config["workflow"]["name"] == "DynamicPreprocessingWorkflow"
    assert config["run_inputs"] == "data/root"


@pytest.mark.parametrize("suffix", [".yaml", ".yml"])
def test_load_preproc_config_from_yaml(tmp_path: Path, suffix: str) -> None:
    config_path = tmp_path / f"job{suffix}"
    config_path.write_text(
        """workflow:
  name: DynamicPreprocessingWorkflow
  kwargs:
    pipeline_params:
      steps: []
run_inputs: data/root
artifacts:
  plan_path: out/plan.duckdb
""",
        encoding="utf-8",
    )

    config = load_preproc_config(config_path)

    assert config == _sample_config()


def test_load_preproc_config_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_preproc_config(tmp_path / "missing.json")


def test_load_preproc_config_rejects_unsupported_extension(tmp_path: Path) -> None:
    config_path = tmp_path / "job.toml"
    config_path.write_text(
        'workflow = { name = "DynamicPreprocessingWorkflow" }', encoding="utf-8"
    )

    with pytest.raises(ValueError, match="Unsupported config extension"):
        load_preproc_config(config_path)


def test_load_preproc_config_rejects_non_mapping_json(tmp_path: Path) -> None:
    config_path = tmp_path / "job.json"
    config_path.write_text('["not", "a", "mapping"]', encoding="utf-8")

    with pytest.raises(TypeError, match="contain a mapping"):
        load_preproc_config(config_path)


def test_load_preproc_config_rejects_non_mapping_yaml(tmp_path: Path) -> None:
    config_path = tmp_path / "job.yaml"
    config_path.write_text("- not\n- a\n- mapping\n", encoding="utf-8")

    with pytest.raises(TypeError, match="must be a mapping"):
        load_preproc_config(config_path)
