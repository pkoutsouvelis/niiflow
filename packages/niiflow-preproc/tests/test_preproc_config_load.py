"""Tests for :func:`~niiflow.preproc.config.load_config`."""

from __future__ import annotations

from pathlib import Path

import pytest

from niiflow.preproc.config import load_config


def _sample_config() -> dict:
    return {
        "settings": {"pipeline_params": {"steps": []}, "num_workers": 1},
        "inputs": "data/root",
        "save_plan_to": "out/plan.duckdb",
    }


def test_load_config_from_json(tmp_path: Path) -> None:
    config_path = tmp_path / "job.json"
    config_path.write_text(
        """{
        "settings": {"pipeline_params": {"steps": []}, "num_workers": 1},
        "inputs": "data/root"
        }""",
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config["settings"]["num_workers"] == 1
    assert config["inputs"] == "data/root"


@pytest.mark.parametrize("suffix", [".yaml", ".yml"])
def test_load_config_from_yaml(tmp_path: Path, suffix: str) -> None:
    config_path = tmp_path / f"job{suffix}"
    config_path.write_text(
        """settings:
  pipeline_params:
    steps: []
  num_workers: 1
inputs: data/root
save_plan_to: out/plan.duckdb
""",
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config == _sample_config()


def test_load_config_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "missing.json")


def test_load_config_rejects_unsupported_extension(tmp_path: Path) -> None:
    config_path = tmp_path / "job.toml"
    config_path.write_text("settings = { num_workers = 1 }", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported config extension"):
        load_config(config_path)


def test_load_config_rejects_non_mapping_json(tmp_path: Path) -> None:
    config_path = tmp_path / "job.json"
    config_path.write_text('["not", "a", "mapping"]', encoding="utf-8")

    with pytest.raises(TypeError, match="contain a mapping"):
        load_config(config_path)


def test_load_config_rejects_non_mapping_yaml(tmp_path: Path) -> None:
    config_path = tmp_path / "job.yaml"
    config_path.write_text("- not\n- a\n- mapping\n", encoding="utf-8")

    with pytest.raises(TypeError, match="must be a mapping"):
        load_config(config_path)
