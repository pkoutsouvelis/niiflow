"""Tests for :func:`dynamic_pipeline` (build + execute forward function)."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import pytest

from niiflow.preproc.pipelines import dynamic_pipeline
from niiflow.preproc.pipelines.pipeline_stages import PipelineStage

# Prefer importlib: package ``__init__`` re-exports ``dynamic_pipeline`` and
# shadows the submodule attribute of the same name.
_dynamic_pipeline_mod = importlib.import_module(
    "niiflow.preproc.pipelines.dynamic_pipeline"
)


class EchoStage(PipelineStage):
    """Test-only stage registered via monkeypatched discovery."""

    REQUIRED_PARAMS = frozenset({"message"})

    def load_param(self, key: str, value: Any) -> Any:
        return value

    def forward(self, **params: Any) -> dict[str, Any]:
        return {"message": params["message"]}

    def save_output(self, key: str, value: Any, output_path: Path) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(str(value), encoding="utf-8")
        return output_path


class RecordingStage(PipelineStage):
    """Records execution order via a shared ``calls`` list injected per test."""

    REQUIRED_PARAMS = frozenset({"label"})
    calls: list[str] = []

    def load_param(self, key: str, value: Any) -> Any:
        return value

    def forward(self, **params: Any) -> dict[str, Any]:
        label = str(params["label"])
        type(self).calls.append(label)
        return {"label": label}

    def save_output(self, key: str, value: Any, output_path: Path) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(str(value), encoding="utf-8")
        return output_path


@pytest.fixture
def registry_with_echo(
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, type[PipelineStage]]:
    registry = {
        **_dynamic_pipeline_mod._discover_stage_classes(),
        "EchoStage": EchoStage,
        "RecordingStage": RecordingStage,
    }
    monkeypatch.setattr(
        _dynamic_pipeline_mod,
        "_discover_stage_classes",
        lambda: registry,
    )
    return registry


@pytest.fixture
def recording_calls() -> list[str]:
    RecordingStage.calls = []
    return RecordingStage.calls


def test_discover_stage_classes_excludes_compose_and_base() -> None:
    registry = _dynamic_pipeline_mod._discover_stage_classes()
    assert "Compose" not in registry
    assert "PipelineStage" not in registry
    assert "RuntimeContext" not in registry
    assert "CheckVoxelSpacing" in registry
    assert issubclass(registry["CheckVoxelSpacing"], PipelineStage)


def test_returns_none_and_executes(
    registry_with_echo: dict[str, type[PipelineStage]],
    tmp_path: Path,
) -> None:
    out = tmp_path / "echo.txt"
    result = dynamic_pipeline(
        {
            "steps": {
                "echo": {
                    "name": "EchoStage",
                    "params": {"message": "hello"},
                    "save_options": {"message": str(out)},
                }
            }
        },
        run_id=str(tmp_path / "active.nii.gz"),
    )
    assert result is None
    assert out.read_text(encoding="utf-8") == "hello"


def test_run_id_is_passed_to_runtime_context(
    registry_with_echo: dict[str, type[PipelineStage]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, Any] = {}
    real_runtime = _dynamic_pipeline_mod.RuntimeContext

    def _capture_runtime(*, run_id: str | None = None, **kwargs: Any) -> Any:
        seen["run_id"] = run_id
        return real_runtime(run_id=run_id, **kwargs)

    monkeypatch.setattr(_dynamic_pipeline_mod, "RuntimeContext", _capture_runtime)
    dynamic_pipeline(
        {
            "steps": {
                "echo": {
                    "name": "EchoStage",
                    "params": {"message": "x"},
                    "save_options": {},
                }
            }
        },
        run_id="sub-01",
    )
    assert seen["run_id"] == "sub-01"


def test_mapping_order_controls_execution_order(
    registry_with_echo: dict[str, type[PipelineStage]],
    recording_calls: list[str],
) -> None:
    dynamic_pipeline(
        {
            "order": ["second", "first"],
            "steps": {
                "first": {
                    "name": "RecordingStage",
                    "params": {"label": "first"},
                    "save_options": {},
                },
                "second": {
                    "name": "RecordingStage",
                    "params": {"label": "second"},
                    "save_options": {},
                },
            },
        }
    )
    assert recording_calls == ["second", "first"]


def test_mapping_without_order_uses_insertion_order(
    registry_with_echo: dict[str, type[PipelineStage]],
    recording_calls: list[str],
) -> None:
    dynamic_pipeline(
        {
            "steps": {
                "first": {
                    "name": "RecordingStage",
                    "params": {"label": "one"},
                    "save_options": {},
                },
                "second": {
                    "name": "RecordingStage",
                    "params": {"label": "two"},
                    "save_options": {},
                },
            }
        }
    )
    assert recording_calls == ["one", "two"]


def test_ordered_list_uses_list_position(
    registry_with_echo: dict[str, type[PipelineStage]],
    recording_calls: list[str],
) -> None:
    dynamic_pipeline(
        {
            "steps": [
                {
                    "name": "RecordingStage",
                    "params": {"label": "one"},
                    "save_options": {},
                },
                {
                    "name": "RecordingStage",
                    "params": {"label": "two"},
                    "save_options": {},
                },
            ],
            "verbose": False,
        }
    )
    assert recording_calls == ["one", "two"]


def test_pipeline_verbose_defaults_applied_to_stages(
    registry_with_echo: dict[str, type[PipelineStage]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pipeline-level verbose is forwarded when a step omits its own verbose."""
    seen_verbose: list[bool] = []
    real_create = _dynamic_pipeline_mod._create_stage

    def _create(*args: Any, **kwargs: Any) -> PipelineStage:
        stage = real_create(*args, **kwargs)
        seen_verbose.append(stage.verbose)
        return stage

    monkeypatch.setattr(_dynamic_pipeline_mod, "_create_stage", _create)
    dynamic_pipeline(
        {
            "steps": [
                {
                    "name": "EchoStage",
                    "params": {"message": "a"},
                    "save_options": {},
                },
                {
                    "name": "EchoStage",
                    "params": {"message": "b"},
                    "save_options": {},
                    "verbose": True,
                },
            ],
            "verbose": False,
        }
    )
    assert seen_verbose == [False, True]


def test_rejects_compose_name(
    registry_with_echo: dict[str, type[PipelineStage]],
) -> None:
    with pytest.raises(ValueError, match="Compose"):
        dynamic_pipeline(
            {
                "order": ["nested"],
                "steps": {
                    "nested": {
                        "name": "Compose",
                        "params": {},
                    }
                },
            }
        )


def test_rejects_instantiated_stage(
    registry_with_echo: dict[str, type[PipelineStage]],
) -> None:
    with pytest.raises(TypeError, match="instantiated"):
        dynamic_pipeline(
            {
                "order": ["bad"],
                "steps": {"bad": EchoStage(params={"message": "x"}, save_options={})},
            }
        )


def test_rejects_unknown_stage_name(
    registry_with_echo: dict[str, type[PipelineStage]],
) -> None:
    with pytest.raises(ValueError, match="Unknown pipeline stage"):
        dynamic_pipeline(
            {
                "order": ["bad"],
                "steps": {"bad": {"name": "NotARealStage", "params": {}}},
            }
        )


def test_rejects_failed_instantiation(
    registry_with_echo: dict[str, type[PipelineStage]],
) -> None:
    with pytest.raises(TypeError, match="Failed to instantiate stage"):
        dynamic_pipeline(
            {
                "steps": [
                    {
                        "name": "EchoStage",
                        "unexpected_kwarg": True,
                    }
                ]
            }
        )


def test_rejects_order_with_unknown_step_id(
    registry_with_echo: dict[str, type[PipelineStage]],
) -> None:
    with pytest.raises(ValueError, match="unknown step id"):
        dynamic_pipeline(
            {
                "order": ["missing"],
                "steps": {
                    "echo": {
                        "name": "EchoStage",
                        "params": {"message": "hello"},
                        "save_options": {},
                    }
                },
            }
        )


def test_rejects_unused_step_in_mapping(
    registry_with_echo: dict[str, type[PipelineStage]],
) -> None:
    with pytest.raises(ValueError, match="not listed in `order`"):
        dynamic_pipeline(
            {
                "order": ["echo"],
                "steps": {
                    "echo": {
                        "name": "EchoStage",
                        "params": {"message": "hello"},
                        "save_options": {},
                    },
                    "unused": {
                        "name": "EchoStage",
                        "params": {"message": "ignored"},
                        "save_options": {},
                    },
                },
            }
        )


def test_rejects_unknown_keys_when_steps_is_list(
    registry_with_echo: dict[str, type[PipelineStage]],
) -> None:
    with pytest.raises(ValueError, match="Unknown pipeline key"):
        dynamic_pipeline(
            {
                "order": ["echo"],
                "steps": [
                    {
                        "name": "EchoStage",
                        "params": {"message": "hello"},
                        "save_options": {},
                    }
                ],
            }
        )

    with pytest.raises(ValueError, match="Unknown pipeline key"):
        dynamic_pipeline(
            {
                "unknown": True,
                "steps": [
                    {
                        "name": "EchoStage",
                        "params": {"message": "hello"},
                        "save_options": {},
                    }
                ],
            }
        )


def test_rejects_unknown_keys_when_steps_is_mapping(
    registry_with_echo: dict[str, type[PipelineStage]],
) -> None:
    with pytest.raises(ValueError, match="Unknown pipeline key"):
        dynamic_pipeline(
            {
                "unknown": True,
                "steps": {
                    "echo": {
                        "name": "EchoStage",
                        "params": {"message": "hello"},
                        "save_options": {},
                    }
                },
            }
        )
