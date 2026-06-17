"""Tests for config-driven dynamic pipeline builder."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from niiflow.preproc.pipelines import create_pipeline, discover_stage_classes
from niiflow.preproc.pipelines.pipeline_stages import (
    CheckVoxelSpacing,
    Compose,
    PipelineStage,
    RuntimeContext,
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


@pytest.fixture
def registry_with_echo(
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, type[PipelineStage]]:
    registry = discover_stage_classes()
    registry = {**registry, "EchoStage": EchoStage}
    monkeypatch.setattr(
        "niiflow.preproc.pipelines.pipeline_factory.discover_stage_classes",
        lambda: registry,
    )
    return registry


def test_discover_stage_classes_excludes_compose_and_base() -> (
    None
):  # TODO: should be shipping-centered; better to test failure cases directly
    registry = discover_stage_classes()
    assert "Compose" not in registry
    assert "PipelineStage" not in registry
    assert "RuntimeContext" not in registry
    assert "CheckVoxelSpacing" in registry
    assert issubclass(registry["CheckVoxelSpacing"], PipelineStage)


def test_build_from_mapping_and_order(
    registry_with_echo: dict[str, PipelineStage],
) -> None:
    pipeline = create_pipeline(
        {
            "order": ["qc", "echo"],
            "steps": {
                "echo": {
                    "name": "EchoStage",
                    "params": {"message": "hello"},
                    "save_options": {},
                },
                "qc": {
                    "name": "CheckVoxelSpacing",
                    "params": {
                        "image": "input.nii.gz",
                        "expected": [1.0, 1.0, 1.0],
                    },
                    "save_options": {},
                    "verbose": False,
                },
            },
            "verbose": True,
        }
    )
    assert isinstance(pipeline, Compose)
    assert pipeline.step_ids == ("qc", "echo")
    assert type(pipeline.stages[0]) is CheckVoxelSpacing
    assert type(pipeline.stages[1]) is EchoStage
    assert pipeline.stages[0].verbose is False
    assert pipeline.stages[1].verbose is True


def test_build_from_mapping_without_order(
    registry_with_echo: dict[str, PipelineStage],
) -> None:
    pipeline = create_pipeline(
        {
            "steps": {
                "first": {
                    "name": "EchoStage",
                    "params": {"message": "one"},
                    "save_options": {},
                },
                "second": {
                    "name": "EchoStage",
                    "params": {"message": "two"},
                    "save_options": {},
                },
            }
        }
    )
    assert pipeline.step_ids == ("first", "second")
    assert [stage.params["message"] for stage in pipeline.stages] == ["one", "two"]


def test_build_from_ordered_list(registry_with_echo: dict[str, PipelineStage]) -> None:
    pipeline = create_pipeline(
        {
            "steps": [
                {
                    "name": "EchoStage",
                    "params": {"message": "one"},
                    "save_options": {},
                },
                {
                    "name": "EchoStage",
                    "params": {"message": "two"},
                    "save_options": {},
                },
            ],
            "verbose": False,
        }
    )
    assert pipeline.step_ids == (None, None)
    assert pipeline.verbose is False
    assert all(stage.verbose is False for stage in pipeline.stages)


def test_build_runs_configured_stages_with_auto_ids(
    registry_with_echo: dict[str, PipelineStage],
) -> None:
    pipeline = create_pipeline(
        {
            "steps": [
                {
                    "name": "EchoStage",
                    "params": {"message": "built"},
                    "save_options": {},
                }
            ]
        }
    )
    ctx = pipeline.run(RuntimeContext())
    assert ctx.artifacts["step_0000"] == {"message": "built"}
    assert ctx.steps_completed == ["step_0000"]


def test_build_runs_configured_stages_with_named_ids(
    registry_with_echo: dict[str, PipelineStage],
) -> None:
    pipeline = create_pipeline(
        {
            "steps": {
                "echo": {
                    "name": "EchoStage",
                    "params": {"message": "built"},
                    "save_options": {},
                }
            }
        }
    )
    ctx = pipeline.run(RuntimeContext())
    assert ctx.artifacts["echo"] == {"message": "built"}
    assert ctx.steps_completed == ["echo"]


def test_build_rejects_compose_name(
    registry_with_echo: dict[str, PipelineStage],
) -> None:
    with pytest.raises(ValueError, match="Compose"):
        create_pipeline(
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


def test_build_rejects_instantiated_stage(
    registry_with_echo: dict[str, PipelineStage],
) -> None:
    with pytest.raises(TypeError, match="instantiated"):
        create_pipeline(
            {
                "order": ["bad"],
                "steps": {"bad": EchoStage(params={"message": "x"}, save_options={})},
            }
        )


def test_build_rejects_unknown_stage_name(
    registry_with_echo: dict[str, PipelineStage],
) -> None:
    with pytest.raises(ValueError, match="Unknown pipeline stage"):
        create_pipeline(
            {
                "order": ["bad"],
                "steps": {"bad": {"name": "NotARealStage", "params": {}}},
            }
        )


def test_build_rejects_failed_instantiation(
    registry_with_echo: dict[str, PipelineStage],
) -> None:
    with pytest.raises(TypeError, match="Failed to instantiate stage"):
        create_pipeline(
            {
                "steps": [
                    {
                        "name": "EchoStage",
                        "unexpected_kwarg": True,
                    }
                ]
            }
        )


def test_build_rejects_order_with_unknown_step_id(
    registry_with_echo: dict[str, PipelineStage],
) -> None:
    with pytest.raises(ValueError, match="unknown step id"):
        create_pipeline(
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


def test_build_rejects_unused_step_in_mapping(
    registry_with_echo: dict[str, PipelineStage],
) -> None:
    with pytest.raises(ValueError, match="not listed in `order`"):
        create_pipeline(
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


def test_build_rejects_unknown_keys_when_steps_is_list(
    registry_with_echo: dict[str, PipelineStage],
) -> None:
    with pytest.raises(ValueError, match="Unknown pipeline key"):
        create_pipeline(
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
        create_pipeline(
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


def test_build_rejects_unknown_keys_when_steps_is_mapping(
    registry_with_echo: dict[str, PipelineStage],
) -> None:
    with pytest.raises(ValueError, match="Unknown pipeline key"):
        create_pipeline(
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
