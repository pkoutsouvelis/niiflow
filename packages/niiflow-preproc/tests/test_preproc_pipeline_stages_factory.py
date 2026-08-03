"""Tests for :mod:`niiflow.preproc.pipelines.pipeline_stages.stage_factory`."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from niiflow.preproc.pipelines.pipeline_stages import (
    CheckVoxelSpacing,
    PipelineStage,
    create_stage,
    discover_stage_classes,
)


class SpyStage(PipelineStage):
    """Test-only stage with no required params, used to assert kwarg forwarding."""

    def load_param(self, key: str, value: Any) -> Any:
        return value

    def forward(self, **params: Any) -> dict[str, Any]:
        return {"message": params.get("message")}

    def save_output(self, key: str, value: Any, output_path: Path) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(str(value), encoding="utf-8")
        return output_path


class TestDiscoverStageClasses:
    """Registry construction by class-name introspection."""

    def test_excludes_base_compose_and_factory(self) -> None:
        registry = discover_stage_classes()
        assert "PipelineStage" not in registry
        assert "Compose" not in registry
        assert "RuntimeContext" not in registry
        # Modules outside `_STAGE_MODULES` (compose / factory / base) are not scanned.
        assert "create_stage" not in registry

    def test_includes_shipped_concrete_stages(self) -> None:
        registry = discover_stage_classes()
        assert registry["CheckVoxelSpacing"] is CheckVoxelSpacing
        assert "ApplyMask" in registry
        assert "SmoothMask" in registry
        assert "RelabelMask" in registry
        assert all(issubclass(cls, PipelineStage) for cls in registry.values())

    def test_registry_is_a_fresh_mapping_per_call(self) -> None:
        first = discover_stage_classes()
        first["Injected"] = SpyStage
        assert "Injected" not in discover_stage_classes()


class TestCreateStage:
    """Instantiation from a registered class name plus constructor kwargs."""

    def test_builds_shipped_stage_with_kwargs(self, tmp_path: Path) -> None:
        report = tmp_path / "report.json"
        stage = create_stage(
            "CheckVoxelSpacing",
            {
                "params": {
                    "image": str(tmp_path / "img.nii.gz"),
                    "expected": (1.0, 1.0, 1.0),
                },
                "save_outputs": {"report": str(report)},
                "verbose": False,
            },
        )
        assert isinstance(stage, CheckVoxelSpacing)
        assert stage.verbose is False
        assert stage.params["expected"] == (1.0, 1.0, 1.0)
        assert stage.save_outputs["report"] == report

    def test_omitted_kwargs_use_constructor_defaults(self) -> None:
        stage = create_stage("SpyStage", registry={"SpyStage": SpyStage})
        assert stage.params == {}
        assert stage.save_outputs == {}
        assert stage.verbose is True

    def test_none_kwargs_is_treated_as_empty(self) -> None:
        stage = create_stage("SpyStage", None, registry={"SpyStage": SpyStage})
        assert isinstance(stage, SpyStage)
        assert stage.params == {}

    def test_accepts_custom_registry(self) -> None:
        stage = create_stage(
            "SpyStage",
            {"params": {"message": "hello"}},
            registry={"SpyStage": SpyStage},
        )
        assert isinstance(stage, SpyStage)
        assert stage.params == {"message": "hello"}

    def test_rejects_compose(self) -> None:
        with pytest.raises(ValueError, match="`Compose` cannot be built"):
            create_stage("Compose")

    def test_rejects_unknown_name(self) -> None:
        with pytest.raises(ValueError, match="Unknown pipeline stage"):
            create_stage("NotARealStage")

    @pytest.mark.parametrize("name", ["", None, 3])
    def test_rejects_invalid_name(self, name: Any) -> None:
        with pytest.raises(ValueError, match="must be a non-empty string"):
            create_stage(name)

    def test_rejects_non_dict_kwargs(self) -> None:
        with pytest.raises(TypeError, match="`stage_kwargs` must be a dictionary"):
            create_stage("CheckVoxelSpacing", ["params"])  # type: ignore[arg-type]

    def test_rejects_unexpected_constructor_kwarg(self) -> None:
        with pytest.raises(TypeError, match="Failed to instantiate stage"):
            create_stage("CheckVoxelSpacing", {"unexpected_kwarg": True})
