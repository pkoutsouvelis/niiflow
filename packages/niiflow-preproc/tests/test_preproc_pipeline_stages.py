"""Tests for :class:`PipelineStage` and concrete stage implementations.

:class:`DummyPipelineStage` (defined in this module) exercises the full
``run()`` / context / persistence contract without ANTs.

Every shipped concrete stage in :data:`SHIPPED_STAGE_CLASSES` is checked for
pipeline-style construction and for compatibility with the shared ``run()``
machinery (via stubbed ``forward`` / ``save_output``). Domain behaviour lives in
per-stage test modules (e.g. ``test_preproc_stages_registration.py``).
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeAlias

import pytest

from niiflow.preproc.pipelines.pipeline_stages import (
    ANTsApplyTransforms,
    ANTsBiasFieldCorrection,
    ANTsBrainExtraction,
    ANTsDenoise,
    ANTsPreprocessBrainImage,
    ANTsRegistration,
    ANTsResample,
    ANTsResampleToTarget,
    ApplyMask,
    CenterCrop,
    CenterPad,
    CheckDimensions,
    CheckVoxelSpacing,
    ClampIntensities,
    Compose,
    CropToMask,
    CropToRange,
    Delete,
    MinmaxNorm,
    PadToRange,
    PipelineStage,
    Reorient,
    RuntimeContext,
    ToNumpy,
    ZTransformNorm,
)
from niiflow.preproc.pipelines.pipeline_stages.pipeline_stage import (
    ArtifactPathRecord,
)

STEP_ID = "step"

StageFactory: TypeAlias = type[PipelineStage]

SHIPPED_STAGE_CLASSES: tuple[StageFactory, ...] = (
    ANTsBiasFieldCorrection,
    ANTsDenoise,
    CropToMask,
    CropToRange,
    PadToRange,
    CenterCrop,
    CenterPad,
    ANTsRegistration,
    ANTsResample,
    ANTsResampleToTarget,
    ANTsBrainExtraction,
    ANTsApplyTransforms,
    ANTsPreprocessBrainImage,
    ClampIntensities,
    ZTransformNorm,
    MinmaxNorm,
    CheckVoxelSpacing,
    CheckDimensions,
    ApplyMask,
    Delete,
    Reorient,
    ToNumpy,
)

PRIMARY_SAVE_KEY: dict[StageFactory, str] = {
    ANTsBiasFieldCorrection: "out_image",
    ANTsDenoise: "out_image",
    CropToMask: "out_image",
    CropToRange: "out_image",
    PadToRange: "out_image",
    CenterCrop: "out_image",
    CenterPad: "out_image",
    ANTsRegistration: "fwdtransforms",
    ANTsResample: "out_image",
    ANTsResampleToTarget: "out_image",
    ANTsPreprocessBrainImage: "out_image",
    ANTsBrainExtraction: "out_image",
    ANTsApplyTransforms: "out_image",
    ClampIntensities: "out_image",
    ZTransformNorm: "out_image",
    MinmaxNorm: "out_image",
    CheckVoxelSpacing: "passed",
    CheckDimensions: "passed",
    ApplyMask: "out_image",
    Delete: "deleted",
    Reorient: "out_image",
    ToNumpy: "array",
}


# ---------------------------------------------------------------------------
# Test-only dummy stage (full PipelineStage contract, no ANTs).
# ---------------------------------------------------------------------------


class DummyPipelineStage(PipelineStage):
    """Minimal stage for ``PipelineStage`` machinery tests."""

    REQUIRED_PARAMS = frozenset({"input_nii"})

    def check_params(self, params: dict[str, Any]) -> None:
        scale = params.get("scale_factor", 1.0)
        if not isinstance(scale, (int, float)):
            raise ValueError(
                f"scale_factor must be numeric, got {type(scale).__name__}"
            )

    def load_param(self, key: str, value: Any) -> Any:
        if key == "input_nii":
            if isinstance(value, (bytes, bytearray)):
                return bytes(value)
            return Path(value).read_bytes()
        return value

    def forward(self, **params: Any) -> dict[str, Any]:
        raw: bytes = params["input_nii"]
        scale = float(params.get("scale_factor", 1.0))
        sample = raw[0] if raw else 0
        return {"output_nii": bytes([min(255, int(sample * scale))])}

    def save_output(self, key: str, value: Any, output_path: Path) -> Path:
        if key != "output_nii":
            raise KeyError(f"Unknown output key {key!r} for {type(self).__name__}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(value)
        return output_path


# ---------------------------------------------------------------------------
# Config builders for shipped stages (construction / stubbed run only).
# ---------------------------------------------------------------------------


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x00")
    return path


def _image_stage_config(tmp_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    image = _touch(tmp_path / "image.nii.gz")
    return {"image": str(image)}, {"out_image": None}


def _registration_stage_config(
    tmp_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    moving = _touch(tmp_path / "moving.nii.gz")
    fixed = _touch(tmp_path / "fixed.nii.gz")
    return (
        {
            "image": str(moving),
            "target": str(fixed),
            "apply_forward": False,
        },
        {
            "fwdtransforms": str(tmp_path / "fwd.json"),
            "invtransforms": str(tmp_path / "inv.json"),
        },
    )


def _apply_transforms_stage_config(
    tmp_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    moving = _touch(tmp_path / "moving.nii.gz")
    fixed = _touch(tmp_path / "fixed.nii.gz")
    transform = _touch(tmp_path / "transform.mat")
    return (
        {
            "image": str(moving),
            "target": str(fixed),
            "transformlist": [str(transform)],
        },
        {"out_image": None},
    )


def _crop_to_range_stage_config(
    tmp_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    image = _touch(tmp_path / "image.nii.gz")
    ranges_path = tmp_path / "ranges.json"
    ranges_path.write_text(json.dumps([[0, 8], [0, 8], [0, 8]]), encoding="utf-8")
    return {"image": str(image), "ranges": str(ranges_path)}, {"out_image": None}


def _crop_to_mask_stage_config(
    tmp_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    image = _touch(tmp_path / "image.nii.gz")
    return {"image": str(image)}, {"out_image": None}


def _center_crop_stage_config(
    tmp_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    image = _touch(tmp_path / "image.nii.gz")
    return {"image": str(image), "shape": (16, 16, 16)}, {"out_image": None}


def _pad_to_range_stage_config(
    tmp_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    image = _touch(tmp_path / "image.nii.gz")
    ranges_path = tmp_path / "pad_ranges.json"
    ranges_path.write_text(json.dumps([[0, 1], [0, 1], [0, 1]]), encoding="utf-8")
    return {"image": str(image), "ranges": str(ranges_path)}, {"out_image": None}


def _center_pad_stage_config(
    tmp_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    image = _touch(tmp_path / "image.nii.gz")
    return {"image": str(image), "shape": (32, 32, 32)}, {"out_image": None}


def _intensity_norm_stage_config(
    tmp_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    image = _touch(tmp_path / "image.nii.gz")
    return {"image": str(image)}, {"out_image": None}


def _check_voxel_spacing_stage_config(
    tmp_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    image = _touch(tmp_path / "image.nii.gz")
    return {"image": str(image), "expected": (1.0, 1.0, 1.0)}, {}


def _check_dimensions_stage_config(
    tmp_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    image = _touch(tmp_path / "image.nii.gz")
    return {"image": str(image), "expected": (64, 64, 64)}, {}


def _delete_stage_config(
    tmp_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    target = _touch(tmp_path / "to_delete.txt")
    return {"paths": [str(target)], "missing_ok": False}, {}


def _reorient_stage_config(
    tmp_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    image = _touch(tmp_path / "image.nii.gz")
    return {"image": str(image), "orientation": "RAS"}, {"out_image": None}


def _apply_mask_stage_config(
    tmp_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    image = _touch(tmp_path / "image.nii.gz")
    mask = _touch(tmp_path / "mask.nii.gz")
    return {"image": str(image), "mask": str(mask)}, {"out_image": None}


def _to_numpy_stage_config(
    tmp_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    image = _touch(tmp_path / "image.nii.gz")
    return {"image": str(image), "dtype": "float32"}, {"array": None}


def _ants_resample_stage_config(
    tmp_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    image = _touch(tmp_path / "image.nii.gz")
    return (
        {"image": str(image), "resample_params": [1.0, 1.0, 1.0]},
        {"out_image": None},
    )


def _ants_resample_to_target_stage_config(
    tmp_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    image = _touch(tmp_path / "image.nii.gz")
    target = _touch(tmp_path / "target.nii.gz")
    return (
        {"image": str(image), "target": str(target)},
        {"out_image": None},
    )


def _ants_preprocess_brain_image_stage_config(
    tmp_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    image = _touch(tmp_path / "image.nii.gz")
    return {"image": str(image)}, {"out_image": None}


STAGE_CONFIG_BUILDERS: dict[
    StageFactory, Callable[[Path], tuple[dict[str, Any], dict[str, Any]]]
] = {
    ANTsBiasFieldCorrection: _image_stage_config,
    ANTsDenoise: _image_stage_config,
    ANTsBrainExtraction: _image_stage_config,
    CropToMask: _crop_to_mask_stage_config,
    CropToRange: _crop_to_range_stage_config,
    PadToRange: _pad_to_range_stage_config,
    CenterCrop: _center_crop_stage_config,
    CenterPad: _center_pad_stage_config,
    ANTsRegistration: _registration_stage_config,
    ANTsResample: _ants_resample_stage_config,
    ANTsResampleToTarget: _ants_resample_to_target_stage_config,
    ANTsPreprocessBrainImage: _ants_preprocess_brain_image_stage_config,
    ANTsApplyTransforms: _apply_transforms_stage_config,
    ClampIntensities: _intensity_norm_stage_config,
    ZTransformNorm: _intensity_norm_stage_config,
    MinmaxNorm: _intensity_norm_stage_config,
    CheckVoxelSpacing: _check_voxel_spacing_stage_config,
    CheckDimensions: _check_dimensions_stage_config,
    Delete: _delete_stage_config,
    ApplyMask: _apply_mask_stage_config,
    Reorient: _reorient_stage_config,
    ToNumpy: _to_numpy_stage_config,
}

STUB_FORWARD_OUTPUTS: dict[StageFactory, dict[str, Any]] = {
    ANTsBiasFieldCorrection: {"out_image": object()},
    ANTsDenoise: {"out_image": object()},
    ANTsBrainExtraction: {"out_image": object(), "brain_mask": object()},
    CropToMask: {"out_image": object(), "ranges": [(0, 4), (0, 4), (0, 4)]},
    CropToRange: {"out_image": object()},
    PadToRange: {"out_image": object()},
    CenterCrop: {"out_image": object(), "ranges": [(0, 4), (0, 4), (0, 4)]},
    CenterPad: {"out_image": object(), "ranges": [(0, 2), (0, 2), (0, 2)]},
    ANTsRegistration: {
        "fwdtransforms": ["stub_fwd.mat"],
        "invtransforms": ["stub_inv.mat"],
    },
    ANTsResample: {"out_image": object()},
    ANTsResampleToTarget: {"out_image": object()},
    ANTsPreprocessBrainImage: {
        "out_image": object(),
        "brain_mask": object(),
        "bias_field": object(),
    },
    ANTsApplyTransforms: {"out_image": object()},
    ClampIntensities: {"out_image": object()},
    ZTransformNorm: {"out_image": object()},
    MinmaxNorm: {"out_image": object()},
    CheckVoxelSpacing: {
        "passed": True,
        "value": (1.0, 1.0, 1.0),
        "report": {
            "check": "CheckVoxelSpacing",
            "passed": True,
            "value": [1.0, 1.0, 1.0],
        },
    },
    CheckDimensions: {
        "passed": True,
        "value": (64, 64, 64),
        "report": {
            "check": "CheckDimensions",
            "passed": True,
            "value": [64, 64, 64],
        },
    },
    Delete: {"deleted": []},
    ApplyMask: {"out_image": object()},
    Reorient: {"out_image": object()},
    ToNumpy: {"array": object(), "metadata": {}},
}


def build_stage_config(
    stage_cls: StageFactory,
    tmp_path: Path,
    *,
    params: dict[str, Any] | None = None,
    save_options: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return ``(params, save_options)`` defaults for a stage class."""
    if stage_cls is DummyPipelineStage:
        input_nii = _touch(tmp_path / "input.nii.gz")
        base_params: dict[str, Any] = {
            "input_nii": str(input_nii),
            "scale_factor": 2.0,
        }
        base_save_options: dict[str, Any] = {"output_nii": None}
    else:
        try:
            builder = STAGE_CONFIG_BUILDERS[stage_cls]
        except KeyError as exc:
            raise NotImplementedError(
                f"Register {stage_cls.__name__} in STAGE_CONFIG_BUILDERS "
                f"before adding it to SHIPPED_STAGE_CLASSES."
            ) from exc
        base_params, base_save_options = builder(tmp_path)

    if params is not None:
        base_params = {**base_params, **params}
    if save_options is not None:
        base_save_options = {**base_save_options, **save_options}
    return base_params, base_save_options


def make_stage(
    stage_cls: StageFactory,
    tmp_path: Path,
    *,
    params: dict[str, Any] | None = None,
    save_options: dict[str, Any] | None = None,
    verbose: bool = True,
) -> PipelineStage:
    p, s = build_stage_config(
        stage_cls, tmp_path, params=params, save_options=save_options
    )
    return stage_cls(params=p, save_options=s, verbose=verbose)


def _stub_save_output(
    key: str, value: Any, output_path: Path
) -> Path | ArtifactPathRecord:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if key in {"fwdtransforms", "invtransforms"}:
        manifest = {
            "format": "ants_transform_chain",
            "transforms": [
                {
                    "path": f"{output_path.stem}_00.mat",
                    "invert": key == "invtransforms",
                }
            ],
        }
        output_path.write_text(json.dumps(manifest), encoding="utf-8")
        payload_path = output_path.parent / f"{output_path.stem}_00.mat"
        payload_path.write_text("stub", encoding="utf-8")
        return ArtifactPathRecord(path=output_path, payload=(payload_path,))
    if output_path.suffix == ".json":
        output_path.write_text(json.dumps(value), encoding="utf-8")
        return output_path
    output_path.write_bytes(b"stub")
    return output_path


def _patch_shipped_stage_for_run(
    stage: PipelineStage,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outputs = STUB_FORWARD_OUTPUTS[type(stage)]
    monkeypatch.setattr(stage, "load_param", lambda _key, value: value)
    monkeypatch.setattr(stage, "forward", lambda **_: dict(outputs))
    monkeypatch.setattr(stage, "save_output", _stub_save_output)


@pytest.fixture(params=SHIPPED_STAGE_CLASSES, ids=lambda cls: cls.__name__)
def shipped_stage_cls(request: pytest.FixtureRequest) -> StageFactory:
    return request.param


@pytest.fixture
def shipped_stage(
    shipped_stage_cls: StageFactory,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> PipelineStage:
    stage = make_stage(shipped_stage_cls, tmp_path)
    _patch_shipped_stage_for_run(stage, monkeypatch)
    return stage


def step_ctx(step_id: str = STEP_ID, **kwargs: Any) -> RuntimeContext:
    return RuntimeContext(step_id=step_id, **kwargs)


# ---------------------------------------------------------------------------
# RuntimeContext
# ---------------------------------------------------------------------------


class TestRuntimeContext:
    # TODO: no need to test independently; should be covered by pipeline stage tests
    def test_defaults_are_empty_dicts_and_no_ids(self) -> None:
        ctx = RuntimeContext()
        assert ctx.run_id is None
        assert ctx.step_id is None
        assert ctx.metadata == {}
        assert ctx.artifacts == {}
        assert ctx.steps_completed == []

    def test_run_id_is_accessible_via_ctx_reference(self, tmp_path: Path) -> None:
        captured: dict[str, Any] = {}

        def _forward(**params: Any) -> dict[str, Any]:
            captured["run_id"] = params["input_nii"]
            return {"output_nii": b"\x01"}

        params, save_options = build_stage_config(
            DummyPipelineStage,
            tmp_path,
            params={"input_nii": "ctx.run_id"},
        )
        stage = DummyPipelineStage(params=params, save_options=save_options)
        stage.forward = _forward  # type: ignore[method-assign]
        stage.load_param = lambda _key, value: value  # type: ignore[method-assign]
        stage.run(RuntimeContext(run_id="sub-001", step_id=STEP_ID))
        assert captured["run_id"] == "sub-001"


# ---------------------------------------------------------------------------
# Dummy stage — full PipelineStage contract (config + run machinery).
# ---------------------------------------------------------------------------


class TestDummyPipelineStageConstruction:  # Again, no need to test base class machinery with dummy stage
    def test_init_with_pipeline_style_config(self, tmp_path: Path) -> None:
        stage = make_stage(DummyPipelineStage, tmp_path)
        assert isinstance(stage.params, dict)
        assert isinstance(stage.save_options, dict)
        assert stage.params
        assert stage.save_options == {}

    def test_params_may_include_extra_keys_when_check_params_allows(  # Too specific; can check if needed in a concrete that needs it
        self, tmp_path: Path
    ) -> None:
        params, save_options = build_stage_config(
            DummyPipelineStage,
            tmp_path,
            params={"algorithm_options": {"verbose": True}},
        )
        stage = DummyPipelineStage(params=params, save_options=save_options)
        assert stage.params["algorithm_options"] == {"verbose": True}

    def test_save_options_accepts_str_and_path_and_drops_none(
        self, tmp_path: Path
    ) -> None:
        params, _ = build_stage_config(DummyPipelineStage, tmp_path)
        as_str = tmp_path / "from_str.nii.gz"
        as_path = tmp_path / "from_path.nii.gz"
        stage = DummyPipelineStage(
            params=params,
            save_options={
                "output_nii": str(as_str),
                "from_path": as_path,
                "skipped": None,
            },
        )
        assert stage.save_options == {
            "output_nii": as_str.resolve(),
            "from_path": as_path.resolve(),
        }

    def test_init_rejects_non_dict_params(self, tmp_path: Path) -> None:
        _, save_options = build_stage_config(DummyPipelineStage, tmp_path)
        with pytest.raises(TypeError, match="params"):
            DummyPipelineStage(
                params="not-a-dict", save_options=save_options  # type: ignore[arg-type]
            )

    def test_init_rejects_non_dict_save_options(self, tmp_path: Path) -> None:
        params, _ = build_stage_config(DummyPipelineStage, tmp_path)
        with pytest.raises(TypeError, match="save_options"):
            DummyPipelineStage(params=params, save_options=[])  # type: ignore[arg-type]

    def test_init_rejects_invalid_save_options_value_type(self, tmp_path: Path) -> None:
        params, save_options = build_stage_config(DummyPipelineStage, tmp_path)
        save_options = {**save_options, "output_nii": 123}
        with pytest.raises(TypeError, match="save_options"):
            DummyPipelineStage(params=params, save_options=save_options)

    def test_check_params_validates_values(self, tmp_path: Path) -> None:
        params, save_options = build_stage_config(
            DummyPipelineStage,
            tmp_path,
            params={"scale_factor": "not-numeric"},
        )
        with pytest.raises(ValueError):
            DummyPipelineStage(params=params, save_options=save_options)

    def test_check_params_enforces_required_keys(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="input_nii"):
            DummyPipelineStage(params={"scale_factor": 1.0}, save_options={})

    def test_params_assignment_is_deep_copied(self, tmp_path: Path) -> None:
        stage = make_stage(DummyPipelineStage, tmp_path)
        external = dict(stage.params)
        external[next(iter(external))] = "mutated"
        assert stage.params != external


class TestDummyPipelineStageRun:  # same; test only shipped
    def test_run_returns_context(self, tmp_path: Path) -> None:
        ctx = make_stage(DummyPipelineStage, tmp_path).run(step_ctx())
        assert isinstance(ctx, RuntimeContext)

    def test_run_publishes_outputs_under_step_id(self, tmp_path: Path) -> None:
        ctx = make_stage(DummyPipelineStage, tmp_path).run(step_ctx("dummy"))
        assert "dummy" in ctx.artifacts
        assert isinstance(ctx.artifacts["dummy"], dict)

    def test_run_auto_generates_step_id_when_omitted(self, tmp_path: Path) -> None:
        ctx = make_stage(DummyPipelineStage, tmp_path).run(RuntimeContext())
        assert ctx.step_id == "step_0000"
        assert "step_0000" in ctx.artifacts

    def test_run_auto_generates_step_id_from_steps_completed(
        self, tmp_path: Path
    ) -> None:
        ctx = RuntimeContext(steps_completed=["denoise", "register"])
        out = make_stage(DummyPipelineStage, tmp_path).run(ctx)
        assert out.step_id == "step_0002"
        assert out.steps_completed == ["denoise", "register", "step_0002"]

    def test_run_rejects_user_provided_step_id_overlapping_completed(
        self, tmp_path: Path
    ) -> None:
        ctx = RuntimeContext(
            step_id="denoise",
            steps_completed=["denoise", "register"],
        )
        with pytest.raises(ValueError, match="user-provided.*index 2.*index 0"):
            make_stage(DummyPipelineStage, tmp_path).run(ctx)

    def test_run_appends_to_steps_completed(self, tmp_path: Path) -> None:
        ctx = RuntimeContext(step_id="b", steps_completed=["a"])
        out = make_stage(DummyPipelineStage, tmp_path).run(ctx)
        assert out.steps_completed == ["a", "b"]

    def test_run_skips_when_enable_false(self, tmp_path: Path) -> None:
        params, save_options = build_stage_config(
            DummyPipelineStage,
            tmp_path,
            params={"enable": False},
        )
        ctx = step_ctx("skipped")
        out = DummyPipelineStage(params=params, save_options=save_options).run(ctx)
        assert "skipped" not in out.artifacts
        assert "skipped" not in out.metadata
        assert out.steps_completed == []

    def test_run_skips_when_enable_resolved_from_ctx(self, tmp_path: Path) -> None:
        params, save_options = build_stage_config(
            DummyPipelineStage,
            tmp_path,
            params={"enable": "ctx.artifacts.qc.passed"},
        )
        ctx = RuntimeContext(
            step_id="conditional",
            artifacts={"qc": {"passed": False}},
        )
        out = DummyPipelineStage(params=params, save_options=save_options).run(ctx)
        assert "conditional" not in out.artifacts
        assert out.steps_completed == []

    def test_run_proceeds_when_enable_resolved_from_ctx(self, tmp_path: Path) -> None:
        params, save_options = build_stage_config(
            DummyPipelineStage,
            tmp_path,
            params={"enable": "ctx.artifacts.qc.passed"},
        )
        ctx = RuntimeContext(
            step_id="conditional",
            artifacts={"qc": {"passed": True}},
        )
        out = DummyPipelineStage(params=params, save_options=save_options).run(ctx)
        assert "conditional" in out.artifacts
        assert out.steps_completed == ["conditional"]

    def test_run_rejects_non_boolean_enable(self, tmp_path: Path) -> None:
        params, save_options = build_stage_config(
            DummyPipelineStage,
            tmp_path,
            params={"enable": "yes"},
        )
        with pytest.raises(TypeError, match="enable"):
            DummyPipelineStage(params=params, save_options=save_options).run(step_ctx())

    def test_run_skips_without_loading_other_params_when_enable_false(
        self, tmp_path: Path
    ) -> None:
        loaded: list[str] = []

        class TrackingStage(DummyPipelineStage):
            def load_param(self, key: str, value: Any) -> Any:
                loaded.append(key)
                return super().load_param(key, value)

        params = {
            "enable": "ctx.artifacts.qc.passed",
            "input_nii": "ctx.artifacts.missing.out_image",
            "scale_factor": 2.0,
        }
        save_options = {"output_nii": str(tmp_path / "out.nii.gz")}
        ctx = RuntimeContext(
            step_id="gated",
            artifacts={"qc": {"passed": False}},
        )
        out = TrackingStage(params=params, save_options=save_options).run(ctx)
        assert loaded == []
        assert "gated" not in out.artifacts
        assert out.steps_completed == []

    def test_call_dunder_delegates_to_run(self, tmp_path: Path) -> None:
        stage = make_stage(DummyPipelineStage, tmp_path)
        assert stage(step_ctx()) == stage.run(step_ctx())

    def test_run_loads_params_via_load_param(self, tmp_path: Path) -> None:
        raw_path = tmp_path / "raw.nii.gz"
        raw_path.write_bytes(b"\x0a")
        params, save_options = build_stage_config(
            DummyPipelineStage,
            tmp_path,
            params={"input_nii": str(raw_path)},
        )
        ctx = DummyPipelineStage(params=params, save_options=save_options).run(
            step_ctx()
        )
        assert ctx.artifacts[STEP_ID]["output_nii"] == bytes(
            [min(255, int(0x0A * 2.0))]
        )

    def test_run_resolves_ctx_reference_in_params(self, tmp_path: Path) -> None:
        template = tmp_path / "template.nii.gz"
        template.write_bytes(b"\x05")
        params, save_options = build_stage_config(
            DummyPipelineStage,
            tmp_path,
            params={"input_nii": "ctx.metadata.template_path"},
        )
        ctx = step_ctx(metadata={"template_path": str(template)})
        out = DummyPipelineStage(params=params, save_options=save_options).run(ctx)
        assert "output_nii" in out.artifacts[STEP_ID]

    def test_run_resolves_ctx_reference_to_prior_in_memory_artifact(
        self, tmp_path: Path
    ) -> None:
        params, save_options = build_stage_config(
            DummyPipelineStage,
            tmp_path,
            params={"input_nii": "ctx.artifacts.first.output_nii"},
        )
        ctx = RuntimeContext(
            step_id="second",
            artifacts={"first": {"output_nii": b"\x03"}},
        )
        out = DummyPipelineStage(params=params, save_options=save_options).run(ctx)
        assert out.artifacts["second"]["output_nii"] == bytes(
            [min(255, int(0x03 * 2.0))]
        )

    def test_run_raises_on_unresolved_ctx_reference(self, tmp_path: Path) -> None:
        params, save_options = build_stage_config(
            DummyPipelineStage,
            tmp_path,
            params={"input_nii": "ctx.artifacts.missing.output_nii"},
        )
        with pytest.raises(KeyError, match="Failed to resolve"):
            DummyPipelineStage(params=params, save_options=save_options).run(step_ctx())

    def test_run_persists_output_when_save_option_set(self, tmp_path: Path) -> None:
        out_file = tmp_path / "out.nii.gz"
        stage = make_stage(
            DummyPipelineStage,
            tmp_path,
            save_options={"output_nii": str(out_file)},
        )
        ctx = stage.run(step_ctx())
        assert out_file.exists()
        assert out_file.stat().st_size > 0
        assert isinstance(ctx.artifacts[STEP_ID]["output_nii"], bytes)
        assert ctx.metadata[STEP_ID]["saved_paths"]["output_nii"] == str(
            out_file.resolve()
        )

    def test_save_output_writes_file_at_target_path(self, tmp_path: Path) -> None:
        stage = make_stage(DummyPipelineStage, tmp_path)
        target = tmp_path / "explicit_target.nii.gz"
        record = stage.save_output("output_nii", b"\x01\x02", target)
        assert target.exists()
        assert target.read_bytes() == b"\x01\x02"
        assert record == target

    def test_run_rejects_invalid_ctx_type(self, tmp_path: Path) -> None:
        stage = make_stage(DummyPipelineStage, tmp_path)
        with pytest.raises(TypeError, match="RuntimeContext"):
            stage.run(ctx="not-a-context")  # type: ignore[arg-type]

    def test_run_rejects_forward_returning_non_dict(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        stage = make_stage(DummyPipelineStage, tmp_path)
        monkeypatch.setattr(stage, "forward", lambda **kw: "nope")
        with pytest.raises(TypeError, match="must return a dict"):
            stage.run(step_ctx())

    def test_run_allows_forward_outputs_not_listed_in_save_options(
        self, tmp_path: Path
    ) -> None:
        stage = make_stage(DummyPipelineStage, tmp_path)

        def _forward(**_: Any) -> dict[str, Any]:
            return {"output_nii": b"x", "extra": 1}

        stage.forward = _forward  # type: ignore[method-assign]
        ctx = stage.run(step_ctx())
        assert ctx.artifacts[STEP_ID]["extra"] == 1


class TestDummyUpdateMetadata:  # same; test only shipped
    def test_default_update_metadata_is_empty(self, tmp_path: Path) -> None:
        ctx = make_stage(DummyPipelineStage, tmp_path).run(step_ctx())
        assert ctx.metadata[STEP_ID] == {}

    def test_run_records_update_metadata_under_step_id(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        stage = make_stage(DummyPipelineStage, tmp_path)
        monkeypatch.setattr(
            stage, "update_metadata", lambda outputs: {"chosen": "linear"}
        )
        ctx = stage.run(step_ctx("m"))
        assert ctx.metadata["m"] == {"chosen": "linear"}

    def test_update_metadata_receives_in_memory_outputs_not_paths(
        self, tmp_path: Path
    ) -> None:
        out_file = tmp_path / "out.nii.gz"
        stage = make_stage(
            DummyPipelineStage,
            tmp_path,
            save_options={"output_nii": str(out_file)},
        )
        seen: dict[str, Any] = {}

        def _capture(outputs: dict[str, Any]) -> dict[str, Any]:
            seen.update(outputs)
            return {}

        stage.update_metadata = _capture  # type: ignore[method-assign]
        ctx = stage.run(step_ctx())
        assert isinstance(seen["output_nii"], bytes)
        assert ctx.metadata[STEP_ID]["saved_paths"]["output_nii"] == str(
            out_file.resolve()
        )

    def test_run_rejects_update_metadata_returning_non_dict(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        stage = make_stage(DummyPipelineStage, tmp_path)
        monkeypatch.setattr(stage, "update_metadata", lambda outputs: "nope")
        with pytest.raises(TypeError, match="update_metadata"):
            stage.run(step_ctx())

    def test_run_preserves_existing_metadata_keys(self, tmp_path: Path) -> None:
        ctx = step_ctx("b", metadata={"a": {"kept": 1}})
        out = make_stage(DummyPipelineStage, tmp_path).run(ctx)
        assert out.metadata["a"] == {"kept": 1}
        assert "b" in out.metadata


# ---------------------------------------------------------------------------
# Shipped stages — construction and shared run() contract (stubbed I/O).
# ---------------------------------------------------------------------------


class TestShippedStageConstruction:
    def test_init_with_pipeline_style_config(
        self, shipped_stage_cls: StageFactory, tmp_path: Path
    ) -> None:
        stage = make_stage(shipped_stage_cls, tmp_path)
        assert isinstance(stage.params, dict)
        assert isinstance(stage.save_options, dict)
        assert stage.params

    def test_save_options_may_omit_outputs(
        self, shipped_stage_cls: StageFactory, tmp_path: Path
    ) -> None:
        params, _ = build_stage_config(shipped_stage_cls, tmp_path)
        stage = shipped_stage_cls(params=params, save_options={})
        assert stage.save_options == {}

    def test_save_options_accepts_str_and_path_and_drops_none(
        self, shipped_stage_cls: StageFactory, tmp_path: Path
    ) -> None:
        params, _ = build_stage_config(shipped_stage_cls, tmp_path)
        key = PRIMARY_SAVE_KEY[shipped_stage_cls]
        as_str = tmp_path / "from_str.nii.gz"
        stage = shipped_stage_cls(
            params=params,
            save_options={key: str(as_str), "skipped": None},
        )
        assert stage.save_options == {key: as_str.resolve()}

    def test_init_rejects_non_dict_params(
        self, shipped_stage_cls: StageFactory, tmp_path: Path
    ) -> None:
        _, save_options = build_stage_config(shipped_stage_cls, tmp_path)
        with pytest.raises(TypeError, match="params"):
            shipped_stage_cls(
                params="not-a-dict", save_options=save_options  # type: ignore[arg-type]
            )

    def test_init_rejects_non_dict_save_options(
        self, shipped_stage_cls: StageFactory, tmp_path: Path
    ) -> None:
        params, _ = build_stage_config(shipped_stage_cls, tmp_path)
        with pytest.raises(TypeError, match="save_options"):
            shipped_stage_cls(params=params, save_options=[])  # type: ignore[arg-type]

    def test_init_rejects_invalid_save_options_value_type(
        self, shipped_stage_cls: StageFactory, tmp_path: Path
    ) -> None:
        params, save_options = build_stage_config(shipped_stage_cls, tmp_path)
        key = PRIMARY_SAVE_KEY[shipped_stage_cls]
        save_options = {**save_options, key: 123}
        with pytest.raises(TypeError, match="save_options"):
            shipped_stage_cls(params=params, save_options=save_options)

    def test_init_rejects_non_bool_verbose(
        self, shipped_stage_cls: StageFactory, tmp_path: Path
    ) -> None:
        params, save_options = build_stage_config(shipped_stage_cls, tmp_path)
        with pytest.raises(TypeError, match="verbose"):
            shipped_stage_cls(
                params=params, save_options=save_options, verbose="yes"  # type: ignore[arg-type]
            )

    def test_params_assignment_is_deep_copied(
        self, shipped_stage_cls: StageFactory, tmp_path: Path
    ) -> None:
        stage = make_stage(shipped_stage_cls, tmp_path)
        external = dict(stage.params)
        external[next(iter(external))] = "mutated"
        assert stage.params != external

    def test_apply_transforms_check_params_enforces_required_keys(
        self, tmp_path: Path
    ) -> None:
        with pytest.raises(ValueError, match="Missing required parameter"):
            ANTsApplyTransforms(
                params={"image": "a.nii.gz"},
                save_options={},
            )

    def test_apply_transforms_check_params_rejects_whichtoinvert_with_manifest(
        self, tmp_path: Path
    ) -> None:
        manifest_path = tmp_path / "fwd.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "format": "ants_transform_chain",
                    "transforms": [{"path": "t.mat", "invert": False}],
                }
            ),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="cannot be set"):
            ANTsApplyTransforms(
                params={
                    "image": "moving.nii.gz",
                    "target": "fixed.nii.gz",
                    "transformlist": str(manifest_path),
                    "whichtoinvert": [True],
                },
                save_options={},
            )


class TestShippedStageRun:
    def test_run_returns_context(self, shipped_stage: PipelineStage) -> None:
        ctx = shipped_stage.run(step_ctx())
        assert isinstance(ctx, RuntimeContext)

    def test_run_publishes_outputs_under_step_id(
        self, shipped_stage: PipelineStage
    ) -> None:
        ctx = shipped_stage.run(step_ctx("shipped"))
        assert "shipped" in ctx.artifacts
        assert set(ctx.artifacts["shipped"]) == set(
            STUB_FORWARD_OUTPUTS[type(shipped_stage)]
        )

    def test_run_auto_generates_step_id_when_omitted(
        self, shipped_stage: PipelineStage
    ) -> None:
        ctx = shipped_stage.run()
        assert ctx.step_id == "step_0000"
        assert "step_0000" in ctx.artifacts

    def test_run_rejects_forward_returning_non_dict(
        self, shipped_stage: PipelineStage, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(shipped_stage, "forward", lambda **kw: "nope")
        with pytest.raises(TypeError, match="must return a dict"):
            shipped_stage.run(step_ctx())

    def test_run_injects_saved_paths_without_custom_metadata(
        self, shipped_stage_cls: StageFactory, tmp_path: Path
    ) -> None:
        key = PRIMARY_SAVE_KEY[shipped_stage_cls]
        if key in {"fwdtransforms", "invtransforms"}:
            out_file = tmp_path / "transforms.json"
        elif key == "deleted":
            out_file = tmp_path / "deleted.json"
        elif key == "array":
            out_file = tmp_path / "saved.npy"
        elif key in {"passed", "value"}:
            out_file = tmp_path / f"qc_{key}.txt"
        elif key == "report":
            out_file = tmp_path / "qc_report.json"
        else:
            out_file = tmp_path / "saved.nii.gz"
        stage = make_stage(
            shipped_stage_cls,
            tmp_path,
            save_options={key: str(out_file)},
        )
        monkeypatch = pytest.MonkeyPatch()
        _patch_shipped_stage_for_run(stage, monkeypatch)
        try:
            ctx = stage.run(step_ctx())
        finally:
            monkeypatch.undo()
        saved = ctx.metadata[STEP_ID]["saved_paths"][key]
        if isinstance(saved, dict):
            assert saved["path"] == str(out_file.resolve())
        else:
            assert saved == str(out_file.resolve())

    def test_run_rejects_invalid_ctx_type(self, shipped_stage: PipelineStage) -> None:
        with pytest.raises(TypeError, match="RuntimeContext"):
            shipped_stage.run(ctx="not-a-context")  # type: ignore[arg-type]


class TestStageLogging:
    @pytest.fixture(
        params=[DummyPipelineStage, *SHIPPED_STAGE_CLASSES], ids=lambda c: c.__name__
    )
    def any_stage_cls(self, request: pytest.FixtureRequest) -> StageFactory:
        return request.param

    def test_verbose_false_suppresses_info_but_not_warning(
        self,
        any_stage_cls: StageFactory,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        stage = make_stage(any_stage_cls, tmp_path, verbose=False)
        with caplog.at_level(logging.DEBUG):
            stage.log("hidden-info", "info")
            stage.log("visible-warning", "warning")
        assert "hidden-info" not in caplog.text
        assert "visible-warning" in caplog.text

    def test_invalid_log_level_raises(
        self, tmp_path: Path
    ) -> None:  # TODO: remove, not representative of shipped usage
        stage = make_stage(DummyPipelineStage, tmp_path)
        with pytest.raises(ValueError, match="level"):
            stage.log("boom", "fatal")  # type: ignore[arg-type]

    def test_run_logs_elapsed_time_with_step_id(
        self,
        any_stage_cls: StageFactory,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        stage = make_stage(any_stage_cls, tmp_path)
        if any_stage_cls is not DummyPipelineStage:
            _patch_shipped_stage_for_run(stage, monkeypatch)
        with caplog.at_level(logging.INFO):
            stage.run(step_ctx("timed"))
        assert "| timed]" in caplog.text
        assert "Finished in" in caplog.text


# ---------------------------------------------------------------------------
# Compose — sequential orchestration of child stages.
# ---------------------------------------------------------------------------


class ScaleStage(PipelineStage):
    """Multiply the first byte of ``input_nii`` by ``scale_factor``."""

    REQUIRED_PARAMS = frozenset({"input_nii"})

    def load_param(self, key: str, value: Any) -> Any:
        if key == "input_nii":
            if isinstance(value, (bytes, bytearray)):
                return bytes(value)
            return Path(value).read_bytes()
        return value

    def forward(self, **params: Any) -> dict[str, Any]:
        raw: bytes = params["input_nii"]
        scale = float(params.get("scale_factor", 1.0))
        sample = raw[0] if raw else 0
        return {"output_nii": bytes([min(255, int(sample * scale))])}

    def save_output(self, key: str, value: Any, output_path: Path) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(value)
        return output_path


def _scale_stage(
    tmp_path: Path,
    *,
    scale_factor: float = 2.0,
    input_ref: str | None = None,
) -> ScaleStage:
    if input_ref is None:
        input_path = tmp_path / "input.nii.gz"
        input_path.write_bytes(b"\x05")
        input_ref = str(input_path)
    return ScaleStage(
        params={"input_nii": input_ref, "scale_factor": scale_factor},
        save_options={},
    )


class TestCompose:
    def test_runs_stages_sequentially_with_explicit_step_ids(
        self, tmp_path: Path
    ) -> None:
        first = _scale_stage(tmp_path, scale_factor=2.0)
        second = _scale_stage(
            tmp_path,
            scale_factor=3.0,
            input_ref="ctx.artifacts.first.output_nii",
        )
        pipeline = Compose([first, second], step_ids=["first", "second"])
        ctx = pipeline.run(step_ctx())
        assert ctx.artifacts["first"]["output_nii"] == bytes([10])
        assert ctx.artifacts["second"]["output_nii"] == bytes([30])
        assert ctx.steps_completed == ["first", "second"]

    def test_auto_generates_child_step_ids_from_parent(self, tmp_path: Path) -> None:
        pipeline = Compose([_scale_stage(tmp_path)])
        ctx = pipeline.run(step_ctx("parent"))
        assert "parent.0" in ctx.artifacts
        assert ctx.artifacts["parent.0"]["output_nii"] == bytes([10])

    def test_none_step_id_uses_auto_generated_step_ids(self, tmp_path: Path) -> None:
        pipeline = Compose(
            [
                _scale_stage(tmp_path, scale_factor=2.0),
                _scale_stage(
                    tmp_path,
                    scale_factor=3.0,
                    input_ref="ctx.artifacts.step_0000.output_nii",
                ),
            ],
            step_ids=[None, None],
        )
        ctx = pipeline.run(RuntimeContext())
        assert ctx.artifacts["step_0000"]["output_nii"] == bytes([10])
        assert ctx.artifacts["step_0001"]["output_nii"] == bytes([30])
        assert ctx.steps_completed == ["step_0000", "step_0001"]

    def test_start_and_end_slice_child_execution(self, tmp_path: Path) -> None:
        pipeline = Compose(
            [
                _scale_stage(tmp_path, scale_factor=2.0),
                _scale_stage(tmp_path, scale_factor=3.0),
                _scale_stage(tmp_path, scale_factor=4.0),
            ],
            step_ids=["a", "b", "c"],
        )
        ctx = pipeline.run(step_ctx(), start=1, end=2)
        assert "a" not in ctx.artifacts
        assert ctx.artifacts["b"]["output_nii"] == bytes([15])
        assert "c" not in ctx.artifacts

    def test_flatten_inlines_nested_compose(self, tmp_path: Path) -> None:
        inner = Compose([_scale_stage(tmp_path)], step_ids=["inner"])
        outer = Compose([inner, _scale_stage(tmp_path)], step_ids=["outer", "tail"])
        flat = outer.flatten()
        assert len(flat) == 2
        assert flat.stages[0] is inner.stages[0]
        assert flat.step_ids == ("inner", "tail")

    def test_len_uses_flattened_stage_count(self, tmp_path: Path) -> None:
        inner = Compose([_scale_stage(tmp_path)], step_ids=["inner"])
        outer = Compose([inner, _scale_stage(tmp_path)], step_ids=["outer", "tail"])
        assert len(outer) == 2

    def test_get_index_of_first_finds_matching_stage(self, tmp_path: Path) -> None:
        pipeline = Compose(
            [
                _scale_stage(tmp_path),
                DummyPipelineStage(*build_stage_config(DummyPipelineStage, tmp_path)),
            ],
            step_ids=["a", "b"],
        )
        index = pipeline.get_index_of_first(
            lambda stage: isinstance(stage, DummyPipelineStage)
        )
        assert index == 1
        assert pipeline.get_index_of_first(lambda stage: False) is None

    def test_rejects_empty_string_step_id(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="non-empty string"):
            Compose([_scale_stage(tmp_path)], step_ids=[""])

    def test_rejects_step_ids_length_mismatch(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="length"):
            Compose([_scale_stage(tmp_path)], step_ids=["a", "b"])

    def test_rejects_invalid_stage_type(self) -> None:
        with pytest.raises(TypeError, match="PipelineStage"):
            Compose([123])  # type: ignore[list-item]

    def test_set_stages_resizes_pipeline(self, tmp_path: Path) -> None:
        pipeline = Compose([_scale_stage(tmp_path)], step_ids=["only"])
        pipeline.set_stages(
            [_scale_stage(tmp_path), _scale_stage(tmp_path, scale_factor=2.0)],
            step_ids=["first", "second"],
        )
        assert len(pipeline.stages) == 2
        assert pipeline.step_ids == ("first", "second")

    def test_bare_stage_auto_generates_step_id_without_parent(
        self, tmp_path: Path
    ) -> None:
        ctx = Compose([_scale_stage(tmp_path)]).run(RuntimeContext())
        assert ctx.step_id == "step_0000"
        assert "step_0000" in ctx.artifacts

    def test_call_delegates_to_run(self, tmp_path: Path) -> None:
        pipeline = Compose([_scale_stage(tmp_path)], step_ids=["only"])
        assert pipeline(step_ctx()) == pipeline.run(step_ctx())


class TestQCStagePersistence:
    @pytest.mark.parametrize(
        "stage_cls",
        [CheckVoxelSpacing, CheckDimensions],
        ids=lambda cls: cls.__name__,
    )
    def test_save_output_writes_passed_txt(
        self, stage_cls: StageFactory, tmp_path: Path
    ) -> None:
        passed_path = tmp_path / "passed.txt"
        stage = make_stage(stage_cls, tmp_path)
        record = stage.save_output("passed", True, passed_path)
        assert record == passed_path.resolve()
        assert passed_path.read_text(encoding="utf-8") == "True\n"

    @pytest.mark.parametrize(
        ("stage_cls", "value", "expected_line"),
        [
            (CheckVoxelSpacing, (1.0, 2.0, 3.0), "1.0,2.0,3.0\n"),
            (CheckDimensions, (64, 128, 256), "64,128,256\n"),
        ],
        ids=["CheckVoxelSpacing", "CheckDimensions"],
    )
    def test_save_output_writes_value_txt(
        self,
        stage_cls: StageFactory,
        value: tuple[float, ...] | tuple[int, ...],
        expected_line: str,
        tmp_path: Path,
    ) -> None:
        value_path = tmp_path / "value.txt"
        stage = make_stage(stage_cls, tmp_path)
        record = stage.save_output("value", value, value_path)
        assert record == value_path.resolve()
        assert value_path.read_text(encoding="utf-8") == expected_line

    @pytest.mark.parametrize(
        "stage_cls",
        [CheckVoxelSpacing, CheckDimensions],
        ids=lambda cls: cls.__name__,
    )
    def test_save_output_rejects_non_txt_extension(
        self, stage_cls: StageFactory, tmp_path: Path
    ) -> None:
        stage = make_stage(stage_cls, tmp_path)
        with pytest.raises(ValueError, match=r"\.txt"):
            stage.save_output("passed", True, tmp_path / "qc.json")

    @pytest.mark.parametrize(
        "stage_cls",
        [CheckVoxelSpacing, CheckDimensions],
        ids=lambda cls: cls.__name__,
    )
    def test_save_output_writes_report_json(
        self, stage_cls: StageFactory, tmp_path: Path
    ) -> None:
        report_path = tmp_path / "report.json"
        stage = make_stage(stage_cls, tmp_path)
        report = {
            "check": stage_cls.__name__,
            "passed": True,
            "value": [1, 1, 1],
            "id": "sub-001",
        }
        record = stage.save_output("report", report, report_path)
        assert record == report_path.resolve()
        assert json.loads(report_path.read_text(encoding="utf-8")) == report

    @pytest.mark.parametrize(
        "stage_cls",
        [CheckVoxelSpacing, CheckDimensions],
        ids=lambda cls: cls.__name__,
    )
    def test_save_output_rejects_non_json_report_extension(
        self, stage_cls: StageFactory, tmp_path: Path
    ) -> None:
        stage = make_stage(stage_cls, tmp_path)
        with pytest.raises(ValueError, match=r"\.json"):
            stage.save_output(
                "report",
                {"check": stage_cls.__name__, "passed": True, "value": [1]},
                tmp_path / "report.txt",
            )


class TestQCStageReport:
    def test_forward_report_omits_id_when_none(self, tmp_path: Path) -> None:
        from types import SimpleNamespace

        stage = make_stage(CheckVoxelSpacing, tmp_path)
        image = SimpleNamespace(spacing=(1.0, 1.0, 1.0))
        outputs = stage.forward(image=image, expected=(1.0, 1.0, 1.0))
        assert outputs["report"] == {
            "check": "CheckVoxelSpacing",
            "passed": True,
            "value": [1.0, 1.0, 1.0],
        }

    def test_forward_report_includes_id_when_set(self, tmp_path: Path) -> None:
        from types import SimpleNamespace

        stage = make_stage(CheckDimensions, tmp_path)
        image = SimpleNamespace(shape=(64, 64, 64))
        outputs = stage.forward(image=image, expected=(64, 64, 64), id="ctx.run_id")
        assert outputs["report"] == {
            "check": "CheckDimensions",
            "passed": True,
            "value": [64, 64, 64],
            "id": "ctx.run_id",
        }
