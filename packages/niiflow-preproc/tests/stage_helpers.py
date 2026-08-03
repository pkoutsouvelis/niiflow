"""Shared pipeline-stage test fixtures data: configs, stubs, and a dummy stage.

Imported by ``conftest.py`` and by the pipeline-stage test modules. Keeping the
per-stage config builders in one place lets the cross-stage contract tests in
``test_preproc_pipeline_stages_contract.py`` parametrize over every shipped stage,
while each per-stage module still overrides only what it needs.
"""

from __future__ import annotations

import json
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
    CropToMask,
    CropToRange,
    Delete,
    GetImage,
    MinmaxNorm,
    PadToRange,
    PipelineStage,
    RelabelMask,
    Rename,
    Reorient,
    RuntimeContext,
    SmoothMask,
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
    SmoothMask,
    RelabelMask,
    Delete,
    GetImage,
    Rename,
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
    SmoothMask: "out_mask",
    RelabelMask: "out_mask",
    Delete: "deleted",
    GetImage: "out_image",
    Rename: "out_path",
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


def touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x00")
    return path


def _image_stage_config(tmp_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    image = touch(tmp_path / "image.nii.gz")
    return {"image": str(image)}, {"out_image": None}


def _registration_stage_config(
    tmp_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    moving = touch(tmp_path / "moving.nii.gz")
    fixed = touch(tmp_path / "fixed.nii.gz")
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
    moving = touch(tmp_path / "moving.nii.gz")
    fixed = touch(tmp_path / "fixed.nii.gz")
    transform = touch(tmp_path / "transform.mat")
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
    image = touch(tmp_path / "image.nii.gz")
    ranges_path = tmp_path / "ranges.json"
    ranges_path.write_text(json.dumps([[0, 8], [0, 8], [0, 8]]), encoding="utf-8")
    return {"image": str(image), "ranges": str(ranges_path)}, {"out_image": None}


def _crop_to_mask_stage_config(
    tmp_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    image = touch(tmp_path / "image.nii.gz")
    return {"image": str(image)}, {"out_image": None}


def _center_crop_stage_config(
    tmp_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    image = touch(tmp_path / "image.nii.gz")
    return {"image": str(image), "shape": (16, 16, 16)}, {"out_image": None}


def _pad_to_range_stage_config(
    tmp_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    image = touch(tmp_path / "image.nii.gz")
    ranges_path = tmp_path / "pad_ranges.json"
    ranges_path.write_text(json.dumps([[0, 1], [0, 1], [0, 1]]), encoding="utf-8")
    return {"image": str(image), "ranges": str(ranges_path)}, {"out_image": None}


def _center_pad_stage_config(
    tmp_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    image = touch(tmp_path / "image.nii.gz")
    return {"image": str(image), "shape": (32, 32, 32)}, {"out_image": None}


def _intensity_norm_stage_config(
    tmp_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    image = touch(tmp_path / "image.nii.gz")
    return {"image": str(image)}, {"out_image": None}


def _check_voxel_spacing_stage_config(
    tmp_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    image = touch(tmp_path / "image.nii.gz")
    return {"image": str(image), "expected": (1.0, 1.0, 1.0)}, {}


def _check_dimensions_stage_config(
    tmp_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    image = touch(tmp_path / "image.nii.gz")
    return {"image": str(image), "expected": (64, 64, 64)}, {}


def _delete_stage_config(
    tmp_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    target = touch(tmp_path / "to_delete.txt")
    return {"paths": [str(target)], "missing_ok": False}, {}


def _rename_stage_config(
    tmp_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    source = touch(tmp_path / "to_rename.txt")
    return (
        {"source": str(source), "dest": str(tmp_path / "renamed.txt")},
        {"out_path": None},
    )


def _get_image_stage_config(
    tmp_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    image = touch(tmp_path / "image.nii.gz")
    return {"image": str(image)}, {"out_image": None}


def _reorient_stage_config(
    tmp_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    image = touch(tmp_path / "image.nii.gz")
    return {"image": str(image), "orientation": "RAS"}, {"out_image": None}


def _apply_mask_stage_config(
    tmp_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    image = touch(tmp_path / "image.nii.gz")
    mask = touch(tmp_path / "mask.nii.gz")
    return {"image": str(image), "mask": str(mask)}, {"out_image": None}


def _smooth_mask_stage_config(
    tmp_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    mask = touch(tmp_path / "mask.nii.gz")
    return {"mask": str(mask), "sigma": 1.0}, {"out_mask": None}


def _relabel_mask_stage_config(
    tmp_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    mask = touch(tmp_path / "mask.nii.gz")
    return (
        {"mask": str(mask), "mapping": {0: 0, 1: 2}, "dtype": "int32"},
        {"out_mask": None},
    )


def _to_numpy_stage_config(
    tmp_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    image = touch(tmp_path / "image.nii.gz")
    return {"image": str(image), "dtype": "float32"}, {"array": None}


def _ants_resample_stage_config(
    tmp_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    image = touch(tmp_path / "image.nii.gz")
    return (
        {"image": str(image), "resample_params": [1.0, 1.0, 1.0]},
        {"out_image": None},
    )


def _ants_resample_to_target_stage_config(
    tmp_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    image = touch(tmp_path / "image.nii.gz")
    target = touch(tmp_path / "target.nii.gz")
    return (
        {"image": str(image), "target": str(target)},
        {"out_image": None},
    )


def _ants_preprocess_brain_image_stage_config(
    tmp_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    image = touch(tmp_path / "image.nii.gz")
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
    Rename: _rename_stage_config,
    GetImage: _get_image_stage_config,
    ApplyMask: _apply_mask_stage_config,
    SmoothMask: _smooth_mask_stage_config,
    RelabelMask: _relabel_mask_stage_config,
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
    Rename: {"out_path": object()},
    GetImage: {"out_image": object()},
    ApplyMask: {"out_image": object()},
    SmoothMask: {"out_mask": object()},
    RelabelMask: {"out_mask": object()},
    Reorient: {"out_image": object()},
    ToNumpy: {"array": object(), "metadata": {}},
}


def build_stage_config(
    stage_cls: StageFactory,
    tmp_path: Path,
    *,
    params: dict[str, Any] | None = None,
    save_outputs: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return ``(params, save_outputs)`` defaults for a stage class."""
    if stage_cls is DummyPipelineStage:
        input_nii = touch(tmp_path / "input.nii.gz")
        base_params: dict[str, Any] = {
            "input_nii": str(input_nii),
            "scale_factor": 2.0,
        }
        base_save_outputs: dict[str, Any] = {"output_nii": None}
    else:
        try:
            builder = STAGE_CONFIG_BUILDERS[stage_cls]
        except KeyError as exc:
            raise NotImplementedError(
                f"Register {stage_cls.__name__} in STAGE_CONFIG_BUILDERS "
                f"before adding it to SHIPPED_STAGE_CLASSES."
            ) from exc
        base_params, base_save_outputs = builder(tmp_path)

    if params is not None:
        base_params = {**base_params, **params}
    if save_outputs is not None:
        base_save_outputs = {**base_save_outputs, **save_outputs}
    return base_params, base_save_outputs


def make_stage(
    stage_cls: StageFactory,
    tmp_path: Path,
    *,
    params: dict[str, Any] | None = None,
    save_outputs: dict[str, Any] | None = None,
    verbose: bool = True,
) -> PipelineStage:
    p, s = build_stage_config(
        stage_cls, tmp_path, params=params, save_outputs=save_outputs
    )
    return stage_cls(params=p, save_outputs=s, verbose=verbose)


def stub_save_output(
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


def patch_shipped_stage_for_run(
    stage: PipelineStage,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outputs = STUB_FORWARD_OUTPUTS[type(stage)]
    monkeypatch.setattr(stage, "load_param", lambda _key, value: value)
    monkeypatch.setattr(stage, "forward", lambda **_: dict(outputs))
    monkeypatch.setattr(stage, "save_output", stub_save_output)


def step_ctx(step_id: str = STEP_ID, **kwargs: Any) -> RuntimeContext:
    return RuntimeContext(step_id=step_id, **kwargs)
