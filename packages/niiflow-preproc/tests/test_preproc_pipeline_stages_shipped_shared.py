"""Shared helpers and contract tests for every shipped pipeline stage.

Owns the shipped-stage registry, config builders, and stubbed-run helpers used by
cross-stage fixtures. Dummy-stage helpers live in
``test_preproc_pipeline_stages_base``. Domain behaviour of concrete stages lives in
the matching ``test_preproc_pipeline_stages_*`` modules.
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
    CheckImageSimilarity,
    CheckVoxelSpacing,
    ClampIntensities,
    CropToMask,
    CropToRange,
    Delete,
    GetImage,
    MinmaxNorm,
    PadToRange,
    PipelineStage,
    PointwiseArithmetic,
    RelabelMask,
    Rename,
    Reorient,
    RuntimeContext,
    SmoothMask,
    SyncMetadata,
    ToNumpy,
    ZTransformNorm,
)
from niiflow.preproc.pipelines.pipeline_stages.pipeline_stage import (
    ArtifactPathRecord,
)
from test_preproc_pipeline_stages_base import (
    STEP_ID,
    DummyPipelineStage,
    build_stage_config as build_dummy_stage_config,
    make_stage as make_dummy_stage,
    step_ctx,
    touch,
)

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
    CheckImageSimilarity,
    ApplyMask,
    SmoothMask,
    RelabelMask,
    PointwiseArithmetic,
    Delete,
    GetImage,
    Rename,
    Reorient,
    ToNumpy,
    SyncMetadata,
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
    CheckImageSimilarity: "passed",
    ApplyMask: "out_image",
    SmoothMask: "out_mask",
    RelabelMask: "out_mask",
    PointwiseArithmetic: "out_image",
    Delete: "deleted",
    GetImage: "out_image",
    Rename: "out_path",
    Reorient: "out_image",
    ToNumpy: "array",
    SyncMetadata: "out_image",
}


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


def _check_image_similarity_stage_config(
    tmp_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    image = touch(tmp_path / "image.nii.gz")
    target = touch(tmp_path / "target.nii.gz")
    return (
        {
            "image": str(image),
            "target": str(target),
            "correlation": 0.5,
        },
        {},
    )


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


def _pointwise_arithmetic_stage_config(
    tmp_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    image = touch(tmp_path / "image.nii.gz")
    return (
        {"image": str(image), "operations": [{"mul": 1.0}]},
        {"out_image": None},
    )


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


def _sync_metadata_stage_config(
    tmp_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    image = touch(tmp_path / "image.nii.gz")
    reference = touch(tmp_path / "reference.nii.gz")
    return (
        {"image": str(image), "reference": str(reference)},
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
    CheckImageSimilarity: _check_image_similarity_stage_config,
    Delete: _delete_stage_config,
    Rename: _rename_stage_config,
    GetImage: _get_image_stage_config,
    ApplyMask: _apply_mask_stage_config,
    SmoothMask: _smooth_mask_stage_config,
    RelabelMask: _relabel_mask_stage_config,
    PointwiseArithmetic: _pointwise_arithmetic_stage_config,
    Reorient: _reorient_stage_config,
    ToNumpy: _to_numpy_stage_config,
    SyncMetadata: _sync_metadata_stage_config,
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
    CheckImageSimilarity: {
        "passed": True,
        "value": {"correlation": 1.0},
        "report": {
            "check": "CheckImageSimilarity",
            "passed": True,
            "value": {"correlation": 1.0},
            "cutoffs": {"correlation": 0.5},
            "combine": "all",
            "metric_passed": {"correlation": True},
        },
    },
    Delete: {"deleted": []},
    Rename: {"out_path": object()},
    GetImage: {"out_image": object()},
    ApplyMask: {"out_image": object()},
    SmoothMask: {"out_mask": object()},
    RelabelMask: {"out_mask": object()},
    PointwiseArithmetic: {"out_image": object()},
    Reorient: {"out_image": object()},
    ToNumpy: {"array": object(), "metadata": {}},
    SyncMetadata: {"out_image": object()},
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
        return build_dummy_stage_config(
            stage_cls, tmp_path, params=params, save_outputs=save_outputs
        )
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
    """Instantiate a stage with defaults from :func:`build_stage_config`."""
    if stage_cls is DummyPipelineStage:
        return make_dummy_stage(
            stage_cls,
            tmp_path,
            params=params,
            save_outputs=save_outputs,
            verbose=verbose,
        )
    p, s = build_stage_config(
        stage_cls, tmp_path, params=params, save_outputs=save_outputs
    )
    return stage_cls(params=p, save_outputs=s, verbose=verbose)


def stub_save_output(
    key: str, value: Any, output_path: Path
) -> Path | ArtifactPathRecord:
    """Write a stub artifact for shipped-stage persistence tests."""
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
    """Stub ``load_param`` / ``forward`` / ``save_output`` for a shipped stage."""
    outputs = STUB_FORWARD_OUTPUTS[type(stage)]
    monkeypatch.setattr(stage, "load_param", lambda _key, value: value)
    monkeypatch.setattr(stage, "forward", lambda **_: dict(outputs))
    monkeypatch.setattr(stage, "save_output", stub_save_output)


@pytest.fixture(params=SHIPPED_STAGE_CLASSES, ids=lambda cls: cls.__name__)
def shipped_stage_cls(request: pytest.FixtureRequest) -> StageFactory:
    """Each shipped concrete stage class, one per parametrized run."""
    return request.param


@pytest.fixture
def shipped_stage(
    shipped_stage_cls: StageFactory,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> PipelineStage:
    """A shipped stage instance with ``forward`` / ``save_output`` stubbed out."""
    stage = make_stage(shipped_stage_cls, tmp_path)
    patch_shipped_stage_for_run(stage, monkeypatch)
    return stage


class TestShippedStageConstruction:
    def test_init_with_pipeline_style_config(
        self, shipped_stage_cls: StageFactory, tmp_path: Path
    ) -> None:
        stage = make_stage(shipped_stage_cls, tmp_path)
        assert isinstance(stage.params, dict)
        assert isinstance(stage.save_outputs, dict)
        assert stage.params

    def test_save_outputs_may_omit_outputs(
        self, shipped_stage_cls: StageFactory, tmp_path: Path
    ) -> None:
        params, _ = build_stage_config(shipped_stage_cls, tmp_path)
        stage = shipped_stage_cls(params=params, save_outputs={})
        assert stage.save_outputs == {}

    def test_save_outputs_accepts_str_and_path_and_drops_none(
        self, shipped_stage_cls: StageFactory, tmp_path: Path
    ) -> None:
        params, _ = build_stage_config(shipped_stage_cls, tmp_path)
        key = PRIMARY_SAVE_KEY[shipped_stage_cls]
        as_str = tmp_path / "from_str.nii.gz"
        stage = shipped_stage_cls(
            params=params,
            save_outputs={key: str(as_str), "skipped": None},
        )
        assert stage.save_outputs == {key: as_str.resolve()}

    def test_init_rejects_non_dict_params(
        self, shipped_stage_cls: StageFactory, tmp_path: Path
    ) -> None:
        _, save_outputs = build_stage_config(shipped_stage_cls, tmp_path)
        with pytest.raises(TypeError, match="params"):
            shipped_stage_cls(
                params="not-a-dict", save_outputs=save_outputs  # type: ignore[arg-type]
            )

    def test_init_rejects_non_dict_save_outputs(
        self, shipped_stage_cls: StageFactory, tmp_path: Path
    ) -> None:
        params, _ = build_stage_config(shipped_stage_cls, tmp_path)
        with pytest.raises(TypeError, match="save_outputs"):
            shipped_stage_cls(params=params, save_outputs=[])  # type: ignore[arg-type]

    def test_init_rejects_invalid_save_outputs_value_type(
        self, shipped_stage_cls: StageFactory, tmp_path: Path
    ) -> None:
        params, save_outputs = build_stage_config(shipped_stage_cls, tmp_path)
        key = PRIMARY_SAVE_KEY[shipped_stage_cls]
        save_outputs = {**save_outputs, key: 123}
        with pytest.raises(TypeError, match="save_outputs"):
            shipped_stage_cls(params=params, save_outputs=save_outputs)

    def test_init_rejects_non_bool_verbose(
        self, shipped_stage_cls: StageFactory, tmp_path: Path
    ) -> None:
        params, save_outputs = build_stage_config(shipped_stage_cls, tmp_path)
        with pytest.raises(TypeError, match="verbose"):
            shipped_stage_cls(
                params=params, save_outputs=save_outputs, verbose="yes"  # type: ignore[arg-type]
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
                save_outputs={},
            )

    def test_apply_transforms_check_params_allows_whichtoinvert_with_manifest(
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
        stage = ANTsApplyTransforms(
            params={
                "image": "moving.nii.gz",
                "target": "fixed.nii.gz",
                "transformlist": str(manifest_path),
                "whichtoinvert": [True],
            },
            save_outputs={},
        )
        assert stage.params["whichtoinvert"] == [True]


class TestShippedStageRun:
    def test_run_returns_context(self, shipped_stage: PipelineStage) -> None:
        ctx = shipped_stage.run(step_ctx())
        assert isinstance(ctx, RuntimeContext)

    def test_run_publishes_outputs_under_step_id(
        self, shipped_stage: PipelineStage
    ) -> None:
        ctx = shipped_stage.run(step_ctx("shipped"))
        assert "shipped" in ctx.outputs
        assert set(ctx.outputs["shipped"]) == set(
            STUB_FORWARD_OUTPUTS[type(shipped_stage)]
        )

    def test_run_auto_generates_step_id_when_omitted(
        self, shipped_stage: PipelineStage
    ) -> None:
        ctx = shipped_stage.run()
        assert ctx.step_id == "step_0000"
        assert "step_0000" in ctx.outputs

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
            save_outputs={key: str(out_file)},
        )
        monkeypatch = pytest.MonkeyPatch()
        patch_shipped_stage_for_run(stage, monkeypatch)
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
            patch_shipped_stage_for_run(stage, monkeypatch)
        with caplog.at_level(logging.INFO):
            stage.run(step_ctx("timed"))
        assert "| timed]" in caplog.text
        assert "Finished in" in caplog.text
