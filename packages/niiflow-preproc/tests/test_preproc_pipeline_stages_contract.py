"""Cross-stage contract tests: every shipped stage honours the base machinery.

Each concrete stage in :data:`stage_helpers.SHIPPED_STAGE_CLASSES` is checked for
pipeline-style construction and for compatibility with the shared ``run()`` flow,
with ``forward`` / ``save_output`` stubbed so no ANTs work happens. The base
behaviour itself is tested in ``test_preproc_pipeline_stages_base.py``; domain
behaviour lives in the ``test_preproc_func_*`` and ``test_preproc_stages_*``
modules.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from niiflow.preproc.pipelines.pipeline_stages import (
    ANTsApplyTransforms,
    PipelineStage,
    RuntimeContext,
)
from stage_helpers import (
    PRIMARY_SAVE_KEY,
    SHIPPED_STAGE_CLASSES,
    STEP_ID,
    STUB_FORWARD_OUTPUTS,
    DummyPipelineStage,
    StageFactory,
    build_stage_config,
    make_stage,
    patch_shipped_stage_for_run,
    step_ctx,
)


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
                save_outputs={},
            )


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
