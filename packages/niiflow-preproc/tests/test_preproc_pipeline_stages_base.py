"""Tests for the :class:`PipelineStage` base class and :class:`RuntimeContext`.

:class:`DummyPipelineStage` exercises the full ``run()`` / context / persistence
contract without ANTs. Shipped stages inherit this machinery unchanged; that
inheritance is verified separately in
``test_preproc_pipeline_stages_contract.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from niiflow.preproc.pipelines.pipeline_stages import RuntimeContext
from stage_helpers import (
    STEP_ID,
    DummyPipelineStage,
    build_stage_config,
    make_stage,
    step_ctx,
)


class TestRuntimeContext:
    def test_defaults_are_empty_dicts_and_no_ids(self) -> None:
        ctx = RuntimeContext()
        assert ctx.run_id is None
        assert ctx.step_id is None
        assert ctx.metadata == {}
        assert ctx.outputs == {}
        assert ctx.steps_completed == []

    def test_run_id_is_accessible_via_ctx_reference(self, tmp_path: Path) -> None:
        captured: dict[str, Any] = {}

        def _forward(**params: Any) -> dict[str, Any]:
            captured["run_id"] = params["input_nii"]
            return {"output_nii": b"\x01"}

        params, save_outputs = build_stage_config(
            DummyPipelineStage,
            tmp_path,
            params={"input_nii": "ctx.run_id"},
        )
        stage = DummyPipelineStage(params=params, save_outputs=save_outputs)
        stage.forward = _forward  # type: ignore[method-assign]
        stage.load_param = lambda _key, value: value  # type: ignore[method-assign]
        stage.run(RuntimeContext(run_id="sub-001", step_id=STEP_ID))
        assert captured["run_id"] == "sub-001"


# ---------------------------------------------------------------------------
# Dummy stage — full PipelineStage contract (config + run machinery).
# ---------------------------------------------------------------------------


class TestDummyPipelineStageConstruction:
    def test_init_with_pipeline_style_config(self, tmp_path: Path) -> None:
        stage = make_stage(DummyPipelineStage, tmp_path)
        assert isinstance(stage.params, dict)
        assert isinstance(stage.save_outputs, dict)
        assert stage.params
        assert stage.save_outputs == {}

    def test_params_may_include_extra_keys_when_check_params_allows(
        self, tmp_path: Path
    ) -> None:
        params, save_outputs = build_stage_config(
            DummyPipelineStage,
            tmp_path,
            params={"algorithm_options": {"verbose": True}},
        )
        stage = DummyPipelineStage(params=params, save_outputs=save_outputs)
        assert stage.params["algorithm_options"] == {"verbose": True}

    def test_save_outputs_accepts_str_and_path_and_drops_none(
        self, tmp_path: Path
    ) -> None:
        params, _ = build_stage_config(DummyPipelineStage, tmp_path)
        as_str = tmp_path / "from_str.nii.gz"
        as_path = tmp_path / "from_path.nii.gz"
        stage = DummyPipelineStage(
            params=params,
            save_outputs={
                "output_nii": str(as_str),
                "from_path": as_path,
                "skipped": None,
            },
        )
        assert stage.save_outputs == {
            "output_nii": as_str.resolve(),
            "from_path": as_path.resolve(),
        }

    def test_init_rejects_non_dict_params(self, tmp_path: Path) -> None:
        _, save_outputs = build_stage_config(DummyPipelineStage, tmp_path)
        with pytest.raises(TypeError, match="params"):
            DummyPipelineStage(
                params="not-a-dict", save_outputs=save_outputs  # type: ignore[arg-type]
            )

    def test_init_rejects_non_dict_save_outputs(self, tmp_path: Path) -> None:
        params, _ = build_stage_config(DummyPipelineStage, tmp_path)
        with pytest.raises(TypeError, match="save_outputs"):
            DummyPipelineStage(params=params, save_outputs=[])  # type: ignore[arg-type]

    def test_init_rejects_invalid_save_outputs_value_type(self, tmp_path: Path) -> None:
        params, save_outputs = build_stage_config(DummyPipelineStage, tmp_path)
        save_outputs = {**save_outputs, "output_nii": 123}
        with pytest.raises(TypeError, match="save_outputs"):
            DummyPipelineStage(params=params, save_outputs=save_outputs)

    def test_check_params_validates_values(self, tmp_path: Path) -> None:
        params, save_outputs = build_stage_config(
            DummyPipelineStage,
            tmp_path,
            params={"scale_factor": "not-numeric"},
        )
        with pytest.raises(ValueError):
            DummyPipelineStage(params=params, save_outputs=save_outputs)

    def test_check_params_enforces_required_keys(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="input_nii"):
            DummyPipelineStage(params={"scale_factor": 1.0}, save_outputs={})

    def test_params_assignment_is_deep_copied(self, tmp_path: Path) -> None:
        stage = make_stage(DummyPipelineStage, tmp_path)
        external = dict(stage.params)
        external[next(iter(external))] = "mutated"
        assert stage.params != external


class TestDummyPipelineStageRun:
    def test_run_returns_context(self, tmp_path: Path) -> None:
        ctx = make_stage(DummyPipelineStage, tmp_path).run(step_ctx())
        assert isinstance(ctx, RuntimeContext)

    def test_run_publishes_outputs_under_step_id(self, tmp_path: Path) -> None:
        ctx = make_stage(DummyPipelineStage, tmp_path).run(step_ctx("dummy"))
        assert "dummy" in ctx.outputs
        assert isinstance(ctx.outputs["dummy"], dict)

    def test_run_auto_generates_step_id_when_omitted(self, tmp_path: Path) -> None:
        ctx = make_stage(DummyPipelineStage, tmp_path).run(RuntimeContext())
        assert ctx.step_id == "step_0000"
        assert "step_0000" in ctx.outputs

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
        params, save_outputs = build_stage_config(
            DummyPipelineStage,
            tmp_path,
            params={"enable": False},
        )
        ctx = step_ctx("skipped")
        out = DummyPipelineStage(params=params, save_outputs=save_outputs).run(ctx)
        assert "skipped" not in out.outputs
        assert "skipped" not in out.metadata
        assert out.steps_completed == []

    def test_run_skips_when_enable_resolved_from_ctx(self, tmp_path: Path) -> None:
        params, save_outputs = build_stage_config(
            DummyPipelineStage,
            tmp_path,
            params={"enable": "ctx.outputs.qc.passed"},
        )
        ctx = RuntimeContext(
            step_id="conditional",
            outputs={"qc": {"passed": False}},
        )
        out = DummyPipelineStage(params=params, save_outputs=save_outputs).run(ctx)
        assert "conditional" not in out.outputs
        assert out.steps_completed == []

    def test_run_proceeds_when_enable_resolved_from_ctx(self, tmp_path: Path) -> None:
        params, save_outputs = build_stage_config(
            DummyPipelineStage,
            tmp_path,
            params={"enable": "ctx.outputs.qc.passed"},
        )
        ctx = RuntimeContext(
            step_id="conditional",
            outputs={"qc": {"passed": True}},
        )
        out = DummyPipelineStage(params=params, save_outputs=save_outputs).run(ctx)
        assert "conditional" in out.outputs
        assert out.steps_completed == ["conditional"]

    def test_run_rejects_non_boolean_enable(self, tmp_path: Path) -> None:
        params, save_outputs = build_stage_config(
            DummyPipelineStage,
            tmp_path,
            params={"enable": "yes"},
        )
        with pytest.raises(TypeError, match="enable"):
            DummyPipelineStage(params=params, save_outputs=save_outputs).run(step_ctx())

    def test_run_skips_without_loading_other_params_when_enable_false(
        self, tmp_path: Path
    ) -> None:
        loaded: list[str] = []

        class TrackingStage(DummyPipelineStage):
            def load_param(self, key: str, value: Any) -> Any:
                loaded.append(key)
                return super().load_param(key, value)

        params = {
            "enable": "ctx.outputs.qc.passed",
            "input_nii": "ctx.outputs.missing.out_image",
            "scale_factor": 2.0,
        }
        save_outputs = {"output_nii": str(tmp_path / "out.nii.gz")}
        ctx = RuntimeContext(
            step_id="gated",
            outputs={"qc": {"passed": False}},
        )
        out = TrackingStage(params=params, save_outputs=save_outputs).run(ctx)
        assert loaded == []
        assert "gated" not in out.outputs
        assert out.steps_completed == []

    def test_call_dunder_delegates_to_run(self, tmp_path: Path) -> None:
        stage = make_stage(DummyPipelineStage, tmp_path)
        assert stage(step_ctx()) == stage.run(step_ctx())

    def test_run_loads_params_via_load_param(self, tmp_path: Path) -> None:
        raw_path = tmp_path / "raw.nii.gz"
        raw_path.write_bytes(b"\x0a")
        params, save_outputs = build_stage_config(
            DummyPipelineStage,
            tmp_path,
            params={"input_nii": str(raw_path)},
        )
        ctx = DummyPipelineStage(params=params, save_outputs=save_outputs).run(
            step_ctx()
        )
        assert ctx.outputs[STEP_ID]["output_nii"] == bytes([min(255, int(0x0A * 2.0))])

    def test_run_resolves_ctx_reference_in_params(self, tmp_path: Path) -> None:
        template = tmp_path / "template.nii.gz"
        template.write_bytes(b"\x05")
        params, save_outputs = build_stage_config(
            DummyPipelineStage,
            tmp_path,
            params={"input_nii": "ctx.metadata.template_path"},
        )
        ctx = step_ctx(metadata={"template_path": str(template)})
        out = DummyPipelineStage(params=params, save_outputs=save_outputs).run(ctx)
        assert "output_nii" in out.outputs[STEP_ID]

    def test_run_resolves_ctx_reference_to_prior_in_memory_output(
        self, tmp_path: Path
    ) -> None:
        params, save_outputs = build_stage_config(
            DummyPipelineStage,
            tmp_path,
            params={"input_nii": "ctx.outputs.first.output_nii"},
        )
        ctx = RuntimeContext(
            step_id="second",
            outputs={"first": {"output_nii": b"\x03"}},
        )
        out = DummyPipelineStage(params=params, save_outputs=save_outputs).run(ctx)
        assert out.outputs["second"]["output_nii"] == bytes([min(255, int(0x03 * 2.0))])

    def test_run_raises_on_unresolved_ctx_reference(self, tmp_path: Path) -> None:
        params, save_outputs = build_stage_config(
            DummyPipelineStage,
            tmp_path,
            params={"input_nii": "ctx.outputs.missing.output_nii"},
        )
        with pytest.raises(KeyError, match="Failed to resolve"):
            DummyPipelineStage(params=params, save_outputs=save_outputs).run(step_ctx())

    def test_run_persists_output_when_save_option_set(self, tmp_path: Path) -> None:
        out_file = tmp_path / "out.nii.gz"
        stage = make_stage(
            DummyPipelineStage,
            tmp_path,
            save_outputs={"output_nii": str(out_file)},
        )
        ctx = stage.run(step_ctx())
        assert out_file.exists()
        assert out_file.stat().st_size > 0
        assert isinstance(ctx.outputs[STEP_ID]["output_nii"], bytes)
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

    def test_run_allows_forward_outputs_not_listed_in_save_outputs(
        self, tmp_path: Path
    ) -> None:
        stage = make_stage(DummyPipelineStage, tmp_path)

        def _forward(**_: Any) -> dict[str, Any]:
            return {"output_nii": b"x", "extra": 1}

        stage.forward = _forward  # type: ignore[method-assign]
        ctx = stage.run(step_ctx())
        assert ctx.outputs[STEP_ID]["extra"] == 1


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
            save_outputs={"output_nii": str(out_file)},
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
