"""Tests for :class:`ProcessingStage` and concrete stage implementations.

These tests pin the public contract of :class:`ProcessingStage` and every
concrete subclass listed in :data:`STAGE_CLASSES`. Append new concrete stages
to that tuple and register defaults in :func:`build_stage_config` to extend
coverage automatically.

Construction uses plain ``params`` / ``save_options`` dicts (as from a
pipeline config), not helpers on the stage classes themselves.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, TypeAlias

import pytest

from niiflow.preproc.pipelines.processing_stages import (
    DummyProcessingStage,
    ProcessingStage,
    RuntimeContext,
)
from niiflow.preproc.pipelines.processing_stages.processing_stage import (
    _resolve_ctx_path,
    _validate_key_set,
)

# ---------------------------------------------------------------------------
# Registry of concrete stage classes covered by the shared test suite.
# Append new concrete stages here when implemented.
# ---------------------------------------------------------------------------

StageFactory: TypeAlias = type[ProcessingStage]

STAGE_CLASSES: tuple[StageFactory, ...] = (DummyProcessingStage,)


@pytest.fixture(params=STAGE_CLASSES, ids=lambda cls: cls.__name__)
def stage_cls(request: pytest.FixtureRequest) -> StageFactory:
    """Concrete stage class under test for this iteration."""
    return request.param


def build_stage_config(
    stage_cls: StageFactory,
    tmp_path: Path,
    *,
    params: dict[str, Any] | None = None,
    save_options: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return ``(params, save_options)`` defaults for a registered concrete stage.

    Override via ``params`` / ``save_options``; keys are merged into the defaults.
    Raises :class:`NotImplementedError` for stages not yet registered (expected
    until their pipeline config fixtures are added).
    """
    if stage_cls is DummyProcessingStage:
        input_nii = tmp_path / "input.nii.gz"
        input_nii.write_bytes(b"\x01\x02\x03")
        base_params: dict[str, Any] = {
            "input_nii": str(input_nii),
            "scale_factor": 2.0,
        }
        base_save_options: dict[str, Any] = {"output_nii": None}
    else:
        raise NotImplementedError(
            f"Register default params/save_options for {stage_cls.__name__} "
            f"in build_stage_config before adding it to STAGE_CLASSES."
        )

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
) -> ProcessingStage:
    """Construct a stage instance from pipeline-style config dicts."""
    p, s = build_stage_config(
        stage_cls, tmp_path, params=params, save_options=save_options
    )
    return stage_cls(params=p, save_options=s, verbose=verbose)


@pytest.fixture
def stage(stage_cls: StageFactory, tmp_path: Path) -> ProcessingStage:
    """A ready-to-run stage instance for the parametrized concrete class."""
    return make_stage(stage_cls, tmp_path)


# ---------------------------------------------------------------------------
# Unit tests for module-level helpers (not parametrized over stage classes).
# ---------------------------------------------------------------------------


class TestResolveCtxPath:
    def test_resolves_input_root_metadata_and_artifacts(self, tmp_path: Path) -> None:
        ctx = RuntimeContext(
            input_root=tmp_path,
            metadata={"subject": "sub-01"},
            artifacts={"mask": tmp_path / "mask.nii.gz"},
        )
        assert _resolve_ctx_path(ctx, "ctx.input_root") == tmp_path.resolve()
        assert _resolve_ctx_path(ctx, "ctx.metadata.subject") == "sub-01"
        assert _resolve_ctx_path(ctx, "ctx.artifacts.mask") == tmp_path / "mask.nii.gz"

    def test_rejects_non_ctx_prefix(self) -> None:
        ctx = RuntimeContext()
        with pytest.raises(ValueError, match="must start with 'ctx.'"):
            _resolve_ctx_path(ctx, "metadata.subject")

    def test_rejects_empty_segments(self) -> None:
        ctx = RuntimeContext()
        with pytest.raises(ValueError, match="empty segments"):
            _resolve_ctx_path(ctx, "ctx.")

    def test_missing_segment_raises_key_error(self) -> None:
        ctx = RuntimeContext()
        with pytest.raises(KeyError, match="Failed to resolve"):
            _resolve_ctx_path(ctx, "ctx.artifacts.missing")


class TestValidateKeySet:
    def test_accepts_exact_match(self) -> None:
        _validate_key_set({"a": 1, "b": 2}, ["b", "a"], "params")

    def test_reports_missing_and_extra(self) -> None:
        with pytest.raises(ValueError, match="missing \\['b'\\]"):
            _validate_key_set({"a": 1, "c": 3}, ["a", "b"], "params")
        with pytest.raises(ValueError, match="unexpected \\['c'\\]"):
            _validate_key_set({"a": 1, "c": 3}, ["a"], "params")


# ---------------------------------------------------------------------------
# Construction / validation (shared across all stage classes).
# ---------------------------------------------------------------------------


class TestStageConstruction:
    """Every concrete stage must accept valid config and reject invalid input."""

    def test_init_with_pipeline_style_config(
        self, stage_cls: StageFactory, tmp_path: Path
    ) -> None:
        stage = make_stage(stage_cls, tmp_path)
        assert isinstance(stage.params, dict)
        assert isinstance(stage.save_options, dict)
        assert set(stage.params) == set(stage_cls.PARAM_KEYS)
        assert set(stage.save_options) == set(stage_cls.OUTPUT_KEYS)

    def test_params_keys_must_match_param_keys(
        self, stage_cls: StageFactory, tmp_path: Path
    ) -> None:
        params, save_options = build_stage_config(stage_cls, tmp_path)
        params.pop(next(iter(params)))
        with pytest.raises(ValueError, match="params"):
            stage_cls(params=params, save_options=save_options)

    def test_save_options_keys_must_match_output_keys(
        self, stage_cls: StageFactory, tmp_path: Path
    ) -> None:
        params, save_options = build_stage_config(stage_cls, tmp_path)
        save_options.pop(next(iter(save_options)))
        with pytest.raises(ValueError, match="save_options"):
            stage_cls(params=params, save_options=save_options)

    def test_init_rejects_non_dict_params(
        self, stage_cls: StageFactory, tmp_path: Path
    ) -> None:
        _, save_options = build_stage_config(stage_cls, tmp_path)
        with pytest.raises(TypeError, match="params"):
            stage_cls(params="not-a-dict", save_options=save_options)  # type: ignore[arg-type]

    def test_init_rejects_non_dict_save_options(
        self, stage_cls: StageFactory, tmp_path: Path
    ) -> None:
        params, _ = build_stage_config(stage_cls, tmp_path)
        with pytest.raises(TypeError, match="save_options"):
            stage_cls(params=params, save_options=[])  # type: ignore[arg-type]

    def test_init_rejects_invalid_save_options_value_type(
        self, stage_cls: StageFactory, tmp_path: Path
    ) -> None:
        params, save_options = build_stage_config(stage_cls, tmp_path)
        key = next(iter(save_options))
        save_options[key] = 123
        with pytest.raises(TypeError, match="save_options"):
            stage_cls(params=params, save_options=save_options)

    def test_init_rejects_non_bool_verbose(
        self, stage_cls: StageFactory, tmp_path: Path
    ) -> None:
        params, save_options = build_stage_config(stage_cls, tmp_path)
        with pytest.raises(TypeError, match="verbose"):
            stage_cls(params=params, save_options=save_options, verbose="yes")  # type: ignore[arg-type]

    def test_check_params_validates_values(
        self, stage_cls: StageFactory, tmp_path: Path
    ) -> None:
        if stage_cls is not DummyProcessingStage:
            pytest.skip("add an invalid-value override for this stage in the test")
        params, save_options = build_stage_config(
            stage_cls, tmp_path, params={"scale_factor": "not-numeric"}
        )
        with pytest.raises(ValueError):
            stage_cls(params=params, save_options=save_options)

    def test_params_assignment_is_deep_copied(self, stage: ProcessingStage) -> None:
        external = dict(stage.params)
        external[next(iter(external))] = "mutated"
        assert stage.params != external


# ---------------------------------------------------------------------------
# RuntimeContext
# ---------------------------------------------------------------------------


class TestRuntimeContext:
    def test_input_root_resolved_to_absolute_path(self, tmp_path: Path) -> None:
        ctx = RuntimeContext(input_root=tmp_path)
        assert ctx.input_root == tmp_path.resolve()

    def test_input_root_none_stays_none(self) -> None:
        ctx = RuntimeContext()
        assert ctx.input_root is None


# ---------------------------------------------------------------------------
# End-to-end run() (shared across all stage classes).
# ---------------------------------------------------------------------------


class TestStageRun:
    """``run()`` must load params, forward, update ctx, and optionally save."""

    def test_run_without_ctx_creates_fresh_context(
        self, stage_cls: StageFactory, tmp_path: Path
    ) -> None:
        ctx = make_stage(stage_cls, tmp_path).run()
        assert isinstance(ctx, RuntimeContext)

    def test_call_dunder_delegates_to_run(
        self, stage_cls: StageFactory, tmp_path: Path
    ) -> None:
        stage = make_stage(stage_cls, tmp_path)
        assert stage() == stage.run()

    def test_run_loads_file_input_via_load_params(
        self, stage_cls: StageFactory, tmp_path: Path
    ) -> None:
        if not stage_cls.FILE_INPUT_KEYS:
            pytest.skip(f"{stage_cls.__name__} has no file inputs")
        file_key = stage_cls.FILE_INPUT_KEYS[0]
        raw_path = tmp_path / "raw.nii.gz"
        raw_path.write_bytes(b"\x0a")
        params, save_options = build_stage_config(
            stage_cls, tmp_path, params={file_key: str(raw_path)}
        )
        ctx = stage_cls(params=params, save_options=save_options).run()
        assert stage_cls.OUTPUT_KEYS[0] in ctx.artifacts

    def test_run_resolves_ctx_reference_in_params(
        self, stage_cls: StageFactory, tmp_path: Path
    ) -> None:
        if not stage_cls.FILE_INPUT_KEYS:
            pytest.skip(f"{stage_cls.__name__} has no file inputs")
        file_key = stage_cls.FILE_INPUT_KEYS[0]
        template = tmp_path / "template.nii.gz"
        template.write_bytes(b"\x05")
        params, save_options = build_stage_config(
            stage_cls,
            tmp_path,
            params={file_key: "ctx.metadata.template_path"},
        )
        ctx = RuntimeContext(metadata={"template_path": str(template)})
        out = stage_cls(params=params, save_options=save_options).run(ctx)
        assert stage_cls.OUTPUT_KEYS[0] in out.artifacts

    def test_run_persists_output_when_save_option_set(
        self, stage_cls: StageFactory, tmp_path: Path
    ) -> None:
        out_key = stage_cls.OUTPUT_KEYS[0]
        out_file = tmp_path / "out.nii.gz"
        stage = make_stage(stage_cls, tmp_path, save_options={out_key: str(out_file)})
        stage.run()
        assert out_file.exists()
        assert out_file.stat().st_size > 0

    def test_run_rejects_invalid_ctx_type(self, stage: ProcessingStage) -> None:
        with pytest.raises(TypeError, match="RuntimeContext"):
            stage.run(ctx="not-a-context")  # type: ignore[arg-type]

    def test_run_rejects_forward_returning_non_dict(
        self, stage: ProcessingStage, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(stage, "forward", lambda **kw: "nope")
        with pytest.raises(TypeError, match="must return a dict"):
            stage.run()

    def test_run_rejects_forward_with_wrong_keys(
        self, stage: ProcessingStage, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(stage, "forward", lambda **kw: {"wrong": 0})
        with pytest.raises(ValueError, match="forward outputs"):
            stage.run()

    def test_run_rejects_update_ctx_removing_prior_keys(
        self, stage: ProcessingStage, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _clear_artifacts(
            outputs: dict[str, Any], ctx: RuntimeContext
        ) -> RuntimeContext:
            ctx.artifacts.clear()
            return ctx

        monkeypatch.setattr(stage, "update_ctx", _clear_artifacts)
        ctx = RuntimeContext(artifacts={"keep": 1})
        with pytest.raises(ValueError, match="removed previously-present"):
            stage.run(ctx)

    def test_run_rejects_update_ctx_returning_wrong_type(
        self, stage: ProcessingStage, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(stage, "update_ctx", lambda o, c: "bad")
        with pytest.raises(TypeError, match="must return a RuntimeContext"):
            stage.run()


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


class TestStageLogging:
    def test_verbose_false_suppresses_info_but_not_warning(
        self, stage_cls: StageFactory, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        stage = make_stage(stage_cls, tmp_path, verbose=False)
        with caplog.at_level(logging.DEBUG):
            stage.log("hidden-info", "info")
            stage.log("visible-warning", "warning")
        assert "hidden-info" not in caplog.text
        assert "visible-warning" in caplog.text

    def test_invalid_log_level_raises(self, stage: ProcessingStage) -> None:
        with pytest.raises(ValueError, match="level"):
            stage.log("boom", "fatal")  # type: ignore[arg-type]
