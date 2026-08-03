"""Tests for :class:`~niiflow.preproc.pipelines.pipeline_stages.compose.Compose`."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from niiflow.preproc.pipelines.pipeline_stages import (
    Compose,
    PipelineStage,
    RuntimeContext,
)
from stage_helpers import DummyPipelineStage, build_stage_config, step_ctx

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
        save_outputs={},
    )


class TestCompose:
    def test_runs_stages_sequentially_with_explicit_step_ids(
        self, tmp_path: Path
    ) -> None:
        first = _scale_stage(tmp_path, scale_factor=2.0)
        second = _scale_stage(
            tmp_path,
            scale_factor=3.0,
            input_ref="ctx.outputs.first.output_nii",
        )
        pipeline = Compose([first, second], step_ids=["first", "second"])
        ctx = pipeline.run(step_ctx())
        assert ctx.outputs["first"]["output_nii"] == bytes([10])
        assert ctx.outputs["second"]["output_nii"] == bytes([30])
        assert ctx.steps_completed == ["first", "second"]

    def test_auto_generates_child_step_ids_from_parent(self, tmp_path: Path) -> None:
        pipeline = Compose([_scale_stage(tmp_path)])
        ctx = pipeline.run(step_ctx("parent"))
        assert "parent.0" in ctx.outputs
        assert ctx.outputs["parent.0"]["output_nii"] == bytes([10])

    def test_none_step_id_uses_auto_generated_step_ids(self, tmp_path: Path) -> None:
        pipeline = Compose(
            [
                _scale_stage(tmp_path, scale_factor=2.0),
                _scale_stage(
                    tmp_path,
                    scale_factor=3.0,
                    input_ref="ctx.outputs.step_0000.output_nii",
                ),
            ],
            step_ids=[None, None],
        )
        ctx = pipeline.run(RuntimeContext())
        assert ctx.outputs["step_0000"]["output_nii"] == bytes([10])
        assert ctx.outputs["step_0001"]["output_nii"] == bytes([30])
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
        assert "a" not in ctx.outputs
        assert ctx.outputs["b"]["output_nii"] == bytes([15])
        assert "c" not in ctx.outputs

    def _three_stage_pipeline(self, tmp_path: Path) -> Compose:
        return Compose(
            [
                _scale_stage(tmp_path, scale_factor=2.0),
                _scale_stage(tmp_path, scale_factor=3.0),
                _scale_stage(tmp_path, scale_factor=4.0),
            ],
            step_ids=["a", "b", "c"],
        )

    def test_default_end_runs_every_stage(self, tmp_path: Path) -> None:
        ctx = self._three_stage_pipeline(tmp_path).run(step_ctx())
        assert ctx.steps_completed == ["a", "b", "c"]

    def test_negative_start_counts_back_from_end(self, tmp_path: Path) -> None:
        ctx = self._three_stage_pipeline(tmp_path).run(step_ctx(), start=-2)
        assert ctx.steps_completed == ["b", "c"]

    def test_negative_end_excludes_trailing_stages(self, tmp_path: Path) -> None:
        ctx = self._three_stage_pipeline(tmp_path).run(step_ctx(), end=-1)
        assert ctx.steps_completed == ["a", "b"]

    def test_negative_start_and_end_combine(self, tmp_path: Path) -> None:
        ctx = self._three_stage_pipeline(tmp_path).run(step_ctx(), start=-3, end=-2)
        assert ctx.steps_completed == ["a"]

    def test_empty_range_runs_nothing(self, tmp_path: Path) -> None:
        ctx = self._three_stage_pipeline(tmp_path).run(step_ctx(), start=1, end=1)
        assert ctx.steps_completed == []

    @pytest.mark.parametrize("start", [4, -4])
    def test_rejects_out_of_range_start(self, tmp_path: Path, start: int) -> None:
        # ``start == len(stages)`` stays legal (empty selection); anything past it
        # is a caller error rather than a silently truncated selection.
        with pytest.raises(ValueError, match=r"`start` .* out of range"):
            self._three_stage_pipeline(tmp_path).run(step_ctx(), start=start)

    @pytest.mark.parametrize("end", [4, -4])
    def test_rejects_out_of_range_end(self, tmp_path: Path, end: int) -> None:
        with pytest.raises(ValueError, match=r"`end` .* out of range"):
            self._three_stage_pipeline(tmp_path).run(step_ctx(), end=end)

    def test_start_equal_to_stage_count_is_allowed(self, tmp_path: Path) -> None:
        ctx = self._three_stage_pipeline(tmp_path).run(step_ctx(), start=3)
        assert ctx.steps_completed == []

    @pytest.mark.parametrize(("start", "end"), [(2, 1), (-1, -2), (2, -2)])
    def test_rejects_end_before_start(
        self, tmp_path: Path, start: int, end: int
    ) -> None:
        with pytest.raises(ValueError, match="must not precede"):
            self._three_stage_pipeline(tmp_path).run(step_ctx(), start=start, end=end)

    def test_rejects_non_int_start(self, tmp_path: Path) -> None:
        with pytest.raises(TypeError, match="`start` must be an int"):
            self._three_stage_pipeline(tmp_path).run(step_ctx(), start="1")  # type: ignore[arg-type]

    def test_rejects_non_int_end(self, tmp_path: Path) -> None:
        with pytest.raises(TypeError, match="`end` must be an int or None"):
            self._three_stage_pipeline(tmp_path).run(step_ctx(), end=1.5)  # type: ignore[arg-type]

    def test_rejects_duplicate_step_ids_at_construction(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match=r"step id 'a' at index 2 duplicates"):
            Compose(
                [
                    _scale_stage(tmp_path),
                    _scale_stage(tmp_path),
                    _scale_stage(tmp_path),
                ],
                step_ids=["a", "b", "a"],
            )

    def test_set_stages_rejects_duplicate_step_ids(self, tmp_path: Path) -> None:
        pipeline = Compose([_scale_stage(tmp_path)], step_ids=["only"])
        with pytest.raises(ValueError, match="duplicates index"):
            pipeline.set_stages(
                [_scale_stage(tmp_path), _scale_stage(tmp_path)],
                step_ids=["dup", "dup"],
            )
        assert pipeline.step_ids == ("only",)

    def test_allows_repeated_none_step_ids(self, tmp_path: Path) -> None:
        pipeline = Compose(
            [_scale_stage(tmp_path), _scale_stage(tmp_path)],
            step_ids=[None, None],
        )
        assert pipeline.step_ids == (None, None)

    def test_nested_duplicate_step_ids_still_raise_at_runtime(
        self, tmp_path: Path
    ) -> None:
        inner = Compose([_scale_stage(tmp_path)], step_ids=["dup"])
        outer = Compose([inner, _scale_stage(tmp_path)], step_ids=["outer", "dup"])
        with pytest.raises(ValueError, match="step ids must be unique"):
            outer.run(step_ctx())

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
        assert "step_0000" in ctx.outputs

    def test_call_delegates_to_run(self, tmp_path: Path) -> None:
        pipeline = Compose([_scale_stage(tmp_path)], step_ids=["only"])
        assert pipeline(step_ctx()) == pipeline.run(step_ctx())
