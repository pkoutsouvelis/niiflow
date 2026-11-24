"""Tests for the QC stages in
:mod:`niiflow.preproc.pipelines.pipeline_stages.qc`.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest

from niiflow.preproc.pipelines.pipeline_stages import (
    AggregateChecks,
    CheckDimensions,
    CheckMetric,
    CheckVoxelSpacing,
)
from test_preproc_pipeline_stages_shipped_shared import StageFactory, make_stage


class TestQCStagePersistence:
    @pytest.mark.parametrize(
        "stage_cls",
        [CheckVoxelSpacing, CheckDimensions, CheckMetric, AggregateChecks],
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
            (CheckMetric, 0.8, "0.8\n"),
            (AggregateChecks, (True, False), "True,False\n"),
        ],
        ids=[
            "CheckVoxelSpacing",
            "CheckDimensions",
            "CheckMetric",
            "AggregateChecks",
        ],
    )
    def test_save_output_writes_value_txt(
        self,
        stage_cls: StageFactory,
        value: tuple[float, ...] | tuple[int, ...] | tuple[bool, ...] | float,
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
        [CheckVoxelSpacing, CheckDimensions, CheckMetric, AggregateChecks],
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
        [CheckVoxelSpacing, CheckDimensions, CheckMetric, AggregateChecks],
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
        [CheckVoxelSpacing, CheckDimensions, CheckMetric, AggregateChecks],
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


class TestRaiseOnFail:
    def test_spacing_raises_on_fail(self, tmp_path: Path) -> None:
        stage = make_stage(CheckVoxelSpacing, tmp_path)
        image = SimpleNamespace(spacing=(2.0, 2.0, 2.0))
        with pytest.raises(
            RuntimeError,
            match=r"CheckVoxelSpacing failed QC.*value=\(2\.0, 2\.0, 2\.0\)",
        ):
            stage.forward(
                image=image,
                expected=(1.0, 1.0, 1.0),
                raise_on_fail=True,
            )

    def test_dimensions_raises_on_fail(self, tmp_path: Path) -> None:
        stage = make_stage(CheckDimensions, tmp_path)
        image = SimpleNamespace(shape=(32, 32, 32))
        with pytest.raises(
            RuntimeError, match=r"CheckDimensions failed QC.*value=\(32, 32, 32\)"
        ):
            stage.forward(
                image=image,
                expected=(64, 64, 64),
                raise_on_fail=True,
            )

    def test_default_allows_soft_fail(self, tmp_path: Path) -> None:
        stage = make_stage(CheckVoxelSpacing, tmp_path)
        image = SimpleNamespace(spacing=(2.0, 2.0, 2.0))
        out = stage.forward(image=image, expected=(1.0, 1.0, 1.0))
        assert out["passed"] is False

    def test_raise_on_fail_skips_when_passed(self, tmp_path: Path) -> None:
        stage = make_stage(CheckVoxelSpacing, tmp_path)
        image = SimpleNamespace(spacing=(1.0, 1.0, 1.0))
        out = stage.forward(image=image, expected=(1.0, 1.0, 1.0), raise_on_fail=True)
        assert out["passed"] is True

    def test_rejects_non_bool_raise_on_fail(self, tmp_path: Path) -> None:
        stage = make_stage(CheckVoxelSpacing, tmp_path)
        with pytest.raises(TypeError, match="raise_on_fail"):
            stage.load_param("raise_on_fail", "yes")


class TestQCStageReport:
    def test_forward_report_omits_id_when_none(self, tmp_path: Path) -> None:
        stage = make_stage(CheckVoxelSpacing, tmp_path)
        image = SimpleNamespace(spacing=(1.0, 1.0, 1.0))
        outputs = stage.forward(image=image, expected=(1.0, 1.0, 1.0))
        assert outputs["report"] == {
            "check": "CheckVoxelSpacing",
            "passed": True,
            "value": [1.0, 1.0, 1.0],
        }

    def test_forward_report_includes_id_when_set(self, tmp_path: Path) -> None:
        stage = make_stage(CheckDimensions, tmp_path)
        image = SimpleNamespace(shape=(64, 64, 64))
        outputs = stage.forward(image=image, expected=(64, 64, 64), id="ctx.run_id")
        assert outputs["report"] == {
            "check": "CheckDimensions",
            "passed": True,
            "value": [64, 64, 64],
            "id": "ctx.run_id",
        }


class TestCheckMetric:
    def test_compares_scalar_with_op(self, tmp_path: Path) -> None:
        stage = make_stage(CheckMetric, tmp_path)
        outputs = stage.forward(metric=0.8, expected=0.7, op=">=")
        assert outputs["passed"] is True
        assert outputs["value"] == 0.8
        assert outputs["report"] == {
            "check": "CheckMetric",
            "passed": True,
            "value": 0.8,
        }

    def test_name_narrows_report_and_failure(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        stage = make_stage(CheckMetric, tmp_path, params={"name": "DSC"})
        with caplog.at_level(logging.INFO):
            outputs = stage.forward(
                metric=0.4,
                expected=0.7,
                op=">=",
                name="DSC",
                log=True,
            )
        assert outputs["passed"] is False
        assert outputs["report"]["check"] == "CheckMetric-DSC"
        assert "[QC CheckMetric-DSC]" in caplog.text
        with pytest.raises(
            RuntimeError,
            match=r"CheckMetric-DSC failed QC.*value=0\.4",
        ):
            stage.forward(
                metric=0.4,
                expected=0.7,
                op=">=",
                name="DSC",
                raise_on_fail=True,
            )

    def test_strips_name(self, tmp_path: Path) -> None:
        stage = make_stage(CheckMetric, tmp_path)
        assert stage.load_param("name", "  DSC  ") == "DSC"
        outputs = stage.forward(metric=1.0, expected=1.0, name="DSC")
        assert outputs["report"]["check"] == "CheckMetric-DSC"

    @pytest.mark.parametrize("name", ["", "   ", 1])
    def test_rejects_blank_or_non_string_name(
        self, tmp_path: Path, name: object
    ) -> None:
        stage = make_stage(CheckMetric, tmp_path)
        with pytest.raises(TypeError, match="name"):
            stage.load_param("name", name)

    def test_rejects_bool_and_non_finite_metric(self, tmp_path: Path) -> None:
        stage = make_stage(CheckMetric, tmp_path)
        with pytest.raises(TypeError, match="metric"):
            stage.load_param("metric", True)
        with pytest.raises(ValueError, match="finite"):
            stage.load_param("metric", float("nan"))
        with pytest.raises(TypeError, match="expected"):
            stage.load_param("expected", (1.0, 1.0))

    def test_default_op_is_greater_equal(self, tmp_path: Path) -> None:
        stage = make_stage(CheckMetric, tmp_path)
        assert stage.forward(metric=1.0, expected=1.0)["passed"] is True
        assert stage.forward(metric=1.5, expected=1.0)["passed"] is True
        assert stage.forward(metric=0.9, expected=1.0)["passed"] is False


class TestAggregateChecks:
    @pytest.mark.parametrize(
        ("flags", "rule", "passed"),
        [
            ([True, True], "AND", True),
            ([True, False], "AND", False),
            ([False, True], "or", True),
            ([False, False], "OR", False),
        ],
    )
    def test_combines_flags(
        self,
        tmp_path: Path,
        flags: list[bool],
        rule: str,
        passed: bool,
    ) -> None:
        stage = make_stage(
            AggregateChecks,
            tmp_path,
            params={"flags": flags, "rule": rule},
        )
        loaded_flags = stage.load_param("flags", flags)
        loaded_rule = stage.load_param("rule", rule)
        outputs = stage.forward(flags=loaded_flags, rule=loaded_rule)
        assert outputs["passed"] is passed
        assert outputs["value"] == tuple(flags)
        assert outputs["report"]["check"] == "AggregateChecks"
        assert outputs["report"]["value"] == flags
        assert outputs["report"]["passed"] is passed

    def test_unresolved_flag_refs_wait_until_load(self, tmp_path: Path) -> None:
        stage = AggregateChecks(
            params={
                "flags": [
                    "ctx.outputs.spacing.passed",
                    "ctx.outputs.dimensions.passed",
                ],
                "rule": "and",
            },
            save_outputs={},
        )
        assert stage.params["rule"] == "and"
        with pytest.raises(TypeError, match=r"flags\[0\]"):
            stage.load_param("flags", stage.params["flags"])
        assert stage.load_param("rule", "and") == "AND"

    def test_rejects_empty_flags(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            AggregateChecks(params={"flags": [], "rule": "AND"}, save_outputs={})

    def test_rejects_non_sequence_flags(self) -> None:
        with pytest.raises(TypeError, match="flags"):
            AggregateChecks(params={"flags": True, "rule": "AND"}, save_outputs={})

    def test_rejects_unknown_rule(self) -> None:
        with pytest.raises(ValueError, match="rule"):
            AggregateChecks(
                params={"flags": [True], "rule": "XOR"},
                save_outputs={},
            )

    def test_raises_on_fail(self, tmp_path: Path) -> None:
        stage = make_stage(AggregateChecks, tmp_path)
        with pytest.raises(
            RuntimeError,
            match=r"AggregateChecks failed QC.*value=\(True, False\)",
        ):
            stage.forward(flags=(True, False), rule="AND", raise_on_fail=True)
