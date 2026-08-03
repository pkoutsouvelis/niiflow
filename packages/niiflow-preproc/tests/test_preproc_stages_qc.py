"""Tests for the QC stages in
:mod:`niiflow.preproc.pipelines.pipeline_stages.qc`.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from niiflow.preproc.pipelines.pipeline_stages import CheckDimensions, CheckVoxelSpacing
from stage_helpers import StageFactory, make_stage


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
