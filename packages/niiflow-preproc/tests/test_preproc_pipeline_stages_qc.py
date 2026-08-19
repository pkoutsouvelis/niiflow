"""Tests for the QC stages in
:mod:`niiflow.preproc.pipelines.pipeline_stages.qc`.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from niiflow.preproc.pipelines.pipeline_stages import (
    CheckDimensions,
    CheckImageSimilarity,
    CheckVoxelSpacing,
)
from test_preproc_pipeline_stages_shipped_shared import StageFactory, make_stage


class TestQCStagePersistence:
    @pytest.mark.parametrize(
        "stage_cls",
        [CheckVoxelSpacing, CheckDimensions, CheckImageSimilarity],
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

    def test_save_output_writes_registration_value_json(self, tmp_path: Path) -> None:
        value_path = tmp_path / "value.json"
        stage = make_stage(CheckImageSimilarity, tmp_path)
        value = {"correlation": 0.9, "mattes_mutual_information": 1.2}
        record = stage.save_output("value", value, value_path)
        assert record == value_path.resolve()
        assert json.loads(value_path.read_text(encoding="utf-8")) == value

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
        [CheckVoxelSpacing, CheckDimensions, CheckImageSimilarity],
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
        [CheckVoxelSpacing, CheckDimensions, CheckImageSimilarity],
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


class TestCheckImageSimilarity:
    def _stage(self, tmp_path: Path, **params: object) -> CheckImageSimilarity:
        base = {
            "image": "image.nii.gz",
            "target": "target.nii.gz",
            "correlation": 0.5,
        }
        base.update(params)
        return make_stage(CheckImageSimilarity, tmp_path, params=base)  # type: ignore[return-value]

    def test_construction_rejects_all_none_cutoffs(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="at least one similarity cutoff"):
            make_stage(
                CheckImageSimilarity,
                tmp_path,
                params={
                    "image": "m.nii.gz",
                    "target": "t.nii.gz",
                    "correlation": None,
                    "mattes_mutual_information": None,
                    "neighborhood_correlation": None,
                },
            )

    def test_construction_rejects_missing_cutoffs(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="at least one similarity cutoff"):
            CheckImageSimilarity(
                params={
                    "image": str(tmp_path / "m.nii.gz"),
                    "target": str(tmp_path / "t.nii.gz"),
                }
            )

    def test_mask_optional(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        stage = self._stage(tmp_path)
        captured: dict[str, object] = {}

        def fake_metrics(image, target, mask=None, metrics=None):
            captured["mask"] = mask
            captured["metrics"] = metrics
            return {"correlation": 1.0}

        monkeypatch.setattr(
            "niiflow.preproc.pipelines.pipeline_stages.qc.ants_similarity_metrics",
            fake_metrics,
        )
        out = stage.forward(image=object(), target=object(), correlation=0.5)
        assert captured["mask"] is None
        assert out["passed"] is True

    def test_construction_rejects_invalid_combine(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="combine"):
            self._stage(tmp_path, combine="majority")

    def test_none_cutoff_skips_metric(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        stage = self._stage(
            tmp_path,
            correlation=0.5,
            mattes_mutual_information=None,
            neighborhood_correlation=0.1,
        )
        calls: list[tuple[str, ...]] = []

        def fake_metrics(image, target, mask=None, metrics=None):
            calls.append(tuple(metrics) if metrics is not None else ())
            return {name: 1.0 for name in metrics or ()}

        monkeypatch.setattr(
            "niiflow.preproc.pipelines.pipeline_stages.qc.ants_similarity_metrics",
            fake_metrics,
        )
        out = stage.forward(
            image=object(),
            target=object(),
            correlation=0.5,
            mattes_mutual_information=None,
            neighborhood_correlation=0.1,
        )
        assert calls == [("correlation", "neighborhood_correlation")]
        assert set(out["value"]) == {"correlation", "neighborhood_correlation"}
        assert "mattes_mutual_information" not in out["report"]["cutoffs"]

    def test_combine_all_and_any(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "niiflow.preproc.pipelines.pipeline_stages.qc.ants_similarity_metrics",
            lambda *a, **k: {"correlation": 0.9, "neighborhood_correlation": 0.2},
        )
        images = {"image": object(), "target": object()}
        all_stage = self._stage(
            tmp_path,
            correlation=0.5,
            neighborhood_correlation=0.5,
            combine="all",
        )
        any_stage = self._stage(
            tmp_path,
            correlation=0.5,
            neighborhood_correlation=0.5,
            combine="any",
        )
        all_out = all_stage.forward(
            **images,
            correlation=0.5,
            neighborhood_correlation=0.5,
            combine="all",
        )
        any_out = any_stage.forward(
            **images,
            correlation=0.5,
            neighborhood_correlation=0.5,
            combine="any",
        )
        assert all_out["passed"] is False
        assert any_out["passed"] is True
        assert all_out["report"]["metric_passed"] == {
            "correlation": True,
            "neighborhood_correlation": False,
        }

    def test_pass_fail_report(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "niiflow.preproc.pipelines.pipeline_stages.qc.ants_similarity_metrics",
            lambda *a, **k: {"correlation": 0.4},
        )
        stage = self._stage(tmp_path, correlation=0.5, id="run-1")
        out = stage.forward(
            image=object(),
            target=object(),
            correlation=0.5,
            id="run-1",
        )
        assert out["passed"] is False
        assert out["value"] == {"correlation": 0.4}
        assert out["report"] == {
            "check": "CheckImageSimilarity",
            "passed": False,
            "value": {"correlation": 0.4},
            "cutoffs": {"correlation": 0.5},
            "combine": "all",
            "metric_passed": {"correlation": False},
            "id": "run-1",
        }

    def test_forward_propagates_physical_space_mismatch(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "niiflow.preproc.pipelines.pipeline_stages.qc.ants_similarity_metrics",
            lambda *a, **k: (_ for _ in ()).throw(
                ValueError("not in the same physical space")
            ),
        )
        stage = self._stage(tmp_path)
        with pytest.raises(ValueError, match="same physical space"):
            stage.forward(
                image=object(),
                target=object(),
                correlation=0.5,
            )
