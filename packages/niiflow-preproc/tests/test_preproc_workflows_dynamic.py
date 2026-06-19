"""Tests for :class:`DynamicPreprocessingWorkflow`."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import pytest

from niiflow.preproc.pipelines import discover_stage_classes
from niiflow.preproc.pipelines.pipeline_stages import PipelineStage
from niiflow.preproc.staging import StagedEntry, StagingErrorRecord
from niiflow.preproc.workflows import (
    DynamicPreprocessingWorkflow,
    PlanningWorkflow,
    ProcessingWorkflow,
    RunPlan,
)


class EchoStage(PipelineStage):
    REQUIRED_PARAMS = frozenset({"message"})

    def load_param(self, key: str, value: Any) -> Any:
        return value

    def forward(self, **params: Any) -> dict[str, Any]:
        return {"message": params["message"]}

    def save_output(self, key: str, value: Any, output_path: Path) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(str(value), encoding="utf-8")
        return output_path


@pytest.fixture
def registry_with_echo(
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, type[PipelineStage]]:
    registry = discover_stage_classes()
    registry = {**registry, "EchoStage": EchoStage}
    monkeypatch.setattr(
        "niiflow.preproc.pipelines.pipeline_factory.discover_stage_classes",
        lambda: registry,
    )
    return registry


def _workflow(
    *,
    staging_params: dict[str, Any] | None = None,
    pipeline_params: dict[str, Any] | None = None,
    explorer_params: dict[str, Any] | None = None,
    **kwargs: Any,
) -> DynamicPreprocessingWorkflow:
    return DynamicPreprocessingWorkflow(
        staging_params=staging_params
        or {"stager_name": "FileStager", "params": {"pointers": {}}},
        pipeline_params=pipeline_params or {"steps": []},
        explorer_params=explorer_params,
        **kwargs,
    )


def _proc_noop(entry: StagedEntry) -> None:
    return None


def _proc_touch_sentinel(entry: StagedEntry) -> None:
    out_dir = Path(entry.params["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{entry.active.name}.done").write_text("ok", encoding="utf-8")


def _proc_record_call(entry: StagedEntry) -> None:
    out_dir = Path(entry.params["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    log = out_dir / "calls.log"
    with log.open("a", encoding="utf-8") as fh:
        fh.write(f"{entry.active.as_posix()}\n")


def _proc_boom(entry: StagedEntry) -> None:
    raise RuntimeError(f"pipeline boom for {entry.active}")


def _proc_fail_on_sub02_t1w(entry: StagedEntry) -> None:
    if entry.active.name == "sub-02_T1w.nii.gz":
        raise RuntimeError("synthetic failure")
    _proc_touch_sentinel(entry)


def _proc_sleep_if_active(entry: StagedEntry) -> None:
    slow_actives = {
        Path(item).resolve() for item in entry.params.get("slow_actives", ())
    }
    if entry.active in slow_actives:
        time.sleep(float(entry.params.get("sleep_seconds", 3.0)))
    _proc_touch_sentinel(entry)


def _proc_log_info(entry: StagedEntry) -> None:
    logging.getLogger().info(entry.params.get("message", "WORKER_INFO"))


def _proc_log_burst(entry: StagedEntry) -> None:
    """Emit several INFO lines per entry, similar to a short pipeline run."""
    from niiflow.preproc.workflows.logging_utils import (
        reset_input_file_context,
        set_input_file_context,
    )

    token = set_input_file_context(entry.active.name)
    try:
        logger = logging.getLogger(
            "niiflow.preproc.pipelines.pipeline_stages.pipeline_stage"
        )
        for index in range(int(entry.params.get("log_lines", 6))):
            logger.info(f"[Stage test | step] line {index}")
    finally:
        reset_input_file_context(token)


@pytest.fixture
def dataset_root(tmp_path: Path) -> Path:
    root = tmp_path / "dataset"
    layout = [
        "sub-01/anat/sub-01_T1w.nii.gz",
        "sub-01/anat/sub-01_T2w.nii.gz",
        "sub-01/anat/sub-01_T1w_seg.nii.gz",
        "sub-02/anat/sub-02_T1w.nii.gz",
        "sub-02/anat/sub-02_T2w.nii",
        "sub-02/anat/sub-02_T1w_seg.nii.gz",
        "sub-03/anat/sub-03_T1w.nii.gz",
        "README.txt",
    ]
    for rel in layout:
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"")
    return root


@pytest.fixture
def logs_dir(tmp_path: Path) -> Path:
    return tmp_path / "logs"


class TestWorkflowPlanningContract:
    def test_dynamic_workflow_is_planning_workflow(self) -> None:
        assert issubclass(DynamicPreprocessingWorkflow, PlanningWorkflow)
        assert issubclass(PlanningWorkflow, ProcessingWorkflow)

    def test_processing_workflow_does_not_define_plan(self) -> None:
        assert "plan" not in ProcessingWorkflow.__dict__

    def test_planning_workflow_run_rejects_non_plan(self, tmp_path: Path) -> None:
        wf = _workflow()
        with pytest.raises(TypeError, match="`plan` must be a RunPlan"):
            PlanningWorkflow.run(wf, [])  # type: ignore[arg-type]


class TestDynamicWorkflowPlan:
    def test_plan_from_single_path(self, tmp_path: Path) -> None:
        file_path = tmp_path / "a.nii.gz"
        file_path.write_bytes(b"")
        wf = _workflow()

        plan = wf.plan(file_path)

        assert len(plan.entries) == 1
        assert plan.entries[0].active == file_path.resolve()

    def test_plan_from_path_list(self, tmp_path: Path) -> None:
        files = [tmp_path / f"img-{index}.nii.gz" for index in range(3)]
        for file_path in files:
            file_path.write_bytes(b"")
        wf = _workflow()

        plan = wf.plan(files)

        assert [entry.active for entry in plan.entries] == [
            path.resolve() for path in files
        ]

    def test_plan_from_explorer_config(self, dataset_root: Path) -> None:
        wf = _workflow(explorer_params={"pattern": "*.nii*"})
        plan = wf.plan(dataset_root)

        names = sorted(entry.active.name for entry in plan.entries)
        assert names == sorted(
            [
                "sub-01_T1w.nii.gz",
                "sub-01_T2w.nii.gz",
                "sub-01_T1w_seg.nii.gz",
                "sub-02_T1w.nii.gz",
                "sub-02_T2w.nii",
                "sub-02_T1w_seg.nii.gz",
                "sub-03_T1w.nii.gz",
            ]
        )

    def test_plan_with_filter_excludes_segmentations(self, dataset_root: Path) -> None:
        wf = _workflow(
            explorer_params={
                "pattern": "sub-*/anat/*T1w*.nii*",
                "filters": {
                    "name": "ExcludeFileRegex",
                    "kwargs": {"regex": r".*_seg\.nii.*"},
                },
            }
        )
        plan = wf.plan(dataset_root)

        names = sorted(entry.active.name for entry in plan.entries)
        assert names == sorted(
            ["sub-01_T1w.nii.gz", "sub-02_T1w.nii.gz", "sub-03_T1w.nii.gz"]
        )

    def test_plan_rejects_directory_without_explorer(self, dataset_root: Path) -> None:
        wf = _workflow()
        with pytest.raises(
            ValueError, match="Directory run inputs require a configured data explorer"
        ):
            wf.plan(dataset_root)

    def test_plan_stages_file_pointers(self, tmp_path: Path) -> None:
        active = tmp_path / "input.nii.gz"
        active.write_bytes(b"nii")
        output = tmp_path / "out" / "result.txt"
        wf = _workflow(
            staging_params={
                "stager_name": "FileStager",
                "params": {
                    "pointers": {
                        "output_path": "output",
                    }
                },
            },
            pipeline_params={
                "steps": [],
                "output_path": str(output),
            },
        )

        plan = wf.plan(active)

        assert plan.entries[0].params["output_path"] == output.resolve()

    def test_plan_rejects_missing_file(self, tmp_path: Path) -> None:
        wf = _workflow()
        with pytest.raises(FileNotFoundError):
            wf.plan(tmp_path / "ghost.nii.gz")

    def test_plan_without_staging_params(self, tmp_path: Path) -> None:
        file_path = tmp_path / "a.nii.gz"
        file_path.write_bytes(b"")
        wf = DynamicPreprocessingWorkflow(
            pipeline_params={"steps": [], "label": "no-staging"},
            num_workers=1,
        )

        plan = wf.plan(file_path)

        assert len(plan.entries) == 1
        assert plan.entries[0].params == {"steps": [], "label": "no-staging"}


class TestDynamicWorkflowRun:
    def test_run_processes_each_entry(
        self,
        dataset_root: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        sentinels = tmp_path / "sentinels"
        wf = _workflow(
            pipeline_params={"steps": [], "out_dir": str(sentinels)},
            explorer_params={"pattern": "*.nii*"},
        )
        monkeypatch.setattr(wf, "process_single", _proc_touch_sentinel)

        wf.run(dataset_root)

        produced = {path.name for path in sentinels.iterdir()}
        expected = {
            f"{entry.active.name}.done" for entry in wf.plan(dataset_root).entries
        }
        assert produced == expected

    def test_run_inputs_returns_generated_plan(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        file_path = tmp_path / "a.nii.gz"
        file_path.write_bytes(b"")
        wf = _workflow(num_workers=1)
        monkeypatch.setattr(wf, "process_single", _proc_noop)

        result = wf.run_inputs(file_path)

        assert isinstance(result, RunPlan)
        assert len(result.entries) == 1
        assert result.entries[0].active == file_path.resolve()
        assert result.entries[0].params == {"steps": []}

    def test_run_inputs_matches_plan(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        file_path = tmp_path / "a.nii.gz"
        file_path.write_bytes(b"")
        wf = _workflow(num_workers=1)
        monkeypatch.setattr(wf, "process_single", _proc_noop)

        plan = wf.plan(file_path)
        result = wf.run_inputs(file_path)

        assert result.entries == plan.entries

    def test_call_dunder_delegates_to_run(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        files = [tmp_path / f"img-{index}.nii.gz" for index in range(2)]
        for file_path in files:
            file_path.write_bytes(b"")
        sentinels = tmp_path / "sentinels"
        wf = _workflow(pipeline_params={"steps": [], "out_dir": str(sentinels)})
        monkeypatch.setattr(wf, "process_single", _proc_touch_sentinel)

        wf(files)

        assert {path.name for path in sentinels.iterdir()} == {
            f"{file_path.name}.done" for file_path in files
        }

    def test_run_with_empty_plan_is_noop(self, monkeypatch: pytest.MonkeyPatch) -> None:
        wf = _workflow()
        called: list[StagedEntry] = []

        def _record(entry: StagedEntry) -> None:
            called.append(entry)

        monkeypatch.setattr(wf, "process_single", _record)
        wf.run(RunPlan(entries=()))

        assert called == []

    def test_run_continues_after_entry_failure(
        self,
        dataset_root: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        sentinels = tmp_path / "sentinels"
        wf = _workflow(
            pipeline_params={"steps": [], "out_dir": str(sentinels)},
            explorer_params={"pattern": "*.nii*"},
            num_workers=1,
        )
        monkeypatch.setattr(wf, "process_single", _proc_fail_on_sub02_t1w)
        wf.run(dataset_root)

        produced = {path.name for path in sentinels.iterdir()}
        assert "sub-02_T1w.nii.gz.done" not in produced
        assert len(produced) == 6

    @pytest.mark.parametrize("num_workers", [2, 4])
    def test_multi_worker_processes_each_entry_once(
        self,
        dataset_root: Path,
        tmp_path: Path,
        num_workers: int,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        sentinels = tmp_path / "sentinels"
        wf = _workflow(
            pipeline_params={"steps": [], "out_dir": str(sentinels)},
            explorer_params={"pattern": "*.nii*"},
            num_workers=num_workers,
        )
        monkeypatch.setattr(wf, "process_single", _proc_touch_sentinel)

        wf.run(dataset_root)

        produced = {path.name for path in sentinels.iterdir()}
        expected = {
            f"{entry.active.name}.done" for entry in wf.plan(dataset_root).entries
        }
        assert produced == expected

    def test_timeout_records_overdue_entry_and_continues(
        self,
        tmp_path: Path,
        logs_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        fast = tmp_path / "fast.nii.gz"
        slow = tmp_path / "slow.nii.gz"
        fast.write_bytes(b"")
        slow.write_bytes(b"")
        sentinels = tmp_path / "sentinels"
        wf = _workflow(
            pipeline_params={
                "steps": [],
                "out_dir": str(sentinels),
                "slow_actives": (str(slow),),
                "sleep_seconds": 2.0,
            },
            num_workers=1,
            logs_root=logs_dir,
            timeout=0.5,
        )
        monkeypatch.setattr(wf, "process_single", _proc_sleep_if_active)

        wf.run([fast, slow])

        status = (logs_dir / "status.log").read_text(encoding="utf-8")
        assert f"{fast.resolve()} | SUCCESS" in status
        assert f"{slow.resolve()} | TIMEOUT" in status
        assert "fast.nii.gz.done" in {path.name for path in sentinels.iterdir()}

    def test_run_plan_skips_entries_with_staging_errors(
        self,
        tmp_path: Path,
        logs_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        ok = tmp_path / "ok.nii.gz"
        bad = tmp_path / "bad.nii.gz"
        ok.write_bytes(b"")
        bad.write_bytes(b"")
        sentinels = tmp_path / "sentinels"
        plan = RunPlan(
            entries=(
                StagedEntry(
                    active=ok.resolve(),
                    params={"steps": [], "out_dir": str(sentinels)},
                ),
                StagedEntry(
                    active=bad.resolve(),
                    params={"steps": [], "out_dir": str(sentinels)},
                    errors=(
                        StagingErrorRecord(
                            active=bad.resolve(),
                            message="mask not found",
                            stage="FileStager",
                        ),
                    ),
                ),
            )
        )
        wf = _workflow(
            pipeline_params={"steps": [], "out_dir": str(sentinels)},
            num_workers=1,
            logs_root=logs_dir,
        )
        monkeypatch.setattr(
            DynamicPreprocessingWorkflow,
            "process_single",
            staticmethod(_proc_touch_sentinel),
        )

        wf.run_plan(plan)

        assert (sentinels / "ok.nii.gz.done").exists()
        assert not (sentinels / "bad.nii.gz.done").exists()
        status = (logs_dir / "status.log").read_text(encoding="utf-8")
        assert f"{bad.resolve()} | STAGING_FAILURE | mask not found" in status

    def test_logs_written_under_logs_root(
        self,
        tmp_path: Path,
        logs_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        file_path = tmp_path / "a.nii.gz"
        file_path.write_bytes(b"")
        wf = _workflow(logs_root=logs_dir, num_workers=1)
        monkeypatch.setattr(wf, "process_single", _proc_noop)

        wf.run(file_path)

        assert (logs_dir / "main.log").exists()
        assert (logs_dir / "status.log").exists()
        assert f"{file_path.resolve()} | SUCCESS" in (
            logs_dir / "status.log"
        ).read_text(encoding="utf-8")

    def test_parallel_worker_logs_capture_burst(
        self,
        dataset_root: Path,
        logs_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        wf = _workflow(
            pipeline_params={"steps": [], "log_lines": 6},
            explorer_params={"pattern": "*.nii*"},
            num_workers=4,
            logs_root=logs_dir,
        )
        monkeypatch.setattr(wf, "process_single", _proc_log_burst)

        wf.run(dataset_root)

        status_lines = [
            line
            for line in (logs_dir / "status.log")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]
        worker_lines = [
            line
            for line in (logs_dir / "workers.log")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]
        expected_worker_lines = len(status_lines) * 6
        assert len(status_lines) == 7
        assert len(worker_lines) >= int(expected_worker_lines * 0.95), (
            f"expected at least {int(expected_worker_lines * 0.95)} worker log lines, "
            f"got {len(worker_lines)}"
        )


class TestDynamicWorkflowPipeline:
    def test_process_single_runs_echo_stage(
        self,
        tmp_path: Path,
        registry_with_echo: dict[str, type[PipelineStage]],
    ) -> None:
        active = tmp_path / "input.nii.gz"
        active.write_bytes(b"nii")
        output = tmp_path / "echo.txt"
        wf = _workflow(
            staging_params={
                "stager_name": "FileStager",
                "params": {
                    "pointers": {
                        "steps.echo.save_options.message": "output",
                    }
                },
            },
            pipeline_params={
                "steps": {
                    "echo": {
                        "name": "EchoStage",
                        "params": {"message": "hello"},
                        "save_options": {"message": str(output)},
                    }
                }
            },
        )
        plan = wf.plan(active)

        wf.process_single(plan.entries[0])

        assert output.read_text(encoding="utf-8") == "hello"
