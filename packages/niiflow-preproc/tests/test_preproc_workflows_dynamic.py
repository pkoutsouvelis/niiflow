"""Tests for :class:`DynamicPreprocessingWorkflow`."""

from __future__ import annotations

import importlib
import logging
import time
from pathlib import Path
from typing import Any

import pytest

from niiflow.preproc.pipelines.pipeline_stages import PipelineStage
from niiflow.preproc.staging import StagedEntry, StagingErrorRecord
from niiflow.preproc.workflows import (
    DynamicPreprocessingWorkflow,
    PlannableWorkflow,
    ProcessingWorkflow,
    RunPlan,
    dynamic_workflow,
)

_dynamic_pipeline_mod = importlib.import_module(
    "niiflow.preproc.pipelines.dynamic_pipeline"
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
    registry = {
        **_dynamic_pipeline_mod._discover_stage_classes(),
        "EchoStage": EchoStage,
    }
    monkeypatch.setattr(
        _dynamic_pipeline_mod,
        "_discover_stage_classes",
        lambda: registry,
    )
    return registry


def _workflow(
    *,
    staging_params: dict[str, Any] | None = None,
    pipeline_params: dict[str, Any] | None = None,
    **kwargs: Any,
) -> DynamicPreprocessingWorkflow:
    return DynamicPreprocessingWorkflow(
        staging_params=staging_params
        or {"stager_name": "FileStager", "params": {"pointers": {}}},
        pipeline_params=pipeline_params or {"steps": []},
        **kwargs,
    )


def _search(
    roots: Path | str | list[Path | str],
    *,
    explorer_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "mode": "search",
        "roots": roots,
        "explorer_params": explorer_params or {"patterns": "*.nii*"},
    }


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
    def test_dynamic_workflow_is_plannable_workflow(self) -> None:
        assert issubclass(DynamicPreprocessingWorkflow, PlannableWorkflow)
        assert issubclass(PlannableWorkflow, ProcessingWorkflow)

    def test_processing_workflow_does_not_define_plan(self) -> None:
        assert "plan" not in ProcessingWorkflow.__dict__

    def test_run_rejects_non_staged_entry(self, tmp_path: Path) -> None:
        wf = _workflow()
        with pytest.raises(
            TypeError, match="`entries` must contain only `StagedEntry`"
        ):
            wf.run([tmp_path / "a.nii.gz"])  # type: ignore[arg-type]


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
        wf = _workflow()
        plan = wf.plan(_search(dataset_root))

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
        wf = _workflow()
        plan = wf.plan(
            _search(
                dataset_root,
                explorer_params={
                    "patterns": "sub-*/anat/*T1w*.nii*",
                    "filters": {
                        "name": "ExcludeFileRegex",
                        "kwargs": {"regex": r".*_seg\.nii.*"},
                    },
                },
            )
        )

        names = sorted(entry.active.name for entry in plan.entries)
        assert names == sorted(
            ["sub-01_T1w.nii.gz", "sub-02_T1w.nii.gz", "sub-03_T1w.nii.gz"]
        )

    def test_plan_rejects_directory_as_explicit_path(self, dataset_root: Path) -> None:
        wf = _workflow()
        with pytest.raises(ValueError, match="Explicit run input must be a file"):
            wf.plan(dataset_root)

    def test_plan_from_file_list(self, tmp_path: Path) -> None:
        files = [tmp_path / f"img-{index}.nii.gz" for index in range(3)]
        for file_path in files:
            file_path.write_bytes(b"")
        list_path = tmp_path / "actives.txt"
        list_path.write_text(
            "\n".join(str(path.resolve()) for path in files),
            encoding="utf-8",
        )
        wf = _workflow()

        plan = wf.plan({"mode": "from_file", "path": list_path})

        assert [entry.active for entry in plan.entries] == [
            path.resolve() for path in files
        ]

    def test_plan_save_active_files(self, tmp_path: Path) -> None:
        file_path = tmp_path / "a.nii.gz"
        file_path.write_bytes(b"")
        out = tmp_path / "found.txt"
        wf = _workflow()

        found = wf.discover_active_files(file_path, save_to=out)

        assert found == [file_path.resolve()]
        assert out.read_text(encoding="utf-8") == f"{file_path.resolve()}\n"

    def test_plan_save_plan(self, tmp_path: Path) -> None:
        file_path = tmp_path / "a.nii.gz"
        file_path.write_bytes(b"")
        plan_path = tmp_path / "job.duckdb"
        files_path = tmp_path / "found.txt"
        wf = _workflow()

        plan = wf.plan(file_path, save_filepaths_to=files_path, save_plan_to=plan_path)

        assert plan_path.exists()
        assert RunPlan.load(plan_path).entries == plan.entries
        assert files_path.read_text(encoding="utf-8") == f"{file_path.resolve()}\n"

    def test_plan_with_none_entry_params(self, tmp_path: Path) -> None:
        file_path = tmp_path / "a.nii.gz"
        file_path.write_bytes(b"")
        wf = DynamicPreprocessingWorkflow(pipeline_params=None, num_workers=1)

        plan = wf.plan(file_path)

        assert len(plan.entries) == 1
        assert plan.entries[0].params == {}

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
        )
        monkeypatch.setattr(wf, "process_single", _proc_touch_sentinel)

        wf.run_plan(wf.plan(_search(dataset_root)))

        produced = {path.name for path in sentinels.iterdir()}
        expected = {
            f"{entry.active.name}.done"
            for entry in wf.plan(_search(dataset_root)).entries
        }
        assert produced == expected

    def test_plan_then_run_plan_executes_entries(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        file_path = tmp_path / "a.nii.gz"
        file_path.write_bytes(b"")
        wf = _workflow(num_workers=1)
        called: list[StagedEntry] = []

        def _record(entry: StagedEntry) -> None:
            called.append(entry)

        monkeypatch.setattr(wf, "process_single", _record)

        wf.run_plan(wf.plan(file_path))

        assert len(called) == 1
        assert called[0].active == file_path.resolve()
        assert called[0].params == {"steps": []}

    def test_plan_save_options_apply_before_run_plan(self, tmp_path: Path) -> None:
        file_path = tmp_path / "a.nii.gz"
        file_path.write_bytes(b"")
        plan_path = tmp_path / "job.duckdb"
        files_path = tmp_path / "found.txt"
        wf = _workflow(num_workers=1)

        wf.run_plan(
            wf.plan(
                file_path,
                save_filepaths_to=files_path,
                save_plan_to=plan_path,
            )
        )

        assert plan_path.exists()
        assert len(RunPlan.load(plan_path).entries) == 1
        assert files_path.read_text(encoding="utf-8") == f"{file_path.resolve()}\n"

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

        plan = wf.plan(files)
        wf(plan.entries)

        assert {path.name for path in sentinels.iterdir()} == {
            f"{file_path.name}.done" for file_path in files
        }

    def test_run_plan_with_empty_plan_is_noop(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        wf = _workflow()
        called: list[StagedEntry] = []

        def _record(entry: StagedEntry) -> None:
            called.append(entry)

        monkeypatch.setattr(wf, "process_single", _record)
        wf.run_plan(RunPlan(entries=()))

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
            num_workers=1,
        )
        monkeypatch.setattr(wf, "process_single", _proc_fail_on_sub02_t1w)
        wf.run_plan(wf.plan(_search(dataset_root)))

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
            num_workers=num_workers,
        )
        monkeypatch.setattr(wf, "process_single", _proc_touch_sentinel)

        wf.run_plan(wf.plan(_search(dataset_root)))

        produced = {path.name for path in sentinels.iterdir()}
        expected = {
            f"{entry.active.name}.done"
            for entry in wf.plan(_search(dataset_root)).entries
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

        wf.run_plan(wf.plan([fast, slow]))

        status = (logs_dir / "status.log").read_text(encoding="utf-8")
        assert f"{fast.resolve()} | SUCCESS" in status
        assert f"{slow.resolve()} | TIMEOUT" in status
        assert "fast.nii.gz.done" in {path.name for path in sentinels.iterdir()}

    @pytest.mark.parametrize("num_workers", [1, 2])
    def test_timeout_measures_worker_processing_not_queue_wait(
        self,
        tmp_path: Path,
        logs_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
        num_workers: int,
    ) -> None:
        """Fast entries must not TIMEOUT while waiting for a worker."""
        first = tmp_path / "first.nii.gz"
        second = tmp_path / "second.nii.gz"
        first.write_bytes(b"")
        second.write_bytes(b"")
        sentinels = tmp_path / "sentinels"
        wf = _workflow(
            pipeline_params={
                "steps": [],
                "out_dir": str(sentinels),
                "slow_actives": (str(first),),
                "sleep_seconds": 1.5,
            },
            num_workers=num_workers,
            logs_root=logs_dir,
            timeout=0.5,
        )
        monkeypatch.setattr(wf, "process_single", _proc_sleep_if_active)

        wf.run_plan(wf.plan([first, second]))

        status = (logs_dir / "status.log").read_text(encoding="utf-8")
        assert f"{first.resolve()} | TIMEOUT" in status
        assert f"{second.resolve()} | SUCCESS" in status
        assert {path.name for path in sentinels.iterdir()} == {
            "first.nii.gz.done",
            "second.nii.gz.done",
        }

    def test_timeout_success_when_processing_finishes_within_limit(
        self,
        tmp_path: Path,
        logs_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Completion after the limit must be TIMEOUT; within the limit, SUCCESS."""
        entry = tmp_path / "entry.nii.gz"
        entry.write_bytes(b"")
        sentinels = tmp_path / "sentinels"

        def _proc_sleep(entry: StagedEntry) -> None:
            time.sleep(float(entry.params["sleep_seconds"]))
            _proc_touch_sentinel(entry)

        wf = _workflow(
            pipeline_params={
                "steps": [],
                "out_dir": str(sentinels),
                "sleep_seconds": 0.2,
            },
            num_workers=1,
            logs_root=logs_dir,
            timeout=0.5,
        )
        monkeypatch.setattr(wf, "process_single", _proc_sleep)

        wf.run_plan(wf.plan(entry))

        status = (logs_dir / "status.log").read_text(encoding="utf-8")
        assert f"{entry.resolve()} | SUCCESS" in status
        assert "TIMEOUT" not in status

    def test_timeout_when_processing_exceeds_limit(
        self,
        tmp_path: Path,
        logs_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        entry = tmp_path / "entry.nii.gz"
        entry.write_bytes(b"")
        sentinels = tmp_path / "sentinels"

        def _proc_sleep(entry: StagedEntry) -> None:
            time.sleep(float(entry.params["sleep_seconds"]))
            _proc_touch_sentinel(entry)

        wf = _workflow(
            pipeline_params={
                "steps": [],
                "out_dir": str(sentinels),
                "sleep_seconds": 0.8,
            },
            num_workers=1,
            logs_root=logs_dir,
            timeout=0.5,
        )
        monkeypatch.setattr(wf, "process_single", _proc_sleep)

        wf.run_plan(wf.plan(entry))

        status = (logs_dir / "status.log").read_text(encoding="utf-8")
        assert f"{entry.resolve()} | TIMEOUT" in status
        assert "SUCCESS" not in status

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

        wf.run_plan(wf.plan(file_path))

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
            num_workers=4,
            logs_root=logs_dir,
        )
        monkeypatch.setattr(wf, "process_single", _proc_log_burst)

        wf.run_plan(wf.plan(_search(dataset_root)))

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


def _driver_settings(
    *,
    pipeline_params: dict[str, Any] | None = None,
    staging_params: dict[str, Any] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    return {
        "staging_params": staging_params
        or {"stager_name": "FileStager", "params": {"pointers": {}}},
        "pipeline_params": pipeline_params or {"steps": []},
        "num_workers": 1,
        **kwargs,
    }


class TestDynamicWorkflow:
    def test_plan_and_execute(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        file_path = tmp_path / "a.nii.gz"
        file_path.write_bytes(b"")
        called: list[StagedEntry] = []

        def _record(entry: StagedEntry) -> None:
            called.append(entry)

        monkeypatch.setattr(
            DynamicPreprocessingWorkflow,
            "process_single",
            staticmethod(_record),
        )

        plan = dynamic_workflow(
            settings=_driver_settings(),
            inputs=file_path,
        )

        assert len(plan.entries) == 1
        assert plan.entries[0].active == file_path.resolve()
        assert called == [plan.entries[0]]

    def test_plan_only_saves_without_executing(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        file_path = tmp_path / "a.nii.gz"
        file_path.write_bytes(b"")
        plan_path = tmp_path / "job.duckdb"
        files_path = tmp_path / "found.txt"
        called: list[StagedEntry] = []

        monkeypatch.setattr(
            DynamicPreprocessingWorkflow,
            "process_single",
            staticmethod(lambda entry: called.append(entry)),
        )

        plan = dynamic_workflow(
            settings=_driver_settings(),
            inputs=file_path,
            save_filepaths_to=files_path,
            save_plan_to=plan_path,
            plan_only=True,
        )

        assert plan_path.exists()
        assert RunPlan.load(plan_path).entries == plan.entries
        assert files_path.read_text(encoding="utf-8") == f"{file_path.resolve()}\n"
        assert called == []

    def test_dry_run_prints_and_skips_save_and_execute(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        file_path = tmp_path / "a.nii.gz"
        file_path.write_bytes(b"")
        plan_path = tmp_path / "job.duckdb"
        called: list[StagedEntry] = []

        monkeypatch.setattr(
            DynamicPreprocessingWorkflow,
            "process_single",
            staticmethod(lambda entry: called.append(entry)),
        )

        plan = dynamic_workflow(
            settings=_driver_settings(),
            inputs=file_path,
            save_plan_to=plan_path,
            dry_run=True,
        )

        captured = capsys.readouterr()
        assert "RunPlan:" in captured.out
        assert str(file_path.resolve()) in captured.out
        assert not plan_path.exists()
        assert called == []
        assert len(plan.entries) == 1

    def test_from_plan_executes_saved_plan(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        file_path = tmp_path / "a.nii.gz"
        file_path.write_bytes(b"")
        plan_path = tmp_path / "job.duckdb"
        plan = _workflow().plan(file_path, save_plan_to=plan_path)
        called: list[StagedEntry] = []

        monkeypatch.setattr(
            DynamicPreprocessingWorkflow,
            "process_single",
            staticmethod(lambda entry: called.append(entry)),
        )

        loaded = dynamic_workflow(
            settings=_driver_settings(),
            from_plan=plan_path,
        )

        assert loaded.entries == plan.entries
        assert called == [plan.entries[0]]

    def test_from_plan_ignores_inputs_and_pipeline_params(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        file_path = tmp_path / "a.nii.gz"
        file_path.write_bytes(b"")
        plan_path = tmp_path / "job.duckdb"
        plan = _workflow().plan(file_path, save_plan_to=plan_path)
        called: list[StagedEntry] = []

        monkeypatch.setattr(
            DynamicPreprocessingWorkflow,
            "process_single",
            staticmethod(lambda entry: called.append(entry)),
        )

        loaded = dynamic_workflow(
            settings=_driver_settings(),
            inputs=tmp_path / "ignored.nii.gz",
            from_plan=plan_path,
        )

        err = capsys.readouterr().err
        assert loaded.entries == plan.entries
        assert called == [plan.entries[0]]
        assert "Ignoring `inputs`" in err
        assert "Ignoring `pipeline_params`" in err

    def test_requires_inputs_without_from_plan(self) -> None:
        with pytest.raises(ValueError, match="`inputs` is required"):
            dynamic_workflow(settings=_driver_settings())

    def test_requires_settings_without_from_plan(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="`settings` is required"):
            dynamic_workflow(inputs=tmp_path / "a.nii.gz")

    def test_requires_pipeline_params_without_from_plan(self, tmp_path: Path) -> None:
        settings = _driver_settings()
        del settings["pipeline_params"]
        with pytest.raises(ValueError, match="non-None `pipeline_params`"):
            dynamic_workflow(settings=settings, inputs=tmp_path / "a.nii.gz")

        settings = _driver_settings()
        settings["pipeline_params"] = None
        with pytest.raises(ValueError, match="non-None `pipeline_params`"):
            dynamic_workflow(settings=settings, inputs=tmp_path / "a.nii.gz")

    def test_rejects_unknown_settings(self) -> None:
        with pytest.raises(
            TypeError,
            match="Failed to instantiate workflow 'DynamicPreprocessingWorkflow'",
        ):
            dynamic_workflow(
                settings={**_driver_settings(), "not_a_real_kwarg": True},
                inputs="/tmp/a.nii.gz",
            )
