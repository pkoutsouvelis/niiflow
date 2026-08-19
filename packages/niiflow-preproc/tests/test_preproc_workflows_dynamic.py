"""Tests for :class:`DynamicProcessingWorkflow`."""

from __future__ import annotations

import importlib
import logging
from pathlib import Path
from typing import Any

import pytest

from niiflow.preproc.pipelines.pipeline_stages import (
    PipelineStage,
    discover_pipeline_stage_classes,
)
from niiflow.preproc.staging import StagedEntry
from niiflow.preproc.workflows import (
    DynamicProcessingWorkflow,
    PlannableWorkflow,
    RunPlan,
    dynamic_workflow,
)
from niiflow.preproc.workflows.mixins import SupportsInputDiscovery, SupportsStaging

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
        **discover_pipeline_stage_classes(),
        "EchoStage": EchoStage,
    }
    monkeypatch.setattr(
        _dynamic_pipeline_mod,
        "discover_pipeline_stage_classes",
        lambda: registry,
    )
    return registry


def _workflow(
    *,
    staging_params: dict[str, Any] | None = None,
    pipeline_params: dict[str, Any] | list[dict[str, Any]] | None = None,
    **kwargs: Any,
) -> DynamicProcessingWorkflow:
    return DynamicProcessingWorkflow(
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


class TestWorkflowPlannableContract:
    def test_dynamic_workflow_is_plannable_workflow(self) -> None:
        assert issubclass(DynamicProcessingWorkflow, PlannableWorkflow)

    def test_supports_discovery_and_staging(self) -> None:
        assert issubclass(DynamicProcessingWorkflow, SupportsInputDiscovery)
        assert issubclass(DynamicProcessingWorkflow, SupportsStaging)


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
        plan = wf.plan(_search(dataset_root))  # type: ignore[arg-type]

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
            )  # type: ignore[arg-type]
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

        found = wf.collect_active_files(file_path, save_to=out)

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
        wf = DynamicProcessingWorkflow(pipeline_params=None, num_workers=1)

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

    def test_plan_rejects_missing_explicit_file(self, tmp_path: Path) -> None:
        wf = _workflow()
        with pytest.raises(FileNotFoundError):
            wf.plan(tmp_path / "ghost.nii.gz")

    def test_plan_without_staging_params(self, tmp_path: Path) -> None:
        file_path = tmp_path / "a.nii.gz"
        file_path.write_bytes(b"")
        wf = DynamicProcessingWorkflow(
            pipeline_params={"steps": [], "label": "no-staging"},
            num_workers=1,
        )

        plan = wf.plan(file_path)

        assert len(plan.entries) == 1
        assert plan.entries[0].params == {"steps": [], "label": "no-staging"}

    def test_plan_expands_active_refs_without_user_stagers(
        self, tmp_path: Path
    ) -> None:
        file_path = tmp_path / "sub-01_T1w.nii.gz"
        file_path.write_bytes(b"")
        wf = DynamicProcessingWorkflow(
            pipeline_params={"steps": [], "subject": "{active.stem}"},
            num_workers=1,
        )

        plan = wf.plan(file_path)

        assert plan.entries[0].params["subject"] == file_path.stem

    def test_plan_aligns_sequence_pipeline_params_with_explicit_files(
        self, tmp_path: Path
    ) -> None:
        files = [tmp_path / f"img-{index}.nii.gz" for index in range(2)]
        for file_path in files:
            file_path.write_bytes(b"")
        wf = DynamicProcessingWorkflow(
            pipeline_params=[
                {"steps": [], "label": "first"},
                {"steps": [], "label": "second"},
            ],
            num_workers=1,
        )

        plan = wf.plan(files)

        assert [entry.params["label"] for entry in plan.entries] == ["first", "second"]
        assert [entry.active for entry in plan.entries] == [
            path.resolve() for path in files
        ]

    def test_plan_aligns_sequence_pipeline_params_with_from_file(
        self, tmp_path: Path
    ) -> None:
        files = [tmp_path / f"img-{index}.nii.gz" for index in range(2)]
        for file_path in files:
            file_path.write_bytes(b"")
        listing = tmp_path / "actives.txt"
        listing.write_text("\n".join(str(path) for path in files), encoding="utf-8")
        wf = DynamicProcessingWorkflow(
            pipeline_params=[
                {"steps": [], "label": "first"},
                {"steps": [], "label": "second"},
            ],
            num_workers=1,
        )

        plan = wf.plan({"mode": "from_file", "path": listing})

        assert [entry.params["label"] for entry in plan.entries] == ["first", "second"]

    def test_plan_sequence_pipeline_params_rejects_search(
        self, dataset_root: Path
    ) -> None:
        wf = DynamicProcessingWorkflow(
            pipeline_params=[{"steps": []}, {"steps": []}],
            num_workers=1,
        )

        with pytest.raises(ValueError, match="search inputs are not allowed"):
            wf.plan(_search(dataset_root))  # type: ignore[arg-type]

    def test_plan_sequence_pipeline_params_must_match_active_count(
        self, tmp_path: Path
    ) -> None:
        files = [tmp_path / f"img-{index}.nii.gz" for index in range(2)]
        for file_path in files:
            file_path.write_bytes(b"")
        wf = DynamicProcessingWorkflow(
            pipeline_params=[{"steps": [], "label": "only-one"}],
            num_workers=1,
        )

        with pytest.raises(ValueError, match="same length"):
            wf.plan(files)

    def test_rejects_invalid_pipeline_params(self) -> None:
        with pytest.raises(TypeError, match="`pipeline_params` must be"):
            DynamicProcessingWorkflow(pipeline_params="steps")  # type: ignore[arg-type]

    def test_rejects_non_mapping_pipeline_params_item(self) -> None:
        with pytest.raises(TypeError, match="Each `pipeline_params` item"):
            DynamicProcessingWorkflow(pipeline_params=["steps"])  # type: ignore[arg-type]


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

        wf.run_plan(wf.plan(_search(dataset_root)))  # type: ignore[arg-type]

        produced = {path.name for path in sentinels.iterdir()}
        expected = {
            f"{entry.active.name}.done"
            for entry in wf.plan(_search(dataset_root)).entries  # type: ignore[arg-type]
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

    def test_plan_save_outputs_apply_before_run_plan(self, tmp_path: Path) -> None:
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

        wf.run_plan(wf.plan(_search(dataset_root)))  # type: ignore[arg-type]

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
                        "steps.echo.save_outputs.message": "output",
                    }
                },
            },
            pipeline_params={
                "steps": {
                    "echo": {
                        "name": "EchoStage",
                        "params": {"message": "hello"},
                        "save_outputs": {"message": str(output)},
                    }
                }
            },
        )
        plan = wf.plan(active)

        wf.process_single(plan.entries[0])

        assert output.read_text(encoding="utf-8") == "hello"


def _driver_settings(
    *,
    pipeline_params: dict[str, Any] | list[dict[str, Any]] | None = None,
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
            DynamicProcessingWorkflow,
            "process_single",
            staticmethod(_record),
        )

        dynamic_workflow(
            settings=_driver_settings(),
            inputs=file_path,
        )

        assert len(called) == 1
        assert called[0].active == file_path.resolve()

    def test_plan_and_execute_with_per_active_pipeline_params(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        files = [tmp_path / f"img-{index}.nii.gz" for index in range(2)]
        for file_path in files:
            file_path.write_bytes(b"")
        called: list[StagedEntry] = []

        monkeypatch.setattr(
            DynamicProcessingWorkflow,
            "process_single",
            staticmethod(lambda entry: called.append(entry)),
        )

        dynamic_workflow(
            settings=_driver_settings(
                pipeline_params=[
                    {"steps": [], "label": "first"},
                    {"steps": [], "label": "second"},
                ]
            ),
            inputs=files,
        )

        assert [entry.params["label"] for entry in called] == ["first", "second"]
        assert [entry.active for entry in called] == [path.resolve() for path in files]

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
            DynamicProcessingWorkflow,
            "process_single",
            staticmethod(lambda entry: called.append(entry)),
        )

        dynamic_workflow(
            settings=_driver_settings(),
            inputs=file_path,
            save_filepaths_to=files_path,
            save_plan_to=plan_path,
            plan_only=True,
        )

        assert plan_path.exists()
        loaded = RunPlan.load(plan_path)
        assert len(loaded.entries) == 1
        assert loaded.entries[0].active == file_path.resolve()
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
            DynamicProcessingWorkflow,
            "process_single",
            staticmethod(lambda entry: called.append(entry)),
        )

        dynamic_workflow(
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
            DynamicProcessingWorkflow,
            "process_single",
            staticmethod(lambda entry: called.append(entry)),
        )

        dynamic_workflow(
            settings=_driver_settings(),
            from_plan=plan_path,
        )

        assert called == [plan.entries[0]]

    def test_from_plan_respects_start_end(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        actives = []
        for index in range(4):
            path = tmp_path / f"img-{index}.nii.gz"
            path.write_bytes(b"")
            actives.append(path)
        plan_path = tmp_path / "job.duckdb"
        plan = _workflow().plan(actives, save_plan_to=plan_path)
        called: list[Path] = []

        monkeypatch.setattr(
            DynamicProcessingWorkflow,
            "process_single",
            staticmethod(lambda entry: called.append(entry.active)),
        )

        dynamic_workflow(
            settings=_driver_settings(),
            from_plan=plan_path,
            start=1,
            end=3,
        )

        assert called == [plan.entries[1].active, plan.entries[2].active]

    def test_plan_only_warns_and_ignores_start_end(
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
            DynamicProcessingWorkflow,
            "process_single",
            staticmethod(lambda entry: called.append(entry)),
        )

        dynamic_workflow(
            settings=_driver_settings(),
            inputs=file_path,
            save_plan_to=plan_path,
            plan_only=True,
            start=1,
        )

        err = capsys.readouterr().err
        assert "Ignoring `start`/`end`" in err
        assert plan_path.exists()
        assert len(RunPlan.load(plan_path).entries) == 1
        assert called == []

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
            DynamicProcessingWorkflow,
            "process_single",
            staticmethod(lambda entry: called.append(entry)),
        )

        dynamic_workflow(
            settings=_driver_settings(),
            inputs=tmp_path / "ignored.nii.gz",
            from_plan=plan_path,
        )

        err = capsys.readouterr().err
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
            match="Failed to instantiate workflow 'DynamicProcessingWorkflow'",
        ):
            dynamic_workflow(
                settings={**_driver_settings(), "not_a_real_kwarg": True},
                inputs="/tmp/a.nii.gz",
            )
