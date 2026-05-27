"""Tests for the preprocessing workflow base and concrete classes.

These tests pin the public contract of :class:`PreprocessingWorkflow` and
every concrete subclass listed in :data:`WORKFLOW_CLASSES`. Append new
concrete workflows to that tuple to bring them under the same suite.

The workflow API exercised here is:

* ``stage(files, staging_config) -> list[StagedEntry]``
  – assembles per-task work units (single file, multimodal bundle, ...).
* ``process_single(entry, pipeline_config) -> None``
  – picklable per-entry hook (must be ``@staticmethod`` on the class
  to survive ``ProcessPoolExecutor``'s default ``spawn`` start method;
  tests substitute it by binding a module-level function as an instance
  attribute via ``monkeypatch``, which is also picklable).
* ``run()`` (aliased to ``__call__``) catches per-entry exceptions and
  reports them through the status logger; the workflow itself never
  re-raises a single entry's failure.

The behaviors exercised here are:

1. Construction from a single :class:`pathlib.Path`, a list of paths,
   a ``str``, a list of ``str``, or an explorer ``dict`` config.
2. Filterable file discovery using ``nifti_finder``'s
   ``AllPurposeFileExplorer`` (patterns + optional filters, including
   ``ComposeFilter``) when the explorer ``dict`` config is used.
3. Staging contract: ``stage`` must return ``list[StagedEntry]``;
   the default 1-file-per-entry behavior is verified.
4. Serial and parallel execution: ``process_single`` is invoked exactly
   once per staged entry, regardless of worker count.
5. Per-entry exception isolation: a failing entry does not abort the
   run; it is recorded as ``FAILURE`` in the status log.
6. Logging wiring: main, status, and worker logs respect the
   ``logs_root`` / ``*_logs`` / ``dev_mode`` flags, and worker records
   are stamped with the per-entry ``input_file`` context.

Module-level helpers used as ``process_single`` doubles are kept
picklable for the multi-worker tests.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, TypeAlias

import pytest

from niiflow.preproc.workflows.workflow import (
    PreprocessFiles,
    PreprocessingWorkflow,
    StagedEntry,
)

# ---------------------------------------------------------------------------
# Registry of concrete workflow classes covered by the shared test suite.
# Append new concrete workflows here to extend coverage automatically.
# ---------------------------------------------------------------------------

WorkflowFactory: TypeAlias = Callable[..., PreprocessingWorkflow]

WORKFLOW_CLASSES: tuple[WorkflowFactory, ...] = (PreprocessFiles,)


@pytest.fixture(params=WORKFLOW_CLASSES, ids=lambda cls: cls.__name__)
def workflow_cls(request: pytest.FixtureRequest) -> WorkflowFactory:
    """Concrete workflow class under test for this iteration."""
    return request.param


# ---------------------------------------------------------------------------
# Module-level test doubles for ``process_single`` (picklable for spawn).
# Each one matches the abstract signature ``(entry, pipeline_config=None)``.
# ---------------------------------------------------------------------------


def _proc_noop(
    entry: StagedEntry, pipeline_config: dict[str, Any] | None = None
) -> None:
    """No-op processor used when only orchestration / logging is under test."""
    return None


def _proc_touch_sentinel(
    entry: StagedEntry, pipeline_config: dict[str, Any] | None = None
) -> None:
    """Write a per-entry sentinel under ``pipeline_config['out_dir']``."""
    cfg = pipeline_config or {}
    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    src = Path(entry.data)
    (out_dir / f"{src.name}.done").write_text("ok", encoding="utf-8")


def _proc_record_call(
    entry: StagedEntry, pipeline_config: dict[str, Any] | None = None
) -> None:
    """Append the entry's path to a shared log file under ``out_dir``."""
    cfg = pipeline_config or {}
    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    log = out_dir / "calls.log"
    with log.open("a", encoding="utf-8") as fh:
        fh.write(f"{Path(entry.data).as_posix()}\n")


def _proc_boom(
    entry: StagedEntry, pipeline_config: dict[str, Any] | None = None
) -> None:
    """Always raise; used to exercise FAILURE/status-log behavior."""
    raise RuntimeError(f"pipeline boom for {entry.label}")


def _proc_log_info(
    entry: StagedEntry, pipeline_config: dict[str, Any] | None = None
) -> None:
    """Emit an INFO record at root level so it flows through the worker queue."""
    cfg = pipeline_config or {}
    msg = cfg.get("message", "WORKER_INFO")
    logging.getLogger().info(msg)


def _proc_log_debug(
    entry: StagedEntry, pipeline_config: dict[str, Any] | None = None
) -> None:
    """Emit a DEBUG record at root level (only captured in dev_mode)."""
    cfg = pipeline_config or {}
    msg = cfg.get("message", "WORKER_DEBUG")
    logging.getLogger().debug(msg)


# ---------------------------------------------------------------------------
# Fixtures: synthetic BIDS-like NIfTI dataset and logs dir.
# ---------------------------------------------------------------------------


@pytest.fixture
def dataset_root(tmp_path: Path) -> Path:
    """A small synthetic BIDS-like dataset on disk.

    Structure::

        dataset/
          sub-01/anat/sub-01_T1w.nii.gz
          sub-01/anat/sub-01_T2w.nii.gz
          sub-01/anat/sub-01_T1w_seg.nii.gz
          sub-02/anat/sub-02_T1w.nii.gz
          sub-02/anat/sub-02_T2w.nii
          sub-02/anat/sub-02_T1w_seg.nii.gz
          sub-03/anat/sub-03_T1w.nii.gz
          README.txt   (should never be picked up by nifti patterns)
    """
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
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"")
    return root


@pytest.fixture
def logs_dir(tmp_path: Path) -> Path:
    """A fresh logs directory for each test (created lazily by the workflow)."""
    return tmp_path / "logs"


# ---------------------------------------------------------------------------
# Construction / input-shape coverage (shared across all workflow classes).
# ---------------------------------------------------------------------------


class TestWorkflowConstruction:
    """Every concrete workflow must accept the documented input shapes."""

    def test_init_with_single_path(
        self, workflow_cls: WorkflowFactory, tmp_path: Path
    ) -> None:
        f = tmp_path / "a.nii.gz"
        f.write_bytes(b"")

        wf = workflow_cls(files=f, num_workers=1)

        assert wf.files == [f.resolve()]
        assert wf.root is None

    def test_init_with_string_path(
        self, workflow_cls: WorkflowFactory, tmp_path: Path
    ) -> None:
        f = tmp_path / "b.nii.gz"
        f.write_bytes(b"")

        wf = workflow_cls(files=str(f), num_workers=1)

        assert wf.files == [f.resolve()]
        assert wf.root is None

    def test_init_with_list_of_paths(
        self, workflow_cls: WorkflowFactory, tmp_path: Path
    ) -> None:
        files = [tmp_path / "a.nii.gz", tmp_path / "b.nii.gz"]
        for f in files:
            f.write_bytes(b"")

        wf = workflow_cls(files=files, num_workers=1)

        assert wf.files == [f.resolve() for f in files]
        assert wf.root is None

    def test_init_with_list_of_strings(
        self, workflow_cls: WorkflowFactory, tmp_path: Path
    ) -> None:
        files = [tmp_path / "a.nii.gz", tmp_path / "b.nii.gz"]
        for f in files:
            f.write_bytes(b"")

        wf = workflow_cls(files=[str(f) for f in files], num_workers=1)

        assert wf.files == [f.resolve() for f in files]
        assert wf.root is None

    def test_init_rejects_missing_file(
        self, workflow_cls: WorkflowFactory, tmp_path: Path
    ) -> None:
        with pytest.raises(FileNotFoundError):
            workflow_cls(files=tmp_path / "ghost.nii.gz", num_workers=1)

    def test_init_rejects_invalid_path_type_in_list(
        self, workflow_cls: WorkflowFactory, tmp_path: Path
    ) -> None:
        f = tmp_path / "a.nii.gz"
        f.write_bytes(b"")
        with pytest.raises(ValueError, match="Path or str"):
            workflow_cls(files=[f, 42], num_workers=1)  # type: ignore[list-item]

    def test_init_rejects_invalid_num_workers(
        self, workflow_cls: WorkflowFactory, tmp_path: Path
    ) -> None:
        f = tmp_path / "a.nii.gz"
        f.write_bytes(b"")
        with pytest.raises(ValueError, match="num_workers"):
            workflow_cls(files=f, num_workers="cpu")  # type: ignore[arg-type]

    def test_init_rejects_non_dict_pipeline_config(
        self, workflow_cls: WorkflowFactory, tmp_path: Path
    ) -> None:
        f = tmp_path / "a.nii.gz"
        f.write_bytes(b"")
        with pytest.raises(ValueError, match="pipeline_config"):
            workflow_cls(
                files=f, num_workers=1, pipeline_config=[1, 2, 3]  # type: ignore[arg-type]
            )

    def test_num_workers_auto_resolves_to_an_integer(
        self, workflow_cls: WorkflowFactory, tmp_path: Path
    ) -> None:
        f = tmp_path / "a.nii.gz"
        f.write_bytes(b"")

        wf = workflow_cls(files=f, num_workers="auto")

        assert isinstance(wf.num_workers, int)  # type: ignore[attr-defined]
        assert wf.num_workers >= 1  # type: ignore[attr-defined]

    def test_root_is_read_only_via_assignment(
        self, workflow_cls: WorkflowFactory, tmp_path: Path
    ) -> None:
        f = tmp_path / "a.nii.gz"
        f.write_bytes(b"")
        wf = workflow_cls(files=f, num_workers=1)
        with pytest.raises(AttributeError, match="read-only"):
            wf.root = tmp_path  # type: ignore[misc]

    def test_root_is_populated_when_using_dict_config(
        self, workflow_cls: WorkflowFactory, dataset_root: Path
    ) -> None:
        wf = workflow_cls(
            files={"root": dataset_root, "patterns": "*.nii*"}, num_workers=1
        )
        assert wf.root == dataset_root.resolve()


# ---------------------------------------------------------------------------
# Filterable file discovery (the ``dict`` explorer config path).
# ---------------------------------------------------------------------------


class TestWorkflowFilterableFileDiscovery:
    """The ``dict`` files config exercises ``nifti_finder``-backed discovery."""

    def test_discovers_all_nifti_files_with_simple_pattern(
        self, workflow_cls: WorkflowFactory, dataset_root: Path
    ) -> None:
        wf = workflow_cls(
            files={"root": dataset_root, "patterns": "*.nii*"}, num_workers=1
        )

        names = sorted(p.name for p in wf.files)
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
        assert all(p.is_absolute() for p in wf.files)
        assert all(p.name != "README.txt" for p in wf.files)

    def test_discovers_with_bids_like_pattern(
        self, workflow_cls: WorkflowFactory, dataset_root: Path
    ) -> None:
        wf = workflow_cls(
            files={
                "root": str(dataset_root),  # str root works too
                "patterns": "sub-*/anat/*T1w.nii*",
            },
            num_workers=1,
        )

        names = sorted(p.name for p in wf.files)
        # ``*T1w.nii*`` matches only names where ``.nii`` immediately follows
        # ``T1w`` (e.g. not ``*_T1w_seg.nii.gz``).
        assert names == sorted(
            ["sub-01_T1w.nii.gz", "sub-02_T1w.nii.gz", "sub-03_T1w.nii.gz"]
        )

    def test_single_filter_excludes_segmentation_files(
        self, workflow_cls: WorkflowFactory, dataset_root: Path
    ) -> None:
        wf = workflow_cls(
            files={
                "root": dataset_root,
                "patterns": "sub-*/anat/*T1w*.nii*",
                "filters": {
                    "name": "ExcludeFileRegex",
                    "kwargs": {"regex": r".*_seg\.nii.*"},
                },
            },
            num_workers=1,
        )

        names = sorted(p.name for p in wf.files)
        assert names == sorted(
            ["sub-01_T1w.nii.gz", "sub-02_T1w.nii.gz", "sub-03_T1w.nii.gz"]
        )

    def test_compose_filter_combines_multiple_criteria(
        self, workflow_cls: WorkflowFactory, dataset_root: Path
    ) -> None:
        wf = workflow_cls(
            files={
                "root": dataset_root,
                "patterns": "*.nii*",
                "filters": {
                    "name": "ComposeFilter",
                    "kwargs": {
                        "filters": [
                            {
                                "name": "ExcludeFileRegex",
                                "kwargs": {"regex": r".*T2w.*"},
                            },
                            {
                                "name": "IncludeFileRegex",
                                "kwargs": {"regex": r".*_seg.*"},
                            },
                        ],
                        "logic": "AND",
                    },
                },
            },
            num_workers=1,
        )

        names = sorted(p.name for p in wf.files)
        assert names == sorted(["sub-01_T1w_seg.nii.gz", "sub-02_T1w_seg.nii.gz"])

    def test_results_are_unique_and_sorted(
        self, workflow_cls: WorkflowFactory, dataset_root: Path
    ) -> None:
        wf = workflow_cls(
            files={
                "root": dataset_root,
                # Two overlapping patterns -> dedup must kick in.
                "patterns": ["*.nii*", "sub-*/anat/*T1w.nii*"],
            },
            num_workers=1,
        )

        assert wf.files == sorted(set(wf.files))

    def test_missing_root_key_raises(self, workflow_cls: WorkflowFactory) -> None:
        with pytest.raises(ValueError, match="root"):
            workflow_cls(files={"patterns": "*.nii*"}, num_workers=1)

    def test_missing_patterns_key_raises(
        self, workflow_cls: WorkflowFactory, dataset_root: Path
    ) -> None:
        with pytest.raises(ValueError, match="patterns"):
            workflow_cls(files={"root": dataset_root}, num_workers=1)

    def test_invalid_root_type_raises(self, workflow_cls: WorkflowFactory) -> None:
        with pytest.raises(ValueError, match="root"):
            workflow_cls(
                files={"root": 12345, "patterns": "*.nii*"},  # type: ignore[dict-item]
                num_workers=1,
            )


# ---------------------------------------------------------------------------
# Staging contract.
# ---------------------------------------------------------------------------


class TestWorkflowStaging:
    """``stage`` must return ``list[StagedEntry]``; otherwise ``run`` rejects it."""

    def test_default_stage_produces_one_entry_per_file(
        self, workflow_cls: WorkflowFactory, tmp_path: Path
    ) -> None:
        files = [tmp_path / f"img-{i}.nii.gz" for i in range(3)]
        for f in files:
            f.write_bytes(b"")

        wf = workflow_cls(files=files, num_workers=1)
        entries = wf.stage(wf.files, None)

        assert len(entries) == len(files)
        assert all(isinstance(e, StagedEntry) for e in entries)
        # ``data`` carries the underlying Path; ``label`` round-trips it.
        assert [e.data for e in entries] == wf.files
        assert [e.label for e in entries] == [str(p) for p in wf.files]

    def test_run_rejects_stage_returning_non_StagedEntry(
        self,
        workflow_cls: WorkflowFactory,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        f = tmp_path / "a.nii.gz"
        f.write_bytes(b"")
        wf = workflow_cls(files=f, num_workers=1)

        monkeypatch.setattr(wf, "stage", lambda files, cfg=None: ["not-an-entry"])

        with pytest.raises(ValueError, match="StagedEntry"):
            wf.run()


# ---------------------------------------------------------------------------
# Pipeline execution: ``run()`` invokes ``process_single`` for every entry.
# ---------------------------------------------------------------------------


class TestWorkflowRun:
    """``run()`` must call ``process_single`` once per staged entry."""

    def test_serial_run_processes_each_file_exactly_once(
        self,
        workflow_cls: WorkflowFactory,
        dataset_root: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        sentinels = tmp_path / "sentinels"
        wf = workflow_cls(
            files={"root": dataset_root, "patterns": "*.nii*"},
            pipeline_config={"out_dir": str(sentinels)},
            num_workers=1,
        )
        monkeypatch.setattr(wf, "process_single", _proc_touch_sentinel)

        wf.run()

        produced = {p.name for p in sentinels.iterdir()}
        expected = {f"{p.name}.done" for p in wf.files}
        assert produced == expected
        assert len(produced) == len(wf.files)

    def test_call_dunder_delegates_to_run(
        self,
        workflow_cls: WorkflowFactory,
        dataset_root: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        sentinels = tmp_path / "sentinels"
        wf = workflow_cls(
            files={"root": dataset_root, "patterns": "*.nii*"},
            pipeline_config={"out_dir": str(sentinels)},
            num_workers=1,
        )
        monkeypatch.setattr(wf, "process_single", _proc_touch_sentinel)

        wf()  # __call__ alias for run

        assert {p.name for p in sentinels.iterdir()} == {
            f"{p.name}.done" for p in wf.files
        }

    def test_process_single_receives_StagedEntry_with_resolved_path(
        self,
        workflow_cls: WorkflowFactory,
        dataset_root: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        wf = workflow_cls(
            files={"root": dataset_root, "patterns": "*.nii*"}, num_workers=1
        )

        seen: list[StagedEntry] = []

        def _capture(
            entry: StagedEntry, pipeline_config: dict[str, Any] | None = None
        ) -> None:
            seen.append(entry)

        monkeypatch.setattr(wf, "process_single", _capture)
        wf.run()

        assert len(seen) == len(wf.files)
        assert all(isinstance(e, StagedEntry) for e in seen)
        seen_paths = sorted(Path(e.data) for e in seen)
        assert seen_paths == sorted(wf.files)
        assert all(Path(e.data).is_absolute() for e in seen)

    def test_run_with_explicit_list_calls_fn_for_each(
        self,
        workflow_cls: WorkflowFactory,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        sentinels = tmp_path / "sentinels"
        files = [tmp_path / f"img-{i}.nii.gz" for i in range(5)]
        for f in files:
            f.write_bytes(b"")

        wf = workflow_cls(
            files=files,
            pipeline_config={"out_dir": str(sentinels)},
            num_workers=1,
        )
        monkeypatch.setattr(wf, "process_single", _proc_touch_sentinel)

        wf.run()

        assert {p.name for p in sentinels.iterdir()} == {
            f"{f.name}.done" for f in files
        }

    def test_run_on_empty_discovery_is_a_noop(
        self,
        workflow_cls: WorkflowFactory,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        empty_root = tmp_path / "empty"
        empty_root.mkdir()

        wf = workflow_cls(
            files={"root": empty_root, "patterns": "*.nii*"}, num_workers=1
        )

        called: list[StagedEntry] = []

        def _record(
            entry: StagedEntry, pipeline_config: dict[str, Any] | None = None
        ) -> None:
            called.append(entry)

        monkeypatch.setattr(wf, "process_single", _record)

        wf.run()

        assert called == []
        assert wf.files == []

    def test_run_catches_per_entry_exceptions_and_continues(
        self,
        workflow_cls: WorkflowFactory,
        dataset_root: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        sentinels = tmp_path / "sentinels"
        wf = workflow_cls(
            files={"root": dataset_root, "patterns": "*.nii*"},
            pipeline_config={"out_dir": str(sentinels)},
            num_workers=1,
        )

        attempted: list[StagedEntry] = []

        def _mostly_ok(
            entry: StagedEntry, pipeline_config: dict[str, Any] | None = None
        ) -> None:
            attempted.append(entry)
            if "sub-02_T1w.nii.gz" in entry.label:
                raise RuntimeError("synthetic failure")
            _proc_touch_sentinel(entry, pipeline_config)

        monkeypatch.setattr(wf, "process_single", _mostly_ok)

        wf.run()  # must not raise

        assert len(attempted) == len(wf.files)
        produced = {p.name for p in sentinels.iterdir()}
        # The failing entry produced no sentinel; every other did.
        assert "sub-02_T1w.nii.gz.done" not in produced
        assert produced == {
            f"{p.name}.done" for p in wf.files if "sub-02_T1w.nii.gz" not in str(p)
        }


# ---------------------------------------------------------------------------
# Multi-worker execution
# ---------------------------------------------------------------------------


class TestWorkflowMultiWorkerExecution:
    """With ``num_workers > 1`` work must still be done exactly once per entry."""

    @pytest.mark.parametrize("num_workers", [2, 4])
    def test_multi_worker_processes_each_file_exactly_once(
        self,
        workflow_cls: WorkflowFactory,
        dataset_root: Path,
        tmp_path: Path,
        num_workers: int,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        sentinels = tmp_path / "sentinels"
        wf = workflow_cls(
            files={"root": dataset_root, "patterns": "*.nii*"},
            pipeline_config={"out_dir": str(sentinels)},
            num_workers=num_workers,
        )
        monkeypatch.setattr(wf, "process_single", _proc_touch_sentinel)

        wf.run()

        produced = {p.name for p in sentinels.iterdir()}
        expected = {f"{p.name}.done" for p in wf.files}
        assert (
            produced == expected
        ), "Every file must be processed exactly once, regardless of worker count"

    def test_multi_worker_calls_process_single_total_n_times(
        self,
        workflow_cls: WorkflowFactory,
        dataset_root: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        wf = workflow_cls(
            files={"root": dataset_root, "patterns": "*.nii*"},
            pipeline_config={"out_dir": str(tmp_path)},
            num_workers=2,
        )
        monkeypatch.setattr(wf, "process_single", _proc_record_call)

        wf.run()

        log = (tmp_path / "calls.log").read_text(encoding="utf-8").splitlines()
        assert sorted(log) == sorted(p.as_posix() for p in wf.files)

    def test_multi_worker_isolates_per_entry_failures(
        self,
        workflow_cls: WorkflowFactory,
        dataset_root: Path,
        logs_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        wf = workflow_cls(
            files={"root": dataset_root, "patterns": "*.nii*"},
            num_workers=2,
            logs_root=logs_dir,
        )
        monkeypatch.setattr(wf, "process_single", _proc_boom)

        wf.run()  # must not raise

        status = (logs_dir / "status.log").read_text(encoding="utf-8")
        failure_lines = [line for line in status.splitlines() if "FAILURE" in line]
        assert len(failure_lines) == len(wf.files)


# ---------------------------------------------------------------------------
# Logging behavior: main, status, and worker log wiring.
# ---------------------------------------------------------------------------


class TestWorkflowLogging:
    """Verify file/handler wiring around main / status / worker logs."""

    def test_no_logs_root_creates_no_log_files(
        self,
        workflow_cls: WorkflowFactory,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        f = tmp_path / "a.nii.gz"
        f.write_bytes(b"")
        monkeypatch.chdir(tmp_path)
        wf = workflow_cls(files=f, num_workers=1, logs_root=None)
        monkeypatch.setattr(wf, "process_single", _proc_noop)

        wf.run()

        assert not (tmp_path / "main.log").exists()
        assert not (tmp_path / "status.log").exists()
        assert not (tmp_path / "workers.log").exists()

    @pytest.mark.parametrize(
        ("main_logs", "status_logs", "worker_logs"),
        [
            (False, True, True),
            (True, False, True),
            (True, True, False),
            (False, False, True),
            (True, False, False),
        ],
    )
    def test_partial_log_toggles_without_logs_root_raise(
        self,
        workflow_cls: WorkflowFactory,
        tmp_path: Path,
        main_logs: bool,
        status_logs: bool,
        worker_logs: bool,
    ) -> None:
        f = tmp_path / "a.nii.gz"
        f.write_bytes(b"")

        with pytest.raises(ValueError, match="logs_root"):
            workflow_cls(
                files=f,
                num_workers=1,
                logs_root=None,
                main_logs=main_logs,
                status_logs=status_logs,
                worker_logs=worker_logs,
            )

    def test_console_includes_main_status_and_worker_logs_without_logs_root(
        self,
        workflow_cls: WorkflowFactory,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        files = [tmp_path / f"img-{i}.nii.gz" for i in range(2)]
        for f in files:
            f.write_bytes(b"")

        wf = workflow_cls(
            files=files,
            pipeline_config={"message": "WORKER_CONSOLE"},
            num_workers=2,
            logs_root=None,
            main_logs=True,
            status_logs=True,
            worker_logs=True,
        )
        monkeypatch.setattr(wf, "process_single", _proc_log_info)

        wf.run()
        captured = capsys.readouterr()
        console = f"{captured.out}\n{captured.err}"

        assert "Starting workflow" in console
        assert "WORKER_CONSOLE" in console
        assert "| SUCCESS" in console
        assert "Workflow complete" in console
    
    def test_console_includes_main_status_and_worker_logs_when_logs_root_set(
        self,
        workflow_cls: WorkflowFactory,
        tmp_path: Path,
        logs_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        files = [tmp_path / f"img-{i}.nii.gz" for i in range(2)]
        for f in files:
            f.write_bytes(b"")

        wf = workflow_cls(
            files=files,
            pipeline_config={"message": "WORKER_CONSOLE"},
            num_workers=2,
            logs_root=logs_dir,
            main_logs=True,
            status_logs=True,
            worker_logs=True,
        )
        monkeypatch.setattr(wf, "process_single", _proc_log_info)

        wf.run()
        captured = capsys.readouterr()
        console = f"{captured.out}\n{captured.err}"

        assert "Starting workflow" in console
        assert "WORKER_CONSOLE" in console
        assert "| SUCCESS" in console
        assert "Workflow complete" in console

    def test_main_log_records_lifecycle_when_logs_root_set(
        self,
        workflow_cls: WorkflowFactory,
        tmp_path: Path,
        logs_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        f = tmp_path / "a.nii.gz"
        f.write_bytes(b"")
        wf = workflow_cls(files=f, num_workers=1, logs_root=logs_dir)
        monkeypatch.setattr(wf, "process_single", _proc_noop)

        wf.run()

        main_log = logs_dir / "main.log"
        assert main_log.exists()
        content = main_log.read_text(encoding="utf-8")
        assert "Starting workflow" in content
        assert "Workflow complete" in content

    def test_main_log_not_written_when_main_logs_disabled(
        self,
        workflow_cls: WorkflowFactory,
        tmp_path: Path,
        logs_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        f = tmp_path / "a.nii.gz"
        f.write_bytes(b"")
        wf = workflow_cls(
            files=f,
            num_workers=1,
            logs_root=logs_dir,
            main_logs=False,
        )
        monkeypatch.setattr(wf, "process_single", _proc_noop)

        wf.run()

        assert not (logs_dir / "main.log").exists()

    def test_status_log_records_success_per_entry(
        self,
        workflow_cls: WorkflowFactory,
        dataset_root: Path,
        logs_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        wf = workflow_cls(
            files={"root": dataset_root, "patterns": "*.nii*"},
            num_workers=1,
            logs_root=logs_dir,
        )
        monkeypatch.setattr(wf, "process_single", _proc_noop)

        wf.run()

        status_log = logs_dir / "status.log"
        assert status_log.exists()
        lines = status_log.read_text(encoding="utf-8").splitlines()
        assert len(lines) == len(wf.files)
        assert all(line.endswith("| SUCCESS") for line in lines)
        # Every entry's label appears exactly once in the status log.
        for entry_label in (str(p) for p in wf.files):
            assert sum(entry_label in line for line in lines) == 1

    def test_status_log_records_failure_with_traceback(
        self,
        workflow_cls: WorkflowFactory,
        tmp_path: Path,
        logs_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        f = tmp_path / "a.nii.gz"
        f.write_bytes(b"")
        wf = workflow_cls(files=f, num_workers=1, logs_root=logs_dir)
        monkeypatch.setattr(wf, "process_single", _proc_boom)

        wf.run()

        content = (logs_dir / "status.log").read_text(encoding="utf-8")
        assert "FAILURE" in content
        assert "RuntimeError" in content
        assert "pipeline boom" in content

    def test_status_log_not_written_when_status_logs_disabled(
        self,
        workflow_cls: WorkflowFactory,
        tmp_path: Path,
        logs_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        f = tmp_path / "a.nii.gz"
        f.write_bytes(b"")
        wf = workflow_cls(
            files=f,
            num_workers=1,
            logs_root=logs_dir,
            status_logs=False,
        )
        monkeypatch.setattr(wf, "process_single", _proc_noop)

        wf.run()

        assert not (logs_dir / "status.log").exists()

    def test_worker_log_stamps_input_file_context_in_parallel(
        self,
        workflow_cls: WorkflowFactory,
        dataset_root: Path,
        logs_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        wf = workflow_cls(
            files={"root": dataset_root, "patterns": "*.nii*"},
            pipeline_config={"message": "WORKER_MSG"},
            num_workers=2,
            logs_root=logs_dir,
        )
        monkeypatch.setattr(wf, "process_single", _proc_log_info)

        wf.run()

        workers_log = (logs_dir / "workers.log")
        assert workers_log.exists()
        worker_lines = workers_log.read_text(encoding="utf-8").splitlines()
        worker_lines = [line for line in worker_lines if "WORKER_MSG" in line]
        # One emission per entry, each stamped with the entry's label
        # (which is ``str(path)`` in the default 1:1 staging).
        assert len(worker_lines) == len(wf.files)
        for entry_label in (str(p) for p in wf.files):
            assert any(
                entry_label in line for line in worker_lines
            ), f"input_file context for {entry_label!r} missing from workers.log"

    def test_worker_logs_disabled_suppresses_worker_info(
        self,
        workflow_cls: WorkflowFactory,
        dataset_root: Path,
        logs_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        wf = workflow_cls(
            files={"root": dataset_root, "patterns": "*.nii*"},
            pipeline_config={"message": "WORKER_INFO_OFF"},
            num_workers=2,
            logs_root=logs_dir,
            worker_logs=False,
        )
        monkeypatch.setattr(wf, "process_single", _proc_log_info)

        wf.run()

        workers_log = logs_dir / "workers.log"
        assert not workers_log.exists()

    def test_dev_mode_captures_worker_debug_records(
        self,
        workflow_cls: WorkflowFactory,
        dataset_root: Path,
        logs_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        wf = workflow_cls(
            files={"root": dataset_root, "patterns": "*.nii*"},
            pipeline_config={"message": "WORKER_DEBUG_MSG"},
            num_workers=2,
            logs_root=logs_dir,
            dev_mode=True,
        )
        monkeypatch.setattr(wf, "process_single", _proc_log_debug)

        wf.run()

        workers_log = (logs_dir / "workers.log")
        assert workers_log.exists()
        assert "WORKER_DEBUG_MSG" in workers_log.read_text(encoding="utf-8")