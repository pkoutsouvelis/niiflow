"""Base and concrete classes for preprocessing workflows."""

from __future__ import annotations

__all__ = [
    "PreprocessFiles",
]

from abc import ABC, abstractmethod
from typing import Callable, Any, Literal
from pathlib import Path
import multiprocessing
import time
from concurrent.futures import (
    FIRST_COMPLETED,
    Future,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    as_completed,
    wait,
)
import traceback
from dataclasses import dataclass

from niiflow.preproc.utils.file import resolve_path
from niiflow.preproc.data import get_data_explorer
from niiflow.preproc.workflows.logging_manager import LoggingManager, ParallelLogging
from niiflow.preproc.workflows.logging_utils import (
    set_input_file_context,
    reset_input_file_context,
)
from niiflow.preproc.utils._types import InputData


def _execute_one(
    process_fn: Callable,
    entry: Any,
    pipeline_config: dict[str, Any] | None = None,
) -> None:
    """Picklable single-entry executor with per-task input file context."""
    token = set_input_file_context(str(entry))
    try:
        process_fn(entry, pipeline_config)
    finally:
        reset_input_file_context(token)


@dataclass(frozen=True)
class StagedEntry:
    """A single input entry for processing.

    Args:
        label: A string label for the entry.
        data: The data for the entry (file-dependent).
    """

    label: str
    data: Any

    def __str__(self) -> str:
        return self.label

    def __repr__(self) -> str:
        return f"StagedEntry(label={self.label!r}, data={self.data!r})"


class PreprocessingWorkflow(ABC):
    """
    Base class for preprocessing workflows containing all machinery for file
    discovery, multi-worker execution, and logging.

    Subclasses must implement the `stage` and `process_single` methods.

    Notes:
        The `files` input can be a single file, a list of files, a string, a list of
        strings, or a dictionary with a `root` key and a `patterns` key.

        If a dictionary is provided, it will be used to instantiate a `nifti_finder`
        explorer, requiring the keys:

        - `root`: The root directory to search for files.
        - `patterns`: The patterns to match for files.
        - `filters` (optional): The filters to apply to the files.

        The example below shows how to use a dictionary `files` input to instantiate an
        explorer with a composed filter configuration:

    Example:
        ```python
        >>> files = {
        ...     "root": "/data/bids",
        ...     "patterns": "*.nii*",
        ...     "filters": {
        ...         "name": "ComposeFilter",
        ...         "kwargs": {
        ...             "logic": "AND",
        ...             "filters": [
        ...                 {
        ...                     "name": "IncludeFileRegex",
        ...                     "kwargs": {"regex": r".*_T1w\\.nii(\\.gz)?$"},
        ...                 },
        ...                 {
        ...                     "name": "ExcludeFileRegex",
        ...                     "kwargs": {"regex": r".*_seg\\.nii(\\.gz)?$"},
        ...                 },
        ...             ],
        ...         },
        ...     },
        ... }
        ```

    Args:
        files (Path | list[Path] | str | list[str] | dict[str, Any]):
            The input files to process. See the example above for more details.
        staging_config (dict[str, Any] | None):
            The configuration for staging; passed to the `stage` method.
        pipeline_config (dict[str, Any] | None):
            The configuration for the pipeline; passed to the `process_single` method.
        num_workers (int | Literal["auto"]):
            The number of workers to use. If `"auto"`, the number of workers is
            set to the number of availableCPU cores.
        logs_root (Path | str | None):
            The root directory for logs. If not provided no logs will be written
            other than to console.
        main_logs (bool):
            Whether to show main logs to console or to `logs_root/main.log` if
            `logs_root` is provided.
        status_logs (bool):
            Whether to show status logs to console or to `logs_root/status.log` if
            `logs_root` is provided.
        worker_logs (bool):
            Whether to show worker logs to console or to `logs_root/worker.log` if
            `logs_root` is provided.
        dev_mode (bool): Whether to enable debug mode. Activates debug-level
            logging for worker processes.
        timeout (float | None): Soft per-entry time limit in seconds. When set,
            entries that exceed the limit are recorded as ``TIMEOUT`` in the
            status log and processing continues with remaining entries. Worker
            processes are not terminated.
    """

    def __init__(
        self,
        files: InputData,
        staging_config: dict[str, Any] | None = None,
        pipeline_config: dict[str, Any] | None = None,
        num_workers: int | Literal["auto"] = "auto",
        logs_root: Path | str | None = None,
        main_logs: bool = True,
        status_logs: bool = True,
        worker_logs: bool = True,
        dev_mode: bool = False,
        timeout: float | None = None,
    ) -> None:
        self._logging_manager = LoggingManager(
            logs_root=logs_root,
            main_logs=main_logs,
            status_logs=status_logs,
            worker_logs=worker_logs,
            dev_mode=dev_mode,
        )
        self._main_logger = self._logging_manager.setup_main_logging()
        self._status_logger = self._logging_manager.setup_status_logging()
        self.files = files
        self.staging_config = staging_config
        self.pipeline_config = pipeline_config
        self.num_workers = num_workers
        self.timeout = timeout

    @property
    def files(self) -> list[Path]:
        return self._files

    @property
    def root(self) -> Path | None:
        return self._root

    @property
    def staging_config(self) -> dict[str, Any] | None:
        return self._staging_config

    @property
    def pipeline_config(self) -> dict[str, Any] | None:
        return self._pipeline_config

    @property
    def num_workers(self) -> int | Literal["auto"]:
        return self._num_workers

    @property
    def timeout(self) -> float | None:
        return self._timeout

    @files.setter
    def files(self, files: InputData) -> None:
        self._root, self._files = self._resolve_files(files)

    @root.setter
    def root(self, root: Path | None) -> None:
        raise AttributeError("`root` is read-only; use `files` with a dictionary input")

    @staging_config.setter
    def staging_config(self, staging_config: dict[str, Any] | None) -> None:
        if staging_config is not None and not isinstance(staging_config, dict):
            raise ValueError(
                f"`staging_config` must be a dictionary or None, got {type(staging_config).__name__}"
            )
        self._staging_config = staging_config

    @pipeline_config.setter
    def pipeline_config(self, pipeline_config: dict[str, Any] | None) -> None:
        if pipeline_config is not None and not isinstance(pipeline_config, dict):
            raise ValueError(
                f"`pipeline_config` must be a dictionary or None, got {type(pipeline_config).__name__}"
            )
        self._pipeline_config = pipeline_config

    @num_workers.setter
    def num_workers(self, num_workers: int | Literal["auto"]) -> None:
        if isinstance(num_workers, str):
            if num_workers != "auto":
                raise ValueError(
                    f"`num_workers` must be 'auto' or an integer, got `{num_workers}`"
                )
            self._num_workers = multiprocessing.cpu_count()
        elif isinstance(num_workers, int):
            if num_workers < 1:
                raise ValueError(
                    f"`num_workers` must be at least 1, got `{num_workers}`"
                )
            self._num_workers = num_workers
        else:
            raise ValueError(
                f"`num_workers` must be 'auto' or an integer, got `{num_workers}`"
            )

    @timeout.setter
    def timeout(self, timeout: float | None) -> None:
        if timeout is not None:
            if not isinstance(timeout, (int, float)):
                raise ValueError(
                    f"`timeout` must be a number or None, got {type(timeout).__name__}"
                )
            if timeout <= 0:
                raise ValueError(f"`timeout` must be positive, got `{timeout}`")
            timeout = float(timeout)
        self._timeout = timeout

    def _resolve_files(self, files: InputData) -> tuple[Path | None, list[Path]]:
        if isinstance(files, (Path, str)):
            f = resolve_path(files)
            if not f.exists():
                raise FileNotFoundError(f"File {f} does not exist")
            return None, [f]

        if isinstance(files, list):
            fs = []
            for f in files:
                f = resolve_path(f)
                if not f.exists():
                    raise FileNotFoundError(f"File {f} does not exist")
                fs.append(f)
            return None, fs

        if isinstance(files, dict):
            self.log("Instantiating data explorer...")

            if "root" not in files:
                raise ValueError("`root` key is required in `data` dictionary")
            if not isinstance(files["root"], (Path, str)):
                raise ValueError(
                    f"`root` must be a Path or str object, got {type(files['root']).__name__}"
                )

            if "pattern" not in files:
                raise ValueError("`pattern` key is required in `data` dictionary")

            try:
                explorer = get_data_explorer(
                    pattern=files["pattern"],
                    filter_kwargs=files.get("filters", None),
                )
            except Exception as e:
                raise RuntimeError(
                    "Failed to bind arguments to data explorer; see "
                    "``nifti_finder``'s documentation of ``AllPurposeFileExplorer`` for "
                    "more details."
                ) from e

            self.log("Data explorer instantiated successfully.")
            self.log("Finding files...")
            root = resolve_path(files["root"])
            files = explorer.list(root, sort=True, unique=True)
            self.log(f"Found {len(files)} unique files.")
            return root, files

        raise ValueError(f"Invalid `files` input: {type(files).__name__}")

    def log(
        self,
        message: str,
        level: Literal["debug", "info", "warning", "error", "critical"] = "info",
    ) -> None:
        getattr(self._main_logger, level)(message)

    def status(
        self,
        message: str,
    ) -> None:
        self._status_logger.info(message)

    def _report_future_result(self, fut: Future[None], entry: StagedEntry) -> None:
        """Map a completed future to a status-log line."""
        try:
            if self._timeout is not None:
                fut.result(timeout=self._timeout)
            else:
                fut.result()
            self.status(f"{entry} | SUCCESS")
        except TimeoutError:
            self.status(f"{entry} | TIMEOUT | exceeded {self._timeout}s")
        except Exception:
            tb = traceback.format_exc().strip().replace("\n", "\n  ")
            self.status(f"{entry} | FAILURE | {tb}")

    def _run_serial(self, entries: list[StagedEntry]) -> None:
        if self._timeout is None:
            for entry in entries:
                try:
                    _execute_one(self.process_single, entry, self._pipeline_config)
                    self.status(f"{entry} | SUCCESS")
                except Exception:
                    tb = traceback.format_exc().strip().replace("\n", "\n  ")
                    self.status(f"{entry} | FAILURE | {tb}")
            return

        with ThreadPoolExecutor(max_workers=1) as executor:
            for entry in entries:
                fut = executor.submit(
                    _execute_one,
                    self.process_single,
                    entry,
                    self._pipeline_config,
                )
                self._report_future_result(fut, entry)

    def _run_parallel(
        self, entries: list[StagedEntry], parallel: ParallelLogging
    ) -> None:
        with ProcessPoolExecutor(
            max_workers=self._num_workers,
            initializer=parallel.worker_init_fn,
        ) as pool:
            futures = {
                pool.submit(
                    _execute_one,
                    self.process_single,
                    entry,
                    self._pipeline_config,
                ): entry
                for entry in entries
            }
            if self._timeout is None:
                for fut in as_completed(futures):
                    self._report_future_result(fut, futures[fut])
                return

            # as_completed only yields finished futures; poll so hung entries
            # can be marked TIMEOUT while others continue (soft timeout).
            # Looping naively over futures would address this but without
            # completion-order handling and responsive status updates.
            submit_times = {fut: time.monotonic() for fut in futures}
            handled: set[Future[None]] = set()
            poll_interval = min(0.5, self._timeout / 10)

            while len(handled) < len(futures):
                remaining = {
                    fut: entry for fut, entry in futures.items() if fut not in handled
                }
                done, not_done = wait(
                    remaining.keys(),
                    timeout=poll_interval,
                    return_when=FIRST_COMPLETED,
                )
                for fut in done:
                    if fut in handled:
                        continue
                    self._report_future_result(fut, remaining[fut])
                    handled.add(fut)

                now = time.monotonic()
                for fut in not_done:
                    if now - submit_times[fut] > self._timeout:
                        entry = remaining[fut]
                        # Parent stops waiting; the worker process is not terminated.
                        self.status(f"{entry} | TIMEOUT | exceeded {self._timeout}s")
                        handled.add(fut)

    def run(self) -> None:
        self.log(
            f"Starting workflow with {len(self.files)} files, {self._num_workers} workers"
        )

        entries = self.stage(self.files, self._staging_config)
        if not isinstance(entries, list) or not all(
            isinstance(e, StagedEntry) for e in entries
        ):
            raise ValueError("`stage` must return a list of `StagedEntry` objects")
        if len(entries) == 0:
            self.log("No entries to process")
            return

        with (
            self._logging_manager.setup_parallel_logging() as parallel
        ):  # listener starts, stops on exit
            if self._num_workers <= 1:
                self._run_serial(entries)
            else:
                self._run_parallel(entries, parallel)

        self.log("Workflow complete")

    __call__ = run

    @abstractmethod
    def stage(
        self, files: list[Path], staging_config: dict[str, Any] | None = None
    ) -> list[StagedEntry]:
        """Stage the input files into a list of input entries for processing.

        Example cases include aggregating multimodal data of a single subject, or
        no-op (single-file processing).

        Args:
            files: The list of files to stage.
            staging_config: The configuration for staging.

        Returns:
            A list of `StagedEntry` objects.
        """
        ...

    @staticmethod
    @abstractmethod
    def process_single(
        entry: StagedEntry, pipeline_config: dict[str, Any] | None = None
    ) -> None:
        """Process a single input entry.

        The method is static to allow for pickling in multi-worker mode.

        Args:
            entry: The input `StagedEntry` to process.
            pipeline_config: The configuration for the pipeline (shared across all entries).
        """
        ...


class PreprocessFiles(PreprocessingWorkflow):
    """Preprocessing workflow that runs a pipeline for each file.

    Args:
        pipeline: The preprocessing pipeline to run.
    """

    def __init__(
        self,
        files: InputData,
        pipeline_config: dict[str, Any] | None = None,
        num_workers: int | Literal["auto"] = "auto",
        logs_root: Path | str | None = None,
        main_logs: bool = True,
        status_logs: bool = True,
        worker_logs: bool = True,
        dev_mode: bool = False,
        timeout: float | None = None,
    ) -> None:
        super().__init__(
            files,
            staging_config=None,
            pipeline_config=pipeline_config,
            num_workers=num_workers,
            logs_root=logs_root,
            main_logs=main_logs,
            status_logs=status_logs,
            worker_logs=worker_logs,
            dev_mode=dev_mode,
            timeout=timeout,
        )

    def stage(
        self, files: list[Path], staging_config: dict[str, Any] | None = None
    ) -> list[StagedEntry]:
        return [StagedEntry(label=str(f), data=f) for f in files]

    @staticmethod
    def process_single(
        entry: StagedEntry, pipeline_config: dict[str, Any] | None = None
    ) -> None:
        print(entry)
