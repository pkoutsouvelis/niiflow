"""Logging configuration for main and parallel preprocessing workflows.

``LoggingManager`` is a stateless builder: it holds configuration and produces fully-
wired loggers / parallel-logging bundles on demand without storing cross-run state.
Lifetime of parallel resources (queue, listener) is owned by the returned
``ParallelLogging`` context manager.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Callable
import logging
from logging.handlers import QueueListener
from multiprocessing import Queue
import warnings

from niiflow.preproc.utils.file import resolve_path
from niiflow.preproc.workflows.logging_utils import (
    setup_worker_logging,
    ColorFilenameFormatter,
    SafeFieldFilter,
)


@dataclass(frozen=True)
class ParallelLogging:
    """Bundle returned by :meth:`LoggingManager.setup_parallel_logging`.

    Owns the queue, listener, and worker initializer for the duration of a parallel run.
    Use as a context manager to guarantee ``listener.stop()`` on exit (including
    exceptions).
    """

    queue: Queue
    listener: QueueListener
    worker_init_fn: Callable[[], None]

    def stop(self) -> None:
        """Flush remaining records and join the listener thread."""
        self.listener.stop()

    def __enter__(self) -> ParallelLogging:
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()


class LoggingManager:
    """Stateless logging configuration builder.

    Holds logging *settings* (paths, flags) and exposes two factory methods:

    * :meth:`setup_main_logging` — returns a configured ``Logger``.
    * :meth:`setup_status_logging` — returns a ``Logger`` for per-file
      outcome tracking (SUCCESS / WARNING / FAILURE + traceback).
    * :meth:`setup_parallel_logging` — returns a :class:`ParallelLogging`
      context manager that bundles the queue, listener, and worker init fn.

    Args:
        logs_root(Path | str | None):
            Root directory for logs. If set, the logs will be written to files
            under this directory, depending on the other flags:
            - `main_logs`: `logs_root/main.log`
            - `worker_logs`: `logs_root/workers.log`
            - `status_logs`: `logs_root/status.log`
            If None, no logs will be written other than to console.
        main_logs(bool):
            Whether to write main-process logs to file or console.
        worker_logs(bool):
            Whether to write worker-process logs to file or console.
        status_logs(bool):
            Whether to write status logs to file or console.
        dev_mode(bool):
            Whether to enable debug mode.
    """

    def __init__(
        self,
        logs_root: Path | str | None,
        main_logs: bool,
        status_logs: bool,
        worker_logs: bool,
        dev_mode: bool,
    ) -> None:
        if logs_root is not None:
            logs_root = resolve_path(logs_root)

        if any(
            not isinstance(p, bool)
            for p in [main_logs, status_logs, worker_logs, dev_mode]
        ):
            raise ValueError(
                "All `{main_logs, status_logs, worker_logs, dev_mode}` must be booleans"
            )

        if logs_root is None and not all([main_logs, status_logs, worker_logs]):
            raise ValueError(
                "`logs_root` must be set when selecting a subset of "
                "`{main_logs, status_logs, worker_logs}`; without `logs_root`, "
                "all three must be enabled for console logging."
            )

        if logs_root is not None and not any([main_logs, status_logs, worker_logs]):
            warnings.warn("`logs_root` is set but no logs will be written")

        self._logs_root = logs_root
        self._main_logs = main_logs
        self._status_logs = status_logs
        self._worker_logs = worker_logs
        self._dev_mode = dev_mode

    def setup_main_logging(self) -> logging.Logger:
        """Build and return the main-process logger.

        Does not call ``logging.basicConfig`` or mutate the root logger. The returned
        logger owns its own handlers with ``propagate=False``.
        """
        level = logging.DEBUG if self._dev_mode else logging.INFO

        base_format = "%(asctime)s | %(levelname)s | "
        format_tail = (
            "%(processName)s | %(message)s" if self._dev_mode else "%(message)s"
        )
        formatter = logging.Formatter(
            base_format + format_tail, datefmt="%Y-%m-%d %H:%M:%S"
        )

        handlers: list[logging.Handler] = [logging.StreamHandler()]
        if self._main_logs and self._logs_root:
            self._logs_root.mkdir(parents=True, exist_ok=True)
            handlers.append(logging.FileHandler(self._logs_root / "main.log"))
        for h in handlers:
            h.setFormatter(formatter)

        logger = logging.getLogger("niiflow.main")
        logger.setLevel(level)
        logger.handlers.clear()
        for h in handlers:
            logger.addHandler(h)
        logger.propagate = False

        return logger

    def setup_status_logging(self) -> logging.Logger:
        """Build and return the status logger.

        Writes one line per processed file to console and, when ``logs_root`` is
        set, to ``logs_root/status.log``::

            sub-01_T1w.nii.gz | SUCCESS
            sub-02_T1w.nii.gz | FAILURE | RuntimeError: boom\\n  traceback...

        Every record is logged at INFO level; the outcome tag
        (SUCCESS / WARNING / FAILURE) is part of the message, not the log
        level.  The logger is main-process-only (single writer, no
        concurrency concerns).
        """
        logger = logging.getLogger("niiflow.status")
        logger.setLevel(logging.DEBUG if self._dev_mode else logging.INFO)
        logger.handlers.clear()
        logger.propagate = False

        if self._status_logs:
            logger.addHandler(logging.StreamHandler())
        if self._status_logs and self._logs_root:
            self._logs_root.mkdir(parents=True, exist_ok=True)
            handler = logging.FileHandler(self._logs_root / "status.log")
            handler.setFormatter(logging.Formatter("%(message)s"))
            logger.addHandler(handler)

        return logger

    def setup_parallel_logging(self) -> ParallelLogging:
        """Create and start the parallel-logging infrastructure.

        Returns a :class:`ParallelLogging` bundle that should be used as a
        context manager around the ``ProcessPoolExecutor`` block::

            with mgr.setup_parallel_logging() as parallel:
                with ProcessPoolExecutor(
                    max_workers=n,
                    initializer=parallel.worker_init_fn,
                ) as pool:
                    ...

        The listener is already started when this method returns.
        """
        queue: Queue = Queue()

        handlers: list[logging.Handler] = [logging.StreamHandler()]
        if self._worker_logs and self._logs_root:
            self._logs_root.mkdir(parents=True, exist_ok=True)
            handlers.append(logging.FileHandler(self._logs_root / "workers.log"))

        if self._dev_mode:
            stream_fmt = (
                "%(asctime)s | %(levelname)s | %(processName)s | "
                "%(colored_input_file)s | %(message)s"
            )
            file_fmt = (
                "%(asctime)s | %(levelname)s | %(processName)s | "
                "%(input_file)s | %(message)s"
            )
        else:
            stream_fmt = (
                "%(asctime)s | %(levelname)s | %(colored_input_file)s | %(message)s"
            )
            file_fmt = "%(asctime)s | %(levelname)s | %(input_file)s | %(message)s"

        for handler in handlers:
            handler.addFilter(SafeFieldFilter())
            if isinstance(handler, logging.FileHandler):
                handler.setFormatter(logging.Formatter(file_fmt, "%Y-%m-%d %H:%M:%S"))
            else:
                handler.setFormatter(
                    ColorFilenameFormatter(stream_fmt, "%Y-%m-%d %H:%M:%S")
                )

        listener = QueueListener(queue, *handlers)
        listener.start()

        init_fn = partial(
            setup_worker_logging,
            log_queue=queue,
            dev_mode=self._dev_mode,
            worker_logs=self._worker_logs,
        )

        return ParallelLogging(
            queue=queue,
            listener=listener,
            worker_init_fn=init_fn,
        )
