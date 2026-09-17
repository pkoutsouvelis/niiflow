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
from typing import Any, Callable
import logging
from logging.handlers import QueueListener
import multiprocessing
from multiprocessing.context import BaseContext
import queue as queue_module
import time
import warnings

from niiflow.preproc.utils.file import resolve_path
from niiflow.preproc.workflows.logging_utils import (
    setup_worker_logging,
    ColorFilenameFormatter,
    SafeFieldFilter,
)

LOG_QUEUE_MAXSIZE = 10_000
LISTENER_STOP_TIMEOUT = 5.0


class FlushingHandler(logging.Handler):
    """Call ``flush`` on the wrapped handler after every emitted record."""

    def __init__(self, handler: logging.Handler) -> None:
        super().__init__(level=logging.NOTSET)
        self._handler = handler

    def emit(self, record: logging.LogRecord) -> None:
        self._handler.emit(record)
        self._handler.flush()


class BoundedQueueListener(QueueListener):
    """Queue listener whose shutdown cannot wait forever for a full queue."""

    def __init__(self, queue: Any, *handlers: logging.Handler) -> None:
        super().__init__(queue, *handlers)
        self.stopped_cleanly = False

    def stop(self, timeout: float = LISTENER_STOP_TIMEOUT) -> None:
        """Request listener shutdown and wait at most ``timeout`` seconds."""
        thread = self._thread
        if thread is None:
            self.stopped_cleanly = True
            return

        deadline = time.monotonic() + timeout
        enqueued = False
        while thread.is_alive():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                self.queue.put(  # type: ignore[attr-defined]
                    self._sentinel,  # type: ignore[attr-defined]
                    block=True,
                    timeout=min(0.1, remaining),
                )
                enqueued = True
                break
            except queue_module.Full:
                continue
            except (OSError, EOFError, ValueError):
                break

        if enqueued:
            thread.join(timeout=max(0.0, deadline - time.monotonic()))
        self.stopped_cleanly = not thread.is_alive()
        if self.stopped_cleanly:
            self._thread = None


@dataclass
class ParallelLogging:
    """Bundle returned by :meth:`LoggingManager.setup_parallel_logging`.

    Owns the queue, listener, and worker initializer for the duration of a parallel run.
    Use as a context manager to guarantee ``listener.stop()`` on exit (including
    exceptions).
    """

    queue: Any
    listener: BoundedQueueListener
    worker_init_fn: Callable[[], None]
    mp_context: BaseContext
    suppressed_count: Any
    diagnostic_logger: logging.Logger
    _stopped: bool = False

    def stop(self) -> None:
        """Stop, flush, and release all parallel-logging resources once."""
        if self._stopped:
            return
        self._stopped = True

        listener_stopped = False
        try:
            self.listener.stop()
            listener_stopped = self.listener.stopped_cleanly
            if not listener_stopped:
                self.diagnostic_logger.warning(
                    "Worker log listener did not stop within %.1fs; "
                    "abandoning queued records",
                    LISTENER_STOP_TIMEOUT,
                )
        except Exception:
            self.diagnostic_logger.exception("Failed to stop worker log listener")

        try:
            suppressed = int(self.suppressed_count.value)
        except Exception:
            suppressed = 0
        if suppressed:
            self.diagnostic_logger.warning(
                "%d worker log records suppressed due to logging backpressure",
                suppressed,
            )

        if listener_stopped:
            for handler in self.listener.handlers:
                target = (
                    handler._handler
                    if isinstance(handler, FlushingHandler)
                    else handler
                )
                try:
                    target.flush()
                finally:
                    target.close()

        try:
            self.queue.close()
        finally:
            if listener_stopped:
                self.queue.join_thread()
            else:
                self.queue.cancel_join_thread()

    def __enter__(self) -> ParallelLogging:
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()


class LoggingManager:
    """Stateless logging configuration builder.

    Holds logging *settings* (paths, flags) and exposes two factory methods:

    * :meth:`setup_main_logging` — returns a configured ``Logger``.
    * :meth:`setup_status_logging` — returns a ``Logger`` for compact per-file
      outcome tracking (SUCCESS / WARNING / FAILURE).
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
        mp_context(BaseContext | None):
            Multiprocessing context to use for the parallel logging. If None, the
            default context (spawn) will be used.
    """

    def __init__(
        self,
        logs_root: Path | str | None,
        main_logs: bool,
        status_logs: bool,
        worker_logs: bool,
        dev_mode: bool,
        mp_context: BaseContext | None = None,
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
        self._mp_context = (
            mp_context
            if mp_context is not None
            else multiprocessing.get_context("spawn")
        )

    @property
    def mp_context(self) -> BaseContext:
        """Multiprocessing context shared by queues, managers, and workers."""
        return self._mp_context

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
            sub-02_T1w.nii.gz | FAILURE | RuntimeError: boom

        Every record is logged at INFO level; the outcome tag
        (SUCCESS / WARNING / FAILURE) is part of the message, not the log
        level. Full tracebacks belong in the diagnostic log. The logger is
        main-process-only (single writer, no concurrency concerns).
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
                    mp_context=parallel.mp_context,
                    initializer=parallel.worker_init_fn,
                ) as pool:
                    ...

        The listener is already started when this method returns. Its queue is
        bounded so slow diagnostic storage cannot create unbounded worker backlog.
        """
        queue = self._mp_context.Queue(maxsize=LOG_QUEUE_MAXSIZE)
        # A lock-free shared counter cannot delay workers. Concurrent increments may
        # under-count slightly, which is acceptable for a diagnostic summary.
        suppressed_count = self._mp_context.Value("L", 0, lock=False)

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

        listener_handlers: list[logging.Handler] = []
        for handler in handlers:
            handler.addFilter(SafeFieldFilter())
            if isinstance(handler, logging.FileHandler):
                handler.setFormatter(logging.Formatter(file_fmt, "%Y-%m-%d %H:%M:%S"))
                listener_handlers.append(handler)
            else:
                handler.setFormatter(
                    ColorFilenameFormatter(stream_fmt, "%Y-%m-%d %H:%M:%S")
                )
                # Flush console only; per-record flush on GPFS files stalls workers.
                listener_handlers.append(FlushingHandler(handler))

        listener = BoundedQueueListener(queue, *listener_handlers)
        listener.start()

        init_fn = partial(
            setup_worker_logging,
            log_queue=queue,
            dev_mode=self._dev_mode,
            worker_logs=self._worker_logs,
            suppressed_count=suppressed_count,
        )

        return ParallelLogging(
            queue=queue,
            listener=listener,
            worker_init_fn=init_fn,
            mp_context=self._mp_context,
            suppressed_count=suppressed_count,
            diagnostic_logger=logging.getLogger("niiflow.main"),
        )
