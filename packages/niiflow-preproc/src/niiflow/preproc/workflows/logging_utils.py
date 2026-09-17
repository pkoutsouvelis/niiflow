"""Logging utilities for worker processes and per-task filename context."""

from __future__ import annotations

__all__ = [
    "BlockingQueueHandler",
    "setup_worker_logging",
    "set_input_file_context",
    "ColorFilenameFormatter",
    "SafeFieldFilter",
]

import contextvars
import logging
from logging.handlers import QueueHandler
from typing import Any

BLUE = "\033[94m"
RESET = "\033[0m"

DEFAULT_ENQUEUE_TIMEOUT = 0.25

_input_file_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "input_file", default="-"
)


class BlockingQueueHandler(QueueHandler):
    """Forward worker records without allowing logging to stall processing.

    A bounded wait absorbs short bursts. If the main-process listener remains
    backlogged, the record is dropped and a best-effort shared counter is incremented.
    Logging transport errors never recurse through :meth:`handleError`.
    """

    def __init__(
        self,
        queue: Any,
        *,
        enqueue_timeout: float = DEFAULT_ENQUEUE_TIMEOUT,
        suppressed_count: Any | None = None,
    ) -> None:
        if enqueue_timeout < 0:
            raise ValueError("`enqueue_timeout` must be non-negative")
        super().__init__(queue)
        self.enqueue_timeout = float(enqueue_timeout)
        self.suppressed_count = suppressed_count
        self.local_suppressed_count = 0

    def enqueue(self, record: logging.LogRecord) -> None:
        try:
            self.queue.put(  # type: ignore[attr-defined]
                record, block=True, timeout=self.enqueue_timeout
            )
        except Exception:
            self._record_suppressed()

    def handleError(self, record: logging.LogRecord) -> None:
        """Drop records that fail preparation without recursive diagnostics."""
        self._record_suppressed()

    def _record_suppressed(self) -> None:
        """Increment local and shared best-effort suppression counts."""
        self.local_suppressed_count += 1
        if self.suppressed_count is None:
            return
        try:
            # Deliberately lock-free: an approximate count is preferable to making
            # worker liveness depend on acquiring another multiprocessing lock.
            self.suppressed_count.value += 1
        except Exception:
            pass


def setup_worker_logging(
    log_queue: Any,
    dev_mode: bool,
    worker_logs: bool,
    enqueue_timeout: float = DEFAULT_ENQUEUE_TIMEOUT,
    suppressed_count: Any | None = None,
) -> None:
    """Configure the worker-process root logger.

    Installs a :class:`BlockingQueueHandler` that forwards every record to the main-
    process :class:`logging.handlers.QueueListener`, and a :class:`ContextVarFilter`
    that stamps each record with the current ``input_file`` context.

    Called once per worker by the pool's ``initializer``.
    """
    if not log_queue:
        raise RuntimeError(
            "Log queue not initialized. "
            "Call LoggingManager.setup_parallel_logging() in the main process first."
        )

    if dev_mode:
        level = logging.DEBUG
    elif worker_logs:
        level = logging.INFO
    else:
        level = logging.WARNING

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()
    handler = BlockingQueueHandler(
        log_queue,
        enqueue_timeout=enqueue_timeout,
        suppressed_count=suppressed_count,
    )
    handler.addFilter(ContextVarFilter())
    root.addHandler(handler)


def set_input_file_context(input_file: str) -> contextvars.Token[str]:
    """Set the per-task ``input_file`` and return a reset token.

    Typical usage inside an active-file-specific task function
    (requires a logger configured with a
    :class:`~niiflow.preproc.workflows.logging_utils.ContextVarFilter` filter)::

        token = set_input_file_context(str(path))
        try:
            ...  # every log record now carries input_file
        finally:
            _input_file_var.reset(token)
    """
    return _input_file_var.set(input_file)


def reset_input_file_context(token: contextvars.Token[str]) -> None:
    """Reset the per-task ``input_file`` context to the previous value."""
    _input_file_var.reset(token)


# ---------------------------------------------------------------------------
# Filters & formatters
# ---------------------------------------------------------------------------


class ContextVarFilter(logging.Filter):
    """Stamps every record with the ``input_file`` from the current context."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.input_file = _input_file_var.get()  # type: ignore[attr-defined]
        return True


class ColorFilenameFormatter(logging.Formatter):
    """Adds ``colored_input_file`` for terminal display."""

    def format(self, record: logging.LogRecord) -> str:
        input_file = getattr(record, "input_file", "")
        record.colored_input_file = (  # type: ignore[attr-defined]
            f"{BLUE}{input_file}{RESET}" if input_file else ""
        )
        return super().format(record)


class SafeFieldFilter(logging.Filter):
    """Default ``input_file`` to ``"-"`` for records that lack one.

    Applied on the listener side so main-process records (which never pass through a
    worker) still render cleanly in the parallel-logging format.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "input_file"):
            record.input_file = "-"  # type: ignore[attr-defined]
        return True
