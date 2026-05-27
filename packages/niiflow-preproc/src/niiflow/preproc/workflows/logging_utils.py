"""Logging utilities for worker processes and per-task filename context."""

from __future__ import annotations

__all__ = [
    "setup_worker_logging",
    "set_input_file_context",
    "ColorFilenameFormatter",
    "SafeFieldFilter",
]

import contextvars
import logging
from logging.handlers import QueueHandler
from multiprocessing import Queue

BLUE = "\033[94m"
RESET = "\033[0m"

_input_file_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "input_file", default="-"
)


def setup_worker_logging(
    log_queue: Queue,
    dev_mode: bool,
    worker_logs: bool,
) -> None:
    """Configure the worker-process root logger.

    Installs a :class:`QueueHandler` that forwards every record to the main-process
    :class:`QueueListener`, and a :class:`ContextVarFilter` that stamps each record with
    the current ``input_file`` context.

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
    handler = QueueHandler(log_queue)
    handler.addFilter(ContextVarFilter())
    root.addHandler(handler)


def set_input_file_context(input_file: str) -> contextvars.Token[str]:
    """Set the per-task ``input_file`` and return a reset token.

    Typical usage inside a per-file task function::

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
