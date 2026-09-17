"""Regression tests for bounded parallel worker logging."""
# !Should follow the same pattern as the other tests; e.g., dummy/test classes first, 
# then helpers/fixtures, then use classes when testing a specific object; e.g., BlockingQueueHandler, 
# BoundedQueueListener, ParallelLogging stop, etc. 

from __future__ import annotations

import logging
import multiprocessing
from pathlib import Path
from queue import Queue
from types import SimpleNamespace
import time

from niiflow.preproc.workflows.logging_manager import (
    BoundedQueueListener,
    LOG_QUEUE_MAXSIZE,
    LoggingManager,
)
from niiflow.preproc.workflows.logging_utils import (
    BlockingQueueHandler,
    setup_worker_logging,
)


class _SlowHandler(logging.Handler):
    """Collect records slowly enough to force bounded-queue saturation."""

    def __init__(self, delay: float) -> None:
        super().__init__()
        self.delay = delay
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        time.sleep(self.delay)
        self.records.append(record)


def _emit_many_worker_records(
    log_queue: object,
    suppressed_count: object,
    record_count: int,
) -> None:
    setup_worker_logging(
        log_queue,  # type: ignore[arg-type]
        dev_mode=False,
        worker_logs=True,
        enqueue_timeout=0.001,
        suppressed_count=suppressed_count,
    )
    logger = logging.getLogger("niiflow.test.saturation")
    for index in range(record_count):
        logger.warning("worker record %d", index)


def test_queue_handler_drops_quickly_without_handle_error_storm() -> None:
    log_queue: Queue[object] = Queue(maxsize=1)
    log_queue.put(object())
    suppressed = SimpleNamespace(value=0)
    handler = BlockingQueueHandler(
        log_queue,
        enqueue_timeout=0.01,
        suppressed_count=suppressed,
    )
    record = logging.LogRecord(
        "worker",
        logging.INFO,
        __file__,
        1,
        "record",
        (),
        None,
    )

    started = time.monotonic()
    handler.emit(record)
    elapsed = time.monotonic() - started

    assert elapsed < 0.2
    assert handler.local_suppressed_count == 1
    assert suppressed.value == 1


def test_saturated_spawn_worker_continues_and_listener_stops() -> None:
    mp_context = multiprocessing.get_context("spawn")
    log_queue = mp_context.Queue(maxsize=2)
    suppressed = mp_context.Value("L", 0, lock=False)
    slow_handler = _SlowHandler(delay=0.02)
    listener = BoundedQueueListener(log_queue, slow_handler)
    listener.start()
    process = mp_context.Process(
        target=_emit_many_worker_records,
        args=(log_queue, suppressed, 250),
    )

    started = time.monotonic()
    process.start()
    process.join(timeout=5)
    elapsed = time.monotonic() - started

    assert not process.is_alive()
    assert process.exitcode == 0
    assert elapsed < 5
    assert suppressed.value > 0
    listener.stop(timeout=3)
    assert listener.stopped_cleanly
    log_queue.close()
    log_queue.join_thread()


def test_parallel_logging_uses_spawn_bounded_queue_and_stops_twice(
    tmp_path: Path,
) -> None:
    mp_context = multiprocessing.get_context("spawn")
    manager = LoggingManager(
        logs_root=tmp_path / "logs",
        main_logs=True,
        status_logs=True,
        worker_logs=True,
        dev_mode=False,
        mp_context=mp_context,
    )
    manager.setup_main_logging()
    parallel = manager.setup_parallel_logging()

    assert parallel.mp_context is mp_context
    assert parallel.queue._maxsize == LOG_QUEUE_MAXSIZE

    parallel.suppressed_count.value = 1_834
    parallel.stop()
    parallel.stop()

    main_log = (tmp_path / "logs" / "main.log").read_text(encoding="utf-8")
    assert "1834 worker log records suppressed due to logging backpressure" in main_log
