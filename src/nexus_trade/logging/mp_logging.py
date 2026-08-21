from __future__ import annotations

import logging
import logging.handlers
import multiprocessing
import sys
from logging import LogRecord
from multiprocessing.queues import Queue as MPQueue

type LogQueue = MPQueue[LogRecord]

LOG_FORMAT: str = "%(asctime)s - %(levelname)s - %(processName)s - %(message)s"


def create_log_queue() -> LogQueue:
    """Create the shared record queue. Call once, in the orchestrator process."""
    return multiprocessing.Queue(-1)


def setup_logging(log_queue: LogQueue) -> logging.handlers.QueueListener:
    """Configure orchestrator-process logging and start the shared listener."""
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    listener = logging.handlers.QueueListener(log_queue, stream_handler, respect_handler_level=True)
    listener.start()
    logging.basicConfig(level=logging.INFO, handlers=[stream_handler], force=True)
    return listener


def configure_worker_logging(log_queue: LogQueue, level: int = logging.INFO) -> None:
    """Route this process's root logger through the shared queue only."""
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(level)
    root.addHandler(logging.handlers.QueueHandler(log_queue))
