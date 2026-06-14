"""AsyncTradeLogger - asynchronous wrapper for TradeLogger writes."""

import logging
import threading
import time
from queue import Empty, Full, Queue

from nexus_trade.core.state import ReconciledTrade
from nexus_trade.logging.logger import CloseData, FillData, PartialCloseData, TradeLogger

logger = logging.getLogger(__name__)
type LogPayload = FillData | CloseData | PartialCloseData


class AsyncTradeLogger:
    """Asynchronous wrapper for TradeLogger with background thread writer."""

    def __init__(self, trade_logger: TradeLogger, max_queue_size: int = 100) -> None:
        self._logger: TradeLogger = trade_logger
        self._queue: Queue[tuple[str, tuple[LogPayload], dict[str, object]]] = Queue(maxsize=max_queue_size)
        self._shutdown_flag: threading.Event = threading.Event()
        self._state_lock: threading.Lock = threading.Lock()
        self._dropped_count: int = 0
        self._last_drop_warning_at: float = 0.0
        self._drop_warning_interval_seconds: float = 5.0
        self._writer_thread: threading.Thread = threading.Thread(
            target=self._writer_loop, name=f"AsyncLogger-{trade_logger.strategy_name}", daemon=False
        )
        self._writer_thread.start()
        logger.debug(f"AsyncLogStart strat={trade_logger.strategy_name} | q={max_queue_size}")

    def _writer_loop(self) -> None:
        """Background thread main loop - continuously dequeues and writes logs."""
        while True:
            if self._shutdown_flag.is_set() and self._queue.empty():
                break

            try:
                item = self._queue.get(timeout=0.25)
            except Empty:
                continue

            method_name, args, kwargs = item
            try:
                getattr(self._logger, method_name)(*args, **kwargs)
            except Exception as e:
                logger.error(
                    f"AsyncLogWriteFail strat={self._logger.strategy_name} | m={method_name} | err={e}",
                    exc_info=True,
                )
            finally:
                self._queue.task_done()
        logger.debug(f"AsyncLogThreadExit strat={self._logger.strategy_name}")

    def log_fill(self, data: FillData) -> None:
        self._enqueue("log_fill", data)

    def log_close(self, data: CloseData) -> None:
        self._enqueue("log_close", data)

    def log_partial_close(self, data: PartialCloseData) -> None:
        self._enqueue("log_partial_close", data)

    def get_open_trades_by_ticket_last_three(self, tickets: list[int]) -> dict[int, ReconciledTrade]:
        """Provide synchronous passthrough for startup reconciliation reads."""
        return self._logger.get_open_trades_by_ticket_last_three(tickets)

    def _enqueue(self, method_name: str, data: LogPayload) -> None:
        with self._state_lock:
            if self._shutdown_flag.is_set():
                logger.warning(f"AsyncLogDrop strat={self._logger.strategy_name} | m={method_name} | reason=shutdown")
                return

            try:
                self._queue.put_nowait((method_name, (data,), {}))
            except Full:
                self._dropped_count += 1
                now = time.monotonic()
                if now - self._last_drop_warning_at >= self._drop_warning_interval_seconds:
                    self._last_drop_warning_at = now
                    logger.error(
                        f"AsyncLogDrop strat={self._logger.strategy_name} | m={method_name} | "
                        f"reason=queue_full | dropped={self._dropped_count}"
                    )
            except Exception as e:
                logger.error(f"AsyncLogQueueErr strat={self._logger.strategy_name} | m={method_name} | err={e}")

    def shutdown(self, timeout: float = 10.0) -> None:
        logger.debug(f"AsyncLogShutdownStart strat={self._logger.strategy_name}")
        timeout = max(0.0, timeout)
        with self._state_lock:
            self._shutdown_flag.set()

        self._writer_thread.join(timeout=timeout)
        if self._writer_thread.is_alive():
            logger.warning(
                f"AsyncLogShutdownWarn strat={self._logger.strategy_name} | timeout={timeout:.1f}s | lost=possible"
            )
        else:
            logger.debug(f"AsyncLogShutdownDone strat={self._logger.strategy_name} | dropped={self._dropped_count}")
