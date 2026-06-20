import logging
import time
from enum import Enum

import MetaTrader5 as mt5
from MetaTrader5 import AccountInfo

# Assuming this exists in your project
from nexus_trade.config.account import AccountConfig
from nexus_trade.config.timings import SYSTEM_TIMINGS

logger = logging.getLogger(__name__)


class ConnectionState(Enum):
    """MT5 connection states for reconnection management."""

    CONNECTED = "connected"
    DISCONNECTED = "disconnected"


class MT5Connection:
    # Define your backoff seconds as a class attribute to fix the missing variable bug

    def __init__(self, config: AccountConfig, connection_check_ttl: int = 60) -> None:
        self.config: AccountConfig = config
        self.state: ConnectionState = ConnectionState.DISCONNECTED
        self.connection_check_ttl: int = connection_check_ttl
        self._cached_connection_state: bool | None = None
        self._cache_timestamp_monotonic: float = 0.0

    def connect(self, max_retries: int = 5) -> bool:
        """Establish MT5 connection with exponential backoff retry logic."""
        if self.is_connected(use_cache=False):
            logger.debug("ConnSkip state=already_connected")
            return True

        for attempt in range(max_retries):
            if attempt > 0:
                mt5.shutdown()
                # Exponential backoff based on the attempt number
                backoff = SYSTEM_TIMINGS.connect_backoff_seconds[
                    min(attempt - 1, len(SYSTEM_TIMINGS.connect_backoff_seconds) - 1)
                ]
                logger.warning(f"Reconnecting... attempt={attempt}/{max_retries} | backoff={backoff}s")
                time.sleep(backoff)

            if not mt5.initialize(
                login=self.config.login,
                password=self.config.password,
                server=self.config.server,
                path=self.config.path,
            ):
                logger.warning(f"ConnInitFail n={attempt + 1}/{max_retries} | err={mt5.last_error()}")
                continue

            account_info = mt5.account_info()
            if not self._validate_account(account_info):
                continue

            logger.debug("ConnOK")
            self._set_connected()
            return True

        logger.error(f"ConnFail tries={max_retries}")
        self._set_disconnected()
        return False

    def disconnect(self) -> None:
        """Graceful disconnection."""
        try:
            mt5.shutdown()
            logger.debug("ConnClosed")
        except Exception as e:
            logger.error(f"ConnCloseErr err={e}", exc_info=True)
        finally:
            self._set_disconnected()

    def is_connected(self, use_cache: bool = True) -> bool:
        """Check if MT5 terminal is alive and authenticated."""
        if use_cache and self._is_cache_valid():
            return bool(self._cached_connection_state)

        terminal_info = mt5.terminal_info()
        if terminal_info is None or not terminal_info.connected:
            self._set_disconnected()
            return False

        account_info = mt5.account_info()
        if not self._validate_account(account_info):
            self._set_disconnected()
            return False

        self._set_connected()
        return True

    def ensure_connected(self) -> bool:
        """Ensure MT5 connection is active, attempt connection if disconnected."""
        if self.is_connected():
            return True

        logger.warning("ConnLost action=reconnect")
        return self.connect()

    def _validate_account(self, account_info: AccountInfo | None) -> bool:
        if account_info is None:
            logger.error("ConnAcctErr reason=account_info_unavailable")
            return False
        if account_info.login != self.config.login:
            logger.error("ConnAcctErr reason=account_mismatch")
            return False
        if account_info.server != self.config.server:
            logger.error("ConnAcctErr reason=server_mismatch")
            return False
        return True

    def _is_cache_valid(self) -> bool:
        if self._cached_connection_state is None:
            return False
        return (time.monotonic() - self._cache_timestamp_monotonic) < self.connection_check_ttl

    def _set_connected(self) -> None:
        self.state = ConnectionState.CONNECTED
        self._cached_connection_state = True
        self._cache_timestamp_monotonic = time.monotonic()

    def _set_disconnected(self) -> None:
        self.state = ConnectionState.DISCONNECTED
        self.invalidate_cache()

    def invalidate_cache(self) -> None:
        self._cached_connection_state = None
        self._cache_timestamp_monotonic = 0.0
