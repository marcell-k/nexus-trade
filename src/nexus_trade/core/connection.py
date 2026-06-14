import logging
import time
from enum import Enum

import MetaTrader5 as mt5

from nexus_trade.config.account import AccountConfig
from nexus_trade.core.protocols import AccountInfo

logger = logging.getLogger(__name__)


class ConnectionState(Enum):
    """MT5 connection states for reconnection management."""

    CONNECTED = "connected"
    DISCONNECTED = "disconnected"


class MT5Connection:
    MAX_BACKOFF_SECONDS: int = 30

    def __init__(
        self, config: AccountConfig, max_reconnection_attempts: int = 5, connection_check_ttl: int = 60
    ) -> None:
        self.config: AccountConfig = config
        self.state: ConnectionState = ConnectionState.DISCONNECTED
        self.reconnection_attempts: int = 0
        self.max_reconnection_attempts: int = max_reconnection_attempts
        self.connection_check_ttl: int = connection_check_ttl
        self._cached_connection_state: bool | None = None
        self._cache_timestamp_monotonic: float = 0.0

    def connect(self, max_retries: int = 3, retry_delay: int = 1) -> bool:
        """Establish MT5 connection with retry logic."""
        if self.is_connected(use_cache=False):
            logger.debug("ConnSkip state=already_connected")
            return True

        for attempt in range(max_retries):
            if attempt > 0:
                mt5.shutdown()

            if not mt5.initialize(
                login=self.config.login,
                password=self.config.password,
                server=self.config.server,
                path=self.config.path,
            ):
                logger.warning(f"ConnInitFail n={attempt + 1}/{max_retries} | err={mt5.last_error()}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (2**attempt))
                continue

            account_info = mt5.account_info()
            if not self._validate_account(account_info):
                mt5.shutdown()
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (2**attempt))
                continue

            logger.debug("ConnOK")
            self._set_connected()
            return True

        logger.error(f"ConnFail tries={max_retries}")
        self._set_disconnected()
        return False

    def reconnect(self) -> bool:
        """Attempt reconnection with exponential backoff."""
        self.reconnection_attempts += 1

        if self.reconnection_attempts > self.max_reconnection_attempts:
            logger.critical(f"ReconnFail tries={self.max_reconnection_attempts} | action=manual")
            self._set_disconnected()
            return False

        backoff_seconds = min(2 ** (self.reconnection_attempts - 1), self.MAX_BACKOFF_SECONDS)
        logger.warning(
            f"ReconnStart n={self.reconnection_attempts}/{self.max_reconnection_attempts} | backoff={backoff_seconds}s"
        )
        time.sleep(backoff_seconds)

        try:
            connected = self.connect(max_retries=3, retry_delay=10)
        except Exception:
            self._set_disconnected()
            raise

        if connected:
            logger.debug(f"ReconnOK n={self.reconnection_attempts}")
            return True

        logger.error(f"ReconnFail n={self.reconnection_attempts}")
        self._set_disconnected()
        return False

    def disconnect(self) -> None:
        """Graceful disconnection."""
        try:
            mt5.shutdown()
            logger.debug("ConnClosed")
            self._set_disconnected()
        except (OSError, RuntimeError) as e:
            logger.error(f"ConnCloseErr err={e}", exc_info=True)
            self._set_disconnected()
        except Exception as e:
            logger.critical(f"ConnCloseFatal err={e}", exc_info=True)
            self._set_disconnected()
            raise

    def is_connected(self, use_cache: bool = True) -> bool:
        """Check if MT5 terminal is alive and authenticated. Uses cached result within TTL."""
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

        logger.debug("ConnCheckOK")
        self._set_connected()
        return True

    def ensure_connected(self) -> bool:
        """Ensure MT5 connection is active, attempt reconnection if disconnected."""
        if self.is_connected():
            return True

        logger.warning("ConnLost action=reconnect")
        return self.reconnect()

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
        """Check if cached connection state is still valid."""
        if self._cached_connection_state is None:
            return False
        cache_age = time.monotonic() - self._cache_timestamp_monotonic
        return cache_age < self.connection_check_ttl

    def _set_connected(self) -> None:
        """Update state to connected and reset connection."""
        self.state = ConnectionState.CONNECTED
        self.reconnection_attempts = 0
        self._cached_connection_state = True
        self._cache_timestamp_monotonic = time.monotonic()

    def _set_disconnected(self) -> None:
        """Update state to disconnected and invalidate cache."""
        self.state = ConnectionState.DISCONNECTED
        self.invalidate_cache()

    def invalidate_cache(self) -> None:
        """Manually invalidate the connection state cache."""
        self._cached_connection_state = None
        self._cache_timestamp_monotonic = 0.0
