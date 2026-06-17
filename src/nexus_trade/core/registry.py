from __future__ import annotations

import importlib
import threading
from typing import TYPE_CHECKING, cast
from zoneinfo import ZoneInfo

from nexus_trade.core.constants import TIMEFRAME_STRING_MAP, TIMEFRAME_TO_MINUTES

if TYPE_CHECKING:
    from nexus_trade.config.strategy import BaseStrategyParams, StrategyConfig
    from nexus_trade.core.protocols import ConfigModule


class StrategyConfigRegistry:
    """Process-local singleton caching parsed strategy configs and derived objects.

    Thread-safe: via single reentrant lock per instance.
    """

    def __init__(self) -> None:
        self._full_configs: dict[str, StrategyConfig[BaseStrategyParams]] = {}
        self._tzs: dict[str, ZoneInfo] = {}
        self._timeframe_minutes: dict[str, int] = {}
        self._lock: threading.RLock = threading.RLock()

    def get_strategy_config(self, strategy_name: str) -> StrategyConfig[BaseStrategyParams]:
        with self._lock:
            if strategy_name not in self._full_configs:
                module = cast(
                    "ConfigModule",
                    cast("object", importlib.import_module(f"nexus_trade.strategies.{strategy_name}.config")),
                )
                self._full_configs[strategy_name] = module.get_config()
            return self._full_configs[strategy_name]

    def get_tz(self, strategy_name: str) -> ZoneInfo:
        with self._lock:
            if strategy_name not in self._tzs:
                cfg = self.get_strategy_config(strategy_name)
                th = cfg.trading_hours
                tz_name: str = (th.timezone if th is not None else None) or cfg.params.timezone or "UTC"

                self._tzs[strategy_name] = ZoneInfo(tz_name)
            return self._tzs[strategy_name]

    def get_timeframe_minutes(self, strategy_name: str) -> int:
        with self._lock:
            if strategy_name not in self._timeframe_minutes:
                cfg = self.get_strategy_config(strategy_name)
                tf_str: str = cfg.params.timeframe
                tf_key = TIMEFRAME_STRING_MAP.get(tf_str.upper())
                if tf_key is None:
                    raise ValueError(f"Unknown timeframe string: {tf_str!r}")
                self._timeframe_minutes[strategy_name] = TIMEFRAME_TO_MINUTES[tf_key]
            return self._timeframe_minutes[strategy_name]

    def invalidate(self, strategy_name: str) -> None:
        """Remove cached entries for a strategy (e.g. after hot-reload in tests)."""
        with self._lock:
            _ = self._full_configs.pop(strategy_name, None)
            _ = self._tzs.pop(strategy_name, None)
            _ = self._timeframe_minutes.pop(strategy_name, None)


# Module-level singleton — one instance per process (each strategy process gets its own).
STRATEGY_CONFIG_REGISTRY: StrategyConfigRegistry = StrategyConfigRegistry()
