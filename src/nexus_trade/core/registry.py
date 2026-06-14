from __future__ import annotations

import importlib
import threading
from typing import TYPE_CHECKING
from zoneinfo import ZoneInfo

from nexus_trade.core.constants import TIMEFRAME_STRING_MAP, TIMEFRAME_TO_MINUTES, string_to_timeframe
from nexus_trade.core.state import RawStrategyConfig

if TYPE_CHECKING:
    from nexus_trade.config.strategy import BaseStrategyParams, StrategyConfig


class StrategyConfigRegistry:
    """Process-local singleton caching parsed strategy configs and derived objects.

    Thread-safe: via single reentrant lock per instance.
    """

    def __init__(self) -> None:
        self._configs: dict[str, RawStrategyConfig] = {}
        self._full_configs: dict[str, StrategyConfig[BaseStrategyParams]] = {}
        self._tzs: dict[str, ZoneInfo] = {}
        self._timeframe_minutes: dict[str, int] = {}
        self._lock: threading.RLock = threading.RLock()

    def get_strategy_config(self, strategy_name: str) -> StrategyConfig[BaseStrategyParams]:
        with self._lock:
            self.get_config(strategy_name)
            return self._full_configs[strategy_name]

    def get_config(self, strategy_name: str) -> RawStrategyConfig:
        with self._lock:
            if strategy_name not in self._configs:
                module = importlib.import_module(f"nexus_trade.strategies.{strategy_name}.config")
                cfg: StrategyConfig[BaseStrategyParams] = module.get_config()
                self._full_configs[strategy_name] = cfg
                self._configs[strategy_name] = self._parse_config_from_obj(cfg)
            return self._configs[strategy_name]

    def get_tz(self, strategy_name: str) -> ZoneInfo:
        with self._lock:
            if strategy_name not in self._tzs:
                config = self.get_config(strategy_name)
                tz_name: str = config.get("timezone") or "UTC"
                self._tzs[strategy_name] = ZoneInfo(tz_name)
            return self._tzs[strategy_name]

    def get_timeframe_minutes(self, strategy_name: str) -> int:
        with self._lock:
            if strategy_name not in self._timeframe_minutes:
                config = self.get_config(strategy_name)
                tf_str: str = config.get("timeframe")
                tf_key = TIMEFRAME_STRING_MAP.get(tf_str.upper())
                if tf_key is None:
                    raise ValueError(f"Unknown timeframe string: {tf_str!r}")
                self._timeframe_minutes[strategy_name] = TIMEFRAME_TO_MINUTES[tf_key]
            return self._timeframe_minutes[strategy_name]

    def invalidate(self, strategy_name: str) -> None:
        """Remove cached entries for a strategy (e.g. after hot-reload in tests)."""
        with self._lock:
            _ = self._configs.pop(strategy_name, None)
            _ = self._full_configs.pop(strategy_name, None)
            _ = self._tzs.pop(strategy_name, None)
            _ = self._timeframe_minutes.pop(strategy_name, None)

    @staticmethod
    def _parse_config_from_obj(config: StrategyConfig[BaseStrategyParams]) -> RawStrategyConfig:
        params = config.params
        trading_hours = config.trading_hours
        execution = config.execution
        news_filter = config.filters.news
        tf_key = string_to_timeframe(params.timeframe or "M15")

        return RawStrategyConfig(
            symbol=params.symbol,
            timeframe=params.timeframe,
            filter_enabled=trading_hours.enabled if trading_hours else False,
            sessions=list(trading_hours.sessions) if trading_hours else [],
            number_of_bars=params.backcandles + 1,
            magic_number=execution.magic_number,
            deviation=execution.deviation,
            timezone=(trading_hours.timezone if trading_hours else None) or params.timezone,
            news_filter_enabled=news_filter.enabled,
            currencies=list(news_filter.currencies),
            buffer_minutes=news_filter.buffer_minutes,
            timeframe_minutes=int(TIMEFRAME_TO_MINUTES.get(tf_key, 15)) if tf_key else 15,
        )


# Module-level singleton — one instance per process (each strategy process gets its own).
STRATEGY_CONFIG_REGISTRY: StrategyConfigRegistry = StrategyConfigRegistry()
