from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo

import MetaTrader5 as mt5
import numpy as np
import pandas as pd

from nexus_trade.core.constants import TIMEFRAME_TO_MINUTES, TimeFrame, string_to_timeframe
from nexus_trade.core.models import Tick
from nexus_trade.core.registry import STRATEGY_CONFIG_REGISTRY
from nexus_trade.core.types import RawStrategyConfig


@dataclass(slots=True)
class _BarCacheEntry:
    bar_time: pd.Timestamp
    timeframe_minutes: int
    next_bar_complete_at: pd.Timestamp
    cache_seeded: bool = False


class DataHandler:
    def __init__(self, broker_tz: ZoneInfo) -> None:
        self.broker_tz: ZoneInfo = broker_tz
        self._latest_bar_cache: dict[tuple[str, int], _BarCacheEntry] = {}
        self._bar_rolling_windows: dict[tuple[str, int], pd.DataFrame | None] = {}

    def _get_cached_window(self, cache_key: tuple[str, int]) -> pd.DataFrame | None:
        """Return cached rolling DataFrame, or None on miss."""
        return self._bar_rolling_windows.get(cache_key)

    def _get_cache_capacity(self, strategy_config: RawStrategyConfig) -> int:
        window_size = strategy_config["number_of_bars"]
        if strategy_config.get("filter_enabled", False):
            return max(window_size, int(window_size * 1.3))
        return window_size

    def _return_cached_window(
        self, cache_key: tuple[str, int], strategy_config: RawStrategyConfig
    ) -> pd.DataFrame | None:
        """Materialize cached window, apply session filter, and trim to size."""
        df = self._get_cached_window(cache_key)
        if df is None or len(df) == 0:
            return df
        if strategy_config.get("filter_enabled", False):
            df = self._filter_session_hours(df, strategy_config)
        return df.tail(strategy_config["number_of_bars"])

    def _update_cache_metadata(
        self, cache_key: tuple[str, int], bar_time_broker: pd.Timestamp, timeframe_minutes: int
    ) -> None:
        """Centralized cache metadata update to eliminate duplication."""
        tolerance_seconds = 2
        next_bar_complete_at = bar_time_broker + pd.Timedelta(minutes=timeframe_minutes, seconds=tolerance_seconds)

        self._latest_bar_cache[cache_key] = _BarCacheEntry(
            bar_time=bar_time_broker,
            timeframe_minutes=timeframe_minutes,
            next_bar_complete_at=next_bar_complete_at,
            cache_seeded=True,
        )

    def _create_tz_aware_dataframe(self, rates: np.ndarray, strategy_tz: ZoneInfo) -> pd.DataFrame:
        """Create timezone-aware DataFrame from MT5 rates array."""
        timestamps = pd.to_datetime(rates["time"], unit="s")
        tz_aware_index = timestamps.tz_localize(self.broker_tz).tz_convert(strategy_tz)
        tz_aware_index.name = "Time"

        df = pd.DataFrame(
            {
                "Open": rates["open"],
                "High": rates["high"],
                "Low": rates["low"],
                "Close": rates["close"],
                "Volume": rates["tick_volume"],
                "spread": rates["spread"],
            },
            index=tz_aware_index,
        )
        # Ensure chronological order in case MT5 returns newest-first.
        return df.sort_index()

    def _check_cache_state(self, cached_metadata: _BarCacheEntry, now_broker: datetime) -> tuple[bool, float]:
        """Return (should_skip, bars_elapsed). should_skip=True means current bar still forming."""
        if now_broker < cached_metadata.next_bar_complete_at:
            return True, 0.0
        bar_close_time = cached_metadata.bar_time + pd.Timedelta(minutes=cached_metadata.timeframe_minutes)
        bars_elapsed = (now_broker - bar_close_time).total_seconds() / 60.0 / cached_metadata.timeframe_minutes
        return False, bars_elapsed

    def get_latest_bars(self, strategy_name: str) -> pd.DataFrame | None:
        """
        Fetch latest bars with predictive caching to eliminate unnecessary MT5 API calls.

        Cache Strategy:
        1. Predictive skip: If current bar incomplete, return cached window without MT5 call
        2. Same bar (cache hit): Bar complete but no new bar yet
        3. New bar (incremental): Fetch 1 bar from MT5, append to window
        4. Gap detected: Full refresh all bars
        """
        strategy_config = STRATEGY_CONFIG_REGISTRY.get_config(strategy_name)
        symbol = strategy_config["symbol"]
        timeframe_mt5 = string_to_timeframe(strategy_config["timeframe"])
        cache_key: tuple[str, int] = (symbol, int(timeframe_mt5))
        strategy_tz = STRATEGY_CONFIG_REGISTRY.get_tz(strategy_name)
        now_broker = datetime.now(self.broker_tz)
        cached_metadata = self._latest_bar_cache.get(cache_key)

        # Predictive skip: bar still forming
        if cached_metadata is not None and self._should_skip_mt5_call(cached_metadata, now_broker):
            cached = self._return_cached_window(cache_key, strategy_config)
            if cached is not None and len(cached) > 0:
                return cached

        # Determine cache strategy: same bar, incremental, or full refresh
        if not (cached_metadata and cached_metadata.cache_seeded):
            return self._full_refresh_bars(symbol, strategy_config, cache_key, strategy_tz)

        should_skip, bars_elapsed = self._check_cache_state(cached_metadata, now_broker)

        if should_skip or bars_elapsed < 1.0:
            cached = self._return_cached_window(cache_key, strategy_config)
            if cached is not None:
                return cached
            return self._full_refresh_bars(symbol, strategy_config, cache_key, strategy_tz)

        if bars_elapsed <= 1.1:
            return self._fetch_and_append_new_bar(symbol, strategy_config, cache_key, strategy_tz)

        return self._full_refresh_bars(symbol, strategy_config, cache_key, strategy_tz)

    def _should_skip_mt5_call(self, cached_metadata: _BarCacheEntry | None, now_broker: datetime) -> bool:
        """Evaluate whether current bar is still forming."""
        if cached_metadata is None:
            return False
        return now_broker < cached_metadata.next_bar_complete_at

    def _fetch_and_append_new_bar(
        self,
        symbol: str,
        strategy_config: RawStrategyConfig,
        cache_key: tuple[str, int],
        strategy_tz: ZoneInfo,
    ) -> pd.DataFrame | None:
        """Fetch single new bar and append to rolling window (incremental update)."""
        timeframe_mt5 = cache_key[1]

        cached_df = self._get_cached_window(cache_key)
        if cached_df is None or len(cached_df) == 0:
            return self._full_refresh_bars(symbol, strategy_config, cache_key, strategy_tz)

        # Fetch 2 bars from position 0 to handle bar transition race condition.
        # At bar boundaries, MT5 may not have created the new forming bar yet, so
        # position 0 could still be the just-completed bar rather than a forming bar.
        rates = mt5.copy_rates_from_pos(symbol, timeframe_mt5, 0, 2)
        if rates is None or len(rates) == 0:
            return self._full_refresh_bars(symbol, strategy_config, cache_key, strategy_tz)

        new_bar_df = self._create_tz_aware_dataframe(rates, strategy_tz)
        new_bar_df = self._filter_complete_bars(new_bar_df, strategy_config)

        # No complete bars available
        if len(new_bar_df) == 0:
            if cached_metadata := self._latest_bar_cache.get(cache_key):
                cached_metadata.next_bar_complete_at = cached_metadata.bar_time + pd.Timedelta(
                    minutes=cached_metadata.timeframe_minutes, seconds=2
                )
            return self._return_cached_window(cache_key, strategy_config)

        # Take the latest complete bar
        latest_complete_bar = new_bar_df.iloc[[-1]]
        new_bar_time = latest_complete_bar.index[0]
        old_last_bar_time = cached_df.index[-1]
        is_new_bar = old_last_bar_time is None or new_bar_time > old_last_bar_time

        if not is_new_bar:
            return self._return_cached_window(cache_key, strategy_config)

        cache_capacity = self._get_cache_capacity(strategy_config)
        updated_df = pd.concat([cached_df, latest_complete_bar], ignore_index=False)
        self._bar_rolling_windows[cache_key] = updated_df.tail(cache_capacity).copy()

        new_bar_time_broker = latest_complete_bar.index[0].tz_convert(self.broker_tz)
        timeframe_minutes = strategy_config.get("timeframe_minutes", TIMEFRAME_TO_MINUTES[TimeFrame(timeframe_mt5)])

        self._update_cache_metadata(cache_key, new_bar_time_broker, timeframe_minutes)

        return self._return_cached_window(cache_key, strategy_config)

    def _full_refresh_bars(
        self,
        symbol: str,
        strategy_config: RawStrategyConfig,
        cache_key: tuple[str, int],
        strategy_tz: ZoneInfo,
    ) -> pd.DataFrame | None:
        """Full refresh of bar data from MT5 (cache miss or gap scenario)."""
        timeframe_mt5 = cache_key[1]
        fetch_count = self._get_cache_capacity(strategy_config)
        rates = mt5.copy_rates_from_pos(symbol, timeframe_mt5, 0, fetch_count)

        if rates is None or len(rates) == 0:
            return None

        df_raw = self._create_tz_aware_dataframe(rates, strategy_tz)
        df_complete = self._filter_complete_bars(df_raw, strategy_config)

        # Cache latest fetched bar first to avoid repeated MT5 polling while a bar forms.
        timeframe_minutes = strategy_config.get("timeframe_minutes", TIMEFRAME_TO_MINUTES[TimeFrame(timeframe_mt5)])

        if len(df_complete) == 0:
            self._bar_rolling_windows[cache_key] = pd.DataFrame(columns=df_raw.columns)
            return pd.DataFrame(columns=df_raw.columns)

        self._bar_rolling_windows[cache_key] = df_complete.tail(fetch_count).copy()

        # Use latest complete bar for elapsed-bar calculations.
        latest_complete_bar_time = df_complete.index[-1].tz_convert(self.broker_tz)
        self._update_cache_metadata(cache_key, latest_complete_bar_time, timeframe_minutes)

        return self._return_cached_window(cache_key, strategy_config)

    def _filter_complete_bars(self, df: pd.DataFrame, strategy_config: RawStrategyConfig) -> pd.DataFrame:
        """Filter out incomplete bars (bars still forming)."""
        if not isinstance(df.index, pd.DatetimeIndex):
            raise TypeError("DataFrame index must be a DatetimeIndex")
        timeframe_minutes = strategy_config.get("timeframe_minutes")
        now_strategy = datetime.now(tz=df.index.tz)

        bar_close_times = df.index + pd.Timedelta(minutes=timeframe_minutes)
        tolerance = pd.Timedelta(seconds=2)
        cutoff_time = pd.Timestamp(now_strategy) + tolerance
        mask_complete = bar_close_times <= cutoff_time

        return df[mask_complete]

    def get_current_tick(self, symbol: str) -> Tick | None:
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return None
        return Tick.from_mt5(tick)

    def _filter_session_hours(self, df: pd.DataFrame, strategy_config: RawStrategyConfig) -> pd.DataFrame:
        """Filter bars to trading session hours. Handles midnight-spanning sessions (e.g., 23:00-01:00)."""
        sessions = strategy_config.get("sessions", [])
        if not strategy_config.get("filter_enabled", False) or not sessions:
            return df

        if not isinstance(df.index, pd.DatetimeIndex):
            raise TypeError("DataFrame index must be a DatetimeIndex")
        bar_minutes = df.index.hour * 60 + df.index.minute
        combined_mask = np.zeros(len(df), dtype=bool)

        for session in sessions:
            start_h, start_m = map(int, session.start.split(":"))
            end_h, end_m = map(int, session.end.split(":"))

            start_min = start_h * 60 + start_m
            end_min = end_h * 60 + end_m

            if end_min >= start_min:
                session_mask = (bar_minutes >= start_min) & (bar_minutes <= end_min)
            else:
                # Midnight-spanning session
                session_mask = (bar_minutes >= start_min) | (bar_minutes <= end_min)

            combined_mask |= session_mask

        return df.loc[combined_mask]
