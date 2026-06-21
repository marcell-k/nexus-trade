from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

import MetaTrader5 as mt
import numpy as np
import pandas as pd

from nexus_trade.config.timings import LOOKBACK, TOLERANCE_SECONDS
from nexus_trade.core.constants import TIMEFRAME_TO_MINUTES, TimeFrame, string_to_timeframe
from nexus_trade.core.models import Tick
from nexus_trade.core.registry import STRATEGY_CONFIG_REGISTRY

if TYPE_CHECKING:
    from pathlib import Path
    from zoneinfo import ZoneInfo

    from nexus_trade.config.strategy import BaseStrategyParams, StrategyConfig


@dataclass(slots=True)
class _BarCacheEntry:
    bar_time: pd.Timestamp
    timeframe_minutes: int
    next_bar_complete_at: pd.Timestamp
    cache_seeded: bool = False


@dataclass(slots=True)
class _RingBuffer:
    """Pre-allocated fixed-size ring buffer for OHLCV bar data."""

    capacity: int
    _size: int = field(default=0, init=False)
    _head: int = field(default=0, init=False)
    _timestamps: np.ndarray = field(init=False)
    _open: np.ndarray = field(init=False)
    _high: np.ndarray = field(init=False)
    _low: np.ndarray = field(init=False)
    _close: np.ndarray = field(init=False)
    _volume: np.ndarray = field(init=False)
    _spread: np.ndarray = field(init=False)

    _is_dirty: bool = field(default=True, init=False)
    _cached_df: pd.DataFrame | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        self._timestamps = np.empty(self.capacity, dtype="datetime64[ns]")
        self._open = np.empty(self.capacity, dtype=np.float64)
        self._high = np.empty(self.capacity, dtype=np.float64)
        self._low = np.empty(self.capacity, dtype=np.float64)
        self._close = np.empty(self.capacity, dtype=np.float64)
        self._volume = np.empty(self.capacity, dtype=np.float64)
        self._spread = np.empty(self.capacity, dtype=np.float64)

    @property
    def size(self) -> int:
        return self._size

    @property
    def is_empty(self) -> bool:
        return self._size == 0

    def latest_timestamp_ns(self) -> int:
        """Return the newest bar timestamp as int64 nanoseconds. Caller checks is_empty first."""
        tail = (self._head + self._size - 1) % self.capacity
        return int(self._timestamps[tail].astype(np.int64))

    def append(
        self, ts_ns: int, open_: float, high: float, low: float, close: float, volume: float, spread: float
    ) -> None:
        if self._size < self.capacity:
            idx = (self._head + self._size) % self.capacity
            self._size += 1
        else:
            idx = self._head
            self._head = (self._head + 1) % self.capacity

        self._is_dirty = True

        self._timestamps[idx] = np.datetime64(ts_ns, "ns")
        self._open[idx] = open_
        self._high[idx] = high
        self._low[idx] = low
        self._close[idx] = close
        self._volume[idx] = volume
        self._spread[idx] = spread

    def to_dataframe(self, tz: ZoneInfo) -> pd.DataFrame:
        """O(1) read if cached, O(n) if dirty."""
        if not self._is_dirty and self._cached_df is not None:
            return self._cached_df

        if self._size == 0:
            return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume", "spread"])

        indices = np.arange(self._head, self._head + self._size) % self.capacity
        ts = self._timestamps[indices].astype("datetime64[ns]")
        index = pd.DatetimeIndex(ts, tz="UTC").tz_convert(tz)
        index.name = "Time"

        self._cached_df = pd.DataFrame(
            {
                "Open": self._open[indices],
                "High": self._high[indices],
                "Low": self._low[indices],
                "Close": self._close[indices],
                "Volume": self._volume[indices],
                "spread": self._spread[indices],
            },
            index=index,
        )
        self._is_dirty = False
        return self._cached_df

    def clear(self) -> None:
        self._size = 0
        self._head = 0
        self._is_dirty = True
        self._cached_df = None


class DataHandler:
    def __init__(self, broker_tz: ZoneInfo, calendar_path: Path | None = None) -> None:
        self.broker_tz: ZoneInfo = broker_tz
        self.calendar_path: Path | None = calendar_path

        self._latest_bar_cache: dict[tuple[str, int, int], _BarCacheEntry] = {}
        self._ring_buffers: dict[tuple[str, int, int], _RingBuffer] = {}

    def _get_capacity(self, strategy_config: StrategyConfig[BaseStrategyParams]) -> int:
        window = strategy_config.params.backcandles
        if strategy_config.trading_hours.enabled:
            return int(window * 1.3)
        return window

    def _get_or_create_ring(self, cache_key: tuple[str, int, int], capacity: int) -> _RingBuffer:
        ring = self._ring_buffers.get(cache_key)
        if ring is None or ring.capacity != capacity:
            ring = _RingBuffer(capacity=capacity)
            self._ring_buffers[cache_key] = ring
        return ring

    def _update_cache_metadata(
        self, cache_key: tuple[str, int, int], bar_time_broker: pd.Timestamp, timeframe_minutes: int
    ) -> None:
        next_bar_complete_at = bar_time_broker + pd.Timedelta(minutes=timeframe_minutes, seconds=TOLERANCE_SECONDS)
        self._latest_bar_cache[cache_key] = _BarCacheEntry(
            bar_time=bar_time_broker,
            timeframe_minutes=timeframe_minutes,
            next_bar_complete_at=next_bar_complete_at,
            cache_seeded=True,
        )

    def _should_skip_mt5_call(self, cached_metadata: _BarCacheEntry, now_broker: datetime) -> bool:
        return now_broker < cached_metadata.next_bar_complete_at

    def _extract_complete_bars(
        self, rates: np.ndarray, timeframe_minutes: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Filter `rates` (as returned by MT5) down to bars that are fully closed.

        Returns (ts_ns, open, high, low, close, volume, spread), oldest first, where ts_ns is
        each bar's *open* time as an absolute epoch-nanosecond instant (timezone-agnostic —
        convert it for display only, never for comparison; comparing raw ints sidesteps any
        clock-drift/jitter issues entirely).
        """
        cutoff_ns = int((pd.Timestamp.now(tz="UTC") + pd.Timedelta(seconds=10)).value)

        broker_offset_s = int(
            pd.Timestamp.now(tz=self.broker_tz).utcoffset().total_seconds()  # type: ignore[union-attr]
        )

        ts_s = rates["time"].astype(np.int64)
        ts_utc_s: np.ndarray = ts_s - broker_offset_s
        bar_close_utc_ns: np.ndarray = (ts_utc_s + timeframe_minutes * 60) * 1_000_000_000
        bar_open_utc_ns: np.ndarray = ts_utc_s * 1_000_000_000

        complete_mask = bar_close_utc_ns <= cutoff_ns
        indices = np.where(complete_mask)[0]

        return (
            bar_open_utc_ns[indices],
            rates["open"][indices].astype(np.float64),
            rates["high"][indices].astype(np.float64),
            rates["low"][indices].astype(np.float64),
            rates["close"][indices].astype(np.float64),
            rates["tick_volume"][indices].astype(np.float64),
            rates["spread"][indices].astype(np.float64),
        )

    def _ring_to_output(
        self,
        ring: _RingBuffer,
        strategy_tz: ZoneInfo,
        strategy_config: StrategyConfig[BaseStrategyParams],
    ) -> pd.DataFrame | None:
        if ring.is_empty:
            return None
        df = ring.to_dataframe(strategy_tz)
        if strategy_config.trading_hours.enabled:
            df = self._filter_session_hours(df, strategy_config)
        return df.tail(strategy_config.params.backcandles) if len(df) > 0 else None

    def get_latest_bars(self, strategy_name: str) -> pd.DataFrame | None:
        strategy_config = STRATEGY_CONFIG_REGISTRY.get_strategy_config(strategy_name)
        symbol = strategy_config.params.symbol
        timeframe_mt5 = string_to_timeframe(strategy_config.params.timeframe)
        capacity = self._get_capacity(strategy_config)
        cache_key: tuple[str, int, int] = (symbol, int(timeframe_mt5), capacity)
        strategy_tz = STRATEGY_CONFIG_REGISTRY.get_tz(strategy_name)
        timeframe_minutes = TIMEFRAME_TO_MINUTES[TimeFrame(int(timeframe_mt5))]
        now_broker = datetime.now(self.broker_tz)

        cached_metadata = self._latest_bar_cache.get(cache_key)
        ring = self._get_or_create_ring(cache_key, capacity)

        if (
            cached_metadata is not None
            and cached_metadata.cache_seeded
            and not ring.is_empty
            and self._should_skip_mt5_call(cached_metadata, now_broker)
        ):
            return self._ring_to_output(ring, strategy_tz, strategy_config)

        if cached_metadata is None or not cached_metadata.cache_seeded or ring.is_empty:
            return self._full_refresh(
                symbol, int(timeframe_mt5), timeframe_minutes, capacity, cache_key, ring, strategy_tz, strategy_config
            )

        # Past the freshness window. Don't guess how many bars elapsed from the clock —
        # fetch a small lookback and let the data tell us: nothing new, a clean append,
        # or a real gap that needs a full refresh. This is what removes the dependence
        # on millisecond-level scheduler/clock timing that caused the old elapsed-based
        # branch to flicker between SAME-BAR / INCREMENTAL / GAP-REFRESH.
        return self._incremental_fetch(
            symbol, int(timeframe_mt5), timeframe_minutes, capacity, cache_key, ring, strategy_tz, strategy_config
        )

    def _incremental_fetch(
        self,
        symbol: str,
        timeframe_mt5: int,
        timeframe_minutes: int,
        capacity: int,
        cache_key: tuple[str, int, int],
        ring: _RingBuffer,
        strategy_tz: ZoneInfo,
        strategy_config: StrategyConfig[BaseStrategyParams],
    ) -> pd.DataFrame | None:
        # Small headroom window: enough to absorb a few missed/delayed polls without
        # paying for a full `capacity`-sized fetch every time.
        rates = mt.copy_rates_from_pos(symbol, timeframe_mt5, 0, LOOKBACK)
        if rates is None or len(rates) == 0:
            return self._ring_to_output(ring, strategy_tz, strategy_config)

        ts_ns, o, h, lo, c, v, sp = self._extract_complete_bars(rates, timeframe_minutes)
        if len(ts_ns) == 0 or ring.is_empty:
            return self._ring_to_output(ring, strategy_tz, strategy_config)

        old_newest_ns = ring.latest_timestamp_ns()
        new_mask = ts_ns > old_newest_ns

        if not new_mask.any():
            return self._ring_to_output(ring, strategy_tz, strategy_config)

        interval_ns = timeframe_minutes * 60 * 1_000_000_000
        new_indices = np.where(new_mask)[0]
        earliest_new_ns = int(ts_ns[new_indices[0]])

        if earliest_new_ns - old_newest_ns > interval_ns:
            # The lookback window didn't reach back far enough to bridge contiguously —
            # bars were genuinely missed (e.g. the bot was paused/blocked). Full refresh.
            return self._full_refresh(
                symbol, timeframe_mt5, timeframe_minutes, capacity, cache_key, ring, strategy_tz, strategy_config
            )

        for i in new_indices:
            ring.append(int(ts_ns[i]), float(o[i]), float(h[i]), float(lo[i]), float(c[i]), float(v[i]), float(sp[i]))

        newest_ns = ring.latest_timestamp_ns()
        newest_ts = pd.Timestamp(newest_ns, unit="ns", tz=self.broker_tz)
        self._update_cache_metadata(cache_key, newest_ts, timeframe_minutes)

        return self._ring_to_output(ring, strategy_tz, strategy_config)

    def _full_refresh(
        self,
        symbol: str,
        timeframe_mt5: int,
        timeframe_minutes: int,
        capacity: int,
        cache_key: tuple[str, int, int],
        ring: _RingBuffer,
        strategy_tz: ZoneInfo,
        strategy_config: StrategyConfig[BaseStrategyParams],
    ) -> pd.DataFrame | None:
        rates = mt.copy_rates_from_pos(symbol, timeframe_mt5, 0, capacity)
        if rates is None or len(rates) == 0:
            return None

        ts_ns, o, h, lo, c, v, sp = self._extract_complete_bars(rates, timeframe_minutes)
        if len(ts_ns) == 0:
            return None

        ring.clear()
        for i in range(len(ts_ns)):
            ring.append(int(ts_ns[i]), float(o[i]), float(h[i]), float(lo[i]), float(c[i]), float(v[i]), float(sp[i]))

        newest_ns = ring.latest_timestamp_ns()
        newest_ts = pd.Timestamp(newest_ns, unit="ns", tz=self.broker_tz)
        self._update_cache_metadata(cache_key, newest_ts, timeframe_minutes)

        return self._ring_to_output(ring, strategy_tz, strategy_config)

    def get_current_tick(self, symbol: str) -> Tick | None:
        tick = mt.symbol_info_tick(symbol)
        if tick is None:
            return None
        return Tick.from_mt5(tick)

    def _filter_session_hours(
        self, df: pd.DataFrame, strategy_config: StrategyConfig[BaseStrategyParams]
    ) -> pd.DataFrame:
        th = strategy_config.trading_hours
        if not th.enabled or not th.sessions:
            return df
        if not isinstance(df.index, pd.DatetimeIndex):
            raise TypeError("DataFrame index must be a DatetimeIndex")

        bar_minutes = df.index.hour * 60 + df.index.minute
        combined_mask = np.zeros(len(df), dtype=bool)

        for session in th.sessions:
            start_h, start_m = map(int, session.start.split(":"))
            end_h, end_m = map(int, session.end.split(":"))
            start_min = start_h * 60 + start_m
            end_min = end_h * 60 + end_m

            if end_min >= start_min:
                combined_mask |= (bar_minutes >= start_min) & (bar_minutes <= end_min)
            else:
                combined_mask |= (bar_minutes >= start_min) | (bar_minutes <= end_min)

        return df.loc[combined_mask]
