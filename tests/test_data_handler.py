"""
Minimal tests for DataHandler internal components.

Tests call private methods directly to bypass the module-level TimeFrame
enum corruption that occurs when the MT5 stub is installed before constants.py
is first imported.  Each test receives explicit `timeframe_minutes` so no
TIMEFRAME_TO_MINUTES look-up is needed.
"""

from __future__ import annotations

import time
from datetime import datetime
from unittest.mock import MagicMock
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from nexus_trade.core.data_handler import DataHandler, _BarCacheEntry, _RingBuffer

_BROKER_TZ = ZoneInfo("UTC")

# Structured array dtype that matches MT5 copy_rates_from_pos output
_RATES_DTYPE = np.dtype(
    [
        ("time", np.int64),
        ("open", np.float64),
        ("high", np.float64),
        ("low", np.float64),
        ("close", np.float64),
        ("tick_volume", np.int64),
        ("spread", np.int32),
    ]
)


# ── helpers ───────────────────────────────────────────────────────────────────


def _now_s() -> int:
    return int(time.time())


def _rates(timestamps_s: list[int], closes: list[float] | None = None) -> np.ndarray:
    """Build a structured rates array with synthetic OHLCV data."""
    n = len(timestamps_s)
    if closes is None:
        closes = [1.1 + i * 0.001 for i in range(n)]
    arr = np.zeros(n, dtype=_RATES_DTYPE)
    for i, (t, c) in enumerate(zip(timestamps_s, closes, strict=True)):
        arr[i]["time"] = t
        arr[i]["open"] = c - 0.0001
        arr[i]["high"] = c + 0.0002
        arr[i]["low"] = c - 0.0002
        arr[i]["close"] = c
        arr[i]["tick_volume"] = 100
        arr[i]["spread"] = 2
    return arr


def _strategy_cfg(backcandles: int = 100, session_enabled: bool = False) -> MagicMock:
    cfg = MagicMock()
    cfg.params.backcandles = backcandles
    cfg.trading_hours.enabled = session_enabled
    return cfg


@pytest.fixture
def dh() -> DataHandler:
    return DataHandler(_BROKER_TZ)


# ── _RingBuffer ───────────────────────────────────────────────────────────────


class TestRingBuffer:
    def test_empty_on_creation(self) -> None:
        ring = _RingBuffer(capacity=5)
        assert ring.is_empty
        assert ring.size == 0

    def test_single_append_increments_size(self) -> None:
        ring = _RingBuffer(capacity=5)
        ring.append(1_000_000_000, 1.1, 1.11, 1.09, 1.105, 100, 2)
        assert ring.size == 1
        assert not ring.is_empty

    def test_overflow_evicts_oldest_entry(self) -> None:
        ring = _RingBuffer(capacity=3)
        for i in range(5):
            ring.append((i + 1) * 1_000_000_000, 1.1, 1.11, 1.09, float(i), 100, 2)
        assert ring.size == 3
        df = ring.to_dataframe(ZoneInfo("UTC"))
        # Oldest two (Close=0.0, 1.0) evicted; last three (2.0, 3.0, 4.0) remain
        assert list(df["Close"]) == [2.0, 3.0, 4.0]

    def test_to_dataframe_index_is_chronological(self) -> None:
        ring = _RingBuffer(capacity=5)
        base_ns = int(time.time()) * 1_000_000_000
        for i in range(3):
            ring.append(base_ns + i * 60_000_000_000, 1.1, 1.11, 1.09, 1.1 + i * 0.001, 100, 2)
        df = ring.to_dataframe(ZoneInfo("UTC"))
        assert len(df) == 3
        assert list(df.index) == sorted(df.index)

    def test_latest_timestamp_ns_returns_newest(self) -> None:
        ring = _RingBuffer(capacity=5)
        ts_old, ts_new = 1_000_000_000_000, 2_000_000_000_000
        ring.append(ts_old, 1.1, 1.11, 1.09, 1.1, 100, 2)
        ring.append(ts_new, 1.2, 1.21, 1.19, 1.2, 100, 2)
        assert ring.latest_timestamp_ns() == ts_new

    def test_clear_resets_to_empty(self) -> None:
        ring = _RingBuffer(capacity=5)
        ring.append(1_000_000, 1.1, 1.11, 1.09, 1.1, 100, 2)
        ring.clear()
        assert ring.is_empty
        assert ring.size == 0

    def test_dataframe_empty_ring_has_correct_columns(self) -> None:
        ring = _RingBuffer(capacity=5)
        df = ring.to_dataframe(ZoneInfo("UTC"))
        assert set(df.columns) == {"Open", "High", "Low", "Close", "Volume", "spread"}
        assert df.empty


# ── _extract_complete_bars ────────────────────────────────────────────────────


class TestExtractCompleteBars:
    """
    With broker_tz=UTC, rates["time"] is treated as UTC epoch seconds.
    A bar at epoch T with timeframe M minutes is complete when
    T + M*60 <= now + 10  (the +10 s grace in the implementation).
    """

    def test_complete_bar_included(self, dh: DataHandler) -> None:
        # Bar from 40 min ago closes at 25 min ago — complete for M15
        ts = [_now_s() - 40 * 60]
        ts_ns, *_ = dh._extract_complete_bars(_rates(ts), 15)
        assert len(ts_ns) == 1

    def test_current_incomplete_bar_excluded(self, dh: DataHandler) -> None:
        # Bar from 5 min ago closes in 10 min — still open
        ts = [_now_s() - 5 * 60]
        ts_ns, *_ = dh._extract_complete_bars(_rates(ts), 15)
        assert len(ts_ns) == 0

    def test_mixed_complete_and_incomplete(self, dh: DataHandler) -> None:
        ts = [_now_s() - 30 * 60, _now_s() - 5 * 60]  # one complete, one not
        ts_ns, *_ = dh._extract_complete_bars(_rates(ts), 15)
        assert len(ts_ns) == 1

    def test_empty_rates_array_returns_empty(self, dh: DataHandler) -> None:
        r = np.zeros(0, dtype=_RATES_DTYPE)
        ts_ns, *_ = dh._extract_complete_bars(r, 15)
        assert len(ts_ns) == 0

    def test_returns_seven_arrays(self, dh: DataHandler) -> None:
        # ts_ns, open, high, low, close, tick_volume, spread
        ts = [_now_s() - 30 * 60]
        result = dh._extract_complete_bars(_rates(ts), 15)
        assert len(result) == 7


# ── _full_refresh ─────────────────────────────────────────────────────────────


class TestFullRefresh:
    _KEY: tuple[str, int, int] = ("EURUSD", 1, 100)

    def _call(
        self,
        dh: DataHandler,
        mt5_mock: MagicMock,
        timestamps_s: list[int],
        backcandles: int = 100,
    ) -> pd.DataFrame | None:
        mt5_mock.copy_rates_from_pos.return_value = _rates(timestamps_s)
        ring = dh._get_or_create_ring(self._KEY, backcandles)
        return dh._full_refresh(
            "EURUSD",
            1,
            15,
            backcandles,
            self._KEY,
            ring,
            ZoneInfo("UTC"),
            _strategy_cfg(backcandles),
        )

    def test_returns_all_complete_bars(self, dh: DataHandler, mt5_mock: MagicMock) -> None:
        ts = [_now_s() - m * 60 for m in (60, 45, 30)]
        df = self._call(dh, mt5_mock, ts)
        assert df is not None
        assert len(df) == 3

    def test_filters_current_incomplete_bar(self, dh: DataHandler, mt5_mock: MagicMock) -> None:
        ts = [_now_s() - 30 * 60, _now_s() - 5 * 60]  # one complete + current bar
        df = self._call(dh, mt5_mock, ts)
        assert df is not None
        assert len(df) == 1

    def test_none_from_mt5_returns_none(self, dh: DataHandler, mt5_mock: MagicMock) -> None:
        mt5_mock.copy_rates_from_pos.return_value = None
        ring = dh._get_or_create_ring(self._KEY, 100)
        result = dh._full_refresh(
            "EURUSD",
            1,
            15,
            100,
            self._KEY,
            ring,
            ZoneInfo("UTC"),
            _strategy_cfg(),
        )
        assert result is None

    def test_seeds_cache_metadata_entry(self, dh: DataHandler, mt5_mock: MagicMock) -> None:
        ts = [_now_s() - 30 * 60]
        mt5_mock.copy_rates_from_pos.return_value = _rates(ts)
        ring = dh._get_or_create_ring(self._KEY, 100)
        dh._full_refresh("EURUSD", 1, 15, 100, self._KEY, ring, ZoneInfo("UTC"), _strategy_cfg())
        assert self._KEY in dh._latest_bar_cache
        assert dh._latest_bar_cache[self._KEY].cache_seeded is True


# ── _incremental_fetch ────────────────────────────────────────────────────────


class TestIncrementalFetch:
    _KEY: tuple[str, int, int] = ("EURUSD", 1, 100)

    def _seed_ring(self, dh: DataHandler, ts_s: int) -> _RingBuffer:
        ring = dh._get_or_create_ring(self._KEY, 100)
        ts_ns = ts_s * 1_000_000_000
        ring.append(ts_ns, 1.1, 1.11, 1.09, 1.1, 100, 2)
        dh._update_cache_metadata(
            self._KEY,
            pd.Timestamp(ts_ns, unit="ns", tz=_BROKER_TZ),
            15,
        )
        return ring

    def test_new_bar_appended_to_ring(self, dh: DataHandler, mt5_mock: MagicMock) -> None:
        old_ts = _now_s() - 45 * 60  # 45 min ago (complete)
        new_ts = _now_s() - 30 * 60  # 30 min ago; gap = 15 min == 1 interval → append, not full-refresh
        ring = self._seed_ring(dh, old_ts)

        mt5_mock.copy_rates_from_pos.return_value = _rates([new_ts])
        df = dh._incremental_fetch("EURUSD", 1, 15, 100, self._KEY, ring, ZoneInfo("UTC"), _strategy_cfg())
        assert df is not None
        assert ring.size == 2
        assert len(df) == 2

    def test_no_new_bar_leaves_ring_unchanged(self, dh: DataHandler, mt5_mock: MagicMock) -> None:
        ts = _now_s() - 30 * 60
        ring = self._seed_ring(dh, ts)

        # Rates contain the bar already in the ring — no new timestamps
        mt5_mock.copy_rates_from_pos.return_value = _rates([ts])
        df = dh._incremental_fetch("EURUSD", 1, 15, 100, self._KEY, ring, ZoneInfo("UTC"), _strategy_cfg())
        assert df is not None
        assert ring.size == 1  # unchanged

    def test_gap_triggers_full_refresh(self, dh: DataHandler, mt5_mock: MagicMock) -> None:
        """Gap of 80 min between ring tail and first new bar (> 1 interval) → full refresh."""
        old_ts = _now_s() - 40 * 60  # 100 min ago
        new_ts = _now_s() - 24 * 60  # 20 min ago; gap = 80 min >> 15 min interval

        ring = self._seed_ring(dh, old_ts)

        # 1st call: incremental (LOOKBACK bars), 2nd call: full refresh (capacity)
        mt5_mock.copy_rates_from_pos.side_effect = [
            _rates([new_ts]),
            _rates([old_ts, new_ts]),
        ]
        dh._incremental_fetch("EURUSD", 1, 15, 100, self._KEY, ring, ZoneInfo("UTC"), _strategy_cfg())
        assert mt5_mock.copy_rates_from_pos.call_count == 2


# ── _should_skip_mt5_call ─────────────────────────────────────────────────────


class TestShouldSkipMt5Call:
    def _entry(self, next_bar_offset_minutes: float, dh: DataHandler) -> _BarCacheEntry:
        tz = ZoneInfo("UTC")
        now = pd.Timestamp.now(tz=tz)
        return _BarCacheEntry(
            bar_time=now,
            timeframe_minutes=15,
            next_bar_complete_at=now + pd.Timedelta(minutes=next_bar_offset_minutes),
            cache_seeded=True,
        )

    def test_skip_when_next_bar_in_future(self, dh: DataHandler) -> None:
        entry = self._entry(10.0, dh)  # next bar due in 10 min
        assert dh._should_skip_mt5_call(entry, datetime.now(ZoneInfo("UTC"))) is True

    def test_no_skip_when_next_bar_overdue(self, dh: DataHandler) -> None:
        entry = self._entry(-1.0, dh)  # next bar was due 1 min ago
        assert dh._should_skip_mt5_call(entry, datetime.now(ZoneInfo("UTC"))) is False


# ── _filter_session_hours ─────────────────────────────────────────────────────


class TestFilterSessionHours:
    @staticmethod
    def _df(hours: list[int]) -> pd.DataFrame:
        idx = pd.DatetimeIndex([pd.Timestamp(f"2025-01-06 {h:02d}:00:00", tz="UTC") for h in hours])
        return pd.DataFrame(
            {"Open": 1.1, "High": 1.11, "Low": 1.09, "Close": 1.1, "Volume": 100, "spread": 2},
            index=idx,
        )

    @staticmethod
    def _session_cfg(*sessions: tuple[str, str]) -> MagicMock:
        cfg = MagicMock()
        cfg.trading_hours.enabled = True
        cfg.trading_hours.sessions = [MagicMock(start=s, end=e) for s, e in sessions]
        return cfg

    def test_out_of_hours_bars_excluded(self, dh: DataHandler) -> None:
        df = self._df([6, 7, 8, 9, 10])
        result = dh._filter_session_hours(df, self._session_cfg(("08:00", "10:00")))
        assert set(result.index.hour) == {8, 9, 10}  # pyright: ignore[reportAttributeAccessIssue]

    def test_all_in_session_bars_included(self, dh: DataHandler) -> None:
        df = self._df([8, 9, 10])
        result = dh._filter_session_hours(df, self._session_cfg(("07:00", "11:00")))
        assert len(result) == 3

    def test_midnight_spanning_session(self, dh: DataHandler) -> None:
        df = self._df([21, 22, 23, 0, 1, 2, 3, 4])
        result = dh._filter_session_hours(df, self._session_cfg(("22:00", "02:00")))
        assert set(result.index.hour) == {22, 23, 0, 1, 2}  # pyright: ignore[reportAttributeAccessIssue]

    def test_multiple_non_overlapping_sessions(self, dh: DataHandler) -> None:
        df = self._df([6, 8, 12, 15])
        result = dh._filter_session_hours(df, self._session_cfg(("08:00", "09:00"), ("14:00", "16:00")))
        assert set(result.index.hour) == {8, 15}  # pyright: ignore[reportAttributeAccessIssue]

    def test_empty_dataframe_returns_empty(self, dh: DataHandler) -> None:
        df = self._df([])
        result = dh._filter_session_hours(df, self._session_cfg(("08:00", "17:00")))
        assert result.empty
