"""Unit tests for NewsFilter and preprocess_calendar_file."""

from __future__ import annotations

import textwrap
from datetime import date, datetime, timedelta
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from nexus_trade.filters.news import NewsFilter, preprocess_calendar_file

if TYPE_CHECKING:
    from pathlib import Path

BROKER_TZ = ZoneInfo("Etc/GMT-3")
STRATEGY_TZ = ZoneInfo("UTC")

# Type values must match what preprocess_calendar_file accepts — INDICATOR/EVENT/HOLIDAY
VALID_CSV = textwrap.dedent("""\
    Date,Time,Currency,Event,Impact,Type
    2025.06.16,14:30,USD,Non-Farm Payrolls,HIGH,INDICATOR
    2025.06.16,08:00,EUR,ECB Rate Decision,HIGH,EVENT
    2025.06.17,00:00,USD,Independence Day,NONE,HOLIDAY
    2025.06.16,12:00,GBP,CPI y/y,MEDIUM,INDICATOR
    2025.06.16,10:00,JPY,BOJ Press Conference,LOW,INDICATOR
""")


@pytest.fixture
def calendar_csv(tmp_path: Path) -> Path:
    path = tmp_path / "calendar.csv"
    path.write_text(VALID_CSV, encoding="utf-16")
    return path


class TestPreprocessCalendarFile:
    def test_parses_valid_csv(self, calendar_csv: Path) -> None:
        df, _ = preprocess_calendar_file(calendar_csv, BROKER_TZ)
        assert not df.empty

    def test_maps_impact_to_priority(self, calendar_csv: Path) -> None:
        df, _ = preprocess_calendar_file(calendar_csv, BROKER_TZ)
        assert "High" in df["priority"].values
        assert "Medium" in df["priority"].values
        assert "Low" in df["priority"].values

    def test_extracts_holidays(self, calendar_csv: Path) -> None:
        _, holidays = preprocess_calendar_file(calendar_csv, BROKER_TZ)
        currencies = {c for c, _ in holidays}
        assert "USD" in currencies

    def test_time_broker_is_tz_aware(self, calendar_csv: Path) -> None:
        df, _ = preprocess_calendar_file(calendar_csv, BROKER_TZ)
        assert df["time_broker"].dt.tz is not None

    def test_missing_required_columns_returns_empty(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.csv"
        bad.write_text("Date,Event\n2025.01.01,Something", encoding="utf-16")
        df, holidays = preprocess_calendar_file(bad, BROKER_TZ)
        assert df.empty
        assert len(holidays) == 0

    def test_invalid_date_rows_dropped(self, tmp_path: Path) -> None:
        content = textwrap.dedent("""\
            Date,Time,Currency,Event,Impact,Type
            INVALID,99:99,USD,Bad Event,HIGH,INDICATOR
            2025.06.16,14:30,USD,NFP,HIGH,INDICATOR
        """)
        path = tmp_path / "mixed.csv"
        path.write_text(content, encoding="utf-16")
        df, _ = preprocess_calendar_file(path, BROKER_TZ)
        assert len(df) == 1

    def test_columns_renamed(self, calendar_csv: Path) -> None:
        df, _ = preprocess_calendar_file(calendar_csv, BROKER_TZ)
        assert "event_name" in df.columns
        assert "currency" in df.columns
        assert "Event" not in df.columns


def _build_news_filter(
    calendar_df: pd.DataFrame,
    holidays: frozenset,
    *,
    buffer_minutes: int = 15,
    currencies: list[str] | None = None,
    fail_open: bool = False,
) -> NewsFilter:
    """Construct NewsFilter with pre-loaded calendar — bypasses file I/O and registry."""
    dh = MagicMock()
    dh.broker_tz = BROKER_TZ

    with patch("nexus_trade.filters.news.STRATEGY_CONFIG_REGISTRY") as reg:
        cfg = MagicMock()
        cfg.params.symbol = "EURUSD"
        cfg.params.timezone = "UTC"
        cfg.filters.news.enabled = True
        cfg.filters.news.currencies = currencies or ["USD"]
        cfg.filters.news.buffer_minutes = buffer_minutes
        cfg.trading_hours = None
        reg.get_strategy_config.return_value = cfg

        nf = NewsFilter(data_handler=dh, strategy_name="test", fail_open=fail_open)

    if not calendar_df.empty:
        calendar_df = calendar_df.copy()
        calendar_df["time_strategy"] = calendar_df["time_broker"].dt.tz_convert(STRATEGY_TZ)
    nf._set_calendar_cache(calendar_df, holidays)
    return nf


def _event_df(event_time: datetime, priority: str = "High", currency: str = "USD") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time_broker": [pd.Timestamp(event_time).tz_convert(BROKER_TZ)],
            "currency": [currency],
            "priority": [priority],
            "Type": ["INDICATOR"],
            "event_name": ["NFP"],
        }
    )


class TestNewsFilterIsSafeToTrade:
    def test_safe_when_no_events(self) -> None:
        nf = _build_news_filter(pd.DataFrame(), frozenset())
        assert nf.is_safe_to_trade() is True

    def test_blocked_within_buffer(self) -> None:
        now = datetime.now(STRATEGY_TZ)
        df = _event_df(now)
        nf = _build_news_filter(df, frozenset(), buffer_minutes=30)
        assert nf.is_safe_to_trade(check_time=now) is False

    def test_safe_after_buffer_expires(self) -> None:
        past = datetime.now(STRATEGY_TZ) - timedelta(hours=2)
        df = _event_df(past)
        nf = _build_news_filter(df, frozenset(), buffer_minutes=30)
        assert nf.is_safe_to_trade() is True

    def test_blocked_on_holiday_matching_currency(self) -> None:
        holidays: frozenset[tuple[str, date]] = frozenset({("USD", date.today())})
        nf = _build_news_filter(pd.DataFrame(), holidays)
        assert nf.is_safe_to_trade() is False

    def test_safe_on_holiday_for_different_currency(self) -> None:
        holidays: frozenset[tuple[str, date]] = frozenset({("EUR", date.today())})
        nf = _build_news_filter(pd.DataFrame(), holidays, currencies=["USD"])
        assert nf.is_safe_to_trade() is True

    def test_fail_open_allows_trading_on_missing_calendar(self) -> None:
        nf = _build_news_filter(pd.DataFrame(), frozenset(), fail_open=True)
        with patch.object(nf, "_load_calendar", return_value=False):
            assert nf.is_safe_to_trade() is True


class TestInvalidateCache:
    def test_clears_all_state(self) -> None:
        df = _event_df(datetime.now(STRATEGY_TZ) + timedelta(hours=1))
        nf = _build_news_filter(df, frozenset())
        assert nf._calendar_cache is not None
        nf.invalidate_cache()
        assert nf._calendar_cache is None
        assert nf._cache_timestamp is None
        assert len(nf._holiday_dates) == 0
        assert nf._high_impact_times_epoch is None
