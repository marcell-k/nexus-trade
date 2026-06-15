"""Integration tests for NewsFilter and preprocess_calendar_file."""

from __future__ import annotations

import textwrap
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from nexus_trade.filters.news import NewsFilter, preprocess_calendar_file

#  preprocess_calendar_file

BROKER_TZ = ZoneInfo("Etc/GMT-3")

VALID_CSV = textwrap.dedent("""\
    Date,Time,Currency,Event,Impact,Type
    2025.06.16,14:30,USD,Non-Farm Payrolls,HIGH,Economic
    2025.06.16,08:00,EUR,ECB Rate Decision,HIGH,Economic
    2025.06.17,00:00,USD,Independence Day,NONE,Holiday
    2025.06.16,12:00,GBP,CPI y/y,MEDIUM,Economic
    2025.06.16,10:00,JPY,BOJ Press Conference,LOW,Economic
""")


@pytest.fixture
def calendar_csv(tmp_path: Path) -> Path:
    path = tmp_path / "calendar.csv"
    path.write_text(VALID_CSV, encoding="utf-16")
    return path


class TestPreprocessCalendarFile:
    def test_parses_valid_csv(self, calendar_csv: Path) -> None:
        df, holidays = preprocess_calendar_file(calendar_csv, BROKER_TZ)
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
            INVALID,99:99,USD,Bad Event,HIGH,Economic
            2025.06.16,14:30,USD,NFP,HIGH,Economic
        """)
        path = tmp_path / "mixed.csv"
        path.write_text(content, encoding="utf-16")
        df, _ = preprocess_calendar_file(path, BROKER_TZ)
        assert len(df) == 1

    def test_event_name_column_renamed(self, calendar_csv: Path) -> None:
        df, _ = preprocess_calendar_file(calendar_csv, BROKER_TZ)
        assert "event_name" in df.columns
        assert "Event" not in df.columns

    def test_currency_column_renamed(self, calendar_csv: Path) -> None:
        df, _ = preprocess_calendar_file(calendar_csv, BROKER_TZ)
        assert "currency" in df.columns


#  NewsFilter.is_safe_to_trade


def _make_news_filter(
    calendar_df: pd.DataFrame,
    holidays: frozenset,
    *,
    buffer_minutes: int = 15,
    currencies: list[str] | None = None,
    strategy_tz: ZoneInfo = ZoneInfo("UTC"),
) -> NewsFilter:
    """Build a NewsFilter with a pre-loaded calendar (bypasses file I/O)."""
    from unittest.mock import MagicMock, patch

    from nexus_trade.core.data_handler import DataHandler

    dh = MagicMock(spec=DataHandler)
    dh.broker_tz = BROKER_TZ

    with patch("nexus_trade.filters.news.STRATEGY_CONFIG_REGISTRY") as reg:
        reg.get_config.return_value = {
            "symbol": "EURUSD",
            "timezone": "UTC",
            "news_filter_enabled": True,
            "currencies": currencies or ["USD"],
            "buffer_minutes": buffer_minutes,
        }
        reg.get_tz.return_value = strategy_tz

        nf = NewsFilter(
            data_handler=dh,
            strategy_name="test",
            shared_state=None,
            fail_open=False,
        )

    # Inject pre-loaded calendar
    if not calendar_df.empty:
        calendar_df["time_strategy"] = calendar_df["time_broker"].dt.tz_convert(strategy_tz)
    nf._set_calendar_cache(calendar_df, holidays)
    return nf


class TestNewsFilterIsSafeToTrade:
    def _make_event_df(self, event_time: datetime, priority: str = "High") -> pd.DataFrame:
        return pd.DataFrame(
            {
                "time_broker": [pd.Timestamp(event_time).tz_convert(BROKER_TZ)],
                "currency": ["USD"],
                "priority": [priority],
                "Type": ["Economic"],
                "event_name": ["NFP"],
            }
        )

    def test_safe_when_no_events(self) -> None:
        nf = _make_news_filter(pd.DataFrame(), frozenset())
        assert nf.is_safe_to_trade() is True

    def test_blocked_within_buffer(self) -> None:
        now = datetime.now(ZoneInfo("UTC"))
        event_time = now  # event is right now
        df = self._make_event_df(event_time)
        df["time_broker"] = df["time_broker"].dt.tz_convert(BROKER_TZ)
        nf = _make_news_filter(df, frozenset(), buffer_minutes=30)
        assert nf.is_safe_to_trade(check_time=now) is False

    def test_safe_after_buffer_expires(self) -> None:

        past_event = datetime.now(ZoneInfo("UTC")).replace(tzinfo=ZoneInfo("UTC")) - __import__("datetime").timedelta(
            hours=2
        )
        df = self._make_event_df(past_event)
        df["time_broker"] = df["time_broker"].dt.tz_convert(BROKER_TZ)
        nf = _make_news_filter(df, frozenset(), buffer_minutes=30)
        assert nf.is_safe_to_trade() is True

    def test_blocked_on_holiday_matching_currency(self) -> None:
        today = date.today()
        holidays: frozenset[tuple[str, date]] = frozenset({("USD", today)})
        nf = _make_news_filter(pd.DataFrame(), holidays)
        assert nf.is_safe_to_trade() is False

    def test_safe_on_holiday_for_different_currency(self) -> None:
        today = date.today()
        holidays: frozenset[tuple[str, date]] = frozenset({("EUR", today)})
        nf = _make_news_filter(pd.DataFrame(), holidays, currencies=["USD"])
        assert nf.is_safe_to_trade() is True

    def test_should_trade_returns_true_when_disabled(self) -> None:
        from unittest.mock import MagicMock, patch

        from nexus_trade.core.data_handler import DataHandler

        dh = MagicMock(spec=DataHandler)
        dh.broker_tz = BROKER_TZ

        with patch("nexus_trade.filters.news.STRATEGY_CONFIG_REGISTRY") as reg:
            reg.get_config.return_value = {
                "symbol": "EURUSD",
                "timezone": "UTC",
                "news_filter_enabled": False,
                "currencies": [],
                "buffer_minutes": 15,
            }
            reg.get_tz.return_value = ZoneInfo("UTC")
            nf = NewsFilter(data_handler=dh, strategy_name="test")

        assert nf.should_trade() is True

    def test_fail_open_returns_true_on_missing_calendar(self) -> None:
        from unittest.mock import MagicMock, patch

        from nexus_trade.core.data_handler import DataHandler

        dh = MagicMock(spec=DataHandler)
        dh.broker_tz = BROKER_TZ

        with patch("nexus_trade.filters.news.STRATEGY_CONFIG_REGISTRY") as reg:
            reg.get_config.return_value = {
                "symbol": "EURUSD",
                "timezone": "UTC",
                "news_filter_enabled": True,
                "currencies": ["USD"],
                "buffer_minutes": 15,
            }
            reg.get_tz.return_value = ZoneInfo("UTC")
            nf = NewsFilter(
                data_handler=dh,
                strategy_name="test",
                fail_open=True,
            )

        # No calendar loaded → fail_open=True means allow trading
        with patch.object(nf, "_load_calendar", return_value=False):
            assert nf.is_safe_to_trade() is True


class TestInvalidateCache:
    def test_clears_all_state(self) -> None:
        from datetime import timedelta

        future = datetime.now(ZoneInfo("UTC")) + timedelta(hours=1)
        df = pd.DataFrame(
            {
                "time_broker": [pd.Timestamp(future).tz_convert(BROKER_TZ)],
                "currency": ["USD"],
                "priority": ["High"],
                "Type": ["Economic"],
                "event_name": ["NFP"],
            }
        )
        nf = _make_news_filter(df, frozenset())
        assert nf._calendar_cache is not None
        nf.invalidate_cache()
        assert nf._calendar_cache is None
        assert nf._cache_timestamp is None
        assert len(nf._holiday_dates) == 0
        assert nf._high_impact_times_epoch is None
