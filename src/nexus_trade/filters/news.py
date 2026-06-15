import logging
import os
import time
from collections.abc import Iterable
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from nexus_trade.core.data_handler import DataHandler
from nexus_trade.core.registry import STRATEGY_CONFIG_REGISTRY
from nexus_trade.core.state import SharedState

logger = logging.getLogger(__name__)


_REQUIRED_COLS: frozenset[str] = frozenset({"Date", "Time", "Currency", "Event", "Impact", "Type"})
_IMPACT_MAPPING: dict[str, str] = {"NONE": "None", "LOW": "Low", "MEDIUM": "Medium", "HIGH": "High"}


def preprocess_calendar_file(
    calendar_path: Path, broker_tz: ZoneInfo
) -> tuple[pd.DataFrame, frozenset[tuple[str, date]]]:
    """Parse and validate economic calendar CSV from MT5 export."""
    df = pd.read_csv(calendar_path, delimiter=",", dtype=str, encoding="utf-16")
    df = df.apply(lambda col: col.str.strip())

    if missing := _REQUIRED_COLS - set(df.columns):
        logger.error(f"CalParseFail reason=missing_cols | cols={sorted(missing)}")
        return pd.DataFrame(), frozenset()

    df = df.dropna(subset=["Date", "Time", "Currency", "Event"]).copy()
    df["Impact"] = df["Impact"].fillna("LOW")
    df["Type"] = df["Type"].fillna("Unknown")
    df["priority"] = df["Impact"].str.upper().map(_IMPACT_MAPPING).fillna("Low")
    df = df.rename(columns={"Event": "event_name", "Currency": "currency"})

    parsed_times = pd.to_datetime(df["Date"] + " " + df["Time"], format="%Y.%m.%d %H:%M", errors="coerce")
    valid = parsed_times.notna()
    if not valid.any():
        return pd.DataFrame(), frozenset()

    df = df.loc[valid].copy()
    df["time_broker"] = parsed_times.loc[valid].dt.tz_localize(broker_tz, ambiguous="NaT", nonexistent="shift_forward")

    holiday_mask = df["Type"].str.contains("Holiday", case=False, na=False)
    holidays = frozenset(
        zip(df.loc[holiday_mask, "currency"], df.loc[holiday_mask, "time_broker"].dt.date, strict=True)
    )

    logger.debug(f"CalPreproc rows={len(df)}")
    return df, holidays


class NewsFilter:
    """News/holiday filter for strategies."""

    CALENDAR_COLUMNS = (
        "time_broker",
        "time_strategy",
        "currency",
        "priority",
        "Type",
        "event_name",
    )

    def __init__(
        self,
        data_handler: DataHandler,
        strategy_name: str,
        cache_ttl_seconds: int = 3600 * 12,
        shared_state: SharedState | None = None,
        fail_open: bool = False,
    ) -> None:
        """
        Initialize news filter with strategy-specific configuration.

        Args:
            data_handler: Data handler instance (provides broker TZ and config)
            strategy_name: Strategy identifier for config lookup
            calendar_path: Override path to calendar CSV (default: MT5 path)
            cache_ttl_seconds: Calendar cache duration in seconds
            shared_state: Orchestrator shared state for calendar cache (optional)
            fail_open: Unable to load calendar


        """
        self.data_handler: DataHandler = data_handler
        self.strategy_name: str = strategy_name
        self.cache_ttl_seconds: int = cache_ttl_seconds
        self.shared_state: SharedState | None = shared_state
        self.calendar_path: Path = Path(os.getenv("MT5_CALENDAR_PATH", ""))
        self.fail_open: bool = fail_open

        # Load strategy configuration
        self._load_configuration()

        self._calendar_cache: pd.DataFrame | None = None
        self._cache_timestamp: float | None = None
        self._holiday_dates: frozenset[tuple[str, date]] = frozenset()
        self._all_holiday_dates: set[date] = set()
        self._holiday_dates_by_currency: dict[str, set[date]] = {}
        self._high_impact_times_epoch: np.ndarray | None = None
        self._event_index_cache: dict[tuple[str, tuple[str, ...] | None], np.ndarray] = {}

        logger.debug(
            f"NewsInit strat={strategy_name} | tz_b={self.broker_tz} | tz_s={self.strategy_tz} "
            f"| cur={self.filter_currencies or 'ALL'}"
        )

    def _load_configuration(self) -> None:
        """Load timezone and filter settings from strategy config."""
        self.broker_tz: ZoneInfo = self.data_handler.broker_tz
        config = STRATEGY_CONFIG_REGISTRY.get_config(self.strategy_name)
        tz_name: str | None = config.get("timezone")
        self.strategy_tz: ZoneInfo = ZoneInfo(tz_name) if tz_name else self.broker_tz
        self.symbol: str | None = config.get("symbol")
        self.enabled: bool = bool(config.get("news_filter_enabled"))
        self.filter_currencies: set[str] = set(config.get("currencies") or [])
        self.buffer_minutes: int = int(config.get("buffer_minutes") or 0)
        self._buffer_seconds: float = float(self.buffer_minutes * 60)

    def _is_cache_valid(self, cache_timestamp: float | None) -> bool:
        """Check if cache timestamp is within TTL."""
        if cache_timestamp is None:
            return False

        now_epoch = time.time()
        return (now_epoch - float(cache_timestamp)) < self.cache_ttl_seconds

    def _normalize_holidays(
        self, holidays_raw: Iterable[tuple[str | None, date | datetime | str | None]]
    ) -> frozenset[tuple[str, date]]:
        normalized: set[tuple[str, date]] = set()
        for item in holidays_raw or []:
            currency, holiday_date = item
            if currency is None or holiday_date is None:
                continue
            parsed_date = (
                holiday_date if isinstance(holiday_date, date) else pd.to_datetime(holiday_date, errors="coerce")
            )
            if pd.isna(parsed_date):
                continue
            normalized_date = parsed_date.date() if isinstance(parsed_date, datetime) else parsed_date
            normalized.add((str(currency), normalized_date))
        return frozenset(normalized)

    @staticmethod
    def _datetime_series_to_epoch_seconds(values: pd.Series) -> np.ndarray:
        dtype_unit = getattr(values.dtype, "unit", "ns")
        unit_divisor = {
            "s": 1.0,
            "ms": 1_000.0,
            "us": 1_000_000.0,
            "ns": 1_000_000_000.0,
        }.get(dtype_unit, 1_000_000_000.0)
        return values.astype("int64").to_numpy(dtype=np.float64) / unit_divisor

    def _set_holiday_caches(self, holidays: frozenset[tuple[str, date]]) -> None:
        self._holiday_dates = holidays
        self._all_holiday_dates = {holiday_date for _, holiday_date in holidays}
        holiday_map: dict[str, set[date]] = {}
        for currency, holiday_date in holidays:
            holiday_map.setdefault(currency, set()).add(holiday_date)
        self._holiday_dates_by_currency = holiday_map

    def _rebuild_event_indexes(self) -> None:
        self._event_index_cache = {}
        self._high_impact_times_epoch = np.empty(0, dtype=np.float64)
        if self._calendar_cache is None or self._calendar_cache.empty:
            return

        high_impact_mask = self._calendar_cache["priority"].eq("High")
        if self.filter_currencies:
            high_impact_mask &= self._calendar_cache["currency"].isin(self.filter_currencies)

        high_df = self._calendar_cache.loc[high_impact_mask, :].sort_values("time_strategy")
        if high_df.empty:
            return

        self._high_impact_times_epoch = self._datetime_series_to_epoch_seconds(high_df["time_strategy"])

    def _set_calendar_cache(self, calendar_df: pd.DataFrame, holidays: frozenset[tuple[str, date]]) -> None:
        if calendar_df.empty:
            self._calendar_cache = calendar_df.copy()
            self._cache_timestamp = time.time()
            self._set_holiday_caches(holidays)
            self._rebuild_event_indexes()
            return

        required_columns = {"time_broker", "currency", "priority", "Type", "event_name"}
        if not required_columns.issubset(calendar_df.columns):
            missing = sorted(required_columns - set(calendar_df.columns))
            raise ValueError(f"Calendar cache missing columns: {missing}")

        normalized_df = calendar_df.copy()
        if "time_strategy" not in normalized_df.columns:
            normalized_df["time_strategy"] = normalized_df["time_broker"].dt.tz_convert(self.strategy_tz)

        week_ago = pd.Timestamp.now(tz=self.strategy_tz) - pd.Timedelta(days=7)
        normalized_df = normalized_df.loc[normalized_df["time_strategy"] >= week_ago, list(self.CALENDAR_COLUMNS)]
        normalized_df = normalized_df.sort_values("time_strategy").reset_index(drop=True)

        self._calendar_cache = normalized_df
        self._cache_timestamp = time.time()
        self._set_holiday_caches(holidays)
        self._rebuild_event_indexes()

    def _normalize_check_time(self, check_time: datetime | None) -> datetime:
        if check_time is None:
            return datetime.now(self.strategy_tz)
        if check_time.tzinfo is None:
            localized = pd.Timestamp(check_time).tz_localize(
                self.strategy_tz, ambiguous=False, nonexistent="shift_forward"
            )
            return localized.to_pydatetime()
        return check_time.astimezone(self.strategy_tz)

    @staticmethod
    def _to_epoch_seconds(value: datetime) -> float:
        return float(value.timestamp())

    def _ensure_holiday_indexes(self) -> None:
        if not self._holiday_dates_by_currency and self._holiday_dates:
            holiday_map: dict[str, set[date]] = {}
            for currency, holiday_date in self._holiday_dates:
                holiday_map.setdefault(currency, set()).add(holiday_date)
            self._holiday_dates_by_currency = holiday_map

    def _resolve_currency_key(self, currencies: list[str] | None) -> tuple[str, ...] | None:
        if currencies:
            return tuple(sorted(set(currencies)))
        if self.filter_currencies:
            return tuple(sorted(self.filter_currencies))
        return None

    def _get_event_index(self, priority: str, currencies: list[str] | None) -> np.ndarray:
        """Return cached epoch-timestamp array for the given priority/currency filter."""
        if self._calendar_cache is None:
            return np.empty(0, dtype=np.float64)

        currency_key = self._resolve_currency_key(currencies)
        cache_key = (priority, currency_key)
        cached = self._event_index_cache.get(cache_key)
        if cached is not None:
            return cached
        event_view = self._calendar_cache[self._calendar_cache["priority"] == priority]
        if currency_key:
            event_view = event_view[event_view["currency"].isin(currency_key)]

        if event_view.empty:
            result = np.empty(0, dtype=np.float64)
        else:
            result = self._datetime_series_to_epoch_seconds(event_view.sort_values("time_strategy")["time_strategy"])

        self._event_index_cache[cache_key] = result
        return result

    def _restore_shared_cache(self) -> bool:
        """Restore calendar from shared state if available and valid."""
        if self.shared_state is None:
            return False

        shared_cache = self.shared_state.get("calendar_cache")
        shared_timestamp = self.shared_state.get("calendar_cache_timestamp", 0.0) or 0.0

        if shared_cache is None:
            return False

        if not self._is_cache_valid(shared_timestamp):
            return False

        restored_df = pd.DataFrame(shared_cache)
        if restored_df.empty:
            return False

        if "time_broker" not in restored_df.columns:
            logger.warning("CalCacheWarn reason=missing_time_broker")
            return False

        parsed_broker_time = pd.to_datetime(restored_df["time_broker"], utc=True, errors="coerce")
        invalid_broker_time = parsed_broker_time.isna()
        if invalid_broker_time.any():
            restored_df = restored_df.loc[~invalid_broker_time].copy()
            parsed_broker_time = parsed_broker_time.loc[~invalid_broker_time]
        if restored_df.empty:
            return False

        restored_df["time_broker"] = parsed_broker_time.dt.tz_convert(self.broker_tz)
        restored_df["time_strategy"] = restored_df["time_broker"].dt.tz_convert(self.strategy_tz)

        holidays = self._normalize_holidays(self.shared_state.get("calendar_holidays", []))
        try:
            self._set_calendar_cache(restored_df, holidays)
        except ValueError as exc:
            logger.warning(f"CalCacheReject err={exc}")
            return False

        cache_age = time.time() - float(shared_timestamp)
        logger.debug(f"CalCacheLoad evt={len(self._calendar_cache)} | age={cache_age:.1f}s")
        return True

    def _load_from_file(self) -> bool:
        """Load and cache calendar from CSV file."""
        if not self.calendar_path.exists():
            logger.warning(f"CalFileMissing path={self.calendar_path}")
            return False

        df, holidays = preprocess_calendar_file(self.calendar_path, self.broker_tz)
        if df.empty:
            return False

        df["time_strategy"] = df["time_broker"].dt.tz_convert(self.strategy_tz)
        self._set_calendar_cache(df, holidays)

        high_impact_count = int(self._calendar_cache["priority"].eq("High").sum())
        logger.info(
            f"CalLoad evt={len(self._calendar_cache)} | hi={high_impact_count} | hol={len(self._holiday_dates)}"
        )

        if self.shared_state is not None:
            self.shared_state["calendar_cache"] = self._calendar_cache.to_dict("records")
            self.shared_state["calendar_holidays"] = list(self._holiday_dates)
            self.shared_state["calendar_cache_timestamp"] = self._cache_timestamp or time.time()

        return True

    def should_trade(self) -> bool:
        """Check if trading is allowed based on news calendar."""
        if not self.enabled:
            return True
        return self.is_safe_to_trade()

    def is_safe_to_trade(self, check_time: datetime | None = None) -> bool:
        """Determine if trading is safe at given time."""
        check_time = self._normalize_check_time(check_time)
        if not self._load_calendar():
            logger.warning(f"NewsCalUnavailable strat={self.strategy_name} | fail_open={self.fail_open}")
            return self.fail_open

        self._ensure_holiday_indexes()
        check_date_strategy = check_time.date()

        if self.filter_currencies:
            for currency in self.filter_currencies:
                holidays_for_currency = self._holiday_dates_by_currency.get(currency)
                if holidays_for_currency and check_date_strategy in holidays_for_currency:
                    return False
        else:
            if check_date_strategy in self._all_holiday_dates:
                return False

        if self._high_impact_times_epoch is None:
            self._rebuild_event_indexes()
        if self._high_impact_times_epoch is None or self._high_impact_times_epoch.size == 0:
            return True

        check_time_epoch = self._to_epoch_seconds(check_time)
        lower_bound = check_time_epoch - self._buffer_seconds
        upper_bound = check_time_epoch + self._buffer_seconds

        first_idx = int(np.searchsorted(self._high_impact_times_epoch, lower_bound, side="left"))
        if first_idx >= self._high_impact_times_epoch.size:
            return True
        return bool(self._high_impact_times_epoch[first_idx] > upper_bound)

    def _load_calendar(self) -> bool:
        """Load economic calendar from shared cache or CSV file."""
        if self._calendar_cache is not None and self._is_cache_valid(self._cache_timestamp):
            return True

        if self._restore_shared_cache():
            return True

        return self._load_from_file()

    def invalidate_cache(self) -> None:
        """Force reload of calendar on next access."""
        self._calendar_cache = None
        self._cache_timestamp = None
        self._holiday_dates = frozenset()
        self._all_holiday_dates = set()
        self._holiday_dates_by_currency = {}
        self._high_impact_times_epoch = None
        self._event_index_cache = {}

    def get_next_event(self, priority: str = "High", currencies: list[str] | None = None) -> dict | None:
        """
        Get next upcoming event matching priority and currency filters.

        Args:
            priority: Event priority ('High', 'Medium', 'Low', 'None')
            currencies: Currency codes to filter (default: use self.filter_currencies)

        Returns:
            dict with event details, or None if no upcoming events

        """
        if not self._load_calendar():
            return None

        now_strategy = datetime.now(self.strategy_tz)
        now_epoch = self._to_epoch_seconds(now_strategy)
        event_times_epoch = self._get_event_index(priority, currencies)
        if event_times_epoch.size == 0:
            return None
        next_idx = int(np.searchsorted(event_times_epoch, now_epoch, side="left"))
        if next_idx >= event_times_epoch.size:
            return None

        # Rebuild filtered view on demand (cold path — called infrequently).
        currency_key = self._resolve_currency_key(currencies)
        view = self._calendar_cache[self._calendar_cache["priority"] == priority]
        if currency_key:
            view = view[view["currency"].isin(currency_key)]
        view = view.sort_values("time_strategy").reset_index(drop=True)
        if next_idx >= len(view):
            return None
        event = view.iloc[next_idx]
        minutes_until = float((event_times_epoch[next_idx] - now_epoch) / 60.0)

        return {
            "time_strategy": event["time_strategy"],
            "time_broker": event["time_broker"],
            "currency": event["currency"],
            "type": event["Type"],
            "event_name": event["event_name"],
            "priority": event["priority"],
            "minutes_until": minutes_until,
        }
