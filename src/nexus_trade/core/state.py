"""Type definitions for multiprocessing primitives, shared state, and MT5 data protocols."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypedDict, TypeVar

if TYPE_CHECKING:
    from datetime import date

    from nexus_trade.config.strategy import SessionConfig
    from nexus_trade.core.models import Order


T = TypeVar("T")


@dataclass
class TTLCache[T]:
    """Generic TTL-based single-value cache entry.

    Type parameter ``_T`` is the concrete type of the cached object, e.g.
    ``TTLCache[AccountInfo]`` or ``TTLCache[SymbolInfo]``.
    """

    value: T | None = None
    timestamp: float = 0.0

    def is_valid(self, ttl: float) -> bool:
        """Return True when a value is present and was stored within ``ttl`` seconds."""
        return self.value is not None and (time.time() - self.timestamp) < ttl

    def set(self, value: T) -> None:
        """Store ``value`` and refresh the timestamp."""
        self.value = value
        self.timestamp = time.time()

    def invalidate(self) -> None:
        """Clear the cached value and reset the timestamp."""
        self.value = None
        self.timestamp = 0.0


class PositionCacheEntry(TypedDict):
    """Single position entry stored in the cross-process position cache."""

    ticket: int
    symbol: str
    type: int
    volume: float
    price_open: float
    sl: float
    tp: float
    profit: float
    swap: float
    magic: int
    time: int


class SharedState(TypedDict, total=False):
    shutdown_flag: bool
    position_cache: dict[int, PositionCacheEntry]
    position_cache_timestamp: float
    heartbeats: dict[str, float]
    calendar_cache: list[dict[str, object]] | None
    calendar_cache_timestamp: float
    calendar_holidays: list[tuple[str, date]]
    daily_trade_counts: dict[str, int]
    daily_equity_high: float
    daily_drawdown: float
    daily_drawdown_current_equity: float
    daily_drawdown_peak_equity: float
    daily_drawdown_last_update: float
    daily_drawdown_initialized: bool
    daily_drawdown_cache_date: date | None
    max_drawdown: float
    max_drawdown_current_equity: float
    max_drawdown_peak_equity: float
    max_drawdown_last_update: float
    max_drawdown_initialized: bool
    drawdown_last_refresh: float
    hist_pnl_sum: float
    hist_peak_equity: float
    last_equity_update: float


_POSITION_FIELDS: tuple[str, ...] = (
    "ticket",
    "symbol",
    "type",
    "volume",
    "price_open",
    "sl",
    "tp",
    "profit",
    "swap",
    "magic",
    "time",
)
_POSITION_DEFAULTS: PositionCacheEntry = {
    "ticket": 0,
    "symbol": "",
    "type": 0,
    "volume": 0.0,
    "price_open": 0.0,
    "sl": 0.0,
    "tp": 0.0,
    "profit": 0.0,
    "swap": 0.0,
    "magic": 0,
    "time": 0,
}
_ORDER_FIELDS: tuple[str, ...] = ("ticket", "symbol", "type", "magic")


@dataclass(slots=True, frozen=True)
class OrderSnapshot:
    """Lightweight pending-order snapshot for bracket cleanup paths."""

    ticket: int
    symbol: str
    type: int
    magic: int


@dataclass(slots=True, frozen=True)
class PartialClosePositionSnapshot:
    """Lightweight position snapshot for partial-close logging paths."""

    ticket: int
    type: int
    symbol: str
    swap: float = 0.0


def normalize_order(
    order: dict[str, object] | Order,
    required_fields: tuple[str, ...] = _ORDER_FIELDS,
) -> OrderSnapshot:
    """Convert MT5 order object or dict to standardized ``OrderSnapshot``."""
    if isinstance(order, dict):
        return OrderSnapshot(
            ticket=int(order["ticket"]),
            symbol=str(order["symbol"]),
            type=int(order["type"]),
            magic=int(order["magic"]),
        )
    return OrderSnapshot(**{field: getattr(order, field) for field in required_fields})


class DrawdownThreshold(TypedDict):
    drawdown_pct: float
    risk_multiplier: float


class AdaptiveSizingConfig(TypedDict):
    enabled: bool
    scope: str
    drawdown_thresholds: list[DrawdownThreshold]


class GlobalRiskPolicy(TypedDict):
    max_total_positions: int
    max_daily_drawdown_pct: float
    strategy_risk: dict[str, float]
    log_root: str
    max_drawdown_pct: float
    max_daily_trades: int
    initial_balance: int
    adaptive_sizing: AdaptiveSizingConfig


class StrategyRiskConfig(TypedDict):
    risk_per_trade: float


class EntryMetadata(TypedDict, total=False):
    submission_time: float
    volume_multiplier: float | None
    ticket: int | None
    position_snapshot: PositionCacheEntry | None
    expected_entry_price: float
    opening_sl: float | None
    expected_buy_entry: float | None
    expected_sell_entry: float | None
    buy_sl: float | None
    sell_sl: float | None


class _RawStrategyConfigRequired(TypedDict):
    symbol: str
    timeframe: str
    number_of_bars: int
    magic_number: int
    timezone: str
    timeframe_minutes: int


class RawStrategyConfig(_RawStrategyConfigRequired, total=False):
    """Parsed strategy config dict produced by DataHandler._load_strategy_config."""

    filter_enabled: bool | None
    sessions: list[SessionConfig]
    deviation: int | None
    news_filter_enabled: bool | None
    currencies: list[str] | None
    buffer_minutes: int | None


class ReconciledTrade(TypedDict):
    trade_id: int
    expected_entry_price: float
    opening_sl: float | None
    volume_multiplier: float | None
