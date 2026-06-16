from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Protocol, TypedDict

if TYPE_CHECKING:
    from nexus_trade.config.strategy import SessionConfig


class PositionType(Enum):
    BUY = "BUY"
    SELL = "SELL"


class MT5Tick(Protocol):
    @property
    def time(self) -> int: ...
    @property
    def bid(self) -> float: ...
    @property
    def ask(self) -> float: ...
    @property
    def last(self) -> float: ...
    @property
    def volume(self) -> int: ...
    @property
    def time_msc(self) -> int: ...
    @property
    def flags(self) -> int: ...
    @property
    def volume_real(self) -> float: ...


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


@dataclass(slots=True, frozen=True)
class PartialClosePositionSnapshot:
    """Lightweight position snapshot for partial-close logging paths."""

    ticket: int
    type: int
    symbol: str
    swap: float = 0.0


@dataclass(slots=True, frozen=True)
class OrderSnapshot:
    """Lightweight pending-order snapshot for bracket cleanup paths."""

    ticket: int
    symbol: str
    type: int
    magic: int


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


class ReconciledTrade(TypedDict):
    trade_id: int
    expected_entry_price: float
    opening_sl: float | None
    volume_multiplier: float | None


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
