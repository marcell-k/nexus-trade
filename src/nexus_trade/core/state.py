from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict, TypeVar

from nexus_trade.core.types import OrderSnapshot

if TYPE_CHECKING:
    from datetime import date

    from MetaTrader5 import TradeOrder

    from nexus_trade.core.types import PositionCacheEntry

T = TypeVar("T")


class _SharedStateRequired(TypedDict):
    shutdown_flag: bool
    position_cache: dict[int, PositionCacheEntry]
    position_cache_timestamp: float
    heartbeats: dict[str, float]


class SharedState(_SharedStateRequired, total=False):
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


def normalize_order(order: TradeOrder) -> OrderSnapshot:
    """Convert MT5 order namedtuple to standardized ``OrderSnapshot``."""
    return OrderSnapshot(
        ticket=int(order.ticket),
        symbol=str(order.symbol),
        type=int(order.type),
        magic=int(order.magic),
    )
