from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict, TypeVar

if TYPE_CHECKING:
    from datetime import date

    from nexus_trade.core.types import PositionCacheEntry

T = TypeVar("T")


class _SharedStateRequired(TypedDict):
    shutdown_flag: bool
    position_cache: dict[int, PositionCacheEntry]
    position_cache_timestamp: float
    daily_drawdown: float
    max_drawdown: float
    daily_trade_counts: dict[str, int]


class SharedState(_SharedStateRequired, total=False):
    calendar_cache: list[dict[str, object]] | None
    calendar_cache_timestamp: float
    calendar_holidays: list[tuple[str, date]]
    daily_equity_high: float
    daily_drawdown_current_equity: float
    daily_drawdown_peak_equity: float
    daily_drawdown_last_update: float
    daily_drawdown_initialized: bool
    daily_drawdown_cache_date: date | None
    max_drawdown_current_equity: float
    max_drawdown_peak_equity: float
    max_drawdown_last_update: float
    max_drawdown_initialized: bool
    drawdown_last_refresh: float
    hist_pnl_sum: float
    hist_peak_equity: float
    last_equity_update: float
