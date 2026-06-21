"""System-wide timing constants. Risk limits live in config/profiles/*.toml."""

from __future__ import annotations

from dataclasses import dataclass

# runner
SL_TP_SNAP_TOLERANCE: float = 0.001
BREAKEVEN_HALF_BAND_RATIO: float = 0.0005
PENDING_ORDER_EXPIRY_GRACE_SECONDS: int = 30

# orchestrator
GRACE_SECONDS = 90

# data handler
LOOKBACK = 5
TOLERANCE_SECONDS = 2

# async logger
MAX_QUEUE_SIZE = 100


@dataclass(frozen=True, slots=True)
class SystemTimings:
    heartbeat_interval: int
    heartbeat_log_interval: int
    cache_staleness_threshold: int
    drawdown_refresh_interval_seconds: int
    drawdown_cache_ttl_seconds: int
    max_strategy_offset_slots: int
    strategy_offset_divisor: float
    symbol_spec_cache_ttl_seconds: float
    account_info_cache_ttl_seconds: int
    risk_manager_symbol_cache_ttl_seconds: int
    news_calendar_cache_ttl_seconds: int
    connect_backoff_seconds: tuple[int, ...]
    order_send_retry_backoff_seconds: tuple[float, ...]


SYSTEM_TIMINGS = SystemTimings(
    heartbeat_interval=60,
    heartbeat_log_interval=900,
    cache_staleness_threshold=60,
    drawdown_refresh_interval_seconds=30,
    drawdown_cache_ttl_seconds=10,
    max_strategy_offset_slots=6,
    strategy_offset_divisor=40.0,
    symbol_spec_cache_ttl_seconds=300.0,
    account_info_cache_ttl_seconds=900,
    risk_manager_symbol_cache_ttl_seconds=14400,
    news_calendar_cache_ttl_seconds=3600 * 12,
    connect_backoff_seconds=(1, 2, 4, 8, 16),
    order_send_retry_backoff_seconds=(0.025, 0.05, 0.10),
)
