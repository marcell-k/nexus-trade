"""System-wide timing constants. Risk limits live in config/profiles/*.toml."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True, slots=True)
class SystemTimings:
    heartbeat_interval: int
    heartbeat_log_interval: int
    cache_staleness_threshold: int
    drawdown_refresh_interval_seconds: int
    max_strategy_offset_slots: int
    strategy_offset_divisor: float
    drawdown_history_start: datetime


SYSTEM_TIMINGS = SystemTimings(
    heartbeat_interval=60,
    heartbeat_log_interval=900,
    cache_staleness_threshold=60,
    drawdown_refresh_interval_seconds=30,
    max_strategy_offset_slots=6,
    strategy_offset_divisor=40.0,
    drawdown_history_start=datetime(2025, 1, 1),
)
