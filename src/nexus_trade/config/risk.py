"""System-wide timing constants. Risk limits live in config/profiles/*.toml."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SystemTimings:
    heartbeat_interval: int
    heartbeat_log_interval: int
    cache_staleness_threshold: int
    drawdown_refresh_interval_seconds: int
    max_strategy_offset_slots: int
    strategy_offset_divisor: float


SYSTEM_TIMINGS = SystemTimings(
    heartbeat_interval=60,
    heartbeat_log_interval=900,
    cache_staleness_threshold=60,
    drawdown_refresh_interval_seconds=30,
    max_strategy_offset_slots=6,
    strategy_offset_divisor=40.0,
)
