from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from datetime import date, datetime
from datetime import time as dt_time
from typing import TYPE_CHECKING

import MetaTrader5 as mt

from nexus_trade.config.timings import SYSTEM_TIMINGS
from nexus_trade.core.symbol import SYMBOL_SPEC_CACHE, SymbolSpec
from nexus_trade.core.types import DrawdownThreshold, GlobalRiskPolicy, NewsEvent, TTLCache
from nexus_trade.filters.costs import MarketCostCalculator
from nexus_trade.filters.news import NewsFilter

if TYPE_CHECKING:
    from zoneinfo import ZoneInfo

    from MetaTrader5 import AccountInfo

    from nexus_trade.config.strategy import BaseStrategyParams, RiskConfig, SessionConfig, StrategyConfig
    from nexus_trade.core.data_handler import DataHandler
    from nexus_trade.core.protocols import AtomicInt, StrategyRunnerProtocol
    from nexus_trade.core.state import SharedState


logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    can_trade: bool
    reason: str
    volume: float = 0.0


class RiskManager:
    """
    Centralized risk management with NewsFilter integration.

    Risk Layers (ordered by cost):
    0. Global limits (atomic counters - <0.001ms)
    1. Strategy limits (local counter - <0.001ms)
    2. NewsFilter (cached datetime comparisons - <0.1ms)
    3. Market conditions (MT5 tick fetch - 1-5ms)
    4. Position sizing (MT5 symbol_info - 1-5ms)
    """

    def __init__(
        self,
        strategy_config: StrategyConfig[BaseStrategyParams],
        global_policy: GlobalRiskPolicy,
        shared_state: SharedState,
        global_trade_count: AtomicInt,
        global_position_count: AtomicInt,
        data_handler: DataHandler,
        broker_tz: ZoneInfo,
        strategy_runner: StrategyRunnerProtocol | None = None,
    ) -> None:
        self.strategy_config: StrategyConfig[BaseStrategyParams] = strategy_config
        self.risk_config: RiskConfig = strategy_config.risk
        self.global_policy: GlobalRiskPolicy = global_policy
        self.shared_state: SharedState = shared_state
        self.global_trade_count: AtomicInt = global_trade_count
        self.global_position_count: AtomicInt = global_position_count
        self.strategy_name: str = strategy_config.name
        self.data_handler: DataHandler = data_handler
        self.broker_tz: ZoneInfo = broker_tz
        self.strategy_runner: StrategyRunnerProtocol | None = strategy_runner

        self._drawdown_refresh_lock: threading.Lock = threading.Lock()

        self.news_filter: NewsFilter | None = self._create_news_filter()
        self.cost_calculator: MarketCostCalculator = MarketCostCalculator(
            max_spread_points=self.risk_config.max_spread_points,
            max_slippage_points=self.risk_config.max_slippage_points,
        )

        self._sorted_drawdown_thresholds: list[DrawdownThreshold] = sorted(
            global_policy["adaptive_sizing"]["drawdown_thresholds"],
            key=lambda x: x["drawdown_pct"],
            reverse=True,
        )

        self._account_cache: TTLCache[AccountInfo] = TTLCache()
        self._drawdown_cache: TTLCache[tuple[float, float]] = TTLCache()

        logger.debug(
            f"RiskInit strat={self.strategy_name} | spr_max={self.cost_calculator.max_spread_points:.1f} | "
            f"slip_max={self.cost_calculator.max_slippage_points:.1f} | tz={self.broker_tz}"
        )

    def _midnight_today(self) -> datetime:
        """Return midnight in broker timezone (timezone-aware)."""
        now_broker = datetime.now(self.broker_tz)
        return now_broker.replace(hour=0, minute=0, second=0, microsecond=0)

    def _broker_today(self) -> date:
        return datetime.now(self.broker_tz).date()

    def _create_news_filter(self) -> NewsFilter | None:
        news_config = self.strategy_config.filters.news
        if not news_config.enabled:
            logger.info(f"NewsCfg strat={self.strategy_name} | enabled=False")
            return None

        nf = NewsFilter(
            data_handler=self.data_handler,
            strategy_name=self.strategy_name,
            shared_state=self.shared_state,
        )
        logger.debug(
            f"NewsInit strat={self.strategy_name} | cur={nf.filter_currencies or 'ALL'} | buf_min={nf.buffer_minutes}"
        )
        return nf

    def validate_trade(
        self,
        strategy_name: str,
        symbol: str,
        expected_price: float,
        sl_price: float,
        signal: int,
    ) -> ValidationResult:
        """Validate trade with layered risk checks (ordered by computational cost)."""
        result = self.check_global_risk()
        if not result.can_trade:
            return result
        result = self._check_trading_hours(strategy_name)
        if not result.can_trade:
            return result
        result = self._check_news_filter(strategy_name)
        if not result.can_trade:
            return result
        result = self._check_market_conditions(symbol, expected_price, signal)
        if not result.can_trade:
            return result
        volume = self.calculate_position_size(symbol, expected_price, sl_price, strategy_name)
        if volume <= 0:
            return ValidationResult(False, "Position size invalid")
        reserved_count = self._check_and_reserve_position_slot()
        if reserved_count is None:
            logger.warning(f"PosSlotRej strat={self.strategy_name} | reason=limit_reached_at_reserve")
            return ValidationResult(False, "Global position limit reached")

        logger.debug(
            f"PosSlotRes strat={self.strategy_name} | cnt={reserved_count}/{self.global_policy['max_total_positions']}"
        )
        return ValidationResult(True, "All checks passed", volume=volume)

    def _check_news_filter(self, strategy_name: str) -> ValidationResult:
        if self.news_filter is None or self.news_filter.should_trade():
            return ValidationResult(True, "NewsFilter passed")

        next_event = self.news_filter.get_next_event()
        event_detail = self._format_news_event(next_event) if next_event else ""
        logger.info(f"Reject strat={strategy_name} | reason=news_filter{event_detail}")
        return ValidationResult(False, f"High-impact news within {self.news_filter.buffer_minutes}-min buffer")

    def _check_market_conditions(self, symbol: str, expected_price: float, signal: int) -> ValidationResult:
        if signal not in (1, -1):
            logger.debug(f"MktCheckSkip strat={self.strategy_name} | sym={symbol} | reason=bracket_order")
            return ValidationResult(True, "Bracket order")

        condition = self.cost_calculator.validate_market_conditions(
            symbol=symbol, expected_price=expected_price, is_buy=(signal == 1), cached_symbol_info=None
        )

        if not condition.is_valid:
            logger.warning(
                f"MktCheckFail strat={self.strategy_name} | sym={symbol} | reason={condition.reason} | "
                f"spr_pts={condition.spread_points:.1f} | spr_px={condition.spread_price:.5f} | "
                f"slip_pts={condition.slippage_points:.1f} | slip_px={condition.slippage_price:.5f}"
            )
            return ValidationResult(False, condition.reason)

        logger.debug(
            f"MktCheckOK strat={self.strategy_name} | sym={symbol} | "
            f"spr_pts={condition.spread_points:.1f} | slip_pts={condition.slippage_points:.1f}"
        )
        return ValidationResult(True, "Market conditions OK")

    def _check_and_reserve_position_slot(self) -> int | None:
        with self.global_position_count.get_lock():
            if self.global_position_count.value >= self.global_policy["max_total_positions"]:
                return None
            self.global_position_count.value += 1
            return self.global_position_count.value

    def _format_news_event(self, event: NewsEvent) -> str:
        currency: str = event.get("currency", "N/A")
        event_name: str = event.get("event_name", "Unknown")
        time_val = event.get("time_strategy")
        time_str = time_val.strftime("%H:%M %Z") if isinstance(time_val, (datetime, dt_time)) else "N/A"
        return f" (Next: {currency} {event_name} at {time_str})"

    def release_position_reservation(self, reason: str = "execution_failed") -> None:
        with self.global_position_count.get_lock():
            if self.global_position_count.value > 0:
                self.global_position_count.value -= 1
                logger.debug(
                    f"PosSlotRel strat={self.strategy_name} | reason={reason} | "
                    f"cnt={self.global_position_count.value}/{self.global_policy['max_total_positions']}"
                )
            else:
                logger.warning(f"PosSlotRelWarn strat={self.strategy_name} | reason=counter_already_zero")

    def check_global_risk(self) -> ValidationResult:
        if self.global_position_count.value >= self.global_policy["max_total_positions"]:
            return ValidationResult(False, "Global position limit reached")
        if self.global_trade_count.value >= self.global_policy["max_daily_trades"]:
            return ValidationResult(False, "Daily trade limit reached")

        if not self._drawdown_cache.is_valid(SYSTEM_TIMINGS.drawdown_refresh_interval_seconds):
            self._drawdown_cache.set(
                (
                    float(self.shared_state["daily_drawdown"]),
                    float(self.shared_state["max_drawdown"]),
                )
            )

        assert self._drawdown_cache.value is not None
        daily_dd, max_dd = self._drawdown_cache.value
        if daily_dd > self.global_policy["max_daily_drawdown_pct"]:
            return ValidationResult(False, f"Daily drawdown {daily_dd * 100:.1f}%")
        if max_dd > self.global_policy["max_drawdown_pct"]:
            return ValidationResult(False, f"Max drawdown {max_dd * 100:.1f}%")
        return ValidationResult(True, "Global checks passed")

    def check_strategy_limits(self, strategy_name: str) -> ValidationResult:
        """Check per-strategy position and trade limits."""
        magic = self.strategy_config.execution.magic_number

        current_positions = self._get_strategy_position_count(strategy_name, magic)
        if current_positions is None:
            return ValidationResult(False, "MT5 API error: positions_get() returned None")

        max_positions = self.risk_config.max_positions
        if current_positions >= max_positions:
            logger.warning(
                f"StratLimitHit strat={strategy_name} | typ=positions | cur={current_positions} | max={max_positions}"
            )
            return ValidationResult(False, "Strategy position limit")

        daily_trade_counts: dict[str, int] = self.shared_state["daily_trade_counts"]
        strategy_trades: int = daily_trade_counts.get(strategy_name, 0)
        max_trades = self.risk_config.max_trades

        if strategy_trades >= max_trades:
            logger.warning(
                f"StratLimitHit strat={strategy_name} | typ=daily_trades | cur={strategy_trades} | max={max_trades}"
            )
            return ValidationResult(False, "Strategy daily trade limit")

        logger.debug(
            f"StratLimitOK strat={strategy_name} | pos={current_positions}/{max_positions} | "
            f"tr={strategy_trades}/{max_trades} | m={magic}"
        )
        return ValidationResult(True, "Strategy checks passed")

    def _get_strategy_position_count(self, strategy_name: str, magic: int) -> int | None:
        if self.strategy_runner is not None:
            logger.debug(f"PosCountSrc strat={strategy_name} | src=local_counter")
            return self.strategy_runner.local_position_count

        logger.warning(f"PosCountSrc strat={strategy_name} | src=mt5_query | reason=no_runner")
        positions = mt.positions_get()
        if positions is None:
            logger.error(f"PosCountFail strat={strategy_name} | op=positions_get | err={mt.last_error()}")
            return None
        return sum(1 for pos in positions if pos.magic == magic)

    def _get_account_info_cached(self) -> AccountInfo | None:
        if self._account_cache.is_valid(SYSTEM_TIMINGS.account_info_cache_ttl_seconds):
            return self._account_cache.value
        result = mt.account_info()
        if result is not None:
            self._account_cache.set(result)
        return result

    def calculate_position_size(self, symbol: str, entry: float, sl: float, strategy_name: str) -> float:
        """Calculate position size: volume = R / (d_SL / tick_size x tick_value)."""
        symbol_info = SYMBOL_SPEC_CACHE.get_spec(symbol)
        if symbol_info is None:
            logger.error(f"PosSizeFail sym={symbol} | reason=symbol_info_unavailable")
            return 0.0

        sl_distance = abs(entry - sl)
        if sl_distance == 0:
            logger.error(f"PosSizeFail strat={strategy_name} | reason=zero_sl_distance")
            return 0.0

        tick_value = symbol_info.tick_value or (symbol_info.tick_size * symbol_info.contract_size)
        if tick_value == 0:
            logger.error(
                f"PosSizeFail strat={strategy_name} | sym={symbol} | reason=tick_value_zero | "
                f"tick_size={symbol_info.tick_size} | contract_size={symbol_info.contract_size}"
            )
            return 0.0

        ticks = sl_distance / symbol_info.tick_size
        strategy_risk = self.global_policy["strategy_risk"][strategy_name]

        if strategy_risk["method"] == "fixed":
            volume = strategy_risk["risk_value"] / (ticks * tick_value)
            logger.debug(
                f"PosSizeFixed strat={strategy_name} | risk=${strategy_risk['risk_value']:.2f} | "
                f"sl_dist={sl_distance:.5f} | vol={volume:.4f}"
            )
            return self._normalize_volume(volume, symbol_info)

        account_info = self._get_account_info_cached()
        if account_info is None:
            logger.error(f"PosSizeFail strat={strategy_name} | reason=account_info_unavailable")
            return 0.0

        risk_multiplier = self._get_adaptive_risk_multiplier(strategy_name)
        adjusted_risk: float = account_info.balance * (strategy_risk["risk_value"] / 100.0) * risk_multiplier
        volume = adjusted_risk / (ticks * tick_value)
        return self._normalize_volume(volume, symbol_info)

    def _normalize_volume(self, volume: float, symbol_info: SymbolSpec) -> float:
        step = symbol_info.volume_step
        volume = round(volume / step) * step
        return max(symbol_info.volume_min, min(volume, symbol_info.volume_max))

    def _get_adaptive_risk_multiplier(self, strategy_name: str) -> float:
        if not self.global_policy["adaptive_sizing"]["enabled"]:
            return 1.0
        current_drawdown: float = float(self.shared_state["max_drawdown"])
        for threshold in self._sorted_drawdown_thresholds:
            if current_drawdown >= threshold["drawdown_pct"]:
                multiplier = threshold["risk_multiplier"]
                logger.info(
                    f"AdaptiveRisk strat={strategy_name} | dd={current_drawdown * 100:.2f}% | "
                    f"thr={threshold['drawdown_pct'] * 100:.1f}% | mul={multiplier:.2f}"
                )
                return multiplier
        return 1.0

    def validate_position_size(self, symbol: str, volume: float) -> float | None:
        symbol_info = SYMBOL_SPEC_CACHE.get_spec(symbol)
        if symbol_info is None:
            logger.error(f"PosNormFail sym={symbol} | reason=symbol_info_unavailable")
            return None
        return self._normalize_volume(volume, symbol_info) if volume >= symbol_info.volume_min else None

    def _check_trading_hours(self, strategy_name: str) -> ValidationResult:
        th = self.strategy_config.trading_hours
        if th is None or not th.enabled or not th.sessions:
            return ValidationResult(True, "TradingHours disabled")

        from zoneinfo import ZoneInfo

        tz = ZoneInfo(th.timezone)
        now: dt_time = datetime.now(tz).time().replace(second=0, microsecond=0)

        for session in th.sessions:
            if self._time_in_session(now, session):
                return ValidationResult(True, "TradingHours passed")

        logger.debug(f"TradingHoursBlock strat={strategy_name} | now={now.strftime('%H:%M')} | tz={th.timezone}")
        return ValidationResult(False, "outside_trading_hours")

    @staticmethod
    def _time_in_session(now: dt_time, session: SessionConfig) -> bool:
        start_h, start_m = map(int, session.start.split(":"))
        end_h, end_m = map(int, session.end.split(":"))
        start = dt_time(start_h, start_m)
        end = dt_time(end_h, end_m)
        if end >= start:
            return start <= now <= end
        return now >= start or now <= end
