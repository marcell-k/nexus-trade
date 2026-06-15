from __future__ import annotations

from nexus_trade.config.strategy import (
    BaseStrategyParams,
    ExecutionConfig,
    FiltersConfig,
    NewsFilterConfig,
    RiskConfig,
    SessionConfig,
    StrategyConfig,
    TradingHoursConfig,
)


class SMAParams(BaseStrategyParams):
    """SMA Crossover strategy parameters."""

    symbol: str = "EURUSD"
    backcandles: int = 100
    timeframe: str = "M15"
    timezone: str = "Europe/London"
    fast_period: int = 10
    slow_period: int = 20
    atr_period: int = 14
    atr_multiplier: float = 1.5
    risk_reward_ratio: float = 2.0
    volume_fixed_lots: float = 0.1


def get_config() -> StrategyConfig[SMAParams]:
    return StrategyConfig.build(
        name="sma_crossover",
        params=SMAParams(),
        execution=ExecutionConfig(
            magic_number=1001,
            deviation=50,
            comment_prefix="SMA",
        ),
        strategy_class="SMACrossoverStrategy",
        symbol="EURUSD",
        order_type="market",
        filters=FiltersConfig(
            news=NewsFilterConfig(
                enabled=False,
                currencies=["EUR", "USD"],
                buffer_minutes=15,
            ),
        ),
        trading_hours=TradingHoursConfig(
            enabled=True,
            timezone="Europe/London",
            sessions=[SessionConfig(start="08:00", end="22:00")],
        ),
        risk=RiskConfig(
            position_sizing_method="fractional",
            max_positions=1,
            max_trades=5,
            max_spread_points=10,
            max_slippage_points=5,
        ),
    )
