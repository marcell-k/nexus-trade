from __future__ import annotations

from typing import Literal

from pydantic import Field, computed_field

from nexus_trade.config._base import FrozenModel


class ExecutionConfig(FrozenModel):
    magic_number: int = Field(gt=0)
    deviation: int = Field(ge=0, default=100)
    min_market_threshold_points: int = Field(default=0, ge=0)


class NewsFilterConfig(FrozenModel):
    enabled: bool = False
    currencies: list[str] = Field(default_factory=list)
    buffer_minutes: int = Field(default=15, ge=0)


class FiltersConfig(FrozenModel):
    market_regime_enabled: bool = False
    news: NewsFilterConfig = Field(default_factory=NewsFilterConfig)


class SessionConfig(FrozenModel):
    start: str = Field(pattern=r"^\d{2}:\d{2}$")
    end: str = Field(pattern=r"^\d{2}:\d{2}$")


class TradingHoursConfig(FrozenModel):
    enabled: bool = False
    timezone: str = "UTC"
    sessions: list[SessionConfig] = Field(default_factory=list)


class RiskConfig(FrozenModel):
    max_positions: int = Field(default=1, gt=0)
    max_trades: int = Field(default=1, gt=0)
    max_spread_points: float = Field(default=100, ge=0)
    max_slippage_points: float = Field(default=100, ge=0)


class BaseStrategyParams(FrozenModel):
    """
    Typed base for all strategy parameter models.

    Extend by subclassing and declaring additional fields — never instantiate directly
    with undeclared kwargs. Inherits frozen=True, strict=True, extra="forbid" from FrozenModel.

    Example::

        class MyParams(BaseStrategyParams):
            fast_period: int = 10
            slow_period: int = 20
            atr_multiplier: float = Field(default=1.5, gt=0.0)
    """

    symbol: str
    backcandles: int = Field(default=100, gt=0)
    timeframe: str = "M15"
    timezone: str = "UTC"


StrategyOrderType = Literal["market", "limit", "stop", "bracket"]


class StrategyConfig[T_Params: BaseStrategyParams](FrozenModel):
    name: str
    order_type: StrategyOrderType
    strategy_class: str
    symbol: str
    params: T_Params
    execution: ExecutionConfig
    filters: FiltersConfig = Field(default_factory=FiltersConfig)
    trading_hours: TradingHoursConfig
    risk: RiskConfig = Field(default_factory=RiskConfig)

    @computed_field
    @property
    def strategy_module(self) -> str:
        return f"nexus_trade.strategies.{self.name}.strategy"

    @classmethod
    def build(
        cls,
        *,
        name: str,
        params: T_Params,
        execution: ExecutionConfig,
        strategy_class: str,
        symbol: str,
        order_type: StrategyOrderType,
        filters: FiltersConfig | None = None,
        trading_hours: TradingHoursConfig | None = None,
        risk: RiskConfig | None = None,
    ) -> StrategyConfig[T_Params]:
        return cls(
            name=name,
            order_type=order_type,
            strategy_class=strategy_class,
            symbol=symbol,
            params=params,
            execution=execution,
            filters=filters or FiltersConfig(),
            trading_hours=trading_hours or TradingHoursConfig(),
            risk=risk or RiskConfig(),
        )
