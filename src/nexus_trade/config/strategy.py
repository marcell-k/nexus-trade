from __future__ import annotations

from typing import ClassVar, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field


class _Frozen(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, strict=True, extra="forbid")


class ExecutionConfig(_Frozen):
    magic_number: int = Field(gt=0)
    deviation: int = Field(ge=0, default=100)
    comment_prefix: str = Field(default="", max_length=15)
    min_market_threshold_points: int = Field(default=0, ge=0)


class NewsFilterConfig(_Frozen):
    enabled: bool = False
    currencies: list[str] = Field(default_factory=list)
    buffer_minutes: int = Field(default=15, ge=0)


class FiltersConfig(_Frozen):
    market_regime_enabled: bool = False
    news: NewsFilterConfig = Field(default_factory=NewsFilterConfig)


class SessionConfig(_Frozen):
    start: str = Field(pattern=r"^\d{2}:\d{2}$")
    end: str = Field(pattern=r"^\d{2}:\d{2}$")


class TradingHoursConfig(_Frozen):
    enabled: bool = False
    timezone: str = "UTC"
    sessions: list[SessionConfig] = Field(default_factory=list)


class RiskConfig(_Frozen):
    position_sizing_method: Literal["fractional", "fixed"] = "fixed"
    max_positions: int = Field(default=1, gt=0)
    max_trades: int = Field(default=1, gt=0)
    max_spread_points: int = Field(default=100, ge=0)
    max_slippage_points: int = Field(default=100, ge=0)
    max_order_fill_time_seconds: int = Field(default=2, gt=0)


class BaseStrategyParams(_Frozen):
    """
    Typed base for all strategy parameter models.

    Extend by subclassing and declaring additional fields — never instantiate directly
    with undeclared kwargs. Inherits frozen=True, strict=True, extra="forbid" from _Frozen.

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


# Legacy TypeVar kept for annotation sites that haven't migrated to PEP 695 syntax.
T_Params = TypeVar("T_Params", bound=BaseStrategyParams)

StrategyOrderType = Literal["market", "limit", "stop", "bracket"]


class StrategyConfig[T_Params: BaseStrategyParams](_Frozen):
    name: str
    order_type: StrategyOrderType
    strategy_module: str
    strategy_class: str
    symbol: str
    params: T_Params
    execution: ExecutionConfig
    filters: FiltersConfig = Field(default_factory=FiltersConfig)
    trading_hours: TradingHoursConfig | None
    risk: RiskConfig = Field(default_factory=RiskConfig)

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
        """Construct a StrategyConfig from parts; derive strategy_module from name."""
        return cls(
            name=name,
            order_type=order_type,
            strategy_module=f"nexus_trade.strategies.{name}.strategy",
            strategy_class=strategy_class,
            symbol=symbol,
            params=params,
            execution=execution,
            filters=filters or FiltersConfig(),
            trading_hours=trading_hours or TradingHoursConfig(),
            risk=risk or RiskConfig(),
        )
