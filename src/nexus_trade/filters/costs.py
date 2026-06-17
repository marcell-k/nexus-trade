from dataclasses import dataclass

import MetaTrader5 as mt

from nexus_trade.core.models import Tick
from nexus_trade.core.symbol import SYMBOL_SPEC_CACHE, SymbolSpec


@dataclass
class MarketCondition:
    """Market condition validation result."""

    spread_points: float
    slippage_points: float
    spread_price: float
    slippage_price: float
    is_valid: bool
    reason: str


class MarketCostCalculator:
    """Validate spread and slippage with automatic normalization per symbol."""

    def __init__(self, max_spread_points: float = 30.0, max_slippage_points: float = 20.0) -> None:
        self.max_spread_points: float = max_spread_points
        self.max_slippage_points: float = max_slippage_points

    def validate_market_conditions(
        self,
        symbol: str,
        expected_price: float,
        is_buy: bool,
        cached_symbol_info: SymbolSpec | None = None,
    ) -> MarketCondition:
        raw_tick = mt.symbol_info_tick(symbol)
        if raw_tick is None:
            raise RuntimeError(f"Tick data unavailable for {symbol!r}")
        tick: Tick = Tick.from_mt5(raw_tick)

        effective_info: SymbolSpec | None = (
            cached_symbol_info if cached_symbol_info is not None else SYMBOL_SPEC_CACHE.get_spec(symbol)
        )
        if effective_info is None:
            raise RuntimeError(f"Symbol info unavailable for {symbol!r}")
        point: float = effective_info.point

        spread_price: float = tick.ask - tick.bid
        spread_points: float = spread_price / point
        slippage_price: float = (tick.ask if is_buy else tick.bid) - expected_price
        slippage_points: float = abs(slippage_price / point)

        if spread_points > self.max_spread_points:
            is_valid = False
            reason = f"Spread too wide: {spread_points:.1f} points (max {self.max_spread_points:.1f})"
        elif slippage_points > self.max_slippage_points:
            is_valid = False
            reason = f"Slippage too high: {slippage_points:.1f} points (max {self.max_slippage_points:.1f})"
        else:
            is_valid = True
            reason = "Market conditions acceptable"

        return MarketCondition(
            spread_points=spread_points,
            slippage_points=slippage_points,
            spread_price=spread_price,
            slippage_price=slippage_price,
            is_valid=is_valid,
            reason=reason,
        )
