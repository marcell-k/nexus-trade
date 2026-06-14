import logging
import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import MetaTrader5 as mt

from nexus_trade.core.constants import OrderFilling

if TYPE_CHECKING:
    from nexus_trade.core.protocols import SymbolInfo


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class SymbolSpec:
    """Data on specific instruement."""

    symbol: str
    description: str
    contract_size: float
    point: float
    digits: int
    volume_min: float
    volume_max: float
    volume_step: float
    bid: float
    ask: float
    spread: int
    spread_float: bool
    tick_size: float
    tick_value: float
    tick_value_profit: float
    tick_value_loss: float
    currency_base: str
    currency_profit: str
    currency_margin: str
    trade_mode: int
    filling_mode: int
    stops_level: int
    freeze_level: int
    swap_long: float
    swap_short: float
    swap_mode: int
    asset_class: str = "unknown"

    @classmethod
    def from_mt5(cls, symbol: str, asset_class: str = "unknown") -> "SymbolSpec | None":
        raw = mt.symbol_info(symbol)
        if raw is None:
            logger.error(f"SymbolInfoFail sym={symbol} | reason=mt5_returned_none")
            return None
        info: SymbolInfo = cast("SymbolInfo", raw)
        return cls(
            symbol=symbol,
            description=str(info.description),
            contract_size=float(info.trade_contract_size),
            point=float(info.point),
            digits=int(info.digits),
            volume_min=float(info.volume_min),
            volume_max=float(info.volume_max),
            volume_step=float(info.volume_step),
            bid=float(info.bid),
            ask=float(info.ask),
            spread=int(info.spread),
            spread_float=bool(info.spread_float),
            tick_size=float(info.trade_tick_size),
            tick_value=float(info.trade_tick_value),
            tick_value_profit=float(info.trade_tick_value_profit),
            tick_value_loss=float(info.trade_tick_value_loss),
            currency_base=str(info.currency_base),
            currency_profit=str(info.currency_profit),
            currency_margin=str(info.currency_margin),
            trade_mode=int(info.trade_mode),
            filling_mode=int(info.filling_mode),
            stops_level=int(info.trade_stops_level),
            freeze_level=int(info.trade_freeze_level),
            swap_long=float(info.swap_long),
            swap_short=float(info.swap_short),
            swap_mode=int(info.swap_mode),
            asset_class=asset_class,
        )

    def filling_modes(self) -> list[OrderFilling]:
        bit_map = [
            (1, OrderFilling.FOK),
            (2, OrderFilling.IOC),
            (4, OrderFilling.RETURN),
            (8, OrderFilling.BOC),
        ]
        return [mode for bit, mode in bit_map if self.filling_mode & bit]


_symbol_cache: dict[str, SymbolSpec | None] = {}
_symbol_lock = threading.Lock()


def get_symbol_spec(symbol: str, asset_class: str = "unknown") -> SymbolSpec | None:
    if symbol not in _symbol_cache:
        with _symbol_lock:
            if symbol not in _symbol_cache:
                _symbol_cache[symbol] = SymbolSpec.from_mt5(symbol, asset_class)
    return _symbol_cache[symbol]


@dataclass(frozen=True, slots=True)
class _CachedEntry:
    spec: SymbolSpec
    filling: OrderFilling
    timestamp: float

    def is_valid(self, ttl: float) -> bool:
        return (time.time() - self.timestamp) < ttl


class SymbolSpecCache:
    """Thread-safe symbol spec cache with configurable TTL."""

    def __init__(self, ttl_seconds: float = 300.0) -> None:
        self._cache: dict[str, _CachedEntry] = {}
        self._lock: threading.Lock = threading.Lock()
        self.ttl: float = ttl_seconds

    def get(self, symbol: str) -> tuple[SymbolSpec, OrderFilling] | None:
        """Return (spec, filling) if cached and fresh."""
        with self._lock:
            entry = self._cache.get(symbol)
            if entry is not None and entry.is_valid(self.ttl):
                return entry.spec, entry.filling
        return None

    def get_or_fetch(self, symbol: str) -> tuple[SymbolSpec, OrderFilling] | None:
        """Return cached entry or fetch from MT5."""
        cached = self.get(symbol)
        if cached is not None:
            return cached
        spec = get_symbol_spec(symbol)
        if spec is None:
            return None
        filling = spec.filling_modes()[0]
        with self._lock:
            self._cache[symbol] = _CachedEntry(spec, filling, time.time())
        return spec, filling
