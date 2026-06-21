import logging
import threading
import time
from dataclasses import dataclass

import MetaTrader5 as mt

from nexus_trade.config.timings import SYSTEM_TIMINGS
from nexus_trade.core.constants import OrderFilling

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
        return cls(
            symbol=symbol,
            description=str(raw.description),
            contract_size=float(raw.trade_contract_size),
            point=float(raw.point),
            digits=int(raw.digits),
            volume_min=float(raw.volume_min),
            volume_max=float(raw.volume_max),
            volume_step=float(raw.volume_step),
            bid=float(raw.bid),
            ask=float(raw.ask),
            spread=int(raw.spread),
            spread_float=bool(raw.spread_float),
            tick_size=float(raw.trade_tick_size),
            tick_value=float(raw.trade_tick_value),
            tick_value_profit=float(raw.trade_tick_value_profit),
            tick_value_loss=float(raw.trade_tick_value_loss),
            currency_base=str(raw.currency_base),
            currency_profit=str(raw.currency_profit),
            currency_margin=str(raw.currency_margin),
            trade_mode=int(raw.trade_mode),
            filling_mode=int(raw.filling_mode),
            stops_level=int(raw.trade_stops_level),
            freeze_level=int(raw.trade_freeze_level),
            swap_long=float(raw.swap_long),
            swap_short=float(raw.swap_short),
            swap_mode=int(raw.swap_mode),
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


@dataclass(frozen=True, slots=True)
class _CachedEntry:
    spec: SymbolSpec
    filling: OrderFilling
    timestamp: float

    def is_valid(self, ttl: float) -> bool:
        return (time.time() - self.timestamp) < ttl


class SymbolSpecCache:
    """Thread-safe symbol spec cache with configurable TTL."""

    def __init__(self) -> None:
        self._cache: dict[str, _CachedEntry] = {}
        self._lock: threading.Lock = threading.Lock()

    def get(self, symbol: str) -> tuple[SymbolSpec, OrderFilling] | None:
        """Return (spec, filling) if cached and fresh."""
        with self._lock:
            entry = self._cache.get(symbol)
            if entry is not None and entry.is_valid(SYSTEM_TIMINGS.symbol_spec_cache_ttl_seconds):
                return entry.spec, entry.filling
        return None

    def get_or_fetch(self, symbol: str, asset_class: str = "unknown") -> tuple[SymbolSpec, OrderFilling] | None:
        """Return cached (spec, filling) or fetch from MT5."""
        cached = self.get(symbol)
        if cached is not None:
            return cached
        spec = SymbolSpec.from_mt5(symbol, asset_class)
        if spec is None:
            return None
        filling_modes = spec.filling_modes()
        if not filling_modes:
            return None
        filling = filling_modes[0]
        with self._lock:
            self._cache[symbol] = _CachedEntry(spec, filling, time.time())
        return spec, filling

    def get_spec(self, symbol: str, asset_class: str = "unknown") -> SymbolSpec | None:
        """Return spec only — thin wrapper over get_or_fetch."""
        result = self.get_or_fetch(symbol, asset_class)
        return result[0] if result is not None else None

    def invalidate(self, symbol: str) -> None:
        """Evict a single symbol — forces re-fetch on next access."""
        with self._lock:
            self._cache.pop(symbol, None)


SYMBOL_SPEC_CACHE: SymbolSpecCache = SymbolSpecCache()
