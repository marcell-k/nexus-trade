from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

from nexus_trade.core.constants import MT5_RETCODE_DONE
from nexus_trade.core.types import (
    OrderSnapshot,
    PartialClosePositionSnapshot,
    PositionCacheEntry,
    PositionType,
)

if TYPE_CHECKING:
    from nexus_trade.core.types import MT5Tick

_CFG = ConfigDict(frozen=True, strict=True, extra="forbid")


@dataclass(slots=True, config=_CFG)
class Position:
    """Position snapshot."""

    ticket: int
    symbol: str
    type: PositionType
    magic: int
    volume: float
    price_open: float
    sl: float | None
    tp: float | None
    profit: float
    swap: float
    time: int

    @classmethod
    def from_mt5(cls, pos: object) -> Position:
        """Convert MT5 position namedtuple."""
        raw_sl = float(getattr(pos, "sl", 0.0))
        raw_tp = float(getattr(pos, "tp", 0.0))
        return cls(
            ticket=int(getattr(pos, "ticket", 0)),
            symbol=str(getattr(pos, "symbol", "")),
            type=PositionType.from_int(int(getattr(pos, "type", 0))),
            magic=int(getattr(pos, "magic", 0)),
            volume=float(getattr(pos, "volume", 0.0)),
            price_open=float(getattr(pos, "price_open", 0.0)),
            sl=raw_sl if raw_sl != 0.0 else None,
            tp=raw_tp if raw_tp != 0.0 else None,
            profit=float(getattr(pos, "profit", 0.0)),
            swap=float(getattr(pos, "swap", 0.0)),
            time=int(getattr(pos, "time", 0)),
        )

    def to_cache_entry(self) -> PositionCacheEntry:
        return PositionCacheEntry(
            ticket=self.ticket,
            symbol=self.symbol,
            type=self.type,
            volume=self.volume,
            price_open=self.price_open,
            sl=self.sl if self.sl is not None else 0.0,
            tp=self.tp if self.tp is not None else 0.0,
            profit=self.profit,
            swap=self.swap,
            magic=self.magic,
            time=self.time,
        )

    def to_partial_snapshot(self) -> PartialClosePositionSnapshot:
        return PartialClosePositionSnapshot(
            ticket=self.ticket,
            type=self.type,
            symbol=self.symbol,
            swap=self.swap,
        )


@dataclass(slots=True, config=_CFG)
class Tick:
    time: int  # seconds in broker TZ
    bid: float
    ask: float
    last: float
    volume: int
    time_msc: int  # milliseconds UTC

    @classmethod
    def from_mt5(cls, raw: MT5Tick) -> Tick:
        """Construct from the namedtuple returned by symbol_info_tick()."""
        return cls(
            time=raw.time,
            bid=raw.bid,
            ask=raw.ask,
            last=raw.last,
            volume=raw.volume,
            time_msc=raw.time_msc,
        )

    @property
    def spread(self) -> float:
        return self.ask - self.bid

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2


def order_succeeded(result: object | None) -> bool:
    """Return True if order_send() result has retcode TRADE_RETCODE_DONE."""
    if result is None:
        return False
    return int(getattr(result, "retcode", -1)) == MT5_RETCODE_DONE


def cache_entry_to_position(entry: PositionCacheEntry) -> Position:
    """Convert shared-cache TypedDict to Pydantic Position model."""
    return Position(
        ticket=entry["ticket"],
        symbol=entry["symbol"],
        type=PositionType.from_int(entry["type"]),
        magic=entry["magic"],
        volume=entry["volume"],
        price_open=entry["price_open"],
        sl=entry["sl"] if entry["sl"] != 0.0 else None,
        tp=entry["tp"] if entry["tp"] != 0.0 else None,
        profit=entry["profit"],
        swap=entry["swap"],
        time=entry["time"],
    )


def normalize_order(order: object) -> OrderSnapshot:
    """Convert MT5 order namedtuple to standardized ``OrderSnapshot``."""
    return OrderSnapshot(
        ticket=int(getattr(order, "ticket", 0)),
        symbol=str(getattr(order, "symbol", "")),
        type=int(getattr(order, "type", 0)),
        magic=int(getattr(order, "magic", 0)),
    )
