from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

from nexus_trade.core.types import PartialClosePositionSnapshot, PositionCacheEntry, PositionType

if TYPE_CHECKING:
    from nexus_trade.core.constants import OrderType
    from nexus_trade.core.types import MT5Tick

_CFG = ConfigDict(frozen=True, strict=True, extra="forbid")


@dataclass(slots=True, config=_CFG)
class Position:
    """Typed Position snapshot."""

    ticket: int
    symbol: str
    type: PositionType
    magic: int
    volume: float
    price_open: float
    price_current: float
    sl: float | None
    tp: float | None
    profit: float


@dataclass(frozen=True, slots=True)
class NormalizedPosition:
    """Single-responsibility position data class — one conversion path."""

    ticket: int
    symbol: str
    type: int  # 0=BUY, 1=SELL
    volume: float
    price_open: float
    sl: float
    tp: float
    profit: float
    swap: float
    magic: int
    time: int

    @classmethod
    def from_mt5(cls, pos: object) -> NormalizedPosition:
        """Convert MT5 position namedtuple."""
        return cls(
            ticket=int(getattr(pos, "ticket", 0)),
            symbol=str(getattr(pos, "symbol", "")),
            type=int(getattr(pos, "type", 0)),
            volume=float(getattr(pos, "volume", 0.0)),
            price_open=float(getattr(pos, "price_open", 0.0)),
            sl=float(getattr(pos, "sl", 0.0)),
            tp=float(getattr(pos, "tp", 0.0)),
            profit=float(getattr(pos, "profit", 0.0)),
            swap=float(getattr(pos, "swap", 0.0)),
            magic=int(getattr(pos, "magic", 0)),
            time=int(getattr(pos, "time", 0)),
        )

    def to_cache_entry(self) -> PositionCacheEntry:
        """Convert to shared-state TypedDict."""
        return PositionCacheEntry(
            ticket=self.ticket,
            symbol=self.symbol,
            type=self.type,
            volume=self.volume,
            price_open=self.price_open,
            sl=self.sl,
            tp=self.tp,
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
class Order:
    """Pending Order."""

    ticket: int
    symbol: str
    type: OrderType
    magic: int
    volume_initial: float
    volume_current: float
    price_open: float
    sl: float | None
    tp: float | None


@dataclass(slots=True, frozen=True)
class _PendingBase:
    symbol: str
    magic: int
    submission_time: float


@dataclass(slots=True, frozen=True)
class StandardPendingTicket(_PendingBase):
    ticket: int


@dataclass(slots=True, frozen=True)
class BracketPendingTicket(_PendingBase):
    buy_order_ticket: int
    sell_order_ticket: int
    expected_volume: float
    buy_stop: float
    sell_stop: float


type PendingTicket = StandardPendingTicket | BracketPendingTicket


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


_RETCODE_DONE: int = 10009


def order_succeeded(result: object | None) -> bool:
    """Return True if order_send() result has retcode TRADE_RETCODE_DONE."""
    if result is None:
        return False
    return int(getattr(result, "retcode", -1)) == _RETCODE_DONE


def order_ticket(result: object) -> int:
    """Return the order ticket from an order_send() result."""
    return int(getattr(result, "order", 0))


def cache_entry_to_position(entry: PositionCacheEntry) -> Position:
    """Convert shared-cache TypedDict to Pydantic Position model."""
    return Position(
        ticket=entry["ticket"],
        symbol=entry["symbol"],
        type=PositionType.BUY if entry["type"] == 0 else PositionType.SELL,
        magic=entry["magic"],
        volume=entry["volume"],
        price_open=entry["price_open"],
        price_current=entry["price_open"],
        sl=entry["sl"] if entry["sl"] != 0.0 else None,
        tp=entry["tp"] if entry["tp"] != 0.0 else None,
        profit=entry["profit"],
    )


@dataclass
class ExitLogData:
    """Parameters for exit logging operations."""

    ticket: int
    expected_exit_price: float
    exit_trigger: str
    expected_entry_price: float
    opening_sl: float
    entry_price: float
    executed_volume: float | None = None
    closed_volume: float | None = None
    deal_id: int | None = None
