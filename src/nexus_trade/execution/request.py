import math
from dataclasses import dataclass
from typing import get_args

from nexus_trade.config.strategy import StrategyOrderType
from nexus_trade.core.types import PartialClosePositionSnapshot, PositionCacheEntry

VALID_ORDER_TYPES: frozenset[str] = frozenset(get_args(StrategyOrderType))


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


@dataclass
class EntryRequest:
    """
    Declarative entry order specification.

    Order type usage:
    - market: Uses signal, sl, tp
    - limit/stop: Uses signal, entry_price, sl, tp
    - bracket: Uses buy_stop, sell_stop, buy_sl, sell_sl (ignores signal, sl, tp)
    """

    strategy_name: str
    order_type: str
    symbol: str
    volume: float
    signal: int  # 1=long, -1=short, 2=bracket (both directions)

    # Single-direction orders (market/limit/stop)
    entry_price: float | None = None
    sl: float | None = None
    tp: float | None = None

    # Bracket orders (OCO) - reuses entry_price fields differently
    buy_stop: float | None = None  # Buy side trigger
    sell_stop: float | None = None  # Sell side trigger
    buy_sl: float | None = None  # Buy side stop loss
    sell_sl: float | None = None  # Sell side stop loss
    buy_tp: float | None = None
    sell_tp: float | None = None

    # Metadata
    comment: str = ""
    expiration_time: str | None = None

    def __post_init__(self) -> None:
        """Validate EntryRequest."""
        if self.order_type not in VALID_ORDER_TYPES:
            raise ValueError(f"Invalid order_type '{self.order_type}', must be one of {VALID_ORDER_TYPES}")
        if self.signal not in {1, -1, 2}:
            raise ValueError(f"Invalid signal {self.signal}, must be 1 (long), -1 (short), or 2 (bracket)")
        if self.volume < 0 or (not math.isfinite(self.volume)):
            raise ValueError(f"Invalid volume {self.volume}, must be >= 0 and finite")
        if self.order_type == "bracket":
            missing = [
                name
                for name, value in (
                    ("buy_stop", self.buy_stop),
                    ("sell_stop", self.sell_stop),
                    ("buy_sl", self.buy_sl),
                    ("sell_sl", self.sell_sl),
                )
                if value is None
            ]
            if missing:
                raise ValueError(f"Bracket order missing required fields: {missing}")


@dataclass
class ExitRequest:
    """
    Declarative exit order specification. Supports full and partial closes.

    Note:
        expected_exit_price is optional. StrategyRunner derives expected exit from live tick
        price and uses this field as fallback when tick data is unavailable.

    """

    ticket: int
    portion: float = 1.0  # 1.0 = full close, 0.5 = half close
    comment: str = ""
    expected_exit_price: float | None = None

    # TODO: implement new_sl and new_tp
    new_sl: float | None = None
    new_tp: float | None = None

    # Metadata
    strategy_name: str | None = None
    exit_reason: str = ""

    def __post_init__(self) -> None:
        """Validate ExitRequest."""
        if self.ticket <= 0:
            raise ValueError(f"Invalid ticket {self.ticket}, must be > 0")
        if not (0 < self.portion <= 1.0):
            raise ValueError(f"Invalid portion {self.portion}, must be in (0, 1.0]")


@dataclass
class ModifyRequest:
    """Position modification request (SL/TP adjustment). Used for trailing stops and breakeven."""

    ticket: int
    new_sl: float | None = None
    new_tp: float | None = None
    comment: str = ""


@dataclass
class ExecutionResult:
    """Result of order execution attempt with fill details for slippage analysis."""

    success: bool
    ticket: int | None = None
    order_tickets: list[int] | None = None
    error_code: int | None = None
    error_message: str = ""

    # Execution details
    executed_volume: float | None = None
    deal_id: int | None = None
    # Request metadata (for logging)
    request_type: str = ""
    symbol: str = ""


type ModifyRequestResult = ModifyRequest | ExitRequest | None


@dataclass
class FillData:
    """Trade fill parameters."""

    trade_id: int
    position: PositionCacheEntry
    expected_entry_price: float
    strategy_name: str
    opening_sl: float | None = None
    fill_time_ms: float | None = None
    volume_multiplier: float | None = None


@dataclass(slots=True)
class CloseData:
    """Trade close parameters."""

    trade_id: int
    position: PositionCacheEntry
    expected_exit_price: float | None
    opening_sl: float | None
    exit_trigger: str
    entry_price: float
    expected_entry_price: float | None


@dataclass(slots=True)
class PartialCloseData:
    """Partial close parameters."""

    trade_id: int
    position: PartialClosePositionSnapshot
    closed_volume: float
    remaining_volume: float
    expected_exit_price: float | None
    opening_sl: float
    strategy_name: str
    exit_trigger: str
    entry_price: float
    expected_entry_price: float
    deal_id: int | None = None


@dataclass(slots=True)
class ExitLogData:
    """Parameters for exit logging operations."""

    ticket: int
    expected_exit_price: float
    exit_trigger: str
    expected_entry_price: float
    opening_sl: float
    entry_price: float
    closed_volume: float | None = None
    deal_id: int | None = None
