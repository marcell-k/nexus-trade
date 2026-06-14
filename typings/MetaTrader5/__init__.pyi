from datetime import datetime
from typing import NamedTuple, overload

import numpy as np
import numpy.typing as npt

TIMEFRAME_M1: int
TIMEFRAME_M2: int
TIMEFRAME_M3: int
TIMEFRAME_M4: int
TIMEFRAME_M5: int
TIMEFRAME_M6: int
TIMEFRAME_M10: int
TIMEFRAME_M12: int
TIMEFRAME_M15: int
TIMEFRAME_M20: int
TIMEFRAME_M30: int
TIMEFRAME_H1: int
TIMEFRAME_H2: int
TIMEFRAME_H3: int
TIMEFRAME_H4: int
TIMEFRAME_H6: int
TIMEFRAME_H8: int
TIMEFRAME_H12: int
TIMEFRAME_D1: int
TIMEFRAME_W1: int
TIMEFRAME_MN1: int

ORDER_TYPE_BUY: int
ORDER_TYPE_SELL: int
ORDER_TYPE_BUY_LIMIT: int
ORDER_TYPE_SELL_LIMIT: int
ORDER_TYPE_BUY_STOP: int
ORDER_TYPE_SELL_STOP: int
ORDER_TYPE_BUY_STOP_LIMIT: int
ORDER_TYPE_SELL_STOP_LIMIT: int

ORDER_FILLING_FOK: int
ORDER_FILLING_IOC: int
ORDER_FILLING_RETURN: int
ORDER_FILLING_BOC: int

TRADE_ACTION_DEAL: int
TRADE_ACTION_PENDING: int
TRADE_ACTION_SLTP: int
TRADE_ACTION_MODIFY: int
TRADE_ACTION_REMOVE: int
TRADE_ACTION_CLOSE_BY: int

ORDER_TIME_GTC: int
ORDER_TIME_DAY: int
ORDER_TIME_SPECIFIED: int
ORDER_TIME_SPECIFIED_DAY: int

POSITION_TYPE_BUY: int
POSITION_TYPE_SELL: int
DEAL_ENTRY_IN: int
DEAL_ENTRY_OUT: int
DEAL_ENTRY_INOUT: int

class SymbolInfo(NamedTuple):
    name: str
    description: str
    trade_contract_size: float
    point: float
    digits: int
    volume_min: float
    volume_max: float
    volume_step: float
    bid: float
    ask: float
    spread: int
    spread_float: bool
    trade_tick_size: float
    trade_tick_value: float
    trade_tick_value_profit: float
    trade_tick_value_loss: float
    currency_base: str
    currency_profit: str
    currency_margin: str
    trade_mode: int
    filling_mode: int
    trade_stops_level: int
    trade_freeze_level: int
    swap_long: float
    swap_short: float
    swap_mode: int
    time: int

class SymbolInfoTick(NamedTuple):
    time: int
    bid: float
    ask: float
    last: float
    volume: int
    time_msc: int
    flags: int
    volume_real: float

class TradePosition(NamedTuple):
    ticket: int
    time: int
    time_msc: int
    time_update: int
    time_update_msc: int
    type: int
    magic: int
    identifier: int
    reason: int
    volume: float
    price_open: float
    sl: float
    tp: float
    price_current: float
    swap: float
    profit: float
    symbol: str
    comment: str
    external_id: str

class TradeOrder(NamedTuple):
    ticket: int
    time_setup: int
    time_setup_msc: int
    time_done: int
    time_done_msc: int
    time_expiration: int
    type: int
    type_time: int
    type_filling: int
    state: int
    magic: int
    volume_current: float
    volume_initial: float
    price_open: float
    sl: float
    tp: float
    price_current: float
    price_stoplimit: float
    symbol: str
    comment: str
    external_id: str

class TradeDeal(NamedTuple):
    ticket: int
    order: int
    time: int
    time_msc: int
    type: int
    entry: int
    magic: int
    position_id: int
    reason: int
    volume: float
    price: float
    commission: float
    swap: float
    profit: float
    fee: float
    symbol: str
    comment: str
    external_id: str

class OrderSendResult(NamedTuple):
    retcode: int
    deal: int
    order: int
    volume: float
    price: float
    bid: float
    ask: float
    comment: str
    request_id: int
    retcode_external: int

class TerminalInfo(NamedTuple):
    community_account: bool
    community_connection: bool
    connected: bool
    dlls_allowed: bool
    trade_allowed: bool
    tradeapi_disabled: bool
    email_enabled: bool
    ftp_enabled: bool
    notifications_enabled: bool
    mqid: bool
    build: int
    maxbars: int
    codepage: int
    ping_last: int
    community_balance: float
    retransmission: float
    company: str
    name: str
    language: str
    path: str
    data_path: str
    commondata_path: str

class AccountInfo(NamedTuple):
    login: int
    trade_mode: int
    leverage: int
    limit_orders: int
    margin_so_mode: int
    trade_allowed: bool
    trade_expert: bool
    margin_mode: int
    currency_digits: int
    fifo_close: bool
    balance: float
    credit: float
    profit: float
    equity: float
    margin: float
    margin_free: float
    margin_level: float
    margin_so_call: float
    margin_so_so: float
    margin_initial: float
    margin_maintenance: float
    assets: float
    liabilities: float
    commission_blocked: float
    name: str
    server: str
    currency: str
    company: str

def initialize(
    path: str | None = ...,
    *,
    login: int | None = ...,
    password: str | None = ...,
    server: str | None = ...,
    timeout: int | None = ...,
    portable: bool = ...,
) -> bool: ...
def shutdown() -> None: ...
def last_error() -> tuple[int, str]: ...
def terminal_info() -> TerminalInfo | None: ...
def account_info() -> AccountInfo | None: ...
def symbol_info(symbol: str) -> SymbolInfo | None: ...
def symbol_info_tick(symbol: str) -> SymbolInfoTick | None: ...
def copy_rates_from_pos(
    symbol: str,
    timeframe: int,
    start_pos: int,
    count: int,
) -> npt.NDArray[np.void] | None: ...
def order_send(request: dict[str, str | int | float | bool]) -> OrderSendResult | None: ...
@overload
def positions_get() -> tuple[TradePosition, ...] | None: ...
@overload
def positions_get(*, symbol: str) -> tuple[TradePosition, ...] | None: ...
@overload
def positions_get(*, ticket: int) -> tuple[TradePosition, ...] | None: ...
def positions_get(
    symbol: str | None = None,
    ticket: int | None = None,
) -> tuple[TradePosition, ...] | None: ...
@overload
def orders_get() -> tuple[TradeOrder, ...] | None: ...
@overload
def orders_get(*, symbol: str) -> tuple[TradeOrder, ...] | None: ...
@overload
def orders_get(*, ticket: int) -> tuple[TradeOrder, ...] | None: ...
def orders_get(
    symbol: str | None = None,
    ticket: int | None = None,
) -> tuple[TradeOrder, ...] | None: ...
@overload
def history_deals_get(from_date: datetime, to_date: datetime) -> tuple[TradeDeal, ...] | None: ...
@overload
def history_deals_get(from_date: datetime, to_date: datetime, *, group: str) -> tuple[TradeDeal, ...] | None: ...
@overload
def history_deals_get(*, position: int) -> tuple[TradeDeal, ...] | None: ...
@overload
def history_deals_get(*, ticket: int) -> tuple[TradeDeal, ...] | None: ...
def history_deals_get(
    from_date: datetime | None = None,
    to_date: datetime | None = None,
    *,
    group: str | None = None,
    position: int | None = None,
    ticket: int | None = None,
) -> tuple[TradeDeal, ...] | None: ...
