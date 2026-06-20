from __future__ import annotations

import sys
from contextlib import ExitStack, suppress
from typing import TYPE_CHECKING, NamedTuple
from unittest.mock import MagicMock, patch

import pytest

if TYPE_CHECKING:
    from collections.abc import Callable, Generator

    from nexus_trade.core.types import PositionCacheEntry


class _AccountInfo(NamedTuple):
    login: int = 12_345_678
    trade_mode: int = 0
    leverage: int = 100
    limit_orders: int = 200
    margin_so_mode: int = 0
    trade_allowed: bool = True
    trade_expert: bool = True
    margin_mode: int = 2
    currency_digits: int = 2
    fifo_close: bool = False
    balance: float = 10_000.0
    credit: float = 0.0
    profit: float = 0.0
    equity: float = 10_000.0
    margin: float = 0.0
    margin_free: float = 10_000.0
    margin_level: float = 0.0
    margin_so_call: float = 50.0
    margin_so_so: float = 30.0
    margin_initial: float = 0.0
    margin_maintenance: float = 0.0
    assets: float = 0.0
    liabilities: float = 0.0
    commission_blocked: float = 0.0
    name: str = "Test Account"
    server: str = "TestBroker-Demo"
    currency: str = "USD"
    company: str = "Test Broker"


class _SymbolInfo(NamedTuple):
    name: str = "EURUSD"
    description: str = "Euro vs US Dollar"
    trade_contract_size: float = 100_000.0
    point: float = 0.00001
    digits: int = 5
    volume_min: float = 0.01
    volume_max: float = 100.0
    volume_step: float = 0.01
    bid: float = 1.10000
    ask: float = 1.10005
    spread: int = 5
    spread_float: bool = True
    trade_tick_size: float = 0.00001
    trade_tick_value: float = 1.0
    trade_tick_value_profit: float = 1.0
    trade_tick_value_loss: float = 1.0
    currency_base: str = "EUR"
    currency_profit: str = "USD"
    currency_margin: str = "EUR"
    trade_mode: int = 4
    filling_mode: int = 1
    trade_stops_level: int = 20
    trade_freeze_level: int = 0
    swap_long: float = -0.5
    swap_short: float = 0.3
    swap_mode: int = 0
    time: int = 1_700_000_000


def _build_mt5_mock() -> MagicMock:
    mock = MagicMock(name="MetaTrader5")

    mock.POSITION_TYPE_BUY = 0
    mock.POSITION_TYPE_SELL = 1
    mock.DEAL_TYPE_BUY = 0
    mock.DEAL_TYPE_SELL = 1
    mock.DEAL_ENTRY_IN = 0
    mock.DEAL_ENTRY_OUT = 1
    mock.DEAL_ENTRY_INOUT = 2
    mock.ORDER_TYPE_BUY = 0
    mock.ORDER_TYPE_SELL = 1
    mock.ORDER_TYPE_BUY_LIMIT = 2
    mock.ORDER_TYPE_SELL_LIMIT = 3
    mock.ORDER_TYPE_BUY_STOP = 4
    mock.ORDER_TYPE_SELL_STOP = 5
    mock.ORDER_TYPE_BUY_STOP_LIMIT = 6
    mock.ORDER_TYPE_SELL_STOP_LIMIT = 7
    mock.ORDER_FILLING_FOK = 0
    mock.ORDER_FILLING_IOC = 1
    mock.ORDER_FILLING_RETURN = 2
    mock.ORDER_FILLING_BOC = 3
    mock.TRADE_ACTION_DEAL = 1
    mock.TRADE_ACTION_PENDING = 5
    mock.TRADE_ACTION_SLTP = 6
    mock.TRADE_ACTION_MODIFY = 7
    mock.TRADE_ACTION_REMOVE = 8
    mock.TRADE_ACTION_CLOSE_BY = 10
    mock.ORDER_TIME_GTC = 0
    mock.ORDER_TIME_DAY = 1
    mock.ORDER_TIME_SPECIFIED = 2
    mock.ORDER_TIME_SPECIFIED_DAY = 3
    mock.TRADE_RETCODE_NO_CHANGES = 10025

    mock.account_info.return_value = _AccountInfo()
    mock.positions_get.return_value = ()
    mock.orders_get.return_value = ()
    mock.history_deals_get.return_value = ()
    mock.history_orders_get.return_value = ()
    mock.symbol_info.return_value = _SymbolInfo()
    mock.symbol_info_tick.return_value = MagicMock(
        bid=1.10000,
        ask=1.10005,
        last=0.0,
        volume=0,
        time=1_700_000_000,
        time_msc=1_700_000_000_000,
        flags=0,
        volume_real=0.0,
    )
    mock.copy_rates_from_pos.return_value = None
    mock.terminal_info.return_value = MagicMock(connected=True)
    mock.last_error.return_value = (1, "")
    mock.initialize.return_value = True
    mock.shutdown.return_value = None

    return mock


def _install_mt5_stub() -> None:
    """Install a placeholder MetaTrader5 module so production code can import it on non-Windows CI."""
    if "MetaTrader5" in sys.modules:
        return
    sys.modules["MetaTrader5"] = _build_mt5_mock()


_install_mt5_stub()


_PATCH_TARGETS: tuple[str, ...] = (
    "nexus_trade.core.data_handler.mt5",
    "nexus_trade.core.connection.mt5",
    "nexus_trade.core.repository.mt",
    "nexus_trade.core.symbol.mt",
    "nexus_trade.core.constants.mt",
    "nexus_trade.filters.costs.mt",
    "nexus_trade.logging.logger.mt",
    "nexus_trade.risk.manager.mt",
    "nexus_trade.execution.executor.mt",
)


@pytest.fixture
def mt5_mock() -> Generator[MagicMock]:
    mock = _build_mt5_mock()
    with ExitStack() as stack:
        stack.enter_context(patch.dict(sys.modules, {"MetaTrader5": mock}))
        for target in _PATCH_TARGETS:
            with suppress(ImportError, AttributeError):
                stack.enter_context(patch(target, mock))
        yield mock


@pytest.fixture
def account_info() -> _AccountInfo:
    return _AccountInfo()


@pytest.fixture
def eurusd_info() -> _SymbolInfo:
    return _SymbolInfo()


@pytest.fixture
def make_position_cache_entry() -> Callable[..., PositionCacheEntry]:
    """Create PositionCacheEntry TypedDicts matching the actual type definition."""

    def _factory(
        *,
        ticket: int = 100_001,
        symbol: str = "EURUSD",
        magic: int = 12345,
        type: int = 0,
        volume: float = 0.10,
        price_open: float = 1.10000,
        sl: float = 1.09900,
        tp: float = 1.10200,
        profit: float = 0.0,
        swap: float = 0.0,
        time: int = 0,
    ) -> PositionCacheEntry:
        from nexus_trade.core.types import PositionCacheEntry as PCE

        return PCE(
            ticket=ticket,
            symbol=symbol,
            type=type,
            volume=volume,
            price_open=price_open,
            sl=sl,
            tp=tp,
            profit=profit,
            swap=swap,
            magic_number=magic,
            time=time,
        )

    return _factory
