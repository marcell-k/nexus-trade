from __future__ import annotations

import sys
import time
from collections.abc import Callable, Generator
from contextlib import ExitStack
from typing import NamedTuple
from unittest.mock import MagicMock, patch

import pytest


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


# ---------------------------------------------------------------------------
# MT5 mock builder
# ---------------------------------------------------------------------------


def _build_mt5_mock() -> MagicMock:
    mock = MagicMock(name="MetaTrader5")

    # Position / deal / order constants
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

    # Filling / action / time-in-force constants
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

    # Default API return values
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


# ---------------------------------------------------------------------------
# mt5_mock
# Patches:
#   sys.modules["MetaTrader5"]       — so test bodies can do sys.modules["MetaTrader5"]
#   each module-level alias          — ``import MetaTrader5 as mt5`` or ``as mt``
# Targets with no matching attr are skipped silently (safe to add extras).
# ---------------------------------------------------------------------------


@pytest.fixture
def mt5_mock() -> Generator[MagicMock]:
    # Run ``grep -r "import MetaTrader5" src/`` to verify / extend this list.
    # Suffix must match the alias used in that module (mt5 or mt).
    _patch_targets: tuple[str, ...] = (
        "nexus_trade.core.data_handler.mt5",  # import MetaTrader5 as mt5
        "nexus_trade.core.connection.mt5",  # import MetaTrader5 as mt5
        "nexus_trade.core.repository.mt",  # import MetaTrader5 as mt
        "nexus_trade.core.symbol.mt",  # import MetaTrader5 as mt
        "nexus_trade.core.constants.mt",  # import MetaTrader5 as mt
        "nexus_trade.filters.costs.mt",  # import MetaTrader5 as mt
        "nexus_trade.logging.logger.mt",  # import MetaTrader5 as mt
        "nexus_trade.risk.manager.mt",  # import MetaTrader5 as mt
        "nexus_trade.execution.executor.mt",  # import MetaTrader5 as mt
    )
    mock = _build_mt5_mock()
    with ExitStack() as stack:
        # Lets test bodies access the mock via sys.modules["MetaTrader5"]
        stack.enter_context(patch.dict(sys.modules, {"MetaTrader5": mock}))
        for target in _patch_targets:
            try:
                stack.enter_context(patch(target, mock))
            except AttributeError:
                pass  # module does not import mt5/mt directly — safe to skip
        yield mock


# ---------------------------------------------------------------------------
# Named fixtures consumed by test_risk_manager.py
# ---------------------------------------------------------------------------


@pytest.fixture
def account_info() -> _AccountInfo:
    """Default AccountInfo namedtuple. Supports ._replace() for param variation."""
    return _AccountInfo()


@pytest.fixture
def eurusd_info() -> _SymbolInfo:
    """EURUSD SymbolInfo namedtuple used in position-sizing tests."""
    return _SymbolInfo()


# ---------------------------------------------------------------------------
# Symbol spec cache isolation
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clear_symbol_cache() -> Generator[None]:
    """Wipe the process-local SymbolSpec cache before and after every test."""
    try:
        from nexus_trade.core.symbol import _symbol_cache

        _symbol_cache.clear()
    except ImportError:
        pass
    yield
    try:
        from nexus_trade.core.symbol import _symbol_cache

        _symbol_cache.clear()
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Position cache entry factory
# ---------------------------------------------------------------------------

_PositionCacheEntry = dict[str, object]


@pytest.fixture
def make_position_cache_entry() -> Callable[..., _PositionCacheEntry]:
    """Factory for position-cache entry dicts.

    Default magic=12345 matches the magic used by test_repository queries.
    Override any field via keyword arg::

        make_position_cache_entry(symbol="GBPUSD", magic=2002)
    """

    def _factory(
        *,
        ticket: int = 100_001,
        symbol: str = "EURUSD",
        magic: int = 12345,
        position_type: int = 0,
        volume: float = 0.10,
        price_open: float = 1.10000,
        sl: float = 1.09900,
        tp: float = 1.10200,
        profit: float = 0.0,
        time_open: float | None = None,
        is_open: bool = True,
    ) -> _PositionCacheEntry:
        return {
            "ticket": ticket,
            "symbol": symbol,
            "magic": magic,
            "position_type": position_type,
            "volume": volume,
            "price_open": price_open,
            "sl": sl,
            "tp": tp,
            "profit": profit,
            "time_open": time_open if time_open is not None else time.time(),
            "is_open": is_open,
        }

    return _factory
