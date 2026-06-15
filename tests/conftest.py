"""Shared pytest fixtures for nexus-trade.

Patch targets in _MT5_PATCH_TARGETS must match every module that does
``import MetaTrader5 as mt5`` at module level.  Run:
    grep -r "import MetaTrader5" src/
and add any missing dotted paths to the tuple.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Generator
from contextlib import ExitStack
from unittest.mock import MagicMock, patch

import pytest

MT5_PATCH_TARGETS: tuple[str, ...] = (
    "nexus_trade.core.data_handler.mt5",
    "nexus_trade.core.repository.mt5",
    "nexus_trade.core.logger.mt5",
    "nexus_trade.core.risk_manager.mt5",
)


def _build_mt5_mock() -> MagicMock:
    mock = MagicMock(name="MetaTrader5")

    # ── integer constants ────────────────────────────────────────────────────
    mock.POSITION_TYPE_BUY = 0
    mock.POSITION_TYPE_SELL = 1
    mock.DEAL_TYPE_BUY = 0
    mock.DEAL_TYPE_SELL = 1
    mock.DEAL_ENTRY_IN = 0
    mock.DEAL_ENTRY_OUT = 1
    mock.ORDER_TYPE_BUY = 0
    mock.ORDER_TYPE_SELL = 1
    mock.RES_S_OK = 1
    mock.TRADE_RETCODE_DONE = 10009

    # ── account_info ─────────────────────────────────────────────────────────
    account = MagicMock(name="account_info")
    account.balance = 10_000.0
    account.equity = 10_000.0
    account.margin = 0.0
    account.margin_free = 10_000.0
    account.profit = 0.0
    account.currency = "USD"
    mock.account_info.return_value = account

    # ── positions_get / history ──────────────────────────────────────────────
    mock.positions_get.return_value = []
    mock.history_deals_get.return_value = []
    mock.history_orders_get.return_value = []

    # ── symbol_info ──────────────────────────────────────────────────────────
    sym = MagicMock(name="symbol_info")
    sym.trade_tick_size = 0.00001
    sym.trade_tick_value = 1.0
    sym.trade_contract_size = 100_000.0
    sym.volume_min = 0.01
    sym.volume_max = 100.0
    sym.volume_step = 0.01
    sym.point = 0.00001
    sym.digits = 5
    sym.spread = 5
    sym.bid = 1.10000
    sym.ask = 1.10005
    mock.symbol_info.return_value = sym

    # ── misc ─────────────────────────────────────────────────────────────────
    mock.last_error.return_value = (1, "")
    mock.initialize.return_value = True
    mock.shutdown.return_value = None

    return mock


@pytest.fixture
def mt5_mock() -> Generator[MagicMock, None, None]:
    """MagicMock replacing MetaTrader5 in every production module that imports it.

    Skips targets where the attribute does not exist (module does not import
    mt5 directly) so adding unused entries to _MT5_PATCH_TARGETS is harmless.
    """
    mock = _build_mt5_mock()
    with ExitStack() as stack:
        for target in _MT5_PATCH_TARGETS:
            try:
                stack.enter_context(patch(target, mock))
            except AttributeError:
                # module in target path does not import mt5 directly – skip
                pass
        yield mock


# ---------------------------------------------------------------------------
# Position-cache factory
# ---------------------------------------------------------------------------

_PositionCacheEntry = dict[str, object]


@pytest.fixture
def make_position_cache_entry() -> Callable[..., _PositionCacheEntry]:
    """Factory that returns a position-cache entry dict.

    All keyword args have safe defaults so callers only specify what matters:

        entry = make_position_cache_entry(symbol="GBPUSD", magic=2002)
    """

    def _factory(
        *,
        ticket: int = 100_001,
        symbol: str = "EURUSD",
        magic: int = 1001,
        position_type: int = 0,  # 0 = BUY, 1 = SELL
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
