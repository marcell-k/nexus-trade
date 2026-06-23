"""
Minimal tests for OrderExecutor.

Coverage: market/pending/bracket entries, close, modify, retry logic, deal-ID
recovery, cancel.  All tests use the project-wide mt5_mock fixture and clear
the module-level SymbolSpecCache between runs to prevent cross-test leakage.
"""

from __future__ import annotations

from collections import namedtuple
from typing import TYPE_CHECKING
from zoneinfo import ZoneInfo

import pytest

from nexus_trade.core.constants import MT5_RETCODE_DONE, RETRYABLE_CODES, OrderType
from nexus_trade.execution.executor import OrderExecutor
from nexus_trade.execution.request import EntryRequest, ExitRequest, ModifyRequest

if TYPE_CHECKING:
    from collections.abc import Generator
    from unittest.mock import MagicMock

BROKER_TZ = ZoneInfo("UTC")

# ── lightweight namedtuple stubs (mirror real MT5 shapes) ────────────────────

_Res = namedtuple(
    "_Res",
    "retcode deal order volume price bid ask comment request_id retcode_external",
)
_Pos = namedtuple(
    "_Pos",
    "ticket time time_msc time_update time_update_msc type magic identifier "
    "reason volume price_open sl tp price_current swap profit symbol comment external_id",
)
_Deal = namedtuple(
    "_Deal",
    "ticket order time time_msc type entry magic position_id reason "
    "volume price commission swap profit fee symbol comment external_id",
)


# ── result builders ───────────────────────────────────────────────────────────


def _ok(order: int = 100_001, deal: int = 200_001) -> _Res:
    return _Res(
        retcode=MT5_RETCODE_DONE,
        deal=deal,
        order=order,
        volume=0.1,
        price=1.1,
        bid=1.1,
        ask=1.1002,
        comment="done",
        request_id=1,
        retcode_external=0,
    )


def _fail(retcode: int = 10016) -> _Res:
    return _Res(
        retcode=retcode,
        deal=0,
        order=0,
        volume=0.0,
        price=0.0,
        bid=0.0,
        ask=0.0,
        comment="reject",
        request_id=0,
        retcode_external=0,
    )


def _position(
    ticket: int = 100_001,
    pos_type: int = 0,
    volume: float = 0.1,
    sl: float = 1.09000,
    tp: float = 1.11000,
) -> _Pos:
    return _Pos(
        ticket=ticket,
        time=0,
        time_msc=0,
        time_update=0,
        time_update_msc=0,
        type=pos_type,
        magic=1001,
        identifier=0,
        reason=0,
        volume=volume,
        price_open=1.10000,
        sl=sl,
        tp=tp,
        price_current=1.10000,
        swap=0.0,
        profit=0.0,
        symbol="EURUSD",
        comment="",
        external_id="",
    )


def _deal_nt(
    ticket: int = 200_001,
    deal_type: int = 1,
    entry: int = 1,
    position_id: int = 100_001,
    volume: float = 0.1,
    time_msc: int = 1_700_000_000_000,
) -> _Deal:
    return _Deal(
        ticket=ticket,
        order=0,
        time=0,
        time_msc=time_msc,
        type=deal_type,
        entry=entry,
        magic=1001,
        position_id=position_id,
        reason=0,
        volume=volume,
        price=1.11000,
        commission=-3.0,
        swap=0.0,
        profit=100.0,
        fee=0.0,
        symbol="EURUSD",
        comment="",
        external_id="",
    )


def _entry_req(order_type: str = "market", signal: int = 1, **kw: object) -> EntryRequest:
    base: dict[str, object] = {
        "strategy_name": "sma_crossover",
        "order_type": order_type,
        "symbol": "EURUSD",
        "volume": 0.1,
        "signal": signal,
        "sl": 1.09,
        "tp": 1.11,
    }
    base.update(kw)
    return EntryRequest(**base)  # type: ignore[arg-type]


# ── fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def executor(mt5_mock: MagicMock) -> OrderExecutor:
    return OrderExecutor(BROKER_TZ)


@pytest.fixture(autouse=True)
def _clear_sym_cache() -> Generator:
    """Evict EURUSD from the module-level SymbolSpecCache before/after each test."""
    from nexus_trade.core.symbol import SYMBOL_SPEC_CACHE

    SYMBOL_SPEC_CACHE.invalidate("EURUSD")
    yield  # type: ignore[misc]
    SYMBOL_SPEC_CACHE.invalidate("EURUSD")


# ── market entry ──────────────────────────────────────────────────────────────


class TestMarketEntry:
    def test_buy_returns_correct_ticket(self, executor: OrderExecutor, mt5_mock: MagicMock) -> None:
        mt5_mock.order_send.return_value = _ok(order=100_001)
        r = executor.execute_entry(_entry_req(signal=1))
        assert r.success
        assert r.ticket == 100_001

    def test_sell_returns_correct_ticket(self, executor: OrderExecutor, mt5_mock: MagicMock) -> None:
        mt5_mock.order_send.return_value = _ok(order=100_002)
        r = executor.execute_entry(_entry_req(signal=-1))
        assert r.success
        assert r.ticket == 100_002

    def test_tick_unavailable_fails(self, executor: OrderExecutor, mt5_mock: MagicMock) -> None:
        mt5_mock.symbol_info_tick.return_value = None
        r = executor.execute_entry(_entry_req())
        assert not r.success

    def test_broker_reject_fails(self, executor: OrderExecutor, mt5_mock: MagicMock) -> None:
        mt5_mock.order_send.return_value = _fail(10016)
        assert not executor.execute_entry(_entry_req()).success


# ── pending entry ─────────────────────────────────────────────────────────────


class TestPendingEntry:
    def test_buy_stop_success(self, executor: OrderExecutor, mt5_mock: MagicMock) -> None:
        mt5_mock.order_send.return_value = _ok(order=100_003)
        r = executor.execute_entry(_entry_req("stop", 1, entry_price=1.105))
        assert r.success
        assert r.ticket == 100_003

    def test_sell_limit_success(self, executor: OrderExecutor, mt5_mock: MagicMock) -> None:
        mt5_mock.order_send.return_value = _ok(order=100_004)
        r = executor.execute_entry(_entry_req("limit", -1, entry_price=1.095))
        assert r.success

    def test_expiry_with_tick_unavailable_fails(self, executor: OrderExecutor, mt5_mock: MagicMock) -> None:
        # expiration_time set + tick None → _fetch_market_data returns None → EXPIRED
        mt5_mock.symbol_info_tick.return_value = None
        r = executor.execute_entry(_entry_req("stop", 1, entry_price=1.105, expiration_time="23:59"))
        assert not r.success

    def test_invalid_signal_for_pending_fails(self, executor: OrderExecutor, mt5_mock: MagicMock) -> None:
        # signal=2 (bracket) + "stop" → combo absent from _PENDING_ORDER_TYPE_MAP
        mt5_mock.order_send.return_value = _ok()
        r = executor.execute_entry(_entry_req("stop", signal=2, entry_price=1.105))
        assert not r.success


# ── bracket entry ─────────────────────────────────────────────────────────────


class TestBracketEntry:
    @staticmethod
    def _bracket() -> EntryRequest:
        return EntryRequest(
            strategy_name="sma_crossover",
            order_type="bracket",
            symbol="EURUSD",
            volume=0.1,
            signal=2,
            buy_stop=1.105,
            sell_stop=1.095,
            buy_sl=1.09,
            sell_sl=1.11,
        )

    def test_both_legs_success(self, executor: OrderExecutor, mt5_mock: MagicMock) -> None:
        mt5_mock.order_send.side_effect = [_ok(order=1), _ok(order=2)]
        r = executor.execute_entry(self._bracket())
        assert r.success
        assert r.order_tickets == [1, 2]

    def test_buy_leg_reject_fails(self, executor: OrderExecutor, mt5_mock: MagicMock) -> None:
        mt5_mock.order_send.return_value = _fail(10016)
        assert not executor.execute_entry(self._bracket()).success

    def test_sell_leg_reject_cancels_buy_and_fails(self, executor: OrderExecutor, mt5_mock: MagicMock) -> None:
        # buy OK → sell fails → cancel(buy) call → total 3 order_send calls
        mt5_mock.order_send.side_effect = [_ok(order=10), _fail(10016), _ok()]
        r = executor.execute_entry(self._bracket())
        assert not r.success
        assert mt5_mock.order_send.call_count == 3

    def test_market_data_unavailable_fails(self, executor: OrderExecutor, mt5_mock: MagicMock) -> None:
        mt5_mock.symbol_info_tick.return_value = None
        assert not executor.execute_entry(self._bracket()).success


# ── execute exit ──────────────────────────────────────────────────────────────


class TestExecuteExit:
    def test_full_close_success(self, executor: OrderExecutor, mt5_mock: MagicMock) -> None:
        mt5_mock.positions_get.return_value = (_position(),)
        mt5_mock.order_send.return_value = _ok(deal=200_001)
        r = executor.execute_exit(ExitRequest(ticket=100_001))
        assert r.success
        assert r.ticket == 100_001

    def test_already_closed_fails_with_message(self, executor: OrderExecutor, mt5_mock: MagicMock) -> None:
        mt5_mock.positions_get.return_value = ()
        r = executor.execute_exit(ExitRequest(ticket=999_999))
        assert not r.success
        assert "already closed" in r.error_message

    def test_partial_close_executed_volume(self, executor: OrderExecutor, mt5_mock: MagicMock) -> None:
        mt5_mock.positions_get.return_value = (_position(volume=0.2),)
        mt5_mock.order_send.return_value = _ok(deal=200_002)
        r = executor.execute_exit(ExitRequest(ticket=100_001, portion=0.5))
        assert r.success
        assert r.executed_volume == pytest.approx(0.1)

    def test_deal_zero_activates_recovery_path(self, executor: OrderExecutor, mt5_mock: MagicMock) -> None:
        mt5_mock.positions_get.return_value = (_position(),)
        mt5_mock.order_send.return_value = _ok(deal=0)  # deal=0 → recovery
        mt5_mock.history_deals_get.return_value = ()  # recovery finds nothing → None
        r = executor.execute_exit(ExitRequest(ticket=100_001))
        assert r.success
        assert r.deal_id is None


# ── modify ────────────────────────────────────────────────────────────────────


class TestModify:
    def test_sl_tp_change_succeeds(self, executor: OrderExecutor, mt5_mock: MagicMock) -> None:
        mt5_mock.positions_get.return_value = (_position(sl=1.09000, tp=1.11000),)
        mt5_mock.order_send.return_value = _ok()
        r = executor.execute_modify(ModifyRequest(ticket=100_001, new_sl=1.095, new_tp=1.115))
        assert r.success

    def test_no_changes_retcode_is_success(self, executor: OrderExecutor, mt5_mock: MagicMock) -> None:
        mt5_mock.positions_get.return_value = (_position(sl=1.09000, tp=1.11000),)
        # 10025 = TRADE_RETCODE_NO_CHANGES (set in conftest mock)
        mt5_mock.order_send.return_value = _fail(retcode=10025)
        r = executor.execute_modify(ModifyRequest(ticket=100_001, new_sl=1.095))
        assert r.success

    def test_same_sl_tp_skips_order_send(self, executor: OrderExecutor, mt5_mock: MagicMock) -> None:
        mt5_mock.positions_get.return_value = (_position(sl=1.09000, tp=1.11000),)
        # Identical values → short-circuit, no order_send call
        r = executor.execute_modify(ModifyRequest(ticket=100_001, new_sl=1.09, new_tp=1.11))
        assert r.success
        mt5_mock.order_send.assert_not_called()

    def test_position_not_found_fails(self, executor: OrderExecutor, mt5_mock: MagicMock) -> None:
        mt5_mock.positions_get.return_value = ()
        assert not executor.execute_modify(ModifyRequest(ticket=99)).success


# ── retry logic ───────────────────────────────────────────────────────────────


class TestRetry:
    def test_retryable_code_retries_then_succeeds(self, executor: OrderExecutor, mt5_mock: MagicMock) -> None:
        retryable = next(iter(RETRYABLE_CODES))
        mt5_mock.order_send.side_effect = [_fail(retryable), _fail(retryable), _ok()]
        r = executor.execute_entry(_entry_req())
        assert r.success
        assert mt5_mock.order_send.call_count == 3

    def test_non_retryable_code_does_not_retry(self, executor: OrderExecutor, mt5_mock: MagicMock) -> None:
        mt5_mock.order_send.return_value = _fail(10016)  # not in RETRYABLE_CODES
        executor.execute_entry(_entry_req())
        assert mt5_mock.order_send.call_count == 1

    def test_all_retries_exhausted_returns_failure(self, executor: OrderExecutor, mt5_mock: MagicMock) -> None:
        retryable = next(iter(RETRYABLE_CODES))
        mt5_mock.order_send.return_value = _fail(retryable)
        r = executor.execute_entry(_entry_req())
        assert not r.success
        assert mt5_mock.order_send.call_count == executor.max_retries


# ── deal ID recovery ──────────────────────────────────────────────────────────


class TestDealIdRecovery:
    def _recover(
        self,
        executor: OrderExecutor,
        mt5_mock: MagicMock,
        deals: tuple[_Deal, ...],
        close_type: OrderType = OrderType.SELL,
        volume: float = 0.1,
    ) -> int | None:
        mt5_mock.history_deals_get.return_value = deals
        return executor._recover_close_deal_id(
            ticket=100_001,
            close_type=close_type,
            close_volume=volume,
            volume_step=0.01,
            max_retries=1,
        )

    def test_matching_deal_returned(self, executor: OrderExecutor, mt5_mock: MagicMock) -> None:
        d = _deal_nt(ticket=999, deal_type=1, entry=1, position_id=100_001, volume=0.1)
        assert self._recover(executor, mt5_mock, (d,)) == 999

    def test_no_deals_returns_none(self, executor: OrderExecutor, mt5_mock: MagicMock) -> None:
        assert self._recover(executor, mt5_mock, ()) is None

    def test_wrong_deal_type_unmatched(self, executor: OrderExecutor, mt5_mock: MagicMock) -> None:
        d = _deal_nt(ticket=888, deal_type=0, volume=0.1)  # BUY type; close_type=SELL
        assert self._recover(executor, mt5_mock, (d,), close_type=OrderType.SELL) is None

    def test_wrong_volume_unmatched(self, executor: OrderExecutor, mt5_mock: MagicMock) -> None:
        d = _deal_nt(ticket=777, deal_type=1, volume=0.5)  # volume mismatch
        assert self._recover(executor, mt5_mock, (d,), volume=0.1) is None

    def test_position_id_match_ranked_higher(self, executor: OrderExecutor, mt5_mock: MagicMock) -> None:
        wrong_pos = _deal_nt(ticket=100, deal_type=1, position_id=999_999, volume=0.1, time_msc=1000)
        correct_pos = _deal_nt(ticket=200, deal_type=1, position_id=100_001, volume=0.1, time_msc=999)
        # correct_pos has lower time_msc but higher rank due to position_id match
        result = self._recover(executor, mt5_mock, (wrong_pos, correct_pos))
        assert result == 200


# ── cancel order ──────────────────────────────────────────────────────────────


class TestCancelOrder:
    def test_success(self, executor: OrderExecutor, mt5_mock: MagicMock) -> None:
        mt5_mock.order_send.return_value = _ok()
        assert executor.cancel_order(12345) is True

    def test_failure(self, executor: OrderExecutor, mt5_mock: MagicMock) -> None:
        mt5_mock.order_send.return_value = _fail()
        assert executor.cancel_order(12345) is False
