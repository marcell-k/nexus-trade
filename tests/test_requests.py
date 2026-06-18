"""Unit tests for nexus_trade.execution.request — validation in __post_init__."""

from __future__ import annotations

import math

import pytest

from nexus_trade.execution.request import (
    EntryRequest,
    ExecutionResult,
    ExitRequest,
    ModifyRequest,
)


class TestEntryRequestValidation:
    def _valid(self, **overrides: object) -> dict:
        base: dict = {
            "strategy_name": "test_strategy",
            "order_type": "market",
            "symbol": "EURUSD",
            "volume": 0.10,
            "signal": 1,
        }
        base.update(overrides)
        return base

    def test_valid_market_buy(self) -> None:
        req = EntryRequest(**self._valid())
        assert req.order_type == "market"
        assert req.signal == 1

    def test_valid_market_sell(self) -> None:
        req = EntryRequest(**self._valid(signal=-1))
        assert req.signal == -1

    def test_valid_bracket_signal(self) -> None:
        req = EntryRequest(**self._valid(order_type="bracket", signal=2))
        assert req.signal == 2

    @pytest.mark.parametrize("order_type", ["limit", "stop", "bracket"])
    def test_valid_order_types(self, order_type: str) -> None:
        sig = 2 if order_type == "bracket" else 1
        EntryRequest(**self._valid(order_type=order_type, signal=sig))

    def test_invalid_order_type_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid order_type"):
            EntryRequest(**self._valid(order_type="OCO"))

    @pytest.mark.parametrize("signal", [0, 3, -2, 99])
    def test_invalid_signal_raises(self, signal: int) -> None:
        with pytest.raises(ValueError, match="Invalid signal"):
            EntryRequest(**self._valid(signal=signal))

    def test_negative_volume_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid volume"):
            EntryRequest(**self._valid(volume=-0.01))

    def test_zero_volume_is_valid(self) -> None:
        req = EntryRequest(**self._valid(volume=0.0))
        assert req.volume == pytest.approx(0.0)

    def test_infinite_volume_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid volume"):
            EntryRequest(**self._valid(volume=math.inf))

    def test_nan_volume_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid volume"):
            EntryRequest(**self._valid(volume=math.nan))

    def test_optional_fields_default_none(self) -> None:
        req = EntryRequest(**self._valid())
        assert req.sl is None
        assert req.tp is None
        assert req.entry_price is None
        assert req.comment == ""

    def test_sl_tp_settable(self) -> None:
        req = EntryRequest(**self._valid(sl=1.09500, tp=1.11000))
        assert req.sl == pytest.approx(1.09500)
        assert req.tp == pytest.approx(1.11000)

    def test_bracket_fields_settable(self) -> None:
        req = EntryRequest(
            strategy_name="test_strategy",
            order_type="bracket",
            symbol="EURUSD",
            volume=0.1,
            signal=2,
            buy_stop=1.10100,
            sell_stop=1.09900,
            buy_sl=1.09500,
            sell_sl=1.10500,
            buy_tp=1.11000,
            sell_tp=1.09000,
        )
        assert req.buy_stop == pytest.approx(1.10100)
        assert req.sell_stop == pytest.approx(1.09900)


class TestExitRequestValidation:
    def test_valid_full_close(self) -> None:
        req = ExitRequest(ticket=100001)
        assert req.portion == pytest.approx(1.0)

    def test_valid_half_close(self) -> None:
        req = ExitRequest(ticket=100001, portion=0.5)
        assert req.portion == pytest.approx(0.5)

    def test_ticket_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid ticket"):
            ExitRequest(ticket=0)

    def test_ticket_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid ticket"):
            ExitRequest(ticket=-1)

    def test_portion_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid portion"):
            ExitRequest(ticket=1, portion=0.0)

    def test_portion_above_one_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid portion"):
            ExitRequest(ticket=1, portion=1.001)

    def test_portion_exactly_one_is_valid(self) -> None:
        req = ExitRequest(ticket=1, portion=1.0)
        assert req.portion == pytest.approx(1.0)

    def test_optional_fields_default(self) -> None:
        req = ExitRequest(ticket=1)
        assert req.comment == ""
        assert req.expected_exit_price is None
        assert req.strategy_name is None
        assert req.exit_reason == ""


class TestModifyRequest:
    def test_all_none_fields(self) -> None:
        req = ModifyRequest(ticket=1)
        assert req.new_sl is None
        assert req.new_tp is None
        assert req.comment == ""

    def test_with_new_sl(self) -> None:
        req = ModifyRequest(ticket=1, new_sl=1.09000)
        assert req.new_sl == pytest.approx(1.09000)


class TestExecutionResult:
    def test_success_result(self) -> None:
        r = ExecutionResult(success=True, ticket=100001)
        assert r.success is True
        assert r.ticket == 100001
        assert r.error_message == ""

    def test_failure_result(self) -> None:
        r = ExecutionResult(success=False, error_message="timeout")
        assert r.success is False
        assert r.error_message == "timeout"
        assert r.ticket is None
