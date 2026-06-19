"""Unit tests for nexus_trade.execution.request — validation in __post_init__."""

from __future__ import annotations

import math

import pytest

from nexus_trade.execution.request import EntryRequest, ExitRequest


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


class TestExitRequestValidation:
    def test_valid_half_close(self) -> None:
        req = ExitRequest(ticket=100001, portion=0.5)
        assert req.portion == pytest.approx(0.5)

    def test_ticket_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid ticket"):
            ExitRequest(ticket=0)

    def test_portion_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid portion"):
            ExitRequest(ticket=1, portion=0.0)

    def test_portion_above_one_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid portion"):
            ExitRequest(ticket=1, portion=1.001)

    def test_portion_exactly_one_is_valid(self) -> None:
        req = ExitRequest(ticket=1, portion=1.0)
        assert req.portion == pytest.approx(1.0)
