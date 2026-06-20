"""Unit tests for nexus_trade.core.models — pure data classes, no MT5 calls."""

from __future__ import annotations

import pytest

from nexus_trade.core.models import (
    NormalizedPosition,
    Position,
    Tick,
    cache_entry_to_position,
    order_succeeded,
)
from nexus_trade.core.types import PositionCacheEntry, PositionType


class TestNormalizedPositionFromMt5:
    def _raw(self, **overrides: object) -> object:
        class _Raw:
            ticket = 100_001
            symbol = "EURUSD"
            type = 0
            volume = 0.10
            price_open = 1.10000
            sl = 1.09500
            tp = 1.11000
            profit = 15.0
            swap = -0.5
            magic = 12345
            time = 1_700_000_000

        for k, v in overrides.items():
            setattr(_Raw, k, v)
        return _Raw()

    def test_all_fields_mapped(self) -> None:
        pos = NormalizedPosition.from_mt5(self._raw())
        assert pos.ticket == 100_001
        assert pos.symbol == "EURUSD"
        assert pos.type == 0
        assert pos.volume == pytest.approx(0.10)
        assert pos.sl == pytest.approx(1.09500)
        assert pos.profit == pytest.approx(15.0)
        assert pos.swap == pytest.approx(-0.5)
        assert pos.magic == 12345
        assert pos.time == 1_700_000_000

    def test_missing_attributes_use_defaults(self) -> None:
        class _Sparse:
            ticket = 99

        pos = NormalizedPosition.from_mt5(_Sparse())
        assert pos.ticket == 99
        assert pos.symbol == ""
        assert pos.volume == pytest.approx(0.0)
        assert pos.magic == 0

    def test_type_coercion(self) -> None:
        pos = NormalizedPosition.from_mt5(self._raw(ticket="55555", magic="99"))
        assert isinstance(pos.ticket, int)
        assert isinstance(pos.magic, int)
        assert pos.ticket == 55555
        assert pos.magic == 99


class TestNormalizedPositionToCacheEntry:
    def _pos(self) -> NormalizedPosition:
        return NormalizedPosition(
            ticket=1,
            symbol="EURUSD",
            type=0,
            volume=0.1,
            price_open=1.1,
            sl=1.09,
            tp=1.11,
            profit=5.0,
            swap=-0.1,
            magic=42,
            time=100,
        )

    def test_all_keys_present(self) -> None:
        required = {"ticket", "symbol", "type", "volume", "price_open", "sl", "tp", "profit", "swap", "magic", "time"}
        assert required.issubset(self._pos().to_cache_entry().keys())


class TestNormalizedPositionToPartialSnapshot:
    def test_fields(self) -> None:
        pos = NormalizedPosition(
            ticket=777,
            symbol="GBPUSD",
            type=1,
            volume=0.2,
            price_open=1.25,
            sl=1.26,
            tp=1.24,
            profit=-3.0,
            swap=0.0,
            magic=9,
            time=200,
        )
        snap = pos.to_partial_snapshot()
        assert snap.ticket == 777
        assert snap.symbol == "GBPUSD"
        assert snap.type == 1
        assert snap.swap == pytest.approx(0.0)


class TestTick:
    class _MockTick:
        time: int = 1_700_000_000
        bid: float = 1.10000
        ask: float = 1.10002
        last: float = 0.0
        volume: int = 100
        time_msc: int = 1_700_000_000_000
        flags: int = 0
        volume_real: int = 1

    def test_from_mt5_fields(self) -> None:
        t = Tick.from_mt5(self._MockTick())
        assert t.bid == pytest.approx(1.10000)
        assert t.ask == pytest.approx(1.10002)
        assert t.time == 1_700_000_000


class TestOrderSucceeded:
    def test_returns_true_on_done(self) -> None:
        class _R:
            retcode = 10009

        assert order_succeeded(_R()) is True

    def test_returns_false_on_other_retcode(self) -> None:
        class _R:
            retcode = 10006

        assert order_succeeded(_R()) is False

    def test_returns_false_on_none(self) -> None:
        assert order_succeeded(None) is False


class TestCacheEntryToPosition:
    def _entry(self, **overrides: object) -> PositionCacheEntry:
        base = PositionCacheEntry(
            ticket=1001,
            symbol="EURUSD",
            type=0,
            volume=0.10,
            price_open=1.10000,
            sl=1.09500,
            tp=1.11000,
            profit=5.0,
            swap=0.0,
            magic=42,
            time=100,
        )
        for k, v in overrides.items():
            base[k] = v  # type: ignore[literal-required]
        return base

    def test_buy_type_conversion(self) -> None:
        assert cache_entry_to_position(self._entry(type=0)).type == PositionType.BUY

    def test_sl_zero_becomes_none(self) -> None:
        assert cache_entry_to_position(self._entry(sl=0.0)).sl is None

    def test_nonzero_sl_tp_preserved(self) -> None:
        pos = cache_entry_to_position(self._entry(sl=1.09500, tp=1.11000))
        assert pos.sl == pytest.approx(1.09500)
        assert pos.tp == pytest.approx(1.11000)


class TestPosition:
    def test_immutable(self) -> None:
        pos = Position(
            ticket=1,
            symbol="EURUSD",
            type=PositionType.BUY,
            magic=1,
            volume=0.1,
            price_open=1.1,
            sl=None,
            tp=None,
            profit=0.0,
        )
        with pytest.raises((TypeError, AttributeError)):
            pos.ticket = 99  # type: ignore[misc]
