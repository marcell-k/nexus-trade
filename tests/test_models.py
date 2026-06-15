"""Unit tests for nexus_trade.core.models — pure data classes, no MT5 calls."""

from __future__ import annotations

import pytest

from nexus_trade.core.models import (
    ExitLogData,
    NormalizedPosition,
    Position,
    PositionType,
    Tick,
    cache_entry_to_position,
    order_succeeded,
    order_ticket,
)
from nexus_trade.core.state import PositionCacheEntry

#  NormalizedPosition.from_mt5 


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
        raw = self._raw()
        pos = NormalizedPosition.from_mt5(raw)
        assert pos.ticket == 100_001
        assert pos.symbol == "EURUSD"
        assert pos.type == 0
        assert pos.volume == pytest.approx(0.10)
        assert pos.price_open == pytest.approx(1.10000)
        assert pos.sl == pytest.approx(1.09500)
        assert pos.tp == pytest.approx(1.11000)
        assert pos.profit == pytest.approx(15.0)
        assert pos.swap == pytest.approx(-0.5)
        assert pos.magic == 12345
        assert pos.time == 1_700_000_000

    def test_missing_attributes_use_defaults(self) -> None:
        """from_mt5 uses getattr with defaults — tolerates incomplete objects."""

        class _Sparse:
            ticket = 99

        pos = NormalizedPosition.from_mt5(_Sparse())
        assert pos.ticket == 99
        assert pos.symbol == ""
        assert pos.type == 0
        assert pos.volume == pytest.approx(0.0)
        assert pos.magic == 0

    def test_type_coercion(self) -> None:
        """String ticket / float magic must be coerced to correct types."""
        raw = self._raw(ticket="55555", magic="99")
        pos = NormalizedPosition.from_mt5(raw)
        assert isinstance(pos.ticket, int)
        assert isinstance(pos.magic, int)
        assert pos.ticket == 55555
        assert pos.magic == 99


#  NormalizedPosition.to_cache_entry 


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

    def test_returns_correct_typed_dict(self) -> None:
        entry = self._pos().to_cache_entry()
        assert entry["ticket"] == 1
        assert entry["symbol"] == "EURUSD"
        assert entry["type"] == 0
        assert entry["volume"] == pytest.approx(0.1)
        assert entry["magic"] == 42

    def test_all_keys_present(self) -> None:
        required = {"ticket", "symbol", "type", "volume", "price_open", "sl", "tp", "profit", "swap", "magic", "time"}
        assert required.issubset(self._pos().to_cache_entry().keys())


#  NormalizedPosition.to_partial_snapshot 


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


#  Tick 


class TestTick:
    def _raw_tick(self) -> object:
        class _T:
            time = 1_700_000_000
            bid = 1.10000
            ask = 1.10002
            last = 0.0
            volume = 100
            time_msc = 1_700_000_000_000

        return _T()

    def test_from_mt5_fields(self) -> None:
        t = Tick.from_mt5(self._raw_tick())
        assert t.bid == pytest.approx(1.10000)
        assert t.ask == pytest.approx(1.10002)
        assert t.time == 1_700_000_000

    def test_spread_property(self) -> None:
        t = Tick.from_mt5(self._raw_tick())
        assert t.spread == pytest.approx(0.00002)

    def test_mid_property(self) -> None:
        t = Tick.from_mt5(self._raw_tick())
        assert t.mid == pytest.approx(1.10001)

    def test_spread_zero_when_bid_equals_ask(self) -> None:
        class _Same:
            time = 0
            bid = 1.1
            ask = 1.1
            last = 0.0
            volume = 0
            time_msc = 0

        t = Tick.from_mt5(_Same())
        assert t.spread == pytest.approx(0.0)


#  order_succeeded 


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

    def test_returns_false_when_no_retcode_attribute(self) -> None:
        assert order_succeeded(object()) is False


#  order_ticket 


class TestOrderTicket:
    def test_extracts_order_attribute(self) -> None:
        class _R:
            order = 99999

        assert order_ticket(_R()) == 99999

    def test_returns_zero_when_missing(self) -> None:
        assert order_ticket(object()) == 0


#  cache_entry_to_position 


class TestCacheEntryToPosition:
    def _entry(self, **overrides: object) -> PositionCacheEntry:
        base: PositionCacheEntry = {
            "ticket": 1001,
            "symbol": "EURUSD",
            "type": 0,
            "volume": 0.10,
            "price_open": 1.10000,
            "sl": 1.09500,
            "tp": 1.11000,
            "profit": 5.0,
            "swap": 0.0,
            "magic": 42,
            "time": 100,
        }
        base.update(overrides)  # type: ignore[typeddict-item]
        return base

    def test_buy_type_conversion(self) -> None:
        pos = cache_entry_to_position(self._entry(type=0))
        assert pos.type == PositionType.BUY

    def test_sell_type_conversion(self) -> None:
        pos = cache_entry_to_position(self._entry(type=1))
        assert pos.type == PositionType.SELL

    def test_sl_zero_becomes_none(self) -> None:
        pos = cache_entry_to_position(self._entry(sl=0.0))
        assert pos.sl is None

    def test_tp_zero_becomes_none(self) -> None:
        pos = cache_entry_to_position(self._entry(tp=0.0))
        assert pos.tp is None

    def test_nonzero_sl_tp_preserved(self) -> None:
        pos = cache_entry_to_position(self._entry(sl=1.09500, tp=1.11000))
        assert pos.sl == pytest.approx(1.09500)
        assert pos.tp == pytest.approx(1.11000)

    def test_price_current_set_to_price_open(self) -> None:
        pos = cache_entry_to_position(self._entry(price_open=1.10500))
        assert pos.price_current == pytest.approx(1.10500)

    def test_all_scalar_fields(self) -> None:
        pos = cache_entry_to_position(self._entry())
        assert pos.ticket == 1001
        assert pos.symbol == "EURUSD"
        assert pos.volume == pytest.approx(0.10)
        assert pos.magic == 42


#  ExitLogData 


class TestExitLogData:
    def test_defaults(self) -> None:
        d = ExitLogData(
            ticket=1,
            expected_exit_price=1.11,
            exit_trigger="TP",
            expected_entry_price=1.10,
            opening_sl=1.09,
            entry_price=1.10,
        )
        assert d.closed_volume is None
        assert d.deal_id is None

    def test_optional_fields_settable(self) -> None:
        d = ExitLogData(
            ticket=1,
            expected_exit_price=1.11,
            exit_trigger="TP",
            expected_entry_price=1.10,
            opening_sl=1.09,
            entry_price=1.10,
            closed_volume=0.05,
            deal_id=9999,
        )
        assert d.closed_volume == pytest.approx(0.05)
        assert d.deal_id == 9999


#  Position frozen dataclass 


class TestPosition:
    def test_immutable(self) -> None:
        pos = Position(
            ticket=1,
            symbol="EURUSD",
            type=PositionType.BUY,
            magic=1,
            volume=0.1,
            price_open=1.1,
            price_current=1.1,
            sl=None,
            tp=None,
            profit=0.0,
        )
        with pytest.raises((TypeError, AttributeError)):
            pos.ticket = 99  # type: ignore[misc]
