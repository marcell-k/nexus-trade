"""Integration tests for PositionRepository — cache hit/miss/stale paths."""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest

from nexus_trade.core.repository import PositionRepository
from nexus_trade.core.state import PositionCacheEntry


def _make_lock() -> MagicMock:
    lock = MagicMock()
    lock.__enter__ = MagicMock(return_value=None)
    lock.__exit__ = MagicMock(return_value=False)
    return lock


def _make_entry(ticket: int, symbol: str = "EURUSD", magic: int = 100) -> PositionCacheEntry:
    return PositionCacheEntry(
        ticket=ticket,
        symbol=symbol,
        type=0,
        volume=0.1,
        price_open=1.1,
        sl=1.09,
        tp=1.11,
        profit=0.0,
        swap=0.0,
        magic=magic,
        time=0,
    )


@pytest.fixture
def fresh_state(make_position_cache_entry) -> dict:
    entry = make_position_cache_entry()
    return {
        "position_cache": {entry["ticket"]: entry},
        "position_cache_timestamp": time.time(),
    }


@pytest.fixture
def repo(fresh_state: dict) -> PositionRepository:
    return PositionRepository(
        shared_state=fresh_state,
        position_cache_lock=_make_lock(),
        cache_staleness_threshold=60,
    )


#  cache hit 


class TestCacheHit:
    def test_returns_matching_symbol_and_magic(self, repo: PositionRepository, make_position_cache_entry) -> None:
        positions = repo.get_strategy_positions(symbol="EURUSD", magic=12345, prefer_cache=True)
        assert positions is not None
        assert len(positions) == 1
        assert positions[0]["symbol"] == "EURUSD"
        assert positions[0]["magic"] == 12345

    def test_filters_by_magic(self, fresh_state: dict) -> None:
        e1 = _make_entry(1, magic=100)
        e2 = _make_entry(2, magic=200)
        fresh_state["position_cache"] = {1: e1, 2: e2}
        repo = PositionRepository(fresh_state, _make_lock(), 60)
        result = repo.get_strategy_positions("EURUSD", magic=100, prefer_cache=True)
        assert result is not None
        assert all(p["magic"] == 100 for p in result)
        assert len(result) == 1

    def test_filters_by_symbol(self, fresh_state: dict) -> None:
        e1 = _make_entry(1, symbol="EURUSD")
        e2 = _make_entry(2, symbol="GBPUSD")
        fresh_state["position_cache"] = {1: e1, 2: e2}
        repo = PositionRepository(fresh_state, _make_lock(), 60)
        result = repo.get_strategy_positions("EURUSD", magic=100, prefer_cache=True)
        assert result is not None
        assert all(p["symbol"] == "EURUSD" for p in result)

    def test_known_tickets_filter_applied(self, fresh_state: dict) -> None:
        e1 = _make_entry(1)
        e2 = _make_entry(2)
        fresh_state["position_cache"] = {1: e1, 2: e2}
        repo = PositionRepository(fresh_state, _make_lock(), 60)
        result = repo.get_strategy_positions("EURUSD", magic=100, known_tickets=frozenset({1}), prefer_cache=True)
        assert result is not None
        assert len(result) == 1
        assert result[0]["ticket"] == 1

    def test_empty_cache_returns_empty_list(self) -> None:
        state = {"position_cache": {}, "position_cache_timestamp": time.time()}
        repo = PositionRepository(state, _make_lock(), 60)
        result = repo.get_strategy_positions("EURUSD", magic=99, prefer_cache=True)
        assert result == []


#  cache stale 


class TestCacheStale:
    def test_stale_cache_falls_through_to_mt5(self, fresh_state: dict, mt5_mock: MagicMock) -> None:
        import sys

        mt5 = sys.modules["MetaTrader5"]
        mt5.positions_get.return_value = ()

        fresh_state["position_cache_timestamp"] = time.time() - 120  # 2 min old
        repo = PositionRepository(fresh_state, _make_lock(), staleness_threshold=60)
        result = repo.get_strategy_positions("EURUSD", magic=99, prefer_cache=True)
        assert result == []
        mt5.positions_get.assert_called()

    def test_fresh_cache_skips_mt5(self, fresh_state: dict, mt5_mock: MagicMock) -> None:
        import sys

        mt5 = sys.modules["MetaTrader5"]
        repo = PositionRepository(fresh_state, _make_lock(), 60)
        repo.get_strategy_positions("EURUSD", magic=12345, prefer_cache=True)
        mt5.positions_get.assert_not_called()


#  cache_age_seconds / is_cache_fresh 


class TestCacheMetrics:
    def test_age_increases_over_time(self, repo: PositionRepository) -> None:
        age = repo.cache_age_seconds()
        assert 0.0 <= age < 5.0

    def test_is_fresh_when_recent(self, repo: PositionRepository) -> None:
        assert repo.is_cache_fresh() is True

    def test_is_stale_when_old(self, fresh_state: dict) -> None:
        fresh_state["position_cache_timestamp"] = time.time() - 120
        repo = PositionRepository(fresh_state, _make_lock(), 60)
        assert repo.is_cache_fresh() is False


#  get_managed_positions 


class TestGetManagedPositions:
    def test_filters_by_magic_numbers(self, mt5_mock: MagicMock) -> None:
        import sys
        from collections import namedtuple

        mt5 = sys.modules["MetaTrader5"]

        RawPos = namedtuple(
            "P",
            "ticket time time_msc time_update time_update_msc type magic identifier "
            "reason volume price_open sl tp price_current swap profit symbol comment external_id",
        )
        p1 = RawPos(1, 0, 0, 0, 0, 0, 100, 0, 0, 0.1, 1.1, 1.09, 1.11, 1.1, 0.0, 0.0, "EURUSD", "", "")
        p2 = RawPos(2, 0, 0, 0, 0, 0, 999, 0, 0, 0.1, 1.1, 1.09, 1.11, 1.1, 0.0, 0.0, "EURUSD", "", "")
        mt5.positions_get.return_value = (p1, p2)

        state: dict = {}
        repo = PositionRepository(state, _make_lock(), 60)
        result = repo.get_managed_positions(frozenset({100}))
        assert result is not None
        assert len(result) == 1
        assert result[0]["magic"] == 100

    def test_returns_none_on_api_failure(self, mt5_mock: MagicMock) -> None:
        import sys

        mt5 = sys.modules["MetaTrader5"]
        mt5.positions_get.return_value = None
        mt5.last_error.return_value = (1, "error")

        repo = PositionRepository({}, _make_lock(), 60)
        result = repo.get_managed_positions(frozenset({100}))
        assert result is None
