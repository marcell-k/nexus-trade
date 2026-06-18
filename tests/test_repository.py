"""Unit tests for PositionRepository — cache hit/miss/stale paths."""

from __future__ import annotations

import time
from collections import namedtuple
from typing import TYPE_CHECKING, cast
from unittest.mock import MagicMock

import pytest

from nexus_trade.core.repository import PositionRepository
from nexus_trade.core.types import PositionCacheEntry

if TYPE_CHECKING:
    from collections.abc import Callable

    from nexus_trade.core.state import SharedState


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
def fresh_state(make_position_cache_entry: Callable[[], PositionCacheEntry]) -> SharedState:
    entry = make_position_cache_entry()
    state = {
        "position_cache": {entry["ticket"]: entry},
        "position_cache_timestamp": time.time(),
    }
    return cast("SharedState", state)


@pytest.fixture
def repo(fresh_state: SharedState) -> PositionRepository:
    return PositionRepository(
        shared_state=fresh_state,
        position_cache_lock=_make_lock(),
        cache_staleness_threshold=60,
    )


class TestCacheHit:
    def test_returns_matching_symbol_and_magic(self, repo: PositionRepository) -> None:
        positions = repo.get_strategy_positions(symbol="EURUSD", magic=12345, prefer_cache=True)
        assert positions is not None
        assert len(positions) == 1
        assert positions[0]["symbol"] == "EURUSD"
        assert positions[0]["magic"] == 12345

    def test_filters_by_magic(self, fresh_state: SharedState) -> None:
        e1 = _make_entry(1, magic=100)
        e2 = _make_entry(2, magic=200)
        fresh_state["position_cache"] = {1: e1, 2: e2}
        repo = PositionRepository(fresh_state, _make_lock(), 60)
        result = repo.get_strategy_positions("EURUSD", magic=100, prefer_cache=True)
        assert result is not None
        assert len(result) == 1
        assert result[0]["magic"] == 100

    def test_filters_by_symbol(self, fresh_state: SharedState) -> None:
        e1 = _make_entry(1, symbol="EURUSD")
        e2 = _make_entry(2, symbol="GBPUSD")
        fresh_state["position_cache"] = {1: e1, 2: e2}
        repo = PositionRepository(fresh_state, _make_lock(), 60)
        result = repo.get_strategy_positions("EURUSD", magic=100, prefer_cache=True)
        assert result is not None
        assert all(p["symbol"] == "EURUSD" for p in result)

    def test_known_tickets_filter_applied(self, fresh_state: SharedState) -> None:
        e1 = _make_entry(1)
        e2 = _make_entry(2)
        fresh_state["position_cache"] = {1: e1, 2: e2}
        repo = PositionRepository(fresh_state, _make_lock(), 60)
        result = repo.get_strategy_positions("EURUSD", magic=100, known_tickets=frozenset({1}), prefer_cache=True)
        assert result is not None
        assert len(result) == 1
        assert result[0]["ticket"] == 1

    def test_empty_cache_returns_empty_list(self) -> None:
        repo = PositionRepository(cast("SharedState", {}), _make_lock(), 60)
        result = repo.get_strategy_positions("EURUSD", magic=99, prefer_cache=True)
        assert result == []


class TestCacheStale:
    def test_stale_cache_falls_through_to_mt5(self, fresh_state: SharedState, _mt5_mock: MagicMock) -> None:
        import sys

        sys.modules["MetaTrader5"].positions_get.return_value = ()
        fresh_state["position_cache_timestamp"] = time.time() - 120
        repo = PositionRepository(fresh_state, _make_lock(), 60)
        result = repo.get_strategy_positions("EURUSD", magic=99, prefer_cache=True)
        assert result == []
        sys.modules["MetaTrader5"].positions_get.assert_called()

    def test_fresh_cache_skips_mt5(self, fresh_state: SharedState, _mt5_mock: MagicMock) -> None:
        import sys

        repo = PositionRepository(fresh_state, _make_lock(), 60)
        repo.get_strategy_positions("EURUSD", magic=12345, prefer_cache=True)
        sys.modules["MetaTrader5"].positions_get.assert_not_called()


class TestCacheMetrics:
    def test_age_is_recent(self, repo: PositionRepository) -> None:
        assert 0.0 <= repo.cache_age_seconds() < 5.0

    def test_is_fresh_when_recent(self, repo: PositionRepository) -> None:
        assert repo.is_cache_fresh() is True

    def test_is_stale_when_old(self, fresh_state: SharedState) -> None:
        fresh_state["position_cache_timestamp"] = time.time() - 120
        repo = PositionRepository(fresh_state, _make_lock(), 60)
        assert repo.is_cache_fresh() is False


class TestGetManagedPositions:
    def test_filters_by_magic_numbers(self, _mt5_mock: MagicMock) -> None:
        import sys

        RawPos = namedtuple(
            "P",
            "ticket time time_msc time_update time_update_msc type magic identifier "
            "reason volume price_open sl tp price_current swap profit symbol comment external_id",
        )
        p1 = RawPos(1, 0, 0, 0, 0, 0, 100, 0, 0, 0.1, 1.1, 1.09, 1.11, 1.1, 0.0, 0.0, "EURUSD", "", "")
        p2 = RawPos(2, 0, 0, 0, 0, 0, 999, 0, 0, 0.1, 1.1, 1.09, 1.11, 1.1, 0.0, 0.0, "EURUSD", "", "")
        sys.modules["MetaTrader5"].positions_get.return_value = (p1, p2)

        repo = PositionRepository(cast("SharedState", {}), _make_lock(), 60)
        result = repo.get_managed_positions(frozenset({100}))
        assert result is not None
        assert len(result) == 1
        assert result[0]["magic"] == 100

    def test_returns_none_on_api_failure(self, _mt5_mock: MagicMock) -> None:
        import sys

        sys.modules["MetaTrader5"].positions_get.return_value = None
        sys.modules["MetaTrader5"].last_error.return_value = (1, "error")
        repo = PositionRepository(cast("SharedState", {}), _make_lock(), 60)
        assert repo.get_managed_positions(frozenset({100})) is None
