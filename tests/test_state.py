"""Unit tests for nexus_trade.core.state — TTLCache, normalize_order."""

from __future__ import annotations

import time

import pytest

from nexus_trade.core.state import OrderSnapshot, TTLCache, normalize_order

#  TTLCache


class TestTTLCache:
    def test_empty_cache_is_invalid(self) -> None:
        cache: TTLCache[int] = TTLCache()
        assert cache.is_valid(ttl=60.0) is False

    def test_set_makes_cache_valid(self) -> None:
        cache: TTLCache[str] = TTLCache()
        cache.set("hello")
        assert cache.is_valid(ttl=60.0) is True
        assert cache.value == "hello"

    def test_expires_after_ttl(self) -> None:
        cache: TTLCache[int] = TTLCache()
        cache.set(42)
        # Backdate timestamp so cache appears expired
        cache.timestamp = time.time() - 100.0
        assert cache.is_valid(ttl=60.0) is False

    def test_within_ttl_is_valid(self) -> None:
        cache: TTLCache[int] = TTLCache()
        cache.set(99)
        cache.timestamp = time.time() - 30.0
        assert cache.is_valid(ttl=60.0) is True

    def test_invalidate_clears_value_and_timestamp(self) -> None:
        cache: TTLCache[float] = TTLCache()
        cache.set(3.14)
        cache.invalidate()
        assert cache.value is None
        assert cache.timestamp == 0.0
        assert cache.is_valid(ttl=60.0) is False

    def test_set_updates_timestamp(self) -> None:
        cache: TTLCache[str] = TTLCache()
        before = time.time()
        cache.set("x")
        after = time.time()
        assert before <= cache.timestamp <= after

    def test_overwrite_resets_ttl(self) -> None:
        cache: TTLCache[int] = TTLCache()
        cache.set(1)
        cache.timestamp = time.time() - 90.0  # expire it
        assert cache.is_valid(ttl=60.0) is False
        cache.set(2)  # fresh write
        assert cache.is_valid(ttl=60.0) is True
        assert cache.value == 2

    def test_zero_ttl_always_invalid(self) -> None:
        cache: TTLCache[int] = TTLCache()
        cache.set(1)
        assert cache.is_valid(ttl=0.0) is False

    def test_none_value_after_init(self) -> None:
        cache: TTLCache[list[int]] = TTLCache()
        assert cache.value is None


#  normalize_order


class TestNormalizeOrder:
    def test_from_dict(self) -> None:
        d = {"ticket": 1, "symbol": "EURUSD", "type": 4, "magic": 99}
        snap = normalize_order(d)
        assert snap.ticket == 1
        assert snap.symbol == "EURUSD"
        assert snap.type == 4
        assert snap.magic == 99

    def test_from_object(self) -> None:
        class _Order:
            ticket = 555
            symbol = "GBPUSD"
            type = 5
            magic = 77

        snap = normalize_order(_Order())
        assert isinstance(snap, OrderSnapshot)
        assert snap.ticket == 555
        assert snap.symbol == "GBPUSD"
        assert snap.type == 5
        assert snap.magic == 77

    def test_dict_coerces_int(self) -> None:
        d = {"ticket": "200", "symbol": "XAUUSD", "type": "4", "magic": "88"}
        snap = normalize_order(d)
        assert isinstance(snap.ticket, int)
        assert snap.ticket == 200

    def test_order_snapshot_is_frozen(self) -> None:
        snap = OrderSnapshot(ticket=1, symbol="X", type=0, magic=0)
        with pytest.raises((TypeError, AttributeError)):
            snap.ticket = 99  # type: ignore[misc]

    def test_result_type(self) -> None:
        snap = normalize_order({"ticket": 1, "symbol": "S", "type": 0, "magic": 0})
        assert isinstance(snap, OrderSnapshot)
