"""Unit tests for nexus_trade.core.types — TTLCache, OrderSnapshot, normalize_order."""

from __future__ import annotations

import time

import pytest

from nexus_trade.core.models import normalize_order
from nexus_trade.core.types import OrderSnapshot, TTLCache


class TestTTLCache:
    def test_empty_cache_is_invalid(self) -> None:
        cache: TTLCache[int] = TTLCache()
        assert cache.is_valid(ttl=60.0) is False

    def test_set_makes_cache_valid(self) -> None:
        cache: TTLCache[str] = TTLCache()
        cache.set("alma")
        assert cache.is_valid(ttl=60.0) is True
        assert cache.value == "alma"

    def test_expires_after_ttl(self) -> None:
        cache: TTLCache[int] = TTLCache()
        cache.set(42)
        cache.timestamp = time.time() - 100.0
        assert cache.is_valid(ttl=60.0) is False

    def test_invalidate_clears_value_and_timestamp(self) -> None:
        cache: TTLCache[float] = TTLCache()
        cache.set(3.14)
        cache.invalidate()
        assert cache.value is None
        assert cache.timestamp == 0.0
        assert cache.is_valid(ttl=60.0) is False

    def test_overwrite_resets_ttl(self) -> None:
        cache: TTLCache[int] = TTLCache()
        cache.set(1)
        cache.timestamp = time.time() - 90.0
        assert cache.is_valid(ttl=60.0) is False
        cache.set(2)
        assert cache.is_valid(ttl=60.0) is True
        assert cache.value == 2


class TestNormalizeOrder:
    def test_from_namedtuple(self) -> None:
        from collections import namedtuple

        Order = namedtuple("Order", "ticket symbol type magic")
        snap = normalize_order(Order(ticket=1, symbol="EURUSD", type=4, magic=99))
        assert snap.ticket == 1
        assert snap.symbol == "EURUSD"
        assert snap.type == 4
        assert snap.magic == 99

    def test_order_snapshot_is_frozen(self) -> None:
        snap = OrderSnapshot(ticket=1, symbol="X", type=0, magic=0)
        with pytest.raises((TypeError, AttributeError)):
            snap.ticket = 99  # type: ignore[misc]
