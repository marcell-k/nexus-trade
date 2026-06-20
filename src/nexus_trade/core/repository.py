"""PositionRepository — single source of truth for MT5 position and order reads."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

import MetaTrader5 as mt

from nexus_trade.core.models import NormalizedPosition, normalize_order

if TYPE_CHECKING:
    from nexus_trade.core.protocols import ProcessLock
    from nexus_trade.core.state import SharedState
    from nexus_trade.core.types import OrderSnapshot, PositionCacheEntry

logger = logging.getLogger(__name__)

_DEFAULT_CACHE_STALENESS_THRESHOLD: int = 60


class PositionRepository:
    """Thread- and process-safe position/order access layer."""

    __slots__: tuple[str, ...] = (
        "_cache_staleness_threshold",
        "_position_cache_lock",
        "_shared_state",
    )

    def __init__(
        self,
        shared_state: SharedState,
        position_cache_lock: ProcessLock,
        cache_staleness_threshold: int = _DEFAULT_CACHE_STALENESS_THRESHOLD,
    ) -> None:
        self._shared_state: SharedState = shared_state
        self._position_cache_lock: ProcessLock = position_cache_lock
        self._cache_staleness_threshold: int = cache_staleness_threshold

    def get_strategy_positions(
        self,
        symbol: str,
        magic: int,
        *,
        known_tickets: frozenset[int] | None = None,
        prefer_cache: bool = True,
    ) -> list[PositionCacheEntry] | None:
        """Return positions matching *symbol* + *magic*."""
        if prefer_cache:
            cached = self._read_cache(symbol=symbol, magic=magic)
            if cached is not None:
                if known_tickets is not None:
                    return [p for p in cached if p["ticket"] in known_tickets]
                return cached
            # Cache stale — fall through.

        return self._query_mt5_positions(symbol=symbol, magic=magic)

    def get_strategy_orders(
        self,
        symbol: str,
        magic: int,
    ) -> list[OrderSnapshot] | None:
        """Return pending orders for *symbol* + *magic* via direct MT5 query."""
        raw = mt.orders_get(symbol=symbol)
        if raw is None:
            logger.error(f"OrdGetFail sym={symbol} | err={mt.last_error()}")
            return None
        if not raw:
            return []
        return [
            normalized
            for order in raw
            if (normalized := normalize_order(order)).magic == magic and normalized.symbol == symbol
        ]

    def get_managed_positions(
        self,
        magic_numbers: frozenset[int],
    ) -> list[PositionCacheEntry] | None:
        """Return all open positions whose magic number is in *magic_numbers*."""
        raw = mt.positions_get()
        if raw is None:
            logger.error(f"PosGetFail op=managed_positions | err={mt.last_error()}")
            return None
        return [
            normalized
            for pos in raw
            if (normalized := NormalizedPosition.from_mt5(pos).to_cache_entry())["magic_number"] in magic_numbers
        ]

    def get_managed_orders(
        self,
        magic_numbers: frozenset[int],
    ) -> list[OrderSnapshot] | None:
        """Return all pending orders whose magic number is in *magic_numbers*."""
        raw = mt.orders_get()
        if raw is None:
            logger.error(f"OrdGetFail op=managed_orders | err={mt.last_error()}")
            return None
        return [normalized for order in raw if (normalized := normalize_order(order)).magic in magic_numbers]

    def get_position_by_ticket(self, ticket: int) -> PositionCacheEntry | None:
        """Return the open position for *ticket*, or ``None`` if not found or on error."""
        raw = mt.positions_get(ticket=ticket)
        if raw is None:
            logger.error(f"PosGetFail ticket={ticket} | err={mt.last_error()}")
            return None
        if not raw:
            return None
        return NormalizedPosition.from_mt5(raw[0]).to_cache_entry()

    def get_positions_by_tickets(
        self,
        tickets: list[int],
    ) -> dict[int, PositionCacheEntry] | None:
        """Bulk-fetch positions for *tickets*."""
        if not tickets:
            return {}
        if len(tickets) == 1:
            raw = mt.positions_get(ticket=tickets[0])
            if raw is None:
                return None
            return {tickets[0]: NormalizedPosition.from_mt5(raw[0]).to_cache_entry()} if raw else {}
        raw_all = mt.positions_get()
        if raw_all is None:
            return None
        ticket_set = frozenset(tickets)
        return {
            pos.ticket: NormalizedPosition.from_mt5(pos).to_cache_entry() for pos in raw_all if pos.ticket in ticket_set
        }

    def _read_cache(
        self,
        symbol: str,
        magic: int,
    ) -> list[PositionCacheEntry] | None:
        """Return cache entries for *symbol* + *magic* if the cache is fresh."""
        try:
            with self._position_cache_lock:
                if "position_cache" not in self._shared_state:
                    return []
                cache_ts: float = float(self._shared_state.get("position_cache_timestamp", 0.0) or 0.0)
                age = time.time() - cache_ts

                if age > self._cache_staleness_threshold:
                    logger.debug(f"CacheStale age={age:.1f}s | threshold={self._cache_staleness_threshold}s")
                    return None

                cache_ref: dict[int, PositionCacheEntry] = self._shared_state.get("position_cache", {})
                result = [pos for pos in cache_ref.values() if pos["magic_number"] == magic and pos["symbol"] == symbol]

            logger.debug(f"CacheHit sym={symbol} | m={magic} | n={len(result)} | age={age:.1f}s")
            return result

        except Exception as exc:
            logger.warning(f"CacheReadFail sym={symbol} | m={magic} | err={exc}")
            return None

    def _query_mt5_positions(
        self,
        symbol: str,
        magic: int,
    ) -> list[PositionCacheEntry] | None:
        """Direct MT5 query, scoped to *symbol* + *magic*."""
        raw = mt.positions_get(symbol=symbol)
        if raw is None:
            logger.error(f"PosGetFail sym={symbol} | err={mt.last_error()}")
            return None
        if not raw:
            return []
        return [
            normalized
            for pos in raw
            if (normalized := NormalizedPosition.from_mt5(pos).to_cache_entry())["magic_number"] == magic
        ]

    def cache_age_seconds(self) -> float:
        """Return seconds since the shared position cache was last written."""
        ts: float = float(self._shared_state.get("position_cache_timestamp", 0.0) or 0.0)
        return time.time() - ts

    def is_cache_fresh(self) -> bool:
        """Return True if the cache age is within the configured staleness threshold."""
        return self.cache_age_seconds() <= self._cache_staleness_threshold
