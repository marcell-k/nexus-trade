"""
Concurrency tests for PositionRepository.

The existing test_repository.py verifies logic paths but uses MagicMock locks,
which never actually serialize access.  These tests use real threading.Lock
objects to verify that:

  1. Concurrent readers always receive a consistent, non-partial cache view.
  2. In-place dict mutations by a writer never produce partial reads under
     contention (the lock prevents interleaved dict iteration + clear/refill).
  3. Staleness detection is consistent across all racing threads.
  4. is_cache_fresh() and cache_age_seconds() reflect the underlying timestamp
     correctly.
"""

from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from unittest.mock import MagicMock

    from nexus_trade.core.protocols import ProcessLock
    from nexus_trade.core.state import SharedState


from nexus_trade.core.repository import PositionRepository
from nexus_trade.core.types import PositionCacheEntry

# ── helpers ───────────────────────────────────────────────────────────────────


def _entry(ticket: int, magic: int = 100, symbol: str = "EURUSD") -> PositionCacheEntry:
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


def _state(
    entries: dict[int, PositionCacheEntry],
    age_seconds: float = 0.0,
) -> SharedState:
    return cast(
        "SharedState",
        {
            "position_cache": dict(entries),
            "position_cache_timestamp": time.time() - age_seconds,
        },
    )


def _repo(
    state: SharedState,
    lock: threading.Lock,
    ttl: int = 60,
) -> PositionRepository:
    return PositionRepository(state, cast("ProcessLock", lock), cache_staleness_threshold=ttl)


# ── concurrent reads ──────────────────────────────────────────────────────────


class TestConcurrentReads:
    def test_twenty_readers_see_identical_results(self) -> None:
        """All 20 simultaneously-started threads read the same 2-entry cache."""
        lock = threading.Lock()
        state = _state({1: _entry(1), 2: _entry(2)})
        repo = _repo(state, lock)

        results: list[list[PositionCacheEntry] | None] = []
        results_lock = threading.Lock()

        def _read() -> None:
            r = repo.get_strategy_positions("EURUSD", 100, prefer_cache=True)
            with results_lock:
                results.append(r)

        threads = [threading.Thread(target=_read) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 20
        assert all(r is not None and len(r) == 2 for r in results)

    def test_no_partial_reads_under_concurrent_in_place_writes(self, mt5_mock: MagicMock) -> None:
        """
        Writer clears the shared dict then repopulates it with N entries, all
        inside the same lock acquisition.  Readers protected by the same lock
        must never observe the intermediate empty or half-filled state.

        Without the lock in _read_cache this test regularly detects 0 < k < N.
        """
        N = 4
        lock = threading.Lock()
        initial = {i: _entry(i) for i in range(N)}
        state = _state(initial)
        repo = _repo(state, lock, ttl=3600)

        partial_reads: list[str] = []
        stop = threading.Event()
        err_lock = threading.Lock()
        batch_counter = [0]

        def _writer() -> None:
            while not stop.is_set():
                batch_counter[0] += 1
                b = batch_counter[0]
                with lock:
                    # In-place mutation: clear then refill — not atomic without lock
                    state["position_cache"].clear()
                    for i in range(N):
                        state["position_cache"][b * N + i] = _entry(b * N + i)
                    state["position_cache_timestamp"] = time.time()
                time.sleep(0)

        def _reader() -> None:
            while not stop.is_set():
                r = repo.get_strategy_positions("EURUSD", 100, prefer_cache=True)
                # r==None → stale fall-through (acceptable); r==[] → MT5 empty (acceptable)
                # Only 0 < len(r) < N would indicate a partial read
                if r is not None and 0 < len(r) < N:
                    with err_lock:
                        partial_reads.append(f"len={len(r)}")
                time.sleep(0)

        writer = threading.Thread(target=_writer, daemon=True)
        readers = [threading.Thread(target=_reader, daemon=True) for _ in range(4)]
        for t in [writer, *readers]:
            t.start()

        time.sleep(0.35)
        stop.set()

        for t in [writer, *readers]:
            t.join(timeout=1.0)

        assert not partial_reads, f"Partial reads observed: {partial_reads[:5]}"


# ── staleness detection under load ───────────────────────────────────────────


class TestStalenessUnderLoad:
    def test_all_threads_agree_cache_is_stale(self, mt5_mock: MagicMock) -> None:
        """10 concurrent readers with a 120 s-old cache (threshold 60 s) all fall
        through to MT5 and receive an empty list.
        """
        lock = threading.Lock()
        state = _state({1: _entry(1)}, age_seconds=120.0)  # always stale
        repo = _repo(state, lock, ttl=60)

        mt5_mock.positions_get.return_value = ()  # MT5 returns no positions

        results: list[list[PositionCacheEntry] | None] = []
        results_lock = threading.Lock()

        def _check() -> None:
            r = repo.get_strategy_positions("EURUSD", 100, prefer_cache=True)
            with results_lock:
                results.append(r)

        threads = [threading.Thread(target=_check) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 10
        # Stale → fall-through to _query_mt5_positions → MT5 empty → []
        assert all(r == [] for r in results)

    def test_cache_age_increases_monotonically(self) -> None:
        """Successive calls to cache_age_seconds() never return a smaller value."""
        lock = threading.Lock()
        state = _state({}, age_seconds=5.0)
        repo = _repo(state, lock)

        ages = [repo.cache_age_seconds() for _ in range(10)]
        assert all(ages[i] <= ages[i + 1] for i in range(len(ages) - 1))


# ── is_cache_fresh correctness ────────────────────────────────────────────────


class TestCacheFreshness:
    def test_freshly_written_state_is_fresh(self) -> None:
        lock = threading.Lock()
        state = _state({}, age_seconds=0.0)
        repo = _repo(state, lock, ttl=60)
        assert repo.is_cache_fresh() is True

    def test_stale_state_is_not_fresh(self) -> None:
        lock = threading.Lock()
        state = _state({}, age_seconds=120.0)
        repo = _repo(state, lock, ttl=60)
        assert repo.is_cache_fresh() is False

    def test_age_boundary_respected(self) -> None:
        lock = threading.Lock()
        # age = 59 s < ttl 60 s → fresh
        state = _state({}, age_seconds=59.0)
        repo = _repo(state, lock, ttl=60)
        assert repo.is_cache_fresh() is True

    def test_cache_age_reflects_timestamp_approximately(self) -> None:
        lock = threading.Lock()
        state = _state({}, age_seconds=30.0)
        repo = _repo(state, lock)
        age = repo.cache_age_seconds()
        assert 29.0 <= age <= 31.5  # 1.5 s tolerance for test-runner jitter

    def test_magic_filter_applied_before_returning(self) -> None:
        """Entries with a different magic number are never returned."""
        lock = threading.Lock()
        state = _state(
            {
                1: _entry(1, magic=100),
                2: _entry(2, magic=200),
            }
        )
        repo = _repo(state, lock)
        result = repo.get_strategy_positions("EURUSD", magic=100, prefer_cache=True)
        assert result is not None
        assert len(result) == 1
        assert result[0]["magic"] == 100
