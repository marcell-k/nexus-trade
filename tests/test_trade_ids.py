"""Unit tests for TradeIDSequenceManager — real SQLite, no MT5 dependency."""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

import pytest

from nexus_trade.execution.trade_ids import TradeIDSequenceManager

if TYPE_CHECKING:
    from collections.abc import Generator
    from pathlib import Path


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "trade_ids.db"


@pytest.fixture
def manager(db_path: Path) -> Generator[TradeIDSequenceManager]:
    mgr = TradeIDSequenceManager(db_path)
    yield mgr
    mgr.close()


#  Basic sequence


class TestGenerateId:
    def test_first_id_is_one(self, manager: TradeIDSequenceManager) -> None:
        assert manager.generate_id() == 1

    def test_sequential_increment(self, manager: TradeIDSequenceManager) -> None:
        ids = [manager.generate_id() for _ in range(5)]
        assert ids == [1, 2, 3, 4, 5]

    def test_persists_across_instances(self, db_path: Path) -> None:
        mgr1 = TradeIDSequenceManager(db_path)
        mgr1.generate_id()
        mgr1.generate_id()
        mgr1.close()

        mgr2 = TradeIDSequenceManager(db_path)
        assert mgr2.generate_id() == 3
        mgr2.close()


#  Batch generation


class TestGenerateBatch:
    def test_batch_sequential(self, manager: TradeIDSequenceManager) -> None:
        ids = manager.generate_batch(4)
        assert ids == [1, 2, 3, 4]

    def test_batch_then_single_continues(self, manager: TradeIDSequenceManager) -> None:
        manager.generate_batch(3)
        next_id = manager.generate_id()
        assert next_id == 4

    def test_batch_no_gaps(self, manager: TradeIDSequenceManager) -> None:
        a = manager.generate_batch(3)
        b = manager.generate_batch(3)
        assert b[0] == a[-1] + 1


#  get_current_id


class TestGetCurrentId:
    def test_reflects_generate_calls(self, manager: TradeIDSequenceManager) -> None:
        manager.generate_id()
        manager.generate_id()
        assert manager.get_current_id() == 2

    def test_does_not_increment(self, manager: TradeIDSequenceManager) -> None:
        manager.get_current_id()
        manager.get_current_id()
        assert manager.get_current_id() == 0


#  reset


class TestReset:
    def test_reset_to_zero(self, manager: TradeIDSequenceManager) -> None:
        manager.generate_id()
        manager.reset(0)
        assert manager.get_current_id() == 0
        assert manager.generate_id() == 1

    def test_negative_value_raises(self, manager: TradeIDSequenceManager) -> None:
        with pytest.raises(ValueError):
            manager.reset(-1)


#  Context manager


class TestContextManager:
    def test_context_manager_closes(self, db_path: Path) -> None:
        with TradeIDSequenceManager(db_path) as mgr:
            mgr.generate_id()
        assert mgr._conn is None


#  Concurrency


class TestConcurrentGeneration:
    def test_no_duplicate_ids_under_contention(self, db_path: Path) -> None:
        """10 threads each generate 10 IDs — all 100 must be unique."""
        results: list[int] = []
        lock = threading.Lock()
        errors: list[Exception] = []

        def worker() -> None:
            mgr = TradeIDSequenceManager(db_path)
            try:
                for _ in range(10):
                    id_ = mgr.generate_id()
                    with lock:
                        results.append(id_)
            except Exception as exc:
                with lock:
                    errors.append(exc)
            finally:
                mgr.close()

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Concurrent errors: {errors}"
        assert len(results) == 100
        assert len(set(results)) == 100, "Duplicate IDs generated"
        assert sorted(results) == list(range(1, 101))
