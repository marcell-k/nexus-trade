"""StrategyRunner entrypoint with a unified, explicit internal architecture."""

from __future__ import annotations

import importlib
import logging
import os
import sqlite3
import threading
import time
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

import MetaTrader5 as mt
import numpy as np

from nexus_trade.config.timings import SYSTEM_TIMINGS
from nexus_trade.core.connection import MT5Connection
from nexus_trade.core.constants import TIMEFRAME_STRING_MAP, TIMEFRAME_TO_MINUTES, TimeFrame
from nexus_trade.core.data_handler import DataHandler
from nexus_trade.core.models import (
    BracketPendingTicket,
    ExitLogData,
    NormalizedPosition,
    PendingTicket,
    Position,
    StandardPendingTicket,
    cache_entry_to_position,
)
from nexus_trade.core.repository import PositionRepository
from nexus_trade.core.types import (
    EntryMetadata,
    GlobalRiskPolicy,
    MT5Tick,
    OrderSnapshot,
    PositionCacheEntry,
    PositionType,
)
from nexus_trade.execution.executor import OrderExecutor
from nexus_trade.execution.request import EntryRequest, ExitRequest, ModifyRequest
from nexus_trade.execution.trade_ids import TradeIDSequenceManager
from nexus_trade.logging.async_logger import AsyncTradeLogger
from nexus_trade.logging.logger import CloseData, FillData, PartialCloseData, TradeLogger
from nexus_trade.risk.manager import RiskManager
from nexus_trade.utils.format import format_price_display

if TYPE_CHECKING:
    from collections.abc import Callable
    from zoneinfo import ZoneInfo

    import pandas as pd

    from nexus_trade.config.account import AccountConfig
    from nexus_trade.config.profile import MetaLabelingCfg
    from nexus_trade.config.strategy import BaseStrategyParams, StrategyConfig, StrategyOrderType
    from nexus_trade.core.protocols import AtomicInt, ProcessLock, StrategyProtocol, XGBClassifierProtocol
    from nexus_trade.core.state import SharedState
    from nexus_trade.tools.calibrator import ProbabilityCalibrator


logger = logging.getLogger(__name__)

_SL_TP_SNAP_TOLERANCE: float = 0.001
_BREAKEVEN_HALF_BAND_RATIO: float = 0.0005
_TRIGGER_TP: str = "TP"
_TRIGGER_SL: str = "SL"
_TRIGGER_BREAKEVEN: str = "BREAKEVEN"
_TRIGGER_SIGNAL: str = "SIGNAL"
_CACHE_STALENESS_THRESHOLD: int = SYSTEM_TIMINGS.cache_staleness_threshold
_BRACKET_EXPIRY_GRACE_SECONDS: int = 30


@dataclass
class RunnerConfig:
    """Consolidated configuration for ``StrategyRunner`` initialization."""

    strategy_name: str
    strategy_config: StrategyConfig[BaseStrategyParams]
    broker_config: AccountConfig
    global_risk_policy: GlobalRiskPolicy
    shared_state: SharedState
    global_trade_count: AtomicInt
    global_position_count: AtomicInt
    position_cache_lock: ProcessLock
    trade_id_db_path: Path
    meta_labeling: MetaLabelingCfg
    strategy_offset_seconds: float = 0.0


class StrategyRunner:
    """Unified strategy runner handling lifecycle, state, entry, and exits."""

    def __init__(self, config: RunnerConfig) -> None:
        self.strategy_name: str = config.strategy_name
        self.config: StrategyConfig[BaseStrategyParams] = config.strategy_config
        self.broker_config: AccountConfig = config.broker_config
        self.global_risk_policy: GlobalRiskPolicy = config.global_risk_policy
        self.shared_state: SharedState = config.shared_state
        self.global_trade_count: AtomicInt = config.global_trade_count
        self.global_position_count: AtomicInt = config.global_position_count
        self.position_cache_lock: ProcessLock = config.position_cache_lock
        self.trade_id_db_path: Path = config.trade_id_db_path
        self._meta_labeling: MetaLabelingCfg = config.meta_labeling
        self.strategy_offset_seconds: float = config.strategy_offset_seconds

        self.connection: MT5Connection
        self.data_handler: DataHandler
        self.trade_id_manager: TradeIDSequenceManager
        self.executor: OrderExecutor
        self.risk_manager: RiskManager

        self.strategy: StrategyProtocol | None = None

        self.position_repo: PositionRepository = PositionRepository(
            shared_state=self.shared_state,
            position_cache_lock=self.position_cache_lock,
        )

        params = self.config.params
        self.symbol: str = params.symbol
        self.timeframe: str = params.timeframe
        self.timeframe_mt5: TimeFrame = TIMEFRAME_STRING_MAP[self.timeframe.upper()]
        self.magic_number: int = self.config.execution.magic_number
        self.order_type: StrategyOrderType = self.config.order_type
        _th = self.config.trading_hours
        self.strategy_tz: str = (_th.timezone if _th else None) or params.timezone
        self.broker_tz: ZoneInfo = self.broker_config.broker_tz

        sync_logger = TradeLogger(
            log_root=Path(self.global_risk_policy["log_root"]) / "trades",
            strategy_name=self.strategy_name,
            strategy_tz=self.strategy_tz,
        )
        self.trade_logger: AsyncTradeLogger = AsyncTradeLogger(trade_logger=sync_logger, max_queue_size=100)
        logger.debug(f"AsyncLog strat={self.strategy_name} | q=100")

        self.entry_metadata: dict[int, EntryMetadata] = {}
        self.ticket_to_trade_id: dict[int, int] = {}
        self.pending_tickets: dict[int, PendingTicket] = {}
        self.pending_by_key: dict[tuple[str, int], list[int]] = {}
        self.pending_by_ticket: dict[int, int] = {}
        self.known_positions: set[int] = set()
        self.position_state_lock: threading.Lock = threading.Lock()

        self.meta_model: XGBClassifierProtocol | None = None
        self.calibration_model: ProbabilityCalibrator | None = None
        self.feature_extractor: Callable[[pd.DataFrame], pd.DataFrame] | None = None
        self.meta_min_confidence: float = 0.0

        self.timeframe_minutes: int = TIMEFRAME_TO_MINUTES[self.timeframe_mt5]
        self.last_processed_bar_time: pd.Timestamp | None = None
        self.local_position_count: int = 0
        self.next_entry_time: datetime | None = None
        self.next_exit_time: datetime | None = None

        self._symbol_tick_cache: dict[str, MT5Tick] = {}
        self._cleanup_done: bool = False

        self.exit_log_data_cls: type[ExitLogData] = ExitLogData

        logger.info(
            f"Init strat={self.strategy_name:<9} | tf={self.timeframe:<3} | "
            f"sym={self.symbol:<7} | m={self.magic_number:>3}"
        )

    def setup(self) -> None:
        """Initialize MT5 connection, data handler, executor, risk manager, and strategy."""
        logger.debug(f"SetupStart strat={self.strategy_name} | pid={os.getpid()}")

        self.trade_id_manager = TradeIDSequenceManager(self.trade_id_db_path)

        self.connection = MT5Connection(self.broker_config)
        if not self.connection.connect():
            raise RuntimeError(f"SetupFail strat={self.strategy_name} | step=mt5_connect")

        strategy_class = self._load_strategy_class()
        self.strategy = cast("StrategyProtocol", strategy_class(params=self.config.params))

        self.data_handler = DataHandler(self.broker_tz)
        self.executor = OrderExecutor(self.broker_tz)
        self.risk_manager = RiskManager(
            strategy_config=self.config,
            global_policy=self.global_risk_policy,
            shared_state=self.shared_state,
            global_trade_count=self.global_trade_count,
            global_position_count=self.global_position_count,
            data_handler=self.data_handler,
            broker_tz=self.broker_config.broker_tz,
            strategy_runner=self,
        )

        self._load_meta_models()

        if "heartbeats" not in self.shared_state:
            self.shared_state["heartbeats"] = {}
        self._update_heartbeat()

    def run(self) -> None:
        """Run the main event loop — timeframe-aligned entry + 1-minute exit monitoring."""
        try:
            try:
                self.setup()
            except RuntimeError:
                return

            startup_positions = self._collect_startup_positions_snapshot()
            self._init_known_positions(startup_positions)

            logger.debug(f"LoopStart strat={self.strategy_name} | entry={self.timeframe_minutes}m | exit=1m")

            self.next_entry_time, self.next_exit_time = self._initialize_schedule()

            while not self.shared_state.get("shutdown_flag", False):
                now = datetime.now(self.broker_tz)
                tolerance = timedelta(seconds=5)

                sleep_sec: float = self._seconds_until_next_event(now)
                if sleep_sec > 0:
                    time.sleep(sleep_sec)
                    now = datetime.now(self.broker_tz)

                data = self._fetch_data()
                if data is None:
                    logger.error(f"DataFetchFail strat={self.strategy_name} | retry=5s")
                    time.sleep(5.0)
                    continue

                self._symbol_tick_cache.clear()

                process_entry = self.next_entry_time and now >= self.next_entry_time - tolerance
                process_exit = now >= self.next_exit_time - tolerance

                self._update_heartbeat()
                preloaded_positions: list[PositionCacheEntry] | None = None

                if process_entry:
                    if self._should_check_entry_signal(data):
                        self._process_entry_signal(data)
                    preloaded_positions = self._load_strategy_positions(mode="cache")
                    self._process_modify_signals(data, preloaded_positions)
                    self.next_entry_time = self._calculate_next_entry_time(now)

                if process_exit:
                    if self.order_type in ("market", "limit", "stop", "bracket"):
                        self._monitor_exits(data, preloaded_positions)
                    self.next_exit_time = self._calculate_next_exit_time(now)

        except KeyboardInterrupt:
            logger.debug(f"RunStop strat={self.strategy_name} | reason=keyboard_interrupt")
        except Exception:
            logger.exception(f"RunCrash strat={self.strategy_name}")
            raise
        finally:
            self.cleanup()

    def cleanup(self) -> None:
        """Close positions, cancel orders, drain logger, disconnect MT5."""
        if self._cleanup_done:
            return
        self._cleanup_done = True

        logger.debug(f"CleanupStart strat={self.strategy_name}")

        self._reconcile_shutdown_state()

        if self.trade_logger and hasattr(self.trade_logger, "shutdown"):
            logger.debug(f"CleanupTradeLog strat={self.strategy_name} | action=drain")
            self.trade_logger.shutdown(timeout=10.0)

        self.trade_id_manager.close()
        self.connection.disconnect()

        logger.debug(f"CleanupDone strat={self.strategy_name}")

    def _load_strategy_class(self) -> type:
        module = importlib.import_module(self.config.strategy_module)
        return getattr(module, self.config.strategy_class)

    def _load_meta_models(self) -> None:
        from nexus_trade.filters.meta import (
            load_calibration_model,
            load_features_extractor,
            load_meta_model,
        )

        cfg = self._meta_labeling
        if not cfg.enabled:
            self.meta_model = None
            self.calibration_model = None
            self.feature_extractor = None
            self.meta_min_confidence = 0.0
            logger.debug(f"MetaCfg strat={self.strategy_name} | enabled=0")
            return

        self.meta_model = load_meta_model(cfg, self.strategy_name)
        self.calibration_model = load_calibration_model(cfg, self.strategy_name)
        self.feature_extractor = load_features_extractor(cfg, self.strategy_name)
        self.meta_min_confidence = cfg.min_confidence
        if self.meta_model is None:
            logger.warning(f"MetaCfgWarn strat={self.strategy_name} | enabled=1 | model=missing")

    def _update_heartbeat(self) -> None:
        self.shared_state["heartbeats"][self.strategy_name] = datetime.now().timestamp()

    def _fetch_data(self) -> pd.DataFrame | None:
        return self.data_handler.get_latest_bars(strategy_name=self.strategy_name)

    def _generate_trade_id(self) -> int:
        return self.trade_id_manager.generate_id()

    def _seconds_until_next_event(self, now: datetime) -> float:
        if not self.next_entry_time and not self.next_exit_time:
            self.next_entry_time, self.next_exit_time = self._initialize_schedule()
            return 1.0
        candidates = [t for t in (self.next_entry_time, self.next_exit_time) if t]
        return (min(candidates) - now).total_seconds()

    def _get_composite_key(self) -> tuple[str, int]:
        return (self.symbol, self.magic_number)

    def _refresh_tracked_position_snapshots(self, positions: list[PositionCacheEntry]) -> None:
        if not positions:
            return
        with self.position_state_lock:
            for pos in positions:
                ticket = pos["ticket"]
                trade_id = self.ticket_to_trade_id.get(ticket)
                if trade_id is None:
                    continue
                metadata = self.entry_metadata.get(trade_id)
                if metadata is None:
                    continue
                metadata["position_snapshot"] = pos

    def _patch_tracked_snapshot_levels(self, ticket: int, new_sl: float | None, new_tp: float | None) -> None:
        if new_sl is None and new_tp is None:
            return
        with self.position_state_lock:
            trade_id = self.ticket_to_trade_id.get(ticket)
            if trade_id is None:
                return
            metadata = self.entry_metadata.get(trade_id)
            if metadata is None:
                return
            snapshot = metadata.get("position_snapshot")
            if snapshot is None:
                return
            if new_sl is not None:
                snapshot["sl"] = new_sl
            if new_tp is not None:
                snapshot["tp"] = new_tp

    def _has_pending_bracket_orders(self) -> bool:
        with self.position_state_lock:
            return any(isinstance(pending, BracketPendingTicket) for pending in self.pending_tickets.values())

    def _seed_startup_position_tracking(
        self,
        trade_id: int,
        pos: Position,
        submission_time: float,
        expected_entry_price: float | None = None,
        opening_sl: float | None = None,
        volume_multiplier: float | None = None,
    ) -> None:
        ticket = pos.ticket
        self.known_positions.add(ticket)
        self.entry_metadata[trade_id] = {
            "submission_time": submission_time,
            "volume_multiplier": volume_multiplier,
            "ticket": ticket,
            "opening_sl": opening_sl if opening_sl is not None else pos.sl,
            "position_snapshot": pos.to_cache_entry(),
            "expected_entry_price": expected_entry_price if expected_entry_price is not None else pos.price_open,
            "entry_request": None,
        }
        self.ticket_to_trade_id[ticket] = trade_id

    def _init_known_positions(self, positions: list[Position] | None = None) -> None:
        if positions is None:
            positions = self._collect_startup_positions_snapshot()
        if positions is None:
            raise RuntimeError(f"{self.strategy_name}: Cannot initialize — MT5 API failure")
        if not positions:
            self.local_position_count = 0
            logger.debug(f"{self.strategy_name:<9}: InitPos n=0")
            return

        position_count = len(positions)
        tracked_tickets = [pos.ticket for pos in positions]
        current_time = time.time()

        try:
            reconciled_map = self.trade_logger.get_open_trades_by_ticket_last_three(tracked_tickets)
        except (sqlite3.Error, RuntimeError):
            logger.error(f"{self.strategy_name:<9}: Startup reconciliation failed", exc_info=True)
            reconciled_map = {}

        reconciled_count = 0
        missing_positions: list[Position] = []

        for pos in positions:
            reconciled = reconciled_map.get(pos.ticket)
            if reconciled is None:
                missing_positions.append(pos)
                continue
            self._seed_startup_position_tracking(
                trade_id=reconciled["trade_id"],
                pos=pos,
                submission_time=current_time,
                expected_entry_price=reconciled.get("expected_entry_price"),
                opening_sl=reconciled.get("opening_sl"),
                volume_multiplier=reconciled.get("volume_multiplier"),
            )
            reconciled_count += 1

        new_trade_ids: list[int] = []
        if missing_positions:
            new_trade_ids = self.trade_id_manager.generate_batch(len(missing_positions))
            for pos, trade_id in zip(missing_positions, new_trade_ids, strict=True):
                self._seed_startup_position_tracking(trade_id=trade_id, pos=pos, submission_time=current_time)
                metadata = self.entry_metadata[trade_id]
                position_obj = pos.to_cache_entry()
                fill_data = self._build_fill_data(
                    trade_id=trade_id,
                    position=position_obj,
                    expected_entry_price=metadata.get("expected_entry_price", 0.0),
                    opening_sl=metadata.get("opening_sl"),
                    fill_time_ms=None,
                    volume_multiplier=metadata.get("volume_multiplier"),
                )
                self.trade_logger.log_fill(fill_data)

        self.local_position_count = position_count
        ticket_str = (
            ", ".join(map(str, tracked_tickets))
            if position_count <= 5
            else f"{tracked_tickets[0]}, {tracked_tickets[1]}, … ({position_count} total)"
        )
        new_ids_str = f"{new_trade_ids[0]}..{new_trade_ids[-1]}" if new_trade_ids else "none"
        logger.info(
            f"{self.strategy_name:<9}: InitPos n={position_count} | rec={reconciled_count} | "
            f"new={len(missing_positions)} | ids={new_ids_str} | t=[{ticket_str}]"
        )

    def _collect_startup_positions_snapshot(self) -> list[Position] | None:
        entries = self._load_strategy_positions(mode="cache", include_unknown=True)
        if entries is None:
            return None
        return [cache_entry_to_position(e) for e in entries]

    def _load_strategy_positions(
        self,
        mode: Literal["cache", "direct"] = "cache",
        include_unknown: bool = False,
    ) -> list[PositionCacheEntry] | None:
        repo: PositionRepository = self.position_repo

        if mode == "cache":
            known: frozenset[int] | None = None if include_unknown else frozenset(self.known_positions)
            positions = repo.get_strategy_positions(
                symbol=self.symbol,
                magic=self.magic_number,
                known_tickets=known,
                prefer_cache=True,
            )
            if positions is not None:
                return positions

        positions = repo.get_strategy_positions(
            symbol=self.symbol,
            magic=self.magic_number,
            prefer_cache=False,
        )
        if positions is None:
            logger.error(f"{self.strategy_name:<9}: PosGetFail reason=mt5_api_failure")
            return None
        if include_unknown:
            return positions
        with self.position_state_lock:
            known_snap = frozenset(self.known_positions)
        return [p for p in positions if p["ticket"] in known_snap]

    def _get_strategy_orders_direct(self) -> list[OrderSnapshot] | None:
        repo: PositionRepository = self.position_repo
        orders = repo.get_strategy_orders(symbol=self.symbol, magic=self.magic_number)
        if orders is None:
            logger.error(f"{self.strategy_name:<9}: OrdGetFail reason=mt5_api_failure")
        return orders

    def _requery_position(self, ticket: int) -> Position | None:
        repo: PositionRepository = self.position_repo
        entry = repo.get_position_by_ticket(ticket)
        if entry is None:
            return None
        return cache_entry_to_position(entry)

    def _invalidate_cache_for_ticket(self, ticket: int) -> None:
        logger.debug(f"{self.strategy_name:<9}: CacheInvalidate t={ticket} | refreshes on next heartbeat")

    def _resolve_entry_prices(self, ticket: int, pos: Position) -> tuple[int | None, float | None, float | None, float]:
        trade_id = self.ticket_to_trade_id.get(ticket)
        if trade_id is None:
            logger.error(f"{self.strategy_name:<9}: OrphanPos t={ticket} | reason=no_metadata")
            return None, pos.price_open, pos.sl, pos.price_open

        metadata = self.entry_metadata[trade_id]
        entry_request = metadata.get("entry_request")
        entry_price = pos.price_open

        if entry_request is None:
            return (
                trade_id,
                metadata.get("expected_entry_price", entry_price),
                metadata.get("opening_sl"),
                entry_price,
            )
        if entry_request.order_type == "bracket":
            is_buy = pos.type == PositionType.BUY
            expected_entry_price = entry_request.buy_stop if is_buy else entry_request.sell_stop
            opening_sl = entry_request.buy_sl if is_buy else entry_request.sell_sl
        else:
            expected_entry_price = entry_request.entry_price or entry_price
            opening_sl = entry_request.sl

        return trade_id, expected_entry_price, opening_sl, entry_price

    def _register_pending_trade(self, trade_id: int, pending_ticket: PendingTicket) -> None:
        self.pending_tickets[trade_id] = pending_ticket
        self.pending_by_key.setdefault(self._get_composite_key(), []).append(trade_id)
        if isinstance(pending_ticket, StandardPendingTicket):
            self.pending_by_ticket[pending_ticket.ticket] = trade_id

    def _store_entry_metadata_bracket(
        self,
        trade_id: int,
        entry_request: EntryRequest,
        volume_multiplier: float | None,
        submission_time: float,
        buy_ticket: int,
        sell_ticket: int,
    ) -> None:
        buy_stop = entry_request.buy_stop
        sell_stop = entry_request.sell_stop
        if buy_stop is None or sell_stop is None:
            raise ValueError(
                f"Bracket entry missing buy_stop or sell_stop: buy_stop={buy_stop!r}, sell_stop={sell_stop!r}"
            )
        metadata: EntryMetadata = {
            "submission_time": submission_time,
            "volume_multiplier": volume_multiplier,
            "ticket": None,
            "opening_sl": None,
            "position_snapshot": None,
            "entry_request": entry_request,
            "expected_entry_price": 0.0,
        }
        with self.position_state_lock:
            self.ticket_to_trade_id.update({buy_ticket: trade_id, sell_ticket: trade_id})
            self.entry_metadata[trade_id] = metadata
            self._register_pending_trade(
                trade_id,
                BracketPendingTicket(
                    symbol=self.symbol,
                    magic=self.magic_number,
                    submission_time=submission_time,
                    buy_order_ticket=buy_ticket,
                    sell_order_ticket=sell_ticket,
                    buy_stop=buy_stop,
                    sell_stop=sell_stop,
                    expected_volume=entry_request.volume,
                ),
            )

    def _store_entry_metadata_standard(
        self,
        trade_id: int,
        ticket: int,
        volume_multiplier: float | None,
        submission_time: float,
        expected_entry_price: float,
        opening_sl: float | None,
    ) -> None:
        with self.position_state_lock:
            self.entry_metadata[trade_id] = {
                "submission_time": submission_time,
                "volume_multiplier": volume_multiplier,
                "ticket": ticket,
                "expected_entry_price": expected_entry_price,
                "opening_sl": opening_sl,
                "position_snapshot": None,
                "entry_request": None,
            }
            self.ticket_to_trade_id[ticket] = trade_id
            self._register_pending_trade(
                trade_id,
                StandardPendingTicket(
                    symbol=self.symbol,
                    magic=self.magic_number,
                    submission_time=submission_time,
                    ticket=ticket,
                ),
            )

    def _cleanup_pending(self, trade_id: int, composite_key: tuple[str, int]) -> None:
        with self.position_state_lock:
            pending_ticket = self.pending_tickets.pop(trade_id, None)
            if isinstance(pending_ticket, StandardPendingTicket):
                self.pending_by_ticket.pop(pending_ticket.ticket, None)
            if composite_key in self.pending_by_key:
                with suppress(ValueError):
                    self.pending_by_key[composite_key].remove(trade_id)
                if not self.pending_by_key[composite_key]:
                    del self.pending_by_key[composite_key]

    def _handle_closed_positions(self, closed_tickets: list[int]) -> None:
        close_logs: list[tuple[int, int, PositionCacheEntry, EntryMetadata]] = []
        closed_count = 0

        with self.position_state_lock:
            for ticket in closed_tickets:
                trade_id = self.ticket_to_trade_id.pop(ticket, None)
                metadata = self.entry_metadata.pop(trade_id, None) if trade_id is not None else None
                snapshot = metadata.get("position_snapshot") if metadata is not None else None
                was_known = ticket in self.known_positions
                self.known_positions.discard(ticket)

                if trade_id is not None and metadata is not None and snapshot is not None:
                    close_logs.append((trade_id, ticket, snapshot, metadata))
                elif was_known or trade_id is not None:
                    logger.warning(
                        f"{self.strategy_name}:{self.symbol} | "
                        f"EXTERNAL CLOSE t={ticket} | reason=missing_metadata_or_snapshot"
                    )

                if was_known:
                    closed_count += 1

            self.local_position_count = max(0, self.local_position_count - closed_count)

        for trade_id, ticket, snapshot, metadata in close_logs:
            close_data = self._build_close_data(
                trade_id=trade_id,
                position=snapshot,
                expected_exit_price=None,
                opening_sl=metadata.get("opening_sl"),
                exit_trigger="EXTERNAL_CLOSE",
                entry_price=snapshot["price_open"],
                expected_entry_price=metadata.get("expected_entry_price")
                if metadata.get("expected_entry_price") is not None
                else snapshot["price_open"],
            )
            self.trade_logger.log_close(close_data)
            logger.info(f"{self.strategy_name}: EXTERNAL CLOSE | id={trade_id} | t={ticket}")

        global_count = self._atomic_decrement_global_positions(closed_count, "external_close")

        if closed_count:
            logger.info(
                f"{self.strategy_name:<9}: PosClosedDetect n={len(closed_tickets)} | "
                f"local={self.local_position_count}/{self.global_risk_policy['max_total_positions']} | "
                f"global={global_count}/{self.global_risk_policy['max_total_positions']}"
            )
        elif closed_tickets:
            logger.warning(
                f"{self.strategy_name}:{self.symbol} | "
                f"Detected {len(closed_tickets)} closed tickets but no tracked positions to decrement"
            )

    def _handle_new_fill(self, pos: Position) -> None:
        ticket = pos.ticket
        with self.position_state_lock:
            trade_id = self.ticket_to_trade_id.get(ticket)
            needs_resolution = (
                trade_id is not None and self.entry_metadata.get(trade_id, {}).get("expected_entry_price") is None
            )
            metadata = self.entry_metadata.get(trade_id) if trade_id is not None else None
        if trade_id is None:
            trade_id = self._resolve_pending_ticket(pos)
        elif needs_resolution:
            resolved: int | None = self._resolve_pending_ticket(pos)
            if resolved is not None:
                trade_id = resolved

        if trade_id is None:
            trade_id = self._handle_orphaned_fill(ticket, pos)
            if trade_id is None:
                logger.error(f"{self.strategy_name:<9}: OrphanFillFail t={ticket} | action=skip")
                return

        if metadata is None:
            logger.error(f"{self.strategy_name:<9}: FillMetaMissing id={trade_id} | action=skip")
            return

        with self.position_state_lock:
            self.local_position_count += 1
            self.known_positions.add(ticket)

        new_trades = self.atomic_increment_trade()

        position_snapshot = pos.to_cache_entry()
        metadata["position_snapshot"] = position_snapshot
        metadata["ticket"] = ticket

        submission_time = metadata.get("submission_time")
        fill_latency_ms = (time.time() - submission_time) * 1000 if submission_time else None

        position_obj = pos.to_cache_entry()
        fill_data = self._build_fill_data(
            trade_id=trade_id,
            position=position_obj,
            expected_entry_price=metadata.get("expected_entry_price", 0.0),
            opening_sl=metadata["opening_sl"],
            fill_time_ms=fill_latency_ms,
            volume_multiplier=metadata.get("volume_multiplier"),
        )
        self.trade_logger.log_fill(fill_data)

        logger.info(
            f"{self.strategy_name:<9}: Fill id={trade_id} | t={ticket} | "
            f"{'B' if pos.type == PositionType.BUY else 'S'} {pos.volume:.2f}@"
            f"{format_price_display(pos.price_open)} | "
            f"pos={self.local_position_count}/{self.global_risk_policy['max_total_positions']} | "
            f"tr={new_trades}/{self.global_risk_policy['max_daily_trades']}"
        )

    def _handle_orphaned_fill(self, ticket: int, pos: Position) -> int | None:
        logger.debug(f"{self.strategy_name:<9}: OrphanFill t={ticket} | action=generate_fallback_id")
        trade_id = self._generate_trade_id()
        with self.position_state_lock:
            self.entry_metadata[trade_id] = {
                "submission_time": time.time(),
                "volume_multiplier": None,
                "ticket": ticket,
                "opening_sl": pos.sl,
                "position_snapshot": None,
                "expected_entry_price": pos.price_open,
                "entry_request": None,
            }
            self.ticket_to_trade_id[ticket] = trade_id
        return trade_id

    def _matches_bracket_conditions(
        self, pos: Position, conditions: BracketPendingTicket, submission_time: float
    ) -> bool:
        expected_volume = conditions.expected_volume
        volume_match = abs(pos.volume - expected_volume) / expected_volume < 0.01
        timing_match = time.time() - submission_time < 10.0
        return volume_match and timing_match

    def _resolve_pending_ticket(self, pos: Position) -> int | None:
        ticket = pos.ticket
        with self.position_state_lock:
            composite_key = self._get_composite_key()
            direct_match = self.pending_by_ticket.get(ticket)
            candidate_ids = tuple(self.pending_by_key.get(composite_key, ()))
            pending_snapshot = {tid: self.pending_tickets.get(tid) for tid in candidate_ids}

        if direct_match is not None:
            self._cleanup_pending(direct_match, composite_key)
            logger.debug(f"{self.strategy_name:<9}: StandardResolved id={direct_match}")
            return direct_match

        if not candidate_ids:
            return None

        for pending_trade_id in candidate_ids:
            conditions = pending_snapshot.get(pending_trade_id)
            if conditions is None or conditions.symbol != pos.symbol or conditions.magic != pos.magic:
                continue
            if isinstance(conditions, BracketPendingTicket) and self._matches_bracket_conditions(
                pos, conditions, conditions.submission_time
            ):
                return self._finalize_bracket_resolution(pending_trade_id, pos, conditions, composite_key)
        return None

    def _finalize_bracket_resolution(
        self,
        pending_trade_id: int,
        pos: Position,
        conditions: BracketPendingTicket,
        composite_key: tuple[str, int],
    ) -> int:
        ticket = pos.ticket
        is_buy = pos.type == PositionType.BUY
        with self.position_state_lock:
            self.ticket_to_trade_id[ticket] = pending_trade_id
            metadata = self.entry_metadata[pending_trade_id]
            metadata["ticket"] = ticket
            metadata["expected_entry_price"] = conditions.buy_stop if is_buy else conditions.sell_stop
            metadata["opening_sl"] = pos.sl if pos.sl else None

        opposite_ticket = conditions.sell_order_ticket if is_buy else conditions.buy_order_ticket
        if opposite_ticket and self.executor._cancel_order(opposite_ticket):
            logger.info(f"{self.strategy_name:<9}: BracketOppCancel t={opposite_ticket}")

        self._cleanup_pending(pending_trade_id, composite_key)
        logger.debug(
            f"{self.strategy_name:<9}: BracketResolved id={pending_trade_id} -> t={ticket} | "
            f"side={'BUY' if is_buy else 'SELL'}"
        )
        return pending_trade_id

    def _check_expired_bracket_orders(
        self, current_tickets: set[int], orders: list[OrderSnapshot] | None = None
    ) -> None:
        with self.position_state_lock:
            bracket_pending: dict[int, BracketPendingTicket] = {
                tid: pending
                for tid, pending in self.pending_tickets.items()
                if isinstance(pending, BracketPendingTicket)
            }
        if not bracket_pending:
            return

        if orders is None:
            orders = self._get_strategy_orders_direct()
        order_tickets: set[int] = {o.ticket for o in orders if o.magic == self.magic_number} if orders else set()

        composite_key = self._get_composite_key()

        for trade_id, pending in bracket_pending.items():
            buy_ticket = pending.buy_order_ticket
            sell_ticket = pending.sell_order_ticket

            if (buy_ticket and buy_ticket in current_tickets) or (sell_ticket and sell_ticket in current_tickets):
                continue

            with self.position_state_lock:
                already_resolved = trade_id in self.ticket_to_trade_id.values()
            if already_resolved:
                continue

            if (buy_ticket and buy_ticket in order_tickets) or (sell_ticket and sell_ticket in order_tickets):
                continue

            elapsed = time.time() - pending.submission_time
            if elapsed < _BRACKET_EXPIRY_GRACE_SECONDS:
                continue

            logger.warning(
                f"{self.strategy_name:<9}: BracketExpired id={trade_id} | "
                f"elapsed={elapsed:.0f}s | buy_t={buy_ticket} | sell_t={sell_ticket}"
            )
            self.risk_manager.release_position_reservation(reason="bracket_expired")
            self._cleanup_pending(trade_id, composite_key)
            self.entry_metadata.pop(trade_id, None)

    def atomic_increment_trade(self) -> int:
        with self.global_trade_count.get_lock():
            self.global_trade_count.value += 1
            return self.global_trade_count.value

    def _atomic_decrement_global_positions(self, count: int, reason: str) -> int:
        if count <= 0:
            return self.global_position_count.value

        with self.global_position_count.get_lock():
            current = self.global_position_count.value
            if current < count:
                logger.warning(
                    f"{self.strategy_name:<9}: PosCountUnderflow prevented | "
                    f"requested=-{count} | current={current} | reason={reason}"
                )
                self.global_position_count.value = 0
            else:
                self.global_position_count.value = current - count
            new_count = self.global_position_count.value

        logger.debug(
            f"{self.strategy_name:<9}: PosCountDec n={new_count}/"
            f"{self.global_risk_policy['max_total_positions']} | delta=-{count} | reason={reason}"
        )
        return new_count

    def _apply_meta_labeling(self, data: pd.DataFrame, entry_request: EntryRequest) -> tuple[float | None, float]:
        if self.meta_model is None:
            return None, entry_request.volume
        if self.feature_extractor is None:
            return None, entry_request.volume

        features = self.feature_extractor(data)
        if features.empty:
            logger.warning(f"{self.strategy_name:<9}: Meta-labeling skipped — feature extractor returned no rows")
            return None, entry_request.volume

        features = features.iloc[[-1]]
        volume_multiplier: float = float(self.meta_model.predict_proba(features)[0, 1])

        if self.calibration_model is not None:
            volume_multiplier = float(self.calibration_model.transform(np.array([volume_multiplier], dtype=float))[0])
        if volume_multiplier < self.meta_min_confidence:
            return None, entry_request.volume

        original_volume = entry_request.volume
        adjusted_volume = self.risk_manager.validate_position_size(self.symbol, original_volume * volume_multiplier)
        if adjusted_volume is None:
            logger.warning(
                f"{self.strategy_name:<9}: MetaLabelReject reason=invalid_pos_size | vol={original_volume * volume_multiplier:.4f}"  # noqa: E501
            )
            return None, entry_request.volume
        logger.info(
            f"{self.strategy_name:<9}: MetaLabel vol={original_volume:.4f} -> "
            f"{adjusted_volume:.4f} | mul={volume_multiplier:.3f}"
        )
        return volume_multiplier, adjusted_volume

    def _process_entry_signal(self, data: pd.DataFrame) -> None:
        assert self.strategy is not None, f"{self.strategy_name}: strategy not initialized"

        entry_request: EntryRequest | None = self.strategy.generate_entry_signal(data)
        if entry_request is None:
            return

        submission_time = time.time()

        if self.order_type == "bracket":
            sl_price = entry_request.buy_sl
            entry_price = entry_request.buy_stop
            if entry_price is None or sl_price is None:
                logger.error(
                    f"{self.strategy_name:<9}: TradeReject reason=bracket_prices_missing"
                    f" | buy_stop={entry_request.buy_stop} buy_sl={entry_request.buy_sl}"
                )
                return
        else:
            sl_price = entry_request.sl
            entry_price = data["Close"].iloc[-1]
            if sl_price is None:
                logger.error(f"{self.strategy_name:<9}: TradeReject reason=sl_missing")
                return

        validation = self.risk_manager.validate_trade(
            strategy_name=self.strategy_name,
            symbol=self.symbol,
            expected_price=entry_price,
            sl_price=sl_price,
            signal=entry_request.signal,
        )
        if not validation.can_trade:
            if validation.reason == "outside_trading_hours":
                logger.debug(f"{self.strategy_name:<9}: TradeSkip reason=outside_trading_hours")
            else:
                logger.warning(f"{self.strategy_name:<9}: TradeReject reason={validation.reason}")
            return

        position_reserved = True
        execution_submitted = False
        result = None

        try:
            entry_request.volume = validation.volume
            volume_multiplier, adjusted_volume = self._apply_meta_labeling(data, entry_request)

            if volume_multiplier is None and self.meta_model is not None:
                self.risk_manager.release_position_reservation(reason="meta_labeling_rejected")
                position_reserved = False
                logger.info(
                    f"{self.strategy_name:<9}: TradeCancel reason=meta_confidence "
                    f"| threshold={self.meta_min_confidence:.3f}"
                )
                return

            entry_request.volume = adjusted_volume
            result = self.executor.execute_entry(entry_request)

            if not result.success:
                self.risk_manager.release_position_reservation(reason="execution_failed")
                position_reserved = False
                logger.error(f"{self.strategy_name:<9}: EntryFail err={result.error_message}")
                return

            execution_submitted = True

        finally:
            if position_reserved and not execution_submitted:
                self.risk_manager.release_position_reservation(reason="exception_during_execution")
                logger.error(f"{self.strategy_name:<9}: PosSlotRel reason=exception_during_execution")

        trade_id = self._generate_trade_id()

        if self.order_type == "bracket":
            assert result.order_tickets is not None, (
                f"successful bracket execution must have order_tickets (trade_id={trade_id})"
            )
            buy_ticket, sell_ticket = result.order_tickets
            self._store_entry_metadata_bracket(
                trade_id, entry_request, volume_multiplier, submission_time, buy_ticket, sell_ticket
            )
            signal_str = "BRACKET"
        else:
            expected_entry_price = entry_request.entry_price or data["Close"].iloc[-1]
            assert result.ticket is not None, f"successful standard execution must have ticket (trade_id={trade_id})"
            self._store_entry_metadata_standard(
                trade_id,
                result.ticket,
                volume_multiplier,
                submission_time,
                expected_entry_price,
                entry_request.sl,
            )
            signal_str = "BUY" if entry_request.signal == 1 else "SELL"

        logger.info(
            f"{self.strategy_name:<9}: ENTRY tid={trade_id} | dir={signal_str} | vol={entry_request.volume:.2f}"
        )

    def _process_modify_signals(
        self, data: pd.DataFrame, preloaded_positions: list[PositionCacheEntry] | None = None
    ) -> None:
        assert self.strategy is not None, f"{self.strategy_name}: strategy not initialized"

        positions: list[PositionCacheEntry]
        if preloaded_positions is not None:
            with self.position_state_lock:
                known = frozenset(self.known_positions)
            positions = [p for p in preloaded_positions if p["ticket"] in known]
        else:
            loaded = self._load_strategy_positions(mode="cache", include_unknown=False)
            if loaded is None:
                logger.error(f"{self.strategy_name:<9}: ModifySkip reason=mt5_api_failure")
                raise RuntimeError(f"{self.strategy_name}: MT5 positions_get() returned None during modify pass")
            positions = loaded

        if not positions:
            return

        for entry in positions:
            pos = cache_entry_to_position(entry)
            request = self.strategy.generate_modify_signal(pos, data)
            if request is None:
                continue
            if isinstance(request, ExitRequest):
                self._execute_and_log_exit(request, pos, "BAR-ALIGNED")
            elif isinstance(request, ModifyRequest):
                self._handle_modify_adjustment(request)
            else:
                logger.warning(f"{self.strategy_name:<9}: UnknownModifyRequest type={type(request).__name__}")

    def _handle_modify_adjustment(self, request: ModifyRequest) -> None:
        result = self.executor.execute_modify(request)
        if result.success:
            self._invalidate_cache_for_ticket(request.ticket)
            self._patch_tracked_snapshot_levels(ticket=request.ticket, new_sl=request.new_sl, new_tp=request.new_tp)
            sl_str = f"{request.new_sl:.5f}" if request.new_sl is not None else "—"
            tp_str = f"{request.new_tp:.5f}" if request.new_tp is not None else "—"
            logger.debug(
                f"{self.strategy_name:<9}: MODIFIED t={request.ticket} | sl={sl_str} tp={tp_str} | {request.comment}"
            )
        else:
            logger.error(f"{self.strategy_name:<9}: ModifyFail t={request.ticket} | err={result.error_message}")

    def _build_fill_data(
        self,
        trade_id: int,
        position: PositionCacheEntry,
        expected_entry_price: float,
        opening_sl: float | None,
        fill_time_ms: float | None = None,
        volume_multiplier: float | None = None,
    ) -> FillData:
        return FillData(
            trade_id=trade_id,
            position=position,
            expected_entry_price=expected_entry_price,
            opening_sl=opening_sl,
            strategy_name=self.strategy_name,
            fill_time_ms=fill_time_ms,
            volume_multiplier=volume_multiplier,
        )

    @staticmethod
    def _build_close_data(
        trade_id: int,
        position: PositionCacheEntry,
        expected_exit_price: float | None,
        opening_sl: float | None,
        exit_trigger: str,
        entry_price: float,
        expected_entry_price: float | None,
    ) -> CloseData:
        return CloseData(
            trade_id=trade_id,
            position=position,
            expected_exit_price=expected_exit_price,
            opening_sl=opening_sl,
            exit_trigger=exit_trigger,
            entry_price=entry_price,
            expected_entry_price=expected_entry_price,
        )

    def _build_partial_close_data(
        self, trade_id: int, position: Position, remaining_volume: float, data: ExitLogData
    ) -> PartialCloseData:
        return PartialCloseData(
            trade_id=trade_id,
            position=NormalizedPosition.from_mt5(position).to_partial_snapshot(),
            closed_volume=data.closed_volume if data.closed_volume is not None else 0.0,
            remaining_volume=remaining_volume,
            expected_exit_price=data.expected_exit_price,
            opening_sl=data.opening_sl,
            strategy_name=self.strategy_name,
            exit_trigger=data.exit_trigger,
            entry_price=position.price_open,
            expected_entry_price=data.expected_entry_price,
            deal_id=data.deal_id,
        )

    def _get_cached_symbol_tick(self, symbol: str) -> MT5Tick | None:
        if symbol in self._symbol_tick_cache:
            return self._symbol_tick_cache[symbol]
        tick = mt.symbol_info_tick(symbol)
        if tick is None:
            return None
        self._symbol_tick_cache[symbol] = tick
        return tick

    def _log_partial_close_execution(self, data: ExitLogData) -> Position | None:
        self._invalidate_cache_for_ticket(data.ticket)
        updated_position = self._requery_position(data.ticket)
        if updated_position is None:
            logger.warning(f"{self.strategy_name:<9}: PartCloseQueryFail t={data.ticket}")
            return None

        trade_id = self.ticket_to_trade_id.get(data.ticket)
        if trade_id is None:
            logger.error(f"{self.strategy_name:<9}: PartCloseLogFail t={data.ticket} | reason=no_trade_id")
            return None

        remaining_volume = updated_position.volume
        partial_data = self._build_partial_close_data(
            trade_id=trade_id,
            position=updated_position,
            remaining_volume=remaining_volume,
            data=data,
        )
        self.trade_logger.log_partial_close(partial_data)

        if trade_id in self.entry_metadata:
            self.entry_metadata[trade_id]["position_snapshot"] = updated_position.to_cache_entry()

        logger.info(
            f"{self.strategy_name:<9}: PARTIAL CLOSE id={trade_id} | t={data.ticket} | "
            f"closed={data.closed_volume:.2f} | rem={remaining_volume:.2f}"
        )
        return updated_position

    def _log_full_close_execution(self, trade_id: int | None, pos: Position, data: ExitLogData) -> None:
        ticket = pos.ticket

        if trade_id is not None:
            close_data = self._build_close_data(
                trade_id=trade_id,
                position=pos.to_cache_entry(),
                expected_exit_price=data.expected_exit_price,
                opening_sl=data.opening_sl,
                exit_trigger=data.exit_trigger,
                entry_price=data.entry_price,
                expected_entry_price=data.expected_entry_price,
            )
            self.trade_logger.log_close(close_data)

        with self.position_state_lock:
            self.known_positions.discard(ticket)
            self.local_position_count = max(0, self.local_position_count - 1)
            self.ticket_to_trade_id.pop(ticket, None)
            if trade_id is not None:
                self.entry_metadata.pop(trade_id, None)

        self._invalidate_cache_for_ticket(ticket)
        max_pos = self.global_risk_policy["max_total_positions"]

        if trade_id is None:
            logger.warning(
                f"{self.strategy_name:<9}: CLOSED t={ticket} | reason=no_trade_id | "
                f"trigger={data.exit_trigger} | pos={self.local_position_count}/{max_pos}"
            )
            return

        logger.info(
            f"{self.strategy_name:<9}: CLOSED id={trade_id} | t={ticket} | "
            f"trigger={data.exit_trigger} | pos={self.local_position_count}/{max_pos} | "
            f"global={self.global_position_count.value}/{max_pos}"
        )

    @staticmethod
    def _snap_to_level(actual: float, level: float) -> bool:
        return abs(actual - level) <= level * _SL_TP_SNAP_TOLERANCE

    def _resolve_expected_exit_price(self, pos: Position) -> tuple[float, str] | None:
        actual: float = pos.price_current
        entry: float = pos.price_open
        sl: float | None = pos.sl if pos.sl else None
        tp: float | None = pos.tp if pos.tp else None
        is_buy: bool = pos.type == PositionType.BUY

        if abs(actual - entry) <= entry * _BREAKEVEN_HALF_BAND_RATIO:
            return entry, _TRIGGER_BREAKEVEN

        if is_buy:
            if tp is not None and actual >= tp * (1.0 - _SL_TP_SNAP_TOLERANCE):
                return tp, _TRIGGER_TP
            if sl is not None and actual <= sl * (1.0 + _SL_TP_SNAP_TOLERANCE):
                return sl, _TRIGGER_SL
        else:
            if tp is not None and actual <= tp * (1.0 + _SL_TP_SNAP_TOLERANCE):
                return tp, _TRIGGER_TP
            if sl is not None and actual >= sl * (1.0 - _SL_TP_SNAP_TOLERANCE):
                return sl, _TRIGGER_SL

        return actual, _TRIGGER_SIGNAL

    def _execute_and_log_exit(self, exit_request: ExitRequest, pos: Position, exit_context: str = "UNKNOWN") -> None:
        ticket = pos.ticket
        exit_price_result = self._resolve_expected_exit_price(pos)
        if exit_price_result is None:
            logger.error(f"{self.strategy_name:<9}: ExitPriceFail t={ticket} | ctx={exit_context}")
            return

        expected_exit_price, price_source = exit_price_result
        logger.debug(
            f"{self.strategy_name:<9}: ExitPrice t={ticket} | src={price_source} | px={expected_exit_price:.5f}"
        )

        trade_id, expected_entry_price, opening_sl, entry_price = self._resolve_entry_prices(ticket, pos)
        result = self.executor.execute_exit(exit_request)
        if not result.success:
            logger.error(f"{self.strategy_name:<9}: ExitFail t={ticket} | err={result.error_message}")
            return

        exit_log_data = self.exit_log_data_cls(
            ticket=ticket,
            expected_exit_price=expected_exit_price,
            exit_trigger=getattr(exit_request, "exit_reason", exit_context),
            expected_entry_price=expected_entry_price if expected_entry_price is not None else entry_price,
            opening_sl=opening_sl if opening_sl is not None else 0.0,
            entry_price=entry_price,
            deal_id=result.deal_id,
        )

        if exit_request.portion < 1.0:
            exit_log_data.closed_volume = pos.volume * exit_request.portion
            self._log_partial_close_execution(exit_log_data)
        else:
            self._log_full_close_execution(trade_id, pos, exit_log_data)

    def _monitor_exits(self, data: pd.DataFrame, preloaded_positions: list[PositionCacheEntry] | None = None) -> None:
        positions = (
            preloaded_positions
            if preloaded_positions is not None
            else self._load_strategy_positions(mode="cache", include_unknown=True)
        )
        if positions is None:
            logger.error(f"{self.strategy_name:<9}: ExitMonFail reason=mt5_api_failure")
            return

        self._refresh_tracked_position_snapshots(positions)
        current_tickets: set[int] = {pos["ticket"] for pos in positions}

        mt5_orders = self._get_strategy_orders_direct() if self._has_pending_bracket_orders() else None

        with self.position_state_lock:
            known_snapshot = set(self.known_positions)

        closed_tickets = [t for t in known_snapshot if t not in current_tickets]
        if closed_tickets:
            self._handle_closed_positions(closed_tickets)

        self._check_expired_bracket_orders(current_tickets, orders=mt5_orders)

        if not positions or data.empty:
            return

        for pos_dict in positions:
            ticket = pos_dict["ticket"]
            pos = cache_entry_to_position(pos_dict)
            if ticket not in known_snapshot:
                self._handle_new_fill(pos)
                known_snapshot.add(ticket)
            self._check_and_execute_exit(pos, data)

        if self.order_type == "bracket" and mt5_orders is not None:
            self._cancel_opposite_bracket_orders(positions=positions, mt5_orders=mt5_orders)

    def _check_and_execute_exit(self, pos: Position, data: pd.DataFrame) -> None:
        assert self.strategy is not None, f"{self.strategy_name}: strategy not initialized"
        exit_request = self.strategy.generate_exit_signal(pos, data)
        if exit_request is None:
            return
        self._execute_and_log_exit(exit_request=exit_request, pos=pos, exit_context="SIGNAL_EXIT")

    def _cancel_opposite_bracket_orders(
        self, positions: list[PositionCacheEntry] | None = None, mt5_orders: list[OrderSnapshot] | None = None
    ) -> None:
        results = self.executor.cancel_bracket_orders(
            symbols=[self.symbol],
            magics=[self.magic_number],
            preloaded_positions=positions,
            preloaded_orders=mt5_orders,
        )
        total_cancelled = sum(n for symbol_results in results.values() for n in symbol_results.values())
        if total_cancelled > 0:
            logger.info(f"{self.strategy_name:<9}: OCOCancel n={total_cancelled}")

    def _reconcile_shutdown_state(self) -> None:
        positions = self._load_strategy_positions(mode="direct", include_unknown=True)
        if positions is None:
            logger.error(f"{self.strategy_name:<9}: ShutdownRecFail reason=positions_get_failed")
            return

        if positions:
            tickets_to_close = [pos["ticket"] for pos in positions]
            results = self.executor.close_positions(tickets=tickets_to_close)
            successful = [t for t, (ok, _) in results.items() if ok]
            if successful:
                self.known_positions.difference_update(successful)
                logger.info(f"{self.strategy_name:<9}: ShutdownPos closed={len(successful)}/{len(tickets_to_close)}")

        orders = self._get_strategy_orders_direct()
        if orders is None:
            logger.error(f"{self.strategy_name:<9}: ShutdownRecFail reason=orders_get_failed")
            return

        cancelled = sum(1 for order in orders if self.executor._cancel_order(order.ticket))
        if cancelled:
            logger.info(f"{self.strategy_name:<9}: ShutdownOrd cancelled={cancelled}")

    def _calculate_next_entry_time(self, from_time: datetime) -> datetime:
        """Calculate next entry time aligned to timeframe boundary from given time."""
        offset_td = timedelta(seconds=self.strategy_offset_seconds)

        next_boundary_minute = ((from_time.minute // self.timeframe_minutes) + 1) * self.timeframe_minutes
        hours_to_add = next_boundary_minute // 60
        adjusted_minute = next_boundary_minute % 60

        next_entry = from_time.replace(minute=adjusted_minute, second=0, microsecond=0)
        if hours_to_add:
            next_entry += timedelta(hours=hours_to_add)
        next_entry += offset_td

        if next_entry <= from_time:
            next_entry += timedelta(minutes=self.timeframe_minutes)

        return next_entry

    def _calculate_next_exit_time(self, from_time: datetime) -> datetime:
        """Calculate next exit time at next minute boundary from given time."""
        offset_td = timedelta(seconds=self.strategy_offset_seconds)

        next_exit = from_time.replace(second=0, microsecond=0) + timedelta(minutes=1)
        next_exit += offset_td

        if next_exit <= from_time:
            next_exit += timedelta(minutes=1)

        return next_exit

    def _initialize_schedule(self) -> tuple[datetime, datetime]:
        """Calculate initial next_entry_time and next_exit_time based on timeframe."""
        now = datetime.now(self.broker_tz)
        next_entry = self._calculate_next_entry_time(now)
        next_exit = self._calculate_next_exit_time(now)
        return next_entry, next_exit

    def _should_check_entry_signal(self, data: pd.DataFrame) -> bool:
        """Return True if bar is new and entry signal should be evaluated."""
        if len(data) == 0:
            return False

        if self.timeframe_minutes == 1:
            return True

        current_bar_time = data.index[-1]

        if self.last_processed_bar_time is not None:
            if current_bar_time.tzinfo is None or self.last_processed_bar_time.tzinfo is None:
                logger.warning(
                    f"BarTZWarn strat={self.strategy_name} | cur_tz={current_bar_time.tzinfo} | "
                    f"last_tz={self.last_processed_bar_time.tzinfo}"
                )
                self.last_processed_bar_time = current_bar_time
                return True

            if current_bar_time <= self.last_processed_bar_time:
                logger.debug(f"BarSkip strat={self.strategy_name} | reason=already_processed")
                return False

        self.last_processed_bar_time = current_bar_time
        return True


def run_strategy_process(config: RunnerConfig) -> None:
    """Entry point for ``multiprocessing.Process``."""
    log_root = Path(config.global_risk_policy["log_root"])
    log_root.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_root / f"{config.strategy_name}.log"),
            logging.StreamHandler(),
        ],
    )

    runner = StrategyRunner(config=config)
    try:
        runner.run()
    except Exception as error:
        logger.exception(f"ProcCrash strat={config.strategy_name} | err={error}")
