"""Orchestrator — multi-process strategy coordination with shared position cache."""

from __future__ import annotations

import ctypes
import logging
import threading
import time
from datetime import datetime, timedelta
from multiprocessing import Lock, Manager, Process, Value
from pathlib import Path
from typing import TYPE_CHECKING, cast

import MetaTrader5 as mt
import numpy as np

from nexus_trade.config.profile import MetaLabelingConfig, RiskProfile
from nexus_trade.config.timings import GRACE_SECONDS, SYSTEM_TIMINGS
from nexus_trade.core.connection import MT5Connection
from nexus_trade.core.registry import STRATEGY_CONFIG_REGISTRY
from nexus_trade.core.repository import PositionRepository
from nexus_trade.execution.executor import OrderExecutor
from nexus_trade.execution.trade_ids import TradeIDSequenceManager
from nexus_trade.filters.news import preprocess_calendar_file
from nexus_trade.runner import RunnerConfig, run_strategy_process
from nexus_trade.utils.format import format_price_display, log_section_header

if TYPE_CHECKING:
    from multiprocessing.managers import SyncManager

    from MetaTrader5 import AccountInfo

    from nexus_trade.config.account import MT5ConnectionConfig
    from nexus_trade.config.strategy import BaseStrategyParams, StrategyConfig
    from nexus_trade.core.protocols import AtomicInt, ProcessLock
    from nexus_trade.core.state import SharedState
    from nexus_trade.core.types import OrderSnapshot, PositionCacheEntry


logger = logging.getLogger(__name__)
HEARTBEAT_INTERVAL_SECONDS: int = SYSTEM_TIMINGS.heartbeat_interval
HEARTBEAT_LOG_INTERVAL_SECONDS: int = SYSTEM_TIMINGS.heartbeat_log_interval


class Orchestrator:
    """Multi-strategy orchestrator with shared position cache."""

    def __init__(self, account_config: MT5ConnectionConfig, profile: RiskProfile, log_root: Path) -> None:
        self.log_root: Path = Path(log_root)
        self.account_config: MT5ConnectionConfig = account_config
        self._profile: RiskProfile = profile

        self.manager: SyncManager = Manager()

        self.global_position_count: AtomicInt = Value(ctypes.c_int, 0)
        self.global_trade_count: AtomicInt = Value(ctypes.c_int, 0)
        self.position_cache_lock: ProcessLock = Lock()

        self.shared_state: SharedState = self._initialize_shared_state()

        self.position_repo: PositionRepository = PositionRepository(
            shared_state=self.shared_state,
            position_cache_lock=self.position_cache_lock,
            cache_staleness_threshold=SYSTEM_TIMINGS.cache_staleness_threshold,
        )

        self.strategy_processes: dict[str, Process] = {}
        self.strategy_configs: dict[str, StrategyConfig[BaseStrategyParams]] = {}
        self.process_crash_count: dict[str, int] = {}

        self.trade_id_db_path: Path = self.log_root / "trade_id_sequence.db"
        self.trade_id_manager: TradeIDSequenceManager = TradeIDSequenceManager(self.trade_id_db_path)

        self.connection: MT5Connection | None = None
        self.executor: OrderExecutor | None = None
        self._shutdown_initiated: bool = False
        self._drift_first_seen: float | None = None
        self._position_drift_value: int | None = None

        logger.info(
            f"OrchInit acct={self._profile.account.type} | "
            f"pos_max={self._profile.limits.max_total_positions} | "
            f"tr_max={self._profile.limits.max_daily_trades}"
        )

    def _initialize_shared_state(self) -> SharedState:
        state: SharedState = cast("SharedState", self.manager.dict())
        state["shutdown_flag"] = False
        state["position_cache"] = cast("dict[int, PositionCacheEntry]", self.manager.dict())
        state["position_cache_timestamp"] = 0.0
        state["daily_drawdown"] = 0.0
        state["max_drawdown"] = 0.0
        state["daily_trade_counts"] = cast("dict[str, int]", self.manager.dict())
        state["calendar_cache"] = None
        state["calendar_cache_timestamp"] = 0.0
        self._reset_shared_daily_state(state)
        return state

    def _reset_shared_daily_state(self, state: SharedState | None = None) -> None:
        target: SharedState = self.shared_state if state is None else state
        target["daily_trade_counts"] = cast("dict[str, int]", self.manager.dict())
        target["daily_equity_high"] = 0.0
        target["daily_drawdown"] = 0.0
        target["daily_drawdown_current_equity"] = 0.0
        target["daily_drawdown_peak_equity"] = 0.0
        target["daily_drawdown_last_update"] = 0.0
        target["daily_drawdown_initialized"] = False
        target["daily_drawdown_cache_date"] = None
        target["max_drawdown"] = 0.0
        target["max_drawdown_current_equity"] = 0.0
        target["max_drawdown_peak_equity"] = 0.0
        target["max_drawdown_last_update"] = 0.0
        target["max_drawdown_initialized"] = False
        target["drawdown_last_refresh"] = 0.0
        target["hist_pnl_sum"] = 0.0
        target["hist_peak_equity"] = float(self._profile.account.initial_balance)

        target["last_equity_update"] = 0.0

    def _get_magic_numbers(self) -> frozenset[int]:
        return frozenset(config.execution.magic_number for config in self.strategy_configs.values())

    def _update_shared_cache(self, positions: list[PositionCacheEntry]) -> None:
        """Atomically write positions into the shared position cache."""
        cache_dict = {entry["ticket"]: entry for entry in positions}
        with self.position_cache_lock:
            cache = self.shared_state.get("position_cache")
            if cache is None:
                cache = self.manager.dict()

            shared_cache = cast("dict[int, PositionCacheEntry]", cache)
            stale = [t for t in list(shared_cache.keys()) if t not in cache_dict]
            for t in stale:
                shared_cache.pop(t, None)
            shared_cache.update(cache_dict)
            self.shared_state["position_cache"] = shared_cache
            self.shared_state["position_cache_timestamp"] = time.time()

    def sync_existing_positions(self, managed_positions: list[PositionCacheEntry]) -> None:
        if not managed_positions:
            logger.debug("PosSync n=0")
            return
        self._update_shared_cache(managed_positions)
        with self.global_position_count.get_lock():
            self.global_position_count.value = len(managed_positions)
        logger.info(f"PosSync n={len(managed_positions)}")
        self._log_position_details(managed_positions)

    def _log_position_details(self, positions: list[PositionCacheEntry]) -> None:
        for pos in positions:
            side = "B" if pos["type"] == 0 else "S"
            price_display = format_price_display(float(pos["price_open"]))
            logger.info(
                f"sym={pos['symbol']:<7} | side={side} | "
                f"vol={pos['volume']:>4.2f} | px={price_display:>10} | "
                f"m={pos['magic']:>3}"
            )
            logger.debug(f"t={pos['ticket']:>10}")

    def preload_calendar_cache(self) -> None:
        """Parse economic calendar CSV and seed shared_state before spawning strategies."""
        calendar_path = self.account_config.calendar_path
        if calendar_path is None:
            logger.info("CalPreloadSkip reason=calendar_path_not_configured")
            return

        df, holidays_frozen = preprocess_calendar_file(calendar_path, self.account_config.broker_tz)
        raw = df.to_dict("records")
        self.shared_state["calendar_cache"] = [{str(k): v for k, v in row.items()} for row in raw]
        self.shared_state["calendar_holidays"] = list(holidays_frozen)
        self.shared_state["calendar_cache_timestamp"] = time.time()

        logger.info(f"CalPreload evt={len(df)} | hi={df['priority'].eq('High').sum()} | hol={len(holidays_frozen)}")

    def _handle_strategy_import_error(self, strategy_name: str, error: Exception, config_module_path: str) -> None:
        logger.critical(f"StratLoadFail name={strategy_name} | mod={config_module_path}")
        logger.critical(f"StratLoadErr name={strategy_name} | err={error}", exc_info=True)
        raise RuntimeError(f"Cannot start with broken enabled strategy: {strategy_name}")

    def discover_strategies(self) -> None:
        for strategy_name in self._profile.enabled_strategy_names:
            config_module_path = f"nexus_trade.strategies.{strategy_name}.config"
            try:
                config: StrategyConfig[BaseStrategyParams] = STRATEGY_CONFIG_REGISTRY.get_strategy_config(strategy_name)
                self.strategy_configs[strategy_name] = config
                params = config.params
                logger.debug(f"StratDisc name={strategy_name} | sym={params.symbol} | tf={params.timeframe}")
            except Exception as e:
                self._handle_strategy_import_error(strategy_name, e, config_module_path)

    def reset_daily_counters(self) -> None:
        with self.global_trade_count.get_lock():
            old_count = self.global_trade_count.value
            self.global_trade_count.value = 0
        logger.info(f"DailyReset tr={old_count}->0")
        self._reset_shared_daily_state()
        logger.info("DailyReset eq_hi=0 | dd=0 | counts=0")

    def spawn_strategy_process(
        self,
        strategy_name: str,
        config: StrategyConfig[BaseStrategyParams],
        strategy_index: int = 0,
    ) -> None:
        strategy_offset_seconds = (
            strategy_index % SYSTEM_TIMINGS.max_strategy_offset_slots
        ) / SYSTEM_TIMINGS.strategy_offset_divisor

        meta_cfg: MetaLabelingConfig = (
            self._profile.strategies[strategy_name].meta_labeling
            if strategy_name in self._profile.strategies
            else MetaLabelingConfig()
        )

        runner_config = RunnerConfig(
            strategy_name=strategy_name,
            strategy_config=config,
            broker_config=self.account_config,
            risk_profile=self._profile,
            log_root=self.log_root,
            shared_state=self.shared_state,
            global_trade_count=self.global_trade_count,
            global_position_count=self.global_position_count,
            position_cache_lock=self.position_cache_lock,
            trade_id_db_path=self.trade_id_db_path,
            meta_labeling=meta_cfg,
            strategy_offset_seconds=strategy_offset_seconds,
        )

        process = Process(target=run_strategy_process, args=(runner_config,))
        process.start()
        self.strategy_processes[strategy_name] = process
        logger.debug(f"ProcSpawn name={strategy_name} | pid={process.pid}")

    def sync_to_next_heartbeat(self) -> None:
        now = datetime.now()
        midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
        seconds_since_midnight = (now - midnight).total_seconds()

        current_multiple = int(seconds_since_midnight // HEARTBEAT_INTERVAL_SECONDS)
        target_seconds = current_multiple * HEARTBEAT_INTERVAL_SECONDS
        if seconds_since_midnight > target_seconds:
            target_seconds += HEARTBEAT_INTERVAL_SECONDS

        sleep_duration = target_seconds - seconds_since_midnight
        if sleep_duration > 0:
            time.sleep(sleep_duration)

    def _reconcile_position_counter(self, actual_count: int) -> None:
        """Correct global_position_count after 90 s of persistent drift."""
        with self.global_position_count.get_lock():
            current_count = self.global_position_count.value

        if current_count == actual_count:
            self._drift_first_seen = None
            self._position_drift_value = None
            return

        drift = current_count - actual_count
        now = time.time()

        if self._position_drift_value != drift:
            self._position_drift_value = drift
            self._drift_first_seen = now
            logger.debug(f"PosDrift cnt={current_count} | mt5={actual_count} | d={drift:+d} | action=waiting")
            return

        if self._drift_first_seen is None:
            self._drift_first_seen = now

        if now - self._drift_first_seen < GRACE_SECONDS:
            return

        with self.global_position_count.get_lock():
            self.global_position_count.value = actual_count
        self._drift_first_seen = None
        self._position_drift_value = None
        logger.info(f"PosDrift cnt={current_count} | mt5={actual_count} | d={drift:+d} | action=reconciled")

    def refresh_position_cache(self) -> None:
        """Fetch managed positions via PositionRepository, write cache, reconcile counter."""
        try:
            magic_numbers = self._get_magic_numbers()
            managed_positions = self.position_repo.get_managed_positions(magic_numbers)
            if managed_positions is None:
                logger.error("PosCacheRefreshFail reason=mt5_api_failure")
                return

            self._update_shared_cache(managed_positions)
            self._reconcile_position_counter(len(managed_positions))
            logger.debug(f"PosCacheRefresh n={len(managed_positions)} | cnt={len(managed_positions)} | age=0s")
        except Exception as exc:
            logger.error(f"PosCacheRefreshFail err={exc}", exc_info=True)

    def _should_log_heartbeat(self, current_time: datetime, last_log_time: float) -> bool:
        interval_seconds = max(1, int(HEARTBEAT_LOG_INTERVAL_SECONDS))
        boundary_tolerance = max(1, int(HEARTBEAT_INTERVAL_SECONDS))
        if time.time() - last_log_time + boundary_tolerance < interval_seconds:
            return False
        midnight = current_time.replace(hour=0, minute=0, second=0, microsecond=0)
        seconds_since_midnight = int((current_time - midnight).total_seconds())
        return seconds_since_midnight % interval_seconds < boundary_tolerance

    def monitor_heartbeats(self) -> None:
        broker_tz = self.account_config.broker_tz
        last_reset_date = datetime.now(tz=broker_tz).date()
        last_log_time = time.time()
        last_drawdown_refresh: float = time.time()

        while not self.shared_state["shutdown_flag"]:
            self.refresh_position_cache()

            now_mt = time.time()
            if now_mt - last_drawdown_refresh >= SYSTEM_TIMINGS.drawdown_refresh_interval_seconds:
                try:
                    self._refresh_shared_drawdown()
                except Exception:
                    logger.exception("DDRefreshFail process=orchestrator")
                last_drawdown_refresh = now_mt

            current_time = datetime.now()
            active = sum(1 for p in self.strategy_processes.values() if p.is_alive())
            cache_pos = self.global_position_count.value
            total_tr = self.global_trade_count.value
            max_pos = self._profile.limits.max_total_positions
            max_tr = self._profile.limits.max_daily_trades

            if self._should_log_heartbeat(current_time, last_log_time):
                logger.info(
                    f"HB t={current_time.strftime('%H:%M:%S')} | "
                    f"proc={active}/{len(self.strategy_processes)} | "
                    f"pos={cache_pos}/{max_pos} | tr={total_tr}/{max_tr}"
                )
                last_log_time = time.time()

            if cache_pos >= max_pos:
                logger.warning(f"LimitHit typ=pos | cur={cache_pos} | max={max_pos}")
            if total_tr >= max_tr:
                logger.warning(f"LimitHit typ=tr | cur={total_tr} | max={max_tr}")

            dead = [(n, p) for n, p in self.strategy_processes.items() if not p.is_alive()]
            if dead:
                for name, proc in dead:
                    self.process_crash_count[name] = self.process_crash_count.get(name, 0) + 1
                    logger.critical(f"Crash strat={name} | pid={proc.pid or 'N/A'} | code={proc.exitcode}")
                logger.critical(f"CrashShutdown n={len(dead)}")
                self.shutdown()
                break

            if current_time.date() > last_reset_date:
                self.reset_daily_counters()
                last_reset_date = current_time.date()

            self.sync_to_next_heartbeat()

    def _log_enabled_strategies(self) -> None:
        log_section_header(
            logger,
            f"ENABLED STRATEGIES (Account: {self._profile.account.type})",
            level=logging.DEBUG,
        )
        for name, config in self.strategy_configs.items():
            params = config.params
            risk_value = self._profile.strategies[name].risk_value
            method = self._profile.strategies[name].position_sizing_method
            logger.debug(
                f"StratConfig name={name} | sym={params.symbol} | tf={params.timeframe} | "
                f"sizing={method} | risk={risk_value}"
            )

    def _connect_to_mt5(self) -> None:
        self.connection = MT5Connection(self.account_config)
        if not self.connection.connect():
            raise RuntimeError("Orchestrator startup failed: Unable to connect to MT5.")
        self.executor = OrderExecutor(self.account_config.broker_tz)
        logger.debug("OrchMT5 conn=ok")

    def _sync_positions(self) -> None:
        log_section_header(logger, "SYNCING EXISTING POSITIONS", level=logging.DEBUG)
        magic_numbers = self._get_magic_numbers()
        managed_positions = self.position_repo.get_managed_positions(magic_numbers) or []
        self.sync_existing_positions(managed_positions=managed_positions)
        time.sleep(3)

    def _spawn_all_strategies(self) -> None:
        for i, (name, config) in enumerate(self.strategy_configs.items()):
            self.spawn_strategy_process(name, config, strategy_index=i)
            if i < len(self.strategy_configs) - 1:
                time.sleep(0.1)

    def _refresh_shared_drawdown(self) -> None:
        account = mt.account_info()
        if account is None:
            logger.error("DrawdownRefreshFail reason=account_info_unavailable")
            if self.connection is None or not self.connection.is_connected():
                logger.critical("DrawdownRefreshFail reason=mt5_disconnected. Activating shutdown flag.")
            else:
                logger.error(
                    "DrawdownRefreshFail reason=transient_api_error_while_connected. Activating shutdown flag."
                )

            self.shared_state["shutdown_flag"] = True
            return
        self._refresh_max_drawdown(account)
        self._refresh_daily_drawdown(account)

    def _refresh_max_drawdown(self, account: AccountInfo) -> None:
        now = datetime.now(tz=self.account_config.broker_tz)

        current_equity = float(account.equity)
        initial_balance = float(self._profile.account.initial_balance)

        last_refresh_ts: float = float(self.shared_state.get("drawdown_last_refresh", 0.0))
        is_full_rescan = last_refresh_ts == 0.0
        query_start = (
            self._profile.account.history_start
            if is_full_rescan
            else datetime.fromtimestamp(last_refresh_ts, tz=self.account_config.broker_tz)
        )
        query_end = now + timedelta(seconds=1)

        new_deals = mt.history_deals_get(query_start, query_end) or ()
        stored_pnl = 0.0 if is_full_rescan else float(self.shared_state.get("hist_pnl_sum", 0.0))
        stored_peak = (
            float(initial_balance)
            if is_full_rescan
            else float(self.shared_state.get("hist_peak_equity", float(initial_balance)))
        )

        pnl_increments: list[float] = [
            float(d.profit + d.commission + d.swap + d.fee) for d in new_deals if d.type in (0, 1)
        ]
        if pnl_increments:
            arr = np.asarray(pnl_increments, dtype=np.float64)
            running = stored_pnl + np.cumsum(arr)
            stored_peak = max(stored_peak, float((initial_balance + running).max()))
            stored_pnl = float(stored_pnl + arr.sum())

        peak_equity = max(stored_peak, current_equity)
        max_dd = (peak_equity - current_equity) / peak_equity if peak_equity > 0 else 0.0
        now_ts = now.timestamp()

        self.shared_state.update(
            {
                "hist_pnl_sum": stored_pnl,
                "hist_peak_equity": peak_equity,
                "max_drawdown": max_dd,
                "max_drawdown_peak_equity": peak_equity,
                "max_drawdown_current_equity": current_equity,
                "max_drawdown_last_update": now_ts,
                "max_drawdown_initialized": True,
                "drawdown_last_refresh": now_ts,
            }
        )
        logger.debug(f"DDMaxRefresh max={max_dd * 100:.2f}% | equity={current_equity:.2f} | peak={peak_equity:.2f}")

    def _refresh_daily_drawdown(self, account: AccountInfo) -> None:
        now = datetime.now(tz=self.account_config.broker_tz)
        current_equity = float(account.equity)
        now_ts = now.timestamp()

        stored_high = float(self.shared_state.get("daily_equity_high", 0.0) or 0.0)
        daily_peak = max(stored_high, current_equity)
        daily_dd = (daily_peak - current_equity) / daily_peak if daily_peak > 0 else 0.0

        self.shared_state.update(
            {
                "daily_equity_high": daily_peak,
                "daily_drawdown": daily_dd,
                "daily_drawdown_peak_equity": daily_peak,
                "daily_drawdown_current_equity": current_equity,
                "daily_drawdown_last_update": now_ts,
                "daily_drawdown_initialized": True,
                "daily_drawdown_cache_date": now.date(),
            }
        )
        logger.debug(
            f"DDDailyRefresh daily={daily_dd * 100:.2f}% | equity={current_equity:.2f} | peak={daily_peak:.2f}"
        )

    def start(self) -> None:
        self.discover_strategies()
        self._log_enabled_strategies()
        self._connect_to_mt5()
        self._sync_positions()
        self.preload_calendar_cache()
        self._refresh_shared_drawdown()
        self._spawn_all_strategies()
        log_section_header(
            logger,
            f"HEARTBEAT MONITOR STARTING (refresh: {HEARTBEAT_INTERVAL_SECONDS}s)",
            level=logging.DEBUG,
        )
        self.monitor_heartbeats()

    def _drain_strategy_processes(self) -> None:
        """Drain all strategy processes in parallel: 3 s graceful → terminate → 5 s → kill."""
        alive: list[tuple[str, Process]] = [
            (name, proc) for name, proc in self.strategy_processes.items() if proc.is_alive()
        ]
        if not alive:
            for name, proc in self.strategy_processes.items():
                logger.debug(f"ProcState name={name} | state=already_exited | code={proc.exitcode}")
            return

        # Phase 1 — parallel graceful join (3 s)
        grace_threads = [threading.Thread(target=proc.join, kwargs={"timeout": 3}, daemon=True) for _, proc in alive]
        for t in grace_threads:
            t.start()
        for t in grace_threads:
            t.join()

        # Phase 2 — terminate stragglers
        stragglers: list[tuple[str, Process]] = [(name, proc) for name, proc in alive if proc.is_alive()]
        for name, proc in stragglers:
            logger.debug(f"ProcState name={name} | state=join_timeout | action=terminate")
            proc.terminate()

        if not stragglers:
            for name, proc in alive:
                logger.debug(f"ProcState name={name} | state=exited | code={proc.exitcode}")
            return

        # Phase 3 — parallel join after terminate (5 s)
        term_threads = [
            threading.Thread(target=proc.join, kwargs={"timeout": 5}, daemon=True) for _, proc in stragglers
        ]
        for t in term_threads:
            t.start()
        for t in term_threads:
            t.join()

        # Phase 4 — kill any still unresponsive
        for name, proc in stragglers:
            if proc.is_alive():
                logger.error(f"ProcKill name={name}")
                proc.kill()

        for name, proc in self.strategy_processes.items():
            logger.debug(f"ProcState name={name} | state=exited | code={proc.exitcode}")

    def _verify_and_close_remaining(self) -> None:
        try:
            magic_numbers = self._get_magic_numbers()
            managed_positions: list[PositionCacheEntry] = self.position_repo.get_managed_positions(magic_numbers) or []
            managed_orders: list[OrderSnapshot] = self.position_repo.get_managed_orders(magic_numbers) or []
            if managed_positions or managed_orders:
                logger.warning(f"ShutdownVerifyOpen pos={len(managed_positions)} | ord={len(managed_orders)}")
                self._force_close_all_immediate(managed_positions, managed_orders)
            else:
                logger.debug("ShutdownVerify ok=1")
        except Exception as exc:
            logger.error(f"ShutdownVerifyFail err={exc}", exc_info=True)

    def shutdown(self) -> None:
        """Graceful shutdown with guaranteed cleanup of managed positions/orders."""
        if self._shutdown_initiated:
            logger.debug("ShutdownSkip reason=already_in_progress")
            return

        self._shutdown_initiated = True

        log_section_header(logger, "ORCHESTRATOR SHUTTING DOWN", level=logging.DEBUG)

        logger.debug("ShutdownPhase n=1 | step=signal_strategies")
        self.shared_state["shutdown_flag"] = True
        time.sleep(0.5)

        logger.debug("ShutdownPhase n=2 | step=wait_process_exit")
        self._drain_strategy_processes()

        logger.debug("ShutdownPhase n=3 | step=verify_managed_items")
        self._verify_and_close_remaining()

        logger.debug("ShutdownPhase n=4 | step=close_mt5")
        if self.connection:
            self.connection.disconnect()
        logger.debug("ShutdownMT5 conn=closed")

        logger.debug("ShutdownPhase n=5 | step=reset_trade_counter")
        with self.global_trade_count.get_lock():
            self.global_trade_count.value = 0

        logger.debug("ShutdownPhase n=6 | step=close_trade_id_db")
        self.trade_id_manager.close()

        log_section_header(logger, "SHUTDOWN COMPLETE", level=logging.DEBUG)

    def _force_close_all_immediate(
        self,
        managed_positions: list[PositionCacheEntry] | None = None,
        managed_orders: list[OrderSnapshot] | None = None,
    ) -> None:
        magic_numbers = self._get_magic_numbers()
        if managed_positions is None:
            managed_positions = self.position_repo.get_managed_positions(magic_numbers) or []
        if managed_orders is None:
            managed_orders = self.position_repo.get_managed_orders(magic_numbers) or []
        if not managed_positions and not managed_orders:
            logger.debug("ForceCloseSkip reason=no_managed_items")
            return

        executor = self.executor or OrderExecutor(self.account_config.broker_tz)

        if managed_positions:
            tickets = [int(p["ticket"]) for p in managed_positions]
            results = executor.close_positions(tickets=tickets)
            closed = sum(success for success, _ in results.values())
            logger.info(f"ForceClosePos ok={closed}/{len(tickets)}")

        if managed_orders:
            tickets = [o.ticket for o in managed_orders]
            cancelled = sum(1 for t in tickets if executor.cancel_order(t))
            logger.info(f"ForceCancelOrd ok={cancelled}/{len(tickets)}")
