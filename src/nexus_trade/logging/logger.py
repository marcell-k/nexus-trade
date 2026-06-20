"""
TradeLogger - SQLite-based trade execution logging with atomic updates.

Schema:
    - Single row per trade (updated on exit)
    - Partial closes create new rows with incremented partial_id
    - ACID transactions prevent data corruption
    - Indexed on ticket for O(log n) lookups
"""

import logging
import sqlite3
import threading
import time
from collections.abc import Generator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import MetaTrader5 as mt
import pandas as pd
from MetaTrader5 import TradeDeal

from nexus_trade.core.constants import MT5_DEAL_ENTRY_OUT
from nexus_trade.core.models import NormalizedPosition
from nexus_trade.core.symbol import SYMBOL_SPEC_CACHE, SymbolSpec
from nexus_trade.core.types import PartialClosePositionSnapshot, PositionCacheEntry, PositionType, ReconciledTrade
from nexus_trade.utils.format import format_price_display

logger = logging.getLogger(__name__)


@dataclass
class FillData:
    """Trade fill parameters."""

    trade_id: int
    position: PositionCacheEntry
    expected_entry_price: float
    strategy_name: str
    opening_sl: float | None = None
    fill_time_ms: float | None = None
    volume_multiplier: float | None = None


@dataclass(slots=True)
class CloseData:
    """Trade close parameters."""

    trade_id: int
    position: PositionCacheEntry
    expected_exit_price: float | None
    opening_sl: float | None
    exit_trigger: str
    entry_price: float
    expected_entry_price: float | None


@dataclass(slots=True)
class PartialCloseData:
    """Partial close parameters."""

    trade_id: int
    position: PartialClosePositionSnapshot
    closed_volume: float
    remaining_volume: float
    expected_exit_price: float | None
    opening_sl: float
    strategy_name: str
    exit_trigger: str
    entry_price: float
    expected_entry_price: float
    deal_id: int | None = None


class TradeLogger:
    """SQLite-based trade logger with single-row-per-trade design."""

    def __init__(self, log_root: Path, strategy_name: str, strategy_tz: str) -> None:
        """Initialize SQLite logger for a specific strategy."""
        self.log_root: Path = Path(log_root)
        self.log_root.mkdir(parents=True, exist_ok=True)

        self.db_path: Path = self.log_root / f"trades_{strategy_name}.db"
        self.strategy_name: str = strategy_name
        self.strategy_tz: ZoneInfo = ZoneInfo(strategy_tz)
        self._local: threading.local = threading.local()
        self._symbol_info_lock: threading.Lock = threading.Lock()

        self._init_database()
        logger.debug(f"TradeLogInit db={self.db_path}")

    def _get_connection(self) -> sqlite3.Connection:
        """Get thread-local database connection."""
        if not hasattr(self._local, "conn"):
            self._local.conn = sqlite3.connect(str(self.db_path), check_same_thread=False, timeout=10.0)
            _ = self._local.conn.execute("PRAGMA foreign_keys = ON")
            _ = self._local.conn.execute("PRAGMA journal_mode = WAL")
        return self._local.conn

    def _init_database(self) -> None:
        """Create trades table with composite primary key."""
        conn = self._get_connection()

        _ = conn.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                trade_id INTEGER NOT NULL,
                partial_sequence INTEGER NOT NULL DEFAULT 0,
                ticket INTEGER,
                entry_date TEXT NOT NULL,
                entry_time TEXT NOT NULL,
                exit_date TEXT,
                exit_time TEXT,
                magic_number INTEGER NOT NULL,
                strategy_name TEXT NOT NULL,
                symbol TEXT NOT NULL,
                size REAL NOT NULL,
                entry_price REAL NOT NULL,
                expected_entry_price REAL NOT NULL,
                entry_spread REAL NOT NULL,
                exit_price REAL,
                expected_exit_price REAL,
                exit_spread REAL,
                opening_sl REAL,
                tp REAL,
                rrr REAL,
                commission REAL,
                swap REAL,
                gross_pnl REAL,
                net_pnl REAL,
                slippage_cost REAL,
                fill_time_mseconds REAL,
                position_type TEXT NOT NULL,
                exit_trigger TEXT,
                volume_multiplier REAL,
                PRIMARY KEY (trade_id, partial_sequence)
            )
        """)

        _ = conn.execute("CREATE INDEX IF NOT EXISTS idx_ticket ON trades(ticket)")
        _ = conn.execute("CREATE INDEX IF NOT EXISTS idx_strategy_date ON trades(strategy_name, entry_date)")
        conn.commit()

        logger.debug(f"TradeLogSchema db={self.db_path}")

    def get_open_trades_by_ticket_last_three(self, tickets: list[int]) -> dict[int, ReconciledTrade]:
        """
        Return open trade metadata for tickets using strict latest-3 row reconciliation.

        Open row rule: exit_time IS NULL.
        Scope rule: only the three most recent rows per ticket (trade_id DESC, partial_sequence DESC).
        """
        if not tickets:
            return {}

        conn = self._get_connection()
        placeholders = ",".join("?" for _ in tickets)
        query = f"""
            SELECT
                ticket,
                trade_id,
                expected_entry_price,
                opening_sl,
                volume_multiplier,
                exit_time
            FROM (
                SELECT
                    ticket,
                    trade_id,
                    expected_entry_price,
                    opening_sl,
                    volume_multiplier,
                    exit_time,
                    ROW_NUMBER() OVER (
                        PARTITION BY ticket
                        ORDER BY trade_id DESC, partial_sequence DESC
                    ) AS recency_rank
                FROM trades
                WHERE ticket IN ({placeholders})
            )
            WHERE recency_rank <= 3
            ORDER BY ticket ASC, trade_id DESC, recency_rank ASC
        """  # noqa: S608

        try:
            rows = conn.execute(query, tickets).fetchall()
        except Exception as e:
            logger.error(f"OpenReconFail err={e}", exc_info=True)
            raise

        candidates: dict[int, list[tuple[object, ...]]] = {}
        for row in rows:
            ticket = row[0]
            candidates.setdefault(ticket, []).append(row)

        reconciled: dict[int, ReconciledTrade] = {}
        for ticket, ticket_rows in candidates.items():
            open_rows = [row for row in ticket_rows if row[5] is None]
            if not open_rows:
                continue
            if len(open_rows) > 1:
                logger.warning(
                    f"OpenReconWarn t={ticket} | open={len(open_rows)} | "
                    f"reason=open rows in latest 3 | use_id={open_rows[0][1]}"
                )

            selected = open_rows[0]
            _ticket, trade_id, expected_entry_price, opening_sl, volume_multiplier, _ = selected
            assert isinstance(trade_id, int)
            assert isinstance(expected_entry_price, float | int)
            reconciled[ticket] = ReconciledTrade(
                trade_id=int(trade_id),
                expected_entry_price=float(expected_entry_price),
                opening_sl=float(opening_sl) if isinstance(opening_sl, float | int) else None,
                volume_multiplier=float(volume_multiplier) if isinstance(volume_multiplier, float | int) else None,
            )
        return reconciled

    @contextmanager
    def _transact(self, operation_id: str) -> Generator[sqlite3.Connection]:
        """Yield an active connection; commit on clean exit, rollback + log on error.

        operation_id is a short label used in error messages (e.g. "fill:42", "close:7").
        """
        conn: sqlite3.Connection = self._get_connection()
        try:
            yield conn
            conn.commit()
        except sqlite3.IntegrityError as exc:
            conn.rollback()
            logger.error(f"DBIntegrity op={operation_id} | err={exc}")
        except Exception as exc:
            conn.rollback()
            logger.error(f"DBFail op={operation_id} | err={exc}", exc_info=True)

    def log_fill(self, data: FillData) -> None:
        """Log position fill with unique trade_id."""
        entry_date, entry_time = self._format_datetime(datetime.now(tz=self.strategy_tz))

        size = data.position["volume"] if data.position["type"] == PositionType.BUY else -data.position["volume"]
        entry_spread = data.position["price_open"] - data.expected_entry_price
        slippage_cost = self._calculate_slippage_cost(
            data.position["symbol"], data.position["volume"], data.position["type"], entry_spread
        )
        fill_time_mseconds = data.fill_time_ms if data.fill_time_ms is not None else None

        with self._transact(f"fill:{data.trade_id}") as conn:
            _ = conn.execute(
                """
                INSERT INTO trades (
                    trade_id, partial_sequence, ticket,
                    entry_date, entry_time,
                    magic_number, strategy_name, symbol, size,
                    entry_price, expected_entry_price, entry_spread,
                    opening_sl, tp,
                    slippage_cost, fill_time_mseconds,
                    position_type, volume_multiplier
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    data.trade_id,
                    0,
                    data.position["ticket"],
                    entry_date,
                    entry_time,
                    data.position["magic_number"],
                    data.strategy_name,
                    data.position["symbol"],
                    size,
                    data.position["price_open"],
                    data.expected_entry_price,
                    entry_spread,
                    data.opening_sl,
                    data.position["tp"] if data.position["tp"] != 0.0 else None,
                    slippage_cost,
                    fill_time_mseconds,
                    "BUY" if data.position["type"] == PositionType.BUY else "SELL",
                    data.volume_multiplier,
                ),
            )
        entry_price_display = format_price_display(data.position["price_open"])
        logger.info(
            f"FillLog id={data.trade_id} | sym={data.position['symbol']} | sz={size:+.2f} | px={entry_price_display}"
        )

    def log_close(self, data: CloseData) -> None:
        """Log position close (updates existing trade_id row)."""
        position_deals = self._get_position_deals(data.position["ticket"], "Exit deal")
        if position_deals is None:
            return

        deal_data = self._get_exit_deal_data(data.position["ticket"], deals=position_deals)
        if deal_data is None:
            return

        actual_exit_price: float = deal_data["exit_price"]
        gross_pnl: float = deal_data["gross_pnl"]

        if data.expected_exit_price is not None:
            expected_exit_price = data.expected_exit_price
        else:
            expected_exit_price = self._infer_expected_exit_price_from_cache(data.position, actual_exit_price)

        raw_exit_spread = self._calculate_exit_spread(actual_exit_price, data.expected_exit_price, position_type=None)
        exit_spread = self._calculate_exit_spread(actual_exit_price, expected_exit_price, data.position["type"])
        commission = self._get_commission(None, deals=position_deals)
        net_pnl = gross_pnl + commission
        rrr = self._calculate_rrr(data.entry_price, actual_exit_price, data.opening_sl, data.position["type"])

        slippage_cost = self._calculate_slippage_cost(
            data.position["symbol"],
            data.position["volume"],
            data.position["type"],
            raw_exit_spread if raw_exit_spread is not None else 0.0,
        )

        exit_date, exit_time = self._format_datetime(datetime.now(tz=self.strategy_tz))

        conn = self._get_connection()

        with self._transact(f"close:{data.trade_id}"):
            before_changes = conn.total_changes
            _ = conn.execute(
                """
                UPDATE trades
                SET exit_date = ?, exit_time = ?, exit_price = ?, expected_exit_price = ?,
                    exit_spread = ?, rrr = ?, commission = ?, swap = ?,
                    gross_pnl = ?, net_pnl = ?, slippage_cost = ?, exit_trigger = ?
                WHERE trade_id = ? AND partial_sequence = 0 AND exit_time IS NULL
            """,
                (
                    exit_date,
                    exit_time,
                    actual_exit_price,
                    expected_exit_price,
                    exit_spread,
                    rrr,
                    commission,
                    data.position["swap"],
                    gross_pnl,
                    net_pnl,
                    slippage_cost,
                    data.exit_trigger,
                    data.trade_id,
                ),
            )

            if conn.total_changes == before_changes:
                logger.warning(
                    f"CloseLogSkip id={data.trade_id} | t={data.position['ticket']} | reason=no_open_entry_row"
                )
                return

        rrr_display = f"{rrr:.2f}" if rrr is not None else "NA"
        slippage_display = f"{slippage_cost:.2f}" if slippage_cost is not None else "NA"
        logger.info(
            f"CloseLog id={data.trade_id} | px={format_price_display(actual_exit_price)} | "
            f"pnl={net_pnl:.2f} | rrr={rrr_display} | slip={slippage_display}"
        )

    def log_partial_close(self, data: PartialCloseData) -> None:
        """Log partial position close (creates new row with incremented partial_sequence)."""
        exit_date, exit_time = self._format_datetime(datetime.now(tz=self.strategy_tz))
        position = NormalizedPosition.from_mt5(data.position).to_partial_snapshot()

        read_conn = self._get_connection()
        next_partial = self._get_next_partial_sequence(read_conn, data.trade_id)
        original = self._get_original_entry_data(read_conn, data.trade_id)
        if original is None:
            logger.error(f"PartCloseFail id={data.trade_id} | t={position.ticket} | reason=original_entry_missing")
            return

        deal_data = self._get_latest_partial_exit_deal(position.ticket, data.deal_id)
        if deal_data is None:
            return

        actual_exit_price: float = deal_data["exit_price"]
        partial_gross_pnl: float = deal_data["profit"]

        total_volume = data.closed_volume + data.remaining_volume
        partial_ratio: float = data.closed_volume / total_volume if total_volume > 0 else 0.0

        expected_exit_price = data.expected_exit_price if data.expected_exit_price is not None else None

        raw_exit_spread = self._calculate_exit_spread(actual_exit_price, expected_exit_price, position_type=None)
        exit_spread = self._calculate_exit_spread(actual_exit_price, expected_exit_price, position.type)

        position_deals = self._get_position_deals(position.ticket, "Partial exit deal", entry_filter=MT5_DEAL_ENTRY_OUT)
        partial_commission: float = 0.0
        if position_deals is not None:
            total_commission = self._get_commission(None, position_deals)
            partial_commission = total_commission * partial_ratio

        rrr = self._calculate_rrr(data.entry_price, actual_exit_price, data.opening_sl, position.type)
        net_pnl = partial_gross_pnl + partial_commission

        slippage_cost = self._calculate_slippage_cost(
            position.symbol,
            data.closed_volume,
            position.type,
            raw_exit_spread if raw_exit_spread is not None else 0.0,
        )

        size = data.closed_volume if position.type == 0 else -data.closed_volume

        with self._transact(f"partial_close:{data.trade_id}:{next_partial}") as conn:
            _ = conn.execute(
                """
                INSERT INTO trades (
                    trade_id, partial_sequence, ticket, entry_date, entry_time, exit_date, exit_time,
                    magic_number, strategy_name, symbol, size, entry_price, expected_entry_price, entry_spread,
                    exit_price, expected_exit_price, exit_spread, opening_sl, tp, rrr,
                    commission, swap, gross_pnl, net_pnl, slippage_cost, fill_time_mseconds,
                    position_type, exit_trigger, volume_multiplier
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    data.trade_id,
                    next_partial,
                    position.ticket,
                    original["entry_date"],
                    original["entry_time"],
                    exit_date,
                    exit_time,
                    original["magic_number"],
                    data.strategy_name,
                    position.symbol,
                    size,
                    data.entry_price,
                    original["expected_entry_price"],
                    original["entry_spread"],
                    actual_exit_price,
                    data.expected_exit_price,
                    exit_spread,
                    original["opening_sl"],
                    original["tp"],
                    rrr,
                    partial_commission,
                    position.swap * partial_ratio,
                    partial_gross_pnl,
                    net_pnl,
                    slippage_cost,
                    original["fill_time_mseconds"],
                    "BUY" if position.type == PositionType.BUY else "SELL",
                    data.exit_trigger,
                    original["volume_multiplier"],
                ),
            )

        slippage_display = f"{slippage_cost:.2f}" if slippage_cost is not None else "NA"
        logger.info(
            f"PartClose id={data.trade_id} | p={next_partial} | "
            f"vol={data.closed_volume:.2f} | px={format_price_display(actual_exit_price)} | "
            f"pnl={net_pnl:.2f} | slip={slippage_display}"
        )

    def _format_datetime(self, dt: datetime) -> tuple[str, str]:
        """Format datetime into date and time strings."""
        return dt.strftime("%Y-%m-%d"), dt.strftime("%H:%M:%S")

    def _get_next_partial_sequence(self, conn: sqlite3.Connection, trade_id: int) -> int:
        """Get next partial_sequence number for a trade."""
        cursor = conn.execute("SELECT MAX(partial_sequence) FROM trades WHERE trade_id = ?", (trade_id,))
        max_partial = cursor.fetchone()[0]
        return (max_partial if max_partial is not None else 0) + 1

    def _get_original_entry_data(self, conn: sqlite3.Connection, trade_id: int) -> dict[str, float | None] | None:
        """Fetch original entry data for a trade."""
        cursor = conn.execute(
            """
            SELECT entry_date, entry_time, magic_number, ticket,
                expected_entry_price, entry_spread, opening_sl, tp,
                fill_time_mseconds, volume_multiplier
            FROM trades
            WHERE trade_id = ? AND partial_sequence = 0
        """,
            (trade_id,),
        )

        row = cursor.fetchone()
        if row is None:
            return None

        return {
            "entry_date": row[0],
            "entry_time": row[1],
            "magic_number": row[2],
            "ticket": row[3],
            "expected_entry_price": row[4],
            "entry_spread": row[5],
            "opening_sl": row[6],
            "tp": row[7],
            "fill_time_mseconds": row[8],
            "volume_multiplier": row[9],
        }

    @staticmethod
    def _calculate_exit_spread(
        actual_exit_price: float,
        expected_exit_price: float | None,
        position_type: int | None = None,
    ) -> float | None:
        """Calculate exit spread as expected minus actual."""
        if expected_exit_price is None:
            return None
        raw_spread = expected_exit_price - actual_exit_price
        if position_type is None:
            return raw_spread
        direction_multiplier = 1 if position_type == 0 else -1
        return direction_multiplier * raw_spread

    @staticmethod
    def _normalize_protective_level(level: float | None) -> float | None:
        """Normalize protective level by treating sentinel zeros as missing."""
        if level is None:
            return None
        return None if abs(level) <= 1e-12 else level

    def _infer_expected_exit_price_from_cache(
        self, position: PositionCacheEntry, actual_exit_price: float
    ) -> float | None:
        """Infer expected exit price from position SL/TP if not provided."""
        normalized_sl = self._normalize_protective_level(position.get("sl"))
        normalized_tp = self._normalize_protective_level(position.get("tp"))

        if normalized_sl is not None:
            sl_distance = abs(position["price_open"] - normalized_sl)
            tolerance = max(sl_distance / 10, 1e-5)
        else:
            tolerance = 1e-5

        if position["type"] == 0:  # BUY
            tp_hit = normalized_tp is not None and actual_exit_price >= (normalized_tp - tolerance)
            sl_hit = normalized_sl is not None and actual_exit_price <= (normalized_sl + tolerance)

            if tp_hit:
                return normalized_tp
            if sl_hit:
                return normalized_sl
        else:  # SELL
            tp_hit = normalized_tp is not None and actual_exit_price <= (normalized_tp + tolerance)
            sl_hit = normalized_sl is not None and actual_exit_price >= (normalized_sl - tolerance)

            if tp_hit:
                return normalized_tp
            if sl_hit:
                return normalized_sl

        return None

    def _get_position_deals(
        self,
        ticket: int,
        context: str,
        max_retries: int = 3,
        entry_filter: int | None = None,
    ) -> list[TradeDeal] | None:
        """Retrieve history deals for a ticket with bounded retry backoff."""
        for attempt in range(max_retries):
            deals = mt.history_deals_get(position=ticket)
            if deals and len(deals) > 0:
                deal_list: list[TradeDeal] = list(deals)
                if entry_filter is not None:
                    deal_list = [deal for deal in deals if getattr(deal, "entry", None) == entry_filter]
                return deal_list
            if attempt < max_retries - 1:
                time.sleep(0.1 * (attempt + 1))
        logger.error(f"DealsFetchFail ctx={context} | t={ticket} | reason=no_deals")
        return None

    def _get_latest_partial_exit_deal(self, ticket: int, deal_id: int | None = None) -> dict[str, float] | None:
        """Retrieve partial exit deal by deal_id (required)."""
        if deal_id is None:
            logger.error(f"PartExitFail t={ticket} | reason=Missing deal_id")
            return None
        deal = mt.history_deals_get(ticket=deal_id)
        if deal and len(deal) > 0:
            return {"exit_price": deal[0].price, "profit": deal[0].profit}
        logger.error(f"PartExitFail deal={deal_id} | t={ticket} | reason=deal_not_found")
        return None

    def _resolve_slippage_params(
        self,
        symbol: str,
        symbol_info: SymbolSpec,
        position_type: int,
    ) -> tuple[float, float, int] | None:
        """Resolve tick parameters for directional slippage calculation."""
        tick_size = symbol_info.tick_size
        tick_value = symbol_info.tick_value
        if tick_size <= 0:
            logger.error(f"SlipCalcFail sym={symbol} | reason=invalid_tick_size | tick_size={tick_size}")
        if tick_value <= 0:
            logger.error(f"SlipCalcFail sym={symbol} | reason=invalid_tick_value | tick_value={tick_value}")
        direction_multiplier = -1 if position_type == 0 else 1
        return tick_size, tick_value, direction_multiplier

    def _calculate_slippage_cost(self, symbol: str, volume: float, position_type: int, *spreads: float) -> float | None:
        """Calculate directional slippage cost in account currency."""
        symbol_info = SYMBOL_SPEC_CACHE.get_spec(symbol)
        if symbol_info is None:
            return None
        params = self._resolve_slippage_params(symbol, symbol_info, position_type)
        if params is None:
            return None

        tick_size, tick_value, direction_multiplier = params
        return direction_multiplier * sum(spreads) * volume * (tick_value / tick_size)

    def _get_commission(self, ticket: int | None, deals: Sequence[TradeDeal]) -> float:
        """Query total commission from history deals."""
        return sum(
            float(getattr(deal, "commission", 0.0))
            for deal in deals
            if ticket is None or getattr(deal, "ticket", None) == ticket
        )

    def _get_exit_deal_data(
        self,
        ticket: int,
        deals: Sequence[TradeDeal],
    ) -> dict[str, float] | None:
        """Retrieve exit price and gross P&L from historical trade deals."""
        exit_deals = [deal for deal in deals if getattr(deal, "entry", None) == MT5_DEAL_ENTRY_OUT]
        if not exit_deals:
            logger.error(f"ExitDealFail t={ticket} | reason=no_exit_deals")
            return None

        gross_pnl = sum(float(getattr(d, "profit", 0.0)) for d in exit_deals)

        return {"exit_price": float(exit_deals[-1].price), "gross_pnl": gross_pnl}

    def _calculate_rrr(
        self,
        entry_price: float,
        exit_price: float,
        opening_sl: float | None,
        position_type: int,
    ) -> float | None:
        """Calculate realized risk-reward ratio."""
        if opening_sl is None:
            return None
        try:
            if position_type == 0:  # BUY
                risk = entry_price - opening_sl
                reward = exit_price - entry_price
            else:  # SELL
                risk = opening_sl - entry_price
                reward = entry_price - exit_price

            return 0.0 if risk == 0 else reward / risk

        except ZeroDivisionError as e:
            logger.error(f"RRRFail err={e}", exc_info=True)
            return None

    def export_to_csv(self, output_path: Path | None = None) -> Path:
        """Export database to CSV format for backward compatibility."""
        if output_path is None:
            output_path = self.log_root / f"trades_{self.strategy_name}.csv"

        conn = self._get_connection()

        try:
            df = pd.read_sql_query("SELECT * FROM trades ORDER BY entry_date, entry_time", conn)
            df.to_csv(output_path, index=False)

            logger.info(f"CsvExport rows={len(df)} | path={output_path}")
            return output_path

        except Exception as e:
            logger.error(f"CsvExportFail err={e}", exc_info=True)
            raise

    def close(self) -> None:
        """Close database connection."""
        if hasattr(self._local, "conn"):
            self._local.conn.close()
            logger.info(f"TradeLogClose db={self.db_path}")
