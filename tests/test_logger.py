"""
Integration tests for TradeLogger — real SQLite, mocked MT5 symbol info and deals.

Math verification section at the bottom tests every computed column:
  RRR · gross_pnl · net_pnl · entry_spread · exit_spread · slippage_cost
"""

from __future__ import annotations

import sqlite3
import sys
import time
from collections import namedtuple
from pathlib import Path

import pytest

from nexus_trade.core.state import PositionCacheEntry
from nexus_trade.logging.logger import CloseData, FillData, TradeLogger

#  Fixtures

STRATEGY_TZ = "UTC"
SYMBOL = "EURUSD"

# EURUSD symbol spec values used in slippage formula
TICK_SIZE = 0.00001
TICK_VALUE = 1.0  # $1 per pip per standard lot
CONTRACT_SIZE = 100_000.0

TradeDeal = namedtuple(
    "TradeDeal",
    [
        "ticket",
        "order",
        "time",
        "time_msc",
        "type",
        "entry",
        "magic",
        "position_id",
        "reason",
        "volume",
        "price",
        "commission",
        "swap",
        "profit",
        "fee",
        "symbol",
        "comment",
        "external_id",
    ],
)


def _make_deal(
    *,
    ticket: int = 2001,
    entry: int = 1,  # DEAL_ENTRY_OUT
    pos_type: int = 1,  # SELL (closing a BUY)
    price: float = 1.11000,
    commission: float = -3.50,
    swap: float = 0.0,
    profit: float = 100.0,
    position_id: int = 100_001,
) -> TradeDeal:
    return TradeDeal(
        ticket=ticket,
        order=1001,
        time=int(time.time()),
        time_msc=int(time.time() * 1000),
        type=pos_type,
        entry=entry,
        magic=12345,
        position_id=position_id,
        reason=0,
        volume=0.10,
        price=price,
        commission=commission,
        swap=swap,
        profit=profit,
        fee=0.0,
        symbol=SYMBOL,
        comment="",
        external_id="",
    )


SymbolInfoNT = namedtuple(
    "SymbolInfo",
    [
        "name",
        "description",
        "trade_contract_size",
        "point",
        "digits",
        "volume_min",
        "volume_max",
        "volume_step",
        "bid",
        "ask",
        "spread",
        "spread_float",
        "trade_tick_size",
        "trade_tick_value",
        "trade_tick_value_profit",
        "trade_tick_value_loss",
        "currency_base",
        "currency_profit",
        "currency_margin",
        "trade_mode",
        "filling_mode",
        "trade_stops_level",
        "trade_freeze_level",
        "swap_long",
        "swap_short",
        "swap_mode",
        "time",
    ],
)

EURUSD_SPEC = SymbolInfoNT(
    name=SYMBOL,
    description="Euro vs US Dollar",
    trade_contract_size=CONTRACT_SIZE,
    point=0.00001,
    digits=5,
    volume_min=0.01,
    volume_max=100.0,
    volume_step=0.01,
    bid=1.10000,
    ask=1.10002,
    spread=2,
    spread_float=True,
    trade_tick_size=TICK_SIZE,
    trade_tick_value=TICK_VALUE,
    trade_tick_value_profit=TICK_VALUE,
    trade_tick_value_loss=TICK_VALUE,
    currency_base="EUR",
    currency_profit="USD",
    currency_margin="EUR",
    trade_mode=4,
    filling_mode=1,
    trade_stops_level=20,
    trade_freeze_level=0,
    swap_long=-0.5,
    swap_short=0.3,
    swap_mode=0,
    time=1_700_000_000,
)


def _configure_mt5(
    mt5_mock,
    *,
    entry_deal: TradeDeal | None = None,
    exit_deal: TradeDeal | None = None,
    extra_deals: list[TradeDeal] | None = None,
) -> None:
    mt5_mock.symbol_info.return_value = EURUSD_SPEC
    deals = []
    if entry_deal:
        deals.append(entry_deal)
    if exit_deal:
        deals.append(exit_deal)
    if extra_deals:
        deals.extend(extra_deals)
    mt5_mock.history_deals_get.return_value = tuple(deals) if deals else ()


@pytest.fixture
def logger_root(tmp_path: Path) -> Path:
    return tmp_path / "logs" / "trades"


@pytest.fixture
def trade_logger(logger_root: Path) -> TradeLogger:
    return TradeLogger(log_root=logger_root, strategy_name="test_strategy", strategy_tz=STRATEGY_TZ)


def _make_entry(
    ticket: int = 100_001,
    pos_type: int = 0,
    price_open: float = 1.10000,
    sl: float = 1.09500,
    tp: float = 1.11000,
    volume: float = 0.10,
    magic: int = 12345,
) -> PositionCacheEntry:
    return PositionCacheEntry(
        ticket=ticket,
        symbol=SYMBOL,
        type=pos_type,
        volume=volume,
        price_open=price_open,
        sl=sl,
        tp=tp,
        profit=0.0,
        swap=0.0,
        magic=magic,
        time=int(time.time()),
    )


def _query(logger: TradeLogger, trade_id: int, partial_seq: int = 0) -> dict | None:
    conn = logger._get_connection()
    row = conn.execute(
        "SELECT * FROM trades WHERE trade_id=? AND partial_sequence=?",
        (trade_id, partial_seq),
    ).fetchone()
    if row is None:
        return None
    cols = [d[0] for d in conn.execute("SELECT * FROM trades LIMIT 0").description]
    return dict(zip(cols, row))


#  Schema


class TestSchema:
    def test_table_created(self, trade_logger: TradeLogger) -> None:
        conn = trade_logger._get_connection()
        tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        names = {row[0] for row in tables}
        assert "trades" in names

    def test_wal_mode_enabled(self, trade_logger: TradeLogger) -> None:
        conn = trade_logger._get_connection()
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        assert mode == "wal"

    def test_ticket_index_exists(self, trade_logger: TradeLogger) -> None:
        conn = trade_logger._get_connection()
        indexes = conn.execute("SELECT name FROM sqlite_master WHERE type='index'").fetchall()
        names = {row[0] for row in indexes}
        assert "idx_ticket" in names

    def test_composite_primary_key(self, trade_logger: TradeLogger) -> None:
        """Inserting duplicate (trade_id, partial_sequence) must fail."""
        conn = trade_logger._get_connection()
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO trades (trade_id, partial_sequence, entry_date, entry_time, "
                "magic_number, strategy_name, symbol, size, entry_price, "
                "expected_entry_price, entry_spread, position_type) "
                "VALUES (1, 0, '2025-01-01', '00:00:00', 1, 'x', 'x', 1, 1, 1, 0, 'BUY')"
            )
            conn.execute(
                "INSERT INTO trades (trade_id, partial_sequence, entry_date, entry_time, "
                "magic_number, strategy_name, symbol, size, entry_price, "
                "expected_entry_price, entry_spread, position_type) "
                "VALUES (1, 0, '2025-01-01', '00:00:00', 1, 'x', 'x', 1, 1, 1, 0, 'BUY')"
            )
            conn.commit()


#  log_fill


class TestLogFill:
    def test_inserts_row(self, trade_logger: TradeLogger, mt5_mock) -> None:
        _configure_mt5(mt5_mock)
        pos = _make_entry()
        trade_logger.log_fill(
            FillData(
                trade_id=1,
                position=pos,
                expected_entry_price=1.10000,
                strategy_name="test_strategy",
                opening_sl=1.09500,
            )
        )
        row = _query(trade_logger, 1)
        assert row is not None

    def test_position_type_buy(self, trade_logger: TradeLogger, mt5_mock) -> None:
        _configure_mt5(mt5_mock)
        pos = _make_entry(pos_type=0)
        trade_logger.log_fill(
            FillData(
                trade_id=1,
                position=pos,
                expected_entry_price=1.10,
                strategy_name="test_strategy",
                opening_sl=1.09,
            )
        )
        row = _query(trade_logger, 1)
        assert row["position_type"] == "BUY"

    def test_position_type_sell(self, trade_logger: TradeLogger, mt5_mock) -> None:
        _configure_mt5(mt5_mock)
        pos = _make_entry(pos_type=1)
        trade_logger.log_fill(
            FillData(
                trade_id=1,
                position=pos,
                expected_entry_price=1.10,
                strategy_name="test_strategy",
                opening_sl=1.10500,
            )
        )
        row = _query(trade_logger, 1)
        assert row["position_type"] == "SELL"
        assert row["size"] == pytest.approx(-0.10)  # SELL → negative size

    def test_buy_size_positive(self, trade_logger: TradeLogger, mt5_mock) -> None:
        _configure_mt5(mt5_mock)
        pos = _make_entry(pos_type=0, volume=0.10)
        trade_logger.log_fill(
            FillData(
                trade_id=1,
                position=pos,
                expected_entry_price=1.10,
                strategy_name="test_strategy",
            )
        )
        row = _query(trade_logger, 1)
        assert row["size"] == pytest.approx(0.10)

    def test_exit_fields_null_after_fill(self, trade_logger: TradeLogger, mt5_mock) -> None:
        _configure_mt5(mt5_mock)
        pos = _make_entry()
        trade_logger.log_fill(
            FillData(
                trade_id=1,
                position=pos,
                expected_entry_price=1.10,
                strategy_name="test_strategy",
            )
        )
        row = _query(trade_logger, 1)
        assert row["exit_date"] is None
        assert row["exit_time"] is None
        assert row["exit_price"] is None
        assert row["gross_pnl"] is None
        assert row["net_pnl"] is None

    def test_entry_spread_computed(self, trade_logger: TradeLogger, mt5_mock) -> None:
        """entry_spread = price_open - expected_entry_price."""
        _configure_mt5(mt5_mock)
        pos = _make_entry(price_open=1.10005)
        trade_logger.log_fill(
            FillData(
                trade_id=1,
                position=pos,
                expected_entry_price=1.10000,
                strategy_name="test_strategy",
            )
        )
        row = _query(trade_logger, 1)
        assert row["entry_spread"] == pytest.approx(0.00005, abs=1e-8)

    def test_fill_time_ms_stored(self, trade_logger: TradeLogger, mt5_mock) -> None:
        _configure_mt5(mt5_mock)
        pos = _make_entry()
        trade_logger.log_fill(
            FillData(
                trade_id=1,
                position=pos,
                expected_entry_price=1.10,
                strategy_name="test_strategy",
                fill_time_ms=123.456,
            )
        )
        row = _query(trade_logger, 1)
        # Stored as fill_time_ms / 1000 = 0.123456
        assert row["fill_time_mseconds"] == pytest.approx(0.123456, rel=1e-4)

    def test_volume_multiplier_stored(self, trade_logger: TradeLogger, mt5_mock) -> None:
        _configure_mt5(mt5_mock)
        pos = _make_entry()
        trade_logger.log_fill(
            FillData(
                trade_id=1,
                position=pos,
                expected_entry_price=1.10,
                strategy_name="test_strategy",
                volume_multiplier=0.75,
            )
        )
        row = _query(trade_logger, 1)
        assert row["volume_multiplier"] == pytest.approx(0.75)


#  log_close


class TestLogClose:
    def _fill_and_close(
        self,
        trade_logger: TradeLogger,
        mt5_mock,
        *,
        trade_id: int = 1,
        ticket: int = 100_001,
        entry: float = 1.10000,
        expected_entry: float = 1.10000,
        sl: float = 1.09500,
        tp: float = 1.11000,
        exit_price: float = 1.11000,
        expected_exit: float = 1.11000,
        gross_pnl: float = 100.0,
        commission: float = -3.50,
    ) -> dict:
        exit_deal = _make_deal(
            entry=1,  # DEAL_ENTRY_OUT
            pos_type=1,  # SELL
            price=exit_price,
            commission=commission,
            profit=gross_pnl,
            position_id=ticket,
        )
        _configure_mt5(mt5_mock, exit_deal=exit_deal)

        pos = _make_entry(ticket=ticket, price_open=entry, sl=sl, tp=tp)
        trade_logger.log_fill(
            FillData(
                trade_id=trade_id,
                position=pos,
                expected_entry_price=expected_entry,
                strategy_name="test_strategy",
                opening_sl=sl,
            )
        )
        trade_logger.log_close(
            CloseData(
                trade_id=trade_id,
                position=pos,
                expected_exit_price=expected_exit,
                opening_sl=sl,
                exit_trigger="TP",
                entry_price=entry,
                expected_entry_price=expected_entry,
            )
        )
        return _query(trade_logger, trade_id)

    def test_updates_exit_fields(self, trade_logger: TradeLogger, mt5_mock) -> None:
        row = self._fill_and_close(trade_logger, mt5_mock)
        assert row["exit_price"] == pytest.approx(1.11000)
        assert row["exit_trigger"] == "TP"
        assert row["exit_date"] is not None

    def test_gross_pnl_from_deal(self, trade_logger: TradeLogger, mt5_mock) -> None:
        row = self._fill_and_close(trade_logger, mt5_mock, gross_pnl=100.0)
        assert row["gross_pnl"] == pytest.approx(100.0)

    def test_net_pnl_equals_gross_plus_commission(self, trade_logger: TradeLogger, mt5_mock) -> None:
        row = self._fill_and_close(trade_logger, mt5_mock, gross_pnl=100.0, commission=-3.50)
        assert row["net_pnl"] == pytest.approx(96.50)

    def test_swap_stored(self, trade_logger: TradeLogger, mt5_mock) -> None:
        """Swap comes from PositionCacheEntry.swap, not the deal."""
        pos_with_swap: PositionCacheEntry = {
            "ticket": 100_001,
            "symbol": SYMBOL,
            "type": 0,
            "volume": 0.10,
            "price_open": 1.10000,
            "sl": 1.09500,
            "tp": 1.11000,
            "profit": 0.0,
            "swap": -1.50,
            "magic": 12345,
            "time": 0,
        }
        exit_deal = _make_deal(profit=100.0, commission=-3.50)
        _configure_mt5(mt5_mock, exit_deal=exit_deal)
        trade_logger.log_fill(
            FillData(
                trade_id=1,
                position=pos_with_swap,
                expected_entry_price=1.10,
                strategy_name="test_strategy",
            )
        )
        trade_logger.log_close(
            CloseData(
                trade_id=1,
                position=pos_with_swap,
                expected_exit_price=1.11,
                opening_sl=1.095,
                exit_trigger="TP",
                entry_price=1.10,
                expected_entry_price=1.10,
            )
        )
        row = _query(trade_logger, 1)
        assert row["swap"] == pytest.approx(-1.50)

    def test_skip_on_missing_exit_deal(self, trade_logger: TradeLogger, mt5_mock) -> None:
        mt5 = sys.modules["MetaTrader5"]
        # Only ENTRY deal, no EXIT deal
        entry_deal = _make_deal(entry=0)  # DEAL_ENTRY_IN
        mt5.history_deals_get.return_value = (entry_deal,)
        mt5.symbol_info.return_value = EURUSD_SPEC

        pos = _make_entry()
        trade_logger.log_fill(
            FillData(
                trade_id=1,
                position=pos,
                expected_entry_price=1.10,
                strategy_name="test_strategy",
            )
        )
        trade_logger.log_close(
            CloseData(
                trade_id=1,
                position=pos,
                expected_exit_price=1.11,
                opening_sl=1.095,
                exit_trigger="TP",
                entry_price=1.10,
                expected_entry_price=1.10,
            )
        )
        row = _query(trade_logger, 1)
        # Row should exist but exit fields null (close was skipped)
        assert row["exit_price"] is None


#  Math verification


class TestMathVerification:
    """
    Exact expected-value checks for every computed column.

    Trade setup:
      BUY EURUSD 0.10 lot
      entry_price  = 1.10000   (no slippage)
      sl           = 1.09500   (50-pip SL)
      tp           = 1.11000   (100-pip TP)
      exit_price   = 1.11000   (hit TP)
      gross_pnl    = 100.0 USD (from MT5 deal)
      commission   = −3.50 USD
      swap         = 0.0

    EURUSD spec: tick_size=0.00001, tick_value=1.0, contract_size=100,000
    """

    @pytest.fixture(autouse=True)
    def setup(self, trade_logger: TradeLogger, mt5_mock) -> None:
        self.logger = trade_logger
        exit_deal = _make_deal(
            entry=1,
            pos_type=1,
            price=1.11000,
            commission=-3.50,
            profit=100.0,
        )
        _configure_mt5(mt5_mock, exit_deal=exit_deal)

        pos = _make_entry(price_open=1.10000, sl=1.09500, tp=1.11000)
        self.logger.log_fill(
            FillData(
                trade_id=1,
                position=pos,
                expected_entry_price=1.10000,
                strategy_name="test_strategy",
                opening_sl=1.09500,
            )
        )
        self.logger.log_close(
            CloseData(
                trade_id=1,
                position=pos,
                expected_exit_price=1.11000,
                opening_sl=1.09500,
                exit_trigger="TP",
                entry_price=1.10000,
                expected_entry_price=1.10000,
            )
        )
        self.row = _query(self.logger, 1)

    def test_rrr_is_two(self) -> None:
        """
        RRR = (exit - entry) / (entry - sl)
            = (1.11000 - 1.10000) / (1.10000 - 1.09500)
            = 0.01000 / 0.00500 = 2.0
        """
        assert self.row["rrr"] == pytest.approx(2.0, rel=1e-6)

    def test_gross_pnl(self) -> None:
        assert self.row["gross_pnl"] == pytest.approx(100.0)

    def test_net_pnl(self) -> None:
        """net_pnl = gross_pnl + commission = 100.0 + (−3.50) = 96.50"""
        assert self.row["net_pnl"] == pytest.approx(96.50)

    def test_entry_spread_zero_no_slippage(self) -> None:
        """entry_spread = price_open − expected_entry = 1.10000 − 1.10000 = 0.0"""
        assert self.row["entry_spread"] == pytest.approx(0.0, abs=1e-8)

    def test_exit_spread_zero_no_slippage(self) -> None:
        """exit_spread (directional, BUY) = +1 × (expected − actual) = 0.0"""
        assert self.row["exit_spread"] == pytest.approx(0.0, abs=1e-8)

    def test_slippage_cost_zero_no_slippage(self) -> None:
        assert self.row["slippage_cost"] == pytest.approx(0.0, abs=1e-8)

    def test_opening_sl_stored(self) -> None:
        assert self.row["opening_sl"] == pytest.approx(1.09500)

    def test_tp_stored(self) -> None:
        assert self.row["tp"] == pytest.approx(1.11000)

    def test_entry_price_stored(self) -> None:
        assert self.row["entry_price"] == pytest.approx(1.10000)

    def test_commission_stored(self) -> None:
        assert self.row["commission"] == pytest.approx(-3.50)

    def test_exit_trigger(self) -> None:
        assert self.row["exit_trigger"] == "TP"


class TestRRRFormula:
    """Parametric RRR verification for BUY and SELL directions."""

    @pytest.mark.parametrize(
        "entry,sl,exit_,pos_type,expected_rrr",
        [
            # BUY: 1:2 trade
            (1.10000, 1.09500, 1.11000, 0, 2.0),
            # BUY: 1:1 trade
            (1.10000, 1.09500, 1.10500, 0, 1.0),
            # BUY: losing trade (−0.5R)
            (1.10000, 1.09500, 1.09750, 0, -0.5),
            # SELL: 1:2 trade
            (1.10000, 1.10500, 1.09000, 1, 2.0),
            # SELL: losing trade
            (1.10000, 1.10500, 1.10250, 1, -0.5),
        ],
    )
    def test_rrr(
        self,
        trade_logger: TradeLogger,
        entry: float,
        sl: float,
        exit_: float,
        pos_type: int,
        expected_rrr: float,
    ) -> None:
        rrr = trade_logger._calculate_rrr(entry, exit_, sl, pos_type)
        assert rrr == pytest.approx(expected_rrr, rel=1e-5)

    def test_zero_risk_returns_zero(self, trade_logger: TradeLogger) -> None:
        """Sl == entry → risk = 0 → return 0 not ZeroDivisionError."""
        rrr = trade_logger._calculate_rrr(1.10, 1.11, 1.10, 0)
        assert rrr == pytest.approx(0.0)


class TestSlippageCostFormula:
    """
    slippage_cost = direction_multiplier × spread × volume × (tick_value / tick_size)

    direction_multiplier:
      BUY  (type=0) → −1  (positive spread = we paid more → negative cost)
      SELL (type=1) → +1
    """

    @pytest.mark.parametrize(
        "pos_type,spread,volume,expected",
        [
            # BUY, 5-pip entry slippage
            (0, 0.00005, 0.10, -0.5),
            # SELL, 5-pip entry slippage
            (1, 0.00005, 0.10, 0.5),
            # BUY, no slippage
            (0, 0.0, 0.10, 0.0),
            # BUY, 1-pip, 1.0 lot
            (0, 0.00010, 1.0, -10.0),
        ],
    )
    def test_slippage_formula(
        self,
        trade_logger: TradeLogger,
        pos_type: int,
        spread: float,
        volume: float,
        expected: float,
        mt5_mock,
    ) -> None:
        mt5 = sys.modules["MetaTrader5"]
        mt5.symbol_info.return_value = EURUSD_SPEC
        cost = trade_logger._calculate_slippage_cost(SYMBOL, volume, pos_type, spread)
        assert cost == pytest.approx(expected, abs=1e-8)


class TestExitSpreadFormula:
    """
    exit_spread (directional):
      direction = +1 for BUY (type=0), −1 for SELL (type=1)
      = direction × (expected − actual)
    """

    @pytest.mark.parametrize(
        "actual,expected_p,pos_type,result",
        [
            (1.11000, 1.11000, 0, 0.0),  # BUY, no slippage
            (1.10995, 1.11000, 0, 0.00005),  # BUY, 0.5-pip unfavorable
            (1.11005, 1.11000, 0, -0.00005),  # BUY, 0.5-pip favorable
            (1.09000, 1.09000, 1, 0.0),  # SELL, no slippage
            (1.09005, 1.09000, 1, 0.00005),  # SELL, 0.5-pip unfavorable
        ],
    )
    def test_exit_spread(
        self,
        trade_logger: TradeLogger,
        actual: float,
        expected_p: float,
        pos_type: int,
        result: float,
    ) -> None:
        spread = trade_logger._calculate_exit_spread(actual, expected_p, pos_type)
        assert spread == pytest.approx(result, abs=1e-8)

    def test_none_expected_returns_none(self, trade_logger: TradeLogger) -> None:
        assert trade_logger._calculate_exit_spread(1.1, None) is None


#  get_open_trades_by_ticket_last_three (reconciliation)


class TestGetOpenTradesByTicket:
    def test_returns_open_trade(self, trade_logger: TradeLogger, mt5_mock) -> None:
        _configure_mt5(mt5_mock)
        pos = _make_entry(ticket=55555)
        trade_logger.log_fill(
            FillData(
                trade_id=7,
                position=pos,
                expected_entry_price=1.10,
                strategy_name="test_strategy",
                opening_sl=1.095,
            )
        )
        result = trade_logger.get_open_trades_by_ticket_last_three([55555])
        assert 55555 in result
        assert result[55555]["trade_id"] == 7

    def test_returns_empty_for_closed_trade(self, trade_logger: TradeLogger, mt5_mock) -> None:
        exit_deal = _make_deal()
        _configure_mt5(mt5_mock, exit_deal=exit_deal)
        pos = _make_entry(ticket=66666)
        trade_logger.log_fill(
            FillData(
                trade_id=8,
                position=pos,
                expected_entry_price=1.10,
                strategy_name="test_strategy",
            )
        )
        trade_logger.log_close(
            CloseData(
                trade_id=8,
                position=pos,
                expected_exit_price=1.11,
                opening_sl=1.095,
                exit_trigger="TP",
                entry_price=1.10,
                expected_entry_price=1.10,
            )
        )
        result = trade_logger.get_open_trades_by_ticket_last_three([66666])
        assert 66666 not in result

    def test_empty_tickets_returns_empty(self, trade_logger: TradeLogger) -> None:
        assert trade_logger.get_open_trades_by_ticket_last_three([]) == {}

    def test_unknown_ticket_not_in_result(self, trade_logger: TradeLogger, mt5_mock) -> None:
        _configure_mt5(mt5_mock)
        result = trade_logger.get_open_trades_by_ticket_last_three([999999])
        assert 999999 not in result
