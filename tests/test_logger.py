"""
Integration tests for TradeLogger — real SQLite, mocked MT5.

Math verification covers: RRR, gross_pnl, net_pnl, entry_spread,
exit_spread, slippage_cost.
"""

from __future__ import annotations

import sqlite3
import time
from collections import namedtuple
from typing import TYPE_CHECKING

import pytest

from nexus_trade.core.types import PositionCacheEntry
from nexus_trade.execution.request import CloseData, FillData
from nexus_trade.logging.trade_logger import TradeLogger

if TYPE_CHECKING:
    from pathlib import Path
    from unittest.mock import MagicMock

STRATEGY_TZ = "UTC"
SYMBOL = "EURUSD"
TICK_SIZE = 0.00001
TICK_VALUE = 1.0
CONTRACT_SIZE = 100_000.0

TradeDeal = namedtuple(
    "TradeDeal",
    "ticket order time time_msc type entry magic position_id reason "
    "volume price commission swap profit fee symbol comment external_id",
)

SymbolInfoNT = namedtuple(
    "SymbolInfo",
    "name description trade_contract_size point digits volume_min volume_max volume_step "
    "bid ask spread spread_float trade_tick_size trade_tick_value trade_tick_value_profit "
    "trade_tick_value_loss currency_base currency_profit currency_margin trade_mode "
    "filling_mode trade_stops_level trade_freeze_level swap_long swap_short swap_mode time",
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


def _make_deal(
    *,
    ticket: int = 2001,
    entry: int = 1,
    pos_type: int = 1,
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


def _configure_mt5(
    mt5_mock: MagicMock,
    *,
    exit_deal: TradeDeal | None = None,
    extra_deals: list[TradeDeal] | None = None,
) -> None:
    mt5_mock.symbol_info.return_value = EURUSD_SPEC
    deals = []
    if exit_deal:
        deals.append(exit_deal)
    if extra_deals:
        deals.extend(extra_deals)
    mt5_mock.history_deals_get.return_value = tuple(deals)


def _make_pos(
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
    return dict(zip(cols, row))  # noqa: B905


@pytest.fixture
def trade_logger(tmp_path: Path) -> TradeLogger:
    return TradeLogger(
        log_root=tmp_path / "logs" / "trades",
        strategy_name="test_strategy",
        strategy_tz=STRATEGY_TZ,
    )


class TestSchema:
    def test_wal_mode_enabled(self, trade_logger: TradeLogger) -> None:
        conn = trade_logger._get_connection()
        assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"

    def test_composite_pk_rejects_duplicate(self, trade_logger: TradeLogger) -> None:
        conn = trade_logger._get_connection()
        insert = (
            "INSERT INTO trades (trade_id, partial_sequence, entry_date, entry_time, "
            "magic_number, strategy_name, symbol, size, entry_price, "
            "expected_entry_price, entry_spread, position_type) "
            "VALUES (1, 0, '2025-01-01', '00:00:00', 1, 'x', 'x', 1, 1, 1, 0, 'BUY')"
        )
        conn.execute(insert)
        conn.commit()
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(insert)
            conn.commit()


class TestLogFill:
    def test_buy_position_type_and_positive_size(self, trade_logger: TradeLogger, mt5_mock: MagicMock) -> None:
        _configure_mt5(mt5_mock)
        trade_logger.log_fill(
            FillData(
                trade_id=1,
                position=_make_pos(pos_type=0, volume=0.10),
                expected_entry_price=1.10,
                strategy_name="test_strategy",
            )
        )
        row = _query(trade_logger, 1)
        assert row is not None
        assert row["position_type"] == "BUY"
        assert row["size"] == pytest.approx(0.10)

    def test_sell_position_type_and_negative_size(self, trade_logger: TradeLogger, mt5_mock: MagicMock) -> None:
        _configure_mt5(mt5_mock)
        trade_logger.log_fill(
            FillData(
                trade_id=1,
                position=_make_pos(pos_type=1, volume=0.10),
                expected_entry_price=1.10,
                strategy_name="test_strategy",
            )
        )
        row = _query(trade_logger, 1)
        assert row is not None
        assert row["position_type"] == "SELL"
        assert row["size"] == pytest.approx(-0.10)

    def test_exit_fields_null_after_fill(self, trade_logger: TradeLogger, mt5_mock: MagicMock) -> None:
        _configure_mt5(mt5_mock)
        trade_logger.log_fill(
            FillData(
                trade_id=1,
                position=_make_pos(),
                expected_entry_price=1.10,
                strategy_name="test_strategy",
            )
        )
        row = _query(trade_logger, 1)
        assert row is not None
        assert row["exit_date"] is None
        assert row["exit_price"] is None
        assert row["gross_pnl"] is None

    def test_entry_spread_computed(self, trade_logger: TradeLogger, mt5_mock: MagicMock) -> None:
        _configure_mt5(mt5_mock)
        trade_logger.log_fill(
            FillData(
                trade_id=1,
                position=_make_pos(price_open=1.10005),
                expected_entry_price=1.10000,
                strategy_name="test_strategy",
            )
        )
        row = _query(trade_logger, 1)
        assert row is not None
        assert row["entry_spread"] == pytest.approx(0.00005, abs=1e-8)


class TestLogClose:
    def _fill_then_close(
        self,
        logger: TradeLogger,
        mt5_mock: MagicMock,
        *,
        trade_id: int = 1,
        exit_price: float = 1.11000,
        commission: float = -3.50,
        profit: float = 100.0,
    ) -> dict:
        deal = _make_deal(price=exit_price, commission=commission, profit=profit)
        _configure_mt5(mt5_mock, exit_deal=deal)
        pos = _make_pos()
        logger.log_fill(
            FillData(
                trade_id=trade_id,
                position=pos,
                expected_entry_price=1.10000,
                strategy_name="test_strategy",
                opening_sl=1.09500,
            )
        )
        logger.log_close(
            CloseData(
                trade_id=trade_id,
                position=pos,
                expected_exit_price=exit_price,
                opening_sl=1.09500,
                exit_trigger="TP",
                entry_price=1.10000,
                expected_entry_price=1.10000,
            )
        )
        row = _query(logger, trade_id)
        assert row is not None
        return row

    def test_exit_fields_populated(self, trade_logger: TradeLogger, mt5_mock: MagicMock) -> None:
        row = self._fill_then_close(trade_logger, mt5_mock)
        assert row["exit_price"] == pytest.approx(1.11000)
        assert row["exit_date"] is not None

    def test_no_row_updated_without_prior_fill(self, trade_logger: TradeLogger, mt5_mock: MagicMock) -> None:
        deal = _make_deal()
        _configure_mt5(mt5_mock, exit_deal=deal)
        pos = _make_pos()
        trade_logger.log_close(
            CloseData(
                trade_id=999,
                position=pos,
                expected_exit_price=1.11,
                opening_sl=1.095,
                exit_trigger="TP",
                entry_price=1.10,
                expected_entry_price=1.10,
            )
        )
        assert _query(trade_logger, 999) is None

    def test_skip_on_missing_exit_deal(self, trade_logger: TradeLogger, mt5_mock: MagicMock) -> None:
        # Only entry deal — no exit deal
        entry_deal = _make_deal(entry=0)
        mt5_mock.symbol_info.return_value = EURUSD_SPEC
        mt5_mock.history_deals_get.return_value = (entry_deal,)
        pos = _make_pos()
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
        assert row is not None
        assert row["exit_price"] is None


class TestMathVerification:
    """Math verification for a BUY EURUSD 0.10 lot trade.

    entry=1.10000, sl=1.09500, tp=1.11000, exit=1.11000 (TP)
    gross_pnl=100.0, commission=-3.50, swap=0.0
    """

    @pytest.fixture(autouse=True)
    def _setup(self, trade_logger: TradeLogger, mt5_mock: MagicMock) -> None:
        exit_deal = _make_deal(entry=1, pos_type=1, price=1.11000, commission=-3.50, profit=100.0)
        _configure_mt5(mt5_mock, exit_deal=exit_deal)
        pos = _make_pos(price_open=1.10000, sl=1.09500, tp=1.11000)
        trade_logger.log_fill(
            FillData(
                trade_id=1,
                position=pos,
                expected_entry_price=1.10000,
                strategy_name="test_strategy",
                opening_sl=1.09500,
            )
        )
        trade_logger.log_close(
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
        row = _query(trade_logger, 1)
        assert row is not None
        self.row = row

    def test_rrr_is_two(self) -> None:
        # (1.11-1.10)/(1.10-1.095) = 0.01/0.005 = 2.0
        assert self.row["rrr"] == pytest.approx(2.0, rel=1e-6)

    def test_gross_pnl(self) -> None:
        assert self.row["gross_pnl"] == pytest.approx(100.0)

    def test_net_pnl(self) -> None:
        assert self.row["net_pnl"] == pytest.approx(96.50)

    def test_entry_spread_zero(self) -> None:
        assert self.row["entry_spread"] == pytest.approx(0.0, abs=1e-8)

    def test_exit_spread_zero(self) -> None:
        assert self.row["exit_spread"] == pytest.approx(0.0, abs=1e-8)

    def test_slippage_cost_zero(self) -> None:
        assert self.row["slippage_cost"] == pytest.approx(0.0, abs=1e-8)


class TestRRRFormula:
    @pytest.mark.parametrize(
        "entry,sl,exit_,pos_type,expected",
        [
            (1.10000, 1.09500, 1.11000, 0, 2.0),  # BUY 1:2
            (1.10000, 1.09500, 1.10500, 0, 1.0),  # BUY 1:1
            (1.10000, 1.09500, 1.09750, 0, -0.5),  # BUY loss
            (1.10000, 1.10500, 1.09000, 1, 2.0),  # SELL 1:2
            (1.10000, 1.10500, 1.10250, 1, -0.5),  # SELL loss
        ],
    )
    def test_rrr(
        self,
        trade_logger: TradeLogger,
        entry: float,
        sl: float,
        exit_: float,
        pos_type: int,
        expected: float,
    ) -> None:
        assert trade_logger._calculate_rrr(entry, exit_, sl, pos_type) == pytest.approx(expected, rel=1e-5)

    def test_zero_risk_returns_zero(self, trade_logger: TradeLogger) -> None:
        assert trade_logger._calculate_rrr(1.10, 1.11, 1.10, 0) == pytest.approx(0.0)


class TestSlippageCostFormula:
    @pytest.mark.parametrize(
        "pos_type,spread,volume,expected",
        [
            (0, 0.00005, 0.10, -0.5),  # BUY 5-pip slippage
            (1, 0.00005, 0.10, 0.5),  # SELL 5-pip slippage
            (0, 0.0, 0.10, 0.0),  # BUY no slippage
            (0, 0.00010, 1.0, -10.0),  # BUY 1-pip 1.0 lot
        ],
    )
    def test_formula(
        self,
        trade_logger: TradeLogger,
        mt5_mock: MagicMock,
        pos_type: int,
        spread: float,
        volume: float,
        expected: float,
    ) -> None:
        mt5_mock.symbol_info.return_value = EURUSD_SPEC
        cost = trade_logger._calculate_slippage_cost(SYMBOL, volume, pos_type, spread)
        assert cost == pytest.approx(expected, abs=1e-8)


class TestExitSpreadFormula:
    @pytest.mark.parametrize(
        "actual,expected_p,pos_type,result",
        [
            (1.11000, 1.11000, 0, 0.0),  # BUY no slippage
            (1.10995, 1.11000, 0, 0.00005),  # BUY unfavorable
            (1.11005, 1.11000, 0, -0.00005),  # BUY favorable
            (1.09000, 1.09000, 1, 0.0),  # SELL no slippage
            (1.09005, 1.09000, 1, 0.00005),  # SELL unfavorable
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
        assert trade_logger._calculate_exit_spread(actual, expected_p, pos_type) == pytest.approx(result, abs=1e-8)

    def test_none_expected_returns_none(self, trade_logger: TradeLogger) -> None:
        assert trade_logger._calculate_exit_spread(1.1, None) is None


class TestGetOpenTradesByTicket:
    def test_returns_open_trade(self, trade_logger: TradeLogger, mt5_mock: MagicMock) -> None:
        _configure_mt5(mt5_mock)
        trade_logger.log_fill(
            FillData(
                trade_id=7,
                position=_make_pos(ticket=55555),
                expected_entry_price=1.10,
                strategy_name="test_strategy",
                opening_sl=1.095,
            )
        )
        result = trade_logger.get_open_trades_by_ticket_last_three([55555])
        assert 55555 in result
        assert result[55555]["trade_id"] == 7

    def test_closed_trade_not_returned(self, trade_logger: TradeLogger, mt5_mock: MagicMock) -> None:
        deal = _make_deal(position_id=66666)
        _configure_mt5(mt5_mock, exit_deal=deal)
        pos = _make_pos(ticket=66666)
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
        assert 66666 not in trade_logger.get_open_trades_by_ticket_last_three([66666])
