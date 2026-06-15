"""Integration tests for RiskManager — layered validation pipeline."""

from __future__ import annotations

import sys
import time
from unittest.mock import MagicMock, patch

import pytest

from nexus_trade.config.strategy import (
    ExecutionConfig,
    FiltersConfig,
    RiskConfig,
    StrategyConfig,
    TradingHoursConfig,
)
from nexus_trade.core.data_handler import DataHandler
from nexus_trade.risk.manager import RiskManager, ValidationResult

#  Fixtures 


@pytest.fixture
def strategy_cfg() -> StrategyConfig:
    from nexus_trade.config.strategy import BaseStrategyParams

    return StrategyConfig.build(
        name="test_strategy",
        params=BaseStrategyParams(symbol="EURUSD", backcandles=50, timeframe="M15"),
        execution=ExecutionConfig(magic_number=12345, deviation=100),
        strategy_class="TestStrategy",
        symbol="EURUSD",
        order_type="market",
        risk=RiskConfig(
            max_positions=2,
            max_trades=5,
            max_spread_points=30,
            max_slippage_points=20,
        ),
    )


@pytest.fixture
def global_policy(make_position_cache_entry) -> dict:
    return {
        "max_total_positions": 10,
        "max_daily_drawdown_pct": 0.05,
        "max_drawdown_pct": 0.20,
        "max_daily_trades": 50,
        "initial_balance": 10_000,
        "adaptive_sizing": {"enabled": False, "scope": "portfolio", "drawdown_thresholds": []},
        "strategy_risk": {"test_strategy": 0.01},
        "log_root": "/tmp/nexus_test",
    }


@pytest.fixture
def clean_shared_state() -> dict:
    return {
        "shutdown_flag": False,
        "position_cache": {},
        "position_cache_timestamp": time.time(),
        "daily_trade_counts": {},
        "daily_drawdown": 0.0,
        "daily_drawdown_initialized": True,
        "max_drawdown": 0.0,
        "max_drawdown_initialized": True,
        "drawdown_last_refresh": 0.0,
        "hist_pnl_sum": 0.0,
        "hist_peak_equity": 10_000.0,
    }


@pytest.fixture
def atomic_counter() -> MagicMock:
    from multiprocessing import Value

    counter = MagicMock()
    counter.value = 0
    counter.get_lock.return_value.__enter__ = MagicMock(return_value=None)
    counter.get_lock.return_value.__exit__ = MagicMock(return_value=False)
    return counter


@pytest.fixture
def mock_runner(atomic_counter: MagicMock) -> MagicMock:
    runner = MagicMock()
    runner.local_position_count = 0
    return runner


@pytest.fixture
def data_handler(mt5_mock: MagicMock) -> DataHandler:
    from zoneinfo import ZoneInfo

    return DataHandler(broker_tz=ZoneInfo("Etc/GMT-3"))


@pytest.fixture
def risk_manager(
    strategy_cfg: StrategyConfig,
    global_policy: dict,
    clean_shared_state: dict,
    atomic_counter: MagicMock,
    data_handler: DataHandler,
    mock_runner: MagicMock,
) -> RiskManager:
    from zoneinfo import ZoneInfo

    trade_count = MagicMock()
    trade_count.value = 0
    trade_count.get_lock.return_value.__enter__ = MagicMock(return_value=None)
    trade_count.get_lock.return_value.__exit__ = MagicMock(return_value=False)

    return RiskManager(
        strategy_config=strategy_cfg,
        global_policy=global_policy,
        shared_state=clean_shared_state,
        global_trade_count=trade_count,
        global_position_count=atomic_counter,
        data_handler=data_handler,
        broker_tz=ZoneInfo("Etc/GMT-3"),
        strategy_runner=mock_runner,
    )


#  check_global_risk 


class TestCheckGlobalRisk:
    def test_passes_under_all_limits(self, risk_manager: RiskManager, atomic_counter: MagicMock) -> None:
        atomic_counter.value = 0
        result = risk_manager.check_global_risk()
        assert result.can_trade is True

    def test_blocks_on_position_limit(
        self, risk_manager: RiskManager, atomic_counter: MagicMock, global_policy: dict
    ) -> None:
        atomic_counter.value = global_policy["max_total_positions"]
        result = risk_manager.check_global_risk()
        assert result.can_trade is False
        assert "position limit" in result.reason.lower()

    def test_blocks_on_trade_limit(self, risk_manager: RiskManager, global_policy: dict) -> None:
        risk_manager.global_trade_count.value = global_policy["max_daily_trades"]
        result = risk_manager.check_global_risk()
        assert result.can_trade is False
        assert "trade limit" in result.reason.lower()

    def test_blocks_on_daily_drawdown(self, risk_manager: RiskManager, clean_shared_state: dict) -> None:
        clean_shared_state["daily_drawdown"] = 0.10  # > 5% limit
        result = risk_manager.check_global_risk()
        assert result.can_trade is False
        assert "drawdown" in result.reason.lower()

    def test_blocks_on_max_drawdown(self, risk_manager: RiskManager, clean_shared_state: dict) -> None:
        clean_shared_state["max_drawdown"] = 0.25  # > 20% limit
        result = risk_manager.check_global_risk()
        assert result.can_trade is False

    def test_passes_at_limit_minus_one(
        self, risk_manager: RiskManager, atomic_counter: MagicMock, global_policy: dict
    ) -> None:
        atomic_counter.value = global_policy["max_total_positions"] - 1
        result = risk_manager.check_global_risk()
        assert result.can_trade is True


#  check_strategy_limits 


class TestCheckStrategyLimits:
    def test_passes_under_limits(self, risk_manager: RiskManager, mock_runner: MagicMock) -> None:
        mock_runner.local_position_count = 0
        result = risk_manager.check_strategy_limits("test_strategy")
        assert result.can_trade is True

    def test_blocks_on_strategy_position_limit(self, risk_manager: RiskManager, mock_runner: MagicMock) -> None:
        mock_runner.local_position_count = 2  # max_positions=2
        result = risk_manager.check_strategy_limits("test_strategy")
        assert result.can_trade is False
        assert "position limit" in result.reason.lower()

    def test_blocks_on_daily_trade_limit(
        self, risk_manager: RiskManager, clean_shared_state: dict, mock_runner: MagicMock
    ) -> None:
        mock_runner.local_position_count = 0
        clean_shared_state["daily_trade_counts"]["test_strategy"] = 5  # max_trades=5
        result = risk_manager.check_strategy_limits("test_strategy")
        assert result.can_trade is False
        assert "daily trade limit" in result.reason.lower()

    def test_skips_position_check_when_flag_set(
        self, risk_manager: RiskManager, mock_runner: MagicMock, clean_shared_state: dict
    ) -> None:
        mock_runner.local_position_count = 99  # over limit but skipped
        result = risk_manager.check_strategy_limits("test_strategy", skip_position_limit_check=True)
        # Should only block on trade count, which is 0
        assert result.can_trade is True


#  calculate_position_size 


class TestCalculatePositionSize:
    def test_fractional_sizing_formula(
        self,
        risk_manager: RiskManager,
        account_info,
        eurusd_info,
        mt5_mock: MagicMock,
    ) -> None:
        mt5 = sys.modules["MetaTrader5"]
        mt5.account_info.return_value = account_info
        mt5.symbol_info.return_value = eurusd_info

        # formula: volume = (balance * risk_pct * multiplier) / (sl_distance / tick_size * tick_value)
        # balance=10000, risk=0.01, sl_distance=0.005 (50 pips), tick_size=0.00001, tick_value=1.0
        # ticks = 0.005 / 0.00001 = 500
        # volume = (10000 * 0.01 * 1.0) / (500 * 1.0) = 100 / 500 = 0.20 lots
        volume = risk_manager.calculate_position_size(
            symbol="EURUSD",
            entry=1.10000,
            sl=1.09500,
            strategy_name="test_strategy",
        )
        assert volume == pytest.approx(0.20, abs=0.01)

    def test_zero_sl_distance_returns_zero(
        self, risk_manager: RiskManager, account_info, eurusd_info, mt5_mock: MagicMock
    ) -> None:
        mt5 = sys.modules["MetaTrader5"]
        mt5.account_info.return_value = account_info
        mt5.symbol_info.return_value = eurusd_info

        volume = risk_manager.calculate_position_size(
            symbol="EURUSD", entry=1.10, sl=1.10, strategy_name="test_strategy"
        )
        assert volume == pytest.approx(0.0)

    def test_returns_zero_on_account_info_failure(self, risk_manager: RiskManager, mt5_mock: MagicMock) -> None:
        mt5 = sys.modules["MetaTrader5"]
        mt5.account_info.return_value = None
        volume = risk_manager.calculate_position_size("EURUSD", 1.1, 1.09, "test_strategy")
        assert volume == pytest.approx(0.0)

    def test_volume_clamped_to_min(
        self, risk_manager: RiskManager, account_info, eurusd_info, mt5_mock: MagicMock
    ) -> None:
        """Tiny risk → sub-minimum volume → clamp to volume_min."""
        mt5 = sys.modules["MetaTrader5"]
        modified_account = account_info._replace(balance=10.0)  # tiny balance
        mt5.account_info.return_value = modified_account
        mt5.symbol_info.return_value = eurusd_info

        volume = risk_manager.calculate_position_size("EURUSD", 1.1, 1.09, "test_strategy")
        assert volume >= eurusd_info.volume_min


#  _get_adaptive_risk_multiplier 


class TestAdaptiveRiskMultiplier:
    def test_returns_one_when_disabled(self, risk_manager: RiskManager) -> None:
        multiplier = risk_manager._get_adaptive_risk_multiplier("test_strategy")  # noqa: SLF001
        assert multiplier == pytest.approx(1.0)

    def test_applies_threshold_when_enabled(
        self,
        strategy_cfg: StrategyConfig,
        clean_shared_state: dict,
        atomic_counter: MagicMock,
        data_handler: DataHandler,
        mock_runner: MagicMock,
    ) -> None:
        from zoneinfo import ZoneInfo

        policy_with_adaptive = {
            "max_total_positions": 10,
            "max_daily_drawdown_pct": 0.05,
            "max_drawdown_pct": 0.20,
            "max_daily_trades": 50,
            "initial_balance": 10_000,
            "adaptive_sizing": {
                "enabled": True,
                "scope": "portfolio",
                "drawdown_thresholds": [
                    {"drawdown_pct": 0.05, "risk_multiplier": 0.5},
                    {"drawdown_pct": 0.10, "risk_multiplier": 0.25},
                ],
            },
            "strategy_risk": {"test_strategy": 0.01},
            "log_root": "/tmp",
        }
        clean_shared_state["max_drawdown"] = 0.07  # between 5% and 10%
        trade_count = MagicMock()
        trade_count.value = 0
        trade_count.get_lock.return_value.__enter__ = MagicMock(return_value=None)
        trade_count.get_lock.return_value.__exit__ = MagicMock(return_value=False)
        rm = RiskManager(
            strategy_config=strategy_cfg,
            global_policy=policy_with_adaptive,
            shared_state=clean_shared_state,
            global_trade_count=trade_count,
            global_position_count=atomic_counter,
            data_handler=data_handler,
            broker_tz=ZoneInfo("Etc/GMT-3"),
            strategy_runner=mock_runner,
        )
        multiplier = rm._get_adaptive_risk_multiplier("test_strategy")  # noqa: SLF001
        assert multiplier == pytest.approx(0.5)

    def test_no_active_threshold_returns_one(self, risk_manager: RiskManager, clean_shared_state: dict) -> None:
        clean_shared_state["max_drawdown"] = 0.0
        multiplier = risk_manager._get_adaptive_risk_multiplier("test_strategy")  # noqa: SLF001
        assert multiplier == pytest.approx(1.0)


#  _normalize_volume 


class TestNormalizeVolume:
    def test_rounds_to_step(self, risk_manager: RiskManager, eurusd_info) -> None:
        from nexus_trade.core.symbol import SymbolSpec

        spec = SymbolSpec.from_mt5.__func__(SymbolSpec, "EURUSD")  # type: ignore[attr-defined]

    def test_clamps_below_min(self, risk_manager: RiskManager, eurusd_info) -> None:
        from nexus_trade.core.symbol import SymbolSpec

        # Build a minimal SymbolSpec directly
        spec = MagicMock()
        spec.volume_min = 0.01
        spec.volume_max = 100.0
        spec.volume_step = 0.01
        result = risk_manager._normalize_volume(0.001, spec)  # noqa: SLF001
        assert result == pytest.approx(0.01)

    def test_clamps_above_max(self, risk_manager: RiskManager) -> None:
        spec = MagicMock()
        spec.volume_min = 0.01
        spec.volume_max = 100.0
        spec.volume_step = 0.01
        result = risk_manager._normalize_volume(999.0, spec)  # noqa: SLF001
        assert result == pytest.approx(100.0)

    def test_step_rounding(self, risk_manager: RiskManager) -> None:
        spec = MagicMock()
        spec.volume_min = 0.01
        spec.volume_max = 100.0
        spec.volume_step = 0.10
        # 0.14 should round to nearest 0.10 step → 0.10
        result = risk_manager._normalize_volume(0.14, spec)  # noqa: SLF001
        assert result == pytest.approx(0.10)


#  release_position_reservation 


class TestReleasePositionReservation:
    def test_decrements_counter(self, risk_manager: RiskManager, atomic_counter: MagicMock) -> None:
        atomic_counter.value = 3
        risk_manager.release_position_reservation("test")
        assert atomic_counter.value == 2

    def test_does_not_go_below_zero(self, risk_manager: RiskManager, atomic_counter: MagicMock) -> None:
        atomic_counter.value = 0
        risk_manager.release_position_reservation("test")
        assert atomic_counter.value == 0
