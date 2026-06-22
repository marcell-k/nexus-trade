"""Unit tests for RiskManager — layered validation pipeline."""

from __future__ import annotations

import sys
import time
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest

from nexus_trade.config.profile import RiskProfile
from nexus_trade.config.strategy import ExecutionConfig, RiskConfig, StrategyConfig
from nexus_trade.core.data_handler import DataHandler
from nexus_trade.risk.manager import RiskManager

if TYPE_CHECKING:
    from MetaTrader5 import AccountInfo, SymbolInfo

    from nexus_trade.core.state import SharedState


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
        risk=RiskConfig(max_positions=2, max_trades=5, max_spread_points=30, max_slippage_points=20),
    )


def risk_profile() -> RiskProfile:
    return RiskProfile.model_validate(
        {
            "account": {"type": "demo", "initial_balance": 10_000},
            "limits": {
                "max_total_positions": 10,
                "max_daily_trades": 50,
                "max_daily_drawdown_pct": 0.05,
                "max_drawdown_pct": 0.20,
            },
            "adaptive_sizing": {"enabled": False, "scope": "portfolio", "thresholds": []},
            "strategies": {
                "test_strategy": {"enabled": True, "position_sizing_method": "fractional", "risk_value": 1.0}
            },
        }
    )


@pytest.fixture
def shared_state() -> dict:
    return {
        "shutdown_flag": False,
        "position_cache": {},
        "position_cache_timestamp": time.time(),
        "daily_trade_counts": {},
        "daily_drawdown": 0.0,
        "max_drawdown": 0.0,
    }


@pytest.fixture
def atomic_counter() -> MagicMock:
    counter = MagicMock()
    counter.value = 0
    counter.get_lock.return_value.__enter__ = MagicMock(return_value=None)
    counter.get_lock.return_value.__exit__ = MagicMock(return_value=False)
    return counter


@pytest.fixture
def mock_runner() -> MagicMock:
    runner = MagicMock()
    runner.local_position_count = 0
    return runner


@pytest.fixture
def data_handler(mt5_mock: MagicMock) -> DataHandler:
    from zoneinfo import ZoneInfo

    return DataHandler(broker_tz=ZoneInfo("Etc/GMT-3"))


def _make_trade_count() -> MagicMock:
    tc = MagicMock()
    tc.value = 0
    tc.get_lock.return_value.__enter__ = MagicMock(return_value=None)
    tc.get_lock.return_value.__exit__ = MagicMock(return_value=False)
    return tc


@pytest.fixture
def risk_manager(
    strategy_cfg: StrategyConfig,
    risk_profile: RiskProfile,
    shared_state: SharedState,
    atomic_counter: MagicMock,
    data_handler: DataHandler,
    mock_runner: MagicMock,
) -> RiskManager:
    from zoneinfo import ZoneInfo

    return RiskManager(
        strategy_config=strategy_cfg,
        risk_profile=risk_profile,
        shared_state=shared_state,
        global_trade_count=_make_trade_count(),
        global_position_count=atomic_counter,
        data_handler=data_handler,
        broker_tz=ZoneInfo("Etc/GMT-3"),
        strategy_runner=mock_runner,
    )


class TestCheckGlobalRisk:
    def test_passes_under_all_limits(self, risk_manager: RiskManager, atomic_counter: MagicMock) -> None:
        atomic_counter.value = 0
        assert risk_manager.check_global_risk().can_trade is True

    def test_blocks_on_position_limit(
        self, risk_manager: RiskManager, atomic_counter: MagicMock, risk_profile: RiskProfile
    ) -> None:
        atomic_counter.value = risk_profile.limits.max_total_positions
        result = risk_manager.check_global_risk()
        assert result.can_trade is False
        assert "position limit" in result.reason.lower()

    def test_blocks_on_trade_limit(self, risk_manager: RiskManager, risk_profile: RiskProfile) -> None:
        risk_manager.global_trade_count.value = risk_profile.limits.max_daily_trades
        result = risk_manager.check_global_risk()
        assert result.can_trade is False
        assert "trade limit" in result.reason.lower()

    def test_blocks_on_daily_drawdown(self, risk_manager: RiskManager, shared_state: dict) -> None:
        shared_state["daily_drawdown"] = 0.10
        result = risk_manager.check_global_risk()
        assert result.can_trade is False

    def test_blocks_on_max_drawdown(self, risk_manager: RiskManager, shared_state: dict) -> None:
        shared_state["max_drawdown"] = 0.25
        result = risk_manager.check_global_risk()
        assert result.can_trade is False


class TestCheckStrategyLimits:
    def test_passes_under_limits(self, risk_manager: RiskManager, mock_runner: MagicMock) -> None:
        mock_runner.local_position_count = 0
        assert risk_manager.check_strategy_limits("test_strategy").can_trade is True

    def test_blocks_on_strategy_position_limit(self, risk_manager: RiskManager, mock_runner: MagicMock) -> None:
        mock_runner.local_position_count = 2  # max_positions=2
        result = risk_manager.check_strategy_limits("test_strategy")
        assert result.can_trade is False
        assert "position limit" in result.reason.lower()

    def test_blocks_on_daily_trade_limit(
        self, risk_manager: RiskManager, shared_state: dict, mock_runner: MagicMock
    ) -> None:
        mock_runner.local_position_count = 0
        shared_state["daily_trade_counts"]["test_strategy"] = 5  # max_trades=5
        result = risk_manager.check_strategy_limits("test_strategy")
        assert result.can_trade is False
        assert "daily trade limit" in result.reason.lower()


class TestCalculatePositionSize:
    def test_fractional_sizing_formula(
        self, risk_manager: RiskManager, account_info: AccountInfo, eurusd_info: SymbolInfo, mt5_mock: MagicMock
    ) -> None:
        mt5 = sys.modules["MetaTrader5"]
        mt5.account_info.return_value = account_info
        mt5.symbol_info.return_value = eurusd_info
        # balance=10000, risk=0.01, sl=50 pips → volume=0.20
        volume = risk_manager.calculate_position_size("EURUSD", 1.10000, 1.09500, "test_strategy")
        assert volume == pytest.approx(0.20, abs=0.01)

    def test_zero_sl_distance_returns_zero(
        self, risk_manager: RiskManager, account_info: AccountInfo, eurusd_info: SymbolInfo, mt5_mock: MagicMock
    ) -> None:
        mt5 = sys.modules["MetaTrader5"]
        mt5.account_info.return_value = account_info
        mt5.symbol_info.return_value = eurusd_info
        assert risk_manager.calculate_position_size("EURUSD", 1.10, 1.10, "test_strategy") == pytest.approx(0.0)

    def test_returns_zero_on_account_info_failure(self, risk_manager: RiskManager, mt5_mock: MagicMock) -> None:
        sys.modules["MetaTrader5"].account_info.return_value = None
        assert risk_manager.calculate_position_size("EURUSD", 1.1, 1.09, "test_strategy") == pytest.approx(0.0)

    def test_volume_clamped_to_min(
        self, risk_manager: RiskManager, account_info: AccountInfo, eurusd_info: SymbolInfo, mt5_mock: MagicMock
    ) -> None:
        mt5 = sys.modules["MetaTrader5"]
        mt5.account_info.return_value = account_info._replace(balance=10.0)
        mt5.symbol_info.return_value = eurusd_info
        volume = risk_manager.calculate_position_size("EURUSD", 1.1, 1.09, "test_strategy")
        assert volume >= eurusd_info.volume_min


class TestAdaptiveRiskMultiplier:
    def test_returns_one_when_disabled(self, risk_manager: RiskManager) -> None:
        assert risk_manager._get_adaptive_risk_multiplier("test_strategy") == pytest.approx(1.0)

    def test_applies_threshold_when_enabled(
        self,
        strategy_cfg: StrategyConfig,
        shared_state: SharedState,
        atomic_counter: MagicMock,
        data_handler: DataHandler,
        mock_runner: MagicMock,
    ) -> None:
        from zoneinfo import ZoneInfo

        profile = RiskProfile.model_validate(
            {
                "account": {"type": "demo", "initial_balance": 10_000},
                "limits": {
                    "max_total_positions": 10,
                    "max_daily_trades": 50,
                    "max_daily_drawdown_pct": 0.05,
                    "max_drawdown_pct": 0.20,
                },
                "adaptive_sizing": {
                    "enabled": True,
                    "scope": "portfolio",
                    "thresholds": [
                        {"drawdown_pct": 0.05, "risk_multiplier": 0.5},
                        {"drawdown_pct": 0.10, "risk_multiplier": 0.25},
                    ],
                },
                "strategies": {
                    "test_strategy": {"enabled": True, "position_sizing_method": "fractional", "risk_value": 1.0}
                },
            }
        )
        shared_state["max_drawdown"] = 0.07
        rm = RiskManager(
            strategy_config=strategy_cfg,
            risk_profile=profile,
            shared_state=shared_state,
            global_trade_count=_make_trade_count(),
            global_position_count=atomic_counter,
            data_handler=data_handler,
            broker_tz=ZoneInfo("Etc/GMT-3"),
            strategy_runner=mock_runner,
        )
        assert rm._get_adaptive_risk_multiplier("test_strategy") == pytest.approx(0.5)

    def test_no_active_threshold_returns_one(self, risk_manager: RiskManager, shared_state: dict) -> None:
        shared_state["max_drawdown"] = 0.0
        assert risk_manager._get_adaptive_risk_multiplier("test_strategy") == pytest.approx(1.0)


class TestNormalizeVolume:
    def _spec(self, *, vol_min: float = 0.01, vol_max: float = 100.0, vol_step: float = 0.01) -> MagicMock:
        spec = MagicMock()
        spec.volume_min = vol_min
        spec.volume_max = vol_max
        spec.volume_step = vol_step
        return spec

    def test_clamps_below_min(self, risk_manager: RiskManager) -> None:
        assert risk_manager._normalize_volume(0.001, self._spec()) == pytest.approx(0.01)

    def test_clamps_above_max(self, risk_manager: RiskManager) -> None:
        assert risk_manager._normalize_volume(999.0, self._spec()) == pytest.approx(100.0)

    def test_step_rounding(self, risk_manager: RiskManager) -> None:
        # 0.14 → nearest 0.10 step = 0.10
        assert risk_manager._normalize_volume(0.14, self._spec(vol_step=0.10)) == pytest.approx(0.10)


class TestReleasePositionReservation:
    def test_decrements_counter(self, risk_manager: RiskManager, atomic_counter: MagicMock) -> None:
        atomic_counter.value = 3
        risk_manager.release_position_reservation("test")
        assert atomic_counter.value == 2

    def test_does_not_go_below_zero(self, risk_manager: RiskManager, atomic_counter: MagicMock) -> None:
        atomic_counter.value = 0
        risk_manager.release_position_reservation("test")
        assert atomic_counter.value == 0
