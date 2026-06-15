"""Unit tests for Pydantic v2 config models — fail-fast validation."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
from pydantic import ValidationError

from nexus_trade.config.profile import (
    DrawdownThresholdCfg,
    MetaLabelingCfg,
    RiskProfile,
    load_profile,
)
from nexus_trade.config.strategy import (
    BaseStrategyParams,
    ExecutionConfig,
    NewsFilterConfig,
    RiskConfig,
    SessionConfig,
    StrategyConfig,
)

#  RiskProfile


def _valid_profile_dict() -> dict:
    return {
        "account": {"type": "demo", "initial_balance": 10000},
        "limits": {
            "max_total_positions": 5,
            "max_daily_trades": 20,
            "max_daily_drawdown_pct": 0.05,
            "max_drawdown_pct": 0.20,
        },
        "strategies": {
            "sma_crossover": {"enabled": True, "risk_pct": 1.0},
        },
    }


class TestRiskProfile:
    def test_valid_profile_parses(self) -> None:
        p = RiskProfile.model_validate(_valid_profile_dict())
        assert p.account.type == "demo"
        assert p.limits.max_total_positions == 5

    def test_enabled_strategy_names(self) -> None:
        d = _valid_profile_dict()
        d["strategies"]["disabled_strat"] = {"enabled": False, "risk_pct": 0.5}
        p = RiskProfile.model_validate(d)
        assert "sma_crossover" in p.enabled_strategy_names
        assert "disabled_strat" not in p.enabled_strategy_names

    def test_no_enabled_strategies_raises(self) -> None:
        d = _valid_profile_dict()
        d["strategies"]["sma_crossover"]["enabled"] = False
        with pytest.raises(ValidationError, match="at least one strategy"):
            RiskProfile.model_validate(d)

    def test_risk_fraction_property(self) -> None:
        p = RiskProfile.model_validate(_valid_profile_dict())
        assert p.strategies["sma_crossover"].risk_fraction == pytest.approx(0.01)

    def test_drawdown_above_one_raises(self) -> None:
        d = _valid_profile_dict()
        d["limits"]["max_daily_drawdown_pct"] = 1.5
        with pytest.raises(ValidationError):
            RiskProfile.model_validate(d)

    def test_zero_max_positions_raises(self) -> None:
        d = _valid_profile_dict()
        d["limits"]["max_total_positions"] = 0
        with pytest.raises(ValidationError):
            RiskProfile.model_validate(d)

    def test_extra_fields_forbidden(self) -> None:
        d = _valid_profile_dict()
        d["unknown_field"] = "oops"
        with pytest.raises(ValidationError):
            RiskProfile.model_validate(d)

    def test_adaptive_sizing_defaults(self) -> None:
        p = RiskProfile.model_validate(_valid_profile_dict())
        assert p.adaptive_sizing.enabled is False
        assert p.adaptive_sizing.thresholds == []


class TestDrawdownThresholdCfg:
    def test_valid(self) -> None:
        t = DrawdownThresholdCfg(drawdown_pct=0.05, risk_multiplier=0.5)
        assert t.drawdown_pct == pytest.approx(0.05)

    def test_zero_drawdown_pct_raises(self) -> None:
        with pytest.raises(ValidationError):
            DrawdownThresholdCfg(drawdown_pct=0.0, risk_multiplier=0.5)

    def test_risk_multiplier_above_one_raises(self) -> None:
        with pytest.raises(ValidationError):
            DrawdownThresholdCfg(drawdown_pct=0.05, risk_multiplier=1.5)


class TestMetaLabelingCfg:
    def test_defaults(self) -> None:
        cfg = MetaLabelingCfg()
        assert cfg.enabled is False
        assert cfg.min_confidence == pytest.approx(0.0)

    def test_min_confidence_out_of_range_raises(self) -> None:
        with pytest.raises(ValidationError):
            MetaLabelingCfg(min_confidence=1.5)


class TestLoadProfile:
    def test_file_not_found_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_profile(tmp_path / "nonexistent.toml")

    def test_loads_valid_toml(self, tmp_path: Path) -> None:
        toml = textwrap.dedent("""
            [account]
            type = "demo"
            initial_balance = 5000

            [limits]
            max_total_positions = 3
            max_daily_trades = 10
            max_daily_drawdown_pct = 0.05
            max_drawdown_pct = 0.20

            [strategies.test_strat]
            enabled = true
            risk_pct = 0.5
        """)
        path = tmp_path / "profile.toml"
        path.write_text(toml)
        p = load_profile(path)
        assert p.account.initial_balance == 5000
        assert "test_strat" in p.enabled_strategy_names


#  StrategyConfig


def _valid_params() -> BaseStrategyParams:
    return BaseStrategyParams(symbol="EURUSD", backcandles=50, timeframe="M15", timezone="UTC")


class TestExecutionConfig:
    def test_valid(self) -> None:
        cfg = ExecutionConfig(magic_number=12345, deviation=100)
        assert cfg.magic_number == 12345

    def test_magic_zero_raises(self) -> None:
        with pytest.raises(ValidationError):
            ExecutionConfig(magic_number=0)

    def test_magic_negative_raises(self) -> None:
        with pytest.raises(ValidationError):
            ExecutionConfig(magic_number=-1)

    def test_negative_deviation_raises(self) -> None:
        with pytest.raises(ValidationError):
            ExecutionConfig(magic_number=1, deviation=-1)

    def test_comment_prefix_max_length(self) -> None:
        with pytest.raises(ValidationError):
            ExecutionConfig(magic_number=1, comment_prefix="x" * 16)


class TestNewsFilterConfig:
    def test_defaults_disabled(self) -> None:
        cfg = NewsFilterConfig()
        assert cfg.enabled is False
        assert cfg.currencies == []
        assert cfg.buffer_minutes == 15

    def test_negative_buffer_raises(self) -> None:
        with pytest.raises(ValidationError):
            NewsFilterConfig(buffer_minutes=-1)


class TestSessionConfig:
    def test_valid_format(self) -> None:
        s = SessionConfig(start="08:00", end="17:00")
        assert s.start == "08:00"

    @pytest.mark.parametrize("bad", ["8:00", "08:0", "800", "08:00:00"])
    def test_invalid_format_raises(self, bad: str) -> None:
        with pytest.raises(ValidationError):
            SessionConfig(start=bad, end="17:00")


class TestBaseStrategyParams:
    def test_defaults(self) -> None:
        p = BaseStrategyParams(symbol="EURUSD")
        assert p.backcandles == 100
        assert p.timeframe == "M15"
        assert p.timezone == "UTC"

    def test_backcandles_zero_raises(self) -> None:
        with pytest.raises(ValidationError):
            BaseStrategyParams(symbol="X", backcandles=0)

    def test_extra_fields_forbidden(self) -> None:
        with pytest.raises(ValidationError):
            BaseStrategyParams(symbol="X", unknown_param=True)


class TestStrategyConfigBuild:
    def test_strategy_module_derived(self) -> None:
        cfg = StrategyConfig.build(
            name="sma_crossover",
            params=_valid_params(),
            execution=ExecutionConfig(magic_number=1),
            strategy_class="SMAStrategy",
            symbol="EURUSD",
            order_type="market",
        )
        assert cfg.strategy_module == "nexus_trade.strategies.sma_crossover.strategy"
        assert cfg.strategy_class == "SMAStrategy"
        assert cfg.name == "sma_crossover"

    def test_invalid_order_type_raises(self) -> None:
        with pytest.raises(ValidationError):
            StrategyConfig.build(
                name="x",
                params=_valid_params(),
                execution=ExecutionConfig(magic_number=1),
                strategy_class="X",
                symbol="EURUSD",
                order_type="invalid",  # type: ignore[arg-type]
            )


class TestRiskConfig:
    def test_defaults(self) -> None:
        r = RiskConfig()
        assert r.max_positions == 1
        assert r.max_trades == 1
        assert r.max_spread_points == 100

    def test_zero_max_positions_raises(self) -> None:
        with pytest.raises(ValidationError):
            RiskConfig(max_positions=0)
