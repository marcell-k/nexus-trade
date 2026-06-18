"""Risk profile — TOML-backed, Pydantic-validated account configuration."""

from __future__ import annotations

import tomllib
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from pathlib import Path

from pydantic import Field, field_validator

from nexus_trade.config._base import FrozenModel


class DrawdownThresholdCfg(FrozenModel):
    drawdown_pct: float = Field(gt=0.0, le=1.0)
    risk_multiplier: float = Field(gt=0.0, le=1.0)


class AdaptiveSizingCfg(FrozenModel):
    enabled: bool = False
    scope: str = "portfolio"
    thresholds: list[DrawdownThresholdCfg] = Field(default_factory=list)


class MetaLabelingCfg(FrozenModel):
    enabled: bool = False
    use_calibration: bool = False
    min_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    recalculate_volume_on_bars: bool = False


class StrategyCfg(FrozenModel):
    enabled: bool
    position_sizing_method: Literal["fractional", "fixed"] = "fractional"
    risk_value: float = Field(
        gt=0.0,
        description=(
            "For 'fractional': % of balance risked per trade (1.0 = 1 %). "
            "For 'fixed': fixed dollar amount risked per trade (e.g. 500 = $500)."
        ),
    )
    meta_labeling: MetaLabelingCfg = Field(default_factory=MetaLabelingCfg)

    @property
    def risk_fraction(self) -> float:
        """Only meaningful for fractional sizing."""
        return self.risk_value / 100.0


class LimitsCfg(FrozenModel):
    max_total_positions: int = Field(gt=0)
    max_daily_trades: int = Field(gt=0)
    max_daily_drawdown_pct: float = Field(gt=0.0, le=1.0)
    max_drawdown_pct: float = Field(gt=0.0, le=1.0)


class AccountCfg(FrozenModel):
    type: str = Field(min_length=1)
    initial_balance: int = Field(gt=0)


class RiskProfile(FrozenModel):
    account: AccountCfg
    limits: LimitsCfg
    adaptive_sizing: AdaptiveSizingCfg = Field(default_factory=AdaptiveSizingCfg)
    strategies: dict[str, StrategyCfg]

    @field_validator("strategies")
    @classmethod
    def _at_least_one_enabled(cls, v: dict[str, StrategyCfg]) -> dict[str, StrategyCfg]:
        if not any(cfg.enabled for cfg in v.values()):
            raise ValueError("profile must enable at least one strategy")
        return v

    @property
    def enabled_strategy_names(self) -> list[str]:
        return sorted(name for name, cfg in self.strategies.items() if cfg.enabled)


def load_profile(path: Path) -> RiskProfile:
    """Parse and validate *path*. Raises ``FileNotFoundError`` or ``ValidationError`` on failure."""
    if not path.exists():
        raise FileNotFoundError(f"Risk profile not found: {path}")
    with path.open("rb") as fh:
        raw = tomllib.load(fh)
    return RiskProfile.model_validate(raw)
