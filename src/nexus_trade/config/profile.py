"""Risk profile — TOML-backed, Pydantic-validated account configuration."""

from __future__ import annotations

import tomllib
from datetime import datetime
from typing import TYPE_CHECKING, Literal
from zoneinfo import ZoneInfo

if TYPE_CHECKING:
    from pathlib import Path

from pydantic import Field, ValidationInfo, field_validator, model_validator

from nexus_trade.config._base import FrozenModel


class DrawdownThresholdConfig(FrozenModel):
    drawdown_pct: float = Field(gt=0.0, le=1.0)
    risk_multiplier: float = Field(gt=0.0, le=1.0)


class AdaptiveSizingConfig(FrozenModel):
    enabled: bool = False
    scope: str = "portfolio"
    thresholds: list[DrawdownThresholdConfig] = Field(default_factory=list)


class MetaLabelingConfig(FrozenModel):
    enabled: bool = False
    use_calibration: bool = False
    min_confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class StrategyRiskSettings(FrozenModel):
    enabled: bool
    position_sizing_method: Literal["fractional", "fixed"] = "fractional"
    risk_value: float = Field(
        gt=0.0,
        description=(
            "For 'fractional': % of balance risked per trade (1.0 = 1 %). "
            "For 'fixed': fixed dollar amount risked per trade (e.g. 500 = $500)."
        ),
    )
    meta_labeling: MetaLabelingConfig = Field(default_factory=MetaLabelingConfig)

    @property
    def risk_fraction(self) -> float:
        """Only meaningful for fractional sizing."""
        return self.risk_value / 100.0


class LimitsConfig(FrozenModel):
    max_total_positions: int = Field(gt=0)
    max_daily_trades: int = Field(gt=0)
    max_daily_drawdown_pct: float = Field(gt=0.0, le=1.0)
    max_drawdown_pct: float = Field(gt=0.0, le=1.0)


class RiskMT5ConnectionConfig(FrozenModel):
    type: str = Field(min_length=1)
    initial_balance: int = Field(gt=0)
    # Default kept UTC for schema/repr; _inject_history_start_default overrides at runtime.
    history_start: datetime = Field(default=datetime(2025, 1, 1, tzinfo=ZoneInfo("UTC")))

    @model_validator(mode="before")
    @classmethod
    def _inject_history_start_default(cls, data: object, info: ValidationInfo) -> object:
        """Inject history_start default using broker_tz from context when field is absent."""
        if not isinstance(data, dict) or "history_start" in data:
            return data
        ctx = info.context
        tz: ZoneInfo = (
            ctx["broker_tz"]
            if isinstance(ctx, dict) and isinstance(ctx.get("broker_tz"), ZoneInfo)
            else ZoneInfo("UTC")
        )
        return {**data, "history_start": datetime(2025, 1, 1, tzinfo=tz)}

    @field_validator("history_start", mode="before")
    @classmethod
    def _ensure_aware(cls, v: object, info: ValidationInfo) -> datetime:
        if isinstance(v, str):
            dt = datetime.fromisoformat(v)
        elif isinstance(v, datetime):
            dt = v
        else:
            raise TypeError(f"history_start must be datetime or ISO-8601 string, got {type(v).__name__}")
        if dt.tzinfo is not None:
            return dt
        ctx = info.context
        tz: ZoneInfo = (
            ctx["broker_tz"]
            if isinstance(ctx, dict) and isinstance(ctx.get("broker_tz"), ZoneInfo)
            else ZoneInfo("UTC")
        )
        return dt.replace(tzinfo=tz)


class RiskProfile(FrozenModel):
    account: RiskMT5ConnectionConfig
    limits: LimitsConfig
    adaptive_sizing: AdaptiveSizingConfig = Field(default_factory=AdaptiveSizingConfig)
    strategies: dict[str, StrategyRiskSettings]

    @field_validator("strategies")
    @classmethod
    def _at_least_one_enabled(cls, v: dict[str, StrategyRiskSettings]) -> dict[str, StrategyRiskSettings]:
        if not any(cfg.enabled for cfg in v.values()):
            raise ValueError("profile must enable at least one strategy")
        return v

    @property
    def enabled_strategy_names(self) -> list[str]:
        return sorted(name for name, cfg in self.strategies.items() if cfg.enabled)


def load_profile(path: Path, broker_tz: ZoneInfo | None = None) -> RiskProfile:
    """Parse and validate *path*, propagating *broker_tz* as validation context."""
    if not path.exists():
        raise FileNotFoundError(f"Risk profile not found: {path}")
    with path.open("rb") as fh:
        raw = tomllib.load(fh)
    context: dict[str, ZoneInfo] | None = {"broker_tz": broker_tz} if broker_tz is not None else None
    return RiskProfile.model_validate(raw, context=context)
