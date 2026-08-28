"""Standalone read-only equity / drawdown / risk-ratio snapshot for a risk profile.

Attaches to an already-running, already-logged-in MT5 terminal (it does not
supply login credentials) and reports current balance/equity, daily and max
drawdown, and annualized Sharpe/Calmar ratios (year-to-date and all-time) —
computed as a one-shot full rescan, since a standalone script has no
persisted running-peak state to build on between invocations.

Ratio methodology:
    - Both ratios are built from a business-day equity curve derived from
      realized deal P&L only (``deal.type in (0, 1)``, matching the
      drawdown calculation below). Non-trading business days forward-fill
      the prior day's equity so their return is exactly zero rather than
      being silently skipped.
    - Sharpe = mean(daily excess return) / std(daily return) * sqrt(252).
    - Calmar = annualized return / max drawdown over the same curve.
    - Both return ``None`` when there isn't enough closed-trade history to
      make the figure meaningful (see ``_sharpe_ratio`` / ``_calmar_ratio``).

Running-Sharpe chart:
    Uses asciichartpy (pure-Python, no plotting deps) to sparkline the expanding
    (day-1-to-date) annualized Sharpe ratio computed on the all-time equity curve,
    resampled to SHARPE_CHART_POINTS equally spaced samples. Not a project
    dependency — install with ``uv add asciichartpy`` or ``pip install asciichartpy``;
    the script degrades gracefully (skips the chart with a note) if it's absent.

Usage:
    uv run python tools/get_stat.py --profile src/nexus_trade/config/profiles/live.toml --env .env
"""

from __future__ import annotations

import argparse
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

import MetaTrader5 as mt
import numpy as np
import pandas as pd

from nexus_trade.config.account import MT5ConnectionConfig, load_account_config_from_env, load_env_file
from nexus_trade.config.profile import RiskProfile, load_profile
from nexus_trade.core.connection import MT5Connection
from nexus_trade.main import resolve_env_path

if TYPE_CHECKING:
    from collections.abc import Sequence
    from zoneinfo import ZoneInfo

    from MetaTrader5 import TradeDeal

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

TRADING_PERIODS_PER_YEAR: int = 252
SHARPE_CHART_POINTS: int = 14
SHARPE_WARMUP_BDAYS: int = 42


@dataclass(frozen=True, slots=True)
class EquitySnapshot:
    """Point-in-time equity and drawdown figures for a single account."""

    balance: float
    equity: float
    daily_drawdown_pct: float
    daily_peak_equity: float
    max_drawdown_pct: float
    max_peak_equity: float


@dataclass(frozen=True, slots=True)
class RatioSnapshot:
    """Annualized risk-adjusted return ratios, year-to-date and all-time."""

    sharpe_ytd: float | None
    calmar_ytd: float | None
    sharpe_all: float | None
    calmar_all: float | None


@dataclass(frozen=True, slots=True)
class EquityCurves:
    """Business-day equity curves used to derive ratios and the running-Sharpe chart."""

    ytd: pd.Series
    all_time: pd.Series


def _load_account(env_path: Path) -> MT5ConnectionConfig:
    """Load account."""
    load_env_file(str(env_path), strict=True, override_existing=False)

    profile_env = os.environ.get("RISK_PROFILE")
    if not profile_env:
        raise RuntimeError("RISK_PROFILE not set in env file")
    risk_profile_path = resolve_env_path(profile_env) or Path(profile_env).expanduser()

    return load_account_config_from_env(risk_profile_path=risk_profile_path)


def _realized_pnl(deals: Sequence[TradeDeal]) -> list[float]:
    """Return per-deal realized P&L (commission/swap/fee inclusive) for entry/exit deals only."""
    return [float(d.profit + d.commission + d.swap + d.fee) for d in deals if d.type in (0, 1)]


def _peak_equity(deals: Sequence[TradeDeal], floor_equity: float) -> float:
    """Walk a deal history forward from *floor_equity*, returning the highest running equity seen."""
    pnl_increments = _realized_pnl(deals)
    if not pnl_increments:
        return floor_equity
    running = floor_equity + np.cumsum(np.asarray(pnl_increments, dtype=np.float64))
    return float(max(floor_equity, running.max()))


def _drawdown(peak_equity: float, current_equity: float) -> float:
    """Fractional drawdown of *current_equity* below *peak_equity* (0.0 if peak is non-positive)."""
    return (peak_equity - current_equity) / peak_equity if peak_equity > 0 else 0.0


def _deal_broker_date(deal_time: int, broker_tz: ZoneInfo) -> pd.Timestamp:
    """Return the broker-local calendar date (naive, midnight) for a deal's epoch timestamp."""
    localized = pd.Timestamp(deal_time, unit="s").tz_localize(broker_tz, ambiguous="NaT", nonexistent="shift_forward")
    return localized.normalize().tz_localize(None)


def _daily_equity_curve(
    deals: Sequence[TradeDeal], broker_tz: ZoneInfo, floor_equity: float, as_of: datetime
) -> pd.Series:
    """Build a business-day-indexed equity curve, forward-filled between trades."""
    as_of_day = pd.Timestamp(as_of).tz_convert(broker_tz).normalize().tz_localize(None)

    pnl_increments = _realized_pnl(deals)
    if not pnl_increments:
        return pd.Series([floor_equity], index=pd.DatetimeIndex([as_of_day]))

    dates = pd.DatetimeIndex([_deal_broker_date(d.time, broker_tz) for d in deals if d.type in (0, 1)])
    daily_pnl = pd.Series(pnl_increments, index=dates).groupby(level=0).sum().sort_index()
    trade_equity = floor_equity + daily_pnl.cumsum()

    seed_day = trade_equity.index[0] - pd.tseries.offsets.BDay(1)
    full_range = pd.bdate_range(start=seed_day, end=max(trade_equity.index[-1], as_of_day))
    curve = pd.Series(np.nan, index=full_range)
    curve.loc[seed_day] = floor_equity
    curve.loc[trade_equity.index] = trade_equity.to_numpy()
    return curve.ffill()


def _sharpe_ratio(
    equity_curve: pd.Series, periods_per_year: int = TRADING_PERIODS_PER_YEAR, risk_free_rate: float = 0.0
) -> float | None:
    """Annualized Sharpe ratio from a daily equity curve, or ``None`` if undefined."""
    if len(equity_curve) < 3:
        return None
    returns = equity_curve.pct_change().dropna()
    std = returns.std(ddof=1)
    if returns.empty or not np.isfinite(std) or std == 0:
        return None
    excess = returns - (risk_free_rate / periods_per_year)
    return float(excess.mean() / std * np.sqrt(periods_per_year))


def _calmar_ratio(equity_curve: pd.Series, periods_per_year: int = TRADING_PERIODS_PER_YEAR) -> float | None:
    """Annualized-return / max-drawdown from a daily equity curve, or ``None`` if undefined."""
    if len(equity_curve) < 2 or equity_curve.iloc[0] <= 0:
        return None
    n_periods = len(equity_curve) - 1
    total_return = equity_curve.iloc[-1] / equity_curve.iloc[0] - 1.0
    annualized_return = (1.0 + total_return) ** (periods_per_year / n_periods) - 1.0

    running_max = equity_curve.cummax()
    drawdown = (equity_curve - running_max) / running_max
    max_dd = float(-drawdown.min())
    if max_dd == 0:
        return None
    return float(annualized_return / max_dd)


def compute_report(profile: RiskProfile, broker_tz: ZoneInfo) -> tuple[EquitySnapshot, EquityCurves]:
    """Compute current equity plus daily and max drawdown for *profile*."""
    account = mt.account_info()
    if account is None:
        raise RuntimeError(f"account_info() failed: {mt.last_error()}")

    now = datetime.now(tz=broker_tz)
    current_equity = float(account.equity)
    current_balance = float(account.balance)
    initial_balance = float(profile.account.initial_balance)

    # --- max drawdown: all-time peak equity vs. current equity ---
    all_deals = mt.history_deals_get(profile.account.history_start, now + timedelta(seconds=1)) or ()
    max_peak = max(_peak_equity(all_deals, initial_balance), current_equity)
    max_dd = _drawdown(max_peak, current_equity)

    # --- daily drawdown: today's peak equity vs. current equity ---
    midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
    todays_deals = mt.history_deals_get(midnight, now + timedelta(seconds=1)) or ()
    todays_pnl = sum(_realized_pnl(todays_deals))
    balance_at_midnight = current_balance - todays_pnl
    daily_peak = max(_peak_equity(todays_deals, balance_at_midnight), current_equity)
    daily_dd = _drawdown(daily_peak, current_equity)

    snapshot = EquitySnapshot(
        balance=current_balance,
        equity=current_equity,
        daily_drawdown_pct=daily_dd,
        daily_peak_equity=daily_peak,
        max_drawdown_pct=max_dd,
        max_peak_equity=max_peak,
    )

    # --- YTD and all-time business-day equity curves, for the ratio calculations ---
    year_start = now.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
    ytd_deals = mt.history_deals_get(year_start, now + timedelta(seconds=1)) or ()
    ytd_pnl = sum(_realized_pnl(ytd_deals))
    balance_at_year_start = current_balance - ytd_pnl
    ytd_curve = _daily_equity_curve(ytd_deals, broker_tz, balance_at_year_start, now)

    all_time_curve = _daily_equity_curve(all_deals, broker_tz, initial_balance, now)

    curves = EquityCurves(ytd=ytd_curve, all_time=all_time_curve)

    return snapshot, curves


def compute_ratios(curves: EquityCurves) -> RatioSnapshot:
    """Compute year-to-date and all-time Sharpe and Calmar ratios from pre-built equity curves."""
    return RatioSnapshot(
        sharpe_ytd=_sharpe_ratio(curves.ytd),
        calmar_ytd=_calmar_ratio(curves.ytd),
        sharpe_all=_sharpe_ratio(curves.all_time),
        calmar_all=_calmar_ratio(curves.all_time),
    )


def _expanding_sharpe_series(
    equity_curve: pd.Series,
    periods_per_year: int = TRADING_PERIODS_PER_YEAR,
    min_periods: int = SHARPE_WARMUP_BDAYS,
) -> pd.Series:
    returns = equity_curve.pct_change().dropna()
    if len(returns) < min_periods:
        return pd.Series(dtype=float)

    running_mean = returns.expanding(min_periods=min_periods).mean()
    running_std = returns.expanding(min_periods=min_periods).std(ddof=1)
    sharpe = (running_mean / running_std) * np.sqrt(periods_per_year)
    return sharpe.replace([np.inf, -np.inf], np.nan).dropna()


def _resample_equally_spaced(series: pd.Series, n_points: int = SHARPE_CHART_POINTS) -> pd.Series:
    """Downsample *series* to *n_points* equally spaced samples (by position, not by date)."""
    if len(series) <= n_points:
        return series
    positions = np.round(np.linspace(0, len(series) - 1, n_points)).astype(int)
    positions = np.unique(positions)
    return series.iloc[positions]


def _print_running_sharpe_chart(equity_curve: pd.Series, n_points: int = SHARPE_CHART_POINTS) -> None:
    """Render the running (expanding, all-time) Sharpe ratio as an ascii sparkline."""
    try:
        import asciichartpy
    except ImportError:
        print("Running Sharpe chart skipped — install with `uv add asciichartpy` (or `pip install asciichartpy`).")
        return

    sharpe_series = _expanding_sharpe_series(equity_curve)
    sampled = _resample_equally_spaced(sharpe_series, n_points)
    if len(sampled) < 2:
        print("Running Sharpe chart skipped — not enough closed-trade history yet.")
        return

    start_label = sampled.index[0].strftime("%Y-%m-%d")
    end_label = sampled.index[-1].strftime("%Y-%m-%d")
    print(f"\nRunning Sharpe (all-time, expanding, {len(sampled)} pts, {start_label} -> {end_label}):")
    print(asciichartpy.plot(sampled.to_list(), {"height": 10}))


def _fmt_ratio(value: float | None) -> str:
    return f"{value:.2f}" if value is not None else "N/A (insufficient trade history)"


def _print_report(profile: RiskProfile, snapshot: EquitySnapshot, ratios: RatioSnapshot) -> None:
    print(f"Account type:     {profile.account.type}")
    print(f"Balance:          {snapshot.balance:,.2f}")
    print(f"Equity:           {snapshot.equity:,.2f}")
    print(
        f"Daily drawdown:   {snapshot.daily_drawdown_pct * 100:.2f}%  "
        f"(limit {profile.limits.max_daily_drawdown_pct * 100:.1f}%, peak {snapshot.daily_peak_equity:,.2f})"
    )
    print(
        f"Max drawdown:     {snapshot.max_drawdown_pct * 100:.2f}%  "
        f"(limit {profile.limits.max_drawdown_pct * 100:.1f}%, peak {snapshot.max_peak_equity:,.2f})"
    )
    print(f"Sharpe (YTD):     {_fmt_ratio(ratios.sharpe_ytd)}")
    print(f"Calmar (YTD):     {_fmt_ratio(ratios.calmar_ytd)}")
    print(f"Sharpe (All):     {_fmt_ratio(ratios.sharpe_all)}")
    print(f"Calmar (All):     {_fmt_ratio(ratios.calmar_all)}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print current equity, drawdown, and risk-ratio stats for an account.")
    _ = parser.add_argument("--env", required=True, help="environment file")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    env_path = resolve_env_path(args.env)
    if env_path is None:
        logger.critical(f"GetStatFail err=environment file '{args.env}' not found")
        return 1

    try:
        account_config = _load_account(env_path)
        assert account_config.risk_profile_path is not None  # guaranteed: _load_account requires RISK_PROFILE
        profile = load_profile(account_config.risk_profile_path, account_config.broker_tz)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        logger.critical(f"GetStatFail err={exc}")
        return 1

    connection = MT5Connection(account_config)
    if not connection.connect():
        logger.critical("GetStatFail err=MT5 connect failed")
        return 1

    try:
        snapshot, curves = compute_report(profile, account_config.broker_tz)
        ratios = compute_ratios(curves)
    except RuntimeError as exc:
        logger.critical(f"GetStatFail err={exc}")
        return 1
    finally:
        connection.disconnect()

    _print_report(profile, snapshot, ratios)
    _print_running_sharpe_chart(curves.all_time)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
