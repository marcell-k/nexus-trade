from __future__ import annotations

import calendar
import logging
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING
from zoneinfo import ZoneInfo

import MetaTrader5 as mt
import pandas as pd

from nexus_trade.config.account import AccountConfig, load_account_config_from_env, load_env_file
from nexus_trade.core.constants import TIMEFRAME_STRING_MAP, TIMEFRAME_TO_MINUTES

if TYPE_CHECKING:
    import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def _load_account() -> AccountConfig:
    env_path = Path.home() / ".config" / "mt5-trading" / ".env"
    load_env_file(str(env_path))
    return load_account_config_from_env()


def _connect(config: AccountConfig) -> None:
    if not mt.initialize(
        login=config.login,
        password=config.password,
        server=config.server,
        path=config.path,
    ):
        raise RuntimeError(f"MT5 init failed: {mt.last_error()}")
    logger.info(f"MT5 connected | account={config.login} | server={config.server} | tz={config.broker_tz.key}")


def validate_symbols(
    symbols: list[str],
    mapping: dict[str, str] | None = None,
) -> tuple[list[tuple[str, str]], dict[str, pd.DataFrame | None]]:
    """Validate symbols exist in MT5. Returns (valid_pairs, partial_results).

    valid_pairs: list of (desired_name, broker_symbol)
    """
    if mapping is None:
        mapping = {}
    results: dict[str, pd.DataFrame | None] = {}
    valid_pairs: list[tuple[str, str]] = []

    logger.info("VALIDATING SYMBOLS")
    for broker in symbols:
        desired = mapping.get(broker, broker)
        info = mt.symbol_info(broker)
        if info is None:
            logger.warning(f"{broker:<20} → {desired:<25} NOT FOUND")
            results[desired] = None
            continue
        valid_pairs.append((desired, broker))
        logger.info(f"{broker:<20} → {desired:<25} OK")

    return valid_pairs, results


def download_range_data(
    broker_symbol: str,
    timeframe: str,
    start_dt: datetime,
    end_dt: datetime,
    timezone: str,
    broker_tz: str,
) -> pd.DataFrame | None:
    """Download a date range via ``copy_rates_range``. Returns OHLCV DataFrame or None."""
    timeframe_enum = TIMEFRAME_STRING_MAP.get(timeframe)
    if timeframe_enum is None:
        logger.error(f"Unknown timeframe: {timeframe!r}")
        return None

    bar_minutes = TIMEFRAME_TO_MINUTES.get(timeframe_enum)
    if bar_minutes is None:
        logger.error(f"No minute mapping for {timeframe!r}")
        return None

    target_tz = ZoneInfo(timezone)
    logger.info(f"Downloading {broker_symbol} {timeframe} {start_dt.date()} → {end_dt.date()} ...")

    rates: np.ndarray | None = mt.copy_rates_range(broker_symbol, timeframe_enum, start_dt, end_dt)
    if rates is None or rates.size == 0:
        logger.warning(f"NO DATA for {broker_symbol}")
        return None

    df = pd.DataFrame(rates)
    df["time"] = (
        pd.to_datetime(df["time"], unit="s")
        .dt.tz_localize(broker_tz, nonexistent="shift_forward")
        .dt.tz_convert(target_tz)
    )
    df = df.rename(columns={"time": "Date"}).set_index("Date")

    # Drop incomplete current bar.
    current_time = pd.Timestamp.now(tz=target_tz)
    df = df[df.index + pd.Timedelta(minutes=bar_minutes) <= current_time]

    df = df[["open", "high", "low", "close", "tick_volume", "spread"]].rename(columns={"tick_volume": "volume"})
    df.columns = pd.Index([c.capitalize() for c in df.columns])

    logger.info(f"OK → {len(df):,} bars for {broker_symbol}")
    return df


def batch_download_month(
    symbols: list[str],
    timeframe: str,
    year: int,
    month: int,
    output_dir: str,
    timezone: str,
    broker_tz: str,
    mapping: dict[str, str] | None = None,
) -> dict[str, pd.DataFrame | None]:
    """Download one calendar month for each symbol."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    valid_pairs, results = validate_symbols(symbols, mapping)
    logger.info(f"DOWNLOADING {len(valid_pairs)} SYMBOLS → {output_dir}")

    target_tz = ZoneInfo(timezone)
    _, last_day = calendar.monthrange(year, month)
    start_dt = datetime(year, month, 1, 0, 0, 0, tzinfo=target_tz)
    end_dt = datetime(year, month, last_day, 23, 59, 59, tzinfo=target_tz)

    success = 0
    for desired, broker in valid_pairs:
        df = download_range_data(broker, timeframe, start_dt, end_dt, timezone, broker_tz)
        if df is not None and len(df) > 0:
            (Path(output_dir) / f"{desired}_{timeframe}_new.csv").write_text(df.to_csv(), encoding="utf-8")
            success += 1
        results[desired] = df

    logger.info(f"SUMMARY: {success}/{len(valid_pairs)} downloaded successfully")
    return results


def batch_download_historical(
    symbols: list[str],
    timeframe: str,
    start_year: int,
    start_month: int,
    output_dir: str,
    timezone: str,
    broker_tz: str,
    mapping: dict[str, str] | None = None,
) -> dict[str, pd.DataFrame | None]:
    """Download full history (start → today) for each symbol in one MT5 call."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    valid_pairs, results = validate_symbols(symbols, mapping)
    if not valid_pairs:
        return results

    target_tz = ZoneInfo(timezone)
    now = datetime.now(target_tz)
    start_dt = datetime(start_year, start_month, 1, 0, 0, 0, tzinfo=target_tz)
    end_dt = datetime(now.year, now.month, now.day, 23, 59, 59, tzinfo=target_tz)

    logger.info(
        f"DOWNLOADING FULL HISTORY | symbols={len(valid_pairs)} | "
        f"from={start_year}-{start_month:02d} | to=today | out={output_dir}"
    )

    success = 0
    for desired, broker in valid_pairs:
        df = download_range_data(broker, timeframe, start_dt, end_dt, timezone, broker_tz)
        if df is not None and len(df) > 0:
            filepath = Path(output_dir) / f"{desired}_{timeframe}.csv"
            df.to_csv(filepath)
            logger.info(f"Saved {filepath.name} ({len(df):,} bars)")
            success += 1
            results[desired] = df
        else:
            results[desired] = None

    logger.info(f"SUMMARY: {success}/{len(valid_pairs)} symbols completed")
    return results


if __name__ == "__main__":
    _account_config = _load_account()
    _connect(_account_config)

    TIMEFRAME = "M5"
    BROKER_TZ: str = _account_config.broker_tz.key

    SYMBOLS: list[str] = [
        "US30",
        "USTEC",
        "US500",
        "US2000",
        "JP225",
        "AUS200",
        "CHINA50",
        "F40",
        "DE40",
        "STOXX50",
        "UK100",
        "ES35",
        "SWI20",
        "IT40",
        "NETH25",
        "XAUUSD",
        "XAGUSD",
        "XBRUSD",
        "XTIUSD",
        "XNGUSD",
        "BTCUSD",
        "ETHUSD",
        "EURUSD",
        "USDJPY",
        "GBPUSD",
        "USDCHF",
        "AUDUSD",
        "USDCAD",
        "NZDUSD",
        "EURGBP",
        "EURJPY",
        "GBPJPY",
        "EURCHF",
    ]

    SYMBOL_MAPPING: dict[str, str] = {
        "USTEC": "US100",
        "JP225": "Japan225",
        "F40": "France40",
        "DE40": "DAX",
        "STOXX50": "Europe50",
        "ES35": "Spain35",
        "SWI20": "Switzerland20",
        "IT40": "Italy40",
        "NETH25": "Netherlands25",
        "XBRUSD": "USBrentCrudeOil",
        "XTIUSD": "USLightCrudeOil",
    }

    # ── Single month download ──────────────────────────────────────────────────
    batch_download_month(
        symbols=SYMBOLS,
        timeframe=TIMEFRAME,
        year=2026,
        month=1,
        output_dir="new_month",
        timezone="UTC",
        broker_tz=BROKER_TZ,
        mapping=SYMBOL_MAPPING,
    )

    # ── Full historical download (uncomment when needed) ───────────────────────
    # batch_download_historical(
    #     symbols=["AAPL", "AMZN", "MSFT", "NVDA", "TSLA", "META", "GOOG"],
    #     timeframe=TIMEFRAME,
    #     start_year=2015,
    #     start_month=1,
    #     output_dir="historical_data",
    #     timezone="UTC",
    #     broker_tz=BROKER_TZ,
    #     mapping=SYMBOL_MAPPING,
    # )

    mt.shutdown()
