"""Entrypoint — MT5 trading system orchestrator."""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Final, Never

from pydantic import ValidationError

from nexus_trade.config.account import load_account_config_from_env, load_env_file
from nexus_trade.config.profile import load_profile
from nexus_trade.logging.mp_logging import create_log_queue, setup_logging
from nexus_trade.utils.format import log_section_header
from nexus_trade.utils.system import WindowsInhibitor

if TYPE_CHECKING:
    from types import FrameType

CONFIG_DIR: Final[Path] = Path("~/.config/mt5-trading").expanduser()
PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent.parent.parent
logger = logging.getLogger(__name__)


def resolve_env_path(env_arg: str) -> Path | None:
    """Resolve env file path: absolute → CONFIG_DIR → project root."""
    candidate = Path(env_arg).expanduser()
    if candidate.is_absolute():
        return candidate if candidate.is_file() else None
    for search_dir in (CONFIG_DIR, PROJECT_ROOT):
        path = search_dir / env_arg
        if path.is_file():
            return path
    return None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MT5 Trading System Orchestrator")
    _ = parser.add_argument(
        "--env",
        type=str,
        required=True,
        help="Path to environment file. Searched in: ~/.config/mt5-trading/, then project root.",
    )
    return parser.parse_args()


def _clean_env_name(env_path: Path) -> str:
    name = env_path.name
    return name.removeprefix(".env.") if name.startswith(".env.") else env_path.stem


def main() -> int:
    """Provide main entry point supporting multiple broker instances via CLI arguments."""
    args = _parse_args()

    env_path = resolve_env_path(args.env)
    if env_path is None:
        print(f"CRITICAL: Environment file '{args.env}' not found. Searched:")
        print(f"  1. {CONFIG_DIR}")
        print(f"  2. {PROJECT_ROOT}")
        return 1

    print(f"Loading environment from: {env_path.name}")
    load_env_file(str(env_path), strict=True, override_existing=False)

    from nexus_trade.orchestrator import Orchestrator

    clean_env_name = _clean_env_name(env_path)
    log_root = PROJECT_ROOT / "logs" / clean_env_name

    log_queue = create_log_queue()
    listener = setup_logging(log_queue)

    log_section_header(
        logger,
        f"TRADING SYSTEM STARTING | Config: {env_path.name}",
        level=logging.INFO,
    )

    orchestrator = None
    shutdown_initiated = False

    def shutdown_once() -> None:
        nonlocal shutdown_initiated
        if shutdown_initiated:
            return
        shutdown_initiated = True
        if orchestrator:
            orchestrator.shutdown()

    def signal_handler(sig: int, frame: FrameType | None) -> Never:
        logger.info("Signal sig=SIGINT | action=shutdown")
        raise KeyboardInterrupt

    previous_sigint = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, signal_handler)

    try:
        profile_env = os.environ.get("RISK_PROFILE")
        if not profile_env:
            logger.critical("MainFail reason=RISK_PROFILE_not_set")
            return 1

        resolved_profile_path: Path = resolve_env_path(profile_env) or Path(profile_env).expanduser()

        try:
            account_config = load_account_config_from_env(risk_profile_path=resolved_profile_path)
        except ValidationError as exc:
            logger.critical(f"ConfigFail err={exc}")
            return 1

        assert account_config.risk_profile_path is not None  # guaranteed: passed resolved path above
        profile = load_profile(account_config.risk_profile_path, account_config.broker_tz)
        logger.info(f"MainStart acct={profile.account.type} | profile={account_config.risk_profile_path.name}")

        stop_file = PROJECT_ROOT / f".stop.{clean_env_name}"
        orchestrator = Orchestrator(
            account_config=account_config,
            profile=profile,
            log_root=log_root,
            stop_file=stop_file,
            log_queue=log_queue,
        )

        with WindowsInhibitor(keep_display=False, away_mode=True, logger=logger):
            orchestrator.start()

    except KeyboardInterrupt:
        logger.debug("MainStop reason=keyboard_interrupt")
    except Exception as e:
        logger.exception(f"MainCrash err={e}")
        return 1
    finally:
        shutdown_once()
        signal.signal(signal.SIGINT, previous_sigint)
        log_section_header(logger, "TRADING SYSTEM STOPPED")
        listener.stop()

    return 0


if __name__ == "__main__":
    sys.exit(main())
