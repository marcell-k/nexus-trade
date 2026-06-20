"""Entrypoint — MT5 trading system orchestrator."""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Final, Never

from nexus_trade.config.account import load_account_config_from_env, load_env_file
from nexus_trade.config.profile import load_profile
from nexus_trade.utils.format import log_section_header
from nexus_trade.utils.system import WindowsInhibitor

if TYPE_CHECKING:
    from types import FrameType

CONFIG_DIR: Final[Path] = Path("~/.config/mt5-trading").expanduser()
PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent.parent.parent
logger = logging.getLogger(__name__)


class _ExcludeHeartbeatFromFileFilter(logging.Filter):
    """Exclude orchestrator heartbeat status lines from file logs."""

    def filter(self, record: logging.LogRecord) -> bool:
        return "HB t=" not in record.getMessage()


def setup_logging(log_root: Path, clean_env_name: str) -> None:
    log_root.mkdir(parents=True, exist_ok=True)
    log_filename = f"orchestrator_{clean_env_name}.log"
    file_handler = logging.FileHandler(log_root / log_filename)
    file_handler.addFilter(_ExcludeHeartbeatFromFileFilter())
    stream_handler = logging.StreamHandler()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[file_handler, stream_handler],
        force=True,
    )


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
    return env_path.stem


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
    relative_log_path = log_root.relative_to(PROJECT_ROOT)
    setup_logging(log_root, clean_env_name)

    log_section_header(
        logger,
        f"TRADING SYSTEM STARTING | Config: {env_path.name} | Log dir: {relative_log_path}",
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

        profile_path = resolve_env_path(profile_env)
        if profile_path is None:
            profile_path = Path(profile_env).expanduser()

        profile = load_profile(profile_path)
        logger.info(f"MainStart acct={profile.account.type} | profile={profile_path.name}")

        account_config = load_account_config_from_env()
        orchestrator = Orchestrator(account_config=account_config, profile=profile, log_root=log_root)

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

    return 0


if __name__ == "__main__":
    sys.exit(main())
