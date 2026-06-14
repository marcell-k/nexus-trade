from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import ClassVar
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import BaseModel, ConfigDict, Field, field_validator

logger = logging.getLogger(__name__)


MT5_ENV_ALLOWED_KEYS = {
    "MT5_LOGIN",
    "MT5_PASSWORD",
    "MT5_SERVER",
    "MT5_PATH",
    "BROKER_TIMEZONE",
    "MT5_CALENDAR_PATH",
    "MT5_CONFIG_DIR",
    "RISK_PROFILE",
}

MT5_ENV_ALLOWED_PREFIXES = ("SYMBOL_",)


def load_env_file(
    filepath: str = ".env",
    allowed_keys: set[str] | None = None,
    strict: bool = True,
    override_existing: bool = False,
) -> None:
    """
    Load environment key/value pairs with optional strict key validation.

    Strict mode:
    1. Ignore keys outside allowlist.
    2. Optionally preserve existing process-level env values.
    """
    env_path = Path(filepath)
    if not env_path.exists():
        return
    effective_allowed_keys = allowed_keys if allowed_keys is not None else MT5_ENV_ALLOWED_KEYS

    def is_allowed_key(key: str) -> bool:
        if key in effective_allowed_keys:
            return True
        return any(key.startswith(prefix) for prefix in MT5_ENV_ALLOWED_PREFIXES)

    with env_path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            normalized_key = key.strip()
            if strict and not is_allowed_key(normalized_key):
                logger.warning(f"EnvSkip key={normalized_key} | reason=not_allowed")
                continue
            if strict and not override_existing and normalized_key in os.environ:
                logger.warning(f"EnvSkip key={normalized_key} | reason=already_set")
                continue
            os.environ[normalized_key] = value.strip(" \"'")


class AccountConfig(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, arbitrary_types_allowed=True)
    login: int = Field(description="Account number")
    password: str = Field(description="Account password")
    server: str = Field(description="Trade server name")
    path: str = Field(description="MT5 application path")
    broker_tz: ZoneInfo

    @field_validator("broker_tz", mode="before")
    @classmethod
    def _coerce_broker_tz(cls, v: object) -> ZoneInfo:
        """Accept a timezone name string and convert it to a  timezone object."""
        if isinstance(v, str):
            try:
                return ZoneInfo(v)
            except ZoneInfoNotFoundError as exc:
                raise ValueError(f"Unknown broker timezone: '{v}'") from exc
        if isinstance(v, ZoneInfo):
            return v
        raise TypeError(f"broker_tz must be str or ZoneInfo, got {type(v).__name__}")


def load_account_config_from_env(prefix: str = "MT5") -> AccountConfig:
    """Load AccountConfig from environment variables.

    Separated from AccountConfig to keep the model a pure data class with
    no coupling to os.environ or variable naming conventions.
    """
    env_data: dict[str, object] = {
        "login": os.environ.get(f"{prefix}_LOGIN"),
        "password": os.environ.get(f"{prefix}_PASSWORD"),
        "server": os.environ.get(f"{prefix}_SERVER"),
        "path": os.environ.get(f"{prefix}_PATH"),
        "broker_tz": os.environ.get("BROKER_TIMEZONE", "UTC"),
    }
    return AccountConfig.model_validate(env_data)
