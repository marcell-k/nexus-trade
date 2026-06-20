from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import ClassVar
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import BaseModel, ConfigDict, Field, field_validator

logger = logging.getLogger(__name__)


ENV_ALLOWED_KEYS = {
    "LOGIN",
    "PASSWORD",
    "SERVER",
    "MT5_PATH",
    "BROKER_TIMEZONE",
    "CALENDAR_PATH",
    "CONFIG_DIR",
    "RISK_PROFILE",
}

ENV_ALLOWED_PREFIXES = ("SYMBOL_",)


def load_env_file(
    filepath: str = ".env",
    allowed_keys: set[str] | None = None,
    strict: bool = True,
    override_existing: bool = False,
) -> None:
    """Load environment key/value pairs with optional strict key validation."""
    env_path = Path(filepath)
    if not env_path.exists():
        return
    effective_allowed_keys = allowed_keys if allowed_keys is not None else ENV_ALLOWED_KEYS

    def is_allowed_key(key: str) -> bool:
        if key in effective_allowed_keys:
            return True
        return any(key.startswith(prefix) for prefix in ENV_ALLOWED_PREFIXES)

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
    calendar_path: Path | None = Field(
        default=None, description="Economic calendar CSV export path (MT5_CALENDAR_PATH)"
    )
    risk_profile_path: Path | None = Field(default=None, description="Risk profile TOML path (RISK_PROFILE)")

    @field_validator("broker_tz", mode="before")
    @classmethod
    def _coerce_broker_tz(cls, v: object) -> ZoneInfo:
        if isinstance(v, str):
            try:
                return ZoneInfo(v)
            except ZoneInfoNotFoundError as exc:
                raise ValueError(f"Unknown broker timezone: '{v}'") from exc
        if isinstance(v, ZoneInfo):
            return v
        raise TypeError(f"broker_tz must be str or ZoneInfo, got {type(v).__name__}")

    @field_validator("calendar_path", mode="before")
    @classmethod
    def _coerce_calendar_path(cls, v: object) -> Path | None:
        if v is None or (isinstance(v, str) and not v.strip()):
            return None
        p = Path(str(v)).expanduser().resolve()
        if not p.exists():
            raise ValueError(f"MT5_CALENDAR_PATH does not exist: '{p}'")
        return p

    @field_validator("risk_profile_path", mode="before")
    @classmethod
    def _coerce_risk_profile_path(cls, v: object) -> Path | None:
        if v is None or (isinstance(v, str) and not v.strip()):
            return None
        p = Path(str(v)).expanduser().resolve()
        if not p.exists():
            raise ValueError(f"RISK_PROFILE path does not exist: '{p}'")
        return p


def load_account_config_from_env(
    *,
    risk_profile_path: str | Path | None = None,
) -> AccountConfig:
    """Load AccountConfig from environment variables."""
    env_data: dict[str, object] = {
        "login": os.environ.get("LOGIN"),
        "password": os.environ.get("PASSWORD"),
        "server": os.environ.get("SERVER"),
        "path": os.environ.get("MT5_PATH"),
        "broker_tz": os.environ.get("BROKER_TIMEZONE", "UTC"),
        "calendar_path": os.environ.get("CALENDAR_PATH"),
        "risk_profile_path": risk_profile_path if risk_profile_path is not None else os.environ.get("RISK_PROFILE"),
    }
    return AccountConfig.model_validate(env_data)
