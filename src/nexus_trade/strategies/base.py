from __future__ import annotations

from abc import ABC, abstractmethod
from functools import lru_cache
from typing import TYPE_CHECKING

import pandas as pd

from nexus_trade.config.strategy import BaseStrategyParams

if TYPE_CHECKING:
    from datetime import time as dt_time

    from nexus_trade.core.models import Position
    from nexus_trade.execution.request import EntryRequest, ExitRequest, ModifyRequestResult


class BaseStrategy[T_Params: BaseStrategyParams](ABC):
    """
    Abstract base for all strategy implementations.

    Type parameter ``T_Params`` is the concrete params class declared in the
    strategy's ``config.py``. Bind it at the subclass level::

        class SMACrossoverStrategy(BaseStrategy[SMAParams]):
            ...

    ``self.params`` is then fully typed as ``SMAParams`` — no dict access, no casting.

    Contract
    --------
    Implement all three abstract hooks. The runner calls them in this order each cycle:

    1. ``generate_entry_signal``  — bar-aligned, only when no position is open (per strategy)
    2. ``generate_modify_signal`` — bar-aligned, for every open position
    3. ``generate_exit_signal``   — every minute, for every open position
    """

    def __init__(self, params: T_Params) -> None:
        self.params: T_Params = params

    @abstractmethod
    def generate_entry_signal(self, data: pd.DataFrame) -> EntryRequest | None:
        """
        Return an ``EntryRequest`` when an entry condition is met, ``None`` otherwise.

        Args:
            data: OHLCV DataFrame with timezone-aware index, length == ``backcandles + 1``.

        """
        ...

    @abstractmethod
    def generate_exit_signal(self, pos: Position, data: pd.DataFrame) -> ExitRequest | None:
        """
        Return an ``ExitRequest`` when an exit condition is met, ``None`` otherwise.

        Called every minute for each open position. Return ``None`` to rely on MT5-side SL/TP.

        Args:
            pos:  Live position snapshot.
            data: Latest OHLCV bars.

        """
        ...

    @abstractmethod
    def generate_modify_signal(self, pos: Position, data: pd.DataFrame) -> ModifyRequestResult:
        """
        Return a ``ModifyRequest`` (SL/TP adjustment), ``ExitRequest`` (partial close),
        or ``None`` if no action is needed.

        Called bar-aligned, before ``generate_exit_signal``. Typical uses:
        trailing stops, breakeven adjustment, scaled partial exits.

        Args:
            pos:  Live position snapshot.
            data: Latest OHLCV bars.

        """
        ...

    # ── Shared time-parsing helpers ───────────────────────────────────────────

    @staticmethod
    @lru_cache(maxsize=32)
    def _parse_time(time_str: str) -> pd.Timestamp:
        """Parse ``HH:MM`` once; subsequent calls return the cached ``Timestamp``."""
        return pd.to_datetime(time_str, format="%H:%M")

    @staticmethod
    def _parse_time_with_offset(time_str: str, offset: pd.Timedelta) -> dt_time:
        """Parse ``time_str`` and apply ``offset``, returning a ``datetime.time``."""
        return (BaseStrategy._parse_time(time_str) + offset).time()

    @staticmethod
    def _parse_time_to_str(time_str: str, offset: pd.Timedelta) -> str:
        """Parse ``time_str``, apply ``offset``, return ``HH:MM`` string."""
        return (BaseStrategy._parse_time(time_str) + offset).strftime("%H:%M")
