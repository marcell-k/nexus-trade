from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    from types import TracebackType

    import numpy as np
    import pandas as pd

    from nexus_trade.config.strategy import BaseStrategyParams, StrategyConfig
    from nexus_trade.core.models import Position
    from nexus_trade.execution.request import EntryRequest, ExitRequest, ModifyRequestResult


@runtime_checkable
class MT5Deal(Protocol):
    """Structural protocol for the namedtuple returned by ``MetaTrader5.history_deals_get()``."""

    @property
    def ticket(self) -> int: ...
    @property
    def order(self) -> int: ...
    @property
    def time(self) -> int: ...
    @property
    def time_msc(self) -> int: ...
    @property
    def type(self) -> int: ...
    @property
    def entry(self) -> int: ...
    @property
    def magic(self) -> int: ...
    @property
    def position_id(self) -> int: ...
    @property
    def reason(self) -> int: ...
    @property
    def volume(self) -> float: ...
    @property
    def price(self) -> float: ...
    @property
    def commission(self) -> float: ...
    @property
    def swap(self) -> float: ...
    @property
    def profit(self) -> float: ...
    @property
    def fee(self) -> float: ...
    @property
    def symbol(self) -> str: ...
    @property
    def comment(self) -> str: ...
    @property
    def external_id(self) -> str: ...


class MT5Position(Protocol):
    """Structural protocol for namedtuples returned by ``MetaTrader5.positions_get()``."""

    ticket: int
    symbol: str
    type: int
    magic: int
    volume: float
    price_open: float
    price_current: float
    sl: float
    tp: float
    profit: float
    swap: float
    time: int


class StrategyRunnerProtocol(Protocol):
    """Structural protocol for ``StrategyRunner`` — consumed by ``RiskManager`` to avoid import cycles."""

    local_position_count: int


class ConfigModule(Protocol):
    def get_config(self) -> StrategyConfig[BaseStrategyParams]: ...


class StrategyProtocol(Protocol):
    """
    Structural protocol for the three runner-facing hooks of any ``BaseStrategy`` subclass.

    ``StrategyRunner`` holds ``self.strategy: StrategyProtocol`` so it remains decoupled
    from the concrete type parameter (``BaseStrategy[SMAParams]``, etc.) and avoids
    generic invariance issues across process boundaries.
    """

    def generate_entry_signal(self, data: pd.DataFrame) -> EntryRequest | None: ...
    def generate_exit_signal(self, pos: Position, data: pd.DataFrame) -> ExitRequest | None: ...
    def generate_modify_signal(self, pos: Position, data: pd.DataFrame) -> ModifyRequestResult: ...


class _AtomicLock(Protocol):
    def acquire(self, block: bool = ..., timeout: float = ...) -> bool: ...
    def release(self) -> None: ...
    def __enter__(self) -> bool: ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
        /,
    ) -> None: ...


class ProcessLock(Protocol):
    def acquire(self, block: bool = True, timeout: float = -1) -> bool: ...
    def release(self) -> None: ...
    def __enter__(self) -> bool: ...  # noqa: D105
    def __exit__(  # noqa: D105
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
        /,
    ) -> None: ...


class AtomicInt(Protocol):
    """Protocol for ``multiprocessing.Value('i', default)``. All ``.value`` ops serialized via lock."""

    value: int

    def get_lock(self) -> _AtomicLock: ...


class AccountInfo(Protocol):
    """Structural protocol for the namedtuple returned by ``MetaTrader5.account_info()``."""

    @property
    def login(self) -> int: ...
    @property
    def trade_mode(self) -> int: ...
    @property
    def leverage(self) -> int: ...
    @property
    def limit_orders(self) -> int: ...
    @property
    def margin_so_mode(self) -> int: ...
    @property
    def trade_allowed(self) -> bool: ...
    @property
    def trade_expert(self) -> bool: ...
    @property
    def margin_mode(self) -> int: ...
    @property
    def currency_digits(self) -> int: ...
    @property
    def fifo_close(self) -> bool: ...
    @property
    def balance(self) -> float: ...
    @property
    def credit(self) -> float: ...
    @property
    def profit(self) -> float: ...
    @property
    def equity(self) -> float: ...
    @property
    def margin(self) -> float: ...
    @property
    def margin_free(self) -> float: ...
    @property
    def margin_level(self) -> float: ...
    @property
    def margin_so_call(self) -> float: ...
    @property
    def margin_so_so(self) -> float: ...
    @property
    def margin_initial(self) -> float: ...
    @property
    def margin_maintenance(self) -> float: ...
    @property
    def assets(self) -> float: ...
    @property
    def liabilities(self) -> float: ...
    @property
    def commission_blocked(self) -> float: ...
    @property
    def name(self) -> str: ...
    @property
    def server(self) -> str: ...
    @property
    def currency(self) -> str: ...
    @property
    def company(self) -> str: ...


class SymbolInfo(Protocol):
    """Structural protocol for the namedtuple returned by ``MetaTrader5.symbol_info()``."""

    name: str
    description: str
    trade_contract_size: float
    point: float
    digits: int
    volume_min: float
    volume_max: float
    volume_step: float
    bid: float
    ask: float
    spread: int
    spread_float: bool
    trade_tick_size: float
    trade_tick_value: float
    trade_tick_value_profit: float
    trade_tick_value_loss: float
    currency_base: str
    currency_profit: str
    currency_margin: str
    trade_mode: int
    filling_mode: int
    trade_stops_level: int
    trade_freeze_level: int
    swap_long: float
    swap_short: float
    swap_mode: int
    time: int


class SupportsPredictProba(Protocol):
    """Protocol for models returning class probabilities."""

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray: ...


class SupportsPredict(Protocol):
    """Protocol for models returning direct mappings (Isotonic)."""

    def predict(self, X: pd.DataFrame) -> np.ndarray: ...


class ClassifierWithProba(SupportsPredictProba, Protocol):
    """Protocol for the underlying ML classifier."""


class XGBClassifierProtocol(SupportsPredictProba, Protocol):
    """Structural protocol for ``xgboost.XGBClassifier``."""

    _estimator_type: str

    def load_model(self, fname: str) -> None: ...
