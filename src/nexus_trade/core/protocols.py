from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd

    from nexus_trade.core.models import Position
    from nexus_trade.execution.request import EntryRequest, ExitRequest, ModifyRequestResult


class MT5Tick(Protocol):
    time: int
    bid: float
    ask: float
    last: float
    volume: int
    time_msc: int
    flags: int
    volume_real: float


@runtime_checkable
class MT5Deal(Protocol):
    """Structural protocol for the namedtuple returned by ``MetaTrader5.history_deals_get()``."""

    ticket: int
    order: int
    time: int
    time_msc: int
    type: int
    entry: int
    magic: int
    position_id: int
    reason: int
    volume: float
    price: float
    commission: float
    swap: float
    profit: float
    fee: float
    symbol: str
    comment: str
    external_id: str


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


class XGBClassifierProtocol(Protocol):
    """Structural protocol for ``xgboost.XGBClassifier``."""

    _estimator_type: str

    def load_model(self, fname: str) -> None: ...
    def predict_proba(self, X: object) -> object: ...


class StrategyRunnerProtocol(Protocol):
    """Structural protocol for ``StrategyRunner`` — consumed by ``RiskManager`` to avoid import cycles."""

    local_position_count: int


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


class ProcessLock(Protocol):
    """Protocol for ``multiprocessing.Lock()``. Supports context manager usage."""

    def acquire(self, block: bool = True, timeout: float = -1) -> bool: ...
    def release(self) -> None: ...
    def __enter__(self) -> bool: ...  # noqa: D105
    def __exit__(self, *args: object) -> None: ...  # noqa: D105


class AtomicInt(Protocol):
    """Protocol for ``multiprocessing.Value('i', default)``. All ``.value`` ops serialized via lock."""

    value: int

    def get_lock(self) -> ProcessLock: ...


class AccountInfo(Protocol):
    """Structural protocol for the namedtuple returned by ``MetaTrader5.account_info()``."""

    login: int
    trade_mode: int
    leverage: int
    limit_orders: int
    margin_so_mode: int
    trade_allowed: bool
    trade_expert: bool
    margin_mode: int
    currency_digits: int
    fifo_close: bool
    balance: float
    credit: float
    profit: float
    equity: float
    margin: float
    margin_free: float
    margin_level: float
    margin_so_call: float
    margin_so_so: float
    margin_initial: float
    margin_maintenance: float
    assets: float
    liabilities: float
    commission_blocked: float
    name: str
    server: str
    currency: str
    company: str


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
    """Protocol for models returning class probabilities (Platt, Beta, Stacking)."""

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray: ...


class SupportsPredict(Protocol):
    """Protocol for models returning direct mappings (Isotonic)."""

    def predict(self, X: pd.DataFrame) -> np.ndarray: ...


class ClassifierWithProba(Protocol):
    """Protocol for the underlying ML classifier."""

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray: ...
