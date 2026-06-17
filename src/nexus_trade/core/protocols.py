from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import numpy as np

if TYPE_CHECKING:
    from types import TracebackType

    import numpy as np
    import pandas as pd

    from nexus_trade.config.strategy import BaseStrategyParams, StrategyConfig
    from nexus_trade.core.models import Position
    from nexus_trade.execution.request import EntryRequest, ExitRequest, ModifyRequestResult


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
