from __future__ import annotations

import pandas as pd

from nexus_trade.core.models import Position, PositionType
from nexus_trade.execution.request import EntryRequest, ExitRequest, ModifyRequest, ModifyRequestResult
from nexus_trade.strategies.base import BaseStrategy
from nexus_trade.strategies.sma_crossover.config import SMAParams


class SMACrossoverStrategy(BaseStrategy[SMAParams]):
    """
    SMA crossover entry with ATR-based SL/TP and breakeven trailing.

    ``self.params`` is ``SMAParams`` — all fields accessed directly, no casting.
    """

    def generate_entry_signal(self, data: pd.DataFrame) -> EntryRequest | None:
        if len(data) < self.params.slow_period:
            return None

        close = data["Close"]
        fast_sma: float = float(close.rolling(self.params.fast_period).mean().iloc[-1])
        slow_sma: float = float(close.rolling(self.params.slow_period).mean().iloc[-1])
        current: float = float(close.iloc[-1])
        atr: float = self._calculate_atr(data, self.params.atr_period)

        if fast_sma > slow_sma:
            sl = current - atr * self.params.atr_multiplier
            tp = current + atr * self.params.atr_multiplier * self.params.risk_reward_ratio
            return EntryRequest(
                strategy_name=self.strategy_name,
                order_type="market",
                symbol=self.params.symbol,
                volume=0.0,
                signal=1,
                sl=sl,
                tp=tp,
                comment="SMA_LO",
            )

        if fast_sma < slow_sma:
            sl = current + atr * self.params.atr_multiplier
            tp = current - atr * self.params.atr_multiplier * self.params.risk_reward_ratio
            return EntryRequest(
                strategy_name=self.strategy_name,
                order_type="market",
                symbol=self.params.symbol,
                volume=0.0,
                signal=-1,
                sl=sl,
                tp=tp,
                comment="SMA_SO",
            )

        return None

    def generate_exit_signal(self, pos: Position, data: pd.DataFrame) -> ExitRequest | None:
        return None  # Rely on MT5-side SL/TP; override to add signal-based exits.

    def generate_modify_signal(self, pos: Position, data: pd.DataFrame) -> ModifyRequestResult:
        """Move SL to breakeven after 0.5 % unrealised profit."""
        if not data.shape[0]:
            return None
        current: float = float(data["Close"].iloc[-1])
        is_long: bool = pos.type == PositionType.BUY
        pnl_pct: float = (current - pos.price_open) / pos.price_open * (1 if is_long else -1)
        if pnl_pct >= 0.005 and (pos.sl is None or pos.sl != pos.price_open):
            return ModifyRequest(ticket=pos.ticket, new_sl=pos.price_open, new_tp=pos.tp, comment="Breakeven")
        return None

    @staticmethod
    def _calculate_atr(data: pd.DataFrame, period: int) -> float:
        high, low, close = data["High"], data["Low"], data["Close"]
        tr = pd.concat(
            [high - low, (high - close.shift()).abs(), (low - close.shift()).abs()],
            axis=1,
        ).max(axis=1)
        return float(tr.rolling(period).mean().iloc[-1])
