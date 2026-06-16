from __future__ import annotations

from enum import IntEnum

import MetaTrader5 as mt


class TimeFrame(IntEnum):
    """MetaTrader 5 timeframe constants."""

    M1 = mt.TIMEFRAME_M1
    M5 = mt.TIMEFRAME_M5
    M15 = mt.TIMEFRAME_M15
    M30 = mt.TIMEFRAME_M30
    H1 = mt.TIMEFRAME_H1
    H4 = mt.TIMEFRAME_H4
    D1 = mt.TIMEFRAME_D1
    W1 = mt.TIMEFRAME_W1
    MN1 = mt.TIMEFRAME_MN1


class OrderType(IntEnum):
    """MetaTrader 5 order-type constants."""

    BUY = mt.ORDER_TYPE_BUY
    SELL = mt.ORDER_TYPE_SELL
    BUY_LIMIT = mt.ORDER_TYPE_BUY_LIMIT
    SELL_LIMIT = mt.ORDER_TYPE_SELL_LIMIT
    BUY_STOP = mt.ORDER_TYPE_BUY_STOP
    SELL_STOP = mt.ORDER_TYPE_SELL_STOP
    BUY_STOP_LIMIT = mt.ORDER_TYPE_BUY_STOP_LIMIT
    SELL_STOP_LIMIT = mt.ORDER_TYPE_SELL_STOP_LIMIT


class OrderFilling(IntEnum):
    """MetaTrader 5 order-filling-mode constants."""

    FOK = mt.ORDER_FILLING_FOK
    IOC = mt.ORDER_FILLING_IOC
    RETURN = mt.ORDER_FILLING_RETURN
    BOC = mt.ORDER_FILLING_BOC


class TradeAction(IntEnum):
    """MetaTrader 5 trade-action constants."""

    DEAL = mt.TRADE_ACTION_DEAL
    PENDING = mt.TRADE_ACTION_PENDING
    SLTP = mt.TRADE_ACTION_SLTP
    MODIFY = mt.TRADE_ACTION_MODIFY
    REMOVE = mt.TRADE_ACTION_REMOVE
    CLOSE_BY = mt.TRADE_ACTION_CLOSE_BY


class TimeInForce(IntEnum):
    """MetaTrader 5 order-lifetime constants."""

    GTC = mt.ORDER_TIME_GTC
    DAY = mt.ORDER_TIME_DAY
    SPECIFIED = mt.ORDER_TIME_SPECIFIED
    SPECIFIED_DAY = mt.ORDER_TIME_SPECIFIED_DAY


MT5_POSITION_TYPE_BUY: int = mt.POSITION_TYPE_BUY
MT5_POSITION_TYPE_SELL: int = mt.POSITION_TYPE_SELL


MT5_DEAL_ENTRY_IN: int = mt.DEAL_ENTRY_IN
MT5_DEAL_ENTRY_OUT: int = mt.DEAL_ENTRY_OUT
MT5_DEAL_ENTRY_INOUT: int = mt.DEAL_ENTRY_INOUT

MT5_RETCODE_DONE: int = 10009


TIMEFRAME_TO_MINUTES: dict[TimeFrame, int] = {
    TimeFrame.M1: 1,
    TimeFrame.M5: 5,
    TimeFrame.M15: 15,
    TimeFrame.M30: 30,
    TimeFrame.H1: 60,
    TimeFrame.H4: 240,
    TimeFrame.D1: 1440,
    TimeFrame.W1: 10080,
    TimeFrame.MN1: 43200,
}

TIMEFRAME_STRING_MAP: dict[str, TimeFrame] = {
    "M1": TimeFrame.M1,
    "M5": TimeFrame.M5,
    "M15": TimeFrame.M15,
    "M30": TimeFrame.M30,
    "H1": TimeFrame.H1,
    "H4": TimeFrame.H4,
    "D1": TimeFrame.D1,
    "W1": TimeFrame.W1,
    "MN1": TimeFrame.MN1,
}


def string_to_timeframe(tf_str: str) -> TimeFrame:
    """Return the TimeFrame enum for *tf_str*, or ``None`` if unknown."""
    return TIMEFRAME_STRING_MAP[tf_str.upper()]
