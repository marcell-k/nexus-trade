"""Order execution engine for MetaTrader 5."""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from enum import Enum, auto
from typing import TYPE_CHECKING

import MetaTrader5 as mt
import pandas as pd

from nexus_trade.config.timings import SYSTEM_TIMINGS
from nexus_trade.core.constants import (
    MT5_RETCODE_DONE,
    RETCODE_NO_CHANGE,
    RETRYABLE_CODES,
    OrderFilling,
    OrderType,
    TimeInForce,
    TradeAction,
)
from nexus_trade.core.models import Position, Tick, normalize_order
from nexus_trade.core.registry import STRATEGY_CONFIG_REGISTRY
from nexus_trade.core.symbol import SYMBOL_SPEC_CACHE, SymbolSpec
from nexus_trade.execution.request import (
    EntryRequest,
    ExecutionResult,
    ExitRequest,
    ModifyRequest,
)
from nexus_trade.utils.format import format_price_display

if TYPE_CHECKING:
    from collections.abc import Callable
    from zoneinfo import ZoneInfo

    from MetaTrader5 import MT5EntryRequest, MT5Request, OrderSendResult, TradePosition

    from nexus_trade.config.strategy import BaseStrategyParams, StrategyConfig
    from nexus_trade.core.types import MT5Tick, OrderSnapshot, PositionCacheEntry


logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class _MarketData:
    tick: Tick
    server_epoch: int | None
    stops_level: int

    @property
    def ask(self) -> float:
        return self.tick.ask

    @property
    def bid(self) -> float:
        return self.tick.bid

    @property
    def tick_epoch(self) -> int:
        return self.tick.time


class _ExpirationOutcome(Enum):
    """Result of expiration processing to replace the ambiguous int | None | bool tri-state."""

    NO_EXPIRATION = auto()
    EXPIRED = auto()


_PENDING_ORDER_TYPE_MAP: dict[tuple[str, int], OrderType] = {
    ("stop", 1): OrderType.BUY_STOP,
    ("stop", -1): OrderType.SELL_STOP,
    ("limit", 1): OrderType.BUY_LIMIT,
    ("limit", -1): OrderType.SELL_LIMIT,
}


class OrderExecutor:
    """Order execution with retry logic, caching."""

    def __init__(self, broker_tz: ZoneInfo) -> None:
        self.broker_tz: ZoneInfo = broker_tz

        self.max_retries: int = 3
        _no_changes = getattr(mt, "TRADE_RETCODE_NO_CHANGES", None)
        self._retcode_no_changes: int = (
            int(_no_changes) if isinstance(_no_changes, int) and _no_changes > 0 else RETCODE_NO_CHANGE
        )

        self._entry_handlers: dict[str, Callable[[EntryRequest], ExecutionResult]] = {
            "market": self._execute_market_entry,
            "bracket": self._execute_bracket_entry,
            "stop": self._execute_pending_entry,
            "limit": self._execute_pending_entry,
        }

        logger.debug(f"ExecInit retry_max={self.max_retries}")

    def execute_entry(self, request: EntryRequest) -> ExecutionResult:
        handler = self._entry_handlers.get(request.order_type)
        if handler is None:
            logger.error(f"EntryFail reason=unknown_order_type | ot={request.order_type}")
            return self._fail_entry(request.symbol, f"Unknown order type: {request.order_type}")
        return handler(request)

    def execute_exit(self, request: ExitRequest) -> ExecutionResult:
        positions = mt.positions_get(ticket=request.ticket)
        if not positions:
            logger.warning(f"ExitSkip t={request.ticket} | reason=already_closed")
            return ExecutionResult(
                success=False,
                ticket=request.ticket,
                error_message=f"Position {request.ticket} already closed",
                request_type="exit",
            )

        position = positions[0]
        attempted_volume: float = position.volume * request.portion
        results = self.close_positions(
            tickets=[request.ticket],
            portions=[request.portion],
            preloaded_positions={request.ticket: position},
        )
        success, deal_id = results.get(request.ticket, (False, None))
        return ExecutionResult(
            success=success,
            ticket=request.ticket,
            executed_volume=attempted_volume if success else None,
            error_message="" if success else f"Failed to close {request.ticket}",
            request_type="exit",
            deal_id=deal_id,
        )

    def execute_modify(self, request: ModifyRequest) -> ExecutionResult:
        success = self.modify_position_sl_tp(ticket=request.ticket, sl=request.new_sl, tp=request.new_tp)
        return ExecutionResult(
            success=success,
            ticket=request.ticket,
            error_message="" if success else f"Failed to modify {request.ticket}",
            request_type="modify",
        )

    def _execute_market_entry(self, request: EntryRequest) -> ExecutionResult:
        symbol = request.symbol
        symbol_spec, type_filling = self._get_cached_symbol_spec(symbol)
        config = self._get_strategy_config(request.strategy_name)

        order_type = OrderType.BUY if request.signal == 1 else OrderType.SELL

        tick = mt.symbol_info_tick(symbol)
        if tick is None:
            logger.error(f"EntryFail sym={symbol} | reason=tick_unavailable")
            return self._fail_entry(symbol, f"{symbol}: tick data unavailable")

        price: float = tick.ask if request.signal == 1 else tick.bid
        sl = round(request.sl, symbol_spec.digits) if request.sl is not None else 0.0
        tp = round(request.tp, symbol_spec.digits) if request.tp is not None else 0.0

        mt5_request: MT5EntryRequest = {
            "action": TradeAction.DEAL,
            "symbol": symbol,
            "volume": request.volume,
            "type": order_type,
            "price": price,
            "sl": sl,
            "tp": tp,
            "deviation": config.execution.deviation,
            "magic": config.execution.magic_number,
            "comment": request.comment,
            "type_filling": type_filling,
            "type_time": TimeInForce.GTC,
        }

        result = self._order_send_with_retry(mt5_request)
        if result is None or result.retcode != MT5_RETCODE_DONE:
            error_msg = result.comment if result is not None else "MT5 returned None"
            return self._fail_entry(symbol, error_msg)
        logger.info(
            f"EntryOK typ={order_type.name} | sym={symbol} | vol={request.volume:.2f} | "
            f"px={format_price_display(price)} | sl={sl} | tp={tp} | t={result.order}"
        )
        return ExecutionResult(success=True, ticket=result.order, request_type="entry", symbol=symbol)

    def _execute_pending_entry(self, request: EntryRequest) -> ExecutionResult:
        """Unified handler for BUY/SELL STOP and LIMIT orders."""
        order_type = _PENDING_ORDER_TYPE_MAP.get((request.order_type, request.signal))
        if order_type is None:
            return self._fail_entry(
                request.symbol,
                f"Invalid pending order combination: order_type={request.order_type!r} signal={request.signal}",
            )

        config = self._get_strategy_config(request.strategy_name)
        symbol = request.symbol
        symbol_spec, type_filling = self._get_cached_symbol_spec(symbol)
        sl: float = round(request.sl, symbol_spec.digits) if request.sl is not None else 0.0
        tp: float = round(request.tp, symbol_spec.digits) if request.tp is not None else 0.0

        expiration = self._resolve_expiration(request, symbol)
        if expiration is _ExpirationOutcome.EXPIRED:
            return self._fail_entry(symbol, "Expiration time has already passed")

        exp_ts = expiration if isinstance(expiration, int) else None
        type_time: TimeInForce = TimeInForce.SPECIFIED if exp_ts else TimeInForce.GTC

        order: MT5EntryRequest = {
            "action": TradeAction.PENDING,
            "symbol": symbol,
            "volume": float(request.volume),
            "type": order_type,
            "price": request.entry_price or 0.0,
            "sl": sl,
            "tp": tp,
            "deviation": config.execution.deviation,
            "magic": config.execution.magic_number,
            "comment": request.comment,
            "type_filling": type_filling,
            "type_time": type_time,
        }
        if exp_ts:
            # expiration only present when needed — use MT5Request for the extended form
            order_with_exp: MT5Request = {**order, "expiration": exp_ts}
            result = self._order_send_with_retry(order_with_exp)
        else:
            result = self._order_send_with_retry(order)
        ok, error_msg = self._validate_order_result(result)
        if not ok:
            return self._fail_entry(symbol, error_msg)

        assert result is not None
        logger.info(
            f"EntryOK typ={order_type.name} | sym={symbol} | vol={request.volume:.2f} | "
            f"px={format_price_display(request.entry_price or 0.0)} | sl={sl} | "
            f"tp={tp} | t={result.order}"
        )
        return ExecutionResult(success=True, ticket=result.order, request_type="entry", symbol=symbol)

    def _execute_bracket_entry(self, request: EntryRequest) -> ExecutionResult:
        """Execute OCO bracket (buy-stop + sell-stop) with dual-fill protection."""
        symbol = request.symbol
        config = self._get_strategy_config(request.strategy_name)
        symbol_spec, type_filling = self._get_cached_symbol_spec(symbol)

        market = self._fetch_market_data(symbol)
        if market is None:
            return self._fail_entry(symbol, f"{symbol}: market data unavailable")

        expiration = self._resolve_expiration(request, symbol, market)
        if expiration is _ExpirationOutcome.EXPIRED:
            return self._fail_entry(symbol, "Expiration time has already passed.")

        exp_ts = expiration if isinstance(expiration, int) else None
        min_threshold = int(config.execution.min_market_threshold_points)

        buy_req, sell_req = self._prepare_bracket_requests(
            request=request,
            symbol_spec=symbol_spec,
            type_filling=type_filling,
            market=market,
            expiration_timestamp=exp_ts,
            strategy_config=config,
            min_market_threshold_points=min_threshold,
        )

        buy_result, sell_result = self._send_bracket_orders(buy_req, sell_req, symbol)
        if buy_result is None or sell_result is None:
            return self._fail_entry(symbol, "Bracket execution failed")

        exp_label = request.expiration_time if request.expiration_time else "GTC"
        buy_order_type = buy_req.get("type", OrderType.BUY_STOP)
        sell_order_type = sell_req.get("type", OrderType.SELL_STOP)
        logger.info(
            f"{request.strategy_name}: BrktOK sym={symbol} | "
            f"buy={OrderType(buy_order_type).name}@{request.buy_stop:.{symbol_spec.digits}f} | "
            f"sell={OrderType(sell_order_type).name}@{request.sell_stop:.{symbol_spec.digits}f} | "
            f"exp={exp_label}"
        )
        logger.debug(f"BrktOKIds sym={symbol} | buy={buy_result.order} | sell={sell_result.order}")
        return ExecutionResult(
            success=True,
            ticket=buy_result.order,
            order_tickets=[buy_result.order, sell_result.order],
            request_type="entry",
            symbol=symbol,
        )

    def _prepare_bracket_requests(
        self,
        request: EntryRequest,
        symbol_spec: SymbolSpec,
        type_filling: OrderFilling,
        market: _MarketData,
        expiration_timestamp: int | None,
        strategy_config: StrategyConfig[BaseStrategyParams],
        min_market_threshold_points: int,
    ) -> tuple[MT5EntryRequest | MT5Request, MT5EntryRequest | MT5Request]:
        buy_stop = round(request.buy_stop or 0.0, symbol_spec.digits)
        sell_stop = round(request.sell_stop or 0.0, symbol_spec.digits)
        buy_sl = round(request.buy_sl or 0.0, symbol_spec.digits)
        sell_sl = round(request.sell_sl or 0.0, symbol_spec.digits)
        buy_tp = round(request.buy_tp or 0.0, symbol_spec.digits)
        sell_tp = round(request.sell_tp or 0.0, symbol_spec.digits)

        stops_level_price_display = format_price_display(market.stops_level * symbol_spec.point)
        logger.debug(
            f"BrktMkt sym={request.symbol} | ask={market.ask:.{symbol_spec.digits}f} | "
            f"bid={market.bid:.{symbol_spec.digits}f} | stp_pts={market.stops_level} | "
            f"stp_px={stops_level_price_display}"
        )

        buy_type, buy_price, buy_action = self._classify_bracket_leg(
            entry_price=buy_stop,
            current_price=market.ask,
            stops_level_points=market.stops_level,
            min_threshold_points=min_market_threshold_points,
            symbol_spec=symbol_spec,
            is_buy=True,
        )
        sell_type, sell_price, sell_action = self._classify_bracket_leg(
            entry_price=sell_stop,
            current_price=market.bid,
            stops_level_points=market.stops_level,
            min_threshold_points=min_market_threshold_points,
            symbol_spec=symbol_spec,
            is_buy=False,
        )

        comment = request.comment
        buy_order = self._build_bracket_request(
            buy_action,
            buy_type,
            request.symbol,
            request.volume,
            buy_price,
            buy_sl,
            buy_tp,
            strategy_config,
            type_filling,
            expiration_timestamp,
            comment,
        )
        sell_order = self._build_bracket_request(
            sell_action,
            sell_type,
            request.symbol,
            request.volume,
            sell_price,
            sell_sl,
            sell_tp,
            strategy_config,
            type_filling,
            expiration_timestamp,
            comment,
        )
        return buy_order, sell_order

    def _classify_bracket_leg(
        self,
        entry_price: float,
        current_price: float,
        stops_level_points: int,
        min_threshold_points: int,
        symbol_spec: SymbolSpec,
        is_buy: bool,
    ) -> tuple[OrderType, float, TradeAction]:
        distance_points = (entry_price - current_price if is_buy else current_price - entry_price) / symbol_spec.point
        side = "B" if is_buy else "S"

        if distance_points < min_threshold_points:
            logger.warning(f"BrktLegMode side={side} | mode=market | reason=too_close_stoplimit")
            return (OrderType.BUY if is_buy else OrderType.SELL), current_price, TradeAction.DEAL

        if distance_points < stops_level_points:
            logger.warning(
                f"BrktLegMode side={side} | mode=limit | reason=stoplevel_violation | "
                f"entry={entry_price:.{symbol_spec.digits}f} | cur={current_price:.{symbol_spec.digits}f}"
            )
            return (OrderType.BUY_LIMIT if is_buy else OrderType.SELL_LIMIT), entry_price, TradeAction.PENDING

        logger.debug(f"BrktLegMode side={side} | mode=stop | entry={entry_price:.{symbol_spec.digits}f}")
        return (OrderType.BUY_STOP if is_buy else OrderType.SELL_STOP), entry_price, TradeAction.PENDING

    def _build_bracket_request(
        self,
        action: TradeAction,
        order_type: OrderType,
        symbol: str,
        volume: float,
        price: float,
        sl: float,
        tp: float,
        strategy_config: StrategyConfig[BaseStrategyParams],
        type_filling: OrderFilling,
        expiration_timestamp: int | None,
        comment: str,
    ) -> MT5EntryRequest | MT5Request:
        is_pending_with_exp = bool(expiration_timestamp and action == TradeAction.PENDING)
        req: MT5EntryRequest = {
            "action": action,
            "symbol": symbol,
            "volume": float(volume),
            "type": order_type,
            "price": float(price),
            "sl": float(sl),
            "tp": float(tp),
            "deviation": strategy_config.execution.deviation,
            "magic": strategy_config.execution.magic_number,
            "comment": comment,
            "type_filling": type_filling,
            "type_time": TimeInForce.SPECIFIED if is_pending_with_exp else TimeInForce.GTC,
        }
        if is_pending_with_exp:
            assert expiration_timestamp is not None
            extra: MT5Request = {"expiration": expiration_timestamp}
            return {**req, **extra}
        return req

    def _send_bracket_orders(
        self, buy_request: MT5EntryRequest | MT5Request, sell_request: MT5EntryRequest | MT5Request, symbol: str
    ) -> tuple[OrderSendResult | None, OrderSendResult | None]:
        buy_raw = self._order_send_with_retry(buy_request)
        ok, error_msg = self._validate_order_result(buy_raw)
        if not ok:
            logger.error(f"BrktBuyFail sym={symbol} | err={error_msg}")
            return None, None

        sell_raw = self._order_send_with_retry(sell_request)
        ok, error_msg = self._validate_order_result(sell_raw)
        if not ok:
            logger.error(f"BrktSellFail sym={symbol} | err={error_msg}")
            if buy_raw is not None:
                self.cancel_order(buy_raw.order)
            return None, None

        return buy_raw, sell_raw

    def _fetch_market_data(self, symbol: str) -> _MarketData | None:
        """Single MT5 round-trip for all market state needed by entry paths."""
        raw_tick: MT5Tick | None = mt.symbol_info_tick(symbol)
        symbol_info = mt.symbol_info(symbol)
        if raw_tick is None or symbol_info is None:
            logger.error(msg="Fetch market data error")
            return None

        tick: Tick = Tick.from_mt5(raw_tick)
        return _MarketData(
            tick=tick,
            server_epoch=symbol_info.time,
            stops_level=symbol_info.trade_stops_level,
        )

    def _resolve_expiration(
        self, request: EntryRequest, symbol: str, market: _MarketData | None = None
    ) -> int | _ExpirationOutcome:
        """
        Convert HH:MM expiration to broker epoch.

        Returns:
            int — valid future expiration timestamp
            _ExpirationOutcome.NO_EXPIRATION — no expiration on the request
            _ExpirationOutcome.EXPIRED — expiration is in the past; caller must reject

        """
        if not request.expiration_time:
            return _ExpirationOutcome.NO_EXPIRATION
        if market is None:
            market = self._fetch_market_data(symbol)
            if market is None:
                return _ExpirationOutcome.EXPIRED
        if market.tick_epoch == 0:
            return _ExpirationOutcome.EXPIRED

        strategy_tz = STRATEGY_CONFIG_REGISTRY.get_tz(request.strategy_name)
        tf_minutes = STRATEGY_CONFIG_REGISTRY.get_timeframe_minutes(request.strategy_name)

        now_broker_display = self._server_epoch_to_broker_display(market.tick_epoch)
        now_strategy = now_broker_display.tz_convert(strategy_tz)
        expiration_pd = self._parse_expiration_hhmm(request.expiration_time, now_strategy, tf_minutes, strategy_tz)
        expiration_broker = expiration_pd.tz_convert(self.broker_tz)

        if expiration_broker <= now_broker_display:
            logger.warning(
                f"ExpPast hhmm={request.expiration_time} | tf_min={tf_minutes} | "
                f"tz={strategy_tz.key} | now={now_broker_display.strftime('%Y-%m-%d %H:%M:%S %Z')} | "
                f"exp={expiration_broker.strftime('%Y-%m-%d %H:%M:%S %Z')}"
            )
            return _ExpirationOutcome.EXPIRED

        if market.server_epoch and self._broker_display_to_mt5_epoch(expiration_broker) <= market.server_epoch:
            logger.error(
                f"ExpFail hhmm={request.expiration_time} | tz={strategy_tz.key} | "
                f"exp={expiration_broker.strftime('%Y-%m-%d %H:%M:%S %Z')} | reason=lte_server_time"
            )
            return _ExpirationOutcome.EXPIRED

        return self._broker_display_to_mt5_epoch(expiration_broker)

    def _server_epoch_to_broker_display(self, srv_epoch: int) -> pd.Timestamp:
        """Convert server epoch (UTC) to broker display timezone."""
        time_utc = pd.to_datetime(srv_epoch, unit="s", utc=True)
        delta = time_utc.tz_convert(self.broker_tz).utcoffset()
        assert delta is not None
        offset = delta.total_seconds()

        return (time_utc - pd.Timedelta(seconds=offset)).tz_convert(self.broker_tz)

    def _broker_display_to_mt5_epoch(self, broker_display: pd.Timestamp) -> int:
        """Convert broker display time back to real UTC epoch."""
        utc_ts = broker_display.tz_convert("UTC")
        delta = utc_ts.tz_convert(self.broker_tz).utcoffset()
        assert delta is not None
        offset = delta.total_seconds()
        display_epoch_ts = utc_ts + pd.Timedelta(seconds=offset)
        return int(display_epoch_ts.timestamp())

    def _parse_expiration_hhmm(
        self, hhmm: str, now_strategy: pd.Timestamp, tf_minutes: int, strategy_tz: ZoneInfo
    ) -> pd.Timestamp:
        hour, minute = int(hhmm[:2]), int(hhmm[3:])
        naive_dt = datetime(now_strategy.year, now_strategy.month, now_strategy.day, hour, minute)
        naive_timestamp = pd.Timestamp(naive_dt) + pd.Timedelta(minutes=tf_minutes) - pd.Timedelta(seconds=1)
        try:
            localized = naive_timestamp.tz_localize(strategy_tz, ambiguous="raise", nonexistent="raise")
        except Exception:
            try:
                localized = naive_timestamp.tz_localize(strategy_tz, ambiguous="raise", nonexistent="shift_forward")
                logger.warning(f"ExpDST hhmm={hhmm} | type=non_existent | use_is_dst=True")
            except Exception:
                logger.warning(f"ExpDST hhmm={hhmm} | type=ambiguous | use_is_dst=False")
                localized = naive_timestamp.tz_localize(strategy_tz, ambiguous=False)
        return localized

    def close_positions(
        self,
        tickets: list[int],
        portions: list[float] | None = None,
        preloaded_positions: dict[int, TradePosition] | None = None,
    ) -> dict[int, tuple[bool, int | None]]:
        """Close positions with optional partial closing. Returns {ticket: (success, deal_id)}."""
        results: dict[int, tuple[bool, int | None]] = {}
        resolved_portions: list[float | None] = [None] * len(tickets) if portions is None else list(portions)
        if len(resolved_portions) != len(tickets):
            return dict.fromkeys(tickets, (False, None))

        position_by_ticket: dict[int, TradePosition] = dict(preloaded_positions or {})
        missing = [t for t in tickets if t not in position_by_ticket]
        if missing:
            loaded = self._load_positions_for_tickets(missing)
            if loaded is None:
                results.update(dict.fromkeys(missing, (False, None)))
                return results
            position_by_ticket.update(loaded)

        tick_cache: dict[str, object] = {}

        for ticket, portion in zip(tickets, resolved_portions, strict=True):
            pos = position_by_ticket.get(ticket)
            results[ticket] = self._close_single_ticket(ticket, portion, pos, tick_cache)

        successful = sum(1 for ok, _ in results.values() if ok)
        if successful:
            logger.info(f"CloseBatch ok={successful}/{len(tickets)}")
        return results

    def _close_single_ticket(
        self,
        ticket: int,
        portion: float | None,
        pos: TradePosition | None,
        tick_cache: dict[str, object],
    ) -> tuple[bool, int | None]:
        """Process the closing of a single ticket position."""
        if pos is None:
            return False, None

        symbol: str = pos.symbol
        symbol_spec, filling_mode = self._get_cached_symbol_spec(symbol)
        pos_volume: float = pos.volume
        raw_close = pos_volume if portion is None else pos_volume * portion
        close_volume = self._normalize_close_volume(raw_close, symbol_spec, portion, pos_volume, symbol)

        tick = tick_cache.get(symbol)
        if tick is None:
            tick = mt.symbol_info_tick(symbol)
            if tick is None:
                logger.error(f"CloseFail sym={symbol} | t={ticket} | reason=tick_unavailable")
                return False, None
            tick_cache[symbol] = tick

        pos_type: int = pos.type
        close_type = OrderType.SELL if pos_type == mt.POSITION_TYPE_BUY else OrderType.BUY
        close_price = tick.bid if close_type == OrderType.SELL else tick.ask  # type: ignore[union-attr]

        order: MT5Request = {
            "action": TradeAction.DEAL,
            "symbol": symbol,
            "volume": float(close_volume),
            "type": close_type,
            "position": ticket,
            "price": close_price,
            "magic": pos.magic,
            "comment": f"Close {portion * 100 if portion else 100:.0f}%",
            "type_filling": filling_mode,
            "type_time": TimeInForce.GTC,
        }

        result = self._order_send_with_retry(order)
        ok, error_msg = self._validate_order_result(result)
        if not ok:
            logger.error(f"CloseFail t={ticket} | err={error_msg}")
            return False, None

        deal_id: int | None = getattr(result, "deal", None) or None
        if deal_id is None:
            deal_id = self._recover_close_deal_id(
                ticket=ticket,
                close_type=close_type,
                close_volume=float(close_volume),
                volume_step=float(symbol_spec.volume_step),
            )
            if deal_id is None:
                logger.warning(f"CloseWarn sym={symbol} | t={ticket} | reason=deal_id_unavailable")
            else:
                logger.info(f"CloseDealRecovered sym={symbol} | t={ticket} | deal={deal_id}")

        logger.info(f"CloseOK t={ticket} | vol={close_volume:.2f} | deal={deal_id}")
        return True, deal_id

    def modify_position_sl_tp(self, ticket: int, sl: float | None, tp: float | None) -> bool:
        positions = mt.positions_get(ticket=ticket)
        if not positions:
            logger.error(f"ModFail t={ticket} | reason=position_not_found")
            return False

        pos = positions[0]
        symbol_spec, _ = self._get_cached_symbol_spec(pos.symbol)
        final_sl = round(sl, symbol_spec.digits) if sl is not None else pos.sl
        final_tp = round(tp, symbol_spec.digits) if tp is not None else pos.tp

        if final_sl == pos.sl and final_tp == pos.tp:
            logger.debug(f"ModSkip t={ticket} | reason=already_set")
            return True

        result = self._order_send_with_retry(
            {
                "action": TradeAction.SLTP,
                "position": ticket,
                "symbol": pos.symbol,
                "sl": final_sl,
                "tp": final_tp,
            },
            success_codes={MT5_RETCODE_DONE, self._retcode_no_changes},
        )
        if result is None:
            logger.error(f"ModFail t={ticket} | reason=no_response")
            return False
        rc = result.retcode
        if rc not in {MT5_RETCODE_DONE, self._retcode_no_changes}:
            logger.error(f"ModFail t={ticket} | err={result.comment}")
            return False
        if rc == self._retcode_no_changes:
            logger.debug(f"ModSkip t={ticket} | reason=mt5_no_changes")
            return True
        logger.info(f"ModOK t={ticket} | sl={final_sl} | tp={final_tp}")
        return True

    def cancel_order(self, ticket: int) -> bool:
        result = self._order_send_with_retry({"action": TradeAction.REMOVE, "order": ticket})
        success = bool(result and result.retcode == MT5_RETCODE_DONE)
        if not success:
            rc = result.retcode if result is not None else None
            comment = result.comment if result is not None else "no_result"
            logger.error(f"CancelFail t={ticket} | rc={rc} | err={comment}")
        return success

    def cancel_bracket_orders(
        self,
        symbols: list[str],
        magics: list[int],
        preloaded_positions: list[PositionCacheEntry] | None = None,
        preloaded_orders: list[OrderSnapshot] | None = None,
    ) -> tuple[dict[str, dict[int, int]], set[int]]:
        """Cancel opposite bracket leg after one side fills. Handles dual-fill by closing both.

        Returns (cancelled_order_counts, dual_fill_closed_tickets) — the second element lets
        the caller label positions closed by dual-fill cleanup distinctly from genuine
        external closes.
        """
        if preloaded_positions is not None:
            normalized_positions = preloaded_positions
        else:
            normalized_positions = [Position.from_mt5(p).to_cache_entry() for p in (mt.positions_get() or ())]

        if preloaded_orders is not None:
            normalized_orders = preloaded_orders
        else:
            normalized_orders = [normalize_order(o) for o in (mt.orders_get() or ())]

        empty: dict[str, dict[int, int]] = {s: dict.fromkeys(magics, 0) for s in symbols}
        if not normalized_positions or not normalized_orders:
            return empty, set()

        position_groups: dict[tuple[str, int], dict[str, list[PositionCacheEntry]]] = defaultdict(
            lambda: {"BUY": [], "SELL": []}
        )
        symbol_set = frozenset(symbols)
        magic_set = frozenset(magics)
        for pos in normalized_positions:
            sym, mag = pos["symbol"], pos["magic"]
            if sym in symbol_set and mag in magic_set:
                side = "BUY" if pos["type"] == mt.POSITION_TYPE_BUY else "SELL"
                position_groups[(sym, mag)][side].append(pos)

        dual_fill_keys, dual_fill_closed_tickets = self._resolve_dual_fills(position_groups)

        dual_cancelled = sum(
            1
            for o in normalized_orders
            if o.symbol in symbol_set
            and o.magic in magic_set
            and (o.symbol, o.magic) in dual_fill_keys
            and self.cancel_order(o.ticket)
        )
        if dual_cancelled:
            logger.info(f"DualFillCancel n={dual_cancelled}")

        position_map: dict[tuple[str, int], int] = {
            (sym, mag): (mt.POSITION_TYPE_BUY if grp["BUY"] else mt.POSITION_TYPE_SELL)
            for (sym, mag), grp in position_groups.items()
            if (sym, mag) not in dual_fill_keys and (grp["BUY"] or grp["SELL"])
        }
        if not position_map:
            return empty, dual_fill_closed_tickets

        results: dict[str, dict[int, int]] = {s: dict.fromkeys(magics, 0) for s in symbols}
        for o in normalized_orders:
            if (o.symbol, o.magic) in dual_fill_keys:
                continue
            if o.symbol not in symbol_set or o.magic not in magic_set:
                continue
            pos_type = position_map.get((o.symbol, o.magic))
            if pos_type is None:
                continue
            should_cancel = (pos_type == mt.POSITION_TYPE_BUY and o.type == mt.ORDER_TYPE_SELL_STOP) or (
                pos_type == mt.POSITION_TYPE_SELL and o.type == mt.ORDER_TYPE_BUY_STOP
            )
            if should_cancel and self.cancel_order(o.ticket):
                results[o.symbol][o.magic] += 1
                logger.debug(
                    f"OCOCancel sym={o.symbol} | m={o.magic} | ot={o.type} | t={o.ticket} | "
                    f"filled={'B' if pos_type == mt.POSITION_TYPE_BUY else 'S'}"
                )
        return results, dual_fill_closed_tickets

    def _resolve_dual_fills(
        self, position_groups: dict[tuple[str, int], dict[str, list[PositionCacheEntry]]]
    ) -> tuple[set[tuple[str, int]], set[int]]:
        dual_keys: set[tuple[str, int]] = set()
        closed_tickets: set[int] = set()
        for (symbol, magic), group in position_groups.items():
            if group["BUY"] and group["SELL"]:
                logger.warning(
                    f"DualFill sym={symbol} | m={magic} | buy={len(group['BUY'])} | sell={len(group['SELL'])}"
                )
                all_tickets = [p["ticket"] for p in group["BUY"] + group["SELL"]]
                close_results = self.close_positions(tickets=all_tickets)
                closed = [t for t, (ok, _) in close_results.items() if ok]
                if closed:
                    logger.info(f"DualFillResolved sym={symbol} | m={magic} | closed={len(closed)}/{len(all_tickets)}")
                    closed_tickets.update(closed)
                else:
                    logger.error(f"DualFillCloseFail sym={symbol} | m={magic}")
                dual_keys.add((symbol, magic))
        return dual_keys, closed_tickets

    def _recover_close_deal_id(
        self,
        ticket: int,
        close_type: OrderType,
        close_volume: float,
        volume_step: float,
        max_retries: int = 3,
    ) -> int | None:
        """Fallback deal_id lookup when order_send returns retcode DONE with deal=0."""
        target_type = int(close_type)
        volume_tolerance = max(float(volume_step) / 2.0, 1e-9)
        exit_entry_code: int | None = getattr(mt, "DEAL_ENTRY_OUT", None)

        for attempt in range(max_retries):
            deals = mt.history_deals_get(position=ticket)
            if deals:
                best_rank: tuple[int, int, int, int] | None = None
                best_deal_ticket: int | None = None
                for deal in deals:
                    rank = self._rank_deal(deal, target_type, close_volume, volume_tolerance, ticket, exit_entry_code)
                    if rank is not None and (best_rank is None or rank > best_rank):
                        best_rank = rank
                        best_deal_ticket = int(deal.ticket)
                if best_deal_ticket is not None:
                    return best_deal_ticket
            if attempt < max_retries - 1:
                time.sleep(0.02 * (attempt + 1))
        return None

    def _rank_deal(
        self,
        deal: object,
        target_type: int,
        close_volume: float,
        volume_tolerance: float,
        ticket: int,
        exit_entry_code: int | None,
    ) -> tuple[int, int, int, int] | None:
        deal_ticket = getattr(deal, "ticket", None)
        if not deal_ticket:
            return None
        if getattr(deal, "type", None) != target_type:
            return None
        deal_vol = float(getattr(deal, "volume", 0.0) or 0.0)
        if abs(deal_vol - close_volume) > volume_tolerance:
            return None

        position_match = int(getattr(deal, "position_id", None) == ticket)
        entry_match = int(exit_entry_code is not None and getattr(deal, "entry", None) == exit_entry_code)
        time_msc = getattr(deal, "time_msc", None) or (int(getattr(deal, "time", 0) or 0) * 1000)
        return (position_match, entry_match, int(time_msc), int(deal_ticket))

    def _load_positions_for_tickets(self, tickets: list[int]) -> dict[int, TradePosition] | None:
        if not tickets:
            return {}
        if len(tickets) == 1:
            rows = mt.positions_get(ticket=tickets[0])
            if rows is None:
                return None
            return {tickets[0]: rows[0]} if rows else {}
        all_pos = mt.positions_get()
        if all_pos is None:
            return None
        ticket_set = set(tickets)
        return {p.ticket: p for p in all_pos if p.ticket in ticket_set}

    def _order_send_with_retry(
        self, mt5_request: MT5EntryRequest | MT5Request, success_codes: set[int] | None = None
    ) -> OrderSendResult | None:
        """Exponential-backoff retry wrapper around mt.order_send."""
        symbol: str = str(mt5_request.get("symbol", ""))
        codes = success_codes if success_codes is not None else {MT5_RETCODE_DONE}
        result: OrderSendResult | None = None

        for attempt in range(self.max_retries):
            result = mt.order_send(mt5_request)
            if result is not None and result.retcode in codes:
                if attempt:
                    logger.info(f"OrderRetryOK sym={symbol} | try={attempt}/{self.max_retries - 1}")
                return result
            if result is not None and result.retcode not in RETRYABLE_CODES:
                logger.warning(f"OrderSendNoRetry sym={symbol} | rc={result.retcode} | err={result.comment}")
                return result
            if attempt < self.max_retries - 1:
                backoff = SYSTEM_TIMINGS.order_send_retry_backoff_seconds[attempt]
                err = result.comment if result else "MT5 returned None"
                logger.warning(
                    f"OrderSendRetry sym={symbol} | try={attempt + 1}/{self.max_retries} | "
                    f"wait_ms={backoff * 1000:.0f} | err={err}"
                )
                time.sleep(backoff)
            else:
                logger.error(
                    f"OrderSendFail sym={symbol} | tries={self.max_retries} | "
                    f"err={result.comment if result else 'MT5 returned None'}"
                )
        return result

    def _validate_order_result(self, result: OrderSendResult | None) -> tuple[bool, str]:
        if result is None:
            return False, "MT5 returned None"
        if result.retcode != MT5_RETCODE_DONE:
            return False, result.comment
        return True, ""

    def _get_cached_symbol_spec(self, symbol: str) -> tuple[SymbolSpec, OrderFilling]:
        result = SYMBOL_SPEC_CACHE.get_or_fetch(symbol)
        if result is None:
            raise RuntimeError(f"SymbolSpec unavailable for {symbol!r}")
        return result

    def _get_strategy_config(self, strategy_name: str) -> StrategyConfig[BaseStrategyParams]:
        return STRATEGY_CONFIG_REGISTRY.get_strategy_config(strategy_name)

    @staticmethod
    def _fail_entry(symbol: str, message: str) -> ExecutionResult:
        return ExecutionResult(success=False, error_message=message, request_type="entry", symbol=symbol)

    @staticmethod
    def _normalize_close_volume(
        raw: float, symbol_spec: SymbolSpec, portion: float | None, pos_volume: float, symbol: str
    ) -> float:
        step = symbol_spec.volume_step
        volume = round(raw / step) * step
        if volume <= symbol_spec.volume_min:
            logger.warning(
                f"CloseVolAdj sym={symbol} | req={raw:.4f} | portion={portion if portion is not None else 1.0:.4f} | "
                f"pos={pos_volume:.4f} | min={symbol_spec.volume_min:.4f} | action=use_min"
            )
            volume = symbol_spec.volume_min
        return min(volume, symbol_spec.volume_max, pos_volume)
