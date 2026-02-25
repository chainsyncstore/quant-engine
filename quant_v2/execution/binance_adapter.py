"""Binance-backed execution adapter for v2 order routing."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal, ROUND_DOWN, ROUND_HALF_UP
import math
from typing import Any

from quant.data.binance_client import BinanceClient
from quant_v2.contracts import OrderPlan
from quant_v2.execution.adapters import ExecutionResult
from quant_v2.execution.idempotency import InMemoryIdempotencyJournal


class BinanceExecutionAdapter:
    """Execution adapter that routes orders through BinanceClient."""

    def __init__(
        self,
        client: BinanceClient,
        *,
        journal: InMemoryIdempotencyJournal | None = None,
    ) -> None:
        self.client = client
        self.journal = journal or InMemoryIdempotencyJournal()

    def place_order(
        self,
        plan: OrderPlan,
        *,
        idempotency_key: str,
        mark_price: float | None = None,
        limit_price: float | None = None,
        post_only: bool = False,
    ) -> ExecutionResult:
        if self.journal.seen(idempotency_key):
            return self.journal.get(idempotency_key)

        if plan.reduce_only:
            # FIX-4: Use slippage-bounded exit instead of raw MARKET orders.
            result = self.close_position_bounded(
                plan.symbol,
                max_slippage_bps=50,
                idempotency_key=idempotency_key,
            )
            return result
        else:
            normalized_qty, skip_reason = self._normalize_quantity_with_filters(
                plan.symbol,
                quantity=float(plan.quantity),
                mark_price=float(mark_price or 0.0),
            )
            if skip_reason is not None:
                result = ExecutionResult(
                    accepted=False,
                    order_id="",
                    idempotency_key=idempotency_key,
                    symbol=plan.symbol,
                    side=plan.side,
                    requested_qty=float(plan.quantity),
                    filled_qty=0.0,
                    avg_price=float(mark_price or 0.0),
                    status="skipped",
                    created_at=datetime.now(timezone.utc).isoformat(),
                    reason=skip_reason,
                )
                self.journal.record(idempotency_key, result)
                return result

            fallback_used = False
            raw = None
            if limit_price is not None:
                try:
                    raw = self.client.place_limit_order(
                        plan.symbol, 
                        plan.side, 
                        normalized_qty, 
                        price=limit_price, 
                        post_only=post_only
                    )
                except Exception as e:
                    import logging
                    logger = logging.getLogger(__name__)
                    # If it's a -2010 "Order would immediately match and take" error, fallback to a slippage-bounded limit
                    if post_only and ("-2010" in str(e) or "immediately match" in str(e).lower()):
                        max_slippage_bps = 15.0  # Allow 15 bps (0.15%) slippage chase
                        slippage_factor = 1.0 + (max_slippage_bps / 10000.0) if plan.side.upper() == "BUY" else 1.0 - (max_slippage_bps / 10000.0)
                        
                        fallback_limit = mark_price * slippage_factor
                        
                        # Apply strict tick-size quantization via Decimal
                        fallback_limit = self._quantize_price(plan.symbol, fallback_limit)
                                
                        logger.warning(
                            "POST_ONLY order matched immediately for %s. Falling back to BOUNDED LIMIT (price=%.4f, max_slippage=15bps).", 
                            plan.symbol, 
                            fallback_limit
                        )
                        fallback_used = True
                        try:
                            # Not post_only anymore, but bounded by fallback_limit to prevent unbounded market slippage
                            raw = self.client.place_limit_order(
                                plan.symbol, 
                                plan.side, 
                                normalized_qty, 
                                price=fallback_limit, 
                                post_only=False
                            )
                        except Exception as e2:
                            logger.error("Bounded fallback limit error for %s: %s", plan.symbol, e2)
                            raise e2
                    else:
                        raise e
            else:
                raw = self.client.place_order(plan.symbol, plan.side, normalized_qty)
                
            accepted = True
            filled_qty = float(raw.get("executedQty", normalized_qty))
            order_id = str(raw.get("orderId", ""))
            status = str(raw.get("status", "filled")).lower()
            
            avg_price = float(raw.get("avgPrice", mark_price or 0.0))
            if status in {"new", "partially_filled"} and limit_price is not None and not fallback_used:
                # If order is still resting, use the limit price as the recorded avg_price 
                # to track the expected execution level.
                avg_price = limit_price
                
            reason = "fallback_to_bounded_limit" if fallback_used else ""
            requested_qty = float(normalized_qty)

        result = ExecutionResult(
            accepted=accepted,
            order_id=order_id,
            idempotency_key=idempotency_key,
            symbol=plan.symbol,
            side=plan.side,
            requested_qty=requested_qty,
            filled_qty=filled_qty,
            avg_price=avg_price,
            status=status,
            created_at=datetime.now(timezone.utc).isoformat(),
            reason=reason,
        )

        self.journal.record(idempotency_key, result)
        return result

    def close_position_bounded(
        self,
        symbol: str,
        *,
        max_slippage_bps: float = 50.0,
        idempotency_key: str = "",
        fill_check_timeout_seconds: float = 5.0,
    ) -> ExecutionResult:
        """Close a position using aggressive limit orders, not raw MARKET.

        1. Fetch current position from exchange.
        2. Fetch top-of-book bid/ask.
        3. Place a LIMIT order at bid * (1 - slippage) for sells, ask * (1 + slippage) for buys.
        4. Wait fill_check_timeout_seconds then check fill status.
        5. If unfilled, cancel and escalate to MARKET as last resort.
        """
        import time as _time

        if idempotency_key and self.journal.seen(idempotency_key):
            return self.journal.get(idempotency_key)

        positions = self.client.get_positions(symbol=symbol)
        if not positions:
            result = ExecutionResult(
                accepted=False,
                order_id="",
                idempotency_key=idempotency_key,
                symbol=symbol,
                side="SELL",
                requested_qty=0.0,
                filled_qty=0.0,
                avg_price=0.0,
                status="no_position",
                created_at=datetime.now(timezone.utc).isoformat(),
                reason="reduce_only_no_position",
            )
            if idempotency_key:
                self.journal.record(idempotency_key, result)
            return result

        pos = positions[0]
        pos_amt = float(pos.get("positionAmt", 0.0))
        if pos_amt == 0.0:
            result = ExecutionResult(
                accepted=False,
                order_id="",
                idempotency_key=idempotency_key,
                symbol=symbol,
                side="SELL",
                requested_qty=0.0,
                filled_qty=0.0,
                avg_price=0.0,
                status="no_position",
                created_at=datetime.now(timezone.utc).isoformat(),
                reason="reduce_only_no_position",
            )
            if idempotency_key:
                self.journal.record(idempotency_key, result)
            return result

        side = "SELL" if pos_amt > 0 else "BUY"
        qty = abs(pos_amt)

        # Fetch top-of-book for slippage-bounded pricing
        top = self.get_orderbook_top(symbol)
        bid, ask = top["bid"], top["ask"]

        import logging as _logging
        _logger = _logging.getLogger(__name__)

        base_slippage_bps = max_slippage_bps
        reason = "bounded_limit_exit"
        raw = None
        max_chase_attempts = 3

        for attempt in range(max_chase_attempts):
            # Re-fetch orderbook on each chase attempt for fresh pricing
            if attempt > 0:
                top = self.get_orderbook_top(symbol)
                bid, ask = top["bid"], top["ask"]

            # Widen slippage by 25bps per retry
            attempt_slippage_bps = base_slippage_bps + (attempt * 25.0)
            slippage_factor = attempt_slippage_bps / 10_000.0

            if side == "SELL":
                aggressive_price = bid * (1.0 - slippage_factor) if bid > 0 else 0.0
            else:
                aggressive_price = ask * (1.0 + slippage_factor) if ask > 0 else 0.0

            if aggressive_price <= 0:
                _logger.warning(
                    "No orderbook data for %s on chase attempt %d — cannot compute price",
                    symbol, attempt + 1,
                )
                continue

            # Apply strict tick-size quantization
            aggressive_price = self._quantize_price(symbol, aggressive_price)

            try:
                raw = self.client.place_limit_order(
                    symbol, side, qty, price=aggressive_price, post_only=False
                )
            except Exception as exc:
                _logger.error(
                    "Bounded limit exit attempt %d failed for %s: %s",
                    attempt + 1, symbol, exc,
                )
                raw = None
                continue

            # Check if immediately filled
            if str(raw.get("status", "")).upper() in {"FILLED"}:
                reason = "bounded_limit_exit"
                break

            if str(raw.get("status", "")).upper() in {"CANCELED", "EXPIRED"}:
                raw = None
                continue

            # Wait and check fill status
            order_id = raw.get("orderId")
            _time.sleep(min(fill_check_timeout_seconds, 5.0))

            open_orders = self.get_open_orders(symbol)
            still_open = any(
                str(o.get("orderId")) == str(order_id) for o in open_orders
            )
            if not still_open:
                # Order filled while we waited
                reason = "bounded_limit_exit"
                break

            # Cancel with verification before chasing
            _logger.warning(
                "Bounded exit for %s unfilled after %.1fs (attempt %d/%d) — cancelling to chase",
                symbol, fill_check_timeout_seconds, attempt + 1, max_chase_attempts,
            )
            try:
                cancel_result = self.client.cancel_order(symbol, order_id)
                cancel_status = str(cancel_result.get("status", "")).upper() if isinstance(cancel_result, dict) else ""
                if cancel_status and cancel_status != "CANCELED":
                    _logger.warning(
                        "Cancel returned status=%s for %s order %s — aborting chase to prevent zombie",
                        cancel_status, symbol, order_id,
                    )
                    reason = "cancel_verification_failed"
                    raw = None
                    break
            except Exception as cancel_exc:
                _logger.error(
                    "Cancel failed for %s order %s: %s — aborting chase to prevent zombie",
                    symbol, order_id, cancel_exc,
                )
                reason = "cancel_verification_failed"
                raw = None
                break

            raw = None  # Reset for next attempt
            reason = "bounded_limit_chase"

        if raw is None:
            # All chase attempts exhausted or aborted — return safe failure
            _logger.error(
                "Chase exhausted for %s after %d attempts — reason: %s",
                symbol, max_chase_attempts, reason,
            )
            final_reason = reason if reason != "bounded_limit_exit" else "chase_exhausted"
            result = ExecutionResult(
                accepted=False,
                order_id="",
                idempotency_key=idempotency_key,
                symbol=symbol,
                side=side,
                requested_qty=qty,
                filled_qty=0.0,
                avg_price=0.0,
                status="chase_exhausted",
                created_at=datetime.now(timezone.utc).isoformat(),
                reason=final_reason,
            )
            if idempotency_key:
                self.journal.record(idempotency_key, result)
            return result

        result = ExecutionResult(
            accepted=True,
            order_id=str(raw.get("orderId", "")),
            idempotency_key=idempotency_key,
            symbol=symbol,
            side=side,
            requested_qty=qty,
            filled_qty=float(raw.get("executedQty", qty)),
            avg_price=float(raw.get("avgPrice", aggressive_price or 0.0)),
            status=str(raw.get("status", "filled")).lower(),
            created_at=datetime.now(timezone.utc).isoformat(),
            reason=reason,
        )
        if idempotency_key:
            self.journal.record(idempotency_key, result)
        return result

    def get_positions(self) -> dict[str, float]:
        raw_positions = self.client.get_positions()
        positions: dict[str, float] = {}
        for pos in raw_positions:
            symbol = str(pos.get("symbol", "")).strip()
            if not symbol:
                continue
            amount = float(pos.get("positionAmt", 0.0))
            if amount != 0.0:
                positions[symbol] = amount
        return positions

    def get_open_orders(self, symbol: str | None = None) -> list[dict]:
        return self.client.get_open_orders(symbol=symbol)

    def cancel_all_orders(self, symbol: str) -> None:
        open_orders = self.get_open_orders(symbol)
        for order in open_orders:
            order_id = order.get("orderId")
            if order_id:
                try:
                    self.client.cancel_order(symbol, order_id)
                except Exception as e:
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.warning("Failed to cancel open order %s for %s: %s", order_id, symbol, e)

    def get_position_metrics(self) -> dict[str, dict[str, float]]:
        raw_positions = self.client.get_positions()
        metrics: dict[str, dict[str, float]] = {}
        for pos in raw_positions:
            symbol = str(pos.get("symbol", "")).strip()
            if not symbol:
                continue
            qty = float(pos.get("positionAmt", 0.0) or 0.0)
            if qty == 0.0:
                continue

            metrics[symbol] = {
                "entry_price": float(pos.get("entryPrice", 0.0) or 0.0),
                "unrealized_pnl_usd": float(pos.get("unrealizedProfit", 0.0) or 0.0),
                "mark_price": float(pos.get("markPrice", 0.0) or 0.0),
            }
        return metrics

    def get_orderbook_top(self, symbol: str) -> dict[str, float]:
        """Retrieve the best bid and best ask for a symbol.

        Returns dict with keys 'bid' and 'ask'. Falls back to mark price
        from position data if the orderbook endpoint is unavailable.
        """
        getter = getattr(self.client, "get_orderbook", None)
        if callable(getter):
            try:
                book = getter(symbol, limit=5)
                bids = book.get("bids", [])
                asks = book.get("asks", [])
                return {
                    "bid": float(bids[0][0]) if bids else 0.0,
                    "ask": float(asks[0][0]) if asks else 0.0,
                }
            except Exception:
                pass

        return {"bid": 0.0, "ask": 0.0}

    def compute_mtm_equity(
        self,
        positions: dict[str, float],
        initial_equity_usd: float = 10_000.0,
    ) -> dict[str, float]:
        """Compute bid/ask-aware MTM equity.

        Long positions valued at bid (the price you could actually sell at).
        Short positions valued at ask (the price you could actually cover at).
        Returns dict with 'bid_mtm_equity_usd', 'ask_mtm_equity_usd', 'mid_mtm_equity_usd'.
        """
        total_bid_value = 0.0
        total_ask_value = 0.0
        total_mid_value = 0.0

        for symbol, qty in positions.items():
            if qty == 0.0:
                continue

            top = self.get_orderbook_top(symbol)
            bid = top["bid"]
            ask = top["ask"]

            if bid <= 0.0 or ask <= 0.0:
                # Fallback to position metrics mark price
                metrics = self.get_position_metrics()
                mark = metrics.get(symbol, {}).get("mark_price", 0.0)
                bid = mark
                ask = mark

            mid = (bid + ask) / 2.0 if bid > 0 and ask > 0 else max(bid, ask)

            if qty > 0:
                # Long: value at bid (conservative)
                total_bid_value += qty * bid
                total_ask_value += qty * ask
            else:
                # Short: value at ask (conservative for covering)
                total_bid_value += abs(qty) * ask
                total_ask_value += abs(qty) * bid

            total_mid_value += abs(qty) * mid

        return {
            "bid_mtm_equity_usd": initial_equity_usd + total_bid_value,
            "ask_mtm_equity_usd": initial_equity_usd + total_ask_value,
            "mid_mtm_equity_usd": initial_equity_usd + total_mid_value,
        }

    def _quantize_price(self, symbol: str, price: float) -> float:
        """Round a price to the symbol's tick_size using Decimal arithmetic."""
        filters_getter = getattr(self.client, "get_symbol_filters", None)
        if not callable(filters_getter):
            return price
        try:
            filters = dict(filters_getter(symbol) or {})
        except Exception:
            return price

        raw_tick = filters.get("tick_size", 0.0)
        tick_size = Decimal(str(raw_tick if raw_tick else 0))
        if tick_size > 0:
            price_dec = Decimal(str(price))
            quantized = (price_dec / tick_size).quantize(
                Decimal("1"), rounding=ROUND_HALF_UP
            ) * tick_size
            return float(quantized)
        return price

    def _normalize_quantity_with_filters(
        self,
        symbol: str,
        *,
        quantity: float,
        mark_price: float,
    ) -> tuple[float, str | None]:
        filters_getter = getattr(self.client, "get_symbol_filters", None)
        if not callable(filters_getter):
            return float(quantity), None

        try:
            filters = dict(filters_getter(symbol) or {})
        except Exception:
            return float(quantity), None

        raw_step = filters.get("step_size", 0.0)
        step_size = Decimal(str(raw_step if raw_step else 0))
        min_qty = max(float(filters.get("min_qty", 0.0) or 0.0), 0.0)
        min_notional = max(float(filters.get("min_notional", 0.0) or 0.0), 0.0)

        qty_dec = Decimal(str(quantity))
        if step_size > 0:
            quantized = (qty_dec / step_size).quantize(
                Decimal("1"), rounding=ROUND_DOWN
            ) * step_size
            normalized = float(quantized)
        else:
            normalized = float(quantity)

        if normalized <= 0.0:
            return 0.0, "skipped_by_filter:step_size"
        if min_qty > 0.0 and normalized + 1e-12 < min_qty:
            return normalized, "skipped_by_filter:min_qty"
        if mark_price > 0.0 and min_notional > 0.0 and (normalized * mark_price) + 1e-12 < min_notional:
            return normalized, "skipped_by_filter:min_notional"
        return normalized, None

    @staticmethod
    def _as_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except Exception:
            return float(default)
