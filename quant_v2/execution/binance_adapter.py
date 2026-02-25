"""Binance-backed execution adapter for v2 order routing."""

from __future__ import annotations

from datetime import datetime, timezone
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
            # We pass limit_price here; client.close_position converts to LIMIT if provided
            raw = self.client.close_position(plan.symbol, limit_price=limit_price)
            accepted = raw is not None
            filled_qty = float(raw.get("executedQty", 0.0)) if raw else 0.0
            order_id = str(raw.get("orderId", "")) if raw else ""
            status = str(raw.get("status", "no_position")) if raw else "no_position"
            avg_price = float(raw.get("avgPrice", 0.0)) if raw else 0.0
            price = float(raw.get("price", avg_price)) if raw else avg_price
            reason = "" if accepted else "reduce_only_no_position"
            requested_qty = float(plan.quantity)
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
                        
                        # Apply symbol price filters (tick size) if necessary
                        if hasattr(self.client, "get_symbol_filters"):
                            filters = dict(self.client.get_symbol_filters(plan.symbol) or {})
                            tick_size = float(filters.get("tick_size", 0.0) or 0.0)
                            if tick_size > 0:
                                fallback_limit = round(fallback_limit / tick_size) * tick_size
                                
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

        step_size = max(float(filters.get("step_size", 0.0) or 0.0), 0.0)
        min_qty = max(float(filters.get("min_qty", 0.0) or 0.0), 0.0)
        min_notional = max(float(filters.get("min_notional", 0.0) or 0.0), 0.0)

        normalized = float(quantity)
        if step_size > 0.0:
            steps = math.floor((normalized / step_size) + 1e-12)
            normalized = max(steps * step_size, 0.0)

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
