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
    ) -> ExecutionResult:
        if self.journal.seen(idempotency_key):
            return self.journal.get(idempotency_key)

        if plan.reduce_only:
            raw = self.client.close_position(plan.symbol)
            accepted = raw is not None
            filled_qty = float(raw.get("executedQty", 0.0)) if raw else 0.0
            order_id = str(raw.get("orderId", "")) if raw else ""
            status = str(raw.get("status", "no_position")) if raw else "no_position"
            avg_price = float(raw.get("avgPrice", 0.0)) if raw else 0.0
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

            raw = self.client.place_order(plan.symbol, plan.side, normalized_qty)
            accepted = True
            filled_qty = float(raw.get("executedQty", normalized_qty))
            order_id = str(raw.get("orderId", ""))
            status = str(raw.get("status", "filled")).lower()
            avg_price = float(raw.get("avgPrice", mark_price or 0.0))
            reason = ""
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
            }
        return metrics

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
