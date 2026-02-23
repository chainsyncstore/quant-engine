"""Binance-backed execution adapter for v2 order routing."""

from __future__ import annotations

from datetime import datetime, timezone
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
        else:
            raw = self.client.place_order(plan.symbol, plan.side, plan.quantity)
            accepted = True
            filled_qty = float(raw.get("executedQty", plan.quantity))
            order_id = str(raw.get("orderId", ""))
            status = str(raw.get("status", "filled")).lower()
            avg_price = float(raw.get("avgPrice", mark_price or 0.0))
            reason = ""

        result = ExecutionResult(
            accepted=accepted,
            order_id=order_id,
            idempotency_key=idempotency_key,
            symbol=plan.symbol,
            side=plan.side,
            requested_qty=plan.quantity,
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

    @staticmethod
    def _as_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except Exception:
            return float(default)
