"""Execution adapter interfaces and a deterministic in-memory paper adapter."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Protocol

from quant_v2.contracts import OrderPlan
from quant_v2.execution.outcomes import ExecutionOutcome, make_replay_result


@dataclass(frozen=True)
class ExecutionResult:
    """Result of an execution adapter order placement."""

    accepted: bool
    order_id: str
    idempotency_key: str
    symbol: str
    side: str
    requested_qty: float
    filled_qty: float
    avg_price: float
    status: str
    created_at: str
    reason: str = ""
    risk_policy_version: str = ""
    request_id: str = ""
    venue_order_id: str = ""
    fill_id: str = ""
    accounting_transaction_id: str = ""
    original_order_id: str = ""
    original_fill_id: str = ""
    outcome: ExecutionOutcome = ExecutionOutcome.NEW_FILL
    newly_filled_qty: float = 0.0
    replayed_at: str = ""

    def __post_init__(self) -> None:
        if not self.request_id:
            object.__setattr__(self, "request_id", str(self.idempotency_key or self.order_id or ""))
        if not self.venue_order_id:
            object.__setattr__(self, "venue_order_id", str(self.order_id or ""))
        if not self.fill_id:
            object.__setattr__(self, "fill_id", str(self.venue_order_id or self.order_id or ""))
        if not self.original_order_id:
            object.__setattr__(self, "original_order_id", str(self.venue_order_id or self.order_id or ""))
        if not self.original_fill_id:
            object.__setattr__(self, "original_fill_id", str(self.fill_id or self.order_id or ""))
        if self.accepted and self.outcome != ExecutionOutcome.IDEMPOTENT_REPLAY and self.newly_filled_qty <= 0.0:
            object.__setattr__(self, "newly_filled_qty", float(self.filled_qty))
        if self.outcome == ExecutionOutcome.IDEMPOTENT_REPLAY and self.newly_filled_qty < 0.0:
            object.__setattr__(self, "newly_filled_qty", 0.0)
        if not self.accepted and self.outcome == ExecutionOutcome.NEW_FILL:
            status = str(self.status or "").strip().lower()
            if status == "skipped":
                inferred = ExecutionOutcome.BLOCKED_PRE_ROUTE
            elif status == "cancelled":
                inferred = ExecutionOutcome.CANCELLED
            elif status in {"rejected", "no_position", "residual_supervision_required", "chase_exhausted", "error"}:
                inferred = ExecutionOutcome.ADAPTER_REJECTED
            else:
                inferred = ExecutionOutcome.UNKNOWN_REQUIRES_RECONCILIATION
            object.__setattr__(self, "outcome", inferred)

    @property
    def economic_filled_qty(self) -> float:
        if self.outcome == ExecutionOutcome.IDEMPOTENT_REPLAY:
            return max(0.0, float(self.newly_filled_qty))
        return max(0.0, float(self.newly_filled_qty or self.filled_qty))

    def replay_copy(self) -> "ExecutionResult":
        return make_replay_result(self)


class ExecutionAdapter(Protocol):
    """Exchange adapter contract for v2 execution paths."""

    def place_order(
        self,
        plan: OrderPlan,
        *,
        idempotency_key: str,
        mark_price: float | None = None,
        limit_price: float | None = None,
        post_only: bool = False,
    ) -> ExecutionResult:
        ...

    def get_positions(self) -> dict[str, float]:
        ...

    def get_open_orders(self, symbol: str | None = None) -> list[dict]:
        ...

    def cancel_all_orders(self, symbol: str) -> None:
        ...


class InMemoryPaperAdapter:
    """Simple deterministic execution adapter used for tests and shadow mode."""

    def __init__(self) -> None:
        self._orders_by_key: dict[str, ExecutionResult] = {}
        self._positions: dict[str, float] = {}
        self._open_orders: list[dict] = []
        self._seq = 0

    def place_order(
        self,
        plan: OrderPlan,
        *,
        idempotency_key: str,
        mark_price: float | None = None,
        limit_price: float | None = None,
        post_only: bool = False,
    ) -> ExecutionResult:
        existing = self._orders_by_key.get(idempotency_key)
        if existing is not None:
            return existing.replay_copy()

        current_pos = float(self._positions.get(plan.symbol, 0.0))
        side_sign = 1.0 if plan.side == "BUY" else -1.0

        accepted = True
        reason = ""
        filled_qty = plan.quantity

        if plan.reduce_only:
            if current_pos == 0.0 or current_pos * side_sign > 0.0:
                accepted = False
                reason = "reduce_only_no_reducible_position"
                filled_qty = 0.0
            else:
                filled_qty = min(plan.quantity, abs(current_pos))

        if accepted and filled_qty > 0.0:
            new_pos = current_pos + (side_sign * filled_qty)
            if abs(new_pos) < 1e-12:
                self._positions.pop(plan.symbol, None)
            else:
                self._positions[plan.symbol] = new_pos

        self._seq += 1
        result = ExecutionResult(
            accepted=accepted,
            order_id=f"paper-{self._seq}",
            idempotency_key=idempotency_key,
            symbol=plan.symbol,
            side=plan.side,
            requested_qty=plan.quantity,
            filled_qty=float(filled_qty),
            avg_price=float(mark_price or 0.0),
            status="filled" if accepted and filled_qty > 0.0 else "rejected",
            created_at=datetime.now(timezone.utc).isoformat(),
            reason=reason,
            request_id=idempotency_key,
            venue_order_id=f"paper-{self._seq}",
            fill_id=f"paper-{self._seq}:0",
            accounting_transaction_id=f"acct-paper-{self._seq}",
            outcome=ExecutionOutcome.NEW_FILL if accepted and filled_qty > 0.0 else ExecutionOutcome.ADAPTER_REJECTED,
            newly_filled_qty=float(filled_qty if accepted else 0.0),
        )
        self._orders_by_key[idempotency_key] = result
        return result

    def get_positions(self) -> dict[str, float]:
        return dict(self._positions)

    def get_open_orders(self, symbol: str | None = None) -> list[dict]:
        return [o for o in self._open_orders if symbol is None or o.get("symbol") == symbol]

    def cancel_all_orders(self, symbol: str) -> None:
        self._open_orders = [o for o in self._open_orders if o.get("symbol") != symbol]
