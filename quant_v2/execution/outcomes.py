"""Typed execution outcomes and immutable fill identifiers."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from datetime import datetime, timezone


class ExecutionOutcome(str, Enum):
    """Typed result taxonomy for execution and replay events."""

    NEW_FILL = "NEW_FILL"
    PARTIAL_FILL = "PARTIAL_FILL"
    IDEMPOTENT_REPLAY = "IDEMPOTENT_REPLAY"
    ADAPTER_REJECTED = "ADAPTER_REJECTED"
    BLOCKED_PRE_ROUTE = "BLOCKED_PRE_ROUTE"
    CANCELLED = "CANCELLED"
    UNKNOWN_REQUIRES_RECONCILIATION = "UNKNOWN_REQUIRES_RECONCILIATION"


@dataclass(frozen=True)
class ExecutionIdentifiers:
    """Immutable identifiers that travel with an execution result."""

    request_id: str
    venue_order_id: str = ""
    fill_id: str = ""
    accounting_transaction_id: str = ""
    original_order_id: str = ""
    original_fill_id: str = ""


@dataclass(frozen=True)
class ExecutionFillRecord:
    """Immutable economic fill record linked to a routed execution."""

    request_id: str
    idempotency_key: str
    symbol: str
    side: str
    requested_qty: float
    newly_filled_qty: float
    cumulative_qty: float
    avg_price: float
    fees: float
    created_at: str
    source_event: str
    policy_version: str
    model_version: str
    outcome: ExecutionOutcome
    venue_order_id: str = ""
    fill_id: str = ""
    fill_sequence: int = 0
    accounting_transaction_id: str = ""
    original_order_id: str = ""
    original_fill_id: str = ""

    def to_payload(self) -> dict[str, object]:
        return {
            "request_id": self.request_id,
            "idempotency_key": self.idempotency_key,
            "symbol": self.symbol,
            "side": self.side,
            "requested_qty": self.requested_qty,
            "newly_filled_qty": self.newly_filled_qty,
            "cumulative_qty": self.cumulative_qty,
            "avg_price": self.avg_price,
            "fees": self.fees,
            "created_at": self.created_at,
            "source_event": self.source_event,
            "policy_version": self.policy_version,
            "model_version": self.model_version,
            "outcome": self.outcome.value,
            "venue_order_id": self.venue_order_id,
            "fill_id": self.fill_id,
            "fill_sequence": self.fill_sequence,
            "accounting_transaction_id": self.accounting_transaction_id,
            "original_order_id": self.original_order_id,
            "original_fill_id": self.original_fill_id,
        }


def utc_now_iso() -> str:
    """Return a UTC timestamp string suitable for record creation."""

    return datetime.now(timezone.utc).isoformat()


def make_replay_result(result, *, replayed_at: str | None = None):
    """Return a transport-stable replay copy of an execution result."""

    return replace(
        result,
        outcome=ExecutionOutcome.IDEMPOTENT_REPLAY,
        newly_filled_qty=0.0,
        original_order_id=str(result.original_order_id or result.venue_order_id or result.order_id),
        original_fill_id=str(result.original_fill_id or result.fill_id or result.order_id),
        replayed_at=replayed_at or utc_now_iso(),
    )
