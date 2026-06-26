"""Accounting schema and projection models for deterministic replay."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from sqlalchemy import Column, DateTime, Float, Integer, MetaData, String, Table, Text

ACCOUNTING_SCHEMA_VERSION = "wp06-ledger-v1"
LEGACY_UNVERIFIABLE = "LEGACY_UNVERIFIABLE"

metadata = MetaData()


class LedgerEventKind(str, Enum):
    ORDER = "order"
    FILL = "fill"
    CASH = "cash"
    FEE = "fee"
    FUNDING = "funding"
    CORRECTION = "correction"
    MARK = "mark"
    LIFECYCLE = "lifecycle"


ledger_events = Table(
    "ledger_events",
    metadata,
    Column("sequence_no", Integer, primary_key=True, autoincrement=True),
    Column("account_id", Integer, index=True, nullable=False),
    Column("kind", String(32), index=True, nullable=False),
    Column("symbol", String(32), index=True),
    Column("occurred_at", DateTime, index=True, nullable=False),
    Column("schema_version", String(64), nullable=False, default=ACCOUNTING_SCHEMA_VERSION),
    Column("legacy_status", String(64), nullable=False, default="ACTIVE"),
    Column("source_event_id", String(128), index=True),
    Column("correlation_id", String(128), index=True),
    Column("payload_json", Text, nullable=False),
)

ledger_orders = Table(
    "ledger_orders",
    metadata,
    Column("sequence_no", Integer, primary_key=True),
    Column("request_id", String(128), index=True, nullable=False),
    Column("venue_order_id", String(128), index=True),
    Column("idempotency_key", String(128), index=True),
    Column("symbol", String(32), index=True, nullable=False),
    Column("side", String(8), nullable=False),
    Column("requested_qty", Float, nullable=False),
    Column("order_status", String(32), nullable=False),
    Column("outcome", String(64), nullable=False),
    Column("submitted_at", DateTime, nullable=False),
    Column("payload_json", Text, nullable=False),
)

ledger_fills = Table(
    "ledger_fills",
    metadata,
    Column("sequence_no", Integer, primary_key=True),
    Column("fill_id", String(128), index=True, nullable=False),
    Column("request_id", String(128), index=True),
    Column("venue_order_id", String(128), index=True),
    Column("symbol", String(32), index=True, nullable=False),
    Column("side", String(8), nullable=False),
    Column("requested_qty", Float, nullable=False),
    Column("newly_filled_qty", Float, nullable=False),
    Column("cumulative_qty", Float, nullable=False),
    Column("avg_price", Float, nullable=False),
    Column("fees_usd", Float, nullable=False),
    Column("outcome", String(64), nullable=False),
    Column("replayed_at", DateTime),
    Column("payload_json", Text, nullable=False),
)

ledger_cash_movements = Table(
    "ledger_cash_movements",
    metadata,
    Column("sequence_no", Integer, primary_key=True),
    Column("movement_type", String(32), index=True, nullable=False),
    Column("currency", String(16), nullable=False, default="USD"),
    Column("amount_usd", Float, nullable=False),
    Column("symbol", String(32), index=True),
    Column("reason", String(128), nullable=False),
    Column("payload_json", Text, nullable=False),
)

ledger_fees = Table(
    "ledger_fees",
    metadata,
    Column("sequence_no", Integer, primary_key=True),
    Column("fee_type", String(32), index=True, nullable=False),
    Column("symbol", String(32), index=True),
    Column("amount_usd", Float, nullable=False),
    Column("venue_order_id", String(128), index=True),
    Column("payload_json", Text, nullable=False),
)

ledger_funding = Table(
    "ledger_funding",
    metadata,
    Column("sequence_no", Integer, primary_key=True),
    Column("symbol", String(32), index=True, nullable=False),
    Column("funding_rate", Float, nullable=False),
    Column("amount_usd", Float, nullable=False),
    Column("funding_time", DateTime, nullable=False),
    Column("payload_json", Text, nullable=False),
)

ledger_corrections = Table(
    "ledger_corrections",
    metadata,
    Column("sequence_no", Integer, primary_key=True),
    Column("corrects_sequence_no", Integer, index=True, nullable=False),
    Column("reason", String(128), nullable=False),
    Column("payload_json", Text, nullable=False),
)

ledger_marks = Table(
    "ledger_marks",
    metadata,
    Column("sequence_no", Integer, primary_key=True),
    Column("symbol", String(32), index=True, nullable=False),
    Column("mark_price", Float, nullable=False),
    Column("marked_at", DateTime, nullable=False),
    Column("payload_json", Text, nullable=False),
)

ledger_lifecycle_events = Table(
    "ledger_lifecycle_events",
    metadata,
    Column("sequence_no", Integer, primary_key=True),
    Column("account_id", Integer, index=True, nullable=False),
    Column("state", String(32), index=True, nullable=False),
    Column("owner", String(64), nullable=False),
    Column("reason", String(128), nullable=False),
    Column("policy_version", String(64), nullable=False),
    Column("created_at", DateTime, nullable=False),
    Column("payload_json", Text, nullable=False),
)

ledger_projection_checkpoints = Table(
    "ledger_projection_checkpoints",
    metadata,
    Column("checkpoint_id", Integer, primary_key=True, autoincrement=True),
    Column("account_id", Integer, index=True, nullable=False),
    Column("upto_sequence_no", Integer, index=True, nullable=False),
    Column("projection_json", Text, nullable=False),
    Column("created_at", DateTime, nullable=False),
)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _json(data: dict[str, Any]) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), default=str)


@dataclass(frozen=True)
class LedgerEvent:
    """Generic append-only ledger record."""

    account_id: int
    kind: LedgerEventKind
    occurred_at: datetime
    payload: dict[str, Any]
    symbol: str = ""
    schema_version: str = ACCOUNTING_SCHEMA_VERSION
    legacy_status: str = "ACTIVE"
    source_event_id: str = ""
    correlation_id: str = ""
    sequence_no: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "account_id", int(self.account_id))
        object.__setattr__(self, "kind", LedgerEventKind(self.kind.value if isinstance(self.kind, LedgerEventKind) else str(self.kind)))
        object.__setattr__(self, "symbol", str(self.symbol or "").strip().upper())
        object.__setattr__(self, "schema_version", str(self.schema_version or "").strip() or ACCOUNTING_SCHEMA_VERSION)
        object.__setattr__(self, "legacy_status", str(self.legacy_status or "ACTIVE").strip() or "ACTIVE")
        object.__setattr__(self, "source_event_id", str(self.source_event_id or "").strip())
        object.__setattr__(self, "correlation_id", str(self.correlation_id or "").strip())
        if self.account_id <= 0:
            raise ValueError("account_id must be positive")
        if not isinstance(self.occurred_at, datetime):
            raise ValueError("occurred_at must be a datetime")

    def to_payload_json(self) -> str:
        return _json(self.payload)

    def to_row(self) -> dict[str, Any]:
        return {
            "account_id": self.account_id,
            "kind": self.kind.value,
            "symbol": self.symbol or None,
            "occurred_at": self.occurred_at,
            "schema_version": self.schema_version,
            "legacy_status": self.legacy_status,
            "source_event_id": self.source_event_id or None,
            "correlation_id": self.correlation_id or None,
            "payload_json": self.to_payload_json(),
        }

    @classmethod
    def from_row(cls, row: Any) -> "LedgerEvent":
        payload = row["payload_json"]
        if isinstance(payload, str):
            payload_dict = json.loads(payload)
        else:
            payload_dict = dict(payload)
        return cls(
            sequence_no=int(row["sequence_no"]),
            account_id=int(row["account_id"]),
            kind=LedgerEventKind(str(row["kind"])),
            occurred_at=row["occurred_at"],
            payload=payload_dict,
            symbol=str(row["symbol"] or ""),
            schema_version=str(row["schema_version"] or ACCOUNTING_SCHEMA_VERSION),
            legacy_status=str(row["legacy_status"] or "ACTIVE"),
            source_event_id=str(row["source_event_id"] or ""),
            correlation_id=str(row["correlation_id"] or ""),
        )


@dataclass
class PositionState:
    quantity: float = 0.0
    average_cost: float = 0.0
    realized_pnl_usd: float = 0.0
    last_mark_price: float = 0.0


@dataclass
class LedgerProjection:
    account_id: int
    cash_usd: float = 0.0
    realized_pnl_usd: float = 0.0
    unrealized_pnl_usd: float = 0.0
    equity_usd: float = 0.0
    total_fees_usd: float = 0.0
    total_funding_usd: float = 0.0
    positions: dict[str, PositionState] = field(default_factory=dict)
    marks: dict[str, float] = field(default_factory=dict)
    lifecycle_states: list[dict[str, Any]] = field(default_factory=list)
    last_sequence_no: int = 0
    legacy_unverifiable_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "account_id": self.account_id,
            "cash_usd": self.cash_usd,
            "realized_pnl_usd": self.realized_pnl_usd,
            "unrealized_pnl_usd": self.unrealized_pnl_usd,
            "equity_usd": self.equity_usd,
            "total_fees_usd": self.total_fees_usd,
            "total_funding_usd": self.total_funding_usd,
            "positions": {symbol: asdict(state) for symbol, state in self.positions.items()},
            "marks": dict(self.marks),
            "lifecycle_states": list(self.lifecycle_states),
            "last_sequence_no": self.last_sequence_no,
            "legacy_unverifiable_count": self.legacy_unverifiable_count,
        }


@dataclass(frozen=True)
class LedgerDifference:
    symbol: str
    ledger_qty: float
    external_qty: float
    checkpoint_qty: float | None
    tolerance: float


@dataclass(frozen=True)
class LedgerReconciliationReport:
    account_id: int
    status: str
    blocked_new_exposure: bool
    ledger_positions: dict[str, float]
    external_positions: dict[str, float]
    checkpoint_positions: dict[str, float]
    cash_delta_usd: float
    equity_delta_usd: float
    differences: tuple[LedgerDifference, ...]
    legacy_unverifiable_count: int
    projection_sequence_no: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "account_id": self.account_id,
            "status": self.status,
            "blocked_new_exposure": self.blocked_new_exposure,
            "ledger_positions": dict(self.ledger_positions),
            "external_positions": dict(self.external_positions),
            "checkpoint_positions": dict(self.checkpoint_positions),
            "cash_delta_usd": self.cash_delta_usd,
            "equity_delta_usd": self.equity_delta_usd,
            "differences": [asdict(diff) for diff in self.differences],
            "legacy_unverifiable_count": self.legacy_unverifiable_count,
            "projection_sequence_no": self.projection_sequence_no,
        }

    def render_text(self) -> str:
        differences = ", ".join(
            f"{diff.symbol}:{diff.ledger_qty:.8f}->{diff.external_qty:.8f}"
            for diff in self.differences
        ) or "none"
        ledger_positions = ", ".join(
            f"{symbol}={qty:.8f}" for symbol, qty in sorted(self.ledger_positions.items())
        ) or "none"
        external_positions = ", ".join(
            f"{symbol}={qty:.8f}" for symbol, qty in sorted(self.external_positions.items())
        ) or "none"
        return (
            f"account_id={self.account_id} status={self.status} blocked_new_exposure={self.blocked_new_exposure}\n"
            f"ledger_positions={ledger_positions}\n"
            f"external_positions={external_positions}\n"
            f"checkpoint_positions={self.checkpoint_positions}\n"
            f"cash_delta_usd={self.cash_delta_usd:.6f} equity_delta_usd={self.equity_delta_usd:.6f}\n"
            f"differences={differences}\n"
            f"legacy_unverifiable_count={self.legacy_unverifiable_count} projection_sequence_no={self.projection_sequence_no}"
        )
