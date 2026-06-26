"""Append-only accounting store and deterministic projection/reconciliation."""

from __future__ import annotations

import json
from datetime import datetime
from dataclasses import replace
from typing import Any

from sqlalchemy import create_engine, insert, select
from sqlalchemy.engine import Engine
from sqlalchemy.pool import StaticPool

from quant_v2.accounting.models import (
    ACCOUNTING_SCHEMA_VERSION,
    LEGACY_UNVERIFIABLE,
    LedgerDifference,
    LedgerEvent,
    LedgerEventKind,
    LedgerProjection,
    LedgerReconciliationReport,
    PositionState,
    ledger_cash_movements,
    ledger_corrections,
    ledger_events,
    ledger_fees,
    ledger_fills,
    ledger_funding,
    ledger_lifecycle_events,
    ledger_marks,
    ledger_orders,
    ledger_projection_checkpoints,
    metadata,
    utc_now,
)


def _make_engine(db_url: str | Engine | None = None) -> Engine:
    if isinstance(db_url, Engine):
        return db_url
    url = str(db_url or "sqlite:///:memory:")
    if url.startswith("sqlite:///:memory:"):
        return create_engine(
            url,
            future=True,
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
    if url.startswith("sqlite:///"):
        return create_engine(url, future=True, connect_args={"check_same_thread": False})
    return create_engine(url, future=True)


def _normalize_side(side: str) -> str:
    return str(side or "").strip().upper()


def _signed_position_fill(current_qty: float, current_avg: float, side: str, qty: float, price: float) -> tuple[float, float, float]:
    """Apply a fill to a position and return (next_qty, next_avg, realized_pnl)."""

    eps = 1e-12
    side = _normalize_side(side)
    if qty <= 0.0:
        return float(current_qty), float(current_avg), 0.0

    if side not in {"BUY", "SELL"}:
        raise ValueError(f"Unsupported side: {side!r}")

    realized = 0.0
    current_qty = float(current_qty)
    current_avg = float(current_avg)
    price = float(price)
    qty = float(qty)

    if side == "BUY":
        if current_qty >= -eps:
            total_qty = current_qty + qty
            if abs(total_qty) <= eps:
                return 0.0, 0.0, 0.0
            if current_qty > eps:
                new_avg = ((current_qty * current_avg) + (qty * price)) / total_qty
            else:
                new_avg = price
            return total_qty, new_avg, 0.0

        close_qty = min(abs(current_qty), qty)
        realized += (current_avg - price) * close_qty
        remaining = qty - close_qty
        short_remaining = abs(current_qty) - close_qty
        if remaining <= eps:
            next_qty = -short_remaining
            next_avg = current_avg if abs(next_qty) > eps else 0.0
            return next_qty, next_avg, realized
        return remaining, price, realized

    if current_qty <= eps:
        total_qty = current_qty - qty
        if abs(total_qty) <= eps:
            return 0.0, 0.0, 0.0
        if current_qty < -eps:
            new_avg = ((abs(current_qty) * current_avg) + (qty * price)) / abs(total_qty)
        else:
            new_avg = price
        return total_qty, new_avg, 0.0

    close_qty = min(current_qty, qty)
    realized += (price - current_avg) * close_qty
    remaining = qty - close_qty
    long_remaining = current_qty - close_qty
    if remaining <= eps:
        next_qty = long_remaining
        next_avg = current_avg if abs(next_qty) > eps else 0.0
        return next_qty, next_avg, realized
    return -remaining, price, realized


class AccountingStore:
    """SQLite-backed append-only accounting store."""

    def __init__(self, db_url: str | Engine | None = None) -> None:
        self.engine = _make_engine(db_url)
        metadata.create_all(self.engine)

    def append_event(self, event: LedgerEvent) -> LedgerEvent:
        with self.engine.begin() as conn:
            return self._append_event(conn, event)

    def _append_event(self, conn, event: LedgerEvent) -> LedgerEvent:
        row = event.to_row()
        payload_json = row["payload_json"]
        if event.source_event_id:
            existing = conn.execute(
                select(ledger_events).where(
                    ledger_events.c.account_id == event.account_id,
                    ledger_events.c.source_event_id == event.source_event_id,
                )
            ).mappings().first()
            if existing is not None:
                return LedgerEvent.from_row(existing)
        result = conn.execute(insert(ledger_events).values(**row))
        sequence_no = int(result.inserted_primary_key[0])
        persisted = LedgerEvent(
            sequence_no=sequence_no,
            account_id=event.account_id,
            kind=event.kind,
            occurred_at=event.occurred_at,
            payload=event.payload,
            symbol=event.symbol,
            schema_version=event.schema_version,
            legacy_status=event.legacy_status,
            source_event_id=event.source_event_id,
            correlation_id=event.correlation_id,
        )
        self._insert_kind_row(conn, persisted, payload_json)
        return persisted

    def _insert_kind_row(self, conn, event: LedgerEvent, payload_json: str) -> None:
        payload = dict(event.payload)
        kind = event.kind
        if kind == LedgerEventKind.ORDER:
            conn.execute(
                insert(ledger_orders).values(
                    sequence_no=event.sequence_no,
                    request_id=str(payload.get("request_id", "")),
                    venue_order_id=str(payload.get("venue_order_id", "")),
                    idempotency_key=str(payload.get("idempotency_key", "")),
                    symbol=event.symbol,
                    side=str(payload.get("side", "")),
                    requested_qty=float(payload.get("requested_qty", 0.0)),
                    order_status=str(payload.get("order_status", "")),
                    outcome=str(payload.get("outcome", "")),
                    submitted_at=payload.get("submitted_at", event.occurred_at),
                    payload_json=payload_json,
                )
            )
        elif kind == LedgerEventKind.FILL:
            conn.execute(
                insert(ledger_fills).values(
                    sequence_no=event.sequence_no,
                    fill_id=str(payload.get("fill_id", "")),
                    request_id=str(payload.get("request_id", "")),
                    venue_order_id=str(payload.get("venue_order_id", "")),
                    symbol=event.symbol,
                    side=str(payload.get("side", "")),
                    requested_qty=float(payload.get("requested_qty", 0.0)),
                    newly_filled_qty=float(payload.get("newly_filled_qty", 0.0)),
                    cumulative_qty=float(payload.get("cumulative_qty", payload.get("newly_filled_qty", 0.0))),
                    avg_price=float(payload.get("avg_price", 0.0)),
                    fees_usd=float(payload.get("fees_usd", 0.0)),
                    outcome=str(payload.get("outcome", "")),
                    replayed_at=payload.get("replayed_at"),
                    payload_json=payload_json,
                )
            )
        elif kind == LedgerEventKind.CASH:
            conn.execute(
                insert(ledger_cash_movements).values(
                    sequence_no=event.sequence_no,
                    movement_type=str(payload.get("movement_type", "")),
                    currency=str(payload.get("currency", "USD")),
                    amount_usd=float(payload.get("amount_usd", 0.0)),
                    symbol=event.symbol or None,
                    reason=str(payload.get("reason", "")),
                    payload_json=payload_json,
                )
            )
        elif kind == LedgerEventKind.FEE:
            conn.execute(
                insert(ledger_fees).values(
                    sequence_no=event.sequence_no,
                    fee_type=str(payload.get("fee_type", "fee")),
                    symbol=event.symbol or None,
                    amount_usd=float(payload.get("amount_usd", 0.0)),
                    venue_order_id=str(payload.get("venue_order_id", "")),
                    payload_json=payload_json,
                )
            )
        elif kind == LedgerEventKind.FUNDING:
            conn.execute(
                insert(ledger_funding).values(
                    sequence_no=event.sequence_no,
                    symbol=event.symbol,
                    funding_rate=float(payload.get("funding_rate", 0.0)),
                    amount_usd=float(payload.get("amount_usd", 0.0)),
                    funding_time=payload.get("funding_time", event.occurred_at),
                    payload_json=payload_json,
                )
            )
        elif kind == LedgerEventKind.CORRECTION:
            conn.execute(
                insert(ledger_corrections).values(
                    sequence_no=event.sequence_no,
                    corrects_sequence_no=int(payload.get("corrects_sequence_no", 0)),
                    reason=str(payload.get("reason", "")),
                    payload_json=payload_json,
                )
            )
        elif kind == LedgerEventKind.MARK:
            conn.execute(
                insert(ledger_marks).values(
                    sequence_no=event.sequence_no,
                    symbol=event.symbol,
                    mark_price=float(payload.get("mark_price", 0.0)),
                    marked_at=payload.get("marked_at", event.occurred_at),
                    payload_json=payload_json,
                )
            )
        elif kind == LedgerEventKind.LIFECYCLE:
            conn.execute(
                insert(ledger_lifecycle_events).values(
                    sequence_no=event.sequence_no,
                    account_id=event.account_id,
                    state=str(payload.get("state", "")),
                    owner=str(payload.get("owner", "")),
                    reason=str(payload.get("reason", "")),
                    policy_version=str(payload.get("policy_version", ACCOUNTING_SCHEMA_VERSION)),
                    created_at=payload.get("created_at", event.occurred_at),
                    payload_json=payload_json,
                )
            )
        else:
            raise ValueError(f"Unsupported ledger kind: {kind!r}")

    def append_order(self, *, account_id: int, symbol: str, request_id: str, side: str, requested_qty: float, order_status: str, outcome: str, idempotency_key: str = "", venue_order_id: str = "", occurred_at: datetime | None = None, source_event_id: str = "", correlation_id: str = "", legacy_status: str = "ACTIVE", extra: dict[str, Any] | None = None) -> LedgerEvent:
        occurred_at = occurred_at or utc_now()
        payload = {
            "request_id": request_id,
            "venue_order_id": venue_order_id,
            "idempotency_key": idempotency_key,
            "side": side,
            "requested_qty": requested_qty,
            "order_status": order_status,
            "outcome": outcome,
            "submitted_at": occurred_at,
        }
        if extra:
            payload.update(extra)
        event = LedgerEvent(
            account_id=account_id,
            kind=LedgerEventKind.ORDER,
            occurred_at=occurred_at,
            payload=payload,
            symbol=symbol,
            legacy_status=legacy_status,
            source_event_id=source_event_id,
            correlation_id=correlation_id,
        )
        return self.append_event(event)

    def append_fill(self, *, account_id: int, symbol: str, side: str, requested_qty: float, newly_filled_qty: float, cumulative_qty: float, avg_price: float, fees_usd: float = 0.0, fill_id: str = "", request_id: str = "", venue_order_id: str = "", outcome: str = "NEW_FILL", replayed_at: datetime | None = None, occurred_at: datetime | None = None, source_event_id: str = "", correlation_id: str = "", legacy_status: str = "ACTIVE", extra: dict[str, Any] | None = None) -> LedgerEvent:
        occurred_at = occurred_at or utc_now()
        payload = {
            "fill_id": fill_id,
            "request_id": request_id,
            "venue_order_id": venue_order_id,
            "side": side,
            "requested_qty": requested_qty,
            "newly_filled_qty": newly_filled_qty,
            "cumulative_qty": cumulative_qty,
            "avg_price": avg_price,
            "fees_usd": fees_usd,
            "outcome": outcome,
            "replayed_at": replayed_at,
        }
        if extra:
            payload.update(extra)
        event = LedgerEvent(
            account_id=account_id,
            kind=LedgerEventKind.FILL,
            occurred_at=occurred_at,
            payload=payload,
            symbol=symbol,
            legacy_status=legacy_status,
            source_event_id=source_event_id,
            correlation_id=correlation_id,
        )
        return self.append_event(event)

    def append_cash_movement(self, *, account_id: int, amount_usd: float, movement_type: str, reason: str, currency: str = "USD", symbol: str = "", occurred_at: datetime | None = None, source_event_id: str = "", correlation_id: str = "", legacy_status: str = "ACTIVE") -> LedgerEvent:
        event = LedgerEvent(
            account_id=account_id,
            kind=LedgerEventKind.CASH,
            occurred_at=occurred_at or utc_now(),
            payload={
                "amount_usd": amount_usd,
                "movement_type": movement_type,
                "reason": reason,
                "currency": currency,
            },
            symbol=symbol,
            legacy_status=legacy_status,
            source_event_id=source_event_id,
            correlation_id=correlation_id,
        )
        return self.append_event(event)

    def append_fee(self, *, account_id: int, amount_usd: float, fee_type: str, symbol: str = "", venue_order_id: str = "", occurred_at: datetime | None = None, source_event_id: str = "", correlation_id: str = "", legacy_status: str = "ACTIVE") -> LedgerEvent:
        event = LedgerEvent(
            account_id=account_id,
            kind=LedgerEventKind.FEE,
            occurred_at=occurred_at or utc_now(),
            payload={"amount_usd": amount_usd, "fee_type": fee_type, "venue_order_id": venue_order_id},
            symbol=symbol,
            legacy_status=legacy_status,
            source_event_id=source_event_id,
            correlation_id=correlation_id,
        )
        return self.append_event(event)

    def append_funding(self, *, account_id: int, symbol: str, funding_rate: float, amount_usd: float, funding_time: datetime | None = None, occurred_at: datetime | None = None, source_event_id: str = "", correlation_id: str = "", legacy_status: str = "ACTIVE") -> LedgerEvent:
        occurred_at = occurred_at or utc_now()
        funding_time = funding_time or occurred_at
        event = LedgerEvent(
            account_id=account_id,
            kind=LedgerEventKind.FUNDING,
            occurred_at=occurred_at,
            payload={
                "funding_rate": funding_rate,
                "amount_usd": amount_usd,
                "funding_time": funding_time,
            },
            symbol=symbol,
            legacy_status=legacy_status,
            source_event_id=source_event_id,
            correlation_id=correlation_id,
        )
        return self.append_event(event)

    def append_correction(self, *, account_id: int, corrects_sequence_no: int, reason: str, occurred_at: datetime | None = None, source_event_id: str = "", correlation_id: str = "", delta: dict[str, Any] | None = None, legacy_status: str = "ACTIVE") -> LedgerEvent:
        payload = {
            "corrects_sequence_no": corrects_sequence_no,
            "reason": reason,
        }
        if delta:
            payload.update(delta)
        event = LedgerEvent(
            account_id=account_id,
            kind=LedgerEventKind.CORRECTION,
            occurred_at=occurred_at or utc_now(),
            payload=payload,
            legacy_status=legacy_status,
            source_event_id=source_event_id,
            correlation_id=correlation_id,
        )
        return self.append_event(event)

    def append_mark(self, *, account_id: int, symbol: str, mark_price: float, marked_at: datetime | None = None, source_event_id: str = "", correlation_id: str = "", legacy_status: str = "ACTIVE") -> LedgerEvent:
        marked_at = marked_at or utc_now()
        event = LedgerEvent(
            account_id=account_id,
            kind=LedgerEventKind.MARK,
            occurred_at=marked_at,
            payload={"mark_price": mark_price, "marked_at": marked_at},
            symbol=symbol,
            legacy_status=legacy_status,
            source_event_id=source_event_id,
            correlation_id=correlation_id,
        )
        return self.append_event(event)

    def append_lifecycle_event(self, *, account_id: int, state: str, owner: str, reason: str, policy_version: str, created_at: datetime | None = None, source_event_id: str = "", correlation_id: str = "", legacy_status: str = "ACTIVE") -> LedgerEvent:
        created_at = created_at or utc_now()
        event = LedgerEvent(
            account_id=account_id,
            kind=LedgerEventKind.LIFECYCLE,
            occurred_at=created_at,
            payload={
                "state": state,
                "owner": owner,
                "reason": reason,
                "policy_version": policy_version,
                "created_at": created_at,
            },
            legacy_status=legacy_status,
            source_event_id=source_event_id,
            correlation_id=correlation_id,
        )
        return self.append_event(event)

    def load_events(self, account_id: int, upto_sequence_no: int | None = None) -> list[LedgerEvent]:
        stmt = select(ledger_events).where(ledger_events.c.account_id == account_id).order_by(ledger_events.c.occurred_at, ledger_events.c.sequence_no)
        if upto_sequence_no is not None:
            stmt = stmt.where(ledger_events.c.sequence_no <= int(upto_sequence_no))
        with self.engine.begin() as conn:
            rows = conn.execute(stmt).mappings().all()
        return [LedgerEvent.from_row(row) for row in rows]

    def replay_projection(self, account_id: int, upto_sequence_no: int | None = None) -> LedgerProjection:
        projection = LedgerProjection(account_id=account_id)
        for event in self.load_events(account_id, upto_sequence_no):
            self._apply_event(projection, event)
        self._refresh_equity(projection)
        return projection

    def save_checkpoint(self, projection: LedgerProjection) -> int:
        with self.engine.begin() as conn:
            return self._save_checkpoint(conn, projection)

    def _save_checkpoint(self, conn, projection: LedgerProjection) -> int:
        payload = json.dumps(projection.to_dict(), sort_keys=True, separators=(",", ":"), default=str)
        result = conn.execute(
            insert(ledger_projection_checkpoints).values(
                account_id=projection.account_id,
                upto_sequence_no=projection.last_sequence_no,
                projection_json=payload,
                created_at=utc_now(),
            )
        )
        return int(result.inserted_primary_key[0])

    def append_event_and_checkpoint(self, event: LedgerEvent, projection: LedgerProjection) -> tuple[LedgerEvent, int]:
        """Persist a ledger event and matching checkpoint in one transaction."""

        previous_sequence_no = projection.last_sequence_no
        try:
            with self.engine.begin() as conn:
                persisted = self._append_event(conn, event)
                projection.last_sequence_no = persisted.sequence_no
                checkpoint_id = self._save_checkpoint(conn, projection)
            return persisted, checkpoint_id
        except Exception:
            projection.last_sequence_no = previous_sequence_no
            raise

    def import_historical_events(
        self,
        events: list[LedgerEvent | dict[str, Any]],
        *,
        legacy_cutover_at: datetime,
    ) -> list[LedgerEvent]:
        """Import historical events while marking pre-cutover history unverifiable."""

        imported: list[LedgerEvent] = []
        for raw in events:
            if isinstance(raw, LedgerEvent):
                event = raw
            else:
                event = LedgerEvent(
                    account_id=int(raw["account_id"]),
                    kind=LedgerEventKind(str(raw["kind"])),
                    occurred_at=raw["occurred_at"],
                    payload=dict(raw["payload"]),
                    symbol=str(raw.get("symbol", "")),
                    schema_version=str(raw.get("schema_version", ACCOUNTING_SCHEMA_VERSION)),
                    legacy_status=str(raw.get("legacy_status", "ACTIVE")),
                    source_event_id=str(raw.get("source_event_id", "")),
                    correlation_id=str(raw.get("correlation_id", "")),
                )
            if event.occurred_at < legacy_cutover_at:
                event = replace(event, legacy_status=LEGACY_UNVERIFIABLE)
            imported.append(self.append_event(event))
        return imported

    def latest_checkpoint(self, account_id: int) -> LedgerProjection | None:
        stmt = (
            select(ledger_projection_checkpoints)
            .where(ledger_projection_checkpoints.c.account_id == account_id)
            .order_by(ledger_projection_checkpoints.c.upto_sequence_no.desc())
            .limit(1)
        )
        with self.engine.begin() as conn:
            row = conn.execute(stmt).mappings().first()
        if row is None:
            return None
        payload = json.loads(row["projection_json"])
        projection = LedgerProjection(account_id=int(payload["account_id"]))
        projection.cash_usd = float(payload["cash_usd"])
        projection.realized_pnl_usd = float(payload["realized_pnl_usd"])
        projection.unrealized_pnl_usd = float(payload["unrealized_pnl_usd"])
        projection.equity_usd = float(payload["equity_usd"])
        projection.total_fees_usd = float(payload["total_fees_usd"])
        projection.total_funding_usd = float(payload["total_funding_usd"])
        projection.positions = {
            symbol: PositionState(**state) for symbol, state in payload.get("positions", {}).items()
        }
        projection.marks = {symbol: float(price) for symbol, price in payload.get("marks", {}).items()}
        projection.lifecycle_states = list(payload.get("lifecycle_states", []))
        projection.last_sequence_no = int(payload.get("last_sequence_no", 0))
        projection.legacy_unverifiable_count = int(payload.get("legacy_unverifiable_count", 0))
        return projection

    def reconcile(
        self,
        account_id: int,
        *,
        adapter_positions: dict[str, float],
        checkpoint: LedgerProjection | None = None,
        symbol_tolerances: dict[str, float] | None = None,
        cash_tolerance: float = 0.01,
        equity_tolerance: float = 0.01,
        open_orders: dict[str, float] | None = None,
    ) -> LedgerReconciliationReport:
        projection = self.replay_projection(account_id)
        checkpoint = checkpoint or self.latest_checkpoint(account_id)
        symbol_tolerances = {str(k).upper(): float(v) for k, v in (symbol_tolerances or {}).items()}
        external_positions = {str(k).upper(): float(v) for k, v in adapter_positions.items()}
        checkpoint_positions = {
            symbol: float(state.quantity)
            for symbol, state in (checkpoint.positions.items() if checkpoint else {})
        }

        differences: list[LedgerDifference] = []
        all_symbols = set(projection.positions) | set(external_positions) | set(checkpoint_positions)
        for symbol in sorted(all_symbols):
            ledger_qty = float(projection.positions.get(symbol, PositionState()).quantity)
            external_qty = float(external_positions.get(symbol, 0.0))
            checkpoint_qty = checkpoint_positions.get(symbol)
            tolerance = symbol_tolerances.get(symbol, 1e-12)
            if abs(ledger_qty - external_qty) > tolerance:
                differences.append(
                    LedgerDifference(
                        symbol=symbol,
                        ledger_qty=ledger_qty,
                        external_qty=external_qty,
                        checkpoint_qty=checkpoint_qty,
                        tolerance=tolerance,
                    )
                )

        cash_delta = 0.0
        equity_delta = 0.0
        if checkpoint is not None:
            cash_delta = projection.cash_usd - float(checkpoint.cash_usd)
            equity_delta = projection.equity_usd - float(checkpoint.equity_usd)

        blocked = bool(differences)
        if abs(cash_delta) > cash_tolerance or abs(equity_delta) > equity_tolerance:
            blocked = True

        if open_orders:
            # open orders do not themselves change ledger state, but they signal unresolved exposure
            for symbol, qty in open_orders.items():
                if abs(float(qty)) > 1e-12 and symbol.upper() not in external_positions:
                    blocked = True

        status = "BLOCKED" if blocked else "OK"
        return LedgerReconciliationReport(
            account_id=account_id,
            status=status,
            blocked_new_exposure=blocked,
            ledger_positions={symbol: state.quantity for symbol, state in projection.positions.items()},
            external_positions=external_positions,
            checkpoint_positions=checkpoint_positions,
            cash_delta_usd=float(cash_delta),
            equity_delta_usd=float(equity_delta),
            differences=tuple(differences),
            legacy_unverifiable_count=projection.legacy_unverifiable_count,
            projection_sequence_no=projection.last_sequence_no,
        )

    def _apply_event(self, projection: LedgerProjection, event: LedgerEvent) -> None:
        projection.last_sequence_no = event.sequence_no
        if event.legacy_status == LEGACY_UNVERIFIABLE:
            projection.legacy_unverifiable_count += 1

        payload = event.payload
        kind = event.kind
        if kind == LedgerEventKind.CASH:
            amount = float(payload.get("amount_usd", 0.0))
            projection.cash_usd += amount
        elif kind == LedgerEventKind.FEE:
            amount = float(payload.get("amount_usd", 0.0))
            projection.cash_usd -= amount
            projection.total_fees_usd += amount
        elif kind == LedgerEventKind.FUNDING:
            amount = float(payload.get("amount_usd", 0.0))
            projection.cash_usd += amount
            projection.total_funding_usd += amount
        elif kind == LedgerEventKind.MARK:
            symbol = event.symbol
            mark_price = float(payload.get("mark_price", 0.0))
            projection.marks[symbol] = mark_price
            if symbol in projection.positions:
                projection.positions[symbol].last_mark_price = mark_price
        elif kind == LedgerEventKind.LIFECYCLE:
            projection.lifecycle_states.append(
                {
                    "sequence_no": event.sequence_no,
                    "account_id": event.account_id,
                    "state": str(payload.get("state", "")),
                    "owner": str(payload.get("owner", "")),
                    "reason": str(payload.get("reason", "")),
                    "policy_version": str(payload.get("policy_version", ACCOUNTING_SCHEMA_VERSION)),
                    "created_at": str(payload.get("created_at", event.occurred_at.isoformat())),
                }
            )
        elif kind in {LedgerEventKind.ORDER, LedgerEventKind.FILL, LedgerEventKind.CORRECTION}:
            self._apply_trade_like_event(projection, event)
        else:
            raise ValueError(f"Unsupported ledger event kind: {kind!r}")

    def _apply_trade_like_event(self, projection: LedgerProjection, event: LedgerEvent) -> None:
        payload = event.payload
        symbol = event.symbol
        if not symbol and event.kind != LedgerEventKind.CORRECTION:
            raise ValueError("Trade-like events require a symbol")

        if event.kind == LedgerEventKind.ORDER:
            return

        if event.kind == LedgerEventKind.CORRECTION:
            delta_cash = float(payload.get("delta_cash_usd", 0.0))
            delta_realized = float(payload.get("delta_realized_pnl_usd", 0.0))
            projection.cash_usd += delta_cash
            projection.realized_pnl_usd += delta_realized
            delta_positions = payload.get("delta_positions", {})
            if isinstance(delta_positions, dict):
                for raw_symbol, delta_state in delta_positions.items():
                    clean_symbol = str(raw_symbol).upper()
                    state = projection.positions.setdefault(clean_symbol, PositionState())
                    state.quantity += float(delta_state.get("quantity", 0.0))
                    state.average_cost = float(delta_state.get("average_cost", state.average_cost))
                    state.realized_pnl_usd += float(delta_state.get("realized_pnl_usd", 0.0))
                    if abs(state.quantity) <= 1e-12:
                        projection.positions.pop(clean_symbol, None)
            return

        side = _normalize_side(str(payload.get("side", "")))
        filled_qty = float(payload.get("newly_filled_qty", payload.get("cumulative_qty", 0.0)))
        avg_price = float(payload.get("avg_price", 0.0))
        fees = float(payload.get("fees_usd", 0.0))
        state = projection.positions.setdefault(symbol, PositionState())
        next_qty, next_avg, realized = _signed_position_fill(
            state.quantity,
            state.average_cost,
            side,
            filled_qty,
            avg_price,
        )
        projection.cash_usd += (-filled_qty * avg_price) if side == "BUY" else (filled_qty * avg_price)
        projection.cash_usd -= fees
        projection.realized_pnl_usd += realized
        projection.total_fees_usd += fees
        state.quantity = next_qty
        state.average_cost = next_avg
        state.realized_pnl_usd += realized
        state.last_mark_price = avg_price
        if abs(state.quantity) <= 1e-12:
            projection.positions.pop(symbol, None)

    def _refresh_equity(self, projection: LedgerProjection) -> None:
        unrealized = 0.0
        for symbol, state in projection.positions.items():
            mark = projection.marks.get(symbol, state.last_mark_price)
            if mark <= 0.0:
                continue
            if state.quantity >= 0:
                unrealized += (mark - state.average_cost) * state.quantity
            else:
                unrealized += (state.average_cost - mark) * abs(state.quantity)
        projection.unrealized_pnl_usd = unrealized
        projection.equity_usd = projection.cash_usd + unrealized


def build_legacy_unverifiable_projection(projection: LedgerProjection) -> LedgerProjection:
    """Mark a projection as containing legacy unverifiable history."""

    projection.legacy_unverifiable_count += 1
    return projection
