from __future__ import annotations

from datetime import datetime, timezone, timedelta

import pytest

from quant_v2.accounting import (
    ACCOUNTING_SCHEMA_VERSION,
    LEGACY_UNVERIFIABLE,
    AccountingStore,
)
from quant_v2.accounting.models import LedgerEvent, LedgerEventKind


def _ts(offset_seconds: int) -> datetime:
    return datetime(2026, 6, 18, 12, 0, 0, tzinfo=timezone.utc) + timedelta(seconds=offset_seconds)


def test_golden_ledger_long_add_partial_close_and_flip() -> None:
    store = AccountingStore()

    store.append_cash_movement(
        account_id=7,
        amount_usd=10_000.0,
        movement_type="deposit",
        reason="initial_funding",
        occurred_at=_ts(0),
        source_event_id="cash-1",
    )
    store.append_fill(
        account_id=7,
        symbol="BTCUSDT",
        side="BUY",
        requested_qty=1.0,
        newly_filled_qty=1.0,
        cumulative_qty=1.0,
        avg_price=100.0,
        fees_usd=1.0,
        fill_id="fill-1",
        request_id="req-1",
        venue_order_id="venue-1",
        occurred_at=_ts(1),
        source_event_id="fill-1",
    )
    store.append_fill(
        account_id=7,
        symbol="BTCUSDT",
        side="BUY",
        requested_qty=1.0,
        newly_filled_qty=1.0,
        cumulative_qty=1.0,
        avg_price=120.0,
        fees_usd=1.0,
        fill_id="fill-2",
        request_id="req-2",
        venue_order_id="venue-2",
        occurred_at=_ts(2),
        source_event_id="fill-2",
    )
    store.append_fill(
        account_id=7,
        symbol="BTCUSDT",
        side="SELL",
        requested_qty=0.5,
        newly_filled_qty=0.5,
        cumulative_qty=0.5,
        avg_price=130.0,
        fees_usd=0.5,
        fill_id="fill-3",
        request_id="req-3",
        venue_order_id="venue-3",
        occurred_at=_ts(3),
        source_event_id="fill-3",
    )
    store.append_mark(
        account_id=7,
        symbol="BTCUSDT",
        mark_price=140.0,
        marked_at=_ts(4),
        source_event_id="mark-1",
    )
    store.append_fill(
        account_id=7,
        symbol="BTCUSDT",
        side="SELL",
        requested_qty=3.0,
        newly_filled_qty=3.0,
        cumulative_qty=3.0,
        avg_price=90.0,
        fees_usd=1.0,
        fill_id="fill-4",
        request_id="req-4",
        venue_order_id="venue-4",
        occurred_at=_ts(5),
        source_event_id="fill-4",
    )

    projection = store.replay_projection(7)
    btc = projection.positions["BTCUSDT"]

    assert projection.cash_usd == pytest.approx(10_111.5)
    assert btc.quantity == pytest.approx(-1.5)
    assert btc.average_cost == pytest.approx(90.0)
    assert projection.realized_pnl_usd == pytest.approx(-20.0)
    assert projection.unrealized_pnl_usd == pytest.approx(-75.0)
    assert projection.equity_usd == pytest.approx(10_036.5)


def test_golden_ledger_funding_correction_and_legacy_import() -> None:
    store = AccountingStore()

    store.append_cash_movement(
        account_id=8,
        amount_usd=1_000.0,
        movement_type="deposit",
        reason="seed_capital",
        occurred_at=_ts(0),
        source_event_id="cash-2",
    )
    store.append_funding(
        account_id=8,
        symbol="ETHUSDT",
        funding_rate=0.0001,
        amount_usd=5.0,
        funding_time=_ts(1),
        occurred_at=_ts(1),
        source_event_id="funding-1",
    )
    store.append_fill(
        account_id=8,
        symbol="ETHUSDT",
        side="BUY",
        requested_qty=1.0,
        newly_filled_qty=1.0,
        cumulative_qty=1.0,
        avg_price=50.0,
        fees_usd=0.5,
        fill_id="fill-5",
        request_id="req-5",
        venue_order_id="venue-5",
        occurred_at=_ts(2),
        source_event_id="fill-5",
    )
    store.append_correction(
        account_id=8,
        corrects_sequence_no=3,
        reason="reverse_erroneous_fill",
        occurred_at=_ts(3),
        source_event_id="corr-1",
        delta={
            "delta_cash_usd": 50.5,
            "delta_realized_pnl_usd": 0.0,
            "delta_positions": {
                "ETHUSDT": {"quantity": -1.0, "average_cost": 0.0, "realized_pnl_usd": 0.0}
            },
        },
    )
    store.append_fill(
        account_id=8,
        symbol="ETHUSDT",
        side="SELL",
        requested_qty=1.0,
        newly_filled_qty=1.0,
        cumulative_qty=1.0,
        avg_price=75.0,
        fees_usd=0.25,
        fill_id="fill-6",
        request_id="req-6",
        venue_order_id="venue-6",
        occurred_at=_ts(4),
        source_event_id="fill-6",
        legacy_status=LEGACY_UNVERIFIABLE,
    )

    projection = store.replay_projection(8)
    assert projection.legacy_unverifiable_count == 1
    assert "ETHUSDT" in projection.positions
    eth = projection.positions["ETHUSDT"]
    assert eth.quantity == pytest.approx(-1.0)
    assert projection.total_funding_usd == pytest.approx(5.0)
    assert projection.total_fees_usd == pytest.approx(0.75)


def test_duplicate_and_out_of_order_delivery_preserve_economic_state() -> None:
    ordered = AccountingStore()
    shuffled = AccountingStore()

    events = [
        ("cash", dict(account_id=9, amount_usd=100.0, movement_type="deposit", reason="seed", occurred_at=_ts(0), source_event_id="cash-9")),
        ("buy", dict(account_id=9, symbol="BTCUSDT", side="BUY", requested_qty=1.0, newly_filled_qty=1.0, cumulative_qty=1.0, avg_price=100.0, fees_usd=1.0, fill_id="fill-9a", request_id="req-9a", venue_order_id="venue-9a", occurred_at=_ts(10), source_event_id="fill-9a")),
        ("sell", dict(account_id=9, symbol="BTCUSDT", side="SELL", requested_qty=0.5, newly_filled_qty=0.5, cumulative_qty=0.5, avg_price=110.0, fees_usd=0.5, fill_id="fill-9b", request_id="req-9b", venue_order_id="venue-9b", occurred_at=_ts(5), source_event_id="fill-9b")),
    ]

    ordered.append_cash_movement(**events[0][1])
    ordered.append_fill(**events[1][1])
    ordered.append_fill(**events[2][1])
    ordered.append_fill(**events[2][1])  # duplicate source event id should be ignored

    shuffled.append_cash_movement(**events[0][1])
    shuffled.append_fill(**events[2][1])
    shuffled.append_fill(**events[1][1])

    ordered_projection = ordered.replay_projection(9)
    shuffled_projection = shuffled.replay_projection(9)

    assert ordered_projection.cash_usd == pytest.approx(shuffled_projection.cash_usd)
    assert ordered_projection.equity_usd == pytest.approx(shuffled_projection.equity_usd)
    assert ordered_projection.positions["BTCUSDT"].quantity == pytest.approx(
        shuffled_projection.positions["BTCUSDT"].quantity
    )


def test_reconciliation_blocks_unexplained_drift_and_passes_with_tolerance() -> None:
    store = AccountingStore()
    store.append_cash_movement(
        account_id=11,
        amount_usd=500.0,
        movement_type="deposit",
        reason="seed",
        occurred_at=_ts(0),
        source_event_id="cash-11",
    )
    store.append_fill(
        account_id=11,
        symbol="BTCUSDT",
        side="BUY",
        requested_qty=1.0,
        newly_filled_qty=1.0,
        cumulative_qty=1.0,
        avg_price=100.0,
        fees_usd=0.0,
        fill_id="fill-11",
        request_id="req-11",
        venue_order_id="venue-11",
        occurred_at=_ts(1),
        source_event_id="fill-11",
    )
    store.append_mark(
        account_id=11,
        symbol="BTCUSDT",
        mark_price=105.0,
        marked_at=_ts(2),
        source_event_id="mark-11",
    )
    checkpoint = store.replay_projection(11)
    store.save_checkpoint(checkpoint)

    aligned = store.reconcile(11, adapter_positions={"BTCUSDT": 1.0}, checkpoint=checkpoint, symbol_tolerances={"BTCUSDT": 1e-6})
    drifted = store.reconcile(11, adapter_positions={"BTCUSDT": 1.1}, checkpoint=checkpoint, symbol_tolerances={"BTCUSDT": 1e-6})

    assert aligned.status == "OK"
    assert aligned.blocked_new_exposure is False
    assert drifted.status == "BLOCKED"
    assert drifted.blocked_new_exposure is True
    assert drifted.differences


def test_checkpoint_roundtrip_and_schema_version_are_stable() -> None:
    store = AccountingStore()
    store.append_cash_movement(
        account_id=12,
        amount_usd=250.0,
        movement_type="deposit",
        reason="seed",
        occurred_at=_ts(0),
        source_event_id="cash-12",
    )
    projection = store.replay_projection(12)
    checkpoint_id = store.save_checkpoint(projection)
    checkpoint = store.latest_checkpoint(12)

    assert checkpoint_id > 0
    assert checkpoint is not None
    assert checkpoint.account_id == 12
    assert ACCOUNTING_SCHEMA_VERSION == "wp06-ledger-v1"


def test_atomic_event_and_checkpoint_recovery_from_checkpoint_failure(monkeypatch) -> None:
    store = AccountingStore()
    store.append_cash_movement(
        account_id=13,
        amount_usd=1_000.0,
        movement_type="deposit",
        reason="seed",
        occurred_at=_ts(0),
        source_event_id="cash-13",
    )
    projection = store.replay_projection(13)
    event = LedgerEvent(
        account_id=13,
        kind=LedgerEventKind.FILL,
        occurred_at=_ts(1),
        payload={
            "fill_id": "fill-13",
            "request_id": "req-13",
            "venue_order_id": "venue-13",
            "side": "BUY",
            "requested_qty": 1.0,
            "newly_filled_qty": 1.0,
            "cumulative_qty": 1.0,
            "avg_price": 100.0,
            "fees_usd": 1.0,
            "outcome": "NEW_FILL",
            "replayed_at": None,
        },
        symbol="BTCUSDT",
        source_event_id="fill-13",
    )

    original_save_checkpoint = store._save_checkpoint
    call_count = {"n": 0}

    def fail_once(conn, projection_arg):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise RuntimeError("simulated checkpoint failure")
        return original_save_checkpoint(conn, projection_arg)

    monkeypatch.setattr(store, "_save_checkpoint", fail_once)

    with pytest.raises(RuntimeError):
        store.append_event_and_checkpoint(event, projection)

    assert projection.last_sequence_no == 1
    assert len(store.load_events(13)) == 1

    persisted, checkpoint_id = store.append_event_and_checkpoint(event, projection)
    assert persisted.sequence_no == 2
    assert checkpoint_id > 0

    replayed = store.replay_projection(13)
    assert replayed.positions["BTCUSDT"].quantity == pytest.approx(1.0)
    assert replayed.cash_usd == pytest.approx(899.0)
    assert len(store.load_events(13)) == 2


def test_historical_import_marks_pre_cutover_history_unverifiable_and_renders_report() -> None:
    store = AccountingStore()
    legacy_buy = LedgerEvent(
        account_id=14,
        kind=LedgerEventKind.FILL,
        occurred_at=_ts(-100),
        payload={
            "fill_id": "legacy-fill-1",
            "request_id": "legacy-req-1",
            "venue_order_id": "legacy-venue-1",
            "side": "BUY",
            "requested_qty": 1.0,
            "newly_filled_qty": 1.0,
            "cumulative_qty": 1.0,
            "avg_price": 100.0,
            "fees_usd": 0.0,
            "outcome": "NEW_FILL",
            "replayed_at": None,
        },
        symbol="BTCUSDT",
        source_event_id="legacy-fill-1",
    )
    modern_sell = LedgerEvent(
        account_id=14,
        kind=LedgerEventKind.FILL,
        occurred_at=_ts(100),
        payload={
            "fill_id": "live-fill-1",
            "request_id": "live-req-1",
            "venue_order_id": "live-venue-1",
            "side": "SELL",
            "requested_qty": 1.0,
            "newly_filled_qty": 1.0,
            "cumulative_qty": 1.0,
            "avg_price": 110.0,
            "fees_usd": 0.5,
            "outcome": "NEW_FILL",
            "replayed_at": None,
        },
        symbol="BTCUSDT",
        source_event_id="live-fill-1",
    )

    imported = store.import_historical_events([legacy_buy, modern_sell], legacy_cutover_at=_ts(0))
    assert imported[0].legacy_status == LEGACY_UNVERIFIABLE
    assert imported[1].legacy_status == "ACTIVE"

    projection = store.replay_projection(14)
    report = store.reconcile(
        14,
        adapter_positions={"BTCUSDT": 0.0},
        checkpoint=projection,
        symbol_tolerances={"BTCUSDT": 1e-6},
    )

    assert projection.legacy_unverifiable_count == 1
    assert projection.positions == {}
    rendered = report.render_text()
    assert "status=OK" in rendered
    assert "legacy_unverifiable_count=1" in rendered
