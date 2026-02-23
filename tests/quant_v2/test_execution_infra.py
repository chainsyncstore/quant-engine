from __future__ import annotations

from quant_v2.contracts import OrderPlan
from quant_v2.execution.adapters import InMemoryPaperAdapter
from quant_v2.execution.idempotency import InMemoryIdempotencyJournal, build_idempotency_key
from quant_v2.execution.reconciler import reconcile_target_exposures


def test_build_idempotency_key_stable_for_same_payload() -> None:
    plan = OrderPlan(symbol="BTCUSDT", side="BUY", quantity=0.01)

    k1 = build_idempotency_key(user_id=123, plan=plan, epoch_minute=100)
    k2 = build_idempotency_key(user_id=123, plan=plan, epoch_minute=100)
    k3 = build_idempotency_key(user_id=999, plan=plan, epoch_minute=100)

    assert k1 == k2
    assert k1 != k3


def test_inmemory_idempotency_journal() -> None:
    journal = InMemoryIdempotencyJournal()

    assert journal.seen("k1") is False
    journal.record("k1", {"status": "ok"})
    assert journal.seen("k1") is True
    assert journal.get("k1") == {"status": "ok"}
    assert journal.size() == 1


def test_inmemory_paper_adapter_idempotent_place_order() -> None:
    adapter = InMemoryPaperAdapter()
    plan = OrderPlan(symbol="BTCUSDT", side="BUY", quantity=0.02)

    first = adapter.place_order(plan, idempotency_key="abc", mark_price=50000.0)
    second = adapter.place_order(plan, idempotency_key="abc", mark_price=51000.0)

    assert first.order_id == second.order_id
    assert first.filled_qty == second.filled_qty
    assert adapter.get_positions()["BTCUSDT"] == 0.02


def test_inmemory_paper_adapter_reduce_only_rejection_when_no_position() -> None:
    adapter = InMemoryPaperAdapter()
    plan = OrderPlan(symbol="ETHUSDT", side="SELL", quantity=0.1, reduce_only=True)

    result = adapter.place_order(plan, idempotency_key="ro")

    assert result.accepted is False
    assert result.status == "rejected"
    assert result.reason == "reduce_only_no_reducible_position"


def test_reconcile_target_exposures_generates_deltas() -> None:
    plans = reconcile_target_exposures(
        {"BTCUSDT": 0.05, "ETHUSDT": -0.03},
        current_positions_qty={"BTCUSDT": 0.002, "ETHUSDT": 0.0},
        prices={"BTCUSDT": 50000.0, "ETHUSDT": 2500.0},
        equity_usd=10000.0,
    )

    by_symbol = {plan.symbol: plan for plan in plans}
    assert by_symbol["BTCUSDT"].side == "BUY"
    assert by_symbol["ETHUSDT"].side == "SELL"
    assert by_symbol["BTCUSDT"].quantity > 0
    assert by_symbol["ETHUSDT"].quantity > 0
