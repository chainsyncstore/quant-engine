from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

from quant_v2.research.model_evaluator import (
    EvaluationPolicy,
    RuntimeBlockers,
    decide_promotion,
    record_shadow_decision,
    resolve_due_shadow_decisions,
    summarize_quarantine,
)


def _payload(symbol: str, signal: str, price: float, ts: datetime) -> dict:
    return {
        "timestamp": ts.isoformat(),
        "symbol": symbol,
        "signal": signal,
        "probability": 0.72 if signal == "BUY" else 0.28,
        "_buy_th": 0.58,
        "_sell_th": 0.42,
        "close_price": price,
        "reason": "test",
    }


def test_shadow_decisions_resolve_and_summarize_candidate_edge(tmp_path: Path) -> None:
    db_path = tmp_path / "eval.db"
    now = datetime(2026, 6, 10, tzinfo=timezone.utc)
    policy = EvaluationPolicy(
        threshold_tuning_hours=72,
        promotion_window_hours=168,
        min_resolved_decisions=2,
        min_actionable_decisions=2,
        min_symbols=2,
        min_edge_bps=10,
        max_drawdown_worse_bps=50,
        round_trip_cost_bps=0,
        max_symbol_concentration=0.60,
    )

    rows = [
        ("candidate", "candidate_a", "BTCUSDT", "BUY", 100.0),
        ("candidate", "candidate_a", "ETHUSDT", "BUY", 100.0),
        ("incumbent", "active_a", "BTCUSDT", "HOLD", 100.0),
        ("incumbent", "active_a", "ETHUSDT", "HOLD", 100.0),
    ]
    for idx, (role, model_id, symbol, signal, entry) in enumerate(rows):
        evaluated_at = now - timedelta(hours=168) + timedelta(minutes=idx)
        assert record_shadow_decision(
            db_path,
            quarantine_version_id="candidate_a",
            model_version_id=model_id,
            baseline_version_id="active_a",
            decision_role=role,
            payload=_payload(symbol, signal, entry, evaluated_at),
            horizon_hours=8,
            evaluated_at=evaluated_at,
        )

    resolved = resolve_due_shadow_decisions(
        db_path,
        prices={"BTCUSDT": 104.0, "ETHUSDT": 103.0},
        now=now,
    )

    assert resolved == 4
    summary = summarize_quarantine(
        db_path,
        quarantine_version_id="candidate_a",
        policy=policy,
        now=now,
    )
    assert summary["candidate_metrics"]["actionable_decisions"] == 2
    assert summary["delta_metrics"]["net_return_bps"] > 10

    decision = decide_promotion(summary, policy=policy, runtime=RuntimeBlockers())
    assert decision.promotion_eligible is True
    assert decision.threshold_tuning_ready is True


def test_promotion_waits_for_rolling_window_even_with_good_edge(tmp_path: Path) -> None:
    db_path = tmp_path / "eval.db"
    now = datetime(2026, 6, 10, tzinfo=timezone.utc)
    policy = EvaluationPolicy(
        threshold_tuning_hours=72,
        promotion_window_hours=168,
        min_resolved_decisions=1,
        min_actionable_decisions=1,
        min_symbols=1,
        min_edge_bps=1,
        max_drawdown_worse_bps=50,
        round_trip_cost_bps=0,
        max_symbol_concentration=1.0,
    )

    for role, model_id in (
        ("candidate", "candidate_a"),
        ("incumbent", "active_a"),
    ):
        evaluated_at = now - timedelta(hours=80)
        assert record_shadow_decision(
            db_path,
            quarantine_version_id="candidate_a",
            model_version_id=model_id,
            baseline_version_id="active_a",
            decision_role=role,
            payload=_payload(
                "BTCUSDT",
                "BUY" if role == "candidate" else "HOLD",
                100.0,
                evaluated_at,
            ),
            horizon_hours=8,
            evaluated_at=evaluated_at,
        )

    resolve_due_shadow_decisions(db_path, prices={"BTCUSDT": 105.0}, now=now)
    summary = summarize_quarantine(
        db_path,
        quarantine_version_id="candidate_a",
        policy=policy,
        now=now,
    )
    decision = decide_promotion(summary, policy=policy, runtime=RuntimeBlockers())

    assert decision.threshold_tuning_ready is True
    assert decision.promotion_eligible is False
    assert any("needs_168h_window" in blocker for blocker in decision.blockers)


def test_runtime_blockers_defer_auto_promotion(tmp_path: Path) -> None:
    summary = {
        "evaluation_window_hours": 168.1,
        "candidate_metrics": {
            "resolved_decisions": 10,
            "actionable_decisions": 5,
            "symbols": 2,
            "max_symbol_concentration": 0.5,
        },
        "incumbent_metrics": {"resolved_decisions": 10},
        "delta_metrics": {
            "net_return_bps": 100,
            "mean_return_bps": 5,
            "max_drawdown_bps": 0,
        },
    }
    policy = EvaluationPolicy(
        min_resolved_decisions=10,
        min_actionable_decisions=5,
        min_symbols=2,
        min_edge_bps=25,
    )

    decision = decide_promotion(
        summary,
        policy=policy,
        runtime=RuntimeBlockers(active_sessions=1, hard_risk_pauses=1),
    )

    assert decision.promotion_eligible is False
    assert "active_sessions=1" in decision.blockers
    assert "hard_risk_pauses=1" in decision.blockers
