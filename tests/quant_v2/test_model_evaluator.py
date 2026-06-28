from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quant_v2.model_registry import ModelRegistry
from quant_v2.models.trainer import save_model_bundle, train
from quant_v2.research.model_evaluator import (
    EvaluationPolicy,
    RuntimeBlockers,
    decide_promotion,
    evaluate_once,
    load_evaluator_control,
    record_registry_evaluation,
    record_shadow_decision,
    resolve_due_shadow_decisions,
    summarize_quarantine,
    write_evaluator_control,
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


def _burst_shadow_rows(
    *,
    start: datetime,
    days: int = 32,
) -> list[tuple[str, str, str, str, float, datetime]]:
    rows: list[tuple[str, str, str, str, float, datetime]] = []
    for day in range(days):
        ts = start + timedelta(days=day)
        base = 100.0 + (day * 0.5)
        rows.extend(
            [
                ("candidate", "candidate_a", "BTCUSDT", "BUY", base, ts),
                ("candidate", "candidate_a", "ETHUSDT", "BUY", base + 1.0, ts),
                ("incumbent", "active_a", "BTCUSDT", "HOLD", base, ts),
                ("incumbent", "active_a", "ETHUSDT", "HOLD", base + 1.0, ts),
            ]
        )
    return rows


def _register_threshold_model(
    registry: ModelRegistry,
    artifact_dir: Path,
    *,
    version_id: str,
    threshold: float,
    monkeypatch,
) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("QUANT_IMAGE", "registry.example/quant-bot@sha256:" + "a" * 64)
    idx = pd.date_range("2025-01-01", periods=220, freq="h", tz="UTC")
    X = pd.DataFrame(
        {
            "f1": np.linspace(-2.0, 2.0, len(idx)),
            "f2": np.sin(np.linspace(0.0, 8.0, len(idx))),
            "f3": np.cos(np.linspace(0.0, 6.0, len(idx))),
        },
        index=idx,
    )
    y = pd.Series((np.sin(np.linspace(0.0, 12.0, len(idx))) > 0.0).astype(int), index=idx)
    trained = train(X, y, horizon=4, calibration_frac=0.2)
    save_model_bundle(
        trained,
        artifact_dir / "model_4m.pkl",
        metadata={
            "dataset_digest": f"{version_id}-digest",
            "threshold": threshold,
            "threshold_policy": {
                "source": "oof_dev_predictions",
                "selected_threshold": threshold,
                "selected_accuracy": 0.63,
            },
        },
    )
    registry.register_version(version_id, artifact_dir, metrics={"promotion_eligible": True})


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
    for role, model_id, symbol, signal, entry in rows:
        evaluated_at = now - timedelta(hours=168) + timedelta(minutes=0 if symbol == "BTCUSDT" else 1)
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
    diagnostics = summary["performance_diagnostics"]
    assert diagnostics["candidate_probability_samples"] == 2
    assert diagnostics["incumbent_probability_samples"] == 2
    assert diagnostics["paired_probability_samples"] == 2
    assert diagnostics["candidate_probability_calibration_mae"] == pytest.approx(0.28, abs=1e-9)
    assert diagnostics["incumbent_probability_calibration_mae"] == pytest.approx(0.28, abs=1e-9)
    assert diagnostics["candidate_incumbent_probability_drift_mae"] == pytest.approx(0.44, abs=1e-9)
    assert diagnostics["candidate_incumbent_directional_agreement"] == pytest.approx(0.0, abs=1e-9)

    decision = decide_promotion(summary, policy=policy, runtime=RuntimeBlockers())
    assert decision.promotion_eligible is True
    assert decision.threshold_tuning_ready is True


def test_replay_summary_requires_coverage_and_beats_benchmark(tmp_path: Path) -> None:
    db_path = tmp_path / "eval.db"
    start = datetime(2026, 5, 1, tzinfo=timezone.utc)
    now = start + timedelta(days=32)
    policy = EvaluationPolicy(
        threshold_tuning_hours=72,
        promotion_window_hours=743,
        min_resolved_decisions=30,
        min_actionable_decisions=30,
        min_symbols=2,
        min_calendar_days=30,
        min_trading_days=20,
        min_symbol_coverage_fraction=0.2,
        min_edge_bps=1,
        max_drawdown_worse_bps=50,
        round_trip_cost_bps=0,
        max_symbol_concentration=0.60,
    )

    for idx, (role, model_id, symbol, signal, price, ts) in enumerate(_burst_shadow_rows(start=start, days=32)):
        evaluated_at = ts + timedelta(minutes=idx % 15)
        assert record_shadow_decision(
            db_path,
            quarantine_version_id="candidate_b",
            model_version_id=model_id,
            baseline_version_id="active_a",
            decision_role=role,
            payload=_payload(symbol, signal, price, evaluated_at),
            horizon_hours=8,
            evaluated_at=evaluated_at,
        )

    resolve_due_shadow_decisions(
        db_path,
        prices={"BTCUSDT": 120.0, "ETHUSDT": 121.0},
        now=now,
    )

    summary = summarize_quarantine(
        db_path,
        quarantine_version_id="candidate_b",
        policy=policy,
        now=now,
    )
    decision = decide_promotion(summary, policy=policy, runtime=RuntimeBlockers())

    assert summary["coverage"]["calendar_days"] >= 30
    assert summary["coverage"]["trading_days"] >= 20
    assert summary["paired_metrics"]["candidate_net_return_bps"] > 0
    assert summary["paired_metrics"]["candidate_minus_benchmark_net_return_bps"] > 0
    assert summary["paired_metrics"]["bootstrap_ci"]["block_count"] >= 30
    assert decision.threshold_tuning_ready is True
    assert decision.promotion_eligible is False
    assert any(blocker.startswith("needs_") for blocker in decision.blockers)


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


def test_summarize_quarantine_uses_registered_threshold_floor(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "eval.db"
    registry_root = tmp_path / "registry"
    candidate_artifact = tmp_path / "candidate"
    incumbent_artifact = tmp_path / "incumbent"
    registry = ModelRegistry(registry_root)
    _register_threshold_model(
        registry,
        candidate_artifact,
        version_id="candidate_a",
        threshold=0.60,
        monkeypatch=monkeypatch,
    )
    _register_threshold_model(
        registry,
        incumbent_artifact,
        version_id="active_a",
        threshold=0.74,
        monkeypatch=monkeypatch,
    )

    now = datetime(2026, 6, 10, tzinfo=timezone.utc)
    for role, model_id in (
        ("candidate", "candidate_a"),
        ("incumbent", "active_a"),
    ):
        evaluated_at = now - timedelta(hours=168)
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
        policy=EvaluationPolicy(
            threshold_tuning_hours=72,
            promotion_window_hours=168,
            min_resolved_decisions=1,
            min_actionable_decisions=1,
            min_symbols=1,
            min_edge_bps=1,
            max_drawdown_worse_bps=50,
            round_trip_cost_bps=0,
            max_symbol_concentration=1.0,
        ),
        registry_root=registry_root,
        now=now,
    )

    assert summary["threshold_policy"]["candidate"]["threshold_floor"] == pytest.approx(0.60)
    assert summary["threshold_policy"]["incumbent"]["threshold_floor"] == pytest.approx(0.74)
    decision = decide_promotion(summary, policy=EvaluationPolicy(
        threshold_tuning_hours=72,
        promotion_window_hours=168,
        min_resolved_decisions=1,
        min_actionable_decisions=1,
        min_symbols=1,
        min_edge_bps=1,
        max_drawdown_worse_bps=50,
        round_trip_cost_bps=0,
        max_symbol_concentration=1.0,
    ), runtime=RuntimeBlockers())
    updated = record_registry_evaluation(
        registry,
        "candidate_a",
        summary=summary,
        decision=decision,
    )
    assert updated.metrics["paper_evaluation"]["threshold_policy"]["candidate"]["threshold_floor"] == pytest.approx(0.60)


def test_persistent_control_cannot_enable_disabled_deployment_policy(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("MODEL_EVAL_AUTO_PROMOTE", "0")
    registry_root = tmp_path / "registry"
    registry_root.mkdir()
    control_path = registry_root / "evaluator_control.json"
    control_path.write_text(json.dumps({"auto_promote": True}), encoding="utf-8")

    loaded = load_evaluator_control(registry_root)
    written = write_evaluator_control(
        registry_root,
        auto_promote=True,
        updated_by="test",
    )

    assert loaded["deployment_auto_promote_allowed"] is False
    assert loaded["auto_promote"] is False
    assert written["auto_promote"] is False
    assert json.loads(control_path.read_text(encoding="utf-8"))["auto_promote"] is False


def test_persistent_control_honors_enabled_deployment_policy(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("MODEL_EVAL_AUTO_PROMOTE", "1")
    registry_root = tmp_path / "registry"
    registry_root.mkdir()

    written = write_evaluator_control(
        registry_root,
        auto_promote=True,
        updated_by="test",
    )
    loaded = load_evaluator_control(registry_root)

    assert written["deployment_auto_promote_allowed"] is True
    assert written["auto_promote"] is True
    assert loaded["deployment_auto_promote_allowed"] is True
    assert loaded["auto_promote"] is True


def test_evaluator_argument_cannot_bypass_disabled_deployment_policy(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("MODEL_EVAL_AUTO_PROMOTE", "false")
    registry_root = tmp_path / "registry"
    artifact = tmp_path / "active"
    artifact.mkdir()
    registry = ModelRegistry(registry_root)
    registry.register_version("active", artifact, status="active")
    registry.set_active_version("active")

    result = asyncio.run(
        evaluate_once(
            model_root=tmp_path / "models",
            registry_root=registry_root,
            db_path=tmp_path / "eval.db",
            collect_shadow=False,
            auto_promote=True,
        )
    )

    assert result["status"] == "ok"
    assert result["auto_promote"] is False


def test_malformed_control_fails_closed_even_when_deployment_allows(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("MODEL_EVAL_AUTO_PROMOTE", "1")
    registry_root = tmp_path / "registry"
    registry_root.mkdir()
    (registry_root / "evaluator_control.json").write_text("not-json", encoding="utf-8")

    control = load_evaluator_control(registry_root)

    assert control["deployment_auto_promote_allowed"] is True
    assert control["auto_promote"] is False
