from __future__ import annotations

from quant_v2.research.model_maintenance import evaluate_active_model_maintenance


def _active_record() -> dict[str, object]:
    return {
        "version_id": "active_v1",
        "metrics": {"status": "active"},
    }


def _candidate(version_id: str, *, quality: str = "pass", paper_eligible: bool = True) -> dict[str, object]:
    return {
        "version_id": version_id,
        "status": "paper_quarantine" if paper_eligible else "candidate",
        "metrics": {
            "paper_evaluation": {"promotion_eligible": paper_eligible},
        },
        "candidate_quality_report": {
            "overall_decision": quality,
            "summary": {"hard_failure_count": 0},
        },
        "benchmark_delta_report": {
            "overall": {
                "candidate_minus_best_nonflat_pnl_usd": 12.0,
                "candidate_minus_same_side_pnl_usd": 11.0,
            }
        },
    }


def _candidate_without_quality(version_id: str) -> dict[str, object]:
    candidate = _candidate(version_id)
    del candidate["candidate_quality_report"]
    return candidate


def test_decayed_active_model_prefers_governed_shadow_promotion() -> None:
    decision = evaluate_active_model_maintenance(
        _active_record(),
        {
            "net_return_bps": -6.0,
            "drawdown_bps": 45.0,
            "candidate_minus_best_nonflat_bps": -3.0,
            "shadow_drift_mae": 0.02,
            "recent_actionable_decisions": 50,
        },
        [_candidate("shadow_v1")],
        hard_risk_pauses=0,
    )

    assert decision.decayed is True
    assert decision.no_trade_required is True
    assert decision.proven_shadow_version_id == "shadow_v1"
    assert decision.recommended_action == "promote_proven_shadow_via_approval_path"


def test_shadow_candidate_requires_passing_quality_report() -> None:
    decision = evaluate_active_model_maintenance(
        _active_record(),
        {
            "net_return_bps": -6.0,
            "drawdown_bps": 45.0,
            "candidate_minus_best_nonflat_bps": -3.0,
            "shadow_drift_mae": 0.02,
            "recent_actionable_decisions": 50,
        },
        [_candidate_without_quality("shadow_v1")],
        hard_risk_pauses=0,
    )

    assert decision.decayed is True
    assert decision.proven_shadow_version_id is None
    assert decision.recommended_action == "remain_no_trade_and_trigger_recovery_retrain"


def test_decayed_active_model_without_shadow_stays_no_trade() -> None:
    decision = evaluate_active_model_maintenance(
        _active_record(),
        {
            "net_return_bps": -6.0,
            "drawdown_bps": 45.0,
            "candidate_minus_best_nonflat_bps": -3.0,
            "shadow_drift_mae": 0.02,
            "recent_actionable_decisions": 50,
        },
        [_candidate("shadow_v1", quality="watch", paper_eligible=False)],
        hard_risk_pauses=0,
    )

    assert decision.decayed is True
    assert decision.proven_shadow_version_id is None
    assert decision.recommended_action == "remain_no_trade_and_trigger_recovery_retrain"


def test_hard_risk_pause_blocks_promotion_recommendation() -> None:
    decision = evaluate_active_model_maintenance(
        _active_record(),
        {
            "net_return_bps": 10.0,
            "drawdown_bps": 1.0,
            "candidate_minus_best_nonflat_bps": 5.0,
            "shadow_drift_mae": 0.01,
            "recent_actionable_decisions": 100,
        },
        [_candidate("shadow_v1")],
        hard_risk_pauses=2,
    )

    assert decision.no_trade_required is True
    assert decision.recommended_action == "remain_no_trade_and_trigger_recovery_retrain"
    assert any("hard_risk_pauses=2" in blocker for blocker in decision.blockers)


def test_same_side_benchmark_decay_blocks_continue() -> None:
    decision = evaluate_active_model_maintenance(
        _active_record(),
        {
            "net_return_bps": 10.0,
            "drawdown_bps": 1.0,
            "candidate_minus_best_nonflat_bps": 5.0,
            "candidate_minus_same_side_bps": -0.5,
            "shadow_drift_mae": 0.01,
            "recent_actionable_decisions": 100,
        },
        [_candidate("shadow_v1")],
        hard_risk_pauses=0,
    )

    assert decision.decayed is True
    assert any("same_side_benchmark_delta_below_floor" in blocker for blocker in decision.blockers)


def test_insufficient_sample_only_decays_when_risk_is_elevated() -> None:
    low_risk = evaluate_active_model_maintenance(
        _active_record(),
        {
            "net_return_bps": 10.0,
            "drawdown_bps": 1.0,
            "candidate_minus_best_nonflat_bps": 5.0,
            "candidate_minus_same_side_bps": 5.0,
            "shadow_drift_mae": 0.01,
            "recent_actionable_decisions": 5,
            "production_risk_elevated": False,
        },
        [_candidate("shadow_v1")],
        hard_risk_pauses=0,
    )
    high_risk = evaluate_active_model_maintenance(
        _active_record(),
        {
            "net_return_bps": 10.0,
            "drawdown_bps": 1.0,
            "candidate_minus_best_nonflat_bps": 5.0,
            "candidate_minus_same_side_bps": 5.0,
            "shadow_drift_mae": 0.01,
            "recent_actionable_decisions": 5,
            "production_risk_elevated": True,
        },
        [_candidate("shadow_v1")],
        hard_risk_pauses=0,
    )

    assert low_risk.decayed is False
    assert "insufficient_recent_actionable" in low_risk.blockers
    assert high_risk.decayed is True
    assert any("insufficient_recent_sample_with_elevated_risk" in blocker for blocker in high_risk.blockers)
