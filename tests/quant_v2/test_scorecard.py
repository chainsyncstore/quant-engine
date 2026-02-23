from __future__ import annotations

import pytest

from quant_v2.research.scorecard import (
    GateInputs,
    ScoreInputs,
    compute_system_score,
    evaluate_claim_70_plus_gates,
)


def test_compute_system_score_weighting() -> None:
    score = compute_system_score(
        ScoreInputs(
            robustness=80.0,
            tradability=75.0,
            risk=70.0,
            generalization=72.0,
            live_readiness=68.0,
        )
    )

    assert score == 74.35


def test_evaluate_claim_70_plus_gates_pass() -> None:
    result = evaluate_claim_70_plus_gates(
        GateInputs(
            dsr_majority=0.61,
            positive_ev_fold_ratio=0.70,
            ruin_probability=0.18,
            single_symbol_ev_dependency=0.30,
            shadow_live_drift_ok=True,
        )
    )

    assert result.passed is True
    assert all(result.checks.values())


def test_evaluate_claim_70_plus_gates_fail() -> None:
    result = evaluate_claim_70_plus_gates(
        GateInputs(
            dsr_majority=0.50,
            positive_ev_fold_ratio=0.70,
            ruin_probability=0.18,
            single_symbol_ev_dependency=0.30,
            shadow_live_drift_ok=True,
        )
    )

    assert result.passed is False
    assert result.checks["dsr_majority"] is False


def test_score_inputs_validate_bounds() -> None:
    with pytest.raises(ValueError):
        ScoreInputs(
            robustness=101.0,
            tradability=50.0,
            risk=50.0,
            generalization=50.0,
            live_readiness=50.0,
        )


def test_gate_inputs_validate_bounds() -> None:
    with pytest.raises(ValueError):
        GateInputs(
            dsr_majority=1.2,
            positive_ev_fold_ratio=0.7,
            ruin_probability=0.2,
            single_symbol_ev_dependency=0.3,
            shadow_live_drift_ok=True,
        )
