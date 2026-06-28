from __future__ import annotations

from quant_v2.research.selection_risk import build_selection_risk_report


def test_selection_risk_report_flags_broad_search_instability() -> None:
    candidates = [
        {
            "candidate_id": "a",
            "score": 1.0,
            "holdout_accuracy": 0.55,
            "fold_ledger": {"mean_fold_accuracy": 0.55, "std_fold_accuracy": 0.01},
        },
        {
            "candidate_id": "b",
            "score": 2.0,
            "holdout_accuracy": 0.70,
            "fold_ledger": {"mean_fold_accuracy": 0.70, "std_fold_accuracy": 0.01},
        },
    ]

    report = build_selection_risk_report(candidates, selected_candidate_id="a", trial_count=64)

    assert report["policy_version"] == "selection_risk_v1"
    assert report["selected_rank_by_holdout"] == 2
    assert "selected_holdout_rank_weak_after_broad_search" in report["blockers"]
    assert report["overfit_risk"] == "high"


def test_selection_risk_report_flags_missing_fold_ledger() -> None:
    candidates = [
        {
            "candidate_id": "a",
            "score": 1.0,
            "holdout_accuracy": 0.55,
            "fold_ledger": {},
        }
    ]

    report = build_selection_risk_report(candidates, selected_candidate_id="a", trial_count=1)

    assert "missing_fold_ledger" in report["blockers"]
