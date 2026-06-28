"""Selection-risk diagnostics for bounded recovery sweeps."""

from __future__ import annotations

from statistics import fmean
from typing import Any, Mapping, Sequence


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        if value is None:
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def build_selection_risk_report(
    candidates: Sequence[Mapping[str, Any]],
    *,
    selected_candidate_id: str | None,
    trial_count: int,
) -> dict[str, Any]:
    candidate_rows = [dict(candidate) for candidate in candidates]
    candidate_count = int(len(candidate_rows))
    if candidate_count == 0:
        return {
            "policy_version": "selection_risk_v1",
            "trial_count": int(trial_count),
            "candidate_count": 0,
            "selected_candidate_id": selected_candidate_id,
            "selected_rank_by_score": None,
            "selected_rank_by_holdout": None,
            "pbo_proxy": 0.0,
            "fold_instability": 0.0,
            "overfit_risk": "low",
            "blockers": ["missing_candidates"],
        }

    scored = sorted(
        candidate_rows,
        key=lambda row: (
            _safe_float(row.get("score"), float("-inf")),
            _safe_float(row.get("holdout_accuracy"), _safe_float(row.get("mean_accuracy"), 0.0)),
            str(row.get("candidate_id", "")),
        ),
        reverse=True,
    )

    by_holdout = sorted(
        candidate_rows,
        key=lambda row: (
            _safe_float(row.get("holdout_accuracy"), _safe_float(row.get("mean_accuracy"), 0.0)),
            _safe_float(row.get("score"), float("-inf")),
            str(row.get("candidate_id", "")),
        ),
        reverse=True,
    )

    selected_score_rank = None
    selected_holdout_rank = None
    selected_row = None
    if selected_candidate_id is not None:
        for idx, row in enumerate(scored, start=1):
            if str(row.get("candidate_id")) == str(selected_candidate_id):
                selected_score_rank = idx
                selected_row = row
                break
        for idx, row in enumerate(by_holdout, start=1):
            if str(row.get("candidate_id")) == str(selected_candidate_id):
                selected_holdout_rank = idx
                break

    fold_std_values = []
    for row in candidate_rows:
        ledger = row.get("fold_ledger") or {}
        if not isinstance(ledger, Mapping):
            continue
        if ledger.get("mean_fold_accuracy") is not None:
            fold_std_values.append(_safe_float(ledger.get("std_fold_accuracy"), 0.0))
    fold_instability = float(fmean(fold_std_values)) if fold_std_values else 0.0

    pbo_proxy = 0.0
    blockers: list[str] = []
    if selected_holdout_rank is not None:
        pbo_proxy = max(0.0, 1.0 - (selected_holdout_rank / max(candidate_count, 1)))
    if trial_count > 50 and selected_holdout_rank is not None and selected_holdout_rank > max(1, int(candidate_count * 0.25)):
        blockers.append("selected_holdout_rank_weak_after_broad_search")
    if fold_instability > 0.20:
        blockers.append("fold_instability")
    if selected_row is None and selected_candidate_id is not None:
        blockers.append("missing_selected_candidate")
    if selected_row is not None and not selected_row.get("fold_ledger"):
        blockers.append("missing_fold_ledger")

    if blockers:
        overfit_risk = "high"
    elif pbo_proxy >= 0.50:
        overfit_risk = "medium"
    else:
        overfit_risk = "low"

    return {
        "policy_version": "selection_risk_v1",
        "trial_count": int(trial_count),
        "candidate_count": candidate_count,
        "selected_candidate_id": selected_candidate_id,
        "selected_rank_by_score": selected_score_rank,
        "selected_rank_by_holdout": selected_holdout_rank,
        "pbo_proxy": float(pbo_proxy),
        "fold_instability": float(fold_instability),
        "overfit_risk": overfit_risk,
        "blockers": blockers,
    }
