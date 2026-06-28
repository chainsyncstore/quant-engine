"""Active-model decay and maintenance lifecycle helpers."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Mapping, Sequence


def _canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def _sha256(payload: Any) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        if value is None:
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _safe_int(value: object, default: int = 0) -> int:
    try:
        if value is None:
            return int(default)
        return int(value)
    except Exception:
        return int(default)


def _read(payload: Mapping[str, Any] | Any, *path: str, default: Any = None) -> Any:
    current: Any = payload
    for key in path:
        if isinstance(current, Mapping) and key in current:
            current = current[key]
            continue
        return default
    return current


@dataclass(frozen=True)
class ModelMaintenanceDecision:
    active_version_id: str
    decayed: bool
    no_trade_required: bool
    proven_shadow_version_id: str | None
    recommended_action: str
    blockers: tuple[str, ...]
    evidence_digest: str
    metrics: dict[str, Any]


def _select_proven_shadow(
    candidate_records: Sequence[Mapping[str, Any] | Any],
    *,
    approval_required: bool,
    hard_risk_pauses: int,
) -> tuple[str | None, list[str]]:
    blockers: list[str] = []
    if hard_risk_pauses > 0:
        blockers.append(f"hard_risk_pauses={int(hard_risk_pauses)}")

    proven: list[tuple[str, float]] = []
    for record in candidate_records:
        version_id = str(_read(record, "version_id", default="") or _read(record, "candidate_id", default="")).strip()
        if not version_id:
            continue
        status = str(_read(record, "status", default="")).strip().lower()
        if status in {"rejected", "expired", "inactive"}:
            continue
        quality = _read(record, "candidate_quality_report", default={}) or {}
        if not isinstance(quality, Mapping) or not quality:
            continue
        quality_decision = str(quality.get("overall_decision") or quality.get("decision") or "").strip().lower()
        if quality_decision != "pass":
            continue
        metrics = _read(record, "metrics", default={}) or {}
        if not isinstance(metrics, Mapping):
            metrics = {}
        paper_eval = _read(metrics, "paper_evaluation", default={}) or {}
        if approval_required:
            if not isinstance(paper_eval, Mapping) or paper_eval.get("promotion_eligible") is not True:
                continue
        benchmark = _read(record, "benchmark_delta_report", "overall", default={}) or {}
        if not isinstance(benchmark, Mapping):
            benchmark = {}
        candidate_minus_best = _safe_float(benchmark.get("candidate_minus_best_nonflat_pnl_usd"), float("-inf"))
        candidate_minus_same = _safe_float(benchmark.get("candidate_minus_same_side_pnl_usd"), float("-inf"))
        if candidate_minus_best <= 0.0 or candidate_minus_same <= 0.0:
            continue
        if quality:
            score = _safe_float(_read(quality, "summary", "hard_failure_count", default=0), 0.0)
        else:
            score = 0.0
        proven.append((version_id, -score + candidate_minus_best + candidate_minus_same))

    if not proven:
        return None, blockers

    proven.sort(key=lambda item: (item[1], item[0]), reverse=True)
    return proven[0][0], blockers


def evaluate_active_model_maintenance(
    active_record: Mapping[str, Any] | Any,
    recent_evidence: Mapping[str, Any],
    candidate_records: Sequence[Mapping[str, Any] | Any],
    *,
    hard_risk_pauses: int = 0,
    approval_required: bool = True,
) -> ModelMaintenanceDecision:
    active_version_id = str(_read(active_record, "version_id", default="") or _read(active_record, "candidate_id", default="")).strip()
    active_metrics = _read(active_record, "metrics", default={}) or {}
    if not isinstance(active_metrics, Mapping):
        active_metrics = {}

    recent_net_bps = _safe_float(_read(recent_evidence, "net_return_bps", default=0.0), 0.0)
    recent_drawdown_bps = _safe_float(_read(recent_evidence, "drawdown_bps", default=0.0), 0.0)
    recent_benchmark_delta_bps = _safe_float(_read(recent_evidence, "candidate_minus_best_nonflat_bps", default=0.0), 0.0)
    recent_same_side_benchmark_delta_bps = _safe_float(
        _read(
            recent_evidence,
            "recent_same_side_benchmark_delta_bps",
            default=_read(recent_evidence, "candidate_minus_same_side_bps", default=0.0),
        ),
        0.0,
    )
    shadow_drift_mae = _safe_float(_read(recent_evidence, "shadow_drift_mae", default=0.0), 0.0)
    recent_actionable = _safe_int(_read(recent_evidence, "recent_actionable_decisions", default=0), 0)
    production_risk_elevated = bool(_read(recent_evidence, "production_risk_elevated", default=False))

    blockers: list[str] = []
    decayed = False
    if recent_net_bps <= _safe_float(_read(recent_evidence, "min_recent_net_bps", default=0.0), 0.0):
        decayed = True
        blockers.append("recent_net_return_below_floor")
    if recent_drawdown_bps >= _safe_float(_read(recent_evidence, "max_drawdown_bps", default=100.0), 100.0):
        decayed = True
        blockers.append("drawdown_exceeds_ceiling")
    if recent_benchmark_delta_bps <= _safe_float(_read(recent_evidence, "min_benchmark_delta_bps", default=0.0), 0.0):
        decayed = True
        blockers.append("benchmark_delta_below_floor")
    if recent_same_side_benchmark_delta_bps <= _safe_float(_read(recent_evidence, "min_same_side_benchmark_delta_bps", default=0.0), 0.0):
        decayed = True
        blockers.append("same_side_benchmark_delta_below_floor")
    if shadow_drift_mae >= _safe_float(_read(recent_evidence, "max_shadow_drift_mae", default=0.10), 0.10):
        decayed = True
        blockers.append("shadow_drift_out_of_tolerance")
    if recent_actionable < _safe_int(_read(recent_evidence, "min_recent_actionable", default=30), 30):
        if production_risk_elevated:
            decayed = True
            blockers.append("insufficient_recent_sample_with_elevated_risk")
        else:
            blockers.append("insufficient_recent_actionable")

    proven_shadow_id, shadow_blockers = _select_proven_shadow(
        candidate_records,
        approval_required=approval_required,
        hard_risk_pauses=hard_risk_pauses,
    )
    blockers.extend(shadow_blockers)

    no_trade_required = bool(decayed or hard_risk_pauses > 0)
    if hard_risk_pauses > 0:
        recommended_action = "remain_no_trade_and_trigger_recovery_retrain"
    elif decayed and proven_shadow_id:
        recommended_action = "promote_proven_shadow_via_approval_path"
    elif decayed:
        recommended_action = "remain_no_trade_and_trigger_recovery_retrain"
    else:
        recommended_action = "continue"

    metrics = {
        "recent_net_return_bps": recent_net_bps,
        "recent_drawdown_bps": recent_drawdown_bps,
        "recent_benchmark_delta_bps": recent_benchmark_delta_bps,
        "recent_same_side_benchmark_delta_bps": recent_same_side_benchmark_delta_bps,
        "shadow_drift_mae": shadow_drift_mae,
        "recent_actionable_decisions": recent_actionable,
        "production_risk_elevated": production_risk_elevated,
        "candidate_count": len(candidate_records),
        "hard_risk_pauses": int(hard_risk_pauses),
        "approval_required": bool(approval_required),
    }
    evidence_digest = _sha256(
        {
            "active_version_id": active_version_id,
            "metrics": metrics,
            "blockers": blockers,
            "proven_shadow_version_id": proven_shadow_id,
            "recommended_action": recommended_action,
        }
    )
    return ModelMaintenanceDecision(
        active_version_id=active_version_id,
        decayed=decayed,
        no_trade_required=no_trade_required,
        proven_shadow_version_id=proven_shadow_id,
        recommended_action=recommended_action,
        blockers=tuple(blockers),
        evidence_digest=evidence_digest,
        metrics=metrics,
    )
