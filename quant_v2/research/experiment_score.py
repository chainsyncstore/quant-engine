"""Compute v2 scorecard and 70+ gate diagnostics from experiment JSON outputs."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Any

from quant_v2.research.scorecard import (
    GateInputs,
    GateResult,
    ScoreInputs,
    compute_system_score,
    evaluate_claim_70_plus_gates,
)


@dataclass(frozen=True)
class ExperimentScoreReport:
    """Computed scorecard from a research experiment file."""

    score_inputs: ScoreInputs
    gate_inputs: GateInputs
    score: float
    gates: GateResult


def _safe_mean(values: list[float], default: float = 0.0) -> float:
    return float(fmean(values)) if values else float(default)


def _clip_0_100(value: float) -> float:
    return max(0.0, min(100.0, float(value)))


def _logistic(x: float, scale: float = 1.0) -> float:
    z = x / max(scale, 1e-9)
    return 1.0 / (1.0 + math.exp(-z))


def build_report_from_experiment(experiment: dict[str, Any]) -> ExperimentScoreReport:
    """Build score and gate diagnostics from experiment JSON dict."""

    results = experiment.get("results", {}) or {}
    horizons = [results[key] for key in sorted(results.keys(), key=lambda k: int(k))]

    dsr_values: list[float] = []
    overall_ev_values: list[float] = []
    overall_win_rate_values: list[float] = []
    fold_positive_flags: list[bool] = []
    fold_ev_std_values: list[float] = []
    horizon_trade_counts: list[int] = []

    for horizon_data in horizons:
        robustness = horizon_data.get("robustness", {}) or {}
        overall = horizon_data.get("overall", {}) or {}
        per_fold = horizon_data.get("per_fold", []) or []

        dsr = float(robustness.get("deflated_sharpe_ratio", 0.0))
        dsr_values.append(max(0.0, dsr))

        overall_ev_values.append(float(overall.get("spread_adjusted_ev", 0.0)))
        overall_win_rate_values.append(float(overall.get("win_rate", 0.0)))
        horizon_trade_counts.append(int(overall.get("n_trades", 0)))

        fold_evs = [float(f.get("spread_adjusted_ev", 0.0)) for f in per_fold]
        fold_positive_flags.extend(ev > 0.0 for ev in fold_evs)
        if len(fold_evs) > 1:
            mean_ev = _safe_mean(fold_evs)
            variance = _safe_mean([(ev - mean_ev) ** 2 for ev in fold_evs])
            fold_ev_std_values.append(math.sqrt(max(variance, 0.0)))

    monte_carlo = experiment.get("monte_carlo", {}) or {}
    ruin_probs = [
        float((mc or {}).get("ruin_probability", 0.0))
        for mc in monte_carlo.values()
        if isinstance(mc, dict)
    ]

    # --- Score components (0..100) ---
    robustness = _clip_0_100(_safe_mean(dsr_values) * 100.0)

    positive_ev_ratio = _safe_mean([1.0 if flag else 0.0 for flag in fold_positive_flags], default=0.0)
    ev_signal = _safe_mean([_logistic(ev, scale=15.0) for ev in overall_ev_values], default=0.5)
    win_rate_signal = _safe_mean(overall_win_rate_values, default=0.5)
    tradability = _clip_0_100((0.50 * positive_ev_ratio + 0.25 * ev_signal + 0.25 * win_rate_signal) * 100.0)

    survival_signal = 1.0 - _safe_mean(ruin_probs, default=0.5)
    risk = _clip_0_100(survival_signal * 100.0)

    stability_signal = 1.0 - _safe_mean([min(std / 25.0, 1.0) for std in fold_ev_std_values], default=0.5)
    generalization = _clip_0_100((0.6 * positive_ev_ratio + 0.4 * stability_signal) * 100.0)

    validation_mode = str((experiment.get("config", {}) or {}).get("validation_mode", "")).strip().lower()
    strict_validation_bonus = 0.10 if validation_mode == "purged_kfold" else 0.0
    trade_coverage_signal = _safe_mean([1.0 if n > 0 else 0.0 for n in horizon_trade_counts], default=0.0)
    live_readiness = _clip_0_100((0.55 * robustness / 100.0 + 0.35 * trade_coverage_signal + strict_validation_bonus) * 100.0)

    score_inputs = ScoreInputs(
        robustness=robustness,
        tradability=tradability,
        risk=risk,
        generalization=generalization,
        live_readiness=live_readiness,
    )

    # --- Gates ---
    dsr_majority = _safe_mean([1.0 if dsr >= 0.55 else 0.0 for dsr in dsr_values], default=0.0)
    positive_ev_fold_ratio = positive_ev_ratio
    ruin_probability = _safe_mean(ruin_probs, default=1.0)

    ev_total = sum(overall_ev_values)
    dominant_share = 1.0
    if abs(ev_total) > 1e-9 and overall_ev_values:
        dominant_share = max(abs(ev) for ev in overall_ev_values) / max(sum(abs(ev) for ev in overall_ev_values), 1e-9)

    # Proxy: high consistency + low ruin + strict validation indicates acceptable shadow/live alignment.
    shadow_live_drift_ok = (
        positive_ev_fold_ratio >= 0.65
        and ruin_probability <= 0.25
        and validation_mode == "purged_kfold"
    )

    gate_inputs = GateInputs(
        dsr_majority=dsr_majority,
        positive_ev_fold_ratio=positive_ev_fold_ratio,
        ruin_probability=ruin_probability,
        single_symbol_ev_dependency=min(max(dominant_share, 0.0), 1.0),
        shadow_live_drift_ok=shadow_live_drift_ok,
    )

    score = compute_system_score(score_inputs)
    gates = evaluate_claim_70_plus_gates(gate_inputs)

    return ExperimentScoreReport(
        score_inputs=score_inputs,
        gate_inputs=gate_inputs,
        score=score,
        gates=gates,
    )


def build_report_from_path(experiment_path: Path | str) -> ExperimentScoreReport:
    """Load experiment JSON and compute score report."""

    path = Path(experiment_path).expanduser()
    data = json.loads(path.read_text(encoding="utf-8"))
    return build_report_from_experiment(data)


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Compute v2 scorecard from experiment JSON")
    parser.add_argument("--experiment", required=True, help="Path to experiment .json file")
    args = parser.parse_args()

    report = build_report_from_path(args.experiment)

    print(f"score={report.score:.2f}")
    print(f"gates_passed={report.gates.passed}")
    print("score_inputs", report.score_inputs)
    print("gate_inputs", report.gate_inputs)
    print("gate_checks", report.gates.checks)


if __name__ == "__main__":
    _cli()
