from __future__ import annotations

from pathlib import Path

from quant_v2.research.experiment_score import (
    build_report_from_experiment,
    build_report_from_path,
)


def test_build_report_from_experiment() -> None:
    experiment = {
        "config": {"validation_mode": "purged_kfold"},
        "results": {
            "1": {
                "overall": {
                    "spread_adjusted_ev": 5.0,
                    "win_rate": 0.58,
                    "n_trades": 150,
                },
                "robustness": {"deflated_sharpe_ratio": 0.62},
                "per_fold": [
                    {"spread_adjusted_ev": 4.0},
                    {"spread_adjusted_ev": 6.0},
                ],
            },
            "4": {
                "overall": {
                    "spread_adjusted_ev": 3.0,
                    "win_rate": 0.55,
                    "n_trades": 120,
                },
                "robustness": {"deflated_sharpe_ratio": 0.58},
                "per_fold": [
                    {"spread_adjusted_ev": 2.0},
                    {"spread_adjusted_ev": 4.0},
                ],
            },
        },
        "monte_carlo": {
            "1": {"ruin_probability": 0.18},
            "4": {"ruin_probability": 0.22},
        },
    }

    report = build_report_from_experiment(experiment)

    assert report.score > 60.0
    assert report.score_inputs.robustness > 50.0
    assert report.gate_inputs.ruin_probability <= 0.25


def test_build_report_from_path_on_existing_baseline() -> None:
    path = Path("experiments/run_20260222_231538_a355958f.json")
    report = build_report_from_path(path)

    assert report.score < 40.0
    assert report.gates.passed is False
    assert report.gates.checks["ruin_probability"] is False
