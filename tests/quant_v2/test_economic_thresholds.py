from __future__ import annotations

import numpy as np

from quant_v2.research.economic_thresholds import EconomicThresholdConfig, select_threshold_by_utility


def test_select_threshold_by_utility_prefers_higher_utility_even_with_lower_accuracy() -> None:
    probabilities = np.array([0.85, 0.75, 0.60, 0.55])
    labels = np.array([1, 1, 1, 0])
    forward = np.array([100.0, 10.0, -200.0, -200.0])

    report = select_threshold_by_utility(probabilities, labels, forward)

    assert report["source"] == "economic_utility"
    assert report["selected_threshold"] == 0.8
    assert report["accuracy_at_selected"] < report["accuracy_optimal_accuracy"]


def test_select_threshold_by_utility_marks_ineligible_thresholds() -> None:
    probabilities = np.array([0.90, 0.40, 0.35])
    labels = np.array([1, 0, 0])
    forward = np.array([5.0, -2.0, -1.0])

    report = select_threshold_by_utility(
        probabilities,
        labels,
        forward,
        config=EconomicThresholdConfig(min_actionable=2, threshold_min=0.5, threshold_max=0.9, threshold_step=0.4),
    )

    diag = {row["threshold"]: row for row in report["thresholds"]}
    assert diag[0.9]["eligible"] is False
    assert report["selected_actionable"] < 2


def test_select_threshold_by_utility_penalizes_symbol_concentration() -> None:
    probabilities = np.array([0.95, 0.90, 0.60, 0.55])
    labels = np.array([1, 1, 1, 1])
    forward = np.array([30.0, 30.0, -10.0, -10.0])
    symbols = ["A", "A", "B", "C"]

    low_penalty = select_threshold_by_utility(
        probabilities,
        labels,
        forward,
        symbols=symbols,
        config=EconomicThresholdConfig(
            min_actionable=1,
            threshold_min=0.5,
            threshold_max=0.9,
            threshold_step=0.4,
            concentration_penalty_weight=0.0,
        ),
    )
    high_penalty = select_threshold_by_utility(
        probabilities,
        labels,
        forward,
        symbols=symbols,
        config=EconomicThresholdConfig(
            min_actionable=1,
            threshold_min=0.5,
            threshold_max=0.9,
            threshold_step=0.4,
            concentration_penalty_weight=100.0,
        ),
    )

    assert low_penalty["selected_threshold"] == 0.9
    assert high_penalty["selected_threshold"] == 0.5


def test_select_threshold_by_utility_includes_fold_instability() -> None:
    probabilities = np.array([0.95, 0.95, 0.60, 0.60])
    labels = np.array([1, 1, 1, 1])
    forward = np.array([20.0, 20.0, -20.0, -20.0])
    fold_ids = ["f1", "f1", "f2", "f2"]

    report = select_threshold_by_utility(
        probabilities,
        labels,
        forward,
        fold_ids=fold_ids,
        config=EconomicThresholdConfig(
            min_actionable=1,
            threshold_min=0.5,
            threshold_max=0.9,
            threshold_step=0.4,
            instability_penalty_weight=50.0,
        ),
    )

    assert any(row["fold_expectancy_std"] > 0.0 for row in report["thresholds"])
    assert report["selected_score"] != 0.0
