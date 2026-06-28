"""Utility-based threshold selection for recovery experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np


@dataclass(frozen=True)
class EconomicThresholdConfig:
    threshold_min: float = 0.50
    threshold_max: float = 0.80
    threshold_step: float = 0.05
    min_actionable: int = 20
    round_trip_cost_bps: float = 8.0
    drawdown_penalty_weight: float = 1.0
    turnover_penalty_bps: float = 0.0
    concentration_penalty_weight: float = 25.0
    instability_penalty_weight: float = 10.0


def _safe_mean(values: np.ndarray | Sequence[float]) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return 0.0
    return float(np.nanmean(arr))


def _safe_std(values: np.ndarray | Sequence[float]) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size <= 1:
        return 0.0
    return float(np.nanstd(arr, ddof=0))


def _max_drawdown_bps(returns_bps: np.ndarray) -> float:
    if returns_bps.size == 0:
        return 0.0
    equity = np.cumsum(np.asarray(returns_bps, dtype=float))
    peaks = np.maximum.accumulate(equity)
    drawdowns = equity - peaks
    return float(np.min(drawdowns))


def select_threshold_by_utility(
    probabilities: np.ndarray,
    labels: np.ndarray,
    forward_return_bps: np.ndarray,
    *,
    symbols: Sequence[str] | None = None,
    fold_ids: Sequence[str] | None = None,
    config: EconomicThresholdConfig | None = None,
) -> dict[str, Any]:
    config = config or EconomicThresholdConfig()
    probs = np.asarray(probabilities, dtype=float).reshape(-1)
    truth = np.asarray(labels, dtype=float).reshape(-1)
    forward = np.asarray(forward_return_bps, dtype=float).reshape(-1)
    if not (len(probs) == len(truth) == len(forward)):
        raise ValueError("probabilities, labels, and forward_return_bps must have the same length")

    if probs.size == 0:
        return {
            "source": "economic_utility",
            "selected_threshold": 0.5,
            "selected_score": 0.0,
            "selected_expectancy_bps": 0.0,
            "selected_actionable": 0,
            "accuracy_at_selected": 0.0,
            "accuracy_optimal_threshold": 0.5,
            "accuracy_optimal_accuracy": 0.0,
            "thresholds": [],
            "config": asdict(config),
        }

    thresholds = np.round(
        np.arange(config.threshold_min, config.threshold_max + config.threshold_step / 2.0, config.threshold_step),
        4,
    )
    min_actionable = max(1, int(config.min_actionable))
    accuracy_by_threshold: list[tuple[float, float]] = []
    diagnostics: list[dict[str, Any]] = []

    truth_binary = np.where(np.isnan(truth), 0.0, truth)
    accuracy_optimal_threshold = float(thresholds[0]) if len(thresholds) else 0.5
    accuracy_optimal_accuracy = -1.0

    for threshold in thresholds:
        action_mask = probs >= float(threshold)
        actionable = int(action_mask.sum())
        acted_labels = truth_binary[action_mask]
        acted_forward = forward[action_mask]

        if actionable:
            signed = np.where(
                acted_labels >= 0.5,
                acted_forward - float(config.round_trip_cost_bps),
                -np.abs(acted_forward) - float(config.round_trip_cost_bps),
            )
            mean_net = _safe_mean(signed)
            drawdown = _max_drawdown_bps(signed)
            action_rate = actionable / max(len(probs), 1)
            if symbols is not None:
                acted_symbols = np.asarray(list(symbols), dtype=str)[action_mask]
                counts = np.unique(acted_symbols, return_counts=True)[1]
                concentration_share = float(np.max(counts) / max(np.sum(counts), 1)) if counts.size else 0.0
            else:
                concentration_share = 0.0
            if fold_ids is not None:
                acted_folds = np.asarray(list(fold_ids), dtype=str)[action_mask]
                fold_means = [
                    float(np.mean(signed[acted_folds == fold])) for fold in sorted(set(acted_folds))
                ]
                fold_std = _safe_std(fold_means)
            else:
                fold_std = 0.0
        else:
            signed = np.asarray([], dtype=float)
            mean_net = 0.0
            drawdown = 0.0
            action_rate = 0.0
            concentration_share = 0.0
            fold_std = 0.0

        eligible = actionable >= min_actionable
        score = float(
            mean_net
            - abs(drawdown) * float(config.drawdown_penalty_weight) / 100.0
            - float(config.turnover_penalty_bps) * action_rate
            - max(0.0, concentration_share - 0.50) * float(config.concentration_penalty_weight)
            - fold_std * float(config.instability_penalty_weight) / 100.0
        )
        accuracy = float(np.mean((probs >= float(threshold)).astype(float) == truth_binary)) if truth_binary.size else 0.0
        accuracy_by_threshold.append((float(threshold), accuracy))
        if accuracy > accuracy_optimal_accuracy or (
            accuracy == accuracy_optimal_accuracy and float(threshold) < accuracy_optimal_threshold
        ):
            accuracy_optimal_accuracy = accuracy
            accuracy_optimal_threshold = float(threshold)

        diagnostics.append(
            {
                "threshold": float(threshold),
                "eligible": bool(eligible),
                "selected_score": score,
                "selected_expectancy_bps": float(mean_net),
                "selected_actionable": actionable,
                "accuracy": accuracy,
                "max_drawdown_bps": float(drawdown),
                "action_rate": float(action_rate),
                "concentration_share": float(concentration_share),
                "fold_expectancy_std": float(fold_std),
            }
        )

    eligible_rows = [row for row in diagnostics if row["eligible"]]
    candidate_rows = eligible_rows if eligible_rows else diagnostics
    selected = max(
        candidate_rows,
        key=lambda row: (
            float(row["selected_score"]),
            float(row["selected_expectancy_bps"]),
            -abs(float(row["max_drawdown_bps"])),
            -float(row["threshold"]),
        ),
    )

    selected_threshold = float(selected["threshold"])
    selected_mask = probs >= selected_threshold
    selected_accuracy = float(np.mean(selected_mask.astype(float) == truth_binary)) if truth_binary.size else 0.0
    selected_actionable = int(selected_mask.sum())
    selected_expectancy = float(
        _safe_mean(
            np.where(
                truth_binary[selected_mask] >= 0.5,
                forward[selected_mask] - float(config.round_trip_cost_bps),
                -np.abs(forward[selected_mask]) - float(config.round_trip_cost_bps),
            )
        )
        if selected_actionable
        else 0.0
    )

    return {
        "source": "economic_utility",
        "selected_threshold": selected_threshold,
        "selected_score": float(selected["selected_score"]),
        "selected_expectancy_bps": selected_expectancy,
        "selected_actionable": selected_actionable,
        "accuracy_at_selected": selected_accuracy,
        "accuracy_optimal_threshold": accuracy_optimal_threshold,
        "accuracy_optimal_accuracy": float(accuracy_optimal_accuracy if accuracy_optimal_accuracy >= 0.0 else 0.0),
        "thresholds": diagnostics,
        "config": asdict(config),
    }
