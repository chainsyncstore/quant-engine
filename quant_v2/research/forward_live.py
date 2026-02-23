"""Forward-live shadow simulation summaries for v2 validation runs."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np


def build_forward_live_simulation(horizon_results: Mapping[int, Any]) -> dict[str, Any]:
    """Build per-horizon and aggregate forward-live simulation summaries."""

    by_horizon: dict[str, dict[str, Any]] = {}
    ev_all: list[float] = []

    for horizon, result in sorted(horizon_results.items(), key=lambda kv: int(kv[0])):
        fold_metrics = [fold.metrics for fold in result.folds]
        ev = [float(m.spread_adjusted_ev) for m in fold_metrics]
        win_rate = [float(m.win_rate) for m in fold_metrics]
        sharpe = [float(m.sharpe) for m in fold_metrics]
        max_dd = [float(m.max_drawdown) for m in fold_metrics]

        if ev:
            ev_mean = float(np.mean(ev))
            ev_std = float(np.std(ev))
            win_rate_mean = float(np.mean(win_rate))
            sharpe_mean = float(np.mean(sharpe))
            worst_drawdown = float(np.min(max_dd))
        else:
            ev_mean = 0.0
            ev_std = 0.0
            win_rate_mean = 0.0
            sharpe_mean = 0.0
            worst_drawdown = 0.0

        ev_all.extend(ev)
        by_horizon[str(horizon)] = {
            "fold_count": int(len(fold_metrics)),
            "ev_mean": ev_mean,
            "ev_std": ev_std,
            "win_rate_mean": win_rate_mean,
            "sharpe_mean": sharpe_mean,
            "worst_drawdown": worst_drawdown,
            "deflated_sharpe_ratio": float(result.robustness.get("deflated_sharpe_ratio", 0.0)),
            "probabilistic_sharpe_ratio": float(result.robustness.get("probabilistic_sharpe_ratio", 0.0)),
        }

    aggregate = {
        "n_horizons": int(len(by_horizon)),
        "ev_mean": float(np.mean(ev_all)) if ev_all else 0.0,
        "ev_std": float(np.std(ev_all)) if ev_all else 0.0,
        "stability_score": float(1.0 / (1.0 + float(np.std(ev_all)))) if ev_all else 0.0,
    }

    return {
        "mode": "shadow_forward_live_simulation",
        "by_horizon": by_horizon,
        "aggregate": aggregate,
    }
