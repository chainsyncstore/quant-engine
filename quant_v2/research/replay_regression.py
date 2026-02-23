"""Replay regression helpers comparing baseline (v1) and current v2 runs."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np


def build_replay_regression_report(
    *,
    current_forward_live: Mapping[str, Any],
    baseline_forward_live: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Build replay deltas between baseline and current forward-live summaries."""

    current_by_h = dict(current_forward_live.get("by_horizon", {}))

    if baseline_forward_live is None:
        baseline_by_h = current_by_h
        baseline_source = "self"
    else:
        baseline_by_h = dict(baseline_forward_live.get("by_horizon", {}))
        baseline_source = "external"

    horizon_deltas: dict[str, dict[str, float]] = {}
    ev_deltas: list[float] = []

    for horizon, current_stats in current_by_h.items():
        base_stats = baseline_by_h.get(horizon)
        if not isinstance(base_stats, dict):
            continue

        ev_delta = float(current_stats.get("ev_mean", 0.0)) - float(base_stats.get("ev_mean", 0.0))
        wr_delta = float(current_stats.get("win_rate_mean", 0.0)) - float(base_stats.get("win_rate_mean", 0.0))
        sharpe_delta = float(current_stats.get("sharpe_mean", 0.0)) - float(base_stats.get("sharpe_mean", 0.0))
        ev_deltas.append(ev_delta)

        horizon_deltas[horizon] = {
            "ev_mean_delta": ev_delta,
            "win_rate_mean_delta": wr_delta,
            "sharpe_mean_delta": sharpe_delta,
        }

    aggregate = {
        "n_compared_horizons": int(len(horizon_deltas)),
        "mean_abs_ev_delta": float(np.mean(np.abs(ev_deltas))) if ev_deltas else 0.0,
        "max_abs_ev_delta": float(np.max(np.abs(ev_deltas))) if ev_deltas else 0.0,
    }

    return {
        "baseline_source": baseline_source,
        "horizon_deltas": horizon_deltas,
        "aggregate": aggregate,
    }
