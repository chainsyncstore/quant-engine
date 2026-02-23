from __future__ import annotations

from types import SimpleNamespace

import pytest

from quant_v2.research.forward_live import build_forward_live_simulation
from quant_v2.research.replay_regression import build_replay_regression_report


def _make_result(ev_values: list[float], *, dsr: float = 0.6, psr: float = 0.7):
    folds = []
    for ev in ev_values:
        folds.append(
            SimpleNamespace(
                metrics=SimpleNamespace(
                    spread_adjusted_ev=ev,
                    win_rate=0.55,
                    sharpe=0.8,
                    max_drawdown=-3.0,
                )
            )
        )
    return SimpleNamespace(
        folds=folds,
        robustness={
            "deflated_sharpe_ratio": dsr,
            "probabilistic_sharpe_ratio": psr,
        },
    )


def test_build_forward_live_simulation_returns_horizon_and_aggregate_views() -> None:
    horizon_results = {
        1: _make_result([1.0, 1.4, 0.8]),
        4: _make_result([0.5, 0.6, 0.4], dsr=0.58, psr=0.66),
    }

    report = build_forward_live_simulation(horizon_results)

    assert report["mode"] == "shadow_forward_live_simulation"
    assert set(report["by_horizon"].keys()) == {"1", "4"}
    assert report["by_horizon"]["1"]["fold_count"] == 3
    assert report["aggregate"]["n_horizons"] == 2
    assert report["aggregate"]["ev_mean"] > 0.0


def test_build_replay_regression_report_compares_baseline_and_current() -> None:
    current = {
        "by_horizon": {
            "1": {"ev_mean": 1.2, "win_rate_mean": 0.58, "sharpe_mean": 0.9},
            "4": {"ev_mean": 0.6, "win_rate_mean": 0.55, "sharpe_mean": 0.7},
        }
    }
    baseline = {
        "by_horizon": {
            "1": {"ev_mean": 1.0, "win_rate_mean": 0.54, "sharpe_mean": 0.8},
            "4": {"ev_mean": 0.7, "win_rate_mean": 0.56, "sharpe_mean": 0.75},
        }
    }

    deltas = build_replay_regression_report(
        current_forward_live=current,
        baseline_forward_live=baseline,
    )

    assert deltas["baseline_source"] == "external"
    assert deltas["aggregate"]["n_compared_horizons"] == 2
    assert "1" in deltas["horizon_deltas"]
    assert deltas["horizon_deltas"]["1"]["ev_mean_delta"] == pytest.approx(0.2)
    assert deltas["aggregate"]["mean_abs_ev_delta"] > 0.0
