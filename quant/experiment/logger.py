"""
Experiment logger — structured JSON experiment tracking.

Logs config, metrics, feature importance, and Monte Carlo results
for each research run.
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

from quant.config import get_path_config, get_research_config
from quant.risk.monte_carlo import MonteCarloResult
from quant.validation.walk_forward import WalkForwardResult

logger = logging.getLogger(__name__)


def determine_verdict(result: WalkForwardResult) -> str:
    """
    Determine research verdict based on walk-forward results.

    Rules:
        EDGE_FOUND:  EV > 0 in ≥1 regime+horizon, stable across ≥60% of folds
        UNSTABLE:    EV > 0 but stable in <60% of folds
        NO_EDGE:     EV ≤ 0 everywhere
    """
    for h, report in result.reports.items():
        # Check if any regime has positive EV
        has_positive_regime = any(rm.ev > 0 for rm in report.per_regime)

        if has_positive_regime:
            # Check stability: what fraction of folds had positive EV?
            positive_folds = sum(
                1 for f in report.per_fold if f.spread_adjusted_ev > 0
            )
            total_folds = len(report.per_fold)

            if total_folds > 0 and (positive_folds / total_folds) >= 0.60:
                return "EDGE_FOUND"
            else:
                return "UNSTABLE"

    return "NO_EDGE"


def save_experiment(
    result: WalkForwardResult,
    mc_results: Dict[int, MonteCarloResult],
    snapshot_path: Optional[str] = None,
    duration_seconds: float = 0.0,
) -> Path:
    """
    Save experiment results to JSON.

    Args:
        result: Walk-forward evaluation result.
        mc_results: Monte Carlo results per horizon.
        snapshot_path: Path to the data snapshot used.
        duration_seconds: Total pipeline execution time.

    Returns:
        Path to the saved experiment log.
    """
    cfg = get_research_config()
    paths = get_path_config()

    run_id = uuid.uuid4().hex[:8]
    ts = datetime.now(timezone.utc)
    ts_str = ts.strftime("%Y%m%d_%H%M%S")

    experiment = {
        "run_id": run_id,
        "timestamp": ts.isoformat(),
        "duration_seconds": duration_seconds,
        "config": {
            "spread_pips": cfg.spread_pips,
            "horizons": cfg.horizons,
            "n_regimes": cfg.n_regimes,
            "wf_train_bars": cfg.wf_train_bars,
            "wf_test_bars": cfg.wf_test_bars,
            "wf_step_bars": cfg.wf_step_bars,
            "model_params": {
                "n_estimators": cfg.lgbm_n_estimators,
                "max_depth": cfg.lgbm_max_depth,
                "learning_rate": cfg.lgbm_learning_rate,
            },
            "data_snapshot": str(snapshot_path) if snapshot_path else None,
        },
        "results": {},
        "monte_carlo": {},
        "feature_importance": {},
        "verdict": determine_verdict(result),
    }

    # Per-horizon results
    for h, report in result.reports.items():
        experiment["results"][str(h)] = {
            "overall": {
                "spread_adjusted_ev": report.overall_ev,
                "win_rate": report.overall_win_rate,
                "sharpe": report.overall_sharpe,
                "max_drawdown": report.overall_max_drawdown,
                "n_trades": report.overall_n_trades,
                "worst_losing_streak": report.overall_worst_streak,
            },
            "per_regime": [asdict(rm) for rm in report.per_regime],
            "per_fold": [asdict(fm) for fm in report.per_fold],
        }

    # Feature importance
    for h, fi in result.feature_importance.items():
        experiment["feature_importance"][str(h)] = fi

    # Monte Carlo
    for h, mc in mc_results.items():
        experiment["monte_carlo"][str(h)] = {
            "ruin_probability": mc.ruin_probability,
            "ev_ci_95": list(mc.ev_ci_95),
            "median_final_pnl": mc.median_final_pnl,
            "p5_final_pnl": mc.p5_final_pnl,
            "p95_final_pnl": mc.p95_final_pnl,
            "worst_streak_p50": mc.worst_streak_p50,
            "worst_streak_p95": mc.worst_streak_p95,
        }

    # Save
    fpath = paths.experiments / f"run_{ts_str}_{run_id}.json"
    fpath.parent.mkdir(parents=True, exist_ok=True)
    with open(fpath, "w") as f:
        json.dump(experiment, f, indent=2, default=str)

    logger.info("Experiment saved: %s (verdict: %s)", fpath, experiment["verdict"])
    return fpath
