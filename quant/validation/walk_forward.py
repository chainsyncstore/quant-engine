"""
Strict walk-forward validation orchestrator.

Train → Validate → Slide Forward → Repeat.
No random shuffling. Regime model and LightGBM are re-fit each fold.

Rolling threshold estimation: each fold uses thresholds discovered
from ALL PRIOR folds only — no lookahead bias.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from quant.config import get_research_config
from quant.features.pipeline import extract_feature_matrix, get_feature_columns
from quant.regime import gmm_regime
from quant.models import trainer as model_trainer
from quant.models.predictor import predict_proba
from quant.validation.metrics import (
    FoldMetrics,
    HorizonReport,
    RegimeMetrics,
    aggregate_fold_metrics,
    compute_metrics,
    compute_trade_pnl,
)

logger = logging.getLogger(__name__)

# Minimum number of past folds before regime thresholds are considered reliable
MIN_FOLDS_FOR_THRESHOLD = 3


@dataclass
class WalkForwardResult:
    """Complete walk-forward evaluation result."""

    reports: Dict[int, HorizonReport]  # horizon → report
    feature_importance: Dict[int, Dict[str, float]]  # horizon → {feat: importance}
    all_pnl: Dict[int, np.ndarray]  # horizon → concatenated PnL
    thresholds: Dict[int, Dict[int, float]]  # horizon → {regime: threshold}


def run_walk_forward(
    df: pd.DataFrame,
    horizons: Optional[List[int]] = None,
    params_override: Optional[Dict] = None,
    feature_subset: Optional[List[str]] = None,
) -> WalkForwardResult:
    """
    Execute strict walk-forward validation with rolling threshold estimation.

    Each fold's regime-gated PnL uses thresholds estimated from prior folds only,
    eliminating lookahead bias. This matches how a live system would operate.

    Args:
        df: Full DataFrame with features + labels (output of pipeline + labeler).
        horizons: Prediction horizons to evaluate (default from config).
        params_override: Optional dict of LightGBM params to override defaults.
        feature_subset: Optional list of feature columns to use (for pruning).

    Returns:
        WalkForwardResult with per-horizon reports.
    """
    cfg = get_research_config()
    horizons = horizons or cfg.horizons

    feature_cols = feature_subset or get_feature_columns(df)
    total_bars = len(df)

    logger.info(
        "Walk-forward: %d total bars, train=%d, test=%d, step=%d",
        total_bars,
        cfg.wf_train_bars,
        cfg.wf_test_bars,
        cfg.wf_step_bars,
    )

    reports: Dict[int, HorizonReport] = {}
    all_feature_importance: Dict[int, Dict[str, float]] = {}
    all_pnl: Dict[int, np.ndarray] = {}
    all_thresholds: Dict[int, Dict[int, float]] = {}

    for h in horizons:
        label_col = f"label_{h}m"
        if label_col not in df.columns:
            raise ValueError(f"Label column '{label_col}' not found. Run labeler first.")

        fold_metrics: List[FoldMetrics] = []
        fold_pnls: List[np.ndarray] = []
        importance_accum: Dict[str, List[float]] = {c: [] for c in feature_cols}

        # --- Rolling regime data: accumulates across folds ---
        # Used for threshold estimation on PAST data only
        regime_trade_data_cumulative: Dict[int, List[Tuple[np.ndarray, np.ndarray, np.ndarray]]] = {}
        # Per-fold current data for gated PnL computation
        fold_regime_data: List[Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]]] = []

        # --- Rolling gated PnL: computed per-fold using past thresholds ---
        gated_pnl_parts: List[np.ndarray] = []
        gated_n_trades_total = 0
        gated_n_wins_total = 0

        fold_idx = 0
        cursor = 0
        completed_folds = 0

        while cursor + cfg.wf_train_bars + cfg.wf_test_bars <= total_bars:
            train_end = cursor + cfg.wf_train_bars
            test_end = train_end + cfg.wf_test_bars

            train_df = df.iloc[cursor:train_end]
            test_df = df.iloc[train_end:test_end]

            X_train = extract_feature_matrix(train_df)
            y_train = train_df[label_col]
            X_test = extract_feature_matrix(test_df)
            y_test = test_df[label_col]

            # --- Filter FLAT labels (-1) from training ---
            train_mask = y_train != -1
            X_train_filtered = X_train[train_mask]
            y_train_filtered = y_train[train_mask]

            if len(X_train_filtered) < 100:
                logger.warning(
                    "Fold %d [%dm]: Skipping — only %d tradeable training samples",
                    fold_idx, h, len(X_train_filtered),
                )
                cursor += cfg.wf_step_bars
                fold_idx += 1
                continue

            # --- Fit regime model on training data only ---
            regime_model = gmm_regime.fit(train_df)

            # --- Add regime labels to test data ---
            test_with_regime = gmm_regime.add_regime_columns(test_df, regime_model)

            # --- Train model (on filtered non-FLAT data) ---
            trained = model_trainer.train(
                X_train_filtered, y_train_filtered, horizon=h,
                params_override=params_override,
            )

            # --- Predict on test ---
            probas = predict_proba(trained, X_test)

            # --- Compute price moves for PnL ---
            price_moves_raw = test_df["close"].shift(-h).values - test_df["close"].values

            # Trim to valid rows (drop NaN from forward shift at tail)
            valid_len = len(test_df) - h
            price_moves = price_moves_raw[:valid_len]
            probas_valid = probas[:valid_len]
            y_valid = y_test.values[:valid_len]

            # --- Filter FLAT labels (-1) from evaluation ---
            eval_mask = y_valid != -1
            price_moves = price_moves[eval_mask]
            probas_valid = probas_valid[eval_mask]
            y_valid = y_valid[eval_mask]

            # --- Compute PnL at default threshold (0.5) ---
            pnl = compute_trade_pnl(
                predictions=probas_valid,
                actuals=y_valid,
                price_moves=price_moves,
                threshold=0.5,
                spread=cfg.spread_price,
            )

            # --- Record fold metrics ---
            fm = compute_metrics(
                pnl=pnl,
                fold=fold_idx,
                train_start=str(train_df.index[0]),
                test_start=str(test_df.index[0]),
                test_end=str(test_df.index[-1]),
            )
            fold_metrics.append(fm)
            fold_pnls.append(pnl)

            # --- Accumulate feature importance ---
            for feat, imp in trained.feature_importance.items():
                if feat in importance_accum:
                    importance_accum[feat].append(imp)

            # --- Collect per-regime data for this fold ---
            regimes_all = test_with_regime["regime"].values[:valid_len]
            regimes = regimes_all[eval_mask]  # filter FLAT from regimes too
            this_fold_regime_data: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
            for r in range(cfg.n_regimes):
                r_mask = regimes == r
                if r_mask.any():
                    this_fold_regime_data[r] = (
                        probas_valid[r_mask],
                        y_valid[r_mask],
                        price_moves[r_mask],
                    )
            fold_regime_data.append(this_fold_regime_data)

            # --- Rolling regime-gated PnL (using PAST thresholds only) ---
            if completed_folds >= MIN_FOLDS_FOR_THRESHOLD:
                # Estimate thresholds from all PRIOR folds (no lookahead)
                rolling_thresholds, rolling_ev = _estimate_rolling_thresholds(
                    regime_trade_data_cumulative, cfg.spread_price, cfg
                )
                # Identify positive-EV regimes from past data
                positive_ev_regimes = {r for r, ev in rolling_ev.items() if ev > 0}

                # Apply past-estimated thresholds to THIS fold's data
                for r, (r_preds, r_actuals, r_moves) in this_fold_regime_data.items():
                    if r not in positive_ev_regimes:
                        continue
                    thresh = rolling_thresholds.get(r, 0.5)
                    mask = r_preds >= thresh
                    n = int(mask.sum())
                    if n > 0:
                        pnl_gated = r_moves[mask] - cfg.spread_price
                        gated_pnl_parts.append(pnl_gated)
                        gated_n_trades_total += n
                        gated_n_wins_total += int((pnl_gated > 0).sum())

            # --- Add this fold's data to cumulative (for FUTURE folds to use) ---
            for r, data in this_fold_regime_data.items():
                if r not in regime_trade_data_cumulative:
                    regime_trade_data_cumulative[r] = []
                regime_trade_data_cumulative[r].append(data)

            completed_folds += 1

            logger.info(
                "Fold %d [%dm]: EV=%.6f, WR=%.1f%%, trades=%d",
                fold_idx,
                h,
                fm.spread_adjusted_ev,
                fm.win_rate * 100,
                fm.n_trades,
            )

            cursor += cfg.wf_step_bars
            fold_idx += 1

        # --- Aggregate feature importance ---
        avg_importance = {
            feat: float(np.mean(vals)) if vals else 0.0
            for feat, vals in importance_accum.items()
        }
        avg_importance = dict(sorted(avg_importance.items(), key=lambda x: x[1], reverse=True))
        all_feature_importance[h] = avg_importance

        # --- Final regime thresholds (from ALL data — used for reporting & production) ---
        regime_metrics_list, regime_thresholds = _optimize_regime_thresholds(
            regime_trade_data_cumulative, cfg.spread_price, cfg
        )
        all_thresholds[h] = regime_thresholds

        # --- Log final regime stats ---
        positive_ev_regimes_final = {
            rm.regime for rm in regime_metrics_list if rm.ev > 0
        }
        for regime in sorted(regime_trade_data_cumulative.keys()):
            if regime not in positive_ev_regimes_final:
                logger.info(
                    "  Regime %d [%dm]: SKIPPED (negative EV)", regime, h
                )
            else:
                rm = next(r for r in regime_metrics_list if r.regime == regime)
                logger.info(
                    "  Regime %d [%dm]: %d trades @ thresh=%.2f, EV=%.6f, WR=%.1f%%",
                    regime, h, rm.n_trades, rm.optimal_threshold, rm.ev, rm.win_rate * 100,
                )

        # --- Use rolling-gated PnL for Monte Carlo (no lookahead) ---
        if gated_pnl_parts:
            gated_pnl = np.concatenate(gated_pnl_parts)
            gated_ev = float(np.mean(gated_pnl))
            gated_wr = gated_n_wins_total / gated_n_trades_total if gated_n_trades_total > 0 else 0.0
            all_pnl[h] = gated_pnl

            logger.info(
                "Horizon %dm ROLLING-GATED: EV=%.6f, WR=%.1f%%, Trades=%d "
                "(skip first %d folds for warmup)",
                h, gated_ev, gated_wr * 100, gated_n_trades_total, MIN_FOLDS_FOR_THRESHOLD,
            )
        else:
            # Fallback to original PnL if no positive-EV regimes
            if fold_pnls:
                all_pnl[h] = np.concatenate(fold_pnls)
            else:
                all_pnl[h] = np.array([])

        # --- Build overall report (from fold metrics — ungated) ---
        overall = aggregate_fold_metrics(fold_metrics)
        reports[h] = HorizonReport(
            horizon=h,
            overall_ev=overall["spread_adjusted_ev"],
            overall_win_rate=overall["win_rate"],
            overall_sharpe=overall["sharpe"],
            overall_max_drawdown=overall["max_drawdown"],
            overall_n_trades=overall["n_trades"],
            overall_worst_streak=overall["worst_losing_streak"],
            per_regime=regime_metrics_list,
            per_fold=fold_metrics,
        )

        logger.info(
            "Horizon %dm complete: %d folds, overall EV=%.6f, WR=%.1f%%",
            h,
            len(fold_metrics),
            overall["spread_adjusted_ev"],
            overall["win_rate"] * 100,
        )

    return WalkForwardResult(
        reports=reports,
        feature_importance=all_feature_importance,
        all_pnl=all_pnl,
        thresholds=all_thresholds,
    )


def _estimate_rolling_thresholds(
    regime_trade_data: Dict[int, List[Tuple[np.ndarray, np.ndarray, np.ndarray]]],
    spread: float,
    cfg,
) -> Tuple[Dict[int, float], Dict[int, float]]:
    """
    Estimate per-regime thresholds from accumulated past data.

    Returns:
        (thresholds, evs) — per-regime threshold and EV estimates.
    """
    from quant.selection.threshold_optimizer import optimize_threshold

    thresholds: Dict[int, float] = {}
    evs: Dict[int, float] = {}

    for regime, data_list in sorted(regime_trade_data.items()):
        all_preds = np.concatenate([d[0] for d in data_list])
        all_moves = np.concatenate([d[2] for d in data_list])

        best_threshold, best_ev = optimize_threshold(
            predictions=all_preds,
            price_moves=all_moves,
            spread=spread,
            threshold_min=cfg.threshold_min,
            threshold_max=cfg.threshold_max,
            threshold_step=cfg.threshold_step,
        )

        thresholds[regime] = best_threshold
        evs[regime] = best_ev

    return thresholds, evs


def _optimize_regime_thresholds(
    regime_trade_data: Dict[int, List[Tuple[np.ndarray, np.ndarray, np.ndarray]]],
    spread: float,
    cfg,
) -> Tuple[List[RegimeMetrics], Dict[int, float]]:
    """Sweep probability thresholds per regime to maximize EV."""
    from quant.selection.threshold_optimizer import optimize_threshold

    metrics_list: List[RegimeMetrics] = []
    thresholds: Dict[int, float] = {}

    for regime, data_list in sorted(regime_trade_data.items()):
        # Concatenate all folds for this regime
        all_preds = np.concatenate([d[0] for d in data_list])
        all_actuals = np.concatenate([d[1] for d in data_list])
        all_moves = np.concatenate([d[2] for d in data_list])

        best_threshold, best_ev = optimize_threshold(
            predictions=all_preds,
            price_moves=all_moves,
            spread=spread,
            threshold_min=cfg.threshold_min,
            threshold_max=cfg.threshold_max,
            threshold_step=cfg.threshold_step,
        )

        # Compute win rate at best threshold
        mask = all_preds >= best_threshold
        n_trades = int(mask.sum())
        if n_trades > 0:
            traded_pnl = all_moves[mask] - spread
            win_rate = float((traded_pnl > 0).sum() / n_trades)
        else:
            win_rate = 0.0

        thresholds[regime] = best_threshold
        metrics_list.append(
            RegimeMetrics(
                regime=regime,
                ev=best_ev,
                win_rate=win_rate,
                n_trades=n_trades,
                optimal_threshold=best_threshold,
            )
        )

    return metrics_list, thresholds
