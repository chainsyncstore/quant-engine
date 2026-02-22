"""
Optuna-based Bayesian hyperparameter optimization for LightGBM.

Runs a fast mini walk-forward (3 folds) per trial to evaluate each
parameter set, using spread-adjusted EV as the objective.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import PredefinedSplit

from quant.config import get_research_config
from quant.features.pipeline import extract_feature_matrix, get_feature_columns
from quant.validation.metrics import compute_trade_pnl

logger = logging.getLogger(__name__)

# Suppress Optuna's verbose logging
optuna.logging.set_verbosity(optuna.logging.WARNING)


def _fast_walk_forward_ev(
    df: pd.DataFrame,
    params: Dict,
    horizon: int,
    n_folds: int = 3,
    feature_cols: Optional[List[str]] = None,
) -> float:
    """
    Run a fast mini walk-forward with given params, return mean EV.

    Uses fewer folds and skips regime modeling for speed.
    """
    cfg = get_research_config()
    label_col = f"label_{horizon}m"
    total_bars = len(df)

    if feature_cols is None:
        feature_cols = get_feature_columns(df)

    fold_evs = []
    cursor = 0

    for fold in range(n_folds):
        if cursor + cfg.wf_train_bars + cfg.wf_test_bars > total_bars:
            break

        train_end = cursor + cfg.wf_train_bars
        test_end = train_end + cfg.wf_test_bars

        train_df = df.iloc[cursor:train_end]
        test_df = df.iloc[train_end:test_end]

        X_train = train_df[feature_cols]
        y_train = train_df[label_col]
        X_test = test_df[feature_cols]
        y_test = test_df[label_col]

        # Filter FLAT labels
        mask = y_train != -1
        X_train = X_train[mask]
        y_train = y_train[mask]

        if len(X_train) < 200:
            cursor += cfg.wf_step_bars
            continue

        # Split for calibration
        cal_frac = cfg.wf_calibration_frac
        n_total = len(X_train)
        n_cal = int(n_total * cal_frac)
        n_fit = n_total - n_cal

        split_index = np.array([-1] * n_fit + [0] * n_cal)
        ps = PredefinedSplit(test_fold=split_index)

        lgbm = LGBMClassifier(**params, random_state=42, verbose=-1)
        calibrated = CalibratedClassifierCV(estimator=lgbm, method="isotonic", cv=ps)
        calibrated.fit(X_train.values, y_train.values)

        probas = calibrated.predict_proba(X_test.values)[:, 1]

        # Compute EV
        price_moves_raw = test_df["close"].shift(-horizon).values - test_df["close"].values
        valid_len = len(test_df) - horizon
        price_moves = price_moves_raw[:valid_len]
        probas_valid = probas[:valid_len]
        y_valid = y_test.values[:valid_len]

        # Filter FLAT from eval
        eval_mask = y_valid != -1
        if eval_mask.sum() < 10:
            cursor += cfg.wf_step_bars
            continue

        pnl = compute_trade_pnl(
            predictions=probas_valid[eval_mask],
            actuals=y_valid[eval_mask],
            price_moves=price_moves[eval_mask],
            threshold=0.5,
            spread=cfg.spread_price,
        )

        if len(pnl) > 0:
            fold_evs.append(float(np.mean(pnl)))

        cursor += cfg.wf_step_bars

    return float(np.mean(fold_evs)) if fold_evs else -1.0


def optimize_hyperparams(
    df: pd.DataFrame,
    horizon: int = 3,
    n_trials: int = 50,
    feature_cols: Optional[List[str]] = None,
) -> Dict:
    """
    Run Optuna Bayesian optimization to find best LightGBM hyperparameters.

    Args:
        df: Full DataFrame with features + labels.
        horizon: Prediction horizon to optimize for.
        n_trials: Number of Optuna trials.
        feature_cols: Optional feature subset to use.

    Returns:
        Dict of best hyperparameters.
    """
    cfg = get_research_config()

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=50),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 200, step=10),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }

        ev = _fast_walk_forward_ev(df, params, horizon, n_folds=3, feature_cols=feature_cols)
        return ev

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = study.best_params
    logger.info(
        "Optuna HPO complete: %d trials, best EV=%.6f",
        n_trials,
        study.best_value,
    )
    logger.info("Best params: %s", best)

    return best
