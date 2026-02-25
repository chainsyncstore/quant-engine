"""
LightGBM model trainer with sigmoid calibration.

Trains separate models for each prediction horizon.
Post-training calibration ensures predicted probabilities are reliable.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import PredefinedSplit

from quant.config import get_research_config

logger = logging.getLogger(__name__)


@dataclass
class TrainedModel:
    """Container for a trained + calibrated model."""

    horizon: int
    model: CalibratedClassifierCV  # calibrated wrapper
    raw_model: LGBMClassifier  # uncalibrated (for feature importance)
    feature_names: list
    feature_importance: Dict[str, float]


def train(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    horizon: int,
    calibration_frac: Optional[float] = None,
    params_override: Optional[Dict] = None,
) -> TrainedModel:
    """
    Train a LightGBM classifier with sigmoid (Platt) calibration.

    The training data is split internally:
        - First (1 - calibration_frac) for LightGBM training
        - Last calibration_frac for sigmoid calibration

    Args:
        X_train: Feature matrix.
        y_train: Binary labels (0/1).
        horizon: Prediction horizon (3 or 5).
        calibration_frac: Fraction of train data for calibration (default from config).
        params_override: Optional dict of LightGBM params to override defaults.

    Returns:
        TrainedModel with calibrated predictions.
    """
    cfg = get_research_config()
    cal_frac = calibration_frac or cfg.wf_calibration_frac

    n_total = len(X_train)
    n_cal = int(n_total * cal_frac)
    n_fit = n_total - n_cal

    # Build LightGBM params (defaults + overrides)
    lgbm_params = {
        "n_estimators": cfg.lgbm_n_estimators,
        "max_depth": cfg.lgbm_max_depth,
        "learning_rate": cfg.lgbm_learning_rate,
        "subsample": cfg.lgbm_subsample,
        "colsample_bytree": cfg.lgbm_colsample_bytree,
        "min_child_samples": cfg.lgbm_min_child_samples,
        "random_state": 42,
        "verbose": -1,
        "importance_type": "gain",
    }
    if params_override:
        lgbm_params.update(params_override)

    # --- Step 1: Feature Importance Pass ---
    # Train raw model on all features to get importance
    lgbm_raw = LGBMClassifier(**lgbm_params)
    lgbm_raw.fit(X_train.values[:n_fit], y_train.values[:n_fit])

    importances = lgbm_raw.feature_importances_
    norm_importances = importances / (importances.sum() + 1e-9)
    feat_names = list(X_train.columns)
    fi = dict(zip(feat_names, norm_importances.tolist()))
    
    # --- Step 2: Pruning (Optional) ---
    selected_features = feat_names
    prune_threshold = params_override.get("prune_threshold", 0.0) if params_override else 0.0

    if prune_threshold > 0:
        selected_features = [f for f, imp in fi.items() if imp >= prune_threshold]
        
        # Safety: keep at least top 5 features if pruning is too aggressive
        if len(selected_features) < 5:
            sorted_feats = sorted(fi.items(), key=lambda x: x[1], reverse=True)
            selected_features = [f for f, _ in sorted_feats[:5]]

        if len(selected_features) < len(feat_names):
            # Retrain raw model on selected subset
            X_pruned = X_train[selected_features]
            lgbm_raw = LGBMClassifier(**lgbm_params)
            lgbm_raw.fit(X_pruned.values[:n_fit], y_train.values[:n_fit])
            
            # Update importance for pruned model
            importances = lgbm_raw.feature_importances_
            norm_importances = importances / (importances.sum() + 1e-9)
            fi = dict(zip(selected_features, norm_importances.tolist()))

    # Sort final importance
    fi = dict(sorted(fi.items(), key=lambda x: x[1], reverse=True))

    # --- Step 3: Calibration ---
    # Train final calibrated model on selected features using the holdout calibration set
    X_cal = X_train[selected_features][n_fit:]
    y_cal = y_train[n_fit:]
    
    # Wrap with sigmoid (Platt) calibration — smooth monotonic mapping
    # that preserves prediction variance (isotonic creates step functions
    # that can collapse all live predictions to a single value).
    # Uses cv="prefit" because lgbm_raw is already fit on the training portion.
    calibrated = CalibratedClassifierCV(
        estimator=lgbm_raw,  # Use the (potentially pruned) raw model as base
        method="sigmoid",
        cv="prefit",
    )
    
    calibrated.fit(X_cal.values, y_cal.values)

    logger.info(
        "Trained model_%dm: %d train, %d cal samples. Features: %d/%d. Top: %s (%.2f)",
        horizon, n_fit, n_cal, len(selected_features), len(feat_names),
        list(fi.keys())[0] if fi else "None",
        list(fi.values())[0] if fi else 0.0,
    )

    return TrainedModel(
        horizon=horizon,
        model=calibrated,
        raw_model=lgbm_raw,
        feature_names=selected_features,
        feature_importance=fi,
    )


def save_model(trained: TrainedModel, path: Path) -> None:
    """Serialize trained model to disk."""
    joblib.dump(trained, path)
    logger.info("Model saved: %s", path)


def load_model(path: Path) -> TrainedModel:
    """Deserialize trained model from disk."""
    return joblib.load(path)
