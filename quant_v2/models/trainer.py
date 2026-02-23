"""v2 ensemble-oriented trainer with fold-local calibration support."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import PredefinedSplit

from quant.config import get_research_config

logger = logging.getLogger(__name__)


@dataclass
class TrainedModel:
    """Container for the v2 trained stack (primary + optional calibration/meta)."""

    horizon: int
    feature_names: list[str]
    feature_importance: dict[str, float]
    primary_model: LGBMClassifier
    calibrated_model: CalibratedClassifierCV | None = None
    meta_model: LogisticRegression | None = None
    calibration_method: str = "sigmoid"
    fit_samples: int = 0
    calibration_samples: int = 0


def train(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    horizon: int,
    calibration_frac: float | None = None,
    params_override: dict[str, Any] | None = None,
) -> TrainedModel:
    """Train v2 model stack with fold-local calibration and meta refinement."""

    if X_train.empty:
        raise ValueError("X_train cannot be empty")
    if len(X_train) != len(y_train):
        raise ValueError("X_train and y_train must have same length")

    y = pd.Series(y_train).astype(int)
    unique_labels = set(int(v) for v in np.unique(y.to_numpy()))
    if not unique_labels.issubset({0, 1}):
        raise ValueError("y_train must be binary labels in {0, 1}")

    cfg = get_research_config()
    cal_frac = float(calibration_frac) if calibration_frac is not None else float(cfg.wf_calibration_frac)
    cal_frac = min(max(cal_frac, 0.05), 0.5)

    n_total = len(X_train)
    n_cal = int(round(n_total * cal_frac))
    n_cal = max(1, min(n_cal, max(n_total - 5, 1)))
    n_fit = max(n_total - n_cal, 1)

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

    X_fit = X_train.iloc[:n_fit]
    y_fit = y.iloc[:n_fit]

    primary = LGBMClassifier(**lgbm_params)
    primary.fit(X_fit.values, y_fit.values)

    importances = np.asarray(primary.feature_importances_, dtype=float)
    if importances.size == 0:
        fi = {}
    else:
        norm = float(importances.sum())
        if norm <= 0.0:
            normed = np.zeros_like(importances)
        else:
            normed = importances / norm
        fi = dict(zip(X_train.columns, normed.tolist(), strict=False))
        fi = dict(sorted(fi.items(), key=lambda kv: kv[1], reverse=True))

    calibrated_model: CalibratedClassifierCV | None = None
    meta_model: LogisticRegression | None = None

    y_cal = y.iloc[n_fit:]
    use_fold_local_calibration = (
        n_fit >= 10
        and n_cal >= 10
        and len(set(int(v) for v in y_fit.to_numpy())) == 2
        and len(set(int(v) for v in y_cal.to_numpy())) == 2
    )

    if use_fold_local_calibration:
        split_index = np.array([-1] * n_fit + [0] * n_cal)
        split = PredefinedSplit(test_fold=split_index)

        calibrated_model = CalibratedClassifierCV(
            estimator=LGBMClassifier(**lgbm_params),
            method="sigmoid",
            cv=split,
        )
        calibrated_model.fit(X_train.values, y.values)

        # Meta-refinement model over primary probabilities.
        primary_cal_proba = primary.predict_proba(X_train.iloc[n_fit:].values)[:, 1]
        meta_features = primary_cal_proba.reshape(-1, 1)
        meta_model = LogisticRegression(max_iter=300, random_state=42)
        meta_model.fit(meta_features, y_cal.values)
    else:
        logger.info(
            "Skipping fold-local calibration/meta for horizon=%s due to insufficient split diversity "
            "(fit=%s, cal=%s)",
            horizon,
            n_fit,
            n_cal,
        )

    logger.info(
        "Trained v2 model_%sm: fit=%s cal=%s features=%s calibrated=%s meta=%s",
        horizon,
        n_fit,
        n_cal,
        len(X_train.columns),
        calibrated_model is not None,
        meta_model is not None,
    )

    return TrainedModel(
        horizon=int(horizon),
        feature_names=list(X_train.columns),
        feature_importance=fi,
        primary_model=primary,
        calibrated_model=calibrated_model,
        meta_model=meta_model,
        fit_samples=n_fit,
        calibration_samples=n_cal,
    )


def save_model(trained: TrainedModel, path: Path) -> None:
    """Persist trained v2 model stack."""

    joblib.dump(trained, path)


def load_model(path: Path) -> TrainedModel:
    """Load trained v2 model stack from disk."""

    return joblib.load(path)
