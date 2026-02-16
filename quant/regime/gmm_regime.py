"""
GMM-based regime detection.

Fits a Gaussian Mixture Model on volatility/momentum features
to identify distinct market regimes. Fitted on training data only;
applied forward without refit during test.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path

from quant.config import get_research_config

logger = logging.getLogger(__name__)

# Feature columns used as GMM input
REGIME_INPUT_FEATURES = [
    "atr_14",
    "rolling_std_20",
    "roc_5",
    "ema_slope_5",
    "vol_ratio",
]


@dataclass
class RegimeModel:
    """Container for a fitted regime model."""

    gmm: GaussianMixture
    scaler: StandardScaler
    n_regimes: int
    input_features: list = field(default_factory=lambda: list(REGIME_INPUT_FEATURES))


def fit(df: pd.DataFrame, n_regimes: Optional[int] = None) -> RegimeModel:
    """
    Fit a GMM regime model on training data.

    Args:
        df: DataFrame with feature columns (output of feature pipeline).
        n_regimes: Number of regimes (default from config).

    Returns:
        Fitted RegimeModel.
    """
    cfg = get_research_config()
    n = n_regimes or cfg.n_regimes

    X = df[REGIME_INPUT_FEATURES].values

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Fit GMM
    gmm = GaussianMixture(
        n_components=n,
        covariance_type="full",
        n_init=5,
        max_iter=200,
        random_state=42,
    )
    gmm.fit(X_scaled)

    logger.info(
        "Fitted GMM with %d regimes on %d samples (BIC: %.1f)",
        n,
        len(X_scaled),
        gmm.bic(X_scaled),
    )

    return RegimeModel(gmm=gmm, scaler=scaler, n_regimes=n)


def predict(model: RegimeModel, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Predict regime labels and probabilities.

    Args:
        model: Fitted RegimeModel.
        df: DataFrame with regime input feature columns.

    Returns:
        (regime_labels, regime_probabilities) — labels are int array,
        probabilities is (n_samples, n_regimes) array.
    """
    X = df[model.input_features].values
    X_scaled = model.scaler.transform(X)

    labels = model.gmm.predict(X_scaled)
    probas = model.gmm.predict_proba(X_scaled)

    return labels, probas


def add_regime_columns(
    df: pd.DataFrame, model: RegimeModel
) -> pd.DataFrame:
    """
    Add regime label and max probability columns to DataFrame.

    Args:
        df: Feature DataFrame.
        model: Fitted RegimeModel.

    Returns:
        DataFrame with 'regime' and 'regime_prob' columns.
    """
    labels, probas = predict(model, df)
    out = df.copy()
    out["regime"] = labels
    out["regime_prob"] = probas.max(axis=1)
    return out


def save_model(model: RegimeModel, path: Path) -> None:
    """Serialize regime model to disk."""
    joblib.dump(model, path)
    logger.info("Regime model saved: %s", path)


def load_model(path: Path) -> RegimeModel:
    """Deserialize regime model from disk."""
    model = joblib.load(path)
    logger.info("Regime model loaded: %s", path)
    return model
