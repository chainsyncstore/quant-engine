"""Inference helpers for v2 model stack."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_v2.models.trainer import TrainedModel


def predict_proba(model: TrainedModel, X: pd.DataFrame) -> np.ndarray:
    """Return class-1 probabilities from v2 primary/meta/calibrated ensemble."""

    missing = set(model.feature_names) - set(X.columns)
    if missing:
        raise ValueError(f"Missing feature columns: {sorted(missing)}")

    X_ordered = X[model.feature_names]
    primary = model.primary_model.predict_proba(X_ordered.values)[:, 1]

    calibrated = primary
    if model.calibrated_model is not None:
        calibrated = model.calibrated_model.predict_proba(X_ordered.values)[:, 1]

    refined = calibrated
    if model.meta_model is not None:
        meta_input = primary.reshape(-1, 1)
        refined = model.meta_model.predict_proba(meta_input)[:, 1]

    if model.calibrated_model is not None and model.meta_model is not None:
        combined = 0.5 * calibrated + 0.5 * refined
    else:
        combined = refined

    return np.clip(np.asarray(combined, dtype=float), 0.0, 1.0)


def predict_proba_with_uncertainty(model: TrainedModel, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return probabilities plus normalized uncertainty in [0, 1]."""

    proba = predict_proba(model, X)
    safe = np.clip(proba, 1e-9, 1.0 - 1e-9)

    entropy = -(safe * np.log2(safe) + (1.0 - safe) * np.log2(1.0 - safe))
    margin_uncertainty = 1.0 - np.abs(2.0 * safe - 1.0)

    uncertainty = 0.5 * entropy + 0.5 * margin_uncertainty
    return proba, np.clip(uncertainty, 0.0, 1.0)
