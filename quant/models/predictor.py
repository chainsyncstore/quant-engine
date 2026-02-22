"""
Model inference wrapper.

Provides a clean interface for generating predictions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant.models.trainer import TrainedModel


def predict_proba(model: TrainedModel, X: pd.DataFrame) -> np.ndarray:
    """
    Generate raw probability of direction = 1 (price goes up).

    Uses the raw LightGBM model directly — sigmoid/isotonic calibration
    compresses production-model predictions into an unusably narrow band
    because the full-dataset calibration is much flatter than per-fold
    calibrations used during walk-forward threshold discovery.

    Args:
        model: TrainedModel from trainer.
        X: Feature matrix (must have same columns as training).

    Returns:
        1D array of probabilities in [0, 1].
    """
    # Validate columns
    missing = set(model.feature_names) - set(X.columns)
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    X_ordered = X[model.feature_names]
    probas = model.raw_model.predict_proba(X_ordered.values)

    # Return probability of class 1 (price up)
    return probas[:, 1]
