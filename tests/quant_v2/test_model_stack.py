from __future__ import annotations

import numpy as np
import pandas as pd

from quant_v2.models.predictor import predict_proba, predict_proba_with_uncertainty
from quant_v2.models.trainer import train


def _sample_training_data(rows: int = 200) -> tuple[pd.DataFrame, pd.Series]:
    idx = pd.date_range("2025-01-01", periods=rows, freq="1h", tz="UTC")
    x1 = np.linspace(-2.0, 2.0, rows)
    x2 = np.sin(np.linspace(0.0, 10.0, rows))
    x3 = np.cos(np.linspace(0.0, 8.0, rows))

    logits = 0.9 * np.sin(np.linspace(0.0, 18.0, rows)) + 0.5 * x2 - 0.2 * x3
    y = (logits > 0.0).astype(int)

    X = pd.DataFrame({"f1": x1, "f2": x2, "f3": x3}, index=idx)
    return X, pd.Series(y, index=idx)


def test_v2_model_stack_train_exposes_primary_calibration_meta_layers() -> None:
    X, y = _sample_training_data(rows=220)

    trained = train(
        X,
        y,
        horizon=1,
        calibration_frac=0.2,
    )

    assert trained.horizon == 1
    assert trained.primary_model is not None
    assert trained.calibrated_model is not None
    assert trained.meta_model is not None
    assert trained.fit_samples > 0
    assert trained.calibration_samples > 0
    assert set(trained.feature_names) == {"f1", "f2", "f3"}
    assert trained.feature_importance


def test_v2_model_stack_predicts_probability_and_uncertainty() -> None:
    X, y = _sample_training_data(rows=180)
    trained = train(X, y, horizon=4, calibration_frac=0.2)

    batch = X.tail(25)
    proba = predict_proba(trained, batch)
    assert proba.shape == (25,)
    assert np.all(np.isfinite(proba))
    assert np.all((proba >= 0.0) & (proba <= 1.0))

    proba2, uncertainty = predict_proba_with_uncertainty(trained, batch)
    assert proba2.shape == (25,)
    assert uncertainty.shape == (25,)
    assert np.all((proba2 >= 0.0) & (proba2 <= 1.0))
    assert np.all((uncertainty >= 0.0) & (uncertainty <= 1.0))
    assert np.allclose(proba, proba2)
