from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from quant_v2.models.confirmation import load_confirmation_model
from quant_v2.models.confirmation_trainer import (
    confirmation_unavailable,
    evaluate_confirmation_profitability,
    train_and_export_confirmation_model,
)


def _training_frames() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(42)
    train = pd.DataFrame(
        {
            "momentum": rng.normal(size=80),
            "volatility": rng.normal(size=80),
            "liquidity": rng.normal(size=80),
        }
    )
    train_y = ((train["momentum"] + 0.5 * train["liquidity"]) > 0).astype(int)
    validation = pd.DataFrame(
        {
            "momentum": rng.normal(size=30),
            "volatility": rng.normal(size=30),
            "liquidity": rng.normal(size=30),
        }
    )
    validation_y = ((validation["momentum"] + 0.5 * validation["liquidity"]) > 0).astype(int)
    return train, train_y, validation, validation_y


def test_train_and_export_confirmation_model_round_trips(tmp_path: Path) -> None:
    X_train, y_train, X_validation, y_validation = _training_frames()

    result = train_and_export_confirmation_model(
        X_train,
        y_train,
        X_validation,
        y_validation,
        tmp_path,
        version_id="fixture",
        horizon=4,
        hidden_dims=(),
        max_epochs=4,
        seed=7,
    )

    assert result.available is True
    assert result.horizon == 4
    assert result.shadow_only is True
    assert result.feature_schema == ["momentum", "volatility", "liquidity"]
    assert 0.0 <= result.validation_accuracy <= 1.0
    assert (tmp_path / result.config_path).is_file()
    assert (tmp_path / result.state_dict_path).is_file()

    loaded = load_confirmation_model(tmp_path / "confirmation")
    prediction = loaded.predict(X_validation.iloc[[0]])

    assert 0.0 <= prediction.probability <= 1.0
    assert 0.0 <= prediction.uncertainty <= 1.0
    assert prediction.model_version == "fixture-confirmation-4m"


def test_train_and_export_records_holdout_profitability(tmp_path: Path) -> None:
    X_train, y_train, X_validation, y_validation = _training_frames()
    forward_returns = [0.01 if label == 1 else -0.01 for label in y_validation]

    result = train_and_export_confirmation_model(
        X_train,
        y_train,
        X_validation,
        y_validation,
        tmp_path,
        version_id="fixture",
        horizon=4,
        hidden_dims=(),
        max_epochs=4,
        seed=7,
        primary_validation_probabilities=[0.9 if label == 1 else 0.1 for label in y_validation],
        validation_forward_returns=forward_returns,
        round_trip_cost_bps=5.0,
        min_profitability_samples=10,
        min_agreement_edge_bps=-100.0,
        min_agreement_win_rate=0.50,
        min_agreement_coverage=0.10,
    )

    assert result.profitability is not None
    assert result.profitability["sample_count"] == len(y_validation)
    assert result.profitability["agreement"]["sample_count"] >= 0
    assert result.profitability["disagreement"]["sample_count"] >= 0
    assert result.profitability["mode"] == "offline_holdout_shadow_evidence"


def test_evaluate_confirmation_profitability_compares_agreement_to_disagreement() -> None:
    metrics = evaluate_confirmation_profitability(
        primary_probabilities=[0.8, 0.8, 0.8, 0.2],
        confirmation_probabilities=[0.7, 0.7, 0.3, 0.7],
        forward_returns=[0.02, 0.01, -0.01, 0.01],
        round_trip_cost_bps=10.0,
        min_profitability_samples=4,
        min_agreement_edge_bps=100.0,
        min_agreement_win_rate=0.50,
        min_agreement_coverage=0.25,
    )

    assert metrics["agreement"]["sample_count"] == 2
    assert metrics["disagreement"]["sample_count"] == 2
    assert metrics["agreement"]["win_rate"] == 1.0
    assert metrics["agreement_edge_bps"] > 100.0
    assert metrics["gate_eligible"] is True


def test_train_confirmation_rejects_schema_mismatch(tmp_path: Path) -> None:
    X_train, y_train, X_validation, y_validation = _training_frames()

    with pytest.raises(ValueError, match="features must match"):
        train_and_export_confirmation_model(
            X_train,
            y_train,
            X_validation.rename(columns={"liquidity": "other"}),
            y_validation,
            tmp_path,
            version_id="fixture",
        )


def test_confirmation_unavailable_payload_is_serializable() -> None:
    payload = confirmation_unavailable("disabled", horizon=4)

    assert payload == {
        "available": False,
        "horizon": 4,
        "reason": "disabled",
        "shadow_only": True,
    }
