"""Tests for multi-horizon model ensemble."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from quant_v2.models.ensemble import HorizonEnsemble


def _mock_trained_model(feature_names: list[str], proba: float, uncertainty: float) -> MagicMock:
    """Create a mock TrainedModel with expected attributes."""
    mock = MagicMock()
    mock.feature_names = feature_names
    mock.primary_model = MagicMock()
    mock.calibrated_model = None
    mock.meta_model = None
    return mock, proba, uncertainty


def test_horizon_ensemble_predict_weighted_average() -> None:
    """Test ensemble combines predictions with weights."""
    # Create mock models with different predictions
    mock_model_2, p2, u2 = _mock_trained_model(["feat1", "feat2"], 0.6, 0.2)
    mock_model_4, p4, u4 = _mock_trained_model(["feat1", "feat2"], 0.7, 0.3)
    mock_model_8, p8, u8 = _mock_trained_model(["feat1", "feat2"], 0.8, 0.4)

    models = {2: mock_model_2, 4: mock_model_4, 8: mock_model_8}

    ensemble = HorizonEnsemble(models)

    # Mock the predict_proba_with_uncertainty function
    def mock_predict(model, X):
        if model is mock_model_2:
            return np.array([p2]), np.array([u2])
        elif model is mock_model_4:
            return np.array([p4]), np.array([u4])
        elif model is mock_model_8:
            return np.array([p8]), np.array([u8])
        return np.array([0.5]), np.array([1.0])

    with patch("quant_v2.models.ensemble.predict_proba_with_uncertainty", side_effect=mock_predict):
        X = pd.DataFrame({"feat1": [1.0], "feat2": [2.0]})
        proba, uncertainty = ensemble.predict(X)

    # Check that result is a weighted average (not equal to any single prediction)
    assert 0.6 < proba < 0.8  # Should be between the min and max
    assert 0.2 < uncertainty < 0.4  # Weighted uncertainty


def test_horizon_ensemble_agreement_bonus() -> None:
    """Test that agreement reduces uncertainty by 20%."""
    # All models agree on direction (proba > 0.5)
    mock_model_2, _, _ = _mock_trained_model(["feat1"], 0.7, 0.2)  # up
    mock_model_4, _, _ = _mock_trained_model(["feat1"], 0.75, 0.3)  # up
    mock_model_8, _, _ = _mock_trained_model(["feat1"], 0.8, 0.4)  # up

    models = {2: mock_model_2, 4: mock_model_4, 8: mock_model_8}
    ensemble = HorizonEnsemble(models)

    def mock_predict(model, X):
        if model is mock_model_2:
            return np.array([0.7]), np.array([0.2])
        elif model is mock_model_4:
            return np.array([0.75]), np.array([0.3])
        elif model is mock_model_8:
            return np.array([0.8]), np.array([0.4])
        return np.array([0.5]), np.array([1.0])

    with patch("quant_v2.models.ensemble.predict_proba_with_uncertainty", side_effect=mock_predict):
        X = pd.DataFrame({"feat1": [1.0]})
        proba, uncertainty = ensemble.predict(X)

    # Without agreement bonus, uncertainty would be higher
    # With all models agreeing, uncertainty should be reduced by 20%
    # (weighted uncertainty would be ~0.29, with bonus ~0.23)
    assert uncertainty < 0.27  # Uncertainty should be reduced


def test_horizon_ensemble_no_agreement_bonus_when_disagree() -> None:
    """Test that disagreement doesn't get uncertainty reduction."""
    # Models disagree: some say up, some say down
    mock_model_2, _, _ = _mock_trained_model(["feat1"], 0.6, 0.2)  # up
    mock_model_4, _, _ = _mock_trained_model(["feat1"], 0.4, 0.3)  # down

    models = {2: mock_model_2, 4: mock_model_4}
    ensemble = HorizonEnsemble(models)

    def mock_predict(model, X):
        if model is mock_model_2:
            return np.array([0.6]), np.array([0.2])
        elif model is mock_model_4:
            return np.array([0.4]), np.array([0.3])
        return np.array([0.5]), np.array([1.0])

    with patch("quant_v2.models.ensemble.predict_proba_with_uncertainty", side_effect=mock_predict):
        X = pd.DataFrame({"feat1": [1.0]})
        proba, uncertainty = ensemble.predict(X)

    # Without agreement, uncertainty should be closer to 0.25 (weighted average of 0.2 and 0.3)
    assert uncertainty > 0.24


def test_horizon_ensemble_graceful_fallback_on_model_failure() -> None:
    """Test ensemble continues if one model fails."""
    mock_model_2, _, _ = _mock_trained_model(["feat1"], 0.6, 0.2)
    mock_model_4, _, _ = _mock_trained_model(["feat1"], 0.7, 0.3)

    models = {2: mock_model_2, 4: mock_model_4}
    ensemble = HorizonEnsemble(models)

    call_count = 0
    def mock_predict(model, X):
        nonlocal call_count
        call_count += 1
        if model is mock_model_2:
            raise ValueError("Model 2 failed!")
        elif model is mock_model_4:
            return np.array([0.7]), np.array([0.3])
        return np.array([0.5]), np.array([1.0])

    with patch("quant_v2.models.ensemble.predict_proba_with_uncertainty", side_effect=mock_predict):
        X = pd.DataFrame({"feat1": [1.0]})
        proba, uncertainty = ensemble.predict(X)

    # Should still return valid predictions from model_4
    assert proba == pytest.approx(0.7)  # Only model_4 contributed
    assert uncertainty == pytest.approx(0.3)


def test_horizon_ensemble_total_failure_returns_neutral() -> None:
    """Test ensemble returns (0.5, 1.0) if all models fail."""
    mock_model_2, _, _ = _mock_trained_model(["feat1"], 0.6, 0.2)
    models = {2: mock_model_2}
    ensemble = HorizonEnsemble(models)

    def mock_predict(model, X):
        raise ValueError("All models failed!")

    with patch("quant_v2.models.ensemble.predict_proba_with_uncertainty", side_effect=mock_predict):
        X = pd.DataFrame({"feat1": [1.0]})
        proba, uncertainty = ensemble.predict(X)

    # Should return neutral values
    assert proba == 0.5
    assert uncertainty == 1.0


def test_horizon_ensemble_from_directory_empty() -> None:
    """Test from_directory returns None for empty directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        artifact_dir = Path(tmpdir)
        ensemble = HorizonEnsemble.from_directory(artifact_dir)
        assert ensemble is None


def test_horizon_ensemble_from_directory_with_models() -> None:
    """Test from_directory loads models from files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        artifact_dir = Path(tmpdir)

        # Create dummy files (content doesn't matter since we mock load_model)
        (artifact_dir / "model_2m.pkl").touch()
        (artifact_dir / "model_4m.pkl").touch()

        # Create mock models to be returned by load_model
        mock_model_2 = MagicMock()
        mock_model_2.feature_names = ["feat1"]
        mock_model_2.primary_model = MagicMock()

        mock_model_4 = MagicMock()
        mock_model_4.feature_names = ["feat1"]
        mock_model_4.primary_model = MagicMock()

        # Mock load_model to return our mock models
        def mock_load_model(path):
            if "model_2m" in str(path):
                return mock_model_2
            elif "model_4m" in str(path):
                return mock_model_4
            raise FileNotFoundError()

        with patch("quant_v2.models.ensemble.load_model", side_effect=mock_load_model):
            ensemble = HorizonEnsemble.from_directory(artifact_dir)

        assert ensemble is not None
        assert ensemble.horizon_count == 2
        assert 2 in ensemble.models
        assert 4 in ensemble.models


def test_horizon_ensemble_rejects_missing_features() -> None:
    """Test ensemble fails closed when a required feature is missing."""
    # Model expects feat1, feat2, feat3 but we only provide feat1, feat2
    mock_model = MagicMock()
    mock_model.feature_names = ["feat1", "feat2", "feat3"]
    mock_model.primary_model = MagicMock()

    models = {2: mock_model}
    ensemble = HorizonEnsemble(models)

    X = pd.DataFrame({"feat1": [1.0], "feat2": [2.0]})
    with pytest.raises(ValueError, match="Missing feature columns"):
        ensemble.predict(X)


def test_horizon_ensemble_weights_normalization() -> None:
    """Test weights are normalized to sum to 1.0."""
    mock_model_2, _, _ = _mock_trained_model(["feat1"], 0.6, 0.2)
    mock_model_4, _, _ = _mock_trained_model(["feat1"], 0.7, 0.3)

    models = {2: mock_model_2, 4: mock_model_4}
    custom_weights = {2: 0.5, 4: 0.5}  # Equal weights

    ensemble = HorizonEnsemble(models, weights=custom_weights)

    # Check weights are normalized
    total_weight = sum(ensemble.weights.values())
    assert total_weight == pytest.approx(1.0)


def test_single_model_ensemble_behaves_like_direct() -> None:
    """Test single-model ensemble behaves like direct prediction."""
    mock_model, expected_proba, expected_uncertainty = _mock_trained_model(["feat1"], 0.75, 0.25)

    models = {4: mock_model}
    ensemble = HorizonEnsemble(models)

    def mock_predict(model, X):
        return np.array([expected_proba]), np.array([expected_uncertainty])

    with patch("quant_v2.models.ensemble.predict_proba_with_uncertainty", side_effect=mock_predict):
        X = pd.DataFrame({"feat1": [1.0]})
        proba, uncertainty = ensemble.predict(X)

    # Single model should pass through directly
    assert proba == pytest.approx(expected_proba)
    assert uncertainty == pytest.approx(expected_uncertainty)
