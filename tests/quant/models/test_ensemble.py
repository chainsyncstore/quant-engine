import json
import joblib
import pandas as pd
import numpy as np
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from quant.models.ensemble import MultiHorizonEnsemble

# Mock the TrainedModel class
class MockTrainedModel:
    def __init__(self, proba_val):
        self.proba_val = proba_val
        self.feature_names = ["f1", "f2"]
        # Mock raw model used by quant.models.predictor.predict_proba
        self.raw_model = MagicMock()
        # Keep calibrated model attr for compatibility with TrainedModel shape
        self.model = MagicMock()
        # predict_proba returns [[1-p, p]]
        self.raw_model.predict_proba.return_value = np.array([[1 - proba_val, proba_val]])
        self.model.predict_proba.return_value = np.array([[1 - proba_val, proba_val]])

@pytest.fixture
def ensemble(tmp_path):
    # Setup a dummy model directory
    model_dir = tmp_path / "model_test"
    model_dir.mkdir()
    
    # Create a config
    config = {
        "horizons": [3, 5, 10],
        "regime_thresholds": {
            "3": {"0": 0.6},
            "5": {"0": 0.6},
            "10": {"0": 0.6},
        },
    }
    with open(model_dir / "config.json", "w") as f:
        json.dump(config, f)
        
    # Mock gmm_regime
    with patch("quant.regime.gmm_regime.load_model") as mock_load_regime:
        mock_load_regime.return_value = MagicMock()
        
        # Instantiate ensemble, but mock the load_model calls
        with patch("quant.models.trainer.load_model") as mock_load_model:
            # We will manually populate models later
            mock_load_model.return_value = MagicMock()
            ens = MultiHorizonEnsemble(model_dir)

    return ens

def test_ensemble_voting_consensus(ensemble):
    # Setup mock models
    # Horizon 3: HIGH confidence (0.8) -> VOTE YES
    model_3 = MockTrainedModel(0.8)
    
    # Horizon 5: HIGH confidence (0.7) -> VOTE YES
    model_5 = MockTrainedModel(0.7)
    
    # Horizon 10: LOW confidence (0.4) -> VOTE NO
    model_10 = MockTrainedModel(0.4)

    ensemble.models = {
        3: model_3,
        5: model_5,
        10: model_10,
    }

    # Mock regime prediction to return 0
    with patch("quant.regime.gmm_regime.predict", return_value=np.array([0])):
        # Predict
        row = pd.Series({"f1": 1, "f2": 2})
        result = ensemble.predict(row)

    # 2 out of 3 voted YES -> Signal should be 1 (BUY)
    assert result.signal == 1
    assert result.horizon_votes[3] is True
    assert result.horizon_votes[5] is True
    assert result.horizon_votes[10] is False
    
    # Confidence should be average of VOTING models (0.8 + 0.7) / 2 = 0.75
    assert result.confidence == 0.75

def test_ensemble_no_consensus(ensemble):
    # Horizon 3: HIGH confidence (0.8) -> VOTE YES
    model_3 = MockTrainedModel(0.8)
    
    # Horizon 5: LOW confidence (0.5) -> VOTE NO (thresh 0.6)
    model_5 = MockTrainedModel(0.5)
    
    # Horizon 10: LOW confidence (0.4) -> VOTE NO
    model_10 = MockTrainedModel(0.4)

    ensemble.models = {
        3: model_3,
        5: model_5,
        10: model_10,
    }

    # Mock regime prediction to return 0
    with patch("quant.regime.gmm_regime.predict", return_value=np.array([0])):
        row = pd.Series({"f1": 1, "f2": 2})
        result = ensemble.predict(row)

    # Only 1 vote -> Signal should be 0 (HOLD)
    assert result.signal == 0
    assert result.confidence == 0.0

def test_ensemble_missing_models(ensemble):
    # Only 2 models loaded
    model_3 = MockTrainedModel(0.9)
    model_5 = MockTrainedModel(0.9)
    
    ensemble.models = {3: model_3, 5: model_5}
    
    # Both vote YES -> Consensus (2/2) -> Signal 1
    with patch("quant.regime.gmm_regime.predict", return_value=np.array([0])):
        row = pd.Series({"f1": 1, "f2": 2})
        result = ensemble.predict(row)
        
    assert result.signal == 1
