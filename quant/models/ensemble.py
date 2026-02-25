"""
Multi-horizon ensemble predictor.

Combines predictions from 3m, 5m, and 10m models.
Signal logic:
    - Independent predictions for each horizon (using regime-specific thresholds).
    - Vote: Signal fires if ≥ 2 horizons agree.
    - Consensus sizing: Average of model confidence or fixed sizing.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from quant.models import trainer as model_trainer
from quant.models.predictor import predict_proba
from quant.regime import gmm_regime

logger = logging.getLogger(__name__)


@dataclass
class EnsembleSignal:
    timestamp: pd.Timestamp
    signal: int  # 1 (BUY), -1 (SELL), 0 (HOLD)
    confidence: float
    horizon_votes: Dict[int, bool]  # which horizons fired
    horizon_probas: Dict[int, float]
    regime: int


class MultiHorizonEnsemble:
    def __init__(self, model_dir: Path):
        self.model_dir = model_dir
        self.models: Dict[int, model_trainer.TrainedModel] = {}
        self.regime_model = None
        self.config = {}
        self._load_artifacts()

    def _load_artifacts(self):
        """Load all models and config."""
        # Load config
        with open(self.model_dir / "config.json", "r") as f:
            self.config = json.load(f)

        # Load regime model
        self.regime_model = gmm_regime.load_model(self.model_dir / "regime_model.joblib")

        # Load horizon models
        horizons = self.config.get("horizons", [3, 5, 10])
        for h in horizons:
            path = self.model_dir / f"model_{h}m.joblib"
            if path.exists():
                self.models[h] = model_trainer.load_model(path)
            else:
                logger.warning("Model for horizon %dm not found at %s", h, path)

        logger.info("Loaded ensemble with horizons: %s", list(self.models.keys()))

    def predict(self, row: pd.Series) -> EnsembleSignal:
        """
        Generate ensemble prediction for a single bar.

        Args:
            row: Feature row (must contain all features required by models).

        Returns:
            EnsembleSignal object.
        """
        # 1. Predict Regime
        # Reshape to DataFrame for compatibility
        df_row = pd.DataFrame([row])
        regime = int(gmm_regime.predict(self.regime_model, df_row)[0])

        votes = {}
        probas = {}
        vote_count = 0

        # 2. Get predictions for each horizon
        for h, model in self.models.items():
            # Get probability (predict_proba handles feature selection)
            prob = predict_proba(model, df_row)[0]
            probas[h] = prob

            # Get threshold for this regime
            thresholds = self.config["regime_thresholds"].get(str(h), {})
            thresh = float(thresholds.get(str(regime), 0.5))

            # Vote
            if prob >= thresh:
                votes[h] = True
                vote_count += 1
            else:
                votes[h] = False

        # 3. Consensus Logic
        # For multi-directional, we count long and short votes separately
        # A long vote is when prob_long >= thresh.
        # If the models output P(UP), then prob_short = 1 - proba.
        # Short vote: if (1 - prob) >= thresh.
        long_votes = {}
        short_votes = {}
        long_vote_count = 0
        short_vote_count = 0
        
        for h, prob in probas.items():
            thresholds = self.config["regime_thresholds"].get(str(h), {})
            thresh = float(thresholds.get(str(regime), 0.5))
            
            if prob >= thresh:
                long_votes[h] = True
                long_vote_count += 1
            else:
                long_votes[h] = False
                
            if (1.0 - prob) >= thresh:
                short_votes[h] = True
                short_vote_count += 1
            else:
                short_votes[h] = False

        n_models = len(self.models)
        required_votes = 2 if n_models >= 3 else n_models
        
        signal_val = 0
        active_votes = {}
        if long_vote_count >= required_votes:
            signal_val = 1 
            active_votes = long_votes
        elif short_vote_count >= required_votes:
            signal_val = -1
            active_votes = short_votes

        # Confidence is only meaningful for actionable consensus signals.
        if signal_val != 0:
            if signal_val == 1:
                voting_probas = [p for h, p in probas.items() if active_votes.get(h)]
            else:
                voting_probas = [(1.0 - p) for h, p in probas.items() if active_votes.get(h)]
            confidence = sum(voting_probas) / len(voting_probas)
        else:
            confidence = 0.0

        # We return the active directional votes as `horizon_votes` for legacy compat
        # The user of this signal will see which horizons agreed on the winning direction
        final_votes = active_votes if signal_val != 0 else long_votes

        return EnsembleSignal(
            timestamp=row.name if hasattr(row, "name") else pd.Timestamp.now(),
            signal=signal_val,
            confidence=confidence,
            horizon_votes=final_votes,
            horizon_probas=probas,
            regime=regime,
        )
