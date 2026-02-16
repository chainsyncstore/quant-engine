"""Tests for GMM regime model."""

import numpy as np

from quant.features.pipeline import build_features
from quant.regime.gmm_regime import fit, predict, add_regime_columns


class TestGMMRegime:
    def test_fit_predict_roundtrip(self, synthetic_ohlcv):
        df = build_features(synthetic_ohlcv)
        model = fit(df, n_regimes=4)

        labels, probas = predict(model, df)
        assert len(labels) == len(df)
        assert probas.shape == (len(df), 4)

    def test_regime_labels_valid(self, synthetic_ohlcv):
        df = build_features(synthetic_ohlcv)
        model = fit(df, n_regimes=4)
        labels, _ = predict(model, df)
        assert set(labels).issubset({0, 1, 2, 3})

    def test_probabilities_sum_to_one(self, synthetic_ohlcv):
        df = build_features(synthetic_ohlcv)
        model = fit(df, n_regimes=4)
        _, probas = predict(model, df)
        np.testing.assert_allclose(probas.sum(axis=1), 1.0, atol=1e-6)

    def test_add_regime_columns(self, synthetic_ohlcv):
        df = build_features(synthetic_ohlcv)
        model = fit(df, n_regimes=4)
        result = add_regime_columns(df, model)
        assert "regime" in result.columns
        assert "regime_prob" in result.columns
        assert (result["regime_prob"] >= 0).all()
        assert (result["regime_prob"] <= 1).all()
