"""Tests for feature engineering pipeline."""

import pandas as pd
import numpy as np

from quant.features.pipeline import build_features, get_feature_columns
from quant.features import momentum, volatility, candle_geometry, trend, volume, time_encoding


class TestIndividualFeatures:
    def test_momentum_produces_expected_columns(self, synthetic_ohlcv):
        result = momentum.compute(synthetic_ohlcv)
        expected = {"roc_1", "roc_3", "roc_5", "roc_10", "roc_20", "momentum_accel"}
        assert expected.issubset(set(result.columns))

    def test_volatility_produces_expected_columns(self, synthetic_ohlcv):
        result = volatility.compute(synthetic_ohlcv)
        expected = {"atr_14", "rolling_std_10", "rolling_std_20", "bb_pct_b", "vol_ratio", "parkinson_vol"}
        assert expected.issubset(set(result.columns))

    def test_candle_geometry_produces_expected_columns(self, synthetic_ohlcv):
        result = candle_geometry.compute(synthetic_ohlcv)
        expected = {"body_range_ratio", "upper_wick_ratio", "lower_wick_ratio", "consec_direction", "gap"}
        assert expected.issubset(set(result.columns))

    def test_trend_produces_expected_columns(self, synthetic_ohlcv):
        result = trend.compute(synthetic_ohlcv)
        expected = {"ema_5", "ema_20", "ema_50", "ema_slope_5", "ema_cross_dist", "price_ema_spread"}
        assert expected.issubset(set(result.columns))

    def test_volume_produces_expected_columns(self, synthetic_ohlcv):
        result = volume.compute(synthetic_ohlcv)
        expected = {"vol_zscore", "vol_ratio_20", "obv_slope", "vwap_dist"}
        assert expected.issubset(set(result.columns))

    def test_time_encoding_produces_expected_columns(self, synthetic_ohlcv):
        result = time_encoding.compute(synthetic_ohlcv)
        expected = {"hour_sin", "hour_cos", "dow_sin", "dow_cos"}
        assert expected.issubset(set(result.columns))


class TestFeaturePipeline:
    def test_feature_count_within_budget(self, synthetic_ohlcv):
        result = build_features(synthetic_ohlcv)
        feature_cols = get_feature_columns(result)
        assert len(feature_cols) <= 40

    def test_no_nan_after_pipeline(self, synthetic_ohlcv):
        result = build_features(synthetic_ohlcv)
        feature_cols = get_feature_columns(result)
        assert not result[feature_cols].isna().any().any()

    def test_pipeline_preserves_ohlcv(self, synthetic_ohlcv):
        result = build_features(synthetic_ohlcv)
        assert "close" in result.columns
        assert "open" in result.columns

    def test_pipeline_drops_warmup_rows(self, synthetic_ohlcv):
        result = build_features(synthetic_ohlcv)
        # Should have fewer rows than input due to warmup
        assert len(result) < len(synthetic_ohlcv)
        assert len(result) > len(synthetic_ohlcv) * 0.9  # But not too many lost
