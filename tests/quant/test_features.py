"""Tests for feature engineering pipeline."""

import pandas as pd
import numpy as np
import pytest

from quant.config import get_research_config
from quant.features.pipeline import build_features, get_feature_columns
from quant.features.schema import (
    FEATURE_CATALOG_SHA256,
    FEATURE_CATALOG_VERSION,
    FEATURE_NAMES,
    feature_catalog_metadata,
    validate_feature_columns,
)
from quant.features import (
    momentum, volatility, candle_geometry, trend, volume, time_encoding,
    microstructure, crypto_session, cross_timeframe, funding_rate,
    open_interest,
)


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

    def test_microstructure_produces_expected_columns(self, synthetic_ohlcv):
        result = microstructure.compute(synthetic_ohlcv)
        expected = {"return_autocorr_5", "return_kurtosis_20", "high_low_range_ratio"}
        assert expected.issubset(set(result.columns))

    def test_crypto_session_produces_expected_columns(self, synthetic_ohlcv):
        result = crypto_session.compute(synthetic_ohlcv)
        expected = {
            "hours_to_funding",
            "hours_to_funding_sin",
            "hours_to_funding_cos",
            "post_funding_window",
            "asia_session",
            "europe_session",
            "us_session",
            "day_of_week_sin",
            "day_of_week_cos",
        }
        assert expected.issubset(set(result.columns))

    def test_cross_timeframe_produces_expected_columns(self, synthetic_ohlcv):
        # cross_timeframe uses EMA columns from trend, so compute trend first
        with_trend = trend.compute(synthetic_ohlcv)
        result = cross_timeframe.compute(with_trend)
        expected = {"roc_60", "atr_ratio_60_14", "trend_alignment"}
        assert expected.issubset(set(result.columns))

    def test_funding_rate_constant_window_produces_neutral_zscore(self):
        idx = pd.date_range("2026-01-01", periods=48, freq="h", tz="UTC")
        df = pd.DataFrame(
            {
                "funding_rate_raw": np.full(len(idx), 0.000025),
            },
            index=idx,
        )

        result = funding_rate.compute(df)
        stable_window = result["funding_rate_zscore"].iloc[24:]

        assert not stable_window.isna().any()
        assert (stable_window == 0.0).all()

    def test_open_interest_missing_history_produces_neutral_features(self):
        idx = pd.date_range("2026-01-01", periods=120, freq="h", tz="UTC")
        df = pd.DataFrame(
            {
                "close": np.linspace(100.0, 112.0, len(idx)),
                "open_interest": [np.nan] * 90 + list(np.linspace(100_000.0, 102_000.0, 30)),
            },
            index=idx,
        )

        result = open_interest.compute(df)
        feature_cols = [
            "oi_roc_1",
            "oi_roc_8",
            "oi_zscore",
            "oi_price_divergence",
            "oi_acceleration",
        ]

        assert not result[feature_cols].isna().any().any()
        assert (result.loc[idx[:90], feature_cols] == 0.0).all().all()


class TestFeaturePipeline:
    def test_feature_count_within_budget(self, synthetic_ohlcv):
        result = build_features(synthetic_ohlcv)
        feature_cols = get_feature_columns(result)
        cfg = get_research_config()
        assert len(feature_cols) >= 30  # Ensure a substantive feature set is produced
        assert len(feature_cols) <= cfg.max_features  # Respect configured budget

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
        # Pipeline should drop warmup/missing-feature rows rather than fabricating values.
        assert len(result) <= len(synthetic_ohlcv)
        assert len(result) > len(synthetic_ohlcv) * 0.9  # But not too many lost
        # Verify no NaN remains in feature columns
        from quant.features.pipeline import get_feature_columns
        feat_cols = get_feature_columns(result)
        assert result[feat_cols].isna().sum().sum() == 0

    def test_pipeline_preserves_rows_when_open_interest_history_is_sparse(self, synthetic_ohlcv):
        sparse = synthetic_ohlcv.copy()
        sparse["funding_rate_raw"] = 0.0001
        sparse["open_interest"] = np.nan
        sparse["open_interest_value"] = np.nan
        sparse.loc[sparse.index[-120:], "open_interest"] = np.linspace(100_000.0, 102_000.0, 120)
        sparse.loc[sparse.index[-120:], "open_interest_value"] = np.linspace(10_000_000.0, 10_200_000.0, 120)

        result = build_features(sparse)
        feature_cols = get_feature_columns(result)

        assert len(result) > len(sparse) * 0.9
        assert not result[feature_cols].isna().any().any()

    def test_feature_schema_metadata_is_explicit_and_stable(self, synthetic_ohlcv):
        result = build_features(synthetic_ohlcv)
        meta = feature_catalog_metadata()

        assert meta["feature_catalog_version"] == FEATURE_CATALOG_VERSION
        assert meta["feature_catalog_sha256"] == FEATURE_CATALOG_SHA256
        assert meta["feature_catalog_feature_names"] == list(FEATURE_NAMES)
        assert result.attrs["feature_catalog_version"] == FEATURE_CATALOG_VERSION
        assert result.attrs["feature_catalog_sha256"] == FEATURE_CATALOG_SHA256
        assert result.attrs["feature_missing_data_policy"] == "drop-core-and-derived-missing-v1"
        assert get_feature_columns(result) == list(FEATURE_NAMES)

    def test_feature_catalog_validation_rejects_missing_columns(self):
        with pytest.raises(ValueError, match="Missing required feature columns"):
            validate_feature_columns(["roc_1", "roc_3"])

    def test_feature_catalog_covers_known_feature_surface(self):
        expected = {
            "liquidations_long_vol",
            "liquidations_short_vol",
            "roc_1",
            "roc_3",
            "roc_5",
            "roc_10",
            "roc_20",
            "momentum_accel",
            "roc_divergence_5_20",
            "atr_14",
            "rolling_std_10",
            "rolling_std_20",
            "bb_pct_b",
            "vol_ratio",
            "parkinson_vol",
            "realized_vol_5",
            "vol_of_vol",
            "garman_klass_vol",
            "body_range_ratio",
            "upper_wick_ratio",
            "lower_wick_ratio",
            "consec_direction",
            "gap",
            "ema_slope_5",
            "ema_cross_dist",
            "price_ema_spread",
            "vol_zscore",
            "vol_ratio_20",
            "obv_slope",
            "vwap_dist",
            "hour_sin",
            "hour_cos",
            "dow_sin",
            "dow_cos",
            "return_autocorr_5",
            "return_kurtosis_20",
            "high_low_range_ratio",
            "trade_imbalance_10",
            "amihud_illiquidity",
            "kyle_lambda_20",
            "roc_60",
            "atr_ratio_60_14",
            "trend_alignment",
            "dist_ema_5_20",
            "dist_ema_20_50",
            "dist_close_ema_20",
            "ema_slope_50",
            "mean_reversion_score",
            "taker_buy_ratio",
            "taker_buy_ratio_ma8",
            "taker_buy_ratio_zscore",
            "cumulative_delta_8",
            "cumulative_delta_8_norm",
            "flow_imbalance_1",
            "flow_imbalance_4",
            "volume_weighted_flow",
            "funding_rate",
            "funding_rate_ma8",
            "funding_rate_zscore",
            "funding_rate_extreme",
            "funding_cumulative_24h",
            "funding_momentum",
            "oi_roc_1",
            "oi_roc_8",
            "oi_zscore",
            "oi_price_divergence",
            "oi_acceleration",
            "liquidation_candle",
            "liquidation_up_pressure",
            "liquidation_down_pressure",
            "post_liquidation_flag",
            "hours_to_funding",
            "hours_to_funding_sin",
            "hours_to_funding_cos",
            "post_funding_window",
            "asia_session",
            "europe_session",
            "us_session",
            "day_of_week_sin",
            "day_of_week_cos",
            "btc_return_4h",
            "btc_divergence_4h",
            "btc_correlation_24h",
            "relative_vol_ratio",
            "oi_funding_pressure",
            "price_position_24h",
            "liquidation_cascade_4h",
            "bid_ask_spread_bps",
            "book_imbalance_5",
            "book_imbalance_20",
            "depth_ratio_5",
            "volume_at_touch_ratio",
            "spread_vol_ratio",
        }

        assert set(FEATURE_NAMES) == expected
