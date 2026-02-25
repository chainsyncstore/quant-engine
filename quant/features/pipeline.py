"""
Feature pipeline — composes all feature modules and validates output.
"""

from __future__ import annotations

import logging
from typing import List

import pandas as pd

from quant.config import get_research_config
from quant.features import (
    momentum,
    volatility,
    candle_geometry,
    trend,
    volume,
    time_encoding,
    microstructure,
    cross_timeframe,
    order_flow,
    funding_rate,
    open_interest,
    liquidation,
    crypto_session,
)

logger = logging.getLogger(__name__)

# Ordered list of feature modules — OHLCV-based first, then crypto-specific
# Note: cross_timeframe must come after trend (uses EMA columns)
_OHLCV_MODULES = [
    momentum,
    volatility,
    candle_geometry,
    trend,
    volume,
    time_encoding,
    microstructure,
    cross_timeframe,
]

# Crypto-specific modules (order flow, funding, OI, liquidation, session)
_CRYPTO_MODULES = [
    order_flow,
    funding_rate,
    open_interest,
    liquidation,
    crypto_session,
]

# Explicit whitelist of permitted feature columns 
# Prevents lookahead bias or data leakage from intermediate calculations
_FEATURE_WHITELIST = {
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
}


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all feature modules sequentially and validate.

    Crypto-only: applies OHLCV modules plus crypto-specific modules.

    Args:
        df: Raw OHLCV DataFrame (session-filtered).

    Returns:
        DataFrame with all feature columns appended, warmup NaN rows dropped.

    Raises:
        ValueError: If total feature count exceeds budget.
    """
    cfg = get_research_config()
    result = df.copy()

    modules = list(_OHLCV_MODULES)
    modules.extend(_CRYPTO_MODULES)

    for mod in modules:
        result = mod.compute(result)

    # Drop warmup NaN rows
    result = result.dropna()

    feature_cols = get_feature_columns(result)
    n_features = len(feature_cols)

    if n_features > cfg.max_features:
        raise ValueError(
            f"Feature count {n_features} exceeds budget of {cfg.max_features}. "
            f"Features: {feature_cols}"
        )

    logger.info(
        "Feature pipeline: %d features, %d rows (dropped %d warmup rows)",
        n_features,
        len(result),
        len(df) - len(result),
    )
    return result


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """Return list of valid feature column names (intersection with whitelist)."""
    return [c for c in df.columns if c in _FEATURE_WHITELIST]


def extract_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Extract only the feature columns as a clean matrix.

    Args:
        df: DataFrame with features computed by build_features().

    Returns:
        DataFrame containing only feature columns.
    """
    cols = get_feature_columns(df)
    return df[cols]
