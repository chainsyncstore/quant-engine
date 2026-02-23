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

# Columns that are NOT features (raw data + intermediates + labels + regime)
_NON_FEATURE_COLS = {
    "open", "high", "low", "close", "volume",
    "ema_5", "ema_20", "ema_50",
    "bid_close", "ask_close", "spread",
    "taker_buy_volume", "taker_sell_volume",
    "funding_rate_raw",
    "open_interest", "open_interest_value",
}
_NON_FEATURE_PREFIXES = ("label_", "regime")


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
    """Return list of feature column names (excludes OHLCV, labels, regime cols)."""
    return [
        c for c in df.columns
        if c not in _NON_FEATURE_COLS and not c.startswith(_NON_FEATURE_PREFIXES)
    ]


def extract_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Extract only the feature columns as a clean matrix.

    Args:
        df: DataFrame with features computed by build_features().

    Returns:
        DataFrame containing only feature columns.
    """
    cols = get_feature_columns(df)
    return df[cols]
