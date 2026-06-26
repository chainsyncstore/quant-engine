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
    liquidation_proximity,   # Phase 1 upgrade
    cross_pair,              # Phase 1 upgrade
    crypto_session,
    order_book,              # Phase B: L2 order book snapshot features
)
from quant.features.schema import (
    FEATURE_NAMES,
    attach_feature_catalog,
    feature_catalog_metadata,
    validate_feature_catalog,
    validate_feature_columns,
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
    liquidation_proximity,   # NEW — Phase 1
    cross_pair,              # NEW — Phase 1 (needs other features, before crypto_session)
    crypto_session,
    order_book,              # Phase B: L2 order book snapshot features
]

# Explicit catalog-backed whitelist of permitted feature columns.
_FEATURE_WHITELIST = set(FEATURE_NAMES)
validate_feature_catalog()


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

    # Drop rows only where core OHLCV is missing; preserve feature missingness
    # until we can either quarantine the candidate or fail closed upstream.
    core_cols = ["open", "high", "low", "close", "volume"]
    core_missing = result[core_cols].isna().any(axis=1)
    result = result[~core_missing].copy()
    validate_feature_columns(result.columns)
    feature_cols = get_feature_columns(result)
    feature_missing_rows = result[feature_cols].isna().any(axis=1)
    if feature_missing_rows.any():
        dropped = int(feature_missing_rows.sum())
        logger.info(
            "Feature pipeline: dropping %d rows with missing derived features instead of fabricating values",
            dropped,
        )
        result = result.loc[~feature_missing_rows].copy()

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
    result = attach_feature_catalog(result)
    return result


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """Return list of valid feature column names (intersection with whitelist)."""
    return [c for c in FEATURE_NAMES if c in df.columns]


def extract_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Extract only the feature columns as a clean matrix.

    Args:
        df: DataFrame with features computed by build_features().

    Returns:
        DataFrame containing only feature columns.
    """
    cols = get_feature_columns(df)
    matrix = df[cols].copy()
    matrix.attrs.update(dict(getattr(df, "attrs", {}) or {}))
    if not matrix.attrs.get("feature_catalog_version"):
        matrix.attrs.update(feature_catalog_metadata())
    if not matrix.attrs.get("feature_missing_data_policy"):
        matrix.attrs["feature_missing_data_policy"] = "drop-core-and-derived-missing-v1"
    return matrix
