"""Session filter for crypto mode (24/7 market pass-through)."""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def filter_sessions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return bars unchanged for crypto 24/7 trading.

    Args:
        df: OHLCV DataFrame with UTC DatetimeIndex.

    Returns:
        Input DataFrame (pass-through).
    """
    if df.empty:
        return df

    logger.info("Session filter: crypto mode — all %d bars passed through", len(df))
    return df
