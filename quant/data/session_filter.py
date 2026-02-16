"""
Session filter — keeps only London + New York trading hours.

Drops weekends and out-of-session bars.
"""

from __future__ import annotations

import logging

import pandas as pd

from quant.config import get_session_config

logger = logging.getLogger(__name__)


def filter_sessions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter DataFrame to London + NY session hours only.

    Args:
        df: OHLCV DataFrame with UTC DatetimeIndex.

    Returns:
        Filtered DataFrame containing only bars within session hours.
    """
    if df.empty:
        return df

    cfg = get_session_config()
    idx = df.index

    # Drop weekends (Saturday=5, Sunday=6)
    weekday_mask = idx.weekday < 5

    # Combined session: earliest start to latest end
    # London 08:00-16:30, NY 13:00-21:00 → combined 08:00-21:00
    london_start_minutes = cfg.london[0] * 60 + cfg.london[1]
    ny_end_minutes = cfg.new_york[2] * 60 + cfg.new_york[3]

    time_in_minutes = idx.hour * 60 + idx.minute
    session_mask = (time_in_minutes >= london_start_minutes) & (time_in_minutes < ny_end_minutes)

    combined_mask = weekday_mask & session_mask
    filtered = df.loc[combined_mask].copy()

    dropped = len(df) - len(filtered)
    logger.info(
        "Session filter: kept %d / %d bars (dropped %d out-of-session)",
        len(filtered),
        len(df),
        dropped,
    )
    return filtered
