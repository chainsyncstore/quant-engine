"""
Candle geometry features.

All features are computed from closed candles only — no lookahead.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute candle geometry features.

    Features (5):
        - body_range_ratio:     |close - open| / (high - low)
        - upper_wick_ratio:     (high - max(open,close)) / (high - low)
        - lower_wick_ratio:     (min(open,close) - low) / (high - low)
        - consec_direction:     Count of consecutive same-direction candles
        - gap:                  open[t] - close[t-1]

    Args:
        df: OHLCV DataFrame with 'open', 'high', 'low', 'close' columns.

    Returns:
        DataFrame with candle geometry feature columns appended.
    """
    out = df.copy()
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]

    candle_range = h - l
    # Avoid division by zero for doji candles
    safe_range = candle_range.replace(0, np.nan)

    # --- Body-to-range ratio ---
    out["body_range_ratio"] = (c - o).abs() / safe_range
    out["body_range_ratio"] = out["body_range_ratio"].fillna(0.0)

    # --- Upper wick ratio ---
    body_top = pd.concat([o, c], axis=1).max(axis=1)
    out["upper_wick_ratio"] = (h - body_top) / safe_range
    out["upper_wick_ratio"] = out["upper_wick_ratio"].fillna(0.0)

    # --- Lower wick ratio ---
    body_bottom = pd.concat([o, c], axis=1).min(axis=1)
    out["lower_wick_ratio"] = (body_bottom - l) / safe_range
    out["lower_wick_ratio"] = out["lower_wick_ratio"].fillna(0.0)

    # --- Consecutive direction count ---
    direction = np.sign(c - o)  # +1 bullish, -1 bearish, 0 doji
    out["consec_direction"] = _consecutive_count(direction)

    # --- Gap (open vs previous close) ---
    out["gap"] = o - c.shift(1)

    return out


def _consecutive_count(direction: pd.Series) -> pd.Series:
    """Count consecutive bars with the same direction sign."""
    result = pd.Series(0, index=direction.index, dtype=int)
    count = 0
    prev = 0
    for i, val in enumerate(direction):
        if val == prev and val != 0:
            count += 1
        else:
            count = 1 if val != 0 else 0
        result.iloc[i] = count * (1 if val >= 0 else -1)
        prev = val
    return result
