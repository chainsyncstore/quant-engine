"""
EMA trend structure features.

All features are computed from closed candles only — no lookahead.
"""

from __future__ import annotations

import pandas as pd


def compute(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute EMA-based trend features.

    Features (6):
        - ema_5:            EMA(5)
        - ema_20:           EMA(20)
        - ema_50:           EMA(50)
        - ema_slope_5:      5-bar difference of EMA(20), normalized
        - ema_cross_dist:   (EMA(5) - EMA(20)) / close
        - price_ema_spread: (close - EMA(50)) / close

    Args:
        df: OHLCV DataFrame with 'close' column.

    Returns:
        DataFrame with trend feature columns appended.
    """
    out = df.copy()
    close = df["close"]

    # --- EMAs ---
    out["ema_5"] = close.ewm(span=5, adjust=False).mean()
    out["ema_20"] = close.ewm(span=20, adjust=False).mean()
    out["ema_50"] = close.ewm(span=50, adjust=False).mean()

    # --- EMA slope (normalized 5-bar diff of EMA20) ---
    out["ema_slope_5"] = out["ema_20"].diff(5) / close

    # --- EMA crossover distance ---
    out["ema_cross_dist"] = (out["ema_5"] - out["ema_20"]) / close

    # --- Price-to-EMA spread ---
    out["price_ema_spread"] = (close - out["ema_50"]) / close

    return out
