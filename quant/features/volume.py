"""
Volume normalization features.

All features are computed from closed candles only — no lookahead.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute volume features.

    Features (4):
        - vol_zscore:   Volume Z-score vs 20-bar rolling window
        - vol_ratio:    Current volume / 20-bar rolling mean
        - obv_slope:    5-bar slope of On-Balance Volume
        - vwap_dist:    Distance from VWAP (session-level proxy)

    Args:
        df: OHLCV DataFrame with 'close' and 'volume' columns.

    Returns:
        DataFrame with volume feature columns appended.
    """
    out = df.copy()
    vol = df["volume"]
    close = df["close"]

    # --- Volume Z-score ---
    vol_mean = vol.rolling(window=20).mean()
    vol_std = vol.rolling(window=20).std()
    out["vol_zscore"] = np.where(vol_std > 0, (vol - vol_mean) / vol_std, 0.0)

    # --- Volume ratio ---
    out["vol_ratio_20"] = np.where(vol_mean > 0, vol / vol_mean, 1.0)

    # --- OBV slope ---
    direction = np.sign(close.diff())
    obv = (vol * direction).cumsum()
    out["obv_slope"] = obv.diff(5)

    # --- VWAP distance (rolling proxy) ---
    typical_price = (df["high"] + df["low"] + close) / 3
    cum_tp_vol = (typical_price * vol).rolling(window=20).sum()
    cum_vol = vol.rolling(window=20).sum()
    vwap = np.where(cum_vol > 0, cum_tp_vol / cum_vol, close)
    out["vwap_dist"] = (close - vwap) / close

    return out
