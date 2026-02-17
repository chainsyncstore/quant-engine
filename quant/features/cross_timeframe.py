"""
Cross-timeframe features.

Higher-timeframe context computed from 1-minute bars using longer lookback
windows. Captures macro momentum and volatility regime alignment.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute cross-timeframe features.

    Features (3):
        - roc_60:           60-bar (1-hour) rate of change
                            Captures higher-timeframe momentum direction
        - atr_ratio_60_14:  ATR(60) / ATR(14) — multi-resolution volatility
                            > 1 = macro vol expanding, < 1 = compressing
        - trend_alignment:  sign(EMA5-EMA20) * sign(EMA20-EMA50)
                            +1 = aligned trend, -1 = conflicting (chop)

    Args:
        df: OHLCV DataFrame with 'high', 'low', 'close', 'ema_5', 'ema_20', 'ema_50'.

    Returns:
        DataFrame with cross-timeframe feature columns appended.
    """
    out = df.copy()
    close = df["close"]
    high = df["high"]
    low = df["low"]
    prev_close = close.shift(1)

    # --- 60-bar rate of change ---
    out["roc_60"] = close.pct_change(periods=60)

    # --- ATR ratio (60 / 14) ---
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr_14 = tr.rolling(window=14).mean()
    atr_60 = tr.rolling(window=60).mean()
    out["atr_ratio_60_14"] = np.where(atr_14 > 0, atr_60 / atr_14, 1.0)

    # --- Trend alignment ---
    # Uses EMAs computed by trend module (ema_5, ema_20, ema_50)
    # If not present yet, compute them inline
    if "ema_5" not in df.columns:
        ema_5 = close.ewm(span=5, adjust=False).mean()
        ema_20 = close.ewm(span=20, adjust=False).mean()
        ema_50 = close.ewm(span=50, adjust=False).mean()
    else:
        ema_5 = df["ema_5"]
        ema_20 = df["ema_20"]
        ema_50 = df["ema_50"]

    short_vs_mid = np.sign(ema_5 - ema_20)
    mid_vs_long = np.sign(ema_20 - ema_50)
    out["trend_alignment"] = short_vs_mid * mid_vs_long

    # --- Distances ---
    # Safe ATR for normalization (avoid division by zero)
    safe_atr = atr_14.replace(0, np.nan)
    
    out["dist_ema_5_20"] = (ema_5 - ema_20) / safe_atr
    out["dist_ema_20_50"] = (ema_20 - ema_50) / safe_atr
    out["dist_close_ema_20"] = (close - ema_20) / safe_atr

    # --- EMA Slope (Trend Strength) ---
    # Normalized slope of long-term EMA
    out["ema_slope_50"] = ema_50.diff() / safe_atr

    # --- Mean Reversion Score ---
    # Distance from 100-bar mean (proxy via EMA_50 * 2 smoothing)
    # Z-score-like: (Close - Mean) / Std
    # Uses 50-bar EMA as 'mean' proxy and ATR as 'std' proxy
    out["mean_reversion_score"] = (close - ema_50) / safe_atr

    return out
