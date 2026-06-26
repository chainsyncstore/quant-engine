"""
Liquidation cascade detection features.

Inferred from price/volume signatures since Binance doesn't provide
a free liquidation history endpoint. A liquidation cascade is identified
when there's a large price move (|roc| > 2 * ATR) coinciding with
abnormally high volume (vol_ratio > 3).
"""

from __future__ import annotations

import pandas as pd
import numpy as np


def compute(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute liquidation cascade proxy features.

    Requires columns: close, high, low, volume.
    """
    out = df.copy()

    # ATR for threshold (14-bar)
    high_low = out["high"] - out["low"]
    high_close = (out["high"] - out["close"].shift(1)).abs()
    low_close = (out["low"] - out["close"].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()

    # Price move magnitude
    price_move = (out["close"] - out["close"].shift(1)).abs()

    # Volume ratio (current bar vs 20-bar MA)
    vol_ma = out["volume"].rolling(20).mean()
    vol_ratio = out["volume"] / vol_ma.replace(0, np.nan)

    # Liquidation candle: big move + high volume
    out["liquidation_candle"] = (
        (price_move > 2.0 * atr) & (vol_ratio > 3.0)
    ).astype(float)

    # Directional liquidation pressure (last 8 bars)
    up_candle = out["liquidation_candle"] * (out["close"] > out["close"].shift(1)).astype(float)
    down_candle = out["liquidation_candle"] * (out["close"] < out["close"].shift(1)).astype(float)
    out["liquidation_up_pressure"] = up_candle.rolling(8).sum()
    out["liquidation_down_pressure"] = down_candle.rolling(8).sum()

    # Post-liquidation flag: 1-4 bars after a cascade (mean reversion window)
    liq_shifted = pd.concat([
        out["liquidation_candle"].shift(i) for i in range(1, 5)
    ], axis=1)
    out["post_liquidation_flag"] = liq_shifted.max(axis=1)
    out["liquidations_long_vol"] = 0.0
    out["liquidations_short_vol"] = 0.0

    return out
