"""
Bid-ask spread features.

Captures liquidity dynamics and regime shifts. Wide spread often precedes
volatility or indicates low liquidity; narrow spread indicates high liquidity
and efficient price discovery.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute bid/ask spread features.

    Features (4):
        - spread_zscore:        Current spread vs 20-bar rolling mean (spike = shock)
        - spread_change_5:      5-bar rate of change in spread (widening = risk off)
        - spread_regime_ratio:  Current spread / 100-bar rolling median
        - spread_to_atr:        Spread / ATR ratio (cost of trading vs opportunity)

    Args:
        df: OHLCV DataFrame with 'spread' column (and 'high', 'low', 'close').

    Returns:
        DataFrame with spread feature columns appended.
    """
    out = df.copy()

    # Fallback if 'spread' not present (e.g. synthetic data)
    if "spread" not in df.columns:
        return out

    spread = df["spread"]
    
    # --- Spread Z-score ---
    # Detects sudden liquidity shocks
    rolling_mean = spread.rolling(window=20).mean()
    rolling_std = spread.rolling(window=20).std()
    out["spread_zscore"] = np.where(
        rolling_std > 0,
        (spread - rolling_mean) / rolling_std,
        0.0
    )

    # --- Spread Change (5-bar) ---
    # Is liquidity improved or worsening?
    out["spread_change_5"] = spread.pct_change(periods=5).fillna(0)

    # --- Spread Regime Ratio ---
    # Current spread vs long-term median (structural liquidity)
    long_term_median = spread.rolling(window=100).median()
    out["spread_regime_ratio"] = np.where(
        long_term_median > 0,
        spread / long_term_median,
        1.0
    )

    # --- Spread to ATR Ratio ---
    # Cost to trade vs potential reward (volatility)
    # High ratio = expensive to trade relative to market movement
    if "high" in df.columns and "low" in df.columns and "close" in df.columns:
        tr = np.maximum(
            df["high"] - df["low"],
            np.abs(df["high"] - df["close"].shift(1)),
            np.abs(df["low"] - df["close"].shift(1))
        )
        atr = tr.rolling(window=14).mean()
        out["spread_to_atr"] = np.where(atr > 0, spread / atr, np.nan)

    return out
