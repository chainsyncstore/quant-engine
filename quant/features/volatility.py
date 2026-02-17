"""
Volatility features.

All features are computed from closed candles only — no lookahead.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute volatility features.

    Features (6):
        - atr_14:          Average True Range (14 bars)
        - rolling_std_10:  Rolling std of returns (10 bars)
        - rolling_std_20:  Rolling std of returns (20 bars)
        - bb_pct_b:        Bollinger Band %B (20-bar, 2-std)
        - vol_ratio:       Volatility ratio (std_10 / std_20)
        - parkinson_vol:   Parkinson volatility estimator (20 bars)

    Args:
        df: OHLCV DataFrame with 'high', 'low', 'close' columns.

    Returns:
        DataFrame with volatility feature columns appended.
    """
    out = df.copy()
    close = df["close"]
    high = df["high"]
    low = df["low"]
    prev_close = close.shift(1)

    # --- ATR(14) ---
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    out["atr_14"] = tr.rolling(window=14).mean()

    # --- Rolling std of returns ---
    returns = close.pct_change()
    out["rolling_std_10"] = returns.rolling(window=10).std()
    out["rolling_std_20"] = returns.rolling(window=20).std()

    # --- Bollinger Band %B ---
    sma_20 = close.rolling(window=20).mean()
    std_20 = close.rolling(window=20).std()
    upper_band = sma_20 + 2 * std_20
    lower_band = sma_20 - 2 * std_20
    band_width = upper_band - lower_band
    out["bb_pct_b"] = np.where(band_width > 0, (close - lower_band) / band_width, 0.5)

    # --- Volatility ratio (short / long) ---
    out["vol_ratio"] = np.where(
        out["rolling_std_20"] > 0,
        out["rolling_std_10"] / out["rolling_std_20"],
        1.0,
    )

    # --- Parkinson volatility (20-bar) ---
    log_hl = np.log(high / low)
    parkinson = log_hl ** 2 / (4 * np.log(2))
    out["parkinson_vol"] = parkinson.rolling(window=20).mean().apply(np.sqrt)

    # --- Realized Volatility (5-bar) ---
    # Standard deviation of returns over 5 bars
    out["realized_vol_5"] = returns.rolling(window=5).std()

    # --- Volatility of Volatility ---
    # Rolling standard deviation of ATR (instability of volatility)
    out["vol_of_vol"] = out["atr_14"].rolling(window=20).std() / out["atr_14"]

    # --- Garman-Klass Volatility ---
    # Efficient estimator using OHLC
    # sigma^2 = 0.5 * (H-L)^2 - (2*ln(2)-1) * (C-O)^2
    log_hl = (np.log(df["high"] / df["low"])) ** 2
    log_co = (np.log(df["close"] / df["open"])) ** 2
    gk_var = 0.5 * log_hl - (2 * np.log(2) - 1) * log_co
    out["garman_klass_vol"] = np.sqrt(gk_var)

    return out
