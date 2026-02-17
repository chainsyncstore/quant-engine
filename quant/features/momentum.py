"""
Momentum features.

All features are computed from closed candles only — no lookahead.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute momentum features.

    Features (6):
        - roc_1:   Rate of change, 1-bar
        - roc_3:   Rate of change, 3-bar
        - roc_5:   Rate of change, 5-bar
        - roc_10:  Rate of change, 10-bar
        - roc_20:  Rate of change, 20-bar
        - momentum_accel: Acceleration (diff of roc_5)

    Args:
        df: OHLCV DataFrame with 'close' column.

    Returns:
        DataFrame with momentum feature columns appended.
    """
    out = df.copy()
    close = df["close"]

    for period in [1, 3, 5, 10, 20]:
        out[f"roc_{period}"] = close.pct_change(periods=period)

    # Momentum acceleration = change in roc_5
    out["momentum_accel"] = out["roc_5"].diff()

    # --- Momentum Divergence ---
    # Sign disagreement between fast (5) and slow (20) momentum
    # Divergence = 1 if signs differ (reversal signal), 0 otherwise
    out["roc_divergence_5_20"] = np.where(
        np.sign(out["roc_5"]) != np.sign(out["roc_20"]),
        1.0,
        0.0
    )

    return out
