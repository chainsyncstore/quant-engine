"""
Funding rate features for crypto perpetual futures.

Funding rates are unique to crypto perps — paid every 8 hours between
longs and shorts. Positive = longs pay shorts (bullish crowding),
negative = shorts pay longs (bearish crowding). Extreme readings
historically precede reversals.
"""

from __future__ import annotations

import pandas as pd
import numpy as np


def compute(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute funding rate features.

    Requires column: funding_rate_raw (forward-filled from 8H to 1H).
    """
    out = df.copy()

    # Skip if funding rate data not available
    if "funding_rate_raw" not in out.columns:
        return out

    fr = out["funding_rate_raw"]

    # Base funding rate (already forward-filled by merge_supplementary)
    out["funding_rate"] = fr

    # Smoothed (8-bar = 8H MA, matches settlement cycle)
    out["funding_rate_ma8"] = fr.rolling(8).mean()

    # Z-score (24-bar = 1 day window)
    fr_mean = fr.rolling(24).mean()
    fr_std = fr.rolling(24).std()
    # Funding can stay constant for long stretches, which makes rolling std = 0.
    # In that case, z-score should be neutral (0.0), not NaN.
    zscore = (fr - fr_mean) / fr_std.replace(0, np.nan)
    out["funding_rate_zscore"] = zscore.fillna(0.0)

    # Extreme flag: |zscore| > 2.0
    out["funding_rate_extreme"] = (out["funding_rate_zscore"].abs() > 2.0).astype(float)

    # Cumulative 24H funding cost (sum of last 3 settlements = 24H)
    out["funding_cumulative_24h"] = fr.rolling(24).sum()

    # Funding momentum: direction of change
    out["funding_momentum"] = fr.diff(8)

    return out
