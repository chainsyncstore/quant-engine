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


def _winsorised_zscore(
    series: pd.Series,
    window: int = 8,
    trim: float = 0.05,
) -> pd.Series:
    """Rolling z-score using winsorised (trimmed) mean and std.

    Trims *trim* fraction from each tail before computing rolling
    statistics, making the z-score robust to outliers in funding-rate
    distributions.
    """
    min_periods = max(3, window // 2)

    def _zscore_of_window(arr: np.ndarray) -> float:
        valid = arr[~np.isnan(arr)]
        if len(valid) < 3:
            return np.nan
        sorted_v = np.sort(valid)
        k = max(1, int(len(sorted_v) * trim))
        if 2 * k >= len(sorted_v):
            k = 0
        trimmed = sorted_v[k : len(sorted_v) - k] if k > 0 else sorted_v
        mu = float(trimmed.mean())
        sigma = float(trimmed.std(ddof=1))
        if sigma <= 0.0:
            return 0.0
        return (float(arr[-1]) - mu) / sigma

    return series.rolling(window, min_periods=min_periods).apply(
        _zscore_of_window,
        raw=True,
    )


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

    # Winsorised z-score (8-bar = 8H rolling window, trim 5% tails)
    zscore = _winsorised_zscore(fr, window=8, trim=0.05)
    # Interpolate interior NaN gaps, then fill any trailing edge with 0
    zscore = zscore.interpolate(method="linear").fillna(0.0)
    out["funding_rate_zscore"] = zscore

    # Extreme flag: |zscore| > 2.0
    out["funding_rate_extreme"] = (out["funding_rate_zscore"].abs() > 2.0).astype(float)

    # Cumulative 24H funding cost (sum of last 3 settlements = 24H)
    out["funding_cumulative_24h"] = fr.rolling(24).sum()

    # Funding momentum: direction of change
    out["funding_momentum"] = fr.diff(8)

    return out
