"""
Cyclical time encoding features.

Encodes time-of-day and day-of-week as sin/cos pairs to preserve
cyclical continuity (23:59 is close to 00:00).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute cyclical time features.

    Features (4):
        - hour_sin: sin(2π * hour/24)
        - hour_cos: cos(2π * hour/24)
        - dow_sin:  sin(2π * dayofweek/5)  (5 trading days)
        - dow_cos:  cos(2π * dayofweek/5)

    Args:
        df: DataFrame with UTC DatetimeIndex.

    Returns:
        DataFrame with time encoding feature columns appended.
    """
    out = df.copy()
    idx = df.index

    # Use fractional hour for finer resolution
    frac_hour = idx.hour + idx.minute / 60.0
    out["hour_sin"] = np.sin(2 * np.pi * frac_hour / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * frac_hour / 24.0)

    # Day of week (Mon=0 .. Fri=4)
    dow = idx.weekday
    out["dow_sin"] = np.sin(2 * np.pi * dow / 5.0)
    out["dow_cos"] = np.cos(2 * np.pi * dow / 5.0)

    return out
