"""
Open interest features for crypto perpetual futures.

OI dynamics reveal whether new money is entering or leaving positions.
Rising OI + rising price = trend confirmation.
Rising OI + falling price = short buildup.
Falling OI = position unwinding.
"""

from __future__ import annotations

import pandas as pd
import numpy as np


def compute(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute open interest features.

    Requires column: open_interest (forward-filled to 1H).
    """
    out = df.copy()

    # Skip if open interest data not available
    if "open_interest" not in out.columns:
        out["oi_roc_1"] = 0.0
        out["oi_roc_8"] = 0.0
        out["oi_zscore"] = 0.0
        out["oi_price_divergence"] = 0.0
        out["oi_acceleration"] = 0.0
        return out

    oi = pd.to_numeric(out["open_interest"], errors="coerce")

    # Rate of change at different horizons
    out["oi_roc_1"] = oi.pct_change(1, fill_method=None)
    out["oi_roc_8"] = oi.pct_change(8, fill_method=None)

    # Z-score relative to 30-bar window
    oi_mean = oi.rolling(30).mean()
    oi_std = oi.rolling(30).std()
    out["oi_zscore"] = (oi - oi_mean) / oi_std.replace(0, np.nan)

    # OI-price divergence: sign(price_roc) * sign(oi_roc)
    # +1 = confirmation (both moving same direction)
    # -1 = divergence (moving opposite directions)
    price_roc = out["close"].pct_change(8)
    out["oi_price_divergence"] = np.sign(price_roc) * np.sign(out["oi_roc_8"])

    # OI acceleration (second derivative)
    out["oi_acceleration"] = out["oi_roc_1"].diff(1)

    oi_feature_cols = [
        "oi_roc_1",
        "oi_roc_8",
        "oi_zscore",
        "oi_price_divergence",
        "oi_acceleration",
    ]
    out[oi_feature_cols] = out[oi_feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return out
