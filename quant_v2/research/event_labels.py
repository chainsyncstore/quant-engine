"""Event-aware label filtering for v2 research datasets."""

from __future__ import annotations

import numpy as np
import pandas as pd


def apply_event_aware_label_filters(
    df: pd.DataFrame,
    *,
    horizons: list[int] | tuple[int, ...],
    funding_abs_threshold: float = 0.0015,
    volatility_shock_quantile: float = 0.98,
) -> pd.DataFrame:
    """Mask labels during funding/volatility shock windows with -1."""

    out = df.copy()
    if out.empty:
        return out

    event_mask = pd.Series(False, index=out.index)
    has_symbol_level = isinstance(out.index, pd.MultiIndex) and "symbol" in out.index.names

    funding_mask = pd.Series(False, index=out.index)
    if "funding_rate_raw" in out.columns:
        funding_series = out["funding_rate_raw"].astype(float)
        funding_mask = funding_series.abs() >= float(funding_abs_threshold)
        event_mask = event_mask | funding_mask

    vol_mask = pd.Series(False, index=out.index)
    if "close" in out.columns:
        if has_symbol_level:
            abs_ret = out.groupby(level="symbol")["close"].pct_change().abs()
        else:
            abs_ret = out["close"].pct_change().abs()
        if abs_ret.notna().any():
            threshold = float(abs_ret.quantile(float(volatility_shock_quantile)))
            if np.isfinite(threshold) and threshold > 0.0:
                vol_mask = abs_ret >= threshold
                event_mask = event_mask | vol_mask.fillna(False)

    # Include one bar after an event to avoid immediate leakage from shock unwind.
    if event_mask.any():
        if has_symbol_level:
            shifted = event_mask.groupby(level="symbol").shift(1, fill_value=False)
        else:
            shifted = event_mask.shift(1, fill_value=False)
        event_mask = event_mask | shifted

    out["event_funding_window_flag"] = funding_mask.astype(float)
    out["event_volatility_shock_flag"] = vol_mask.fillna(False).astype(float)
    out["event_exclusion_flag"] = event_mask.astype(float)

    for horizon in horizons:
        label_col = f"label_{int(horizon)}m"
        if label_col in out.columns:
            out.loc[event_mask, label_col] = -1

    return out
