"""Regime-context feature layer for v2 multi-symbol datasets."""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_regime_context_features(df: pd.DataFrame) -> pd.DataFrame:
    """Append market-wide stress/volatility context features by timestamp."""

    if not isinstance(df.index, pd.MultiIndex) or list(df.index.names) != ["timestamp", "symbol"]:
        raise ValueError("df must be MultiIndex with levels ['timestamp', 'symbol']")
    if "close" not in df.columns or "volume" not in df.columns:
        raise ValueError("df must contain 'close' and 'volume' columns")

    out = df.copy()
    ts = out.index.get_level_values("timestamp")

    ret_1h = out.groupby(level="symbol")["close"].pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    mkt_ret_mean = ret_1h.groupby(ts).transform("mean")
    mkt_ret_std = ret_1h.groupby(ts).transform(lambda values: float(values.std(ddof=0))).fillna(0.0)
    mkt_ret_abs = ret_1h.abs().groupby(ts).transform("mean")

    log_vol = np.log1p(out["volume"].clip(lower=0.0))
    mkt_log_vol = log_vol.groupby(ts).transform("mean")

    # Rolling market context on per-timestamp aggregates.
    grouped = pd.DataFrame(
        {
            "mkt_ret_mean": mkt_ret_mean.groupby(ts).first(),
            "mkt_ret_std": mkt_ret_std.groupby(ts).first(),
            "mkt_ret_abs": mkt_ret_abs.groupby(ts).first(),
            "mkt_log_vol": mkt_log_vol.groupby(ts).first(),
        }
    ).sort_index()

    grouped["regime_trend_24h"] = grouped["mkt_ret_mean"].rolling(24, min_periods=3).mean().fillna(0.0)
    grouped["regime_volatility_24h"] = grouped["mkt_ret_std"].rolling(24, min_periods=3).mean().fillna(0.0)
    grouped["regime_stress_24h"] = grouped["mkt_ret_abs"].rolling(24, min_periods=3).mean().fillna(0.0)

    vol_threshold = grouped["regime_volatility_24h"].rolling(96, min_periods=10).quantile(0.80)
    grouped["regime_high_vol_flag"] = (
        grouped["regime_volatility_24h"] >= vol_threshold.fillna(grouped["regime_volatility_24h"])
    ).astype(float)

    merge_cols = grouped[
        [
            "regime_trend_24h",
            "regime_volatility_24h",
            "regime_stress_24h",
            "regime_high_vol_flag",
        ]
    ]

    out = out.join(merge_cols, on="timestamp")
    out[
        [
            "regime_trend_24h",
            "regime_volatility_24h",
            "regime_stress_24h",
            "regime_high_vol_flag",
        ]
    ] = out[
        [
            "regime_trend_24h",
            "regime_volatility_24h",
            "regime_stress_24h",
            "regime_high_vol_flag",
        ]
    ].fillna(0.0)

    return out
