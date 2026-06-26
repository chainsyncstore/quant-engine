"""Cross-sectional and market-context feature enrichment for multi-symbol datasets."""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_cross_sectional_features(df: pd.DataFrame) -> pd.DataFrame:
    """Append cross-sectional features over timestamp groups.

    Expected input index is MultiIndex [timestamp, symbol] with at least
    `close` and `volume` columns.
    """

    if not isinstance(df.index, pd.MultiIndex) or list(df.index.names) != ["timestamp", "symbol"]:
        raise ValueError("df must be MultiIndex with levels ['timestamp', 'symbol']")
    if "close" not in df.columns or "volume" not in df.columns:
        raise ValueError("df must contain 'close' and 'volume' columns")

    out = df.copy()

    ret_1h = out.groupby(level="symbol")["close"].pct_change()
    ret_1h = ret_1h.replace([np.inf, -np.inf], np.nan)

    ts_level = out.index.get_level_values("timestamp")

    ret_mean = ret_1h.groupby(ts_level).transform("mean")
    ret_std = ret_1h.groupby(ts_level).transform(lambda values: float(values.std(ddof=0)))
    ret_std = ret_std.replace(0.0, np.nan)

    vol_log = np.log1p(out["volume"].clip(lower=0.0))
    vol_mean = vol_log.groupby(ts_level).transform("mean")
    vol_std = vol_log.groupby(ts_level).transform(lambda values: float(values.std(ddof=0)))
    vol_std = vol_std.replace(0.0, np.nan)

    out["xs_ret_1h_z"] = ((ret_1h - ret_mean) / ret_std).replace([np.inf, -np.inf], np.nan)
    out["xs_ret_1h_rank"] = ret_1h.groupby(ts_level).rank(method="average", pct=True)

    out["xs_volume_z"] = ((vol_log - vol_mean) / vol_std).replace([np.inf, -np.inf], np.nan)
    zero_vol_spread = vol_mean.notna() & vol_std.isna()
    out.loc[zero_vol_spread, "xs_volume_z"] = 0.0
    out["xs_volume_rank"] = vol_log.groupby(ts_level).rank(method="average", pct=True)

    out["xs_dispersion_ret_1h"] = ret_1h.groupby(ts_level).transform(
        lambda values: float(values.std(ddof=0)) if values.notna().any() else np.nan
    )
    out["xs_breadth_up_ret_1h"] = ret_1h.groupby(ts_level).transform(
        lambda values: float((values > 0.0).mean()) if values.notna().any() else np.nan
    )

    return out
