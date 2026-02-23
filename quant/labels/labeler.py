"""
Binary directional labeling engine.

Labels are simple: did price go up within the horizon?
No pip thresholds, no spread adjustment at label time.
Spread is accounted for during evaluation.
"""

from __future__ import annotations

import logging
from typing import List

import pandas as pd

logger = logging.getLogger(__name__)


def add_labels(df: pd.DataFrame, horizons: List[int] | None = None) -> pd.DataFrame:
    """
    Add ternary directional labels for each horizon.

    Label definition (with dead zone):
        move = close[t + h] - close[t]
        label_{h}m =  1  if  move >  dead_zone   (UP — tradeable long)
        label_{h}m =  0  if  move < -dead_zone   (DOWN — tradeable short)
        label_{h}m = -1  if  |move| <= dead_zone  (FLAT — excluded from training)

    The dead zone eliminates sub-spread moves that are untradeable noise.

    Rows where the forward label cannot be computed (tail) are dropped.

    Args:
        df: OHLCV DataFrame with 'close' column.
        horizons: List of forward horizons in bars (default: [3, 5]).

    Returns:
        DataFrame with label columns added, tail rows dropped.
    """
    from quant.config import get_research_config

    cfg = get_research_config()
    horizons = horizons or cfg.horizons

    out = df.copy()
    max_horizon = max(horizons)

    # Crypto-only dynamic dead zone: percentage of close price per bar
    dead_zone_series = out["close"] * cfg.dead_zone_pct

    for h in horizons:
        future_close = out["close"].shift(-h)
        move = future_close - out["close"]

        # Ternary label: 1=UP, 0=DOWN, -1=FLAT (dead zone)
        label = pd.Series(-1, index=out.index, dtype=int)
        label[move > dead_zone_series] = 1
        label[move < -dead_zone_series] = 0
        out[f"label_{h}m"] = label

    # Drop rows where labels are unavailable (tail)
    out = out.iloc[: len(out) - max_horizon].copy()

    for h in horizons:
        labels = out[f"label_{h}m"]
        n_up = (labels == 1).sum()
        n_down = (labels == 0).sum()
        n_flat = (labels == -1).sum()
        total = len(labels)
        dz_info = f"dead_zone_pct={cfg.dead_zone_pct}"
        logger.info(
            "Label %dh: %d rows — UP=%.1f%%, DOWN=%.1f%%, FLAT=%.1f%% (%s)",
            h, total,
            n_up / total * 100,
            n_down / total * 100,
            n_flat / total * 100,
            dz_info,
        )

    return out
