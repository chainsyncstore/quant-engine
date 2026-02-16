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
    Add binary directional labels for each horizon.

    Label definition:
        label_{h}m = 1  if  close[t + h] > close[t]
        label_{h}m = 0  otherwise

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

    for h in horizons:
        future_close = out["close"].shift(-h)
        out[f"label_{h}m"] = (future_close > out["close"]).astype(int)

    # Drop rows where labels are unavailable
    out = out.iloc[: len(out) - max_horizon].copy()

    for h in horizons:
        pos_rate = out[f"label_{h}m"].mean()
        logger.info("Label %dm: %d rows, %.1f%% positive", h, len(out), pos_rate * 100)

    return out
