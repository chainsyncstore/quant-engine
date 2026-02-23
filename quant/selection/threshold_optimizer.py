"""
Per-regime probability threshold optimizer.

Sweeps thresholds to find the one that maximizes spread-adjusted EV.
"""

from __future__ import annotations

import logging
from typing import Tuple

import numpy as np

logger = logging.getLogger(__name__)


def optimize_threshold(
    predictions: np.ndarray,
    price_moves: np.ndarray,
    spread: float | np.ndarray,
    threshold_min: float = 0.50,
    threshold_max: float = 0.80,
    threshold_step: float = 0.05,
    allow_short: bool = False,
) -> Tuple[float, float]:
    """
    Find the probability threshold that maximizes spread-adjusted EV.

    Args:
        predictions: Calibrated P(up) for each bar.
        price_moves: Actual price moves.
        spread: Spread in price units (float or array).
        threshold_min: Lower bound of sweep.
        threshold_max: Upper bound of sweep.
        threshold_step: Step size.
        allow_short: When True, optimize bidirectional long/short EV.

    Returns:
        (best_threshold, best_ev) — best threshold and its EV.
        If no threshold yields positive EV, returns (threshold_max, best_ev).
    """
    best_threshold = threshold_max
    best_ev = -np.inf
    best_n = 0

    if isinstance(spread, np.ndarray) and len(spread) != len(predictions):
        raise ValueError("Spread array length must match predictions length")

    thresholds = np.round(np.arange(threshold_min, threshold_max + threshold_step / 2, threshold_step), 2)

    for t in thresholds:
        long_mask = predictions >= t
        if allow_short:
            short_mask = predictions <= (1.0 - t)
            mask = long_mask | short_mask
        else:
            mask = long_mask

        n_trades = int(mask.sum())

        if n_trades < 10:  # minimum trade count for reliability
            continue

        traded_moves = price_moves[mask]
        if allow_short:
            directions = np.where(long_mask[mask], 1.0, -1.0)
            gross_pnl = traded_moves * directions
        else:
            gross_pnl = traded_moves

        if isinstance(spread, np.ndarray):
            pnl = gross_pnl - spread[mask]
        else:
            pnl = gross_pnl - spread
        ev = float(np.mean(pnl))

        if ev > best_ev:
            best_ev = ev
            best_threshold = float(t)
            best_n = int(n_trades)

    logger.debug(
        "Threshold optimization: best=%.2f, EV=%.6f, trades=%d",
        best_threshold,
        best_ev,
        best_n,
    )

    return best_threshold, best_ev if best_ev > -np.inf else 0.0
