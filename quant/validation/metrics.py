"""
Performance metrics for directional prediction.

All metrics account for spread when evaluating PnL.
"""

from __future__ import annotations

import math
import logging
from statistics import NormalDist
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
_NORM = NormalDist()
_EULER_GAMMA = 0.5772156649


@dataclass
class FoldMetrics:
    """Metrics for a single walk-forward fold."""

    fold: int
    train_start: str
    test_start: str
    test_end: str
    spread_adjusted_ev: float
    win_rate: float
    n_trades: int
    sharpe: float
    max_drawdown: float
    worst_losing_streak: int


@dataclass
class RegimeMetrics:
    """Metrics for a single regime."""

    regime: int
    ev: float
    win_rate: float
    n_trades: int
    optimal_threshold: float


@dataclass
class HorizonReport:
    """Complete metrics report for one horizon."""

    horizon: int
    overall_ev: float
    overall_win_rate: float
    overall_sharpe: float
    overall_max_drawdown: float
    overall_n_trades: int
    overall_worst_streak: int
    per_regime: List[RegimeMetrics]
    per_fold: List[FoldMetrics]


def compute_trade_pnl(
    predictions: np.ndarray,
    actuals: np.ndarray,
    price_moves: np.ndarray,
    threshold: float,
    spread: float | np.ndarray,
    allow_short: bool = False,
) -> np.ndarray:
    """
    Compute per-trade PnL for thresholded directional trades.

    Args:
        predictions: Calibrated P(up) for each bar.
        actuals: Actual labels (0/1).
        price_moves: Actual price moves (close[t+h] - close[t]).
        threshold: Minimum probability to take a trade.
        spread: Spread in price units (e.g. 0.00008).
        allow_short: When True, take SELL trades for P(up) <= (1-threshold).

    Returns:
        Array of PnL values for each traded bar (empty if no trades).
    """
    if isinstance(spread, np.ndarray) and len(spread) != len(predictions):
        raise ValueError("Spread array length must match predictions length")

    long_mask = predictions >= threshold
    if allow_short:
        short_mask = predictions <= (1.0 - threshold)
        trade_mask = long_mask | short_mask
    else:
        trade_mask = long_mask

    if not trade_mask.any():
        return np.array([])

    traded_moves = price_moves[trade_mask]
    if allow_short:
        trade_dirs = np.where(long_mask[trade_mask], 1.0, -1.0)
        gross_pnl = traded_moves * trade_dirs
    else:
        gross_pnl = traded_moves

    if isinstance(spread, np.ndarray):
        traded_spread = spread[trade_mask]
        pnl = gross_pnl - traded_spread
    else:
        pnl = gross_pnl - spread

    return pnl


def compute_metrics(
    pnl: np.ndarray,
    fold: int = 0,
    train_start: str = "",
    test_start: str = "",
    test_end: str = "",
) -> FoldMetrics:
    """
    Compute standard metrics from a PnL array.

    Args:
        pnl: Array of per-trade PnL values.
        fold: Fold index.
        train_start: Training window start timestamp.
        test_start: Test window start timestamp.
        test_end: Test window end timestamp.

    Returns:
        FoldMetrics dataclass.
    """
    n = len(pnl)

    if n == 0:
        return FoldMetrics(
            fold=fold,
            train_start=train_start,
            test_start=test_start,
            test_end=test_end,
            spread_adjusted_ev=0.0,
            win_rate=0.0,
            n_trades=0,
            sharpe=0.0,
            max_drawdown=0.0,
            worst_losing_streak=0,
        )

    ev = float(np.mean(pnl))
    win_rate = float((pnl > 0).sum() / n)
    sharpe = float(np.mean(pnl) / np.std(pnl)) if np.std(pnl) > 0 else 0.0

    # Max drawdown from cumulative PnL
    cum_pnl = np.cumsum(pnl)
    peak = np.maximum.accumulate(cum_pnl)
    drawdown = cum_pnl - peak
    max_dd = float(drawdown.min()) if len(drawdown) > 0 else 0.0

    # Worst losing streak
    worst_streak = _worst_losing_streak(pnl)

    return FoldMetrics(
        fold=fold,
        train_start=train_start,
        test_start=test_start,
        test_end=test_end,
        spread_adjusted_ev=ev,
        win_rate=win_rate,
        n_trades=n,
        sharpe=sharpe,
        max_drawdown=max_dd,
        worst_losing_streak=worst_streak,
    )


def aggregate_fold_metrics(folds: List[FoldMetrics]) -> dict:
    """Aggregate metrics across all folds.

    Returns:
        Dict with overall metrics.
    """
    if not folds:
        return {
            "spread_adjusted_ev": 0.0,
            "win_rate": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
            "n_trades": 0,
            "worst_losing_streak": 0,
        }

    # Weighted by trade count
    total_trades = sum(f.n_trades for f in folds)
    if total_trades == 0:
        return {
            "spread_adjusted_ev": 0.0,
            "win_rate": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
            "n_trades": 0,
            "worst_losing_streak": 0,
        }

    weighted_ev = sum(f.spread_adjusted_ev * f.n_trades for f in folds) / total_trades
    weighted_wr = sum(f.win_rate * f.n_trades for f in folds) / total_trades

    # Sharpe: average across folds (not weighted — each fold is independent)
    avg_sharpe = float(np.mean([f.sharpe for f in folds if f.n_trades > 0]))

    return {
        "spread_adjusted_ev": weighted_ev,
        "win_rate": weighted_wr,
        "sharpe": avg_sharpe,
        "max_drawdown": min(f.max_drawdown for f in folds),
        "n_trades": total_trades,
        "worst_losing_streak": max(f.worst_losing_streak for f in folds),
    }


def probabilistic_sharpe_ratio(
    pnl: np.ndarray,
    benchmark_sharpe: float = 0.0,
) -> float:
    """
    Compute Probabilistic Sharpe Ratio (PSR).

    PSR estimates P(SR_true > benchmark_sharpe), adjusting for skew/kurtosis.
    """
    n = len(pnl)
    if n < 2:
        return 0.5

    std = float(np.std(pnl))
    if std <= 0:
        return 0.5

    sharpe = float(np.mean(pnl) / std)
    standardized = (pnl - np.mean(pnl)) / std
    skew = float(np.mean(standardized ** 3))
    kurt = float(np.mean(standardized ** 4))

    denom_sq = 1.0 - (skew * sharpe) + (((kurt - 1.0) / 4.0) * sharpe * sharpe)
    denom = math.sqrt(max(1e-12, denom_sq))
    z = ((sharpe - benchmark_sharpe) * math.sqrt(n - 1)) / denom
    return float(_NORM.cdf(z))


def deflated_sharpe_ratio(
    pnl: np.ndarray,
    n_trials: int,
) -> float:
    """
    Compute Deflated Sharpe Ratio (DSR) using a multiple-testing-adjusted benchmark.

    n_trials should approximate the number of strategy/threshold variants explored.
    """
    n_trials = max(int(n_trials), 1)
    if n_trials <= 1:
        return probabilistic_sharpe_ratio(pnl, benchmark_sharpe=0.0)

    n = len(pnl)
    if n < 2:
        return 0.5

    std = float(np.std(pnl))
    if std <= 0:
        return 0.5

    sharpe = float(np.mean(pnl) / std)
    standardized = (pnl - np.mean(pnl)) / std
    skew = float(np.mean(standardized ** 3))
    kurt = float(np.mean(standardized ** 4))

    sr_var = (1.0 - (skew * sharpe) + (((kurt - 1.0) / 4.0) * sharpe * sharpe)) / max(n - 1, 1)
    sr_std = math.sqrt(max(1e-12, sr_var))

    p1 = min(max(1.0 - 1.0 / n_trials, 1e-6), 1.0 - 1e-6)
    p2 = min(max(1.0 - 1.0 / (n_trials * math.e), 1e-6), 1.0 - 1e-6)
    z1 = _NORM.inv_cdf(p1)
    z2 = _NORM.inv_cdf(p2)
    sharpe_star = sr_std * ((1.0 - _EULER_GAMMA) * z1 + (_EULER_GAMMA * z2))

    return probabilistic_sharpe_ratio(pnl, benchmark_sharpe=sharpe_star)


def _worst_losing_streak(pnl: np.ndarray) -> int:
    """Find the longest consecutive losing streak."""
    max_streak = 0
    current = 0
    for p in pnl:
        if p <= 0:
            current += 1
            max_streak = max(max_streak, current)
        else:
            current = 0
    return max_streak
