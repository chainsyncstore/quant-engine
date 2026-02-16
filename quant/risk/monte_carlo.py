"""
Monte Carlo simulation for risk assessment.

Uses empirical PnL distribution from walk-forward to model:
    - Ruin probability
    - Equity curve distributions
    - Streak behaviour
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from quant.config import get_research_config

logger = logging.getLogger(__name__)


@dataclass
class MonteCarloResult:
    """Results from Monte Carlo simulation."""

    n_simulations: int
    n_trades_per_sim: int
    ruin_probability: float  # fraction of paths hitting ruin
    ev_ci_95: tuple  # (lower, upper) 95% CI of mean PnL
    median_final_pnl: float
    p5_final_pnl: float
    p95_final_pnl: float
    worst_streak_p50: int
    worst_streak_p95: int


def simulate(
    pnl_distribution: np.ndarray,
    n_trades: int | None = None,
    ruin_threshold: float = -0.05,
    flat_stake: float = 1.0,
) -> MonteCarloResult:
    """
    Run Monte Carlo simulation on empirical PnL distribution.

    Args:
        pnl_distribution: Array of observed per-trade PnL values.
        n_trades: Number of trades per simulation path
                  (default: same as observed count).
        ruin_threshold: Cumulative PnL threshold for ruin
                        (fraction of initial equity, e.g. -0.05 = -5%).
        flat_stake: Fixed stake per trade (no compounding).

    Returns:
        MonteCarloResult with simulation statistics.
    """
    cfg = get_research_config()
    n_sims = cfg.mc_n_simulations
    n_trades = n_trades or len(pnl_distribution)

    if len(pnl_distribution) == 0:
        logger.warning("Empty PnL distribution — returning zero results")
        return MonteCarloResult(
            n_simulations=n_sims,
            n_trades_per_sim=n_trades,
            ruin_probability=1.0,
            ev_ci_95=(0.0, 0.0),
            median_final_pnl=0.0,
            p5_final_pnl=0.0,
            p95_final_pnl=0.0,
            worst_streak_p50=0,
            worst_streak_p95=0,
        )

    rng = np.random.default_rng(seed=42)

    # Sample trade indices with replacement
    sampled_indices = rng.choice(len(pnl_distribution), size=(n_sims, n_trades), replace=True)
    sampled_pnl = pnl_distribution[sampled_indices] * flat_stake

    # Cumulative PnL per path
    cum_pnl = np.cumsum(sampled_pnl, axis=1)

    # Final PnL
    final_pnl = cum_pnl[:, -1]

    # Ruin: did cumulative PnL ever hit ruin_threshold?
    min_cum_pnl = np.min(cum_pnl, axis=1)
    ruin_count = (min_cum_pnl <= ruin_threshold).sum()
    ruin_prob = float(ruin_count / n_sims)

    # EV confidence interval (95%)
    path_means = np.mean(sampled_pnl, axis=1)
    ev_lower = float(np.percentile(path_means, 2.5))
    ev_upper = float(np.percentile(path_means, 97.5))

    # Worst losing streaks per path
    worst_streaks = np.array([_worst_streak(sampled_pnl[i]) for i in range(n_sims)])

    result = MonteCarloResult(
        n_simulations=n_sims,
        n_trades_per_sim=n_trades,
        ruin_probability=ruin_prob,
        ev_ci_95=(ev_lower, ev_upper),
        median_final_pnl=float(np.median(final_pnl)),
        p5_final_pnl=float(np.percentile(final_pnl, 5)),
        p95_final_pnl=float(np.percentile(final_pnl, 95)),
        worst_streak_p50=int(np.median(worst_streaks)),
        worst_streak_p95=int(np.percentile(worst_streaks, 95)),
    )

    logger.info(
        "Monte Carlo (%d sims, %d trades): ruin=%.2f%%, median_pnl=%.4f, EV_CI=[%.6f, %.6f]",
        n_sims,
        n_trades,
        ruin_prob * 100,
        result.median_final_pnl,
        ev_lower,
        ev_upper,
    )

    return result


def _worst_streak(pnl: np.ndarray) -> int:
    """Find worst losing streak in a PnL array."""
    max_streak = 0
    current = 0
    for p in pnl:
        if p <= 0:
            current += 1
            if current > max_streak:
                max_streak = current
        else:
            current = 0
    return max_streak
