"""Tests for Monte Carlo simulation."""

import numpy as np

from quant.risk.monte_carlo import simulate, _worst_streak


class TestMonteCarlo:
    def test_simulation_produces_results(self):
        pnl = np.random.default_rng(42).normal(0.0001, 0.001, size=200)
        result = simulate(pnl, n_trades=100)

        assert result.n_simulations == 10_000
        assert result.n_trades_per_sim == 100
        assert 0.0 <= result.ruin_probability <= 1.0
        assert result.ev_ci_95[0] <= result.ev_ci_95[1]

    def test_percentile_ordering(self):
        pnl = np.random.default_rng(42).normal(0.0001, 0.001, size=200)
        result = simulate(pnl, n_trades=100)
        assert result.p5_final_pnl <= result.median_final_pnl <= result.p95_final_pnl

    def test_empty_pnl_gives_ruin(self):
        result = simulate(np.array([]), n_trades=50)
        assert result.ruin_probability == 1.0

    def test_worst_streak(self):
        pnl = np.array([1, -1, -1, -1, 1, -1])
        assert _worst_streak(pnl) == 3

    def test_all_positive_low_ruin(self):
        # Deterministic positive PnL should have very low ruin
        pnl = np.ones(100) * 0.001
        result = simulate(pnl, n_trades=50)
        assert result.ruin_probability == 0.0
