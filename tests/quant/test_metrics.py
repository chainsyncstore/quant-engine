"""Tests for metrics module."""

import numpy as np

from quant.validation.metrics import (
    compute_metrics,
    compute_trade_pnl,
    aggregate_fold_metrics,
    probabilistic_sharpe_ratio,
    deflated_sharpe_ratio,
    _worst_losing_streak,
)


class TestMetrics:
    def test_spread_adjusted_ev_known_values(self):
        # 4 trades: +10, +5, -3, +2 pips, spread = 1 pip each
        pnl = np.array([9, 4, -4, 1])  # after spread
        m = compute_metrics(pnl, fold=0)
        expected_ev = np.mean(pnl)
        assert abs(m.spread_adjusted_ev - expected_ev) < 1e-10

    def test_win_rate_known_values(self):
        pnl = np.array([1.0, -1.0, 1.0, 1.0])  # 3/4 wins
        m = compute_metrics(pnl)
        assert abs(m.win_rate - 0.75) < 1e-10

    def test_sharpe_calculation(self):
        pnl = np.array([0.01, 0.02, 0.01, 0.03])
        m = compute_metrics(pnl)
        expected_sharpe = np.mean(pnl) / np.std(pnl)
        assert abs(m.sharpe - expected_sharpe) < 1e-10

    def test_worst_losing_streak(self):
        pnl = np.array([1, -1, -1, -1, 1, -1, -1])
        assert _worst_losing_streak(pnl) == 3

    def test_max_drawdown(self):
        pnl = np.array([1.0, 1.0, -3.0, 1.0])
        m = compute_metrics(pnl)
        # Cumulative: 1, 2, -1, 0. Peak: 1, 2, 2, 2. DD: 0, 0, -3, -2
        assert m.max_drawdown == -3.0

    def test_empty_pnl(self):
        m = compute_metrics(np.array([]))
        assert m.n_trades == 0
        assert m.spread_adjusted_ev == 0.0

    def test_trade_pnl_with_threshold(self):
        predictions = np.array([0.4, 0.6, 0.7, 0.3])
        actuals = np.array([0, 1, 1, 0])
        price_moves = np.array([-0.001, 0.002, 0.001, -0.003])
        spread = 0.00008

        pnl = compute_trade_pnl(predictions, actuals, price_moves, threshold=0.5, spread=spread)
        assert len(pnl) == 2  # Only indices 1, 2 pass threshold
        np.testing.assert_allclose(pnl[0], 0.002 - 0.00008)

    def test_trade_pnl_bidirectional_threshold(self):
        predictions = np.array([0.9, 0.2, 0.7, 0.1])
        actuals = np.array([1, 0, 1, 0])
        price_moves = np.array([0.003, -0.002, 0.001, -0.004])
        spread = 0.0001

        pnl = compute_trade_pnl(
            predictions,
            actuals,
            price_moves,
            threshold=0.7,
            spread=spread,
            allow_short=True,
        )
        assert len(pnl) == 4
        assert float(np.mean(pnl)) > 0

    def test_probabilistic_and_deflated_sharpe_ranges(self):
        pnl = np.array([0.02, 0.01, 0.015, -0.001, 0.03, 0.005, 0.01, 0.025])
        psr = probabilistic_sharpe_ratio(pnl)
        dsr = deflated_sharpe_ratio(pnl, n_trials=20)

        assert 0.0 <= psr <= 1.0
        assert 0.0 <= dsr <= 1.0
        assert dsr <= psr

    def test_aggregate_fold_metrics(self):
        m1 = compute_metrics(np.array([0.01, 0.02]), fold=0)
        m2 = compute_metrics(np.array([-0.01, 0.03, 0.01]), fold=1)
        agg = aggregate_fold_metrics([m1, m2])
        assert agg["n_trades"] == 5
        assert agg["spread_adjusted_ev"] != 0.0
