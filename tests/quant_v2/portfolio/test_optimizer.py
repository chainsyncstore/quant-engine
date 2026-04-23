"""Tests for quant_v2 portfolio optimizer."""

import numpy as np
import pandas as pd
import pytest

from quant_v2.portfolio.optimizer import RiskParityOptimizer


class TestMinNotionalFilter:
    """Tests for the min-notional floor behavior."""

    def test_dampened_two_symbol_allocation_survives_min_notional(self) -> None:
        """
        Verify that dampened allocations (0.30x Kelly split across 2 symbols)
        survive the min-notional filter at ~$10k equity with the 0.5% floor.
        """
        optimizer = RiskParityOptimizer()

        rng = np.random.default_rng(42)
        base_prices = 100 + np.arange(100) * 0.01

        price_histories = {
            "FOOUSDT": pd.Series(base_prices + rng.standard_normal(100) * 0.05),
            "BARUSDT": pd.Series(base_prices + rng.standard_normal(100) * 0.05),
        }

        target_exposures = {"FOOUSDT": 0.018, "BARUSDT": 0.018}
        equity_usd = 10_000.0

        result = optimizer.optimize(target_exposures, price_histories, equity_usd)

        assert "FOOUSDT" in result.weights, "FOOUSDT should survive min-notional filter"
        assert "BARUSDT" in result.weights, "BARUSDT should survive min-notional filter"

        for sym, w in result.weights.items():
            notional = abs(w) * equity_usd
            assert notional >= 49.0, f"{sym} notional {notional:.2f} below floor"

    def test_min_notional_still_drops_uneconomic_positions(self) -> None:
        """
        Verify that truly uneconomic positions (yielding $10 notional at $10k equity)
        are still dropped by the min-notional filter.
        """
        optimizer = RiskParityOptimizer()

        rng = np.random.default_rng(42)
        base_prices = 100 + np.arange(100) * 0.01

        price_histories = {
            "TINYUSDT": pd.Series(base_prices + rng.standard_normal(100) * 0.05),
        }

        target_exposures = {"TINYUSDT": 0.001}
        equity_usd = 10_000.0

        result = optimizer.optimize(target_exposures, price_histories, equity_usd)

        assert result.weights == {}, "TINYUSDT should be dropped due to low notional"
        assert "TINYUSDT" in result.dropped_symbols, "TINYUSDT should be in dropped_symbols"
