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


class TestFlattenHeldNoSignal:
    """Tests for synthesised flatten of held positions with no incoming signal (P0-3)."""

    def test_held_symbol_with_no_signal_is_flattened(self) -> None:
        """Held position + empty target_exposures → explicit zero-weight flatten intent."""
        optimizer = RiskParityOptimizer()

        rng = np.random.default_rng(42)
        base_prices = 100 + np.arange(100) * 0.01
        price_histories = {
            "BNBUSDT": pd.Series(base_prices + rng.standard_normal(100) * 0.05),
        }

        result = optimizer.optimize(
            target_exposures={},
            price_histories=price_histories,
            equity_usd=10_000.0,
            current_positions={"BNBUSDT": 0.05},
        )

        assert result.weights == {"BNBUSDT": 0.0}
        assert "flatten_held_no_signal" in result.constraints_applied

    def test_held_symbol_with_existing_signal_is_untouched(self) -> None:
        """Explicit SELL target must win over synthesis; flatten sentinel absent."""
        optimizer = RiskParityOptimizer()

        rng = np.random.default_rng(42)
        base_prices = 100 + np.arange(100) * 0.01
        price_histories = {
            "BNBUSDT": pd.Series(base_prices + rng.standard_normal(100) * 0.05),
        }

        result = optimizer.optimize(
            target_exposures={"BNBUSDT": -0.08},
            price_histories=price_histories,
            equity_usd=10_000.0,
            current_positions={"BNBUSDT": 0.05},
        )

        assert "BNBUSDT" in result.weights
        assert result.weights["BNBUSDT"] < 0, "Explicit SELL should produce negative weight"
        assert "flatten_held_no_signal" not in result.constraints_applied

    def test_flatten_bypasses_min_notional_filter(self) -> None:
        """Flatten intent must survive the min-notional filter at any equity."""
        optimizer = RiskParityOptimizer()

        rng = np.random.default_rng(42)
        base_prices = 100 + np.arange(100) * 0.01
        price_histories = {
            "BNBUSDT": pd.Series(base_prices + rng.standard_normal(100) * 0.05),
        }

        # 0.001 position × $10k equity = $10 notional, below the 0.5% floor ($50).
        result = optimizer.optimize(
            target_exposures={},
            price_histories=price_histories,
            equity_usd=10_000.0,
            current_positions={"BNBUSDT": 0.001},
        )

        assert "BNBUSDT" in result.weights
        assert result.weights["BNBUSDT"] == 0.0
        assert "BNBUSDT" not in result.dropped_symbols
        assert "flatten_held_no_signal" in result.constraints_applied

    def test_current_positions_none_is_backwards_compatible(self) -> None:
        """Omitting current_positions must reproduce pre-patch behaviour exactly."""
        optimizer = RiskParityOptimizer()

        rng = np.random.default_rng(42)
        base_prices = 100 + np.arange(100) * 0.01
        price_histories = {
            "FOOUSDT": pd.Series(base_prices + rng.standard_normal(100) * 0.05),
            "BARUSDT": pd.Series(base_prices + rng.standard_normal(100) * 0.05),
        }

        target_exposures = {"FOOUSDT": 0.018, "BARUSDT": 0.018}
        equity_usd = 10_000.0

        # Reference: call without the new kwarg (positional-only args) → must match explicit None.
        reference = optimizer.optimize(target_exposures, price_histories, equity_usd)
        with_none = optimizer.optimize(
            target_exposures,
            price_histories,
            equity_usd,
            current_positions=None,
        )

        assert with_none.weights == reference.weights
        assert with_none.dropped_symbols == reference.dropped_symbols
        assert with_none.constraints_applied == reference.constraints_applied
        assert "flatten_held_no_signal" not in reference.constraints_applied
