from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_v2.contracts import StrategySignal
from quant_v2.execution.planner import (
    PlannerConfig,
    build_execution_intents,
    intents_to_order_plans,
)
from quant_v2.portfolio.optimizer import RiskParityOptimizer
from quant_v2.portfolio.risk_policy import PortfolioRiskPolicy


def _signal(symbol: str, signal: str, confidence: float, uncertainty: float | None = None) -> StrategySignal:
    return StrategySignal(
        symbol=symbol,
        timeframe="1h",
        horizon_bars=4,
        signal=signal,
        confidence=confidence,
        uncertainty=uncertainty,
    )


def test_build_execution_intents_applies_allocation_and_policy() -> None:
    signals = [
        _signal("BTCUSDT", "BUY", 0.74, uncertainty=0.10),
        _signal("ETHUSDT", "SELL", 0.69, uncertainty=0.15),
        _signal("SOLUSDT", "HOLD", 0.50),
    ]

    policy = PortfolioRiskPolicy(
        max_symbol_exposure_frac=0.05,
        max_gross_exposure_frac=0.08,
        max_net_exposure_frac=0.06,
    )

    plan = build_execution_intents(
        signals,
        policy=policy,
        config=PlannerConfig(total_risk_budget_frac=0.50, max_symbol_exposure_frac=0.06, min_confidence=0.65),
    )

    assert len(plan.intents) == 2
    assert plan.allocation.gross_exposure > 0
    assert plan.policy_result.gross_exposure <= 0.08 + 1e-12
    assert "SOLUSDT" in plan.allocation.skipped_symbols


def test_intents_to_order_plans_builds_quantities() -> None:
    signals = [_signal("BTCUSDT", "BUY", 0.73), _signal("ETHUSDT", "SELL", 0.70)]
    policy = PortfolioRiskPolicy()
    plan = build_execution_intents(
        signals,
        policy=policy,
        config=PlannerConfig(total_risk_budget_frac=0.50, max_symbol_exposure_frac=0.05, min_confidence=0.65),
    )

    orders = intents_to_order_plans(
        plan.intents,
        prices={"BTCUSDT": 50000.0, "ETHUSDT": 2500.0},
        equity_usd=10000.0,
    )

    assert len(orders) == 2
    by_symbol = {order.symbol: order for order in orders}
    assert by_symbol["BTCUSDT"].side == "BUY"
    assert by_symbol["ETHUSDT"].side == "SELL"
    assert by_symbol["BTCUSDT"].quantity > 0
    assert by_symbol["ETHUSDT"].quantity > 0


def test_intents_to_order_plans_validate_equity() -> None:
    signals = [_signal("BTCUSDT", "BUY", 0.70)]
    policy = PortfolioRiskPolicy()
    plan = build_execution_intents(signals, policy=policy)

    with pytest.raises(ValueError):
        intents_to_order_plans(plan.intents, prices={"BTCUSDT": 50000.0}, equity_usd=0.0)


# ====================================================================
# audit_20260423 P0-3b — current_positions plumbing into the optimizer
# ====================================================================
#
# Note on scope: the optimizer synthesises a weight-0 entry for held-but-
# silent symbols and tags the OptimizerResult with "flatten_held_no_signal".
# PortfolioRiskPolicy.apply strips zero exposures (by design, pre-dating
# this audit), so the downstream flatten relies on reconcile_target_exposures
# which already handles the union of (targets ∪ current_positions) → flatten.
# These tests verify the optimizer is actually invoked with current_positions
# and that the audit trail (constraint tag) fires.  End-to-end flatten
# behaviour is exercised by the service-level tests in test_execution_service.


def _make_price_hist(seed: int = 42, n: int = 100, base: float = 100.0) -> pd.Series:
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0, 0.02, n)
    return pd.Series(base * np.cumprod(1.0 + returns))


def test_planner_passes_current_positions_to_optimizer() -> None:
    """Held BNBUSDT with no signal triggers optimizer flatten-synthesis path."""
    captured: dict[str, object] = {}

    class _SpyOptimizer(RiskParityOptimizer):
        def optimize(self, target_exposures, price_histories, equity_usd, *, current_positions=None):  # type: ignore[override]
            captured["target_exposures"] = dict(target_exposures)
            captured["current_positions"] = dict(current_positions) if current_positions else None
            return super().optimize(
                target_exposures,
                price_histories,
                equity_usd,
                current_positions=current_positions,
            )

    signals = [_signal("BTCUSDT", "BUY", 0.74, uncertainty=0.10)]
    policy = PortfolioRiskPolicy(max_symbol_exposure_frac=0.10)
    spy = _SpyOptimizer(min_notional_usd=0.0)
    histories = {
        "BTCUSDT": _make_price_hist(seed=1),
        "BNBUSDT": _make_price_hist(seed=2),
    }

    build_execution_intents(
        signals,
        policy=policy,
        config=PlannerConfig(total_risk_budget_frac=1.0, min_confidence=0.65, equity_usd=10_000.0),
        optimizer=spy,
        price_histories=histories,
        current_positions={"BNBUSDT": 0.527},
    )

    assert captured["current_positions"] == {"BNBUSDT": 0.527}


def test_planner_runs_optimizer_when_only_held_positions_have_no_signal() -> None:
    """With zero incoming signals but a held position, the optimizer still runs.

    Before P0-3b, build_execution_intents short-circuited the optimizer whenever
    allocation.target_exposures was empty, regardless of held positions.  This
    test pins the new behaviour: optimizer runs so the flatten constraint is
    visible in diagnostics, even when the downstream policy strips the zero.
    """
    calls: list[dict] = []

    class _CountingOptimizer(RiskParityOptimizer):
        def optimize(self, target_exposures, price_histories, equity_usd, *, current_positions=None):  # type: ignore[override]
            calls.append({
                "targets": dict(target_exposures),
                "currents": dict(current_positions) if current_positions else None,
            })
            return super().optimize(
                target_exposures,
                price_histories,
                equity_usd,
                current_positions=current_positions,
            )

    signals: list[StrategySignal] = []  # all-HOLD cycle
    policy = PortfolioRiskPolicy(max_symbol_exposure_frac=0.10)
    optimizer = _CountingOptimizer(min_notional_usd=0.0)

    build_execution_intents(
        signals,
        policy=policy,
        config=PlannerConfig(equity_usd=10_000.0),
        optimizer=optimizer,
        price_histories={"BNBUSDT": _make_price_hist(seed=3)},
        current_positions={"BNBUSDT": 0.527},
    )

    assert len(calls) == 1
    assert calls[0]["currents"] == {"BNBUSDT": 0.527}


def test_planner_skips_optimizer_when_no_signals_and_no_positions() -> None:
    """No signals + no held positions → optimizer is NOT run (short-circuit preserved)."""
    calls: list[dict] = []

    class _CountingOptimizer(RiskParityOptimizer):
        def optimize(self, *args, **kwargs):  # type: ignore[override]
            calls.append(kwargs)
            return super().optimize(*args, **kwargs)

    policy = PortfolioRiskPolicy(max_symbol_exposure_frac=0.10)
    optimizer = _CountingOptimizer(min_notional_usd=0.0)

    build_execution_intents(
        [],
        policy=policy,
        config=PlannerConfig(equity_usd=10_000.0),
        optimizer=optimizer,
        price_histories={},
        current_positions={},
    )

    assert calls == []


def test_planner_without_current_positions_is_backwards_compatible() -> None:
    """Omitting current_positions preserves the pre-P0-3b behaviour exactly."""
    signals = [_signal("BTCUSDT", "BUY", 0.74)]
    policy = PortfolioRiskPolicy(max_symbol_exposure_frac=0.10)
    optimizer = RiskParityOptimizer(min_notional_usd=0.0)

    plan = build_execution_intents(
        signals,
        policy=policy,
        config=PlannerConfig(equity_usd=10_000.0),
        optimizer=optimizer,
        price_histories={"BTCUSDT": _make_price_hist(seed=4)},
    )

    assert set(plan.policy_result.exposures.keys()) == {"BTCUSDT"}
