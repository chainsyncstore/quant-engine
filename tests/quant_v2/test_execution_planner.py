from __future__ import annotations

import pytest

from quant_v2.contracts import StrategySignal
from quant_v2.execution.planner import (
    PlannerConfig,
    build_execution_intents,
    intents_to_order_plans,
)
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
        config=PlannerConfig(total_risk_budget_frac=0.12, max_symbol_exposure_frac=0.06, min_confidence=0.55),
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
        config=PlannerConfig(total_risk_budget_frac=0.10, max_symbol_exposure_frac=0.05, min_confidence=0.55),
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
