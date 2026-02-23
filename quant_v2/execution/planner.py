"""Portfolio-aware execution planning from v2 strategy signals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from quant_v2.contracts import ExecutionIntent, OrderPlan, StrategySignal
from quant_v2.portfolio.allocation import AllocationDecision, allocate_signals
from quant_v2.portfolio.risk_policy import PolicyResult, PortfolioRiskPolicy


@dataclass(frozen=True)
class IntentPlan:
    """Combined allocation and risk-policy output."""

    intents: tuple[ExecutionIntent, ...]
    allocation: AllocationDecision
    policy_result: PolicyResult


@dataclass(frozen=True)
class PlannerConfig:
    """Execution planner tuning knobs."""

    total_risk_budget_frac: float = 0.15
    max_symbol_exposure_frac: float = 0.05
    min_confidence: float = 0.55


def build_execution_intents(
    signals: Iterable[StrategySignal],
    *,
    policy: PortfolioRiskPolicy,
    config: PlannerConfig = PlannerConfig(),
    bucket_map: dict[str, str] | None = None,
    reduce_only: bool = False,
) -> IntentPlan:
    """Convert strategy signals into policy-compliant execution intents."""

    allocation = allocate_signals(
        signals,
        total_risk_budget_frac=config.total_risk_budget_frac,
        max_symbol_exposure_frac=config.max_symbol_exposure_frac,
        min_confidence=config.min_confidence,
    )

    policy_result = policy.apply(allocation.target_exposures, bucket_map=bucket_map)

    intents: list[ExecutionIntent] = []
    signal_map = {signal.symbol: signal for signal in signals}
    for symbol, exposure_frac in policy_result.exposures.items():
        signal = signal_map.get(symbol)
        if signal is None or not signal.actionable:
            continue
        intents.append(
            ExecutionIntent(
                symbol=symbol,
                signal=signal,
                target_notional_usd=abs(exposure_frac),
                risk_budget_frac=abs(exposure_frac),
                reduce_only=reduce_only,
            )
        )

    return IntentPlan(
        intents=tuple(intents),
        allocation=allocation,
        policy_result=policy_result,
    )


def intents_to_order_plans(
    intents: Iterable[ExecutionIntent],
    *,
    prices: dict[str, float],
    equity_usd: float,
    min_qty: float = 0.0,
) -> tuple[OrderPlan, ...]:
    """Translate intents into quantity-based order plans."""

    if equity_usd <= 0.0:
        raise ValueError("equity_usd must be positive")

    plans: list[OrderPlan] = []
    for intent in intents:
        price = float(prices.get(intent.symbol, 0.0))
        if price <= 0.0:
            continue

        signed_exposure = intent.risk_budget_frac if intent.signal.signal == "BUY" else -intent.risk_budget_frac
        notional = abs(signed_exposure) * equity_usd
        quantity = notional / price
        if quantity <= min_qty:
            continue

        side = "BUY" if signed_exposure > 0 else "SELL"
        plans.append(
            OrderPlan(
                symbol=intent.symbol,
                side=side,
                quantity=quantity,
                reduce_only=intent.reduce_only,
            )
        )

    return tuple(plans)
