"""Portfolio-aware execution planning from v2 strategy signals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING
from typing import Iterable

from quant_v2.contracts import ExecutionIntent, OrderPlan, StrategySignal
from quant_v2.portfolio.allocation import AllocationDecision, allocate_signals
from quant_v2.portfolio.optimizer import RiskParityOptimizer
from quant_v2.portfolio.risk_policy import OperatingRiskLimits, PolicyResult, PortfolioRiskPolicy

if TYPE_CHECKING:
    import pandas as pd


@dataclass(frozen=True)
class IntentPlan:
    """Combined allocation and risk-policy output."""

    intents: tuple[ExecutionIntent, ...]
    allocation: AllocationDecision
    policy_result: PolicyResult


@dataclass(frozen=True)
class PlannerConfig:
    """Execution planner tuning knobs."""

    total_risk_budget_frac: float = 1.0
    max_symbol_exposure_frac: float = 0.15
    min_confidence: float = 0.65
    enable_optimizer: bool = True
    equity_usd: float = 300.0
    target_headroom_ratio: float = 0.85
    fee_reserve_frac: float = 0.0
    slippage_reserve_frac: float = 0.0
    rounding_reserve_frac: float = 0.0
    min_quantity_reserve_frac: float = 0.0
    adverse_mark_buffer_frac: float = 0.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.total_risk_budget_frac <= 1.0:
            raise ValueError("total_risk_budget_frac must be in [0, 1]")
        if not 0.0 < self.max_symbol_exposure_frac <= 1.0:
            raise ValueError("max_symbol_exposure_frac must be in (0, 1]")
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError("min_confidence must be in [0, 1]")
        if self.equity_usd <= 0.0:
            raise ValueError("equity_usd must be positive")
        if not 0.0 < self.target_headroom_ratio <= 1.0:
            raise ValueError("target_headroom_ratio must be within (0, 1]")
        for name in (
            "fee_reserve_frac",
            "slippage_reserve_frac",
            "rounding_reserve_frac",
            "min_quantity_reserve_frac",
            "adverse_mark_buffer_frac",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value < 1.0:
                raise ValueError(f"{name} must be within [0, 1)")

    @property
    def reserve_capacity_frac(self) -> float:
        reserve = (
            float(self.fee_reserve_frac)
            + float(self.slippage_reserve_frac)
            + float(self.rounding_reserve_frac)
            + float(self.min_quantity_reserve_frac)
        )
        return min(max(reserve, 0.0), 0.95)

    @property
    def effective_operating_ratio(self) -> float:
        return float(self.target_headroom_ratio) * (1.0 - self.reserve_capacity_frac)


def build_execution_intents(
    signals: Iterable[StrategySignal],
    *,
    policy: PortfolioRiskPolicy,
    config: PlannerConfig = PlannerConfig(),
    bucket_map: dict[str, str] | None = None,
    reduce_only: bool = False,
    optimizer: RiskParityOptimizer | None = None,
    price_histories: dict[str, "pd.Series"] | None = None,
    current_positions: dict[str, float] | None = None,
) -> IntentPlan:
    """Convert strategy signals into policy-compliant execution intents.

    When *optimizer* is provided (and config.enable_optimizer is True),
    the allocation exposures are adjusted using risk-parity weights before
    the risk policy is applied.

    *current_positions* (symbol → quantity) lets the optimizer synthesise
    flatten targets for held positions that produced no incoming signal,
    preventing silent HOLDs from trapping a position indefinitely.  See
    audit_20260423 P0-3 / P0-3b.
    """
    allocation = allocate_signals(
        signals,
        total_risk_budget_frac=config.total_risk_budget_frac,
        max_symbol_exposure_frac=config.max_symbol_exposure_frac,
        min_confidence=config.min_confidence,
        equity_usd=config.equity_usd,
        current_positions=current_positions,
    )

    # --- Risk-parity optimization pass ---
    optimized_exposures = allocation.target_exposures
    # Run the optimizer whenever there are incoming targets OR we need to
    # synthesise a flatten for a held position that lost its signal.
    has_held_without_signal = bool(current_positions) and any(
        abs(qty) > 1e-12 and sym not in allocation.target_exposures
        for sym, qty in (current_positions or {}).items()
    )
    if (
        config.enable_optimizer
        and optimizer is not None
        and (allocation.target_exposures or has_held_without_signal)
    ):
        opt_result = optimizer.optimize(
            target_exposures=allocation.target_exposures,
            price_histories=price_histories or {},
            equity_usd=config.equity_usd,
            current_positions=current_positions,
        )
        if opt_result.weights:
            optimized_exposures = opt_result.weights

    if isinstance(policy, OperatingRiskLimits) and policy.limit_source == "operating_headroom":
        policy_result = policy.apply(
            optimized_exposures,
            bucket_map=bucket_map,
        )
    elif isinstance(policy, OperatingRiskLimits):
        operating_policy = OperatingRiskLimits.from_hard_limits(
            policy.hard_limits,
            max_symbol_exposure_frac=policy.max_symbol_exposure_frac,
            max_gross_exposure_frac=policy.max_gross_exposure_frac,
            max_net_exposure_frac=policy.max_net_exposure_frac,
            correlation_bucket_caps=policy.correlation_bucket_caps,
            limit_source=policy.limit_source,
            target_headroom_ratio=config.target_headroom_ratio,
            fee_reserve_frac=config.fee_reserve_frac,
            slippage_reserve_frac=config.slippage_reserve_frac,
            rounding_reserve_frac=config.rounding_reserve_frac,
            min_quantity_reserve_frac=config.min_quantity_reserve_frac,
            adverse_mark_buffer_frac=config.adverse_mark_buffer_frac,
        )
        policy_result = operating_policy.scaled(operating_policy.effective_headroom_ratio).apply(
            optimized_exposures,
            bucket_map=bucket_map,
        )
    else:
        policy_result = policy.scaled(config.effective_operating_ratio).apply(
            optimized_exposures,
            bucket_map=bucket_map,
        )

    intents: list[ExecutionIntent] = []
    signal_map = {signal.symbol: signal for signal in signals}
    import math as _math
    for symbol, exposure_frac in policy_result.exposures.items():
        if _math.isnan(exposure_frac):
            continue
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
