from typing import Protocol, Tuple

from config.execution_policies import ExecutionPolicy
from config.execution_policy_guard import ExecutionPolicyGuard, PolicyOrderContext
from hypotheses.base import IntentType, TradeIntent
from portfolio.models import PortfolioAllocation, PortfolioState


class RiskRule(Protocol):
    """
    Protocol for risk management rules.
    """

    def can_execute(
        self,
        intent: TradeIntent,
        allocation: PortfolioAllocation,
        portfolio_state: PortfolioState,
    ) -> Tuple[bool, str]:
        """
        Check if a trade intent can be executed given the current state.

        Returns:
            Tuple of (allowed, reason)
        """

    def on_trade_allowed(
        self,
        intent: TradeIntent,
        allocation: PortfolioAllocation,
        portfolio_state: PortfolioState,
    ) -> None:
        """
        Optional hook invoked after a trade intent passes all risk checks.
        """


class MaxDrawdownRule:
    """
    Rejects NEW risk (Entries) if portfolio drawdown exceeds limit.
    """

    def __init__(self, max_drawdown_pct: float):
        self.max_drawdown_pct = max_drawdown_pct

    def can_execute(
        self,
        intent: TradeIntent,
        allocation: PortfolioAllocation,
        portfolio_state: PortfolioState,
    ) -> Tuple[bool, str]:
        # Always allow exits (risk reduction)
        if intent.type in [IntentType.CLOSE, IntentType.SELL]:  # Assuming Sell is exit or short
            # Note: Shorting adds risk, but CLOSE reduces it.
            # If intention is purely CLOSE existing, allow.
            # If intention is SELL (Short entry), check risk.
            if intent.type == IntentType.CLOSE:
                return True, "Risk reduction allowed"

            # If SELL means entering SHORT, we check drawdown.

        if portfolio_state.drawdown_pct > self.max_drawdown_pct:
            return False, f"Portfolio Drawdown {portfolio_state.drawdown_pct:.2f}% > Limit {self.max_drawdown_pct:.2f}%"

        return True, "Drawdown within limits"


class ExecutionPolicyRule:
    """
    Composite risk rule enforcing ExecutionPolicy constraints at the portfolio/risk layer.
    """

    def __init__(self, policy: ExecutionPolicy):
        self.policy = policy
        self._guard = ExecutionPolicyGuard(policy)

    def can_execute(
        self,
        intent: TradeIntent,
        allocation: PortfolioAllocation,
        portfolio_state: PortfolioState,
    ) -> Tuple[bool, str]:
        if intent.type == IntentType.CLOSE:
            return True, "Risk reduction allowed"

        ctx = self._build_context(intent, allocation, portfolio_state)
        self._guard.observe_equity(portfolio_state.timestamp, portfolio_state.total_capital)
        return self._guard.evaluate_order(ctx)

    def on_trade_allowed(
        self,
        intent: TradeIntent,
        allocation: PortfolioAllocation,
        portfolio_state: PortfolioState,
    ) -> None:
        if intent.type == IntentType.CLOSE:
            return
        ctx = self._build_context(intent, allocation, portfolio_state)
        self._guard.record_entry(ctx)

    def _build_context(
        self, intent: TradeIntent, allocation: PortfolioAllocation, portfolio_state: PortfolioState
    ) -> PolicyOrderContext:
        reference_price = allocation.reference_price
        if reference_price is None:
            raise ValueError("PortfolioAllocation.reference_price is required for execution policy evaluation.")
        notional = intent.size * reference_price
        return PolicyOrderContext(
            timestamp=portfolio_state.timestamp,
            symbol=allocation.symbol,
            notional=abs(notional),
            is_entry=intent.type in {IntentType.BUY, IntentType.SELL},
        )
