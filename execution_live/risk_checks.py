"""
Generic risk checks that operate on ExecutionIntents.
"""

from __future__ import annotations

from typing import Protocol, Tuple

from config.execution_policies import ExecutionPolicy
from config.execution_policy_guard import ExecutionPolicyGuard, PolicyOrderContext
from execution_live.order_models import AccountState, ExecutionIntent, IntentAction


class RiskCheck(Protocol):
    """Protocol for adapter-level risk checks."""

    def evaluate(self, intent: ExecutionIntent, account_state: AccountState) -> Tuple[bool, str]:
        ...


class NotionalLimitCheck:
    """Rejects orders whose notional exceeds the configured limit."""

    def __init__(self, max_notional: float):
        self.max_notional = max_notional

    def evaluate(self, intent: ExecutionIntent, account_state: AccountState) -> Tuple[bool, str]:
        reference_price = intent.reference_price or intent.limit_price
        if reference_price is None:
            return False, "Reference price is required for risk evaluation."

        notional = abs(intent.quantity) * reference_price
        if notional > self.max_notional:
            return False, f"Order notional {notional:,.2f} exceeds limit {self.max_notional:,.2f}"

        return True, "Within notional limit"


class CashAvailabilityCheck:
    """Ensures available cash/buying power is sufficient for the trade."""

    def evaluate(self, intent: ExecutionIntent, account_state: AccountState) -> Tuple[bool, str]:
        reference_price = intent.reference_price or intent.limit_price
        if reference_price is None:
            return False, "Reference price is required for cash check."

        required = abs(intent.quantity) * reference_price
        if required > account_state.cash:
            return False, f"Insufficient cash. Required {required:,.2f}, available {account_state.cash:,.2f}"

        return True, "Cash available"


class ExecutionPolicyCheck:
    """
    Adapter-level risk check backed by ExecutionPolicyGuard.

    Keeps intra-day state (drawdown, trade counts) in sync with account snapshots
    and rejects intents that would violate the configured policy.
    """

    def __init__(self, policy: ExecutionPolicy):
        self._guard = ExecutionPolicyGuard(policy)

    def evaluate(self, intent: ExecutionIntent, account_state: AccountState) -> Tuple[bool, str]:
        # Track latest equity for drawdown calculations
        self._guard.observe_equity(account_state.timestamp, account_state.equity)

        reference_price = intent.reference_price or intent.limit_price
        if reference_price is None:
            return False, "Reference price required for execution policy evaluation."

        notional = abs(intent.quantity) * reference_price
        ctx = PolicyOrderContext(
            timestamp=intent.timestamp,
            symbol=intent.symbol,
            notional=notional,
            is_entry=intent.action in {IntentAction.BUY, IntentAction.SELL},
        )

        allowed, reason = self._guard.evaluate_order(ctx)
        if allowed and ctx.is_entry:
            # Record immediately so multiple checks in the same bar don't bypass trade limits.
            self._guard.record_entry(ctx)
        return allowed, reason

    @property
    def policy_label(self) -> str:
        return self._guard.label()

    def policy_snapshot(self):
        """Serialized policy for logging/debugging."""
        return self._guard.serialize_policy()
