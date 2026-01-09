"""
Generic risk checks that operate on ExecutionIntents.
"""

from __future__ import annotations

from typing import Protocol, Tuple

from execution_live.order_models import AccountState, ExecutionIntent


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
