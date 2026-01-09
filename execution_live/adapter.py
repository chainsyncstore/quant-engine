"""
Execution adapter abstraction.

Research components must only depend on this interface (via callbacks).
Concrete broker or paper implementations live behind the boundary.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from execution_live.order_models import (
    AccountState,
    ExecutionIntent,
    ExecutionReport,
    PositionSnapshot,
)


class ExecutionAdapter(ABC):
    """
    Abstract adapter representing a trading venue or broker connection.

    Concrete implementations (paper, live, sandbox) must handle state, risk,
    and persistence while exposing a minimal surface area to the research engine.
    """

    @abstractmethod
    def get_account_state(self) -> AccountState:
        """Return the latest account snapshot."""

    @abstractmethod
    def get_positions(self) -> List[PositionSnapshot]:
        """Return all currently open positions."""

    @abstractmethod
    def place_order(self, intent: ExecutionIntent) -> ExecutionReport:
        """
        Submit an execution intent.

        Returns:
            ExecutionReport describing acceptance/fill or rejection.
        """

    @abstractmethod
    def cancel_order(self, order_id: str) -> ExecutionReport:
        """Attempt to cancel an existing order."""
