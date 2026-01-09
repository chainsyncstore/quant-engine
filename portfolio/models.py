from datetime import datetime
from typing import Dict, Optional
from pydantic import BaseModel, ConfigDict
from state.position_state import Position

class PortfolioAllocation(BaseModel):
    """Allocation detail for a single hypothesis."""

    hypothesis_id: str
    allocated_capital: float
    available_capital: float = 0.0
    symbol: Optional[str] = None
    reference_price: Optional[float] = None
    pending_notional: Optional[float] = None
    current_position: Optional[Position] = None
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0

class PortfolioState(BaseModel):
    """
    Immutable snapshot of the portfolio state at a specific point in time.
    """

    model_config = ConfigDict(frozen=True)

    timestamp: datetime
    total_capital: float  # Cash + Market Value of Positions
    cash: float
    allocations: Dict[str, PortfolioAllocation]  # hypothesis_id -> state

    # Portfolio level metrics
    total_realized_pnl: float
    total_unrealized_pnl: float
    drawdown_pct: float

    def get_allocation(self, hypothesis_id: str) -> Optional[PortfolioAllocation]:
        return self.allocations.get(hypothesis_id)
