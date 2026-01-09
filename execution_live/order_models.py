"""
Order and execution domain models for the execution boundary.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class IntentAction(str, Enum):
    """High-level instruction emitted by research components."""

    BUY = "BUY"
    SELL = "SELL"
    CLOSE = "CLOSE"


class OrderType(str, Enum):
    """Supported order types."""

    MARKET = "MARKET"
    LIMIT = "LIMIT"


class TimeInForce(str, Enum):
    """Order lifetime policies."""

    DAY = "DAY"
    GTC = "GTC"


class OrderStatus(str, Enum):
    """Lifecycle states for an order."""

    ACCEPTED = "ACCEPTED"
    REJECTED = "REJECTED"
    FILLED = "FILLED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    CANCELLED = "CANCELLED"


class PositionSnapshot(BaseModel):
    """Lightweight immutable view of an open position."""

    model_config = ConfigDict(frozen=True)

    symbol: str
    quantity: float
    average_price: float
    side: str
    unrealized_pnl: float = 0.0


class AccountState(BaseModel):
    """Adapter-level account snapshot."""

    model_config = ConfigDict(frozen=True)

    equity: float
    cash: float
    buying_power: float
    timestamp: datetime
    positions: List[PositionSnapshot] = Field(default_factory=list)


class ExecutionIntent(BaseModel):
    """
    Instruction emitted by orchestrators/meta engines instead of direct broker calls.
    """

    model_config = ConfigDict(frozen=True)

    symbol: str
    action: IntentAction
    quantity: float = Field(gt=0.0)
    timestamp: datetime
    order_type: OrderType = OrderType.MARKET
    time_in_force: TimeInForce = TimeInForce.DAY
    reference_price: Optional[float] = Field(default=None)
    limit_price: Optional[float] = Field(default=None)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    reference_id: Optional[str] = Field(
        default=None,
        description="Optional upstream identifier for tracing.",
    )


class ExecutionReport(BaseModel):
    """Result of attempting to place or cancel an order."""

    model_config = ConfigDict(frozen=True)

    order_id: str
    status: OrderStatus
    intent: ExecutionIntent
    filled_quantity: float = 0.0
    avg_fill_price: Optional[float] = None
    message: Optional[str] = None
    realized_pnl: Optional[float] = None
    cost_paid: Optional[float] = None


class ExecutionEvent(BaseModel):
    """Event emitted by the adapter for auditability."""

    model_config = ConfigDict(frozen=True)

    event_type: str
    timestamp: datetime
    payload: Dict[str, Any]
