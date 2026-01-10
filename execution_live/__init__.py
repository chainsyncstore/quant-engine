"""
Execution boundary for live or paper adapters.

This package introduces the ExecutionAdapter abstraction that isolates
research components from broker-specific integrations.
"""

from execution_live.adapter import ExecutionAdapter
from execution_live.paper_broker import PaperExecutionAdapter
from execution_live.order_models import (
    ExecutionIntent,
    IntentAction,
    OrderStatus,
    OrderType,
    TimeInForce,
    ExecutionReport,
    ExecutionEvent,
    AccountState,
    PositionSnapshot,
)
from execution_live.event_logger import ExecutionEventLogger
from execution_live.events import (
    COMPETITION_DAILY_HALT,
    COMPETITION_PROFILE_LOADED,
    COMPETITION_TRADE_BLOCKED,
)

__all__ = [
    "ExecutionAdapter",
    "PaperExecutionAdapter",
    "ExecutionIntent",
    "IntentAction",
    "OrderStatus",
    "OrderType",
    "TimeInForce",
    "ExecutionReport",
    "ExecutionEvent",
    "AccountState",
    "PositionSnapshot",
    "ExecutionEventLogger",
    "COMPETITION_TRADE_BLOCKED",
    "COMPETITION_DAILY_HALT",
    "COMPETITION_PROFILE_LOADED",
]
