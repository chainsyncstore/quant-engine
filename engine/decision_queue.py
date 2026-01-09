"""
Decision queue management.

Buffers trade intents to enforce execution delays (preventing look-ahead bias).
"""

from typing import List
from datetime import datetime
from pydantic import BaseModel, ConfigDict

from hypotheses.base import TradeIntent


class QueuedDecision(BaseModel):
    """
    A decision waiting in the queue.
    
    Wraps a TradeIntent with timing information.
    """
    model_config = ConfigDict(frozen=True)
    
    intent: TradeIntent
    decision_timestamp: datetime
    decision_bar_index: int


class DecisionQueue:
    """
    Manages the buffer of trade decisions.
    
    Enforces the `execution_delay_bars` rule.
    """
    
    def __init__(self, execution_delay_bars: int = 1):
        """
        Initialize queue.
        
        Args:
            execution_delay_bars: Number of bars to wait before execution.
                                  Default 1 means "execute on NEXT bar".
        """
        self._execution_delay_bars = execution_delay_bars
        self._queue: List[QueuedDecision] = []
    
    def enqueue(self, intent: TradeIntent, decision_timestamp: datetime, decision_bar_index: int):
        """
        Add a decision to the queue.
        
        Args:
            intent: The trade intent
            decision_timestamp: When the decision was made
            decision_bar_index: Index of bar when decision was made
        """
        decision = QueuedDecision(
            intent=intent,
            decision_timestamp=decision_timestamp,
            decision_bar_index=decision_bar_index
        )
        self._queue.append(decision)
    
    def get_executable_decisions(self, current_bar_index: int) -> List[QueuedDecision]:
        """
        Get decisions that are ready to execute.
        
        Ready means: current_bar_index >= decision_bar_index + delay
        
        Args:
            current_bar_index: The index of the bar ABOUT TO BE PROCESSED.
                               
        Returns:
            List of executable decisions (removed from queue)
        """
        executable = []
        remaining = []
        
        for decision in self._queue:
            # Logic:
            # If delay=1 (execute next bar)
            # Decision at index 0. Target = 0 + 1 = 1.
            # If current_bar_index is 1, we are at start of Bar 1.
            # 1 >= 1. True. Execute.
            
            if current_bar_index >= decision.decision_bar_index + self._execution_delay_bars:
                executable.append(decision)
            else:
                remaining.append(decision)
        
        self._queue = remaining
        return executable
        
    def has_pending_decisions(self) -> int:
        """Count pending decisions."""
        return len(self._queue)
    
    def clear(self):
        """Clear the queue."""
        self._queue.clear()
