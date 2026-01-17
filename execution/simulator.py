"""
Execution simulator.

Handles trade execution, cost application, and PnL tracking.
"""

from typing import List, Optional
from pydantic import BaseModel, ConfigDict

from data.schemas import Bar
from engine.decision_queue import QueuedDecision
from execution.cost_model import CostModel, CostSide
from hypotheses.base import IntentType
from state.position_state import PositionState, PositionSide


class CompletedTrade(BaseModel):
    """
    Record of a completed trade.
    
    Immutable record for evaluation.
    """
    model_config = ConfigDict(frozen=True)
    
    trade_type: str  # ENTRY or EXIT
    side: str        # LONG or SHORT
    execution_price: float
    size: float
    execution_timestamp: object # datetime
    decision_timestamp: object # datetime
    cost_bps: float
    total_cost: float
    
    # Optional exit fields
    entry_price: Optional[float] = None
    entry_timestamp: Optional[object] = None
    realized_pnl: Optional[float] = None
    trade_duration_days: Optional[float] = None


class ExecutionSimulator:
    """
    Simulates trade execution with realistic costs.
    
    Responsibilities:
    - Execute trades at next-bar open (default)
    - Apply transaction costs
    - Update position state
    - Track completed trades
    """
    
    def __init__(
        self,
        cost_model: CostModel,
        initial_capital: float
    ):
        """
        Initialize execution simulator.
        
        Args:
            cost_model: Cost model for transaction costs
            initial_capital: Starting capital
        """
        self._cost_model = cost_model
        self._initial_capital = initial_capital
        self._available_capital = initial_capital
        self._completed_trades: List[CompletedTrade] = []
    
    def execute_decisions(
        self,
        decisions: List[QueuedDecision],
        execution_bar: Bar,
        position_state: PositionState
    ) -> List[CompletedTrade]:
        """
        Execute a list of decisions at the current bar.
        
        Args:
            decisions: Decisions to execute
            execution_bar: Bar to execute on (uses open price)
            position_state: Position state to update
            
        Returns:
            List of completed trades
        """
        trades: List[CompletedTrade] = []
        
        for decision in decisions:
            intent = decision.intent
            
            # Execute based on intent type
            if intent.type == IntentType.BUY:
                trade = self._execute_entry(
                    side=PositionSide.LONG,
                    execution_bar=execution_bar,
                    decision=decision,
                    position_state=position_state,
                    size=intent.size
                )
                if trade:
                    trades.append(trade)
            
            elif intent.type == IntentType.SELL:
                trade = self._execute_entry(
                    side=PositionSide.SHORT,
                    execution_bar=execution_bar,
                    decision=decision,
                    position_state=position_state,
                    size=intent.size
                )
                if trade:
                    trades.append(trade)
            
            elif intent.type == IntentType.CLOSE:
                trade = self._execute_exit(
                    execution_bar=execution_bar,
                    decision=decision,
                    position_state=position_state
                )
                if trade:
                    trades.append(trade)
        
        return trades
    
    def _execute_entry(
        self,
        side: PositionSide,
        execution_bar: Bar,
        decision: QueuedDecision,
        position_state: PositionState,
        size: float
    ) -> Optional[CompletedTrade]:
        """Execute a position entry."""
        # Can't enter if already have a position
        if position_state.has_position:
            return None
        
        # Use open price of the execution bar
        base_price = execution_bar.open
        
        # Apply costs
        cost_side = CostSide.BUY if side == PositionSide.LONG else CostSide.SELL
        effective_price = self._cost_model.apply_costs(base_price, cost_side)
        
        # Calculate position size based on available capital
        # For simplicity, use full available capital
        capital_to_deploy = self._available_capital
        position_size = capital_to_deploy / effective_price
        
        # Calculate total cost
        total_cost = self._cost_model.calculate_cost_amount(
            base_price,
            position_size,
            cost_side
        )
        
        # Open position
        position_state.open_position(
            side=side,
            entry_price=effective_price,
            size=position_size,
            entry_timestamp=execution_bar.timestamp,
            entry_capital=capital_to_deploy
        )
        
        # Update capital
        self._available_capital = 0.0  # All capital is now deployed
        
        # Record trade
        trade = CompletedTrade(
            trade_type="ENTRY",
            side=side.value,
            execution_price=effective_price,
            size=position_size,
            execution_timestamp=execution_bar.timestamp,
            decision_timestamp=decision.decision_timestamp,
            cost_bps=self._cost_model.get_total_cost_bps(),
            total_cost=total_cost
        )
        
        self._completed_trades.append(trade)
        return trade
    
    def _execute_exit(
        self,
        execution_bar: Bar,
        decision: QueuedDecision,
        position_state: PositionState
    ) -> Optional[CompletedTrade]:
        """Execute a position exit."""
        # Can't exit if no position
        if not position_state.has_position:
            return None
        
        position = position_state.position
        
        # Use open price of the execution bar
        base_price = execution_bar.open
        
        # Apply costs (opposite side of entry)
        cost_side = CostSide.SELL if position.side == PositionSide.LONG else CostSide.BUY
        effective_price = self._cost_model.apply_costs(base_price, cost_side)
        
        # Calculate realized PnL
        if position.side == PositionSide.LONG:
            realized_pnl = (effective_price - position.entry_price) * position.size
        else:  # SHORT
            realized_pnl = (position.entry_price - effective_price) * position.size
        
        # Calculate total cost
        total_cost = self._cost_model.calculate_cost_amount(
            base_price,
            position.size,
            cost_side
        )
        
        # Calculate trade duration
        duration = execution_bar.timestamp - position.entry_timestamp
        duration_days = duration.total_seconds() / (24 * 3600)
        
        # Update capital
        self._available_capital += position.entry_capital + realized_pnl
        
        # Close position
        closed_position = position_state.close_position()
        
        # Record trade
        trade = CompletedTrade(
            trade_type="EXIT",
            side=closed_position.side.value,
            execution_price=effective_price,
            size=closed_position.size,
            execution_timestamp=execution_bar.timestamp,
            decision_timestamp=decision.decision_timestamp,
            cost_bps=self._cost_model.get_total_cost_bps(),
            total_cost=total_cost,
            entry_price=closed_position.entry_price,
            entry_timestamp=closed_position.entry_timestamp,
            realized_pnl=realized_pnl,
            trade_duration_days=duration_days
        )
        
        self._completed_trades.append(trade)
        return trade
    
    def get_completed_trades(self) -> List[CompletedTrade]:
        """Get all completed trades."""
        return self._completed_trades.copy()
    
    def get_available_capital(self) -> float:
        """Get currently available capital."""
        return self._available_capital
    
    def get_total_capital(self, current_price: float, position_state: PositionState) -> float:
        """
        Get total capital including unrealized PnL.
        
        Args:
            current_price: Current market price
            position_state: Current position state
            
        Returns:
            Total capital (available + [entry_capital + unrealized_pnl])
        """
        if position_state.has_position:
            unrealized_pnl = position_state.get_unrealized_pnl(current_price)
            position = position_state.position
            return self._available_capital + position.entry_capital + unrealized_pnl
        
        return self._available_capital
    
    def reset(self) -> None:
        """Reset simulator state."""
        self._available_capital = self._initial_capital
        self._completed_trades.clear()
