"""
Position state management.

Tracks the current open position (if any) and calculates unrealized PnL.
"""

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class PositionSide(str, Enum):
    """Position direction."""
    LONG = "LONG"
    SHORT = "SHORT"


class Position(BaseModel):
    """
    Represents an open position.
    
    All fields are immutable to prevent accidental modification.
    """
    model_config = ConfigDict(frozen=True)
    
    side: PositionSide = Field(
        description="Position direction (LONG or SHORT)"
    )
    
    entry_price: float = Field(
        gt=0.0,
        description="Entry price"
    )
    
    size: float = Field(
        gt=0.0,
        description="Position size (number of units)"
    )
    
    entry_timestamp: datetime = Field(
        description="When the position was opened"
    )
    
    entry_capital: float = Field(
        gt=0.0,
        description="Capital deployed to open this position"
    )
    
    def unrealized_pnl(self, current_price: float) -> float:
        """
        Calculate unrealized profit/loss.
        
        Args:
            current_price: Current market price
            
        Returns:
            Unrealized PnL (positive = profit, negative = loss)
        """
        if self.side == PositionSide.LONG:
            return (current_price - self.entry_price) * self.size
        else:  # SHORT
            return (self.entry_price - current_price) * self.size
    
    def unrealized_pnl_pct(self, current_price: float) -> float:
        """
        Calculate unrealized PnL as a percentage of entry capital.
        
        Args:
            current_price: Current market price
            
        Returns:
            Unrealized PnL percentage
        """
        pnl = self.unrealized_pnl(current_price)
        return (pnl / self.entry_capital) * 100.0


class PositionState:
    """
    Tracks the current position state.
    
    Only one position can be open at a time (v0 constraint).
    """
    
    def __init__(self):
        """Initialize empty position state."""
        self._position: Optional[Position] = None
    
    @property
    def has_position(self) -> bool:
        """
        Check if there is an open position.
        
        Returns:
            True if a position is open, False otherwise
        """
        return self._position is not None
    
    @property
    def position(self) -> Position:
        """
        Get the current position.
        
        Returns:
            Current position
            
        Raises:
            RuntimeError: If no position is open
        """
        if self._position is None:
            raise RuntimeError("No position is currently open")
        
        return self._position
    
    def get_position(self) -> Optional[Position]:
        """
        Get the current position (safe version).
        
        Returns:
            Current position or None if no position is open
        """
        return self._position
    
    def open_position(
        self,
        side: PositionSide,
        entry_price: float,
        size: float,
        entry_timestamp: datetime,
        entry_capital: float
    ) -> None:
        """
        Open a new position.
        
        Args:
            side: Position direction
            entry_price: Entry price
            size: Position size
            entry_timestamp: Entry timestamp
            entry_capital: Capital deployed
            
        Raises:
            RuntimeError: If a position is already open
        """
        if self.has_position:
            raise RuntimeError(
                "Cannot open new position: a position is already open. "
                "Close the existing position first."
            )
        
        self._position = Position(
            side=side,
            entry_price=entry_price,
            size=size,
            entry_timestamp=entry_timestamp,
            entry_capital=entry_capital
        )
    
    def close_position(self) -> Position:
        """
        Close the current position.
        
        Returns:
            The closed position
            
        Raises:
            RuntimeError: If no position is open
        """
        if not self.has_position:
            raise RuntimeError("Cannot close position: no position is open")
        
        closed_position = self._position
        self._position = None
        assert closed_position is not None  # mypy: guarded by has_position check
        return closed_position
    
    def get_unrealized_pnl(self, current_price: float) -> float:
        """
        Get unrealized PnL for the current position.
        
        Args:
            current_price: Current market price
            
        Returns:
            Unrealized PnL, or 0.0 if no position is open
        """
        if not self.has_position:
            return 0.0
        
        return self.position.unrealized_pnl(current_price)
    
    def reset(self) -> None:
        """
        Reset position state.
        
        Should only be used for testing or starting a new evaluation run.
        """
        self._position = None
