from typing import Protocol, Tuple
from hypotheses.base import TradeIntent, IntentType
from portfolio.models import PortfolioState, PortfolioAllocation

class RiskRule(Protocol):
    """
    Protocol for risk management rules.
    """
    def can_execute(
        self, 
        intent: TradeIntent, 
        allocation: PortfolioAllocation, 
        portfolio_state: PortfolioState
    ) -> Tuple[bool, str]:
        """
        Check if a trade intent can be executed given the current state.
        
        Returns:
            Tuple of (allowed, reason)
        """
        ...

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
        portfolio_state: PortfolioState
    ) -> Tuple[bool, str]:
        
        # Always allow exits (risk reduction)
        if intent.type in [IntentType.CLOSE, IntentType.SELL]: # Assuming Sell is exit or short
            # Note: Shorting adds risk, but CLOSE reduces it.
            # If intention is purely CLOSE existing, allow.
            # If intention is SELL (Short entry), check risk.
            if intent.type == IntentType.CLOSE:
                return True, "Risk reduction allowed"
            
            # If SELL means entering SHORT, we check drawdown.
        
        if portfolio_state.drawdown_pct > self.max_drawdown_pct:
            return False, f"Portfolio Drawdown {portfolio_state.drawdown_pct:.2f}% > Limit {self.max_drawdown_pct:.2f}%"
            
        return True, "Drawdown within limits"
