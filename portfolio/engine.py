import logging
from typing import Dict, List, Optional

from data.schemas import Bar
from hypotheses.base import TradeIntent, Hypothesis
from execution.simulator import ExecutionSimulator
from execution.cost_model import CostModel
from state.market_state import MarketState
from state.position_state import PositionState
from evaluation.policy import ResearchPolicy
from portfolio.models import PortfolioState, PortfolioAllocation
from portfolio.risk import RiskRule
from clock.clock import Clock

logger = logging.getLogger(__name__)

class PortfolioEngine:
    """
    Orchestrates the simulation of a portfolio of hypotheses.
    """
    def __init__(
        self,
        hypotheses: List[Hypothesis],
        initial_capital: float,
        policy: ResearchPolicy,
        risk_rules: Optional[List[RiskRule]] = None
    ):
        self.hypotheses = hypotheses
        self.initial_capital = initial_capital
        self.policy = policy
        self.risk_rules = list(risk_rules) if risk_rules else []
        
        # State Initialization
        capital_per_hypothesis = initial_capital / len(hypotheses) if hypotheses else 0
        
        self.simulators: Dict[str, ExecutionSimulator] = {}
        self.position_states: Dict[str, PositionState] = {}
        
        cost_model = CostModel(
            transaction_cost_bps=policy.transaction_cost_bps,
            slippage_bps=policy.slippage_bps
        )
        
        for h in hypotheses:
            self.simulators[h.hypothesis_id] = ExecutionSimulator(
                cost_model=cost_model,
                initial_capital=capital_per_hypothesis
            )
            self.position_states[h.hypothesis_id] = PositionState()

        # Global Clock & Market State
        self.clock = Clock()
        self.market_state = MarketState(lookback_window=policy.train_window_bars) # Use policy window

    def run(self, bars: List[Bar]) -> List[PortfolioState]:
        """
        Run the simulation over the provided bars.
        """
        history: List[PortfolioState] = []
        
        peak_equity = self.initial_capital
        
        for bar_idx, bar in enumerate(bars):
            self.clock.set_time(bar.timestamp)
            self.market_state.update(bar)
            
            # 1. Collect Intents
            intents: Dict[str, TradeIntent] = {}
            for h in self.hypotheses:
                intent = h.on_bar(
                    self.market_state, 
                    self.position_states[h.hypothesis_id], 
                    self.clock
                )
                if intent:
                    intents[h.hypothesis_id] = intent

            # 2. Risk Checks & Execution
            
            # Calculate current portfolio stats for risk checks (using PREVIOUS bar's state or Current Open?)
            # We use current bar open for execution, so state is technically "Pre-Bar Execution"
            
            # Aggregate State Update Loop
            for h in self.hypotheses:
                hid = h.hypothesis_id
                sim = self.simulators[hid]
                pos_state = self.position_states[hid]
                
                # Check Risk if Intent exists
                if hid in intents:
                    intent = intents[hid]
                    
                    # Create temporary allocation view for rule
                    dummy_allocation = PortfolioAllocation(
                        hypothesis_id=hid,
                        allocated_capital=sim.get_total_capital(bar.open, pos_state), # Approx
                        available_capital=sim.get_available_capital(),
                        symbol=bar.symbol,
                        reference_price=bar.open,
                        current_position=pos_state.position if pos_state.has_position else None,
                    )
                    
                    # Construct current Global State (Snapshot)
                    # Ideally we pass the PREVIOUS history[-1] but update with current market price
                    current_portfolio_snapshot = self._create_snapshot(bar, peak_equity)
                    
                    allowed = True
                    for rule in self.risk_rules:
                        can, reason = rule.can_execute(intent, dummy_allocation, current_portfolio_snapshot)
                        if not can:
                            allowed = False
                            logger.info(f"Rejected trade for {hid}: {reason}")
                            break
                    
                    if allowed:
                        self._notify_trade_approved(intent, dummy_allocation, current_portfolio_snapshot)
                        # Convert Intent to QueuedDecision (Immediate execution in this simple engine)
                        # We need to bridge the gap: Simulator takes QueuedDecision
                        from engine.decision_queue import QueuedDecision
                        decision = QueuedDecision(
                            intent=intent,
                            decision_timestamp=bar.timestamp,
                            # For immediate execution, index doesn't matter much to simulator 
                            # (sim uses intent & execution_bar), but queue needs valid index.
                            decision_bar_index=bar_idx 
                        )
                        
                        sim.execute_decisions([decision], bar, pos_state)

            # 3. Snapshot State AFTER execution
            snapshot = self._create_snapshot(bar, peak_equity)
            if snapshot.total_capital > peak_equity:
                peak_equity = snapshot.total_capital
                
            # Re-calculate drawdown with new peak
            # Since snapshot is frozen, we might need to adjust or rely on next bar. 
            # (Simplification: We calculate drawdown in _create_snapshot compared to PASSED peak)
            
            history.append(snapshot)
            
        return history

    def _create_snapshot(self, bar: Bar, peak_equity: float) -> PortfolioState:
        total_cap = 0.0
        total_cash = 0.0
        total_real = 0.0
        total_unreal = 0.0
        
        allocations = {}
        
        for h in self.hypotheses:
            hid = h.hypothesis_id
            sim = self.simulators[hid]
            pos_state = self.position_states[hid]
            
            cap = sim.get_total_capital(bar.close, pos_state) # Valuate at Close
            total_cap += cap
            available_cap = sim.get_available_capital()
            total_cash += available_cap
            
            # PnL logic in simulator is a bit hidden in trades list. 
            # We assume initial_capital per sim + pnl = current cap.
            # Realized PnL is (Current Cap - Unrealized PnL - Initial).
            
            unreal = 0.0
            if pos_state.has_position:
                unreal = pos_state.get_unrealized_pnl(bar.close)
                
            current_real = (cap - unreal) - sim._initial_capital
            
            total_unreal += unreal
            total_real += current_real
            
            allocations[hid] = PortfolioAllocation(
                hypothesis_id=hid,
                allocated_capital=cap,
                available_capital=available_cap,
                symbol=bar.symbol,
                reference_price=bar.close,
                current_position=pos_state.position if pos_state.has_position else None,
                unrealized_pnl=unreal,
                realized_pnl=current_real
            )

        drawdown = 0.0
        if peak_equity > 0:
            drawdown = max(0.0, (peak_equity - total_cap) / peak_equity * 100.0)

        return PortfolioState(
            timestamp=bar.timestamp,
            total_capital=total_cap,
            cash=total_cash,
            allocations=allocations,
            total_realized_pnl=total_real,
            total_unrealized_pnl=total_unreal,
            drawdown_pct=drawdown
        )

    def _notify_trade_approved(
        self,
        intent: TradeIntent,
        allocation: PortfolioAllocation,
        portfolio_state: PortfolioState,
    ) -> None:
        for rule in self.risk_rules:
            callback = getattr(rule, "on_trade_allowed", None)
            if callable(callback):
                callback(intent, allocation, portfolio_state)
