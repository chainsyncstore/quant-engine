from typing import Dict, List
from hypotheses.base import Hypothesis, TradeIntent, IntentType
from portfolio.weighting import WeightingStrategy
from storage.repositories import EvaluationRepository
from promotion.models import HypothesisStatus

class Ensemble:
    """
    Container for a collection of hypotheses and their weighting logic.
    """
    def __init__(
        self,
        hypotheses: List[Hypothesis],
        weighting_strategy: WeightingStrategy,
        repo: EvaluationRepository,
        policy_id: str
    ):
        self.hypotheses = hypotheses
        self.weighting_strategy = weighting_strategy
        self.repo = repo
        self.policy_id = policy_id
        self.weights: Dict[str, float] = {}
        self.current_statuses: Dict[str, HypothesisStatus] = {}
        
        # Initial Status Load
        # In PROD, we fetch from DB. 
        # For simplicity, we assume they start as PROMOTED (or pass in).
        # We'll check repo.
        for h in hypotheses:
            # Need a way to get current status. Repo has get_hypothesis_status_history?
            # get_hypotheses_by_status checks active status.
            # We assume they are PROMOTED initially if loaded by runner.
            self.current_statuses[h.hypothesis_id] = HypothesisStatus.PROMOTED
            
        # Initial weight calculation
        self.update_weights()
        
    def update_weights(self):
        """Recalculate weights based on current strategy."""
        self.weights = self.weighting_strategy.calculate_weights(
            self.hypotheses, self.repo, self.policy_id, self.current_statuses
        )
        
    def set_status(self, hypothesis_id: str, status: HypothesisStatus):
        """Update status and weights."""
        self.current_statuses[hypothesis_id] = status
        self.update_weights()

    def aggregate_signal(
        self,
        intents: Dict[str, TradeIntent],
        current_allocations: Dict[str, float],
        total_capital: float
    ) -> float:
        """
        Aggregates individual intents into a single net MetaTradeIntent.
        
        Logic:
        - Each hypothesis desires a position size = Weight * Total_Capital * Intent.Size
        - Note: Intent.Size is usually relative (0-1). 
        - We output a Net Intent relative to Total Capital (or standardized size).
        
        Simplified C3 Approach:
        - Calculate Target Net Position (Signed Size).
        - If (Target - Current) > Threshold -> Emit ACTION.
        
        Wait, `TradeIntent` is an ACTION (Buy/Sell/Close). Not a TARGET POSITION.
        However, for Meta-Level, converting actions to net actions is complex if we don't track state.
        
        Better Approach:
        - We track `Target Portfolio Exposure`.
        - Sum(Weight * Side * Size) = Target Net Exposure (-1.0 to +1.0).
        - The `MetaPortfolioEngine` compares this target to current meta-position and generates the trade.
        
        So this method returns `Target Net Exposure`.
        """
        net_exposure = 0.0
        
        for h in self.hypotheses:
            hid = h.hypothesis_id
            weight = self.weights.get(hid, 0.0)
            if weight == 0:
                continue

            intent = intents.get(hid)
            if not intent:
                continue

            direction = 0.0
            if intent.type == IntentType.BUY:
                direction = 1.0
            elif intent.type == IntentType.SELL:
                direction = -1.0
            elif intent.type == IntentType.CLOSE:
                direction = 0.0

            net_exposure += weight * direction * intent.size

        return float(net_exposure)
