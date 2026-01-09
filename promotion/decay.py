from typing import Tuple
from batch.models import RankedHypothesis
from evaluation.policy import ResearchPolicy
from promotion.rules import PromotionRule

class DecayRule(PromotionRule):
    """
    Rule to check if a hypothesis has decayed based on recent performance.
    """
    def evaluate(self, h: RankedHypothesis, policy: ResearchPolicy, total_batch_size: int = 1) -> Tuple[bool, str]:
        reasons = []
        passed = True
        
        # 1. Access Metrics - Use IN-SAMPLE metrics which effectively represent the "New Data" window
        sharpe = h.oos_sharpe # In longitudinal check, the "new data" run result is put into oos slots by convention or we need a new struct?
                              # Let's assume the longitudinal checker constructs a RankedHypothesis 
                              # where oos_sharpe is the performance on the *new* data.
                              
        # 2. Check Sharpe Decay (if enabled)
        # We can also check against absolute minimums for staying active.
        # e.g. if Sharpe drops below 0.5 (even if it was 2.0), it's bad.
        
        if sharpe < (policy.promotion_min_sharpe * 0.75):
             passed = False
             reasons.append(f"Sharpe {sharpe:.2f} < Maintenance Threshold {(policy.promotion_min_sharpe * 0.75):.2f}")

        # 3. Check Drawdown
        if h.oos_max_drawdown > (policy.promotion_max_drawdown * 1.5):
            passed = False
            reasons.append(f"Drawdown {h.oos_max_drawdown:.2f}% > Maintenance Threshold {(policy.promotion_max_drawdown * 1.5):.2f}%")
            
        if not passed:
            return False, f"DECAYED: {'; '.join(reasons)}"

        return True, "Passed Maintenance Check"
