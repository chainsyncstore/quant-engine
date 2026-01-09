from typing import List
from batch.models import RankedHypothesis
from evaluation.policy import ResearchPolicy
from promotion.models import PromotionDecision, HypothesisStatus
from promotion.rules import (
    AbsolutePerformanceRule,
    RobustnessRule,
    RelativeStandingRule,
    BenchmarkFilterRule,
    PromotionRule,
)

class PromotionEvaluator:
    def __init__(self, policy: ResearchPolicy, batch_id: str):
        self.policy = policy
        self.batch_id = batch_id
        self.rules: List[PromotionRule] = [
            RobustnessRule(),
            BenchmarkFilterRule(),
            AbsolutePerformanceRule(),
            RelativeStandingRule(top_percentile=0.5) # Default to top 50%? PRD "Optional but Recommended"
        ]
        
    def evaluate(self, ranked_hypotheses: List[RankedHypothesis]) -> List[PromotionDecision]:
        decisions = []
        total_count = len(ranked_hypotheses)
        
        for h in ranked_hypotheses:
            all_passed = True
            reasons = []
            
            for rule in self.rules:
                passed, reason = rule.evaluate(h, self.policy, total_count)
                if not passed:
                    all_passed = False
                    reasons.append(reason)
                    # We might want to break early, or collect all failures.
                    # PRD example reasons: "OOS Sharpe...".
                    # Let's collect explicit reasons for passing too?
                    # PRD usage implies reasons are why it WAS promoted or NOT.
                    break 
                else:
                    reasons.append(reason) # Track success reasons too?
            
            status = HypothesisStatus.PROMOTED if all_passed else HypothesisStatus.EVALUATED
            
            decisions.append(PromotionDecision(
                hypothesis_id=h.hypothesis_id,
                batch_id=self.batch_id,
                decision=status,
                reasons=reasons
            ))
            
        return decisions
