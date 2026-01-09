from abc import ABC, abstractmethod
from typing import Tuple
from batch.models import RankedHypothesis, GuardrailStatus
from evaluation.policy import ResearchPolicy

class PromotionRule(ABC):
    @abstractmethod
    def evaluate(self, hypothesis: RankedHypothesis, policy: ResearchPolicy, total_batch_size: int) -> Tuple[bool, str]:
        """
        Evaluate rule. Returns (Passed, Reason/Failure Message).
        """
        pass

class AbsolutePerformanceRule(PromotionRule):
    def evaluate(self, h: RankedHypothesis, policy: ResearchPolicy, total_batch_size: int) -> Tuple[bool, str]:
        reasons = []
        passed = True
        
        if h.oos_sharpe < policy.promotion_min_sharpe:
            passed = False
            reasons.append(f"Sharpe {h.oos_sharpe:.2f} < {policy.promotion_min_sharpe}")
            
        if h.oos_mean_return < policy.promotion_min_return_pct:
            passed = False
            reasons.append(f"Return {h.oos_mean_return:.2f}% < {policy.promotion_min_return_pct}%")
            
        if h.oos_max_drawdown > policy.promotion_max_drawdown:
            passed = False
            reasons.append(f"Drawdown {h.oos_max_drawdown:.2f}% > {policy.promotion_max_drawdown}%")
            
        if not passed:
            return False, "; ".join(reasons)
            
        return True, "Passed Absolute Performance"

class RobustnessRule(PromotionRule):
    def evaluate(self, h: RankedHypothesis, policy: ResearchPolicy, total_batch_size: int) -> Tuple[bool, str]:
        if h.guardrail_status == GuardrailStatus.FAIL:
            return False, "Failed Guardrails"
        
        if h.decay_flag:
            return False, "Decay Detected"
            
        return True, "Passed Robustness"

class RelativeStandingRule(PromotionRule):
    """
    Optional: Must be in top N% of batch.
    """
    def __init__(self, top_percentile: float = 0.5):
        self.top_percentile = top_percentile # e.g. 0.5 means top 50%

    def evaluate(self, h: RankedHypothesis, policy: ResearchPolicy, total_batch_size: int) -> Tuple[bool, str]:
        # h.rank is 1-based (1 is best)
        # Percentile rank: (total - rank + 1) / total
        # e.g. Rank 1 of 10: (10-1+1)/10 = 1.0 (100th percentile)
        # e.g. Rank 10 of 10: (10-10+1)/10 = 0.1 (10th percentile)
        
        # Or simply: rank <= total * (percentile)
        # Validation: top 10% of 10 items = rank <= 1.
        
        # User PRD says: "Top N (e.g. top 10%) OR score >= percentile threshold"
        # Let's use simple rank percentile.
        
        cutoff_rank = max(1, int(total_batch_size * self.top_percentile))
        
        if h.rank > cutoff_rank:
             return False, f"Rank {h.rank} not in top {int(self.top_percentile*100)}% (Cutoff: {cutoff_rank})"
             
        return True, f"Rank {h.rank} in top {int(self.top_percentile*100)}%"

class BenchmarkFilterRule(PromotionRule):
    """
    Enforces risk-adjusted return standards relative to benchmark.
    - Alpha > Min
    - Beta < Max
    - Information Ratio > Min
    """
    def evaluate(self, h: RankedHypothesis, policy: ResearchPolicy, total_batch_size: int) -> Tuple[bool, str]:
        reasons = []
        passed = True
        
        # Beta Check (Max cap)
        if h.oos_beta > policy.promotion_max_beta:
            passed = False
            reasons.append(f"Beta {h.oos_beta:.2f} > {policy.promotion_max_beta}")
            
        # Alpha Check (Min floor)
        if h.oos_alpha < policy.promotion_min_alpha:
            passed = False
            reasons.append(f"Alpha {h.oos_alpha:.2f} < {policy.promotion_min_alpha}")
            
        # Information Ratio Check (Min floor)
        if h.oos_ir < policy.promotion_min_information_ratio:
            passed = False
            reasons.append(f"IR {h.oos_ir:.2f} < {policy.promotion_min_information_ratio}")
            
        if not passed:
            return False, "; ".join(reasons)
            
        return True, "Passed Benchmark Filters"
