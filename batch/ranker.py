from typing import List
import math
from batch.models import AggregatedHypothesisResult, RankedHypothesis, GuardrailStatus

def _min_max_scale(values: List[float]) -> List[float]:
    min_val = min(values)
    max_val = max(values)
    if math.isclose(max_val, min_val):
        return [0.5 for _ in values] # Logic for equal values? Or 0? 0.5 is neutral.
    return [(v - min_val) / (max_val - min_val) for v in values]

def rank_hypotheses(batch_id: str, results: List[AggregatedHypothesisResult]) -> List[RankedHypothesis]:
    """
    Ranks hypotheses deterministically based on OOS metrics.
    """
    # 1. Filter ineligible hypotheses
    eligible_results = [r for r in results if r.guardrail_status == GuardrailStatus.PASS]
    failed_results = [r for r in results if r.guardrail_status == GuardrailStatus.FAIL]
    
    if not eligible_results:
        # Return all as unranked/failed
        return [
            RankedHypothesis(
                batch_id=batch_id,
                hypothesis_id=r.hypothesis_id,
                research_score=-float('inf'),
                rank=999,
                oos_sharpe=r.oos_sharpe,
                oos_mean_return=r.oos_mean_return,
                oos_max_drawdown=r.oos_max_drawdown,
                oos_alpha=r.oos_alpha,
                oos_beta=r.oos_beta,
                oos_ir=r.oos_ir,
                decay_flag=r.decay_detected,
                guardrail_status=GuardrailStatus.FAIL
            ) for r in results
        ]

    # 2. Extract metrics for normalization
    sharpes = [r.oos_sharpe for r in eligible_results]
    means = [r.oos_mean_return for r in eligible_results]
    profit_factors = [r.profit_factor for r in eligible_results]
    regimes = [float(r.regime_coverage_count) for r in eligible_results]
    
    norm_sharpes = _min_max_scale(sharpes)
    norm_means = _min_max_scale(means)
    norm_profit_factors = _min_max_scale(profit_factors)
    norm_regimes = _min_max_scale(regimes)
    
    ranked_list = []
    
    # 3. Compute Scores
    for i, r in enumerate(eligible_results):
        decay_penalty = 1.0 if r.decay_detected else 0.0
        
        # Formula
        score = (
            0.30 * norm_sharpes[i] +
            0.20 * norm_means[i] +
            0.15 * r.profitable_window_ratio + # Already 0-1
            0.15 * norm_profit_factors[i] +
            0.10 * norm_regimes[i] -
            0.10 * decay_penalty
        )
        
        ranked_list.append(RankedHypothesis(
            batch_id=batch_id,
            hypothesis_id=r.hypothesis_id,
            research_score=score,
            rank=0, # Placeholder
            oos_sharpe=r.oos_sharpe,
            oos_mean_return=r.oos_mean_return,
            oos_max_drawdown=r.oos_max_drawdown,
            oos_alpha=r.oos_alpha,
            oos_beta=r.oos_beta,
            oos_ir=r.oos_ir,
            decay_flag=r.decay_detected,
            guardrail_status=GuardrailStatus.PASS
        ))
        
    # 4. Sort and Assign Ranks
    # Sort distinctively by score (desc), then ID (asc) for determinism
    ranked_list.sort(key=lambda x: (-x.research_score, x.hypothesis_id))
    
    for i, rh in enumerate(ranked_list):
        rh.rank = i + 1
        
    # 5. Append failed
    for r in failed_results:
        ranked_list.append(RankedHypothesis(
            batch_id=batch_id,
            hypothesis_id=r.hypothesis_id,
            research_score=-float('inf'), # Minimum score
            rank=999,
            oos_sharpe=r.oos_sharpe,
            oos_mean_return=r.oos_mean_return,
            oos_max_drawdown=r.oos_max_drawdown,
            oos_alpha=r.oos_alpha,
            oos_beta=r.oos_beta,
            oos_ir=r.oos_ir,
            decay_flag=r.decay_detected,
            guardrail_status=GuardrailStatus.FAIL
        ))
        
    return ranked_list
