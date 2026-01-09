
import numpy as np
from typing import Dict, Any
from batch.models import AggregatedHypothesisResult, GuardrailStatus

def aggregate_results(hypothesis_id: str, run_output: Dict[str, Any]) -> AggregatedHypothesisResult:
    """
    Aggregates OOS metrics per hypothesis from the orchestrator run output.
    """
    if run_output.get("mode") != "WALK_FORWARD":
         # Fallback for single pass? PRD mandates Walk-Forward for batch runs.
         # But maybe we should handle it gracefully or raise error. 
         # PRD: "Walk-forward evaluation is mandatory for batch runs"
         # We will assume caller ensures this, but let's handle if it's missing.
         raise ValueError("Batch execution requires Walk-Forward mode.")

    windows = run_output.get("windows", [])
    if not windows:
        return AggregatedHypothesisResult(
            hypothesis_id=hypothesis_id,
            oos_mean_return=0.0,
            oos_median_return=0.0,
            oos_sharpe=0.0,
            oos_max_drawdown=0.0,
            oos_alpha=0.0,
            oos_beta=0.0,
            oos_ir=0.0,
            profit_factor=0.0,
            profitable_window_ratio=0.0,
            regime_coverage_count=0,
            decay_detected=False,
            guardrail_status=GuardrailStatus.FAIL # Fail if no windows
        )

    # Extract OOS (test) metrics
    oos_metrics_list = [w["test_metrics"] for w in windows]
    
    # --- Aggregate Metrics ---
    # We can average Sharpe, or re-compute? 
    # Usually Sharpe is not additive. But "Avg OOS Sharpe" is a common metric.
    # PRD doesn't specify aggregation method details other than "Computed... using OOS windows only".
    # Let's assume arithmetic mean for returns and Sharpe for now, 
    # or better: concatenate all OOS equity curves and compute global OOS metrics?
    # PRD says "AggregatedHypothesisResult... oos_mean_return... oos_sharpe". 
    # Interpretation: Mean of the window metrics is safer for "stability" than global concatenation which might mask bad windows.
    # Actually, "oos_mean_return" probably implies the mean of the returns distribution OF THE AGGREGATED SERIES?
    # Or the mean of the window returns?
    # Let's stick to averaging window metrics for simplicity and robustness against outliers in one window.
    
    # Metric keys from EvaluationMetrics (inference from run_evaluation.py): 
    # 'sharpe_ratio', 'total_return', 'max_drawdown', 'profit_factor'
    
    sharpes = [m.get("sharpe_ratio", 0.0) for m in oos_metrics_list]
    returns = [m.get("total_return", 0.0) for m in oos_metrics_list]
    drawdowns = [m.get("max_drawdown", 0.0) for m in oos_metrics_list] # Usually positive number in metrics? or negative?
                                                                       # Standard is usually positive percentage for DD.
    profit_factors = [m.get("profit_factor", 0.0) for m in oos_metrics_list]
    
    oos_mean_return = float(np.mean(returns))
    oos_median_return = float(np.median(returns))
    oos_sharpe = float(np.mean(sharpes))
    oos_max_drawdown = float(np.max(drawdowns)) # Max of max drawdowns (conservative)
    avg_profit_factor = float(np.mean(profit_factors))
    
    alphas = [m.get("alpha", 0.0) for m in oos_metrics_list]
    betas = [m.get("beta", 0.0) for m in oos_metrics_list]
    irs = [m.get("information_ratio", 0.0) for m in oos_metrics_list]

    oos_alpha = float(np.mean(alphas))
    oos_beta = float(np.mean(betas))
    oos_ir = float(np.mean(irs))
    
    # Profitable Window Ratio
    profitable_windows = sum(1 for r in returns if r > 0)
    profitable_window_ratio = profitable_windows / len(windows) if windows else 0.0
    
    # Regime Coverage
    regimes = set()
    for w in windows:
        if "market_regime" in w and w["market_regime"]:
             regimes.add(w["market_regime"])
    
    regime_coverage_count = len(regimes)
    
    # Decay Detected
    decay_detected = any(w.get("decay", {}).result_tag == "FAIL" for w in windows if w.get("decay"))
    
    # Guardrails
    guardrail_status = GuardrailStatus.PASS
    if oos_sharpe < -10.0:
        guardrail_status = GuardrailStatus.FAIL
        
    return AggregatedHypothesisResult(
        hypothesis_id=hypothesis_id,
        oos_mean_return=oos_mean_return,
        oos_median_return=oos_median_return,
        oos_sharpe=oos_sharpe,
        oos_max_drawdown=oos_max_drawdown,
        oos_alpha=oos_alpha,
        oos_beta=oos_beta,
        oos_ir=oos_ir,
        profit_factor=avg_profit_factor,
        profitable_window_ratio=profitable_window_ratio,
        regime_coverage_count=regime_coverage_count,
        decay_detected=decay_detected,
        guardrail_status=guardrail_status
    )
