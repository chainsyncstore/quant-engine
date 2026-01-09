"""
Benchmark Filter Unit Test

Verifies that PromotionEvaluator correctly applies BenchmarkFilterRule.
"""
import sys
sys.path.insert(0, '.')

from promotion.evaluator import PromotionEvaluator, HypothesisStatus
from batch.models import RankedHypothesis, GuardrailStatus
from evaluation.policy import ResearchPolicy, EvaluationMode

def create_mock_hypothesis(hid, alpha, beta, ir):
    return RankedHypothesis(
        batch_id="test_batch",
        hypothesis_id=hid,
        research_score=0.0,
        rank=1,
        oos_sharpe=1.0, # Good Sharpe
        oos_mean_return=10.0,
        oos_max_drawdown=5.0,
        oos_alpha=alpha,
        oos_beta=beta,
        oos_ir=ir,
        decay_flag=False,
        guardrail_status=GuardrailStatus.PASS
    )

def main():
    print("="*80)
    print("BENCHMARK FILTER UNIT TEST")
    print("="*80)
    
    # Policy with Filters
    policy = ResearchPolicy(
        policy_id="FILTER_POLICY",
        description="Strict filters",
        evaluation_mode=EvaluationMode.SINGLE_PASS,
        train_window_bars=0, test_window_bars=0, step_size_bars=0,
        execution_delay_bars=0, transaction_cost_bps=0, slippage_bps=0,
        min_trades=0, min_regimes=0, max_sharpe_decay=0.0,
        
        # FILTERS
        promotion_min_alpha=0.05, # Require 5% Alpha
        promotion_max_beta=1.5,   # Max Beta 1.5
        promotion_min_information_ratio=0.5 # Min IR 0.5
    )
    
    evaluator = PromotionEvaluator(policy, "test_batch")
    
    scenarios = [
        # HID, Alpha, Beta, IR, Expected
        ("PASS_CASE",   0.10, 1.0, 1.0, True),
        ("FAIL_ALPHA",  0.00, 1.0, 1.0, False), # Alpha 0 < 0.05
        ("FAIL_BETA",   0.10, 2.0, 1.0, False), # Beta 2 > 1.5
        ("FAIL_IR",     0.10, 1.0, 0.2, False), # IR 0.2 < 0.5
    ]
    
    failures = []
    
    for hid, alpha, beta, ir, expected_pass in scenarios:
        h = create_mock_hypothesis(hid, alpha, beta, ir)
        decisions = evaluator.evaluate([h])
        decision = decisions[0]
        
        passed = (decision.decision == HypothesisStatus.PROMOTED)
        
        status_str = "PROMOTED" if passed else "REJECTED"
        reasons = "; ".join(decision.reasons)
        
        print(f"Scenario {hid:<12} | Alpha={alpha:>5.2f} Beta={beta:>5.2f} IR={ir:>5.2f} | -> {status_str:<10} | {reasons}")
        
        if passed != expected_pass:
            failures.append(f"{hid} failed: Expected {expected_pass}, got {passed}")
            
    if failures:
        print("\n✗ FAIL: Verification failed.")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print("\n✓ PASS: All benchmark filters verified.")
        sys.exit(0)

if __name__ == "__main__":
    main()
