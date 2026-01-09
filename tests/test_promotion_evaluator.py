from batch.models import RankedHypothesis, GuardrailStatus
from evaluation.policy import ResearchPolicy, EvaluationMode
from promotion.evaluator import PromotionEvaluator

def _mock_policy():
    return ResearchPolicy(
        policy_id="TEST_POL", description="Test", evaluation_mode=EvaluationMode.SINGLE_PASS, # Ignored
        train_window_bars=0, test_window_bars=0, step_size_bars=0, execution_delay_bars=0,
        transaction_cost_bps=0, slippage_bps=0, min_trades=10, min_regimes=1, max_sharpe_decay=0.5,
        promotion_min_sharpe=0.5, promotion_min_profit_factor=1.2, promotion_min_return_pct=0.0,
        promotion_max_drawdown=20.0, promotion_min_trades=15
    )

def test_evaluator_flow():
    policy = _mock_policy()
    evaluator = PromotionEvaluator(policy, "batch_1")
    
    # H1: Perfect -> PROMOTED
    h1 = RankedHypothesis(
        batch_id="b1", hypothesis_id="h1", research_score=0, rank=1,
        oos_sharpe=1.0, oos_mean_return=5.0, oos_max_drawdown=5.0,
        decay_flag=False, guardrail_status=GuardrailStatus.PASS
    )
    
    # H2: Fails Sharpe -> EVALUATED
    h2 = RankedHypothesis(
        batch_id="b1", hypothesis_id="h2", research_score=0, rank=2,
        oos_sharpe=0.1, oos_mean_return=5.0, oos_max_drawdown=5.0,
        decay_flag=False, guardrail_status=GuardrailStatus.PASS
    )
    
    decisions = evaluator.evaluate([h1, h2])
    
    assert len(decisions) == 2
    
    assert decisions[0].hypothesis_id == "h1"
    assert decisions[0].decision.value == "PROMOTED"
    
    assert decisions[1].hypothesis_id == "h2"
    assert decisions[1].decision.value == "EVALUATED"
    assert any("Sharpe" in r for r in decisions[1].reasons)
