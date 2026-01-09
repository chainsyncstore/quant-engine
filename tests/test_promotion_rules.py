from batch.models import RankedHypothesis, GuardrailStatus
from evaluation.policy import ResearchPolicy, EvaluationMode
from promotion.rules import AbsolutePerformanceRule, RobustnessRule, RelativeStandingRule

def _mock_policy():
    return ResearchPolicy(
        policy_id="TEST_POL", description="Test", evaluation_mode=EvaluationMode.SINGLE_PASS, # Ignored
        train_window_bars=0, test_window_bars=0, step_size_bars=0, execution_delay_bars=0,
        transaction_cost_bps=0, slippage_bps=0, min_trades=10, min_regimes=1, max_sharpe_decay=0.5,
        promotion_min_sharpe=0.5, promotion_min_profit_factor=1.2, promotion_min_return_pct=0.0,
        promotion_max_drawdown=20.0, promotion_min_trades=15
    )

def _mock_hypothesis(sharpe=1.0, ret=5.0, dd=10.0, pf=1.5, status=GuardrailStatus.PASS, decay=False, rank=1):
    return RankedHypothesis(
        batch_id="b1", hypothesis_id="h1", research_score=0, rank=rank,
        oos_sharpe=sharpe, oos_mean_return=ret, oos_max_drawdown=dd,
        decay_flag=decay, guardrail_status=status
    )

def test_absolute_performance_rule():
    rule = AbsolutePerformanceRule()
    policy = _mock_policy()
    
    # Pass
    h_pass = _mock_hypothesis(sharpe=0.6, ret=1.0, dd=10.0, pf=1.3)
    passed, reason = rule.evaluate(h_pass, policy, 10)
    assert passed
    
    # Fail Sharpe
    h_fail_sharpe = _mock_hypothesis(sharpe=0.4, ret=1.0, dd=10.0, pf=1.3)
    passed, reason = rule.evaluate(h_fail_sharpe, policy, 10)
    assert not passed
    assert "Sharpe" in reason
    
    # Fail Return
    h_fail_ret = _mock_hypothesis(sharpe=0.6, ret=-1.0)
    passed, reason = rule.evaluate(h_fail_ret, policy, 10)
    assert not passed
    assert "Return" in reason

def test_robustness_rule():
    rule = RobustnessRule()
    policy = _mock_policy()
    
    # Pass
    h = _mock_hypothesis()
    assert rule.evaluate(h, policy, 10)[0]
    
    # Fail Guardrail
    h_fail = _mock_hypothesis(status=GuardrailStatus.FAIL)
    assert not rule.evaluate(h_fail, policy, 10)[0]
    
    # Fail Decay
    h_decay = _mock_hypothesis(decay=True)
    assert not rule.evaluate(h_decay, policy, 10)[0]

def test_relative_standing_rule():
    rule = RelativeStandingRule(top_percentile=0.2) # Top 20%
    policy = _mock_policy()
    
    # Total 10. Cutoff rank <= 2.
    
    # Rank 1 (Pass)
    h1 = _mock_hypothesis(rank=1)
    passed, r = rule.evaluate(h1, policy, 10)
    assert passed
    
    # Rank 2 (Pass)
    h2 = _mock_hypothesis(rank=2)
    assert rule.evaluate(h2, policy, 10)[0]
    
    # Rank 3 (Fail)
    h3 = _mock_hypothesis(rank=3)
    passed, r = rule.evaluate(h3, policy, 10)
    assert not passed
    assert "not in top" in r
