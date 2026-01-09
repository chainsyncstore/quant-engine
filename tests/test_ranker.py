
from batch.ranker import rank_hypotheses
from batch.models import AggregatedHypothesisResult, GuardrailStatus

def _make_res(hid, sharpe, ret, pf=1.5, regime=1, decay=False, status=GuardrailStatus.PASS):
    return AggregatedHypothesisResult(
        hypothesis_id=hid,
        oos_sharpe=sharpe,
        oos_mean_return=ret,
        oos_median_return=ret,
        oos_max_drawdown=0.0,
        profit_factor=pf,
        profitable_window_ratio=1.0, # Fixed
        regime_coverage_count=regime,
        decay_detected=decay,
        guardrail_status=status
    )

def test_ranker_basic():
    # H1: Best
    # H2: Medium
    # H3: Worst
    
    r1 = _make_res("h1", sharpe=2.0, ret=20)
    r2 = _make_res("h2", sharpe=1.0, ret=10)
    r3 = _make_res("h3", sharpe=0.5, ret=5)
    
    rankings = rank_hypotheses("b1", [r1, r2, r3]) # Order shouldn't matter for output logic, but lets shuffle
    
    assert rankings[0].hypothesis_id == "h1"
    assert rankings[0].rank == 1
    assert rankings[1].hypothesis_id == "h2"
    assert rankings[2].hypothesis_id == "h3"
    
    # Check normalized scores
    # Max sharpe=2.0, Min=0.5
    # Max ret=20, Min=5
    # H1 norm sharpe = 1.0, norm ret = 1.0
    # Score = 0.3*1 + 0.2*1 + ...
    
def test_ranker_guardrail_exclusion():
    r1 = _make_res("h1", sharpe=2.0, ret=20)
    r2 = _make_res("h2", sharpe=1.0, ret=10, status=GuardrailStatus.FAIL)
    
    rankings = rank_hypotheses("b1", [r2, r1])
    
    assert rankings[0].hypothesis_id == "h1"
    assert rankings[0].rank == 1
    
    assert rankings[1].hypothesis_id == "h2"
    assert rankings[1].rank == 999
    assert rankings[1].research_score == -float('inf')

def test_ranker_decay_penalty():
    # Two identical results, but one has decay
    r1 = _make_res("h1", sharpe=1.0, ret=10, decay=False)
    r2 = _make_res("h2", sharpe=1.0, ret=10, decay=True)
    
    rankings = rank_hypotheses("b1", [r1, r2])
    
    assert rankings[0].hypothesis_id == "h1"
    assert rankings[1].hypothesis_id == "h2"
    assert rankings[0].research_score > rankings[1].research_score
    # Diff should be exactly 0.10
    assert abs(rankings[0].research_score - rankings[1].research_score - 0.10) < 1e-6
