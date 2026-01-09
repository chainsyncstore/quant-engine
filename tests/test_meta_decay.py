import pytest
from datetime import datetime, timedelta
from portfolio.meta_engine import MetaPortfolioEngine
from portfolio.ensemble import Ensemble
from portfolio.weighting import EqualWeighting
from hypotheses.base import Hypothesis, TradeIntent, IntentType
from data.schemas import Bar
from execution.cost_model import CostModel
from storage.repositories import EvaluationRepository
from promotion.models import HypothesisStatus

# --- Mocks ---

class AlwaysLossMock(Hypothesis):
    """Buys and holds. Market crashes."""
    @property
    def hypothesis_id(self): return "loser"
    @property
    def parameters(self): return {}
    def on_bar(self, ms, ps, c):
        if not ps.has_position:
            return TradeIntent(type=IntentType.BUY, size=1.0)
        return None

@pytest.fixture
def mock_repo(tmp_path):
    db_path = tmp_path / "test_decay.db"
    return EvaluationRepository(str(db_path))

@pytest.fixture
def crash_bars():
    start = datetime(2023, 1, 1)
    bars = []
    # 100 -> 50 in 10 days (50% DD)
    prices = [100 - i*5 for i in range(11)]
    for i, p in enumerate(prices):
        bars.append(Bar(
            timestamp=start + timedelta(days=i),
            open=p, high=p+1, low=p-1, close=p, volume=1000
        ))
    return bars

def test_dynamic_decay(mock_repo, crash_bars):
    """
    Verify that a strategy crashing > 25% is marked DECAYED and weight becomes 0.
    """
    h_loss = AlwaysLossMock()
    
    ensemble = Ensemble(
        hypotheses=[h_loss],
        weighting_strategy=EqualWeighting(),
        repo=mock_repo,
        policy_id="TEST"
    )
    
    # Initially PROMOTED
    assert ensemble.current_statuses["loser"] == HypothesisStatus.PROMOTED
    assert ensemble.weights["loser"] == 1.0
    
    cost_model = CostModel(0.0, 0.0)
    # Check every day
    engine = MetaPortfolioEngine(ensemble, 100000.0, cost_model, decay_check_interval=1)
    
    # Run simulation
    history = engine.run(crash_bars)
    
    # Check Final Status
    assert ensemble.current_statuses["loser"] == HypothesisStatus.DECAYED
    
    # Check Final Weight
    assert ensemble.weights["loser"] == 0.0
    
    # Check Allocations: Meta should be flat eventually?
    # Decay trigger happens bar-by-bar.
    # Once Decayed, weight is 0.
    # Target Exposure = 0.
    # Engine should Close position.
    
    final = history[-1]
    alloc_meta = final.allocations["META_PORTFOLIO"]
    
    # Position should be closed (None or Size 0)
    if alloc_meta.current_position:
        assert alloc_meta.current_position.size == 0
