import pytest
from datetime import datetime, timedelta
from portfolio.meta_engine import MetaPortfolioEngine
from portfolio.ensemble import Ensemble
from portfolio.weighting import EqualWeighting
from hypotheses.base import Hypothesis, TradeIntent, IntentType
from state.position_state import PositionSide
from data.schemas import Bar
from execution.cost_model import CostModel
from storage.repositories import EvaluationRepository

# --- Mocks ---

class LongMock(Hypothesis):
    @property
    def hypothesis_id(self): return "long"
    @property
    def parameters(self): return {}
    def on_bar(self, ms, ps, c):
        if not ps.has_position:
            return TradeIntent(type=IntentType.BUY, size=1.0)
        return None

class ShortMock(Hypothesis):
    @property
    def hypothesis_id(self): return "short"
    @property
    def parameters(self): return {}
    def on_bar(self, ms, ps, c):
        if not ps.has_position:
            return TradeIntent(type=IntentType.SELL, size=1.0)
        return None

@pytest.fixture
def mock_repo(tmp_path):
    db_path = tmp_path / "test_meta.db"
    return EvaluationRepository(str(db_path))

@pytest.fixture
def mock_bars():
    start = datetime(2023, 1, 1)
    bars = []
    # Flat price
    for i in range(10):
        bars.append(Bar(
            timestamp=start + timedelta(days=i),
            open=100.0, high=101.0, low=99.0, close=100.0, volume=1000
        ))
    return bars

def test_meta_netting(mock_repo, mock_bars):
    """
    Verify that if H1 is Long and H2 is Short (Equal Weight),
    Meta Portfolio stays Flat (Net Exposure = 0).
    """
    h_long = LongMock()
    h_short = ShortMock()
    
    ensemble = Ensemble(
        hypotheses=[h_long, h_short],
        weighting_strategy=EqualWeighting(),
        repo=mock_repo,
        policy_id="TEST"
    )
    
    cost_model = CostModel(0.0, 0.0)
    engine = MetaPortfolioEngine(ensemble, 100000.0, cost_model)
    
    history = engine.run(mock_bars)
    
    # Check Result
    final_state = history[-1]
    
    # Shadow allocations should simulate trades
    alloc_long = final_state.allocations["long"]
    alloc_short = final_state.allocations["short"]
    
    # Long should be invested (virtually)
    assert alloc_long.current_position is not None
    assert alloc_long.current_position.side == PositionSide.LONG
    
    # Short should be invested (virtually)
    assert alloc_short.current_position is not None
    assert alloc_short.current_position.side == PositionSide.SHORT
    
    # Meta Portfolio should be Flat!
    # Net Exposure = (0.5 * 1.0) + (0.5 * -1.0) = 0.0
    alloc_meta = final_state.allocations["META_PORTFOLIO"]
    assert alloc_meta.current_position is None # OR size 0
    if alloc_meta.current_position:
         assert alloc_meta.current_position.size == 0
         
    # Capital should remain 100,000 (minus maybe 0 cost)
    assert final_state.total_capital == 100000.0

def test_meta_sizing(mock_repo, mock_bars):
    """
    Verify single Long hypothesis results in ~100% exposure.
    """
    h_long = LongMock()
    
    ensemble = Ensemble(
        hypotheses=[h_long],
        weighting_strategy=EqualWeighting(),
        repo=mock_repo,
        policy_id="TEST"
    )
    
    cost_model = CostModel(0.0, 0.0)
    engine = MetaPortfolioEngine(ensemble, 100000.0, cost_model)
    
    history = engine.run(mock_bars)
    final_state = history[-1]
    
    alloc_meta = final_state.allocations["META_PORTFOLIO"]
    assert alloc_meta.current_position is not None
    assert alloc_meta.current_position.side == PositionSide.LONG
    
    # Size should be ~1000 units (100k / 100 price)
    assert 990 <= alloc_meta.current_position.size <= 1010
