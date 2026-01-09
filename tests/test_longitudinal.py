
import pytest
from datetime import datetime, timedelta
from typing import List

from evaluation.longitudinal import LongitudinalTracker
from evaluation.policy import ResearchPolicy
from storage.repositories import EvaluationRepository
from promotion.models import HypothesisStatus
from config.settings import Settings
from hypotheses.registry import registry
from hypotheses.base import Hypothesis, TradeIntent, IntentType
from state.market_state import MarketState
from state.position_state import PositionState
from clock.clock import Clock
from data.schemas import Bar

class TradingHypothesis(Hypothesis):
    """Hypothesis that enters and exits to generate trades."""
    @property
    def hypothesis_id(self) -> str:
        return "trading_test"
    
    @property
    def parameters(self) -> dict:
        return {}
        
    def __init__(self, **kwargs):
        pass

    def on_bar(self, market_state: MarketState, position_state: PositionState, clock: Clock) -> TradeIntent:
        # Buy if flat, Sell if long
        if not position_state.has_position:
            return TradeIntent(type=IntentType.BUY, size=1.0)
        else:
            return TradeIntent(type=IntentType.CLOSE, size=1.0)

# --- Mock Data ---

def create_mock_bars(start_date: datetime, count: int, price_trajectory: List[float]) -> List[Bar]:
    bars = []
    current_date = start_date
    for price in price_trajectory:
        bars.append(Bar(
            timestamp=current_date,
            open=price,
            high=price,
            low=price,
            close=price,
            volume=1000
        ))
        current_date += timedelta(days=1)
    return bars

@pytest.fixture
def temp_db(tmp_path):
    db_path = tmp_path / "test_longitudinal.db"
    return str(db_path)

@pytest.fixture
def repo(temp_db):
    return EvaluationRepository(temp_db)

@pytest.fixture
def policy():
    return ResearchPolicy(
        policy_id="TEST_POLICY",
        description="Test Policy",
        evaluation_mode="SINGLE_PASS",
        train_window_bars=100,
        test_window_bars=20,
        step_size_bars=20,
        execution_delay_bars=0,
        transaction_cost_bps=0.0,
        slippage_bps=0.0,
        min_trades=5,
        min_regimes=1,
        max_sharpe_decay=0.5,
        promotion_min_sharpe=1.0,
        promotion_min_return_pct=5.0,
        promotion_max_drawdown=10.0
    )

@pytest.fixture
def settings(temp_db, tmp_path):
    # Mock settings to point to temp DB
    return Settings(
        database_path=str(temp_db),
        starting_capital=10000.0,
        lookback_window=10
    )

def test_longitudinal_decay_check_maintained(repo, policy, settings, monkeypatch, tmp_path):
    """Test checks for decay but finds good performance, maintaining PROMOTED status."""

    
    # 1. Setup Data
    # History: Good performance
    # New Data: Good performance (Price goes up)
    
    start_date = datetime(2023, 1, 1)
    
    # Create History (Evaluation 1)
    
    # Create History (Evaluation 1)
    h_cls = TradingHypothesis
    # Ensure registered (it is by default usually, but let's be safe if registry is global)
    if not registry.is_registered(h_cls().hypothesis_id):
        registry.register(h_cls)
        
    hid = h_cls().hypothesis_id
    
    # Store Hypothesis
    repo.store_hypothesis(hid, h_cls().parameters)
    
    # Store Initial Evaluation (Good) -> PROMOTED
    repo.store_evaluation(
        hypothesis_id=hid,
        parameters=h_cls().parameters,
        market_symbol="TEST",
        test_start_timestamp=start_date,
        test_end_timestamp=start_date + timedelta(days=10),
        metrics={"sharpe_ratio": 2.0},
        benchmark_metrics={},
        assumed_costs_bps=0,
        initial_capital=10000,
        final_equity=11000,
        bars_processed=10,
        policy_id=policy.policy_id
    )
    
    repo.store_hypothesis_status(
        hid,
        HypothesisStatus.PROMOTED.value,
        policy_id=policy.policy_id,
        rationale=["Initial promotion"]
    )
    
    # Write New Data to CSV for MarketLoader (since tracker loads from CSV)
    # Price goes UP -> Good Sharpe
    new_data_start = start_date + timedelta(days=11)
    prices = [100 + i for i in range(30)] # Steady up trend
    bars = create_mock_bars(new_data_start, 30, prices)
    
    import pandas as pd
    df = pd.DataFrame([vars(b) for b in bars])
    # Reorder for CSV
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    df.to_csv(tmp_path / "mock_data.csv", index=False)
    
    # 2. Run Tracker
    # Patch settings in tracker
    tracker = LongitudinalTracker(repo, policy, settings)
    # monkeypatch.setattr(tracker, 'settings', settings)
    
    # Run with current time after new data
    results = tracker.run_checks(
        data_path=str(tmp_path / "mock_data.csv"),
        symbol="TEST", 
        current_time=new_data_start + timedelta(days=30)
    )
    
    # 3. Assertions
    assert len(results) == 1
    assert results[0]['hypothesis_id'] == hid
    assert results[0]['status'] == "MAINTAINED"
    
    # Check DB status
    current_status = repo.get_hypotheses_by_status("PROMOTED")
    assert hid in current_status
    
    decayed_status = repo.get_hypotheses_by_status("DECAYED")
    assert hid not in decayed_status

def test_longitudinal_decay_check_decayed(repo, policy, settings, monkeypatch, tmp_path):
    """Test checks for decay and finds bad performance, downgrading to DECAYED."""
    
    start_date = datetime(2023, 1, 1)
    start_date = datetime(2023, 1, 1)
    h_cls = TradingHypothesis
    hid = h_cls().hypothesis_id
    
    repo.store_hypothesis(hid, h_cls().parameters)
    repo.store_evaluation(
        hypothesis_id=hid,
        parameters=h_cls().parameters,
        market_symbol="TEST",
        test_start_timestamp=start_date,
        test_end_timestamp=start_date + timedelta(days=10),
        metrics={"sharpe_ratio": 2.0},
        benchmark_metrics={},
        assumed_costs_bps=0,
        initial_capital=10000,
        final_equity=11000,
        bars_processed=10,
        policy_id=policy.policy_id
    )
    repo.store_hypothesis_status(
        hid,
        HypothesisStatus.PROMOTED.value,
        policy_id=policy.policy_id,
        rationale=["Initial promotion"]
    )
    
    # Write New Data causing DECAY
    # Price Choppy/Down -> Bad Sharpe
    new_data_start = start_date + timedelta(days=11)
    # Sawtooth pattern or drop
    prices = [100, 90, 100, 90, 80, 70, 60] + [60]*20 # Big drop then flat
    bars = create_mock_bars(new_data_start, len(prices), prices)
    
    import pandas as pd
    df = pd.DataFrame([vars(b) for b in bars])
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    df.to_csv(tmp_path / "mock_data.csv", index=False)
    
    tracker = LongitudinalTracker(repo, policy, settings)
    # monkeypatch.setattr(tracker, 'settings', settings)
    
    results = tracker.run_checks(
        data_path=str(tmp_path / "mock_data.csv"),
        symbol="TEST", 
        current_time=new_data_start + timedelta(days=len(prices))
    )
    
    assert len(results) == 1
    assert results[0]['status'] == "DECAYED"
    assert "DECAY DETECTED" in results[0]['reason'] or "Sharpe" in results[0]['reason']
    
    # Check DB
    promoted = repo.get_hypotheses_by_status("PROMOTED")
    assert hid not in promoted
    
    decayed = repo.get_hypotheses_by_status("DECAYED")
    assert hid in decayed

