"""
Tests for the main ReplayEngine.
"""

from datetime import datetime

from clock.clock import Clock
from data.bar_iterator import BarIterator
from data.market_loader import MarketDataLoader
from engine.decision_queue import DecisionQueue
from engine.replay_engine import ReplayEngine
from hypotheses.examples.always_long import AlwaysLongHypothesis
from execution.simulator import ExecutionSimulator
from execution.cost_model import CostModel
from state.market_state import MarketState
from state.position_state import PositionState


def test_deterministic_replay():
    """Test that replay results are deterministic given same data/seed."""
    bars = MarketDataLoader.create_synthetic_data(
        "TEST", datetime(2020, 1, 1), 100, 100.0, seed=42
    )
    
    def run_replay():
        hypothesis = AlwaysLongHypothesis()
        clock = Clock()
        bar_iterator = BarIterator(bars)
        decision_queue = DecisionQueue(execution_delay_bars=1)
        
        market_state = MarketState()
        position_state = PositionState()
        
        engine = ReplayEngine(
            hypothesis=hypothesis,
            bar_iterator=bar_iterator,
            clock=clock,
            decision_queue=decision_queue,
            market_state=market_state,
            position_state=position_state
        )
        return engine.run()
    
    result1 = run_replay()
    result2 = run_replay()
    
    assert result1["bars_processed"] == result2["bars_processed"]
    assert result1["decisions_made"] == result2["decisions_made"]
    assert result1["end_time"] == result2["end_time"]


def test_no_look_ahead_market_state():
    """Test that hypothesis cannot see future data."""
    bars = MarketDataLoader.create_synthetic_data(
        "TEST", datetime(2020, 1, 1), 10, 100.0, seed=42
    )
    
    from hypotheses.base import Hypothesis
    
    class PeekCheckHypothesis(Hypothesis):
        def __init__(self):
            # No super().__init__ needed for abstract base? 
            # Check base class definition. It's ABC.
            pass
            
        @property
        def hypothesis_id(self): return "peek_check"
        
        @property
        def parameters(self): return {}
        
        def on_bar(self, market_state, position_state, clock):
            # Market state should only contain data up to current time
            # Since market_state is list of bars, check the last one
            if len(market_state.get_history()) > 0:
                last_bar = market_state.get_history()[-1]
                assert last_bar.timestamp <= clock.now()
            return None

    hypothesis = PeekCheckHypothesis()
    clock = Clock()
    bar_iterator = BarIterator(bars)
    decision_queue = DecisionQueue()
    
    engine = ReplayEngine(hypothesis, bar_iterator, clock, decision_queue)
    engine.run()


def test_execution_delay_enforcement():
    """Test that trades are executed after the specified delay."""
    bars = MarketDataLoader.create_synthetic_data(
        "TEST", datetime(2020, 1, 1), 20, 100.0, seed=42
    )
    
    hypothesis = AlwaysLongHypothesis()
    clock = Clock()
    bar_iterator = BarIterator(bars)
    decision_queue = DecisionQueue(execution_delay_bars=2)  # Delay 2 bars
    cost_model = CostModel(0, 0)
    executor = ExecutionSimulator(cost_model, 100000)
    
    market_state = MarketState()
    position_state = PositionState()
    
    execution_times = []
    
    def on_execution(decisions, bar, index, mkt, pos):
        executor.execute_decisions(decisions, bar, pos)
        execution_times.append(index)
        
    engine = ReplayEngine(
        hypothesis,
        bar_iterator,
        clock,
        decision_queue,
        execution_delay_bars=2,
        market_state=market_state,
        position_state=position_state
    )
    stats = engine.run(on_execution_callback=on_execution)
    
    # AlwaysLong decides on bar 0.
    # Intent enqueued on bar 0.
    # Delay 2 means execute when current_index >= decision_index + 2
    # So execute on bar 2.
    assert 2 in execution_times
    assert stats["executions_triggered"] > 0
    
    # Verify capital/trades using passed state/executor
    trades = executor.get_completed_trades()
    assert len(trades) > 0


def test_market_state_lookback_window():
    """Test that market state respects lookback window."""
    bars = MarketDataLoader.create_synthetic_data(
        "TEST", datetime(2020, 1, 1), 50, 100.0, seed=42
    )
    
    # Lookback 10
    market_state = MarketState(lookback_window=10)
    
    hypothesis = AlwaysLongHypothesis()
    clock = Clock()
    bar_iterator = BarIterator(bars)
    decision_queue = DecisionQueue()
    
    engine = ReplayEngine(
        hypothesis, bar_iterator, clock, decision_queue, 
        market_state=market_state
    )
    engine.run()
    
    history = market_state.get_history()
    assert len(history) <= 10
