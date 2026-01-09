"""
Sanity tests for basic system validation.

Tests:
- Always-long hypothesis produces expected behavior
- Zero-cost environment matches buy-and-hold
- Basic metrics calculation
"""

from datetime import datetime

import pytest

from clock.clock import Clock
from data.bar_iterator import BarIterator
from data.market_loader import MarketDataLoader
from engine.decision_queue import DecisionQueue
from engine.replay_engine import ReplayEngine
from evaluation.benchmark import BenchmarkCalculator
from evaluation.metrics import EvaluationMetrics
from execution.cost_model import CostModel
from execution.simulator import ExecutionSimulator
from hypotheses.examples.always_long import AlwaysLongHypothesis
from state.market_state import MarketState
from state.position_state import PositionState


def test_always_long_basic():
    """Test that always_long hypothesis produces exactly one entry."""
    # Generate synthetic data
    bars = MarketDataLoader.create_synthetic_data(
        symbol="TEST",
        start_date=datetime(2020, 1, 1),
        num_bars=100,
        initial_price=100.0,
        seed=42
    )
    
    # Setup
    hypothesis = AlwaysLongHypothesis()
    clock = Clock()
    bar_iterator = BarIterator(bars)
    decision_queue = DecisionQueue(execution_delay_bars=1)
    cost_model = CostModel(transaction_cost_bps=0.0, slippage_bps=0.0)  # Zero cost
    executor = ExecutionSimulator(cost_model, initial_capital=100_000.0)
    
    market_state = MarketState()
    position_state = PositionState()
    
    # Run replay
    def on_execution(decisions, bar, bar_index, mkt_state, pos_state):
        executor.execute_decisions(decisions, bar, pos_state)
    
    engine = ReplayEngine(
        hypothesis=hypothesis,
        bar_iterator=bar_iterator,
        clock=clock,
        decision_queue=decision_queue,
        market_state=market_state,
        position_state=position_state
    )
    
    stats = engine.run(on_execution_callback=on_execution)
    
    # Assertions
    assert stats["bars_processed"] == 100
    assert stats["decisions_made"] == 1  # Should only decide to buy once
    
    trades = executor.get_completed_trades()
    assert len(trades) == 1  # Should have 1 entry
    assert trades[0].trade_type == "ENTRY"
    assert trades[0].side == "LONG"


def test_zero_cost_vs_benchmark():
    """Test that zero-cost always-long matches buy-and-hold benchmark."""
    # Generate synthetic data
    bars = MarketDataLoader.create_synthetic_data(
        symbol="TEST",
        start_date=datetime(2020, 1, 1),
        num_bars=252,
        initial_price=100.0,
        trend=0.0005,
        seed=42
    )
    
    initial_capital = 100_000.0
    
    # Setup hypothesis evaluation
    hypothesis = AlwaysLongHypothesis()
    clock = Clock()
    bar_iterator = BarIterator(bars)
    decision_queue = DecisionQueue(execution_delay_bars=1)
    cost_model = CostModel(transaction_cost_bps=0.0, slippage_bps=0.0)
    executor = ExecutionSimulator(cost_model, initial_capital)
    
    market_state = MarketState()
    position_state = PositionState()
    
    # Run replay
    def on_execution(decisions, bar, bar_index, mkt_state, pos_state):
        executor.execute_decisions(decisions, bar, pos_state)
    
    engine = ReplayEngine(
        hypothesis=hypothesis,
        bar_iterator=bar_iterator,
        clock=clock,
        decision_queue=decision_queue,
        market_state=market_state,
        position_state=position_state
    )
    
    engine.run(on_execution_callback=on_execution)
    
    # Get final capital (including unrealized PnL)
    # Important: Must include unrealized PnL because AlwaysLong never exits in this simulation
    final_capital = executor.get_total_capital(bars[-1].close, position_state)
    
    # Calculate benchmark
    benchmark = BenchmarkCalculator.calculate_buy_and_hold_return(
        bars=bars,
        initial_capital=initial_capital,
        include_costs=False
    )
    
    # They should be very close (within 1.0% due to execution at open vs close)
    hypothesis_return = ((final_capital - initial_capital) / initial_capital) * 100
    benchmark_return = benchmark["benchmark_return_pct"]
    
    # Allow small difference due to execution timing
    assert abs(hypothesis_return - benchmark_return) < 1.0


def test_metrics_calculation():
    """Test that metrics are calculated correctly."""
    # Generate data with known characteristics
    bars = MarketDataLoader.create_synthetic_data(
        symbol="TEST",
        start_date=datetime(2020, 1, 1),
        num_bars=100,
        initial_price=100.0,
        volatility=0.01,
        seed=42
    )
    
    initial_capital = 100_000.0
    
    # Run evaluation
    hypothesis = AlwaysLongHypothesis()
    clock = Clock()
    bar_iterator = BarIterator(bars)
    decision_queue = DecisionQueue(execution_delay_bars=1)
    cost_model = CostModel(transaction_cost_bps=10.0, slippage_bps=5.0)
    executor = ExecutionSimulator(cost_model, initial_capital)
    
    market_state = MarketState()
    position_state = PositionState()
    
    def on_execution(decisions, bar, bar_index, mkt_state, pos_state):
        executor.execute_decisions(decisions, bar, pos_state)
    
    engine = ReplayEngine(
        hypothesis=hypothesis,
        bar_iterator=bar_iterator,
        clock=clock,
        decision_queue=decision_queue,
        market_state=market_state,
        position_state=position_state
    )
    
    engine.run(on_execution_callback=on_execution)
    
    # Calculate metrics
    final_capital = executor.get_total_capital(bars[-1].close, position_state)
    
    trades = executor.get_completed_trades()
    metrics = EvaluationMetrics(trades, initial_capital, final_capital)
    
    # Basic assertions
    assert metrics.trade_count() > 0
    assert metrics.final_equity() == final_capital
    assert isinstance(metrics.sharpe_ratio(), float)
    assert isinstance(metrics.max_drawdown(), float)
    assert metrics.max_drawdown() >= 0.0


def test_clock_monotonicity():
    """Test that clock time only moves forward."""
    clock = Clock()
    
    # Set initial time
    t1 = datetime(2020, 1, 1)
    clock.set_time(t1)
    assert clock.now() == t1
    
    # Move forward
    t2 = datetime(2020, 1, 2)
    clock.set_time(t2)
    assert clock.now() == t2
    
    # Try to move backward (should raise)
    with pytest.raises(ValueError):
        clock.set_time(t1)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
