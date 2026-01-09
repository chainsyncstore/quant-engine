"""
Demo script that populates the database tables to show the full pipeline.
Runs evaluation, promotion, portfolio simulation, and meta simulation.
"""
import sys
sys.path.insert(0, '.')

from datetime import datetime
from storage.repositories import EvaluationRepository
from config.settings import get_settings
from config.policies import get_policy
from promotion.models import HypothesisStatus
from portfolio.models import PortfolioAllocation, PortfolioState

# Get settings
settings = get_settings()
repo = EvaluationRepository(settings.database_path)
policy = get_policy("WF_V1")

print("=== Demo: Populating Database Tables ===\n")

# 1. Store a hypothesis
print("1. Storing hypothesis...")
repo.store_hypothesis(
    "simple_momentum", 
    {"hold_bars": 3}, 
    "SimpleMomentumHypothesis"
)
print("   Stored: simple_momentum")

# 2. Store an evaluation
print("\n2. Storing evaluation metrics...")
metrics = {
    "total_trades": 50,
    "win_rate_pct": 55.0,
    "sharpe_ratio": 1.25,
    "max_drawdown_pct": 12.5,
    "profit_factor": 1.8,
    "total_return_pct": 18.5,
    "final_equity": 118500.0,
    "total_pnl": 18500.0
}

repo.store_evaluation(
    hypothesis_id="simple_momentum",
    parameters={"hold_bars": 3},
    market_symbol="SYNTHETIC",
    test_start_timestamp=datetime(2023, 1, 1),
    test_end_timestamp=datetime(2023, 12, 31),
    metrics=metrics,
    benchmark_metrics={"benchmark_return_pct": 10.0},
    assumed_costs_bps=10,
    initial_capital=100000.0,
    final_equity=118500.0,
    bars_processed=252,
    result_tag="STABLE",
    sample_type="OUT_OF_SAMPLE",
    policy_id=policy.policy_id
)
print(f"   Stored evaluation: Sharpe={metrics['sharpe_ratio']}, Return={metrics['total_return_pct']}%")

# 3. Store promotion status
print("\n3. Storing promotion decision...")
repo.store_hypothesis_status(
    hypothesis_id="simple_momentum",
    status=HypothesisStatus.PROMOTED.value,
    rationale=["Passed all thresholds"],
    policy_id=policy.policy_id
)
print("   Status: PROMOTED")

# 4. Store portfolio evaluation 
print("\n4. Storing portfolio evaluation...")

snapshot = PortfolioState(
    timestamp=datetime(2023, 12, 31),
    total_capital=115000.0,
    cash=5000.0,
    allocations={
        "simple_momentum": PortfolioAllocation(
            hypothesis_id="simple_momentum",
            allocated_capital=110000.0,
            current_position=None,
            unrealized_pnl=0.0,
            realized_pnl=15000.0
        )
    },
    total_realized_pnl=15000.0,
    total_unrealized_pnl=0.0,
    drawdown_pct=5.2
)

repo.store_portfolio_evaluation(snapshot, "DEMO_RUN", policy.policy_id)
print(f"   Stored portfolio: Capital=${snapshot.total_capital:,.2f}, DD={snapshot.drawdown_pct:.1f}%")

print("\n=== Demo Complete ===")
print("\nRun the following to inspect tables:")
print('  python -c "...[same inspect query as before]..."')
