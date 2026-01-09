
import pytest
import sqlite3
from batch.batch_runner import BatchRunner
from batch.batch_config import BatchConfig

@pytest.fixture
def temp_db(tmp_path):
    return str(tmp_path / "test_batch.db")

@pytest.fixture
def setup_policy():
    # Register a test policy
    from config.policies import _POLICIES, ResearchPolicy, EvaluationMode
    
    policy = ResearchPolicy(
        policy_id="TEST_INTEG_POLICY",
        description="Integration Test",
        evaluation_mode=EvaluationMode.WALK_FORWARD,
        train_window_bars=50,
        test_window_bars=20,
        step_size_bars=20,
        execution_delay_bars=0,
        transaction_cost_bps=0,
        slippage_bps=0,
        min_trades=10,
        min_regimes=1,
        max_sharpe_decay=0.5,
        promotion_min_sharpe=0.5,
        promotion_min_profit_factor=1.0,
        promotion_min_return_pct=0.0,
        promotion_max_drawdown=20.0,
        promotion_min_trades=10
    )
    _POLICIES["TEST_INTEG_POLICY"] = policy
    return policy

def test_batch_execution_e2e(temp_db, setup_policy):
    # Setup Config
    
    config = BatchConfig(
        batch_id="test_batch_001",
        policy_id="TEST_INTEG_POLICY",
        market_symbol="SYNTHETIC",
        hypotheses=["always_long"],
        assumed_costs_bps=0,
        synthetic=True,
        synthetic_bars=100
    )
    
    runner = BatchRunner(config, db_path=temp_db)
    rankings = runner.run()
    
    # Verify outputs
    assert len(rankings) == 1
    assert rankings[0].hypothesis_id == "always_long"
    assert rankings[0].rank == 1
    
    # Verify DB persistence
    conn = sqlite3.connect(temp_db)
    conn.row_factory = sqlite3.Row
    
    # Check Batches
    batches = conn.execute("SELECT * FROM batches").fetchall()
    assert len(batches) == 1
    assert batches[0]["batch_id"] == "test_batch_001"
    assert batches[0]["policy_id"] == "TEST_INTEG_POLICY"
    
    # Check Rankings
    db_rankings = conn.execute("SELECT * FROM batch_rankings").fetchall()
    assert len(db_rankings) == 1
    assert db_rankings[0]["hypothesis_id"] == "always_long"
    assert db_rankings[0]["rank"] == 1
    
    # Check Evaluations were stored
    evals = conn.execute("SELECT * FROM evaluations").fetchall()
    assert len(evals) > 0

def test_batch_execution_with_promotion(temp_db, setup_policy):
    # Create a batch config
    config = BatchConfig(
        batch_id="promo_batch",
        policy_id="TEST_INTEG_POLICY", # Re-use the one created above
        market_symbol="SYNTHETIC",
        hypotheses=["always_long"],
        assumed_costs_bps=0,
        synthetic=True,
        synthetic_bars=100
    )
    
    runner = BatchRunner(config, db_path=temp_db)
    # Run WITH promotion
    runner.run(promote=True)
    
    conn = sqlite3.connect(temp_db)
    conn.row_factory = sqlite3.Row
    
    # Check history
    history = conn.execute("SELECT * FROM hypothesis_status_history").fetchall()
    assert len(history) == 1
    assert history[0]["hypothesis_id"] == "always_long"
    assert history[0]["status"] in ["PROMOTED", "EVALUATED"]

