"""
Meta-Strategy Regime Shift Test

Validates C2/C3 Logic:
- Can the MetaPortfolioEngine handle a portfolio of strategies?
- Does it detect decay when regimes shift?
- Does it re-allocate capital (Decay -> Zero Weight)?

Scenario:
1. Force-promote 3 hypotheses (Simulated False Positives).
2. Run through 3 regimes: 
   - Bull (0-100): Biased strategies perform well.
   - Choppy (100-200): Trend followers die, Counter-trend shines.
   - Bear (200-300): Long-only dies.
3. Verify that weights adjust dynamically.
"""
import sys
sys.path.insert(0, '.')

import sqlite3
import random
from datetime import datetime, timedelta

from hypotheses.registry import get_hypothesis
from portfolio.meta_engine import MetaPortfolioEngine
from portfolio.ensemble import Ensemble
from portfolio.weighting import RobustnessWeighting
from storage.repositories import EvaluationRepository
from data.schemas import Bar
from promotion.models import HypothesisStatus

# Config
DB_PATH = "results/test_meta_regime.db"
START_DATE = datetime(2024, 1, 1)

def setup_db():
    """Setup a fresh DB for this test."""
    import os
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
        
    # Initialize schema
    conn = sqlite3.connect(DB_PATH)
    with open('storage/schema.sql', 'r') as f:
        conn.executescript(f.read())
    conn.close()
    
    return EvaluationRepository(DB_PATH)

def generate_regime_bars() -> list:
    """Generate 300 bars: Bull -> Choppy -> Bear."""
    random.seed(42)
    bars = []
    price = 100.0
    
    # 1. Bull (0-100)
    for i in range(100):
        change = random.gauss(0.003, 0.01) # Strong drift up
        price *= (1 + change)
        bars.append(create_bar(START_DATE + timedelta(days=i), price, change))
        
    # 2. Choppy (100-200)
    for i in range(100):
        change = random.gauss(0.0, 0.025) # No drift, high vol
        price *= (1 + change)
        bars.append(create_bar(START_DATE + timedelta(days=100+i), price, change))
        
    # 3. Bear (200-300)
    for i in range(100):
        change = random.gauss(-0.003, 0.012) # Strong drift down
        price *= (1 + change)
        bars.append(create_bar(START_DATE + timedelta(days=200+i), price, change))
        
    return bars

def create_bar(timestamp, close, change):
    open_p = close / (1 + change/2) # Approx
    high_p = max(open_p, close) * (1 + abs(random.gauss(0, 0.005)))
    low_p = min(open_p, close) * (1 - abs(random.gauss(0, 0.005)))
    return Bar(
        timestamp=timestamp,
        open=round(open_p, 2),
        high=round(high_p, 2),
        low=round(low_p, 2),
        close=round(close, 2),
        volume=1000000
    )

def inject_faked_promotions(repo: EvaluationRepository, hypotheses: list):
    """Force promote hypotheses so they are picked up by the Ensemble."""
    for hid in hypotheses:
        # 1. Register hypothesis
        repo.store_hypothesis(hid, {}, "Test Hypothesis")
        
        # 2. Store fake evaluation (High Sharpe to start)
        repo.store_evaluation(
            hypothesis_id=hid,
            parameters={},
            market_symbol="SYNTHETIC",
            test_start_timestamp=START_DATE,
            test_end_timestamp=START_DATE,
            metrics={"sharpe_ratio": 2.0}, # High initial score
            benchmark_metrics={},
            assumed_costs_bps=10,
            initial_capital=100000,
            final_equity=110000,
            bars_processed=100,
            result_tag="STABLE",
            sample_type="OUT_OF_SAMPLE",
            policy_id="TEST_POLICY"
        )
        
        # 3. Store Promoted Status
        repo.store_hypothesis_status(
            hypothesis_id=hid,
            status="PROMOTED",
            rationale=["Simulated Promotion"],
            policy_id="TEST_POLICY"
        )

def main():
    print("="*80)
    print("META-STRATEGY REGIME SHIFT TEST")
    print("="*80)
    print("Scenario: Bull (0-100) -> Choppy (100-200) -> Bear (200-300)")
    
    # 1. Setup
    repo = setup_db()
    
    # Hypotheses to test
    # - simple_momentum: Should thrive in Bull, die in Choppy/Bear
    # - counter_trend: Should thrive in Choppy, die in Trends? Or maybe survive.
    # - time_exit: Random noise (Control)
    test_hypotheses = ["simple_momentum", "counter_trend", "time_exit"]
    
    print(f"Injecting fake promotions for: {test_hypotheses}")
    inject_faked_promotions(repo, test_hypotheses)
    
    # 2. Initialize Ensemble & Engine
    print("Initializing Meta Engine...")
    hypotheses_objs = [get_hypothesis(hid)() for hid in test_hypotheses]
    
    ensemble = Ensemble(
        hypotheses=hypotheses_objs,
        weighting_strategy=RobustnessWeighting(),
        repo=repo,
        policy_id="TEST_POLICY"
    )
    
    from execution.cost_model import CostModel
    
    engine = MetaPortfolioEngine(
        ensemble=ensemble,
        initial_capital=100000.0,
        cost_model=CostModel(transaction_cost_bps=10, slippage_bps=5),
        decay_check_interval=20 # Check every 20 days
    )
    
    # 3. Run Simulation
    print("Running simulation (300 bars)...")
    bars = generate_regime_bars()
    history = engine.run(bars)
    
    # 4. Analyze Allocation History
    print("\n" + "="*80)
    print("ALLOCATION & DECAY ANALYSIS")
    print("="*80)
    
    # Track when weights drop to zero
    allocations = {hid: [] for hid in test_hypotheses}
    timestamps = [s.timestamp for s in history]
    
    for snap in history:
        for hid in test_hypotheses:
            if hid in snap.allocations:
                allocations[hid].append(snap.allocations[hid].allocated_capital)
            else:
                allocations[hid].append(0.0)
                
    # Sample points
    points = [50, 150, 250, 299] # Mid-Bull, Mid-Choppy, Mid-Bear, End
    
    print(f"{'Date':<12} | {'Regime':<10} | {'Momentum':>12} | {'Counter':>12} | {'TimeExit':>12}")
    print("-" * 75)
    
    regime_map = {50: "BULL", 150: "CHOPPY", 250: "BEAR", 299: "END"}
    
    for idx in points:
        date_str = timestamps[idx].strftime("%Y-%m-%d")
        regime = regime_map.get(idx, "?")
        
        row = f"{date_str:<12} | {regime:<10}"
        for hid in test_hypotheses:
            cap = allocations[hid][idx]
            status = "DEAD" if cap == 0 else f"${cap:,.0f}"
            row += f" | {status:>12}"
        print(row)
        
    # Check final statuses
    print("-" * 75)
    print("\nFinal Statuses:")
    for hid in test_hypotheses:
        status = ensemble.current_statuses.get(hid)
        print(f"  {hid:20}: {status}")
        
    print("\n" + "="*80)
    
    # Verification
    decayed_count = sum(1 for h in test_hypotheses if ensemble.current_statuses.get(h) == HypothesisStatus.DECAYED)
    if decayed_count > 0:
        print(f"✓ PASS: System successfully decayed {decayed_count} strategies online.")
    else:
        print("⚠ FAIL: No strategies were decayed. Thresholds might be too loose.")

if __name__ == "__main__":
    main()
