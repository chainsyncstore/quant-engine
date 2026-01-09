"""
Regime Gating Verification Test

Validates that strategies are blocked from the Meta-Portfolio in forbidden regimes,
but continue to be tracked in Shadow Simulators.

Scenario:
1. Regime: BULL (0-100) -> CHOPPY (100-200).
2. Hypothesis: counter_trend (Allowed: CHOPPY, NEUTRAL).
3. Expectation:
   - BULL Phase: Shadow trades (loss), Meta Portfolio stays FLAT (weight ignored).
   - CHOPPY Phase: Shadow trades (gain), Meta Portfolio trades (gain).
"""
import sys
sys.path.insert(0, '.')

import sqlite3
import random
from datetime import datetime, timedelta

from hypotheses.registry import get_hypothesis
from portfolio.meta_engine import MetaPortfolioEngine
from portfolio.ensemble import Ensemble
from portfolio.weighting import EqualWeighting
from storage.repositories import EvaluationRepository
from data.schemas import Bar
from execution.cost_model import CostModel

DB_PATH = "results/test_regime_gating.db"
START_DATE = datetime(2024, 1, 1)

def setup_db():
    import os
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
    conn = sqlite3.connect(DB_PATH)
    with open('storage/schema.sql', 'r') as f:
        conn.executescript(f.read())
    conn.close()
    return EvaluationRepository(DB_PATH)

def create_bar(timestamp, close, change):
    open_p = close / (1 + change/2)
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

def generate_bars() -> list:
    random.seed(42)
    bars = []
    price = 100.0
    
    # Phase 0: Warmup (200 bars)
    for i in range(200):
        change = random.gauss(0.001, 0.01)
        price *= (1 + change)
        bars.append(create_bar(START_DATE + timedelta(days=i), price, change))
        
    start_test = 200
    
    # Phase 1: BULL (200-300) - Strong Up
    # Should result in Price > SMA200 and SMA50 > SMA200
    for i in range(100):
        change = random.gauss(0.005, 0.01) 
        price *= (1 + change)
        bars.append(create_bar(START_DATE + timedelta(days=start_test+i), price, change))
        
    # Phase 2: CHOPPY (300-500) - Low vol to drop ADX < 20
    # Adjusted to 0.5% daily vol to ensure ADX drops
    for i in range(200):
        change = random.gauss(0.0, 0.005) 
        price *= (1 + change)
        bars.append(create_bar(START_DATE + timedelta(days=start_test+100+i), price, change))
        
    return bars

def inject_promotion(repo, hid):
    repo.store_hypothesis(hid, {}, "Test")
    repo.store_hypothesis_status(
        hid,
        "PROMOTED",
        "Test",
        "TEST_POLICY",
        rationale=["Injected promotion"]
    )

def main():
    print("="*80)
    print("REGIME GATING VERIFICATION")
    print("="*80)
    
    repo = setup_db()
    inject_promotion(repo, "counter_trend")
    
    ensemble = Ensemble(
        hypotheses=[get_hypothesis("counter_trend")()],
        weighting_strategy=EqualWeighting(), # 100% weight to counter_trend
        repo=repo,
        policy_id="TEST_POLICY"
    )
    
    engine = MetaPortfolioEngine(
        ensemble=ensemble,
        initial_capital=100000.0,
        cost_model=CostModel(0, 0), # Zero cost for clean math
        decay_check_interval=0
    )
    
    print("Generating 500 bars (200 warmup, 100 Bull, 200 Chop)...")
    bars = generate_bars()
    
    print("Running simulation...")
    history = engine.run(bars)
    
    warmup_end = 200
    bull_end = 300
    
    cap_start = history[warmup_end].total_capital
    cap_bull_end = history[bull_end].total_capital
    cap_chop_end = history[-1].total_capital
    
    shadow_start = history[warmup_end].allocations["counter_trend"].allocated_capital
    shadow_bull_end = history[bull_end].allocations["counter_trend"].allocated_capital
    shadow_chop_end = history[-1].allocations["counter_trend"].allocated_capital
    
    print("\nResults:")
    print(f"{'Phase':<10} | {'Meta PnL':>12} | {'Shadow PnL':>12}")
    print("-" * 40)
    print(f"{'BULL':<10} | {cap_bull_end - cap_start:>12.2f} | {shadow_bull_end - shadow_start:>12.2f}")
    print(f"{'CHOP':<10} | {cap_chop_end - cap_bull_end:>12.2f} | {shadow_chop_end - shadow_bull_end:>12.2f}")
    
    gating_worked = abs(cap_bull_end - cap_start) < 1.0 # Tolerance
    shadow_active = abs(shadow_bull_end - shadow_start) > 10.0
    meta_active_chop = abs(cap_chop_end - cap_bull_end) > 10.0
    
    print("\nVerifications:")
    print(f"1. Meta Gated in BULL: {gating_worked} (PnL ~ 0)")
    print(f"2. Shadow Active in BULL: {shadow_active}")
    print(f"3. Meta Active in CHOP: {meta_active_chop}")
    
    # DEBUG: Check Classification
    from market.regime import RegimeClassifier
    from state.market_state import MarketState
    
    ms = MarketState(lookback_window=500)
    classifier = RegimeClassifier()
    
    print("\nRegime Diagnosis:")
    for i, bar in enumerate(bars):
        ms.update(bar)
        if i % 20 == 0 and i >= 200: # Check more frequently
            regime = classifier.classify(ms)
            df = ms.to_dataframe(300)
            
            # Manually check ADX
            try:
                adx_series = classifier._calculate_adx(df)
                adx = adx_series.iloc[-1] if not adx_series.empty else 0.0
            except (KeyError, ValueError, ZeroDivisionError):
                adx = 0.0
            
            print(f"Bar {i}: Regime={regime}, ADX={adx:.2f}, Price={bar.close:.2f}")

    if gating_worked and shadow_active and meta_active_chop:
        print("\n✓ PASS: Regime Gating successfully blocked trades in BULL and allowed in CHOP.")
    else:
        print("\n✗ FAIL: Verification failed.")

if __name__ == "__main__":
    main()
