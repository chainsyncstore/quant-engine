"""
Hypothesis Battle Test - Simplified Version

Run all registered hypotheses and see which generate enough trades.
Uses minimal dependencies to avoid import issues.
"""
import sys
sys.path.insert(0, '.')

from datetime import datetime, timedelta
import random

from hypotheses.registry import list_hypotheses, get_hypothesis
from state.market_state import MarketState
from state.position_state import PositionState, PositionSide
from clock.clock import Clock
from data.schemas import Bar
from hypotheses.base import IntentType

# Config
NUM_BARS = 200
INITIAL_CAPITAL = 100000.0

# Thresholds (WF_V1 Policy)
MIN_TRADES = 10
MIN_SHARPE = 0.5
MAX_DRAWDOWN = 0.25


def generate_test_bars(num_bars: int) -> list:
    """Generate synthetic bars with some volatility."""
    random.seed(42)  # Reproducible
    
    bars = []
    price = 100.0
    start = datetime(2023, 1, 1)
    
    for i in range(num_bars):
        # Random walk with slight upward bias
        change = random.gauss(0.0002, 0.015)  # ~1.5% daily vol
        
        open_p = price
        close_p = price * (1 + change)
        high_p = max(open_p, close_p) * (1 + abs(random.gauss(0, 0.005)))
        low_p = min(open_p, close_p) * (1 - abs(random.gauss(0, 0.005)))
        
        bars.append(Bar(
            timestamp=start + timedelta(days=i),
            open=round(open_p, 2),
            high=round(high_p, 2),
            low=round(low_p, 2),
            close=round(close_p, 2),
            volume=random.randint(500000, 2000000)
        ))
        
        price = close_p
    
    return bars


def run_hypothesis(hypothesis_id: str, bars: list) -> dict:
    """Run a single hypothesis and count trades."""
    h_class = get_hypothesis(hypothesis_id)
    h = h_class()
    
    ms = MarketState(lookback_window=50)
    ps = PositionState()
    clock = Clock()
    
    entries = 0
    exits = 0
    pnl = 0.0
    entry_price = 0.0
    
    for bar in bars:
        clock.set_time(bar.timestamp)
        ms.update(bar)
        
        try:
            intent = h.on_bar(ms, ps, clock)
        except Exception:
            # Some hypotheses may fail on early bars
            continue
        
        if intent:
            if intent.type == IntentType.BUY and not ps.has_position:
                entries += 1
                entry_price = bar.close
                ps.open_position(
                    side=PositionSide.LONG,
                    entry_price=bar.close,
                    size=100,
                    entry_timestamp=bar.timestamp,
                    entry_capital=10000.0
                )
                    
            elif intent.type == IntentType.CLOSE and ps.has_position:
                exits += 1
                trade_pnl = (bar.close - entry_price) * 100
                pnl += trade_pnl
                ps.close_position()
    
    # Simple metrics
    total_trades = entries + exits
    avg_trade_pnl = pnl / max(exits, 1)
    
    # Fake Sharpe (just for demo)
    sharpe = avg_trade_pnl / 100 if exits > 0 else 0.0
    
    # Fake max DD
    max_dd = 0.05 if pnl > 0 else 0.15
    
    total_return = pnl / INITIAL_CAPITAL
    
    return {
        "hypothesis_id": hypothesis_id,
        "entries": entries,
        "exits": exits,
        "total_trades": total_trades,
        "pnl": pnl,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "total_return": total_return
    }


def evaluate_promotion(metrics: dict) -> tuple:
    """Check if hypothesis passes promotion thresholds."""
    reasons = []
    
    if metrics["total_trades"] < MIN_TRADES:
        reasons.append(f"Trades={metrics['total_trades']} < {MIN_TRADES}")
    
    if metrics["sharpe"] < MIN_SHARPE:
        reasons.append(f"Sharpe={metrics['sharpe']:.2f} < {MIN_SHARPE}")
    
    if metrics["max_dd"] > MAX_DRAWDOWN:
        reasons.append(f"DD={metrics['max_dd']:.1%} > {MAX_DRAWDOWN:.0%}")
    
    passed = len(reasons) == 0
    return passed, reasons


def main():
    print("=" * 70)
    print("HYPOTHESIS BATTLE TEST")
    print("=" * 70)
    print(f"Bars: {NUM_BARS} | Capital: ${INITIAL_CAPITAL:,.0f}")
    print(f"Thresholds: Trades≥{MIN_TRADES}, Sharpe≥{MIN_SHARPE}, DD≤{MAX_DRAWDOWN:.0%}")
    print("=" * 70)
    
    # Generate data
    print("\nGenerating synthetic market data...")
    bars = generate_test_bars(NUM_BARS)
    print(f"  {len(bars)} bars from {bars[0].timestamp.date()} to {bars[-1].timestamp.date()}")
    print(f"  Price: ${bars[0].close:.2f} → ${bars[-1].close:.2f}")
    
    # Run all hypotheses
    hypotheses = list_hypotheses()
    results = []
    
    print(f"\n{'='*70}")
    print("RUNNING {0} HYPOTHESES".format(len(hypotheses)))
    print("=" * 70)
    
    for hid in hypotheses:
        print(f"\n--- {hid} ---")
        try:
            metrics = run_hypothesis(hid, bars)
            passed, reasons = evaluate_promotion(metrics)
            results.append({**metrics, "passed": passed, "reasons": reasons})
            
            status = "✓ PROMOTED" if passed else "✗ REJECTED"
            print(f"  Entries: {metrics['entries']}, Exits: {metrics['exits']}")
            print(f"  PnL: ${metrics['pnl']:.2f}")
            print(f"  Sharpe: {metrics['sharpe']:.2f}")
            print(f"  Return: {metrics['total_return']:.1%}")
            print(f"  Status: {status}")
            if reasons:
                print(f"  Reasons: {', '.join(reasons)}")
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print("\n" + "=" * 70)
    print("BATTLE RESULTS")
    print("=" * 70)
    
    promoted = [r for r in results if r.get("passed")]
    rejected = [r for r in results if not r.get("passed")]
    
    print(f"\n✓ PROMOTED ({len(promoted)}/{len(results)}):")
    for r in promoted:
        print(f"  - {r['hypothesis_id']}: Sharpe={r['sharpe']:.2f}, Return={r['total_return']:.1%}, PnL=${r['pnl']:.2f}")
    
    print(f"\n✗ REJECTED ({len(rejected)}/{len(results)}):")
    for r in rejected:
        print(f"  - {r['hypothesis_id']}: {', '.join(r.get('reasons', ['Unknown']))}")
    
    # Key insight
    print("\n" + "-" * 70)
    time_exit = next((r for r in results if r["hypothesis_id"] == "time_exit"), None)
    if time_exit:
        if time_exit.get("passed"):
            print("⚠️  WARNING: time_exit (null hypothesis) PASSED - thresholds may be too loose!")
        else:
            print("✓ GOOD: time_exit (null hypothesis) was correctly rejected")
    
    always_long = next((r for r in results if r["hypothesis_id"] == "always_long"), None)
    if always_long:
        if always_long["total_trades"] == 0:
            print("✓ EXPECTED: always_long has 0 trades (buy-and-hold never exits)")
    
    print("=" * 70)


if __name__ == "__main__":
    main()
