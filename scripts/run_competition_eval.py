"""Run competition evaluation pipeline for candidate hypotheses."""

from orchestrator.run_evaluation import run_evaluation
from config.settings import get_settings
from storage.repositories import EvaluationRepository
from promotion.models import HypothesisStatus
from config.policies import get_policy

DATA_PATH = r"C:\Users\HP\AppData\Roaming\MetaQuotes\Terminal\10CE948A1DFC9A8C27E56E827008EBD4\MQL5\Files\results\live_eurusd.csv"
POLICY_ID = "COMPETITION_EVAL"
SYMBOL = "EURUSD"

CANDIDATES = [
    "volatility_expansion_breakout",
    "mean_reversion_exhaustion",
    "session_open_impulse",
    "volatility_compression",
]


def main():
    settings = get_settings()
    repo = EvaluationRepository(settings.database_path)
    policy = get_policy(POLICY_ID)
    
    print("=" * 60)
    print("COMPETITION EVALUATION PIPELINE")
    print("=" * 60)
    print(f"Policy: {POLICY_ID}")
    print(f"Data: {DATA_PATH}")
    print(f"Candidates: {CANDIDATES}")
    print()
    
    results = []
    
    for hid in CANDIDATES:
        print(f"\n--- Evaluating: {hid} ---")
        try:
            result = run_evaluation(
                hypothesis_id=hid,
                policy_id=POLICY_ID,
                data_path=DATA_PATH,
                symbol=SYMBOL,
                verbose=True
            )
            
            windows = result.get("windows", [])
            print(f"Walk-forward windows: {len(windows)}")
            
            # Aggregate test metrics
            total_trades = 0
            total_return = 0.0
            sharpe_sum = 0.0
            max_dd = 0.0
            
            for w in windows:
                tm = w.get("test_metrics", {})
                total_trades += tm.get("trade_count", 0)
                total_return += tm.get("total_return", 0.0)
                sharpe_sum += tm.get("sharpe_ratio", 0.0)
                dd = tm.get("max_drawdown_pct", 0.0)
                if dd > max_dd:
                    max_dd = dd
            
            avg_sharpe = sharpe_sum / len(windows) if windows else 0.0
            
            results.append({
                "hypothesis_id": hid,
                "windows": len(windows),
                "total_trades": total_trades,
                "total_return": total_return,
                "avg_sharpe": avg_sharpe,
                "max_dd": max_dd,
            })
            
            print(f"  Total trades: {total_trades}")
            print(f"  Total return: {total_return:.2f}%")
            print(f"  Avg Sharpe: {avg_sharpe:.2f}")
            print(f"  Max DD: {max_dd:.2f}%")
            
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                "hypothesis_id": hid,
                "error": str(e)
            })
    
    # Promotion decisions
    print("\n" + "=" * 60)
    print("PROMOTION DECISIONS")
    print("=" * 60)
    
    for r in results:
        hid = r["hypothesis_id"]
        if "error" in r:
            status = HypothesisStatus.EVALUATED
            reason = f"Evaluation error: {r['error']}"
        elif r["total_trades"] < policy.promotion_min_trades:
            status = HypothesisStatus.EVALUATED
            reason = f"Insufficient trades: {r['total_trades']} < {policy.promotion_min_trades}"
        elif r["max_dd"] > policy.promotion_max_drawdown:
            status = HypothesisStatus.EVALUATED
            reason = f"Excessive drawdown: {r['max_dd']:.1f}% > {policy.promotion_max_drawdown}%"
        elif r["avg_sharpe"] < policy.promotion_min_sharpe:
            status = HypothesisStatus.EVALUATED
            reason = f"Sharpe too low: {r['avg_sharpe']:.2f} < {policy.promotion_min_sharpe}"
        else:
            status = HypothesisStatus.PROMOTED
            reason = f"Passed: trades={r['total_trades']}, sharpe={r['avg_sharpe']:.2f}, dd={r['max_dd']:.1f}%"
        
        repo.store_hypothesis_status(
            hypothesis_id=hid,
            status=status.value,
            policy_id=POLICY_ID,
            rationale=[reason]
        )
        
        symbol = "✓" if status == HypothesisStatus.PROMOTED else "✗"
        print(f"  {symbol} {hid}: {status.value} - {reason}")
    
    # Final summary
    print("\n" + "=" * 60)
    promoted = repo.get_hypotheses_by_status("PROMOTED", policy_id=POLICY_ID)
    print(f"PROMOTED under {POLICY_ID}: {promoted}")


if __name__ == "__main__":
    main()
