
import argparse
import uuid
from batch.batch_config import BatchConfig
from batch.batch_runner import BatchRunner

def main():
    parser = argparse.ArgumentParser(description="Hypothesis Batch Execution")
    
    parser.add_argument("--batch-id", help="Optional Batch ID (generated if not provided)")
    parser.add_argument("--market", required=True, help="Market symbol (e.g. OPTS)")
    parser.add_argument("--hypotheses", nargs="+", required=True, help="List of hypothesis IDs")
    
    # Policy Params
    parser.add_argument("--policy", required=True, help="Policy ID (e.g. WF_V1)")
    
    # Costs & Data
    # The assumed_costs_bps argument is removed as it's now derived from the policy.
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data")
    parser.add_argument("--synthetic-bars", type=int, help="Number of synthetic bars")
    
    parser.add_argument("--promote", action="store_true", help="Evaluate hypothesis promotion")
    
    # Optional Data Path (Not in PRD but useful)
    parser.add_argument("--data-path", help="Path to market data CSV (if not synthetic)")

    args = parser.parse_args()
    
    # Validate args
    if args.synthetic and not args.synthetic_bars:
        parser.error("--synthetic-bars is required when --synthetic is set.")

    batch_id = args.batch_id or str(uuid.uuid4())[:8]
    
    # Ensure policy exists (fail early)
    from config.policies import get_policy
    try:
        policy = get_policy(args.policy)
    except ValueError as e:
        parser.error(str(e))

    config = BatchConfig(
        batch_id=batch_id,
        policy_id=args.policy,
        market_symbol=args.market,
        hypotheses=args.hypotheses,
        assumed_costs_bps=policy.transaction_cost_bps,
        synthetic=args.synthetic,
        synthetic_bars=args.synthetic_bars
    )
    
    print(f"Initializing Batch {batch_id}...")
    runner = BatchRunner(config) 
    
    try:
        rankings = runner.run(promote=args.promote)
        
        print("\n" + "="*50)
        print(f"BATCH RANKINGS (ID: {batch_id})")
        print("="*50)
        print(f"{'Rank':<5} {'Hypothesis':<20} {'Score':<10} {'Sharpe':<10} {'Status':<10}")
        print("-" * 60)
        
        for r in rankings:
            print(f"{r.rank:<5} {r.hypothesis_id:<20} {r.research_score:>8.4f} {r.oos_sharpe:>8.4f} {r.guardrail_status.value:<10}")
            
    except Exception as e:
        print(f"BATCH FAILED: {e}")
        import traceback
        traceback.print_exc()
        # sys.exit(1)
