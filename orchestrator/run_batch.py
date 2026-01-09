import argparse
import logging

from config.settings import get_settings
from config.policies import get_policy
from storage.repositories import EvaluationRepository
from orchestrator.run_evaluation import run_evaluation
from promotion.models import HypothesisStatus

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Run batch hypothesis evaluation.")
    parser.add_argument("--market", required=True, help="Market symbol (e.g., OPTS, SPY)")
    parser.add_argument("--policy", required=True, help="Research Policy ID")
    parser.add_argument("--hypotheses", nargs="+", required=True, help="List of hypothesis IDs")
    parser.add_argument("--data-path", default=None, help="Path to market data CSV")
    parser.add_argument("--use-synthetic", action="store_true", help="Use synthetic data")
    parser.add_argument("--promote", action="store_true", help="Run promotion after evaluation")
    
    args = parser.parse_args()
    
    settings = get_settings()
    policy = get_policy(args.policy)
    repo = EvaluationRepository(settings.database_path)
    
    logger.info(f"=== Batch Evaluation: {args.market} / {policy.policy_id} ===")
    logger.info(f"Hypotheses: {args.hypotheses}")
    
    # 1. Run each hypothesis
    results = []
    
    for hid in args.hypotheses:
        logger.info(f"\n--- Evaluating {hid} ---")
        try:
            result = run_evaluation(
                hypothesis_id=hid,
                policy_id=args.policy,
                data_path=args.data_path,
                symbol=args.market,
                use_synthetic=args.use_synthetic or (args.data_path is None),
                verbose=True
            )
            if result:
                results.append((hid, result))
                logger.info(f"{hid}: Evaluation complete")
        except Exception as e:
            logger.error(f"Failed to evaluate {hid}: {e}")
            import traceback
            traceback.print_exc()
            
    # 2. Promotion (if requested)
    if args.promote and results:
        logger.info("\n--- Running Promotion ---")
        
        # Get latest metrics for each hypothesis from DB
        for hid, _ in results:
            latest = repo.get_latest_evaluation(hid, policy_id=policy.policy_id)
            if latest:
                sharpe = latest.get('sharpe_ratio', 0.0) or 0.0
                total_return = latest.get('total_return_pct', 0.0) or 0.0
                max_dd = latest.get('max_drawdown_pct', 100.0) or 100.0
                total_trades = latest.get('total_trades', 0) or 0
                
                # Simple threshold check
                promoted = (
                    sharpe >= policy.promotion_min_sharpe and
                    total_return >= policy.promotion_min_return_pct and
                    max_dd <= policy.promotion_max_drawdown and
                    total_trades >= policy.min_trades
                )
                
                status = HypothesisStatus.PROMOTED if promoted else HypothesisStatus.REJECTED
                reason = f"Sharpe={sharpe:.2f}, Return={total_return:.1f}%, DD={max_dd:.1f}%, Trades={total_trades}"
                
                repo.store_hypothesis_status(
                    hid,
                    status.value,
                    policy_id=policy.policy_id,
                    rationale=[reason]
                )
                
                status_str = "✓ PROMOTED" if promoted else "✗ REJECTED"
                logger.info(f"{hid}: {status_str} ({reason})")
            else:
                logger.warning(f"{hid}: No evaluation found, skipping promotion")
            
    logger.info("\n=== Batch Complete ===")
