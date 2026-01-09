import argparse
import logging
from config.settings import get_settings
from config.policies import get_policy
from storage.repositories import EvaluationRepository
from evaluation.longitudinal import LongitudinalTracker

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def main():
    parser = argparse.ArgumentParser(description="Run longitudinal hypothesis decay checks.")
    parser.add_argument("--policy", required=True, help="Research Policy ID")
    parser.add_argument("--symbol", default="SYNTHETIC", help="Market Symbol")
    parser.add_argument("--data-path", required=True, help="Path to market data CSV")
    
    args = parser.parse_args()
    
    settings = get_settings()
    repo = EvaluationRepository(settings.database_path)
    policy = get_policy(args.policy)
    
    print(f"Starting Longitudinal Check for policy: {policy.policy_id}")
    
    tracker = LongitudinalTracker(repo, policy, settings)
    results = tracker.run_checks(data_path=args.data_path, symbol=args.symbol)
    
    print("\n--- Results ---")
    for res in results:
        print(f"Hypothesis: {res['hypothesis_id']}")
        print(f"  Status Update: {res['status']}")
        print(f"  Reason: {res['reason']}")
        print(f"  New Metrics: Sharpe={res['new_metrics'].get('sharpe_ratio'):.2f}, Return={res['new_metrics'].get('total_return'):.2f}%")
        print("-" * 30)
