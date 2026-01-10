import argparse
import logging

from config.competition_flags import COMPETITION_MODE
from config.settings import get_settings
from config.policies import get_policy
from config.execution_policies import get_execution_policy
from storage.repositories import EvaluationRepository
from portfolio.engine import PortfolioEngine
from portfolio.risk import (
    ExecutionPolicyRule,
    LossStreakGuard,
    MaxDrawdownRule,
    TradeThrottle,
)
from data.market_loader import MarketDataLoader
from hypotheses.registry import get_hypothesis
from promotion.models import HypothesisStatus
from execution_live.events import COMPETITION_PROFILE_LOADED

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Run portfolio simulation.")
    parser.add_argument("--policy", required=True, help="Research Policy ID")
    parser.add_argument("--symbol", default="SYNTHETIC", help="Market Symbol")
    parser.add_argument("--data-path", default=None, help="Path to market data CSV")
    parser.add_argument("--use-synthetic", action="store_true", help="Use synthetic data")
    parser.add_argument("--synthetic-bars", type=int, default=252, help="Number of synthetic bars")
    parser.add_argument("--capital", type=float, default=100000.0, help="Initial Capital")
    parser.add_argument("--max-drawdown", type=float, default=0.20, help="Max Drawdown Limit")
    parser.add_argument("--execution-policy", default="RESEARCH", help="Execution policy ID to enforce")
    parser.add_argument("--tag", default="MANUAL_RUN", help="Portfolio Tag")
    
    args = parser.parse_args()
    
    settings = get_settings()
    repo = EvaluationRepository(settings.database_path)
    policy = get_policy(args.policy)
    execution_policy_id = args.execution_policy
    if COMPETITION_MODE:
        logger.info(
            "COMPETITION_MODE enabled. Overriding requested execution policy %s with COMPETITION_5PERCENTERS",
            execution_policy_id,
        )
        execution_policy_id = "COMPETITION_5PERCENTERS"
    execution_policy = get_execution_policy(execution_policy_id)

    telemetry_hook = None
    if COMPETITION_MODE:
        def _log_competition_event(event_type: str, payload: dict) -> None:
            logger.info(
                "competition_telemetry | event=%s payload=%s",
                event_type,
                payload,
            )
        telemetry_hook = _log_competition_event
    
    logger.info(
        "Starting Portfolio Simulation for %s on %s with execution policy %s",
        policy.policy_id,
        args.symbol,
        execution_policy.label,
    )
    
    # 1. Fetch PROMOTED Hypotheses
    promoted_ids = repo.get_hypotheses_by_status(
        HypothesisStatus.PROMOTED.value,
        policy_id=policy.policy_id
    )
    
    if not promoted_ids:
        logger.warning("No PROMOTED hypotheses found.")
        return

    logger.info(f"Found {len(promoted_ids)} promoted hypotheses: {promoted_ids}")
    
    # 2. Instantiate Hypotheses
    import json
    hypotheses = []
    for hid in promoted_ids:
        details = repo.get_hypothesis_details(hid)
        params = {}
        if details and 'parameters_json' in details:
            try:
                params = json.loads(details['parameters_json'])
            except json.JSONDecodeError:
                logger.error(f"Failed to load params for {hid}")
        
        h_cls = get_hypothesis(hid)
        hypotheses.append(h_cls(**params))

    # 3. Load Market Data
    if args.use_synthetic or args.data_path is None:
        bars = MarketDataLoader.create_synthetic(
            num_bars=args.synthetic_bars, 
            symbol=args.symbol
        )
        logger.info(f"Generated {len(bars)} synthetic bars.")
    else:
        bars = MarketDataLoader.load_from_csv(args.data_path, symbol=args.symbol)
        if not bars:
            logger.error("No market data found.")
            return
        logger.info(f"Loaded {len(bars)} bars.")
    
    # 4. Initialize Engine
    risk_rules = [
        MaxDrawdownRule(max_drawdown_pct=args.max_drawdown),
        TradeThrottle(telemetry_hook=telemetry_hook),
        LossStreakGuard(telemetry_hook=telemetry_hook),
        ExecutionPolicyRule(execution_policy),
    ]
    if COMPETITION_MODE and telemetry_hook:
        telemetry_hook(
            COMPETITION_PROFILE_LOADED,
            {
                "policy_id": execution_policy.policy_id,
                "label": execution_policy.label,
                "tag": args.tag,
            },
        )
    
    engine = PortfolioEngine(
        hypotheses=hypotheses,
        initial_capital=args.capital,
        policy=policy,
        risk_rules=risk_rules
    )
    
    # 5. Run
    history = engine.run(bars)
    
    # 6. Store Results
    logger.info(f"Simulation complete. Storing {len(history)} portfolio snapshots...")
    for state in history:
        repo.store_portfolio_evaluation(state, args.tag, policy.policy_id)
        
    # 7. Summary
    final = history[-1]
    logger.info("--- Portfolio Result ---")
    logger.info(f"Final Capital: ${final.total_capital:,.2f}")
    logger.info(f"Return: {((final.total_capital - args.capital) / args.capital * 100):.2f}%")
    logger.info(f"Max Drawdown: {final.drawdown_pct:.2f}%")
