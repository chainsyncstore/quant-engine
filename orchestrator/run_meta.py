import argparse
import logging

from config.settings import get_settings
from config.policies import get_policy
from config.execution_policies import get_execution_policy
from storage.repositories import EvaluationRepository
from portfolio.meta_engine import MetaPortfolioEngine
from portfolio.ensemble import Ensemble
from portfolio.weighting import EqualWeighting, RobustnessWeighting
from portfolio.risk import MaxDrawdownRule, ExecutionPolicyRule
from data.market_loader import MarketDataLoader
from hypotheses.registry import get_hypothesis
from promotion.models import HypothesisStatus
from execution.cost_model import CostModel
from execution_live import ExecutionEventLogger, PaperExecutionAdapter
from execution_live.risk_checks import CashAvailabilityCheck, NotionalLimitCheck, ExecutionPolicyCheck
from execution_live.service import PaperExecutionService

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Run meta-strategy simulation.")
    parser.add_argument("--policy", required=True, help="Research Policy ID")
    parser.add_argument("--symbol", default="SYNTHETIC", help="Market Symbol")
    parser.add_argument("--data-path", required=True, help="Path to market data CSV")
    parser.add_argument("--capital", type=float, default=100000.0, help="Initial Capital")
    parser.add_argument("--max-drawdown", type=float, default=0.20, help="Max Drawdown Limit")
    parser.add_argument("--weighting", default="equal", choices=["equal", "robustness"], help="Weighting strategy")
    parser.add_argument("--tag", default="META_RUN", help="Meta Run Tag")
    parser.add_argument("--paper", action="store_true", help="Enable paper execution boundary")
    parser.add_argument("--paper-log", default=None, help="Optional path to append execution events as JSON lines")
    parser.add_argument("--paper-max-notional", type=float, default=0.0, help="Max notional per order for paper execution (0 disables check)")
    parser.add_argument("--execution-policy", default="RESEARCH", help="Execution policy ID to enforce")
    
    args = parser.parse_args()
    
    settings = get_settings()
    repo = EvaluationRepository(settings.database_path)
    policy = get_policy(args.policy)
    execution_policy = get_execution_policy(args.execution_policy)
    
    logger.info(
        "Starting Meta-Strategy Simulation for policy %s on %s with execution policy %s",
        policy.policy_id,
        args.symbol,
        execution_policy.label,
    )
    
    # 1. Fetch Hypotheses
    promoted_ids = repo.get_hypotheses_by_status(
        HypothesisStatus.PROMOTED.value,
        policy_id=policy.policy_id
    )
    
    if not promoted_ids:
        logger.warning("No PROMOTED hypotheses found.")
        return

    logger.info(f"Found {len(promoted_ids)} promoted hypotheses: {promoted_ids}")
    
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
        
    # 2. Configure Weighting
    weighting_strategy = EqualWeighting()
    if args.weighting == "robustness":
        weighting_strategy = RobustnessWeighting()
        
    ensemble = Ensemble(
        hypotheses=hypotheses,
        weighting_strategy=weighting_strategy,
        repo=repo,
        policy_id=policy.policy_id
    )
    
    logger.info(f"Initial Weights: {ensemble.weights}")
    
    # 3. Load Data
    bars = MarketDataLoader.load_from_csv(args.data_path, symbol=args.symbol)
    if not bars:
        logger.error("No market data found.")
        return
        
    # 4. Init Engine
    cost_model = CostModel(
        transaction_cost_bps=policy.transaction_cost_bps,
        slippage_bps=policy.slippage_bps
    )
    
    risk_rules = [
        MaxDrawdownRule(max_drawdown_pct=args.max_drawdown),
        ExecutionPolicyRule(execution_policy)
    ]
    
    paper_adapter = None
    execution_sink = None
    
    if args.paper:
        logger.info("Paper execution enabled.")
        event_logger = ExecutionEventLogger(persist_path=args.paper_log)
        event_logger.log(
            "execution_policy_loaded",
            {
                "policy_id": execution_policy.policy_id,
                "label": execution_policy.label,
                "config": execution_policy.serialize(),
            },
        )

        adapter_risk_checks = [CashAvailabilityCheck(), ExecutionPolicyCheck(execution_policy)]
        if args.paper_max_notional > 0:
            adapter_risk_checks.append(NotionalLimitCheck(args.paper_max_notional))
        
        paper_adapter = PaperExecutionAdapter(
            cost_model=cost_model,
            initial_equity=args.capital,
            risk_checks=adapter_risk_checks,
            event_logger=event_logger,
        )
        paper_service = PaperExecutionService(paper_adapter)
        
        def execution_sink(intent):
            logger.info(
                "ExecutionIntent -> symbol=%s action=%s qty=%.2f bar=%s",
                intent.symbol,
                intent.action.value,
                intent.quantity,
                intent.metadata.get("bar_index"),
            )
            report = paper_service.handle_intent(intent)
            logger.info(
                "ExecutionReport <- order=%s status=%s filled=%.2f msg=%s",
                report.order_id,
                report.status.value,
                report.filled_quantity,
                report.message or "",
            )
    
    engine = MetaPortfolioEngine(
        ensemble=ensemble,
        initial_capital=args.capital,
        cost_model=cost_model,
        risk_rules=risk_rules,
        symbol=args.symbol,
        execution_intent_sink=execution_sink,
    )
    
    # 5. Run
    history = engine.run(bars)
    
    # 6. Store
    logger.info(f"Simulation complete. Storing {len(history)} portfolio snapshots...")
    for state in history:
        repo.store_portfolio_evaluation(state, args.tag, policy.policy_id)
        
    final = history[-1]
    logger.info("--- Meta Portfolio Result ---")
    logger.info(f"Final Capital: ${final.total_capital:,.2f}")
    logger.info(f"Return: {((final.total_capital - args.capital) / args.capital * 100):.2f}%")
    logger.info(f"Max Drawdown: {final.drawdown_pct:.2f}%")
    
    if paper_adapter:
        account = paper_adapter.get_account_state()
        logger.info("--- Paper Execution Account ---")
        logger.info(
            "Equity: %.2f | Cash: %.2f | Positions: %d",
            account.equity,
            account.cash,
            len(account.positions),
        )
