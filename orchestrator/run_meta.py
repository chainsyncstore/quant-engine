import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import pandas as pd

from config.competition_flags import COMPETITION_MODE
from config.settings import get_settings
from config.policies import get_policy
from config.execution_policies import get_execution_policy
from storage.repositories import EvaluationRepository
from portfolio.meta_engine import MetaPortfolioEngine
from portfolio.ensemble import Ensemble
from portfolio.weighting import EqualWeighting, RobustnessWeighting
from portfolio.risk import (
    ExecutionPolicyRule,
    LossStreakGuard,
    MaxDrawdownRule,
    TradeThrottle,
)
from data.market_loader import MarketDataLoader
from hypotheses.registry import get_hypothesis
from promotion.models import HypothesisStatus
from execution.cost_model import CostModel
from execution_live import ExecutionEventLogger, PaperExecutionAdapter
from execution_live.events import COMPETITION_PROFILE_LOADED
from execution_live.risk_checks import CashAvailabilityCheck, NotionalLimitCheck, ExecutionPolicyCheck
from execution_live.service import PaperExecutionService


def filter_new_bars(df: pd.DataFrame, last_seen_ts: datetime | None):
    """
    Idempotent guard to drop bars that have already been processed.
    """
    if df.empty:
        return df

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    if last_seen_ts is None:
        return df
    return df[df["timestamp"] > last_seen_ts]


def _load_last_seen_timestamp(state_path: Path) -> datetime | None:
    if not state_path.exists():
        return None

    try:
        raw = state_path.read_text(encoding="utf-8").strip()
        if not raw:
            return None
        return datetime.fromisoformat(raw)
    except Exception as exc:
        logger.warning("Failed to parse last_seen state file %s: %s", state_path, exc)
        return None


def _persist_last_seen_timestamp(state_path: Path, timestamp: datetime) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(timestamp.isoformat(), encoding="utf-8")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
DEBUG_TRACE_PATH = Path("results/run_meta_debug.log")

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
    parser.add_argument(
        "--state-path",
        default=None,
        help="Optional path to persist last processed bar timestamp (defaults to <data-path>.state)",
    )
    
    args = parser.parse_args()
    
    DEBUG_TRACE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with DEBUG_TRACE_PATH.open("a", encoding="utf-8") as handle:
        handle.write(f"{datetime.now().isoformat()} | ARGS: {args}\n")
    
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

    telemetry_hook: Optional[Callable[[str, dict], None]] = None
    if COMPETITION_MODE:
        def _emit_competition_event(event_type: str, payload: dict) -> None:
            logger.info(
                "competition_telemetry | event=%s payload=%s",
                event_type,
                payload,
            )

        telemetry_hook = _emit_competition_event
    
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
    with DEBUG_TRACE_PATH.open("a", encoding="utf-8") as handle:
        handle.write(f"{datetime.now().isoformat()} | PROMOTED_COUNT: {len(promoted_ids)}\n")
    
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
    
    # 3. Load Data (only process new bars)
    csv_df = pd.read_csv(args.data_path)
    state_path = Path(args.state_path) if args.state_path else Path(args.data_path).with_suffix(".state")
    last_seen_ts = _load_last_seen_timestamp(state_path)
    new_rows = filter_new_bars(csv_df, last_seen_ts)

    if new_rows.empty:
        logger.info("No new bars detected in %s (last_seen=%s)", args.data_path, last_seen_ts)
        return

    bars = MarketDataLoader.load_from_dataframe(new_rows, symbol=args.symbol)
    if not bars:
        logger.error("No market data found after filtering new bars.")
        return
        
    # 4. Init Engine
    cost_model = CostModel(
        transaction_cost_bps=policy.transaction_cost_bps,
        slippage_bps=policy.slippage_bps
    )
    
    paper_adapter = None
    execution_sink = None
    event_logger: Optional[ExecutionEventLogger] = None
    
    if args.paper:
        if args.paper_log:
            Path(args.paper_log).parent.mkdir(parents=True, exist_ok=True)
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
        if COMPETITION_MODE:
            telemetry_hook = event_logger.log
            event_logger.log(
                COMPETITION_PROFILE_LOADED,
                {
                    "policy_id": execution_policy.policy_id,
                    "label": execution_policy.label,
                    "tag": args.tag,
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
    elif telemetry_hook and COMPETITION_MODE:
        telemetry_hook(
            COMPETITION_PROFILE_LOADED,
            {
                "policy_id": execution_policy.policy_id,
                "label": execution_policy.label,
                "tag": args.tag,
            },
        )

    risk_rules = [
        MaxDrawdownRule(max_drawdown_pct=args.max_drawdown),
        TradeThrottle(telemetry_hook=telemetry_hook),
        LossStreakGuard(telemetry_hook=telemetry_hook),
        ExecutionPolicyRule(execution_policy),
    ]
    
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
    
    # 6. Store + persist progress
    logger.info(f"Simulation complete. Storing {len(history)} portfolio snapshots...")
    for state in history:
        repo.store_portfolio_evaluation(state, args.tag, policy.policy_id)

    latest_bar_ts = pd.to_datetime(new_rows["timestamp"]).max()
    if isinstance(latest_bar_ts, pd.Timestamp):
        latest_bar_ts = latest_bar_ts.to_pydatetime()
    if latest_bar_ts:
        _persist_last_seen_timestamp(state_path, latest_bar_ts)
        logger.info("Persisted last_seen timestamp %s to %s", latest_bar_ts, state_path)
        
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


if __name__ == "__main__":
    main()
