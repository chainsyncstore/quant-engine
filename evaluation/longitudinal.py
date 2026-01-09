import json
import logging
from datetime import datetime
from typing import Any, List, Optional
from batch.models import RankedHypothesis, GuardrailStatus
from evaluation.policy import ResearchPolicy
from orchestrator.run_evaluation import _run_single_pass
from promotion.decay import DecayRule
from promotion.models import HypothesisStatus
from storage.repositories import EvaluationRepository
from hypotheses.registry import get_hypothesis
from data.market_loader import MarketDataLoader
from config.settings import get_settings

logger = logging.getLogger(__name__)

class LongitudinalTracker:
    def __init__(self, repo: EvaluationRepository, policy: ResearchPolicy, settings: Optional[Any] = None):
        self.repo = repo
        self.policy = policy
        self.settings = settings or get_settings()
        self.decay_rule = DecayRule()

    def run_checks(self, data_path: str, symbol: str, current_time: Optional[datetime] = None) -> List[dict]:
        """
        Run longitudinal checks on all PROMOTED hypotheses.
        """
        if current_time is None:
            current_time = datetime.now()

        promoted_ids = self.repo.get_hypotheses_by_status(
            HypothesisStatus.PROMOTED.value, 
            policy_id=self.policy.policy_id
        )
        results = []

        if not promoted_ids:
            logger.info(f"No PROMOTED hypotheses found for policy {self.policy.policy_id}.")
            return []

        # Load all data once - optimization
        # In a real system, we might need to load per-symbol if hypotheses target different symbols.
        # For now, we assume single symbol or passed symbol.
        all_bars = MarketDataLoader.load_from_csv(data_path, symbol=symbol)
        if not all_bars:
             logger.warning(f"No market data found for {symbol}")
             return []

        # Convert to simple list for easier filtering
        # Assume bars are sorted
        
        for hid in promoted_ids:
            latest_eval = self.repo.get_latest_evaluation(hid, policy_id=self.policy.policy_id)
            if not latest_eval:
                logger.warning(f"Promoted hypothesis {hid} has no evaluation record. Skipping.")
                continue

            last_end_time = datetime.fromisoformat(latest_eval['test_end_timestamp'])
            
            # Select "New Data"
            new_bars = [b for b in all_bars if b.timestamp > last_end_time and b.timestamp <= current_time]
            
            # Check availability
            # TODO: Configurable minimum bars for re-evaluation?
            if len(new_bars) < 20: # Arbitrary small number
                logger.info(f"Insufficient new data for {hid} (Only {len(new_bars)} bars). Last run: {last_end_time}")
                continue

            logger.info(f"Checking decay for {hid} on {len(new_bars)} new bars...")

            # Match params
            hypothesis_details = self.repo.get_hypothesis_details(hid)
            params = {}
            if hypothesis_details and 'parameters_json' in hypothesis_details:
                try:
                    params = json.loads(hypothesis_details['parameters_json'])
                except json.JSONDecodeError:
                    logger.error(f"Failed to load params for {hid}")

            hypothesis_cls = get_hypothesis(hid)
            hypothesis = hypothesis_cls(**params) 
            
            result = _run_single_pass(
                hypothesis=hypothesis,
                bars=new_bars,
                policy=self.policy,
                settings=self.settings,
                symbol=symbol,
                repo=None, # Don't store this RUN as an evaluation? Or do we? 
                           # We probably should store it as a "MONITORING" run.
                verbose=False,
                sample_type="MONITORING"
            )

            metrics = result["metrics"]
            
            # Create a "pseudo" RankedHypothesis for the rule
            # DecayRule checks: oos_sharpe, oos_max_drawdown, decay_flag
            # We map metrics to these fields.
            
            pseudo_ranked = RankedHypothesis(
                batch_id="MONITORING",
                hypothesis_id=hid,
                research_score=0.0,
                rank=0,
                oos_sharpe=metrics.get('sharpe_ratio', 0.0),
                oos_mean_return=metrics.get('mean_return_per_trade', 0.0),
                oos_max_drawdown=metrics.get('max_drawdown', 0.0),
                oos_alpha=metrics.get('alpha', 0.0),
                oos_beta=metrics.get('beta', 0.0),
                oos_ir=metrics.get('information_ratio', 0.0),
                decay_flag=False,
                guardrail_status=GuardrailStatus.PASS
            )
            
            passed, reason = self.decay_rule.evaluate(pseudo_ranked, self.policy)
            
            status_update = "MAINTAINED"
            if not passed:
                status_update = "DECAYED"
                self.repo.store_hypothesis_status(
                    hypothesis_id=hid,
                    status=HypothesisStatus.DECAYED.value,
                    policy_id=self.policy.policy_id,
                    rationale=[reason]
                )
                logger.info(f"Hypothesis {hid} DECAYED: {reason}")
            else:
                 logger.info(f"Hypothesis {hid} MAINTAINED.")

            results.append({
                "hypothesis_id": hid,
                "status": status_update,
                "reason": reason,
                "new_metrics": metrics
            })
            
            # Store the monitoring run?
            self.repo.store_evaluation(
                 hypothesis_id=hid,
                 parameters=hypothesis.parameters,
                 market_symbol=symbol,
                 test_start_timestamp=new_bars[0].timestamp,
                 test_end_timestamp=new_bars[-1].timestamp,
                 metrics=metrics,
                 benchmark_metrics={}, # TODO
                 assumed_costs_bps=self.policy.transaction_cost_bps,
                 initial_capital=self.settings.starting_capital,
                 final_equity=metrics.get('final_equity'),
                 bars_processed=len(new_bars),
                 result_tag="MONITORING",
                 sample_type="MONITORING",
                 policy_id=self.policy.policy_id
            )

        return results
