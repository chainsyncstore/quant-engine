
import hashlib
import json
from datetime import datetime
from typing import List, Optional

from config.settings import get_settings
from storage.repositories import EvaluationRepository
from orchestrator.run_evaluation import run_evaluation
from evaluation.policy import ResearchPolicy
from data.market_loader import MarketDataLoader
from batch.batch_config import BatchConfig
from batch.models import AggregatedHypothesisResult, RankedHypothesis, GuardrailStatus
from batch.aggregation import aggregate_results
from batch.ranker import rank_hypotheses
from promotion.evaluator import PromotionEvaluator

class BatchRunner:
    """
    Executes a batch of hypotheses under shared conditions.
    """
    def __init__(self, config: BatchConfig, db_path: Optional[str] = None):
        self.config = config
        self.settings = get_settings()
        self._db_path = db_path or self.settings.database_path
        self.repo = EvaluationRepository(self._db_path)
        
    def run(self, promote: bool = False) -> List[RankedHypothesis]:
        """
        Run the batch execution flow.
        """
        print(f"Starting Batch: {self.config.batch_id}")
        
        
        
        # 1. Load Shared Research Policy
        policy = self._get_policy()
        self.repo.store_policy(policy)
        # register_policy(policy) # No longer needed if loaded from registry? 
        # Actually validation: config.policies.get_policy loads from registry.
        # But for Orchestrator to see it, it must be in registry. It IS in registry if we got it via get_policy.
        # So we don't need register_policy unless we want to ensure it's there? 
        # get_policy raises if not found.
        
        # 2. Store Batch Config
        config_hash = self._compute_config_hash()
        self.repo.store_batch(
            batch_id=self.config.batch_id, 
            market_symbol=self.config.market_symbol, 
            config_hash=config_hash,
            policy_id=policy.policy_id
        )
        
        # 3. Setup/Load Data ONCE
        print("Loading Shared Data...")
        bars = self._load_data()
        
        
        aggregated_results: List[AggregatedHypothesisResult] = []
        
        # 4. Execute Hypotheses
        for hypothesis_id in self.config.hypotheses:
            print(f"Running Hypothesis: {hypothesis_id}")
            try:
                # Invoke Orchestrator
                # Note: run_evaluation expects string dates or uses default.
                # BatchConfig assumes full range available in data or specific windows?
                # run_evaluation logic filters bars if start/end date provided.
                # Our BatchConfig doesn't expose strict start/end date, so we use all bars loaded.
                
                run_output = run_evaluation(
                    hypothesis_id=hypothesis_id,
                    policy_id=policy.policy_id,
                    symbol=self.config.market_symbol,
                    preloaded_bars=bars,
                    output_db=self._db_path,
                    verbose=False # Reduce noise
                )
                
                # 5. Aggregate Results
                agg_result = aggregate_results(hypothesis_id, run_output)
                aggregated_results.append(agg_result)
                
            except Exception as e:
                print(f"Error executing hypothesis {hypothesis_id}: {e}")
                import traceback
                traceback.print_exc()
                # Treat as failed
                aggregated_results.append(AggregatedHypothesisResult(
                    hypothesis_id=hypothesis_id,
                    oos_mean_return=0.0,
                    oos_median_return=0.0,
                    oos_sharpe=0.0,
                    oos_max_drawdown=0.0,
                    oos_alpha=0.0,
                    oos_beta=0.0,
                    oos_ir=0.0,
                    profit_factor=0.0,
                    profitable_window_ratio=0.0,
                    regime_coverage_count=0,
                    decay_detected=False,
                    guardrail_status=GuardrailStatus.FAIL
                ))

        # 6. Rank Results
        print("Ranking Results...")
        rankings = rank_hypotheses(self.config.batch_id, aggregated_results)
        
        # 7. Persist Rankings
        for r in rankings:
            self.repo.store_batch_ranking(vars(r))
            
        # 8. Promotion (Optional)
        if promote:
            print("Evaluating Promotions...")
            evaluator = PromotionEvaluator(policy, self.config.batch_id)
            decisions = evaluator.evaluate(rankings)
            for d in decisions:
                self.repo.store_hypothesis_status(
                    hypothesis_id=d.hypothesis_id,
                    status=d.decision.value,
                    batch_id=self.config.batch_id,
                    policy_id=policy.policy_id,
                    rationale=d.reasons
                )
                if d.decision.value == "PROMOTED":
                    print(f"  [PROMOTED] {d.hypothesis_id}")
            
        return rankings

    def _load_data(self):
        if self.config.synthetic:
            return MarketDataLoader.create_synthetic_data(
                symbol=self.config.market_symbol,
                start_date=datetime(2020, 1, 1),
                num_bars=self.config.synthetic_bars or 252
            )
        else:
            # BatchConfig defaults to "SYNTHETIC" or expects a path?
            # BatchConfig has `market_symbol`. 
            # It seems we need a `data_path` if not synthetic.
            # But BatchConfig doesn't have `data_path` field in the PRD example:
            # BatchConfig(batch_id, market_symbol, hypotheses, ..., synthetic, synthetic_bars)
            # CLI args in PRD: --market OPTS (implies symbol?) --synthetic ...
            # Wait, if not synthetic, where does it load from?
            # Existing `MarketDataLoader.load_from_csv` requires a path.
            # The PRD CLI interface:
            # --market OPTS
            # Usually "OPTS" is a symbol. Where is the CSV?
            # Maybe the system assumes a standard data directory or the CLI passes it?
            # The PRD doesn't explicitly modify BatchConfig to have `data_path`.
            # But the Orchestrator needs it.
            # I will assume `settings.data_directory` or similar, OR I will assume `market_symbol`
            # maps to a known file if not provided.
            # Let's check `MarketDataLoader`.
            # It has `load_from_csv(file_path, symbol)`.
            # Typically, we might construct path from symbol: `data/{symbol}.csv`.
            # For now, I'll rely on `self.settings` or a convention.
            # Or I'll add `data_path` to `BatchConfig` if needed? 
            # PRD: "BatchConfig... market_symbol: str".
            # Let's assume for now we only support Synthetic or we need to add `data_path` to config or usage.
            # I will add logic to try finding the file.
            
            p = f"data/{self.config.market_symbol}.csv"
            return MarketDataLoader.load_from_csv(p, self.config.market_symbol)

    
    def _get_policy(self) -> ResearchPolicy:
        # Load implicit policy from config
        from config.policies import get_policy
        return get_policy(self.config.policy_id)

    def _compute_config_hash(self) -> str:
        # Simple hash of the config object (excluding ephemeral)
        d = vars(self.config).copy()
        s = json.dumps(d, sort_keys=True, default=str)
        return hashlib.sha256(s.encode()).hexdigest()
