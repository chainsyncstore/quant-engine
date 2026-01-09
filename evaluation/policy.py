"""
Research Policy Definition.

Defines the immutable configuration for an evaluation run.
Ensures reproducibility and auditability.
"""

import hashlib
import json
from enum import Enum
from pydantic import BaseModel, ConfigDict


class EvaluationMode(str, Enum):
    WALK_FORWARD = "WALK_FORWARD"
    SINGLE_PASS = "SINGLE_PASS"


class ResearchPolicy(BaseModel):
    """
    Immutable configuration for research evaluation.
    
    Acts as the source of truth for:
    - Walk-forward parameters
    - Execution constraints (delay, costs)
    - Quality guardrails (min trades, regime coverage, decay)
    """
    model_config = ConfigDict(frozen=True)
    
    policy_id: str
    description: str
    evaluation_mode: EvaluationMode
    
    # Walk-Forward Config (Ignored if SINGLE_PASS)
    train_window_bars: int
    test_window_bars: int
    step_size_bars: int
    
    # Execution Constraints
    execution_delay_bars: int
    transaction_cost_bps: float
    slippage_bps: float
    
    # Research Guardrails
    min_trades: int
    min_regimes: int
    max_sharpe_decay: float
    
    # Promotion Thresholds
    promotion_min_sharpe: float = 0.5
    promotion_min_profit_factor: float = 1.2
    promotion_min_return_pct: float = 0.0 # Positive return
    promotion_max_drawdown: float = 20.0 # Max 20% DD
    promotion_min_trades: int = 30 # Must meet min trades for promotion (can be higher than research min)
    
    # Benchmark Filters
    promotion_min_information_ratio: float = 0.0 # Positive IR
    promotion_min_alpha: float = 0.0 # Positive Alpha
    promotion_max_beta: float = 2.0 # Cap Beta (avoid purely leveraged market exposure)
    
    def compute_hash(self) -> str:
        """
        Compute a deterministic hash of the policy configuration.
        
        Returns:
            SHA256 hash of the JSON representation (sorted keys).
        """
        # Exclude policy_id and description from hash? 
        # Usually yes, if we want "structural equality" to mean same hash.
        # But policy_id should probably imply the structure.
        # Let's hash the *content* so if two IDs have same content they collide or we verify content match.
        # Actually, let's include everything to be safe, or just the structural params.
        # Including ID makes it unique per ID.
        # Let's hash the structural parameters only to detect duplicates under different names if needed,
        # but for lineage, we want to know exact config used.
        
        data = self.model_dump(exclude={"policy_id", "description"})
        json_str = json.dumps(data, sort_keys=True)
        return hashlib.sha256(json_str.encode("utf-8")).hexdigest()
