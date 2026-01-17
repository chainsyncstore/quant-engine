"""
Research Policy Registry.

Defines standard research policies for the engine.
"""

from typing import Dict
from evaluation.policy import ResearchPolicy, EvaluationMode

_POLICIES: Dict[str, ResearchPolicy] = {}


def register_policy(policy: ResearchPolicy) -> None:
    """Register a policy."""
    _POLICIES[policy.policy_id] = policy


def get_policy(policy_id: str) -> ResearchPolicy:
    """Get a policy by ID."""
    if policy_id not in _POLICIES:
        raise ValueError(f"Unknown policy: {policy_id}. Available: {list(_POLICIES.keys())}")
    return _POLICIES[policy_id]


# --- Standard Policies ---

WF_V1 = ResearchPolicy(
    policy_id="WF_V1",
    description="Standard Walk-Forward V1: 252/63/63, Strict Guardrails",
    evaluation_mode=EvaluationMode.WALK_FORWARD,
    train_window_bars=252,
    test_window_bars=63,
    step_size_bars=63,
    execution_delay_bars=1,
    transaction_cost_bps=5.0,
    slippage_bps=5.0,
    min_trades=30,
    min_regimes=2,
    max_sharpe_decay=0.5,
    promotion_min_sharpe=0.5,
    promotion_min_profit_factor=1.2,
    promotion_min_return_pct=0.0,
    promotion_max_drawdown=25.0,
    promotion_min_trades=30
)

SINGLE_PASS_V1 = ResearchPolicy(
    policy_id="SINGLE_PASS_V1",
    description="Standard Single Pass: Full History, No Walk-Forward",
    evaluation_mode=EvaluationMode.SINGLE_PASS,
    train_window_bars=0, # Ignored
    test_window_bars=0, # Ignored
    step_size_bars=0, # Ignored
    execution_delay_bars=1,
    transaction_cost_bps=5.0,
    slippage_bps=5.0,
    min_trades=30,
    min_regimes=1, # Less strict
    max_sharpe_decay=1.0 # Ignored basically
)

register_policy(WF_V1)
register_policy(SINGLE_PASS_V1)

# Competition policy for limited data scenarios (~250 M5 bars)
COMPETITION_EVAL = ResearchPolicy(
    policy_id="COMPETITION_EVAL",
    description="Competition evaluation: Small walk-forward windows for limited live data",
    evaluation_mode=EvaluationMode.WALK_FORWARD,
    train_window_bars=100,  # ~8 hours of M5 data
    test_window_bars=50,    # ~4 hours of M5 data  
    step_size_bars=50,
    execution_delay_bars=1,
    transaction_cost_bps=5.0,
    slippage_bps=5.0,
    min_trades=3,           # Relaxed for low-frequency strategies
    min_regimes=1,
    max_sharpe_decay=0.75,
    promotion_min_sharpe=0.0,        # Allow any positive edge
    promotion_min_profit_factor=1.0, # Break-even or better
    promotion_min_return_pct=-5.0,   # Allow small losses during eval
    promotion_max_drawdown=15.0,     # Stricter DD for competition
    promotion_min_trades=1           # At least 1 trade to show activity
)
register_policy(COMPETITION_EVAL)
