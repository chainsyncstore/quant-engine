from dataclasses import dataclass
from enum import Enum

class GuardrailStatus(Enum):
    PASS = "PASS"
    FAIL = "FAIL"

@dataclass
class AggregatedHypothesisResult:
    """
    Computed after execution, using OOS windows only.
    """
    hypothesis_id: str
    oos_mean_return: float
    oos_median_return: float
    oos_sharpe: float
    oos_max_drawdown: float
    oos_alpha: float
    oos_beta: float
    oos_ir: float
    profit_factor: float
    profitable_window_ratio: float
    regime_coverage_count: int
    decay_detected: bool
    guardrail_status: GuardrailStatus

@dataclass
class RankedHypothesis:
    """
    Final output persisted and exposed.
    """
    batch_id: str
    hypothesis_id: str
    research_score: float
    rank: int
    oos_sharpe: float
    oos_mean_return: float
    oos_max_drawdown: float
    oos_alpha: float
    oos_beta: float
    oos_ir: float
    decay_flag: bool
    guardrail_status: GuardrailStatus
