"""Scorecard and gate evaluation for the 70-82 target rubric."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ScoreInputs:
    """Inputs aligned to the v2 scoring rubric."""

    robustness: float
    tradability: float
    risk: float
    generalization: float
    live_readiness: float

    def __post_init__(self) -> None:
        for field_name in (
            "robustness",
            "tradability",
            "risk",
            "generalization",
            "live_readiness",
        ):
            value = getattr(self, field_name)
            if not 0.0 <= value <= 100.0:
                raise ValueError(f"{field_name} must be within [0, 100]")


@dataclass(frozen=True)
class GateInputs:
    """Hard gates for claiming >=70 score readiness."""

    dsr_majority: float
    positive_ev_fold_ratio: float
    ruin_probability: float
    single_symbol_ev_dependency: float
    shadow_live_drift_ok: bool

    def __post_init__(self) -> None:
        if not 0.0 <= self.dsr_majority <= 1.0:
            raise ValueError("dsr_majority must be in [0, 1]")
        if not 0.0 <= self.positive_ev_fold_ratio <= 1.0:
            raise ValueError("positive_ev_fold_ratio must be in [0, 1]")
        if not 0.0 <= self.ruin_probability <= 1.0:
            raise ValueError("ruin_probability must be in [0, 1]")
        if not 0.0 <= self.single_symbol_ev_dependency <= 1.0:
            raise ValueError("single_symbol_ev_dependency must be in [0, 1]")


@dataclass(frozen=True)
class GateResult:
    """Gate pass/fail diagnostics."""

    passed: bool
    checks: dict[str, bool]


def compute_system_score(inputs: ScoreInputs) -> float:
    """Compute weighted composite score from rubric components."""

    return round(
        0.30 * inputs.robustness
        + 0.25 * inputs.tradability
        + 0.20 * inputs.risk
        + 0.15 * inputs.generalization
        + 0.10 * inputs.live_readiness,
        2,
    )


def evaluate_claim_70_plus_gates(inputs: GateInputs) -> GateResult:
    """Evaluate hard gates that must pass before claiming >=70 quality."""

    checks = {
        "dsr_majority": inputs.dsr_majority >= 0.55,
        "positive_ev_fold_ratio": inputs.positive_ev_fold_ratio >= 0.65,
        "ruin_probability": inputs.ruin_probability <= 0.25,
        "single_symbol_ev_dependency": inputs.single_symbol_ev_dependency <= 0.35,
        "shadow_live_drift_ok": bool(inputs.shadow_live_drift_ok),
    }
    return GateResult(passed=all(checks.values()), checks=checks)
