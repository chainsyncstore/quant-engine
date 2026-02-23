"""Signal-to-exposure allocation logic for v2 portfolio routing."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from quant_v2.contracts import StrategySignal


@dataclass(frozen=True)
class AllocationDecision:
    """Signed exposure targets produced by the allocator."""

    target_exposures: dict[str, float]
    gross_exposure: float
    net_exposure: float
    skipped_symbols: dict[str, str] = field(default_factory=dict)


def allocate_signals(
    signals: Iterable[StrategySignal],
    *,
    total_risk_budget_frac: float = 0.15,
    max_symbol_exposure_frac: float = 0.05,
    min_confidence: float = 0.55,
) -> AllocationDecision:
    """Allocate portfolio exposures from model signals under simple caps."""

    if not 0.0 <= total_risk_budget_frac <= 1.0:
        raise ValueError("total_risk_budget_frac must be in [0, 1]")
    if not 0.0 < max_symbol_exposure_frac <= 1.0:
        raise ValueError("max_symbol_exposure_frac must be in (0, 1]")
    if not 0.0 <= min_confidence <= 1.0:
        raise ValueError("min_confidence must be in [0, 1]")

    actionable: list[tuple[str, float, float]] = []
    skipped: dict[str, str] = {}

    for signal in signals:
        if signal.signal == "HOLD":
            skipped[signal.symbol] = "hold"
            continue
        if signal.signal == "DRIFT_ALERT":
            skipped[signal.symbol] = "drift_alert"
            continue
        if signal.confidence < min_confidence:
            skipped[signal.symbol] = f"confidence<{min_confidence:.2f}"
            continue

        direction = 1.0 if signal.signal == "BUY" else -1.0
        base_score = max(signal.confidence - min_confidence, 0.0)
        uncertainty_factor = 1.0 - signal.uncertainty if signal.uncertainty is not None else 1.0
        score = base_score * max(uncertainty_factor, 0.0)
        if score <= 0.0:
            skipped[signal.symbol] = "zero_score"
            continue
        actionable.append((signal.symbol, direction, score))

    if not actionable or total_risk_budget_frac == 0.0:
        return AllocationDecision(
            target_exposures={},
            gross_exposure=0.0,
            net_exposure=0.0,
            skipped_symbols=skipped,
        )

    score_sum = sum(item[2] for item in actionable)
    if score_sum <= 0.0:
        return AllocationDecision(
            target_exposures={},
            gross_exposure=0.0,
            net_exposure=0.0,
            skipped_symbols=skipped,
        )

    exposures: dict[str, float] = {}
    for symbol, direction, score in actionable:
        raw = total_risk_budget_frac * (score / score_sum)
        capped = min(raw, max_symbol_exposure_frac)
        exposures[symbol] = direction * capped

    gross = float(sum(abs(v) for v in exposures.values()))
    net = float(sum(exposures.values()))
    return AllocationDecision(
        target_exposures=exposures,
        gross_exposure=gross,
        net_exposure=net,
        skipped_symbols=skipped,
    )
