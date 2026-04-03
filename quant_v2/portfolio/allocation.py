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
    total_risk_budget_frac: float = 1.0,
    max_symbol_exposure_frac: float = 0.15,
    min_confidence: float = 0.65,
) -> AllocationDecision:
    """Allocate portfolio exposures from model signals under confidence-scaled caps."""

    if not 0.0 <= total_risk_budget_frac <= 1.0:
        raise ValueError("total_risk_budget_frac must be in [0, 1]")
    if not 0.0 < max_symbol_exposure_frac <= 1.0:
        raise ValueError("max_symbol_exposure_frac must be in (0, 1]")
    if not 0.0 <= min_confidence <= 1.0:
        raise ValueError("min_confidence must be in [0, 1]")

    actionable: list[tuple[str, float]] = []
    skipped: dict[str, str] = {}
    kelly_reference_confidence = 0.80
    kelly_reference_edge = max((2.0 * kelly_reference_confidence) - 1.0, 1e-12)
    kelly_scale = max_symbol_exposure_frac / kelly_reference_edge

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

        raw_edge = max((2.0 * float(signal.confidence)) - 1.0, 0.0)
        uncertainty_factor = 1.0 - float(signal.uncertainty) if signal.uncertainty is not None else 1.0
        adjusted_edge = raw_edge * max(uncertainty_factor, 0.0)
        if adjusted_edge <= 0.0:
            skipped[signal.symbol] = "zero_edge"
            continue

        signed_exposure = kelly_scale * adjusted_edge
        if signal.signal == "SELL":
            signed_exposure *= -1.0
        capped_exposure = max(-max_symbol_exposure_frac, min(signed_exposure, max_symbol_exposure_frac))
        if abs(capped_exposure) <= 0.0:
            skipped[signal.symbol] = "zero_edge"
            continue
        actionable.append((signal.symbol, capped_exposure))

    if not actionable or total_risk_budget_frac == 0.0:
        return AllocationDecision(
            target_exposures={},
            gross_exposure=0.0,
            net_exposure=0.0,
            skipped_symbols=skipped,
        )

    gross_requested = sum(abs(exposure) for _, exposure in actionable)
    if gross_requested <= 0.0:
        return AllocationDecision(
            target_exposures={},
            gross_exposure=0.0,
            net_exposure=0.0,
            skipped_symbols=skipped,
        )

    exposures: dict[str, float] = {}
    scale_to_budget = min(1.0, total_risk_budget_frac / gross_requested)
    for symbol, signed_exposure in actionable:
        exposures[symbol] = signed_exposure * scale_to_budget

    gross = float(sum(abs(v) for v in exposures.values()))
    net = float(sum(exposures.values()))
    return AllocationDecision(
        target_exposures=exposures,
        gross_exposure=gross,
        net_exposure=net,
        skipped_symbols=skipped,
    )
