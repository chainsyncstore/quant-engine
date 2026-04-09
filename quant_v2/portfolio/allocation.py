"""Signal-to-exposure allocation logic for v2 portfolio routing."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Iterable

from quant_v2.contracts import StrategySignal

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Session-edge windows (UTC hours)
# ---------------------------------------------------------------------------
# High-edge: overlaps with major session opens where breakout follow-through
# is historically strongest.  Crypto vol clusters around US equity open
# (13:30–15:00 UTC) and Asian open / daily close (00:00–02:00 UTC).
_HIGH_EDGE_HOURS: frozenset[int] = frozenset({0, 1, 13, 14, 15})
# Low-edge: weekend-like dead zones where signals have weaker follow-through.
_LOW_EDGE_HOURS: frozenset[int] = frozenset({3, 4, 5, 9, 10, 11})

_SESSION_BOOST: float = 1.0    # multiplier for high-edge hours
_SESSION_NORMAL: float = 0.85  # multiplier for neutral hours
_SESSION_DAMPEN: float = 0.65  # multiplier for low-edge hours

# ---------------------------------------------------------------------------
# Regime directional bias
# ---------------------------------------------------------------------------
_REGIME_ALIGN_MULT: float = 1.0   # signal aligns with momentum
_REGIME_NEUTRAL_MULT: float = 0.85  # no clear momentum
_REGIME_OPPOSE_MULT: float = 0.55  # signal fights the prevailing trend

# ---------------------------------------------------------------------------
# Symbol prediction accuracy dampening
# ---------------------------------------------------------------------------
_ACCURACY_STRONG_THRESHOLD: float = 0.55   # hit rate above this → full allocation
_ACCURACY_WEAK_THRESHOLD: float = 0.45     # hit rate below this → heavy dampening
_ACCURACY_STRONG_MULT: float = 1.0
_ACCURACY_NEUTRAL_MULT: float = 0.60
_ACCURACY_WEAK_MULT: float = 0.30

# ---------------------------------------------------------------------------
# Event gate dampening
# ---------------------------------------------------------------------------
_EVENT_DEFAULT_MULT: float = 1.0  # no event data → neutral

# ---------------------------------------------------------------------------
# Model agreement dampening
# ---------------------------------------------------------------------------
_AGREEMENT_STRONG_THRESHOLD: float = 0.8
_AGREEMENT_STRONG_MULT: float = 1.0
_AGREEMENT_NEUTRAL_MULT: float = 0.85
_AGREEMENT_WEAK_MULT: float = 0.60


def _session_multiplier(hour_utc: int | None) -> float:
    """Return a soft session-edge multiplier for the given UTC hour."""
    if hour_utc is None:
        return _SESSION_NORMAL
    if hour_utc in _HIGH_EDGE_HOURS:
        return _SESSION_BOOST
    if hour_utc in _LOW_EDGE_HOURS:
        return _SESSION_DAMPEN
    return _SESSION_NORMAL


def _symbol_accuracy_multiplier(hit_rate: float | None) -> float:
    """Return a soft allocation multiplier based on rolling prediction accuracy.

    Returns 1.0 (neutral) when *hit_rate* is None (insufficient data).
    """
    if hit_rate is None:
        return _ACCURACY_STRONG_MULT
    if hit_rate >= _ACCURACY_STRONG_THRESHOLD:
        return _ACCURACY_STRONG_MULT
    if hit_rate >= _ACCURACY_WEAK_THRESHOLD:
        return _ACCURACY_NEUTRAL_MULT
    return _ACCURACY_WEAK_MULT


def _event_gate_multiplier(event_gate_mult: float | None) -> float:
    """Return the event-gate multiplier, or 1.0 if no event data."""
    if event_gate_mult is None:
        return _EVENT_DEFAULT_MULT
    return max(0.0, min(event_gate_mult, 1.0))


def _model_agreement_multiplier(agreement: float | None) -> float:
    """Return allocation multiplier based on model agreement level.

    agreement >= 0.8 → 1.0× (strong agreement, full allocation)
    agreement >= 0.5 → 0.85× (mild agreement)
    agreement < 0.5  → 0.60× (disagreement, dampen)
    None             → 1.0× (no ensemble data, neutral pass-through)
    """
    if agreement is None:
        return 1.0  # no ensemble data → no penalty (neutral pass-through)
    if agreement >= _AGREEMENT_STRONG_THRESHOLD:
        return _AGREEMENT_STRONG_MULT
    if agreement >= 0.5:
        return _AGREEMENT_NEUTRAL_MULT
    return _AGREEMENT_WEAK_MULT


def _regime_multiplier(signal_direction: str, momentum_bias: float | None) -> float:
    """Return a soft regime-alignment multiplier.

    *momentum_bias* is in [-1, 1]:
      positive = bullish momentum, negative = bearish momentum.
    A BUY signal aligned with positive momentum gets 1.0×;
    a BUY signal opposing negative momentum gets the dampened multiplier.
    """
    if momentum_bias is None:
        return _REGIME_NEUTRAL_MULT
    abs_bias = abs(momentum_bias)
    if abs_bias < 0.10:
        return _REGIME_NEUTRAL_MULT

    if signal_direction == "BUY":
        return _REGIME_ALIGN_MULT if momentum_bias > 0 else _REGIME_OPPOSE_MULT
    elif signal_direction == "SELL":
        return _REGIME_ALIGN_MULT if momentum_bias < 0 else _REGIME_OPPOSE_MULT
    return _REGIME_NEUTRAL_MULT


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
    enable_session_filter: bool = True,
    enable_regime_bias: bool = True,
    enable_symbol_accuracy: bool = True,
    enable_event_gate: bool = True,
    enable_model_agreement: bool = True,
) -> AllocationDecision:
    """Allocate portfolio exposures from model signals under confidence-scaled caps.

    When *enable_session_filter* is True, signals carry a ``session_hour_utc``
    field that modulates exposure based on intraday session-edge windows.

    When *enable_regime_bias* is True, signals carry a ``momentum_bias`` field
    that scales exposure based on trend alignment.

    When *enable_symbol_accuracy* is True, signals carry a ``symbol_hit_rate``
    field that dampens allocation for symbols with poor rolling prediction accuracy.
    """

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

        # --- Session volatility filter ---
        sess_mult = 1.0
        if enable_session_filter:
            sess_mult = _session_multiplier(signal.session_hour_utc)

        # --- Regime directional bias ---
        regime_mult = 1.0
        if enable_regime_bias:
            regime_mult = _regime_multiplier(signal.signal, signal.momentum_bias)

        # --- Symbol prediction accuracy ---
        accuracy_mult = 1.0
        if enable_symbol_accuracy:
            accuracy_mult = _symbol_accuracy_multiplier(signal.symbol_hit_rate)

        # --- Event gate ---
        event_mult = 1.0
        if enable_event_gate:
            event_mult = _event_gate_multiplier(signal.event_gate_mult)

        # --- Model agreement ---
        agreement_mult = 1.0
        if enable_model_agreement:
            agreement_mult = _model_agreement_multiplier(signal.model_agreement)

        signed_exposure *= sess_mult * regime_mult * accuracy_mult * event_mult * agreement_mult

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
