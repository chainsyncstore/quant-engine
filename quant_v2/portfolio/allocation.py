"""Signal-to-exposure allocation logic for v2 portfolio routing."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Iterable

from quant_v2.contracts import MarketRiskSnapshot, ModelSourceDetails, StrategySignal
from quant_v2.portfolio.cost_model import BinanceCostModel, confidence_to_edge_bps, get_default_cost_model

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
_REGIME2_SELL_ALLOCATION_MULT_ENV = "BOT_V2_REGIME2_SELL_ALLOCATION_MULT"
_REGIME2_SELL_ALLOCATION_MULT_DEFAULT: float = 0.50
_CHRONOS_REQUIRE_AGREEMENT_ENV = "BOT_V2_REQUIRE_CHRONOS_AGREEMENT_FOR_ENTRY"
_CHRONOS_DISAGREEMENT_MULT_ENV = "BOT_V2_CHRONOS_DISAGREEMENT_MULT"
_CHRONOS_DISAGREEMENT_MULT_DEFAULT: float = 0.25
_CHRONOS_EXTREME_CONFIDENCE_ENV = "BOT_V2_CHRONOS_EXTREME_CONFIDENCE"
_CHRONOS_EXTREME_CONFIDENCE_DEFAULT: float = 0.80

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


def _bounded_float_env(name: str, default: float, *, minimum: float, maximum: float) -> float:
    raw = os.getenv(name, str(default)).strip() or str(default)
    try:
        value = float(raw)
    except ValueError:
        value = float(default)
    return max(float(minimum), min(value, float(maximum)))


def _regime2_sell_allocation_multiplier() -> float:
    return _bounded_float_env(
        _REGIME2_SELL_ALLOCATION_MULT_ENV,
        _REGIME2_SELL_ALLOCATION_MULT_DEFAULT,
        minimum=0.0,
        maximum=1.0,
    )


def _bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _chronos_disagreement_multiplier() -> float:
    return _bounded_float_env(
        _CHRONOS_DISAGREEMENT_MULT_ENV,
        _CHRONOS_DISAGREEMENT_MULT_DEFAULT,
        minimum=0.0,
        maximum=1.0,
    )


def _chronos_extreme_confidence() -> float:
    return _bounded_float_env(
        _CHRONOS_EXTREME_CONFIDENCE_ENV,
        _CHRONOS_EXTREME_CONFIDENCE_DEFAULT,
        minimum=0.5,
        maximum=1.0,
    )


def _lgbm_directional_confidence(signal_direction: str, details: ModelSourceDetails) -> float | None:
    if details.lgbm_probability is None:
        return None
    if signal_direction == "BUY":
        return float(details.lgbm_probability)
    if signal_direction == "SELL":
        return 1.0 - float(details.lgbm_probability)
    return None


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
    constraints_applied: tuple[str, ...] = ()


def _active_market_short_guard(signals: Iterable[StrategySignal]) -> MarketRiskSnapshot | None:
    for signal in signals:
        market_risk = signal.market_risk
        if market_risk is not None and market_risk.broad_selloff:
            return market_risk
    return None


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
    enable_cost_gate: bool = True,
    equity_usd: float = 300.0,
    cost_model: BinanceCostModel | None = None,
    current_positions: dict[str, float] | None = None,
) -> AllocationDecision:
    """Allocate portfolio exposures from model signals under confidence-scaled caps.

    When *enable_session_filter* is True, signals carry a ``session_hour_utc``
    field that modulates exposure based on intraday session-edge windows.

    When *enable_regime_bias* is True, signals carry a ``momentum_bias`` field
    that scales exposure based on trend alignment.

    When *enable_symbol_accuracy* is True, signals carry a ``symbol_hit_rate``
    field that dampens allocation for symbols with poor rolling prediction accuracy.

    When *enable_cost_gate* is True, signals whose expected edge does not cover
    1.5× round-trip transaction costs are filtered out.
    """

    if not 0.0 <= total_risk_budget_frac <= 1.0:
        raise ValueError("total_risk_budget_frac must be in [0, 1]")
    if not 0.0 < max_symbol_exposure_frac <= 1.0:
        raise ValueError("max_symbol_exposure_frac must be in (0, 1]")
    if not 0.0 <= min_confidence <= 1.0:
        raise ValueError("min_confidence must be in [0, 1]")

    signal_list = tuple(signals)
    market_short_guard = _active_market_short_guard(signal_list)
    current_positions = current_positions or {}

    actionable: list[tuple[str, float]] = []
    skipped: dict[str, str] = {}
    constraints: list[str] = []
    kelly_reference_confidence = 0.80
    kelly_reference_edge = max((2.0 * kelly_reference_confidence) - 1.0, 1e-12)
    kelly_scale = max_symbol_exposure_frac / kelly_reference_edge

    for signal in signal_list:
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

        # --- Transaction cost gate + cost-aware Kelly ---
        cost_drag_frac = 0.0
        if enable_cost_gate:
            _cm = cost_model or get_default_cost_model()
            edge_bps = confidence_to_edge_bps(signal.confidence, signal.uncertainty)
            notional_usd = abs(kelly_scale * adjusted_edge) * equity_usd
            economic, cost_est = _cm.is_economic(signal.symbol, notional_usd, edge_bps)
            if not economic:
                skipped[signal.symbol] = (
                    f"cost_gate({edge_bps:.1f}bps_edge<{cost_est.min_edge_bps:.1f}bps_min)"
                )
                logger.debug(
                    "Cost gate: skipping %s edge=%.1fbps < min=%.1fbps (notional=%.0f)",
                    signal.symbol, edge_bps, cost_est.min_edge_bps, notional_usd,
                )
                continue
            # Subtract cost from edge before Kelly sizing (plan item A.3)
            if edge_bps > 0:
                cost_drag_frac = min(cost_est.round_trip_cost_bps / edge_bps, 0.95)

        cost_adjusted_edge = adjusted_edge * (1.0 - cost_drag_frac)
        signed_exposure = kelly_scale * cost_adjusted_edge

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

        model_sources = signal.model_sources
        if (
            model_sources is not None
            and model_sources.agreement is False
            and model_sources.chronos_enabled
        ):
            extreme_threshold = _chronos_extreme_confidence()
            lgbm_confidence = _lgbm_directional_confidence(signal.signal, model_sources)
            extreme_override = (
                lgbm_confidence is not None
                and lgbm_confidence >= extreme_threshold
                and model_sources.lgbm_direction == signal.signal
            )
            current_qty = float(current_positions.get(signal.symbol, 0.0) or 0.0)
            opposing_position = (
                (signal.signal == "SELL" and current_qty > 1e-12)
                or (signal.signal == "BUY" and current_qty < -1e-12)
            )

            if extreme_override:
                constraints.append("chronos_disagreement_extreme_override")
                logger.info(
                    "Chronos disagreement overridden for %s %s: lgbm_confidence=%.3f "
                    "threshold=%.2f chronos_direction=%s",
                    signal.symbol,
                    signal.signal,
                    lgbm_confidence,
                    extreme_threshold,
                    model_sources.chronos_direction,
                )
            elif opposing_position:
                skipped[signal.symbol] = "chronos_disagreement_flatten_only"
                constraints.append("chronos_disagreement_flatten_only")
                logger.info(
                    "Chronos disagreement allows flatten-only for %s %s: current_qty=%.8f "
                    "lgbm_p=%s chronos_p=%s final_p=%.3f",
                    signal.symbol,
                    signal.signal,
                    current_qty,
                    (
                        f"{model_sources.lgbm_probability:.3f}"
                        if model_sources.lgbm_probability is not None
                        else "n/a"
                    ),
                    (
                        f"{model_sources.chronos_probability:.3f}"
                        if model_sources.chronos_probability is not None
                        else "n/a"
                    ),
                    model_sources.final_probability,
                )
                continue
            elif _bool_env(_CHRONOS_REQUIRE_AGREEMENT_ENV, True):
                skipped[signal.symbol] = "chronos_disagreement_entry_block"
                constraints.append("chronos_disagreement_entry_block")
                logger.info(
                    "Chronos disagreement blocked %s entry for %s: lgbm_p=%s chronos_p=%s "
                    "final_p=%.3f lgbm_dir=%s chronos_dir=%s",
                    signal.signal,
                    signal.symbol,
                    (
                        f"{model_sources.lgbm_probability:.3f}"
                        if model_sources.lgbm_probability is not None
                        else "n/a"
                    ),
                    (
                        f"{model_sources.chronos_probability:.3f}"
                        if model_sources.chronos_probability is not None
                        else "n/a"
                    ),
                    model_sources.final_probability,
                    model_sources.lgbm_direction,
                    model_sources.chronos_direction,
                )
                continue
            else:
                disagreement_mult = _chronos_disagreement_multiplier()
                signed_exposure *= disagreement_mult
                constraints.append("chronos_disagreement_dampen")
                logger.info(
                    "Chronos disagreement dampened %s %s: multiplier=%.2f lgbm_p=%s "
                    "chronos_p=%s final_p=%.3f",
                    signal.signal,
                    signal.symbol,
                    disagreement_mult,
                    (
                        f"{model_sources.lgbm_probability:.3f}"
                        if model_sources.lgbm_probability is not None
                        else "n/a"
                    ),
                    (
                        f"{model_sources.chronos_probability:.3f}"
                        if model_sources.chronos_probability is not None
                        else "n/a"
                    ),
                    model_sources.final_probability,
                )

        if (
            model_sources is not None
            and model_sources.agreement is None
            and model_sources.chronos_enabled
            and model_sources.chronos_probability is None
            and _bool_env(_CHRONOS_REQUIRE_AGREEMENT_ENV, True)
        ):
            current_qty = float(current_positions.get(signal.symbol, 0.0) or 0.0)
            opposing_position = (
                (signal.signal == "SELL" and current_qty > 1e-12)
                or (signal.signal == "BUY" and current_qty < -1e-12)
            )
            if opposing_position:
                skipped[signal.symbol] = "chronos_unavailable_flatten_only"
                constraints.append("chronos_unavailable_flatten_only")
                logger.info(
                    "Chronos unavailable; allowing flatten-only for %s %s: current_qty=%.8f",
                    signal.symbol,
                    signal.signal,
                    current_qty,
                )
                continue
            skipped[signal.symbol] = "chronos_unavailable_entry_block"
            constraints.append("chronos_unavailable_entry_block")
            logger.info(
                "Chronos unavailable; blocked %s entry for %s because agreement is required",
                signal.signal,
                signal.symbol,
            )
            continue

        if signal.signal == "SELL":
            if signal.regime == 2:
                current_qty = float(current_positions.get(signal.symbol, 0.0) or 0.0)
                if current_qty > 1e-12:
                    skipped[signal.symbol] = "regime2_sell_flatten_only"
                    constraints.append("regime2_sell_flatten_only")
                    logger.info(
                        "Regime 2 SELL treated as flatten-only for %s: current_qty=%.8f",
                        signal.symbol,
                        current_qty,
                    )
                    continue

                regime2_mult = _regime2_sell_allocation_multiplier()
                signed_exposure *= regime2_mult
                constraints.append("regime2_sell_dampen")
                logger.info(
                    "Regime 2 SELL dampened for %s: multiplier=%.2f confidence=%.3f",
                    signal.symbol,
                    regime2_mult,
                    signal.confidence,
                )

            if market_short_guard is not None and signal.confidence < market_short_guard.strong_short_confidence:
                skipped[signal.symbol] = (
                    "market_short_guard:"
                    f"broad_selloff weak_sell confidence<{market_short_guard.strong_short_confidence:.2f}"
                )
                constraints.append("market_short_guard_block")
                logger.info(
                    "Market short guard blocked SELL %s: confidence=%.3f down_ratio=%.2f "
                    "median_return=%.4f btc_return=%s lookback=%sh current_qty=%.8f",
                    signal.symbol,
                    signal.confidence,
                    market_short_guard.down_ratio,
                    market_short_guard.median_return,
                    (
                        f"{market_short_guard.btc_return:.4f}"
                        if market_short_guard.btc_return is not None
                        else "n/a"
                    ),
                    market_short_guard.lookback_hours,
                    float(current_positions.get(signal.symbol, 0.0) or 0.0),
                )
                continue
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
            constraints_applied=tuple(dict.fromkeys(constraints)),
        )

    gross_requested = sum(abs(exposure) for _, exposure in actionable)
    if gross_requested <= 0.0:
        return AllocationDecision(
            target_exposures={},
            gross_exposure=0.0,
            net_exposure=0.0,
            skipped_symbols=skipped,
            constraints_applied=tuple(dict.fromkeys(constraints)),
        )

    exposures: dict[str, float] = {}
    scale_to_budget = min(1.0, total_risk_budget_frac / gross_requested)
    for symbol, signed_exposure in actionable:
        exposures[symbol] = signed_exposure * scale_to_budget

    if market_short_guard is not None:
        long_sum = sum(v for v in exposures.values() if v > 0.0)
        short_sum = sum(abs(v) for v in exposures.values() if v < 0.0)
        target_short_sum = float(market_short_guard.short_net_cap_frac) + long_sum
        if short_sum > target_short_sum and short_sum > 0.0:
            scale = max(target_short_sum, 0.0) / short_sum
            for symbol, value in list(exposures.items()):
                if value < 0.0:
                    exposures[symbol] = value * scale
            constraints.append("market_short_guard_net_cap")
            logger.info(
                "Market short guard capped net short exposure: short_sum=%.4f target=%.4f "
                "scale=%.3f down_ratio=%.2f median_return=%.4f btc_return=%s lookback=%sh",
                short_sum,
                target_short_sum,
                scale,
                market_short_guard.down_ratio,
                market_short_guard.median_return,
                (
                    f"{market_short_guard.btc_return:.4f}"
                    if market_short_guard.btc_return is not None
                    else "n/a"
                ),
                market_short_guard.lookback_hours,
            )

    gross = float(sum(abs(v) for v in exposures.values()))
    net = float(sum(exposures.values()))
    return AllocationDecision(
        target_exposures=exposures,
        gross_exposure=gross,
        net_exposure=net,
        skipped_symbols=skipped,
        constraints_applied=tuple(dict.fromkeys(constraints)),
    )
