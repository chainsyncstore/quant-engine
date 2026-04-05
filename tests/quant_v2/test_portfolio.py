from __future__ import annotations

import pytest

from quant_v2.contracts import StrategySignal
from quant_v2.portfolio.allocation import allocate_signals
from quant_v2.portfolio.risk_policy import PortfolioRiskPolicy


def _signal(symbol: str, side: str, confidence: float, uncertainty: float | None = None) -> StrategySignal:
    return StrategySignal(
        symbol=symbol,
        timeframe="1h",
        horizon_bars=4,
        signal=side,
        confidence=confidence,
        uncertainty=uncertainty,
    )


def test_allocate_signals_builds_signed_exposures_with_caps() -> None:
    decision = allocate_signals(
        [
            _signal("BTCUSDT", "BUY", 0.72, uncertainty=0.10),
            _signal("ETHUSDT", "SELL", 0.68, uncertainty=0.15),
            _signal("SOLUSDT", "HOLD", 0.50),
        ],
        total_risk_budget_frac=0.50,
        max_symbol_exposure_frac=0.05,
        min_confidence=0.65,
    )

    assert "BTCUSDT" in decision.target_exposures
    assert "ETHUSDT" in decision.target_exposures
    assert decision.target_exposures["BTCUSDT"] > 0
    assert decision.target_exposures["ETHUSDT"] < 0
    assert abs(decision.target_exposures["BTCUSDT"]) <= 0.05
    assert abs(decision.target_exposures["ETHUSDT"]) <= 0.05
    assert abs(decision.target_exposures["BTCUSDT"]) > abs(decision.target_exposures["ETHUSDT"])
    assert decision.gross_exposure <= 0.10 + 1e-12
    assert decision.skipped_symbols["SOLUSDT"] == "hold"


def test_allocate_signals_skips_low_confidence_and_drift() -> None:
    decision = allocate_signals(
        [
            _signal("BTCUSDT", "BUY", 0.52),
            _signal("ETHUSDT", "DRIFT_ALERT", 0.80),
        ],
        total_risk_budget_frac=0.50,
        min_confidence=0.65,
    )

    assert decision.target_exposures == {}
    assert decision.gross_exposure == 0.0
    assert decision.skipped_symbols["BTCUSDT"].startswith("confidence<")
    assert decision.skipped_symbols["ETHUSDT"] == "drift_alert"


def test_allocate_signals_confidence_scales_exposure_before_cap() -> None:
    # Disable session/regime filters to test pure Kelly scaling
    decision = allocate_signals(
        [
            _signal("BTCUSDT", "BUY", 0.80, uncertainty=0.0),
            _signal("ETHUSDT", "BUY", 0.65, uncertainty=0.0),
        ],
        total_risk_budget_frac=0.50,
        max_symbol_exposure_frac=0.05,
        min_confidence=0.65,
        enable_session_filter=False,
        enable_regime_bias=False,
        enable_symbol_accuracy=False,
    )

    assert decision.target_exposures["BTCUSDT"] == pytest.approx(0.05)
    assert decision.target_exposures["ETHUSDT"] == pytest.approx(0.025)
    assert decision.gross_exposure == pytest.approx(0.075)


def test_allocate_signals_validate_args() -> None:
    with pytest.raises(ValueError):
        allocate_signals([], total_risk_budget_frac=-0.1)

    with pytest.raises(ValueError):
        allocate_signals([], max_symbol_exposure_frac=0.0)

    with pytest.raises(ValueError):
        allocate_signals([], min_confidence=1.1)


def test_portfolio_risk_policy_caps_symbol_bucket_gross_and_net() -> None:
    exposures = {
        "BTCUSDT": 0.08,
        "ETHUSDT": 0.07,
        "SOLUSDT": -0.05,
        "XRPUSDT": -0.04,
    }
    bucket_map = {
        "BTCUSDT": "majors",
        "ETHUSDT": "majors",
        "SOLUSDT": "alts",
        "XRPUSDT": "alts",
    }

    policy = PortfolioRiskPolicy(
        max_symbol_exposure_frac=0.06,
        max_gross_exposure_frac=0.15,
        max_net_exposure_frac=0.05,
        correlation_bucket_caps={"majors": 0.08, "alts": 0.08},
    )

    result = policy.apply(exposures, bucket_map=bucket_map)

    assert all(abs(v) <= 0.06 + 1e-12 for v in result.exposures.values())
    assert sum(abs(v) for v in result.exposures.values()) <= 0.15 + 1e-12
    assert abs(sum(result.exposures.values())) <= 0.05 + 1e-12
    assert "symbol_cap" in result.constraints_applied


def test_allocate_signals_symbol_accuracy_dampens_weak_pairs() -> None:
    """Signals with poor rolling hit rate get dampened allocation."""
    strong_signal = StrategySignal(
        symbol="BTCUSDT", timeframe="1h", horizon_bars=4,
        signal="BUY", confidence=0.80, uncertainty=0.0,
        symbol_hit_rate=0.60,  # > 55% → mult 1.0
    )
    weak_signal = StrategySignal(
        symbol="ETHUSDT", timeframe="1h", horizon_bars=4,
        signal="BUY", confidence=0.80, uncertainty=0.0,
        symbol_hit_rate=0.40,  # < 45% → mult 0.30
    )

    decision = allocate_signals(
        [strong_signal, weak_signal],
        total_risk_budget_frac=1.0,
        max_symbol_exposure_frac=0.15,
        min_confidence=0.65,
        enable_session_filter=False,
        enable_regime_bias=False,
        enable_symbol_accuracy=True,
    )

    # Strong pair should get full allocation; weak pair ~0.30× of that
    btc = abs(decision.target_exposures["BTCUSDT"])
    eth = abs(decision.target_exposures["ETHUSDT"])
    assert btc > eth
    assert eth == pytest.approx(btc * 0.30, rel=0.01)


def test_allocate_signals_no_data_symbol_accuracy_is_neutral() -> None:
    """symbol_hit_rate=None should not dampen (neutral 1.0×)."""
    with_data = StrategySignal(
        symbol="BTCUSDT", timeframe="1h", horizon_bars=4,
        signal="BUY", confidence=0.80, uncertainty=0.0,
        symbol_hit_rate=0.60,
    )
    no_data = StrategySignal(
        symbol="ETHUSDT", timeframe="1h", horizon_bars=4,
        signal="BUY", confidence=0.80, uncertainty=0.0,
        symbol_hit_rate=None,  # no data → neutral 1.0
    )

    decision = allocate_signals(
        [with_data, no_data],
        total_risk_budget_frac=1.0,
        max_symbol_exposure_frac=0.15,
        min_confidence=0.65,
        enable_session_filter=False,
        enable_regime_bias=False,
        enable_symbol_accuracy=True,
    )

    btc = abs(decision.target_exposures["BTCUSDT"])
    eth = abs(decision.target_exposures["ETHUSDT"])
    assert btc == pytest.approx(eth)  # both get 1.0× multiplier


def test_portfolio_risk_policy_validate_args() -> None:
    with pytest.raises(ValueError):
        PortfolioRiskPolicy(max_symbol_exposure_frac=0.0)

    with pytest.raises(ValueError):
        PortfolioRiskPolicy(max_gross_exposure_frac=0.1, max_net_exposure_frac=0.2)
