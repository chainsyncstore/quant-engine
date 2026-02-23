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
        total_risk_budget_frac=0.12,
        max_symbol_exposure_frac=0.05,
        min_confidence=0.55,
    )

    assert "BTCUSDT" in decision.target_exposures
    assert "ETHUSDT" in decision.target_exposures
    assert decision.target_exposures["BTCUSDT"] > 0
    assert decision.target_exposures["ETHUSDT"] < 0
    assert abs(decision.target_exposures["BTCUSDT"]) <= 0.05
    assert abs(decision.target_exposures["ETHUSDT"]) <= 0.05
    assert decision.skipped_symbols["SOLUSDT"] == "hold"


def test_allocate_signals_skips_low_confidence_and_drift() -> None:
    decision = allocate_signals(
        [
            _signal("BTCUSDT", "BUY", 0.52),
            _signal("ETHUSDT", "DRIFT_ALERT", 0.80),
        ],
        total_risk_budget_frac=0.10,
        min_confidence=0.55,
    )

    assert decision.target_exposures == {}
    assert decision.gross_exposure == 0.0
    assert decision.skipped_symbols["BTCUSDT"].startswith("confidence<")
    assert decision.skipped_symbols["ETHUSDT"] == "drift_alert"


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


def test_portfolio_risk_policy_validate_args() -> None:
    with pytest.raises(ValueError):
        PortfolioRiskPolicy(max_symbol_exposure_frac=0.0)

    with pytest.raises(ValueError):
        PortfolioRiskPolicy(max_gross_exposure_frac=0.1, max_net_exposure_frac=0.2)
