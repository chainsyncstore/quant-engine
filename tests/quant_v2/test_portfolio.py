from __future__ import annotations

import pytest

from quant_v2.contracts import MarketRiskSnapshot, ModelSourceDetails, StrategySignal
from quant_v2.portfolio.allocation import allocate_signals
from quant_v2.portfolio.risk_policy import PortfolioRiskPolicy


def _signal(
    symbol: str,
    side: str,
    confidence: float,
    uncertainty: float | None = None,
    regime: int | None = None,
) -> StrategySignal:
    return StrategySignal(
        symbol=symbol,
        timeframe="1h",
        horizon_bars=4,
        signal=side,
        confidence=confidence,
        uncertainty=uncertainty,
        regime=regime,
    )


def _selloff_snapshot(
    *,
    active: bool = True,
    strong_short_confidence: float = 0.75,
    short_net_cap_frac: float = 0.15,
) -> MarketRiskSnapshot:
    return MarketRiskSnapshot(
        lookback_hours=30,
        symbols_evaluated=5,
        down_ratio=0.80 if active else 0.40,
        median_return=-0.025 if active else 0.002,
        btc_return=-0.03 if active else 0.01,
        broad_selloff=active,
        strong_short_confidence=strong_short_confidence,
        short_net_cap_frac=short_net_cap_frac,
    )


def _sources(
    *,
    lgbm: float | None,
    chronos: float | None,
    final: float,
    agreement: bool | None,
    chronos_enabled: bool = True,
) -> ModelSourceDetails:
    def direction(value: float | None) -> str | None:
        if value is None:
            return None
        if value > 0.5:
            return "BUY"
        if value < 0.5:
            return "SELL"
        return "HOLD"

    return ModelSourceDetails(
        lgbm_probability=lgbm,
        chronos_probability=chronos,
        final_probability=final,
        lgbm_direction=direction(lgbm),  # type: ignore[arg-type]
        chronos_direction=direction(chronos),  # type: ignore[arg-type]
        agreement=agreement,
        chronos_enabled=chronos_enabled,
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
        enable_event_gate=False,
        enable_model_agreement=False,
        enable_cost_gate=False,
    )

    assert decision.target_exposures["BTCUSDT"] == pytest.approx(0.05)
    assert decision.target_exposures["ETHUSDT"] == pytest.approx(0.025)
    assert decision.gross_exposure == pytest.approx(0.075)


def test_market_short_guard_inactive_allows_normal_sell() -> None:
    signal = StrategySignal(
        symbol="ETHUSDT",
        timeframe="1h",
        horizon_bars=4,
        signal="SELL",
        confidence=0.70,
        uncertainty=0.0,
        market_risk=_selloff_snapshot(active=False),
    )

    decision = allocate_signals(
        [signal],
        min_confidence=0.65,
        enable_session_filter=False,
        enable_regime_bias=False,
        enable_cost_gate=False,
    )

    assert decision.target_exposures["ETHUSDT"] < 0.0
    assert decision.constraints_applied == ()


def test_market_short_guard_blocks_weak_new_sell() -> None:
    signal = StrategySignal(
        symbol="ETHUSDT",
        timeframe="1h",
        horizon_bars=4,
        signal="SELL",
        confidence=0.70,
        uncertainty=0.0,
        market_risk=_selloff_snapshot(active=True),
    )

    decision = allocate_signals(
        [signal],
        min_confidence=0.65,
        enable_session_filter=False,
        enable_regime_bias=False,
        enable_cost_gate=False,
    )

    assert decision.target_exposures == {}
    assert "market_short_guard" in decision.skipped_symbols["ETHUSDT"]
    assert "market_short_guard_block" in decision.constraints_applied


def test_market_short_guard_allows_very_strong_sell_but_caps_net_short() -> None:
    snapshot = _selloff_snapshot(active=True, short_net_cap_frac=0.05)
    signals = [
        StrategySignal(
            symbol="ETHUSDT",
            timeframe="1h",
            horizon_bars=4,
            signal="SELL",
            confidence=0.82,
            uncertainty=0.0,
            market_risk=snapshot,
        ),
        StrategySignal(
            symbol="SOLUSDT",
            timeframe="1h",
            horizon_bars=4,
            signal="SELL",
            confidence=0.84,
            uncertainty=0.0,
            market_risk=snapshot,
        ),
    ]

    decision = allocate_signals(
        signals,
        total_risk_budget_frac=1.0,
        max_symbol_exposure_frac=0.15,
        min_confidence=0.65,
        enable_session_filter=False,
        enable_regime_bias=False,
        enable_cost_gate=False,
    )

    assert decision.target_exposures
    assert all(value < 0.0 for value in decision.target_exposures.values())
    assert decision.net_exposure >= -0.05 - 1e-12
    assert "market_short_guard_net_cap" in decision.constraints_applied


def test_regime2_sell_allocation_is_dampened_by_default(monkeypatch) -> None:
    monkeypatch.delenv("BOT_V2_REGIME2_SELL_ALLOCATION_MULT", raising=False)

    regime1 = allocate_signals(
        [_signal("ETHUSDT", "SELL", 0.80, uncertainty=0.0, regime=1)],
        total_risk_budget_frac=1.0,
        max_symbol_exposure_frac=0.15,
        min_confidence=0.65,
        enable_session_filter=False,
        enable_regime_bias=False,
        enable_symbol_accuracy=False,
        enable_event_gate=False,
        enable_model_agreement=False,
        enable_cost_gate=False,
    )
    regime2 = allocate_signals(
        [_signal("ETHUSDT", "SELL", 0.80, uncertainty=0.0, regime=2)],
        total_risk_budget_frac=1.0,
        max_symbol_exposure_frac=0.15,
        min_confidence=0.65,
        enable_session_filter=False,
        enable_regime_bias=False,
        enable_symbol_accuracy=False,
        enable_event_gate=False,
        enable_model_agreement=False,
        enable_cost_gate=False,
    )

    assert regime1.target_exposures["ETHUSDT"] == pytest.approx(-0.15)
    assert regime2.target_exposures["ETHUSDT"] == pytest.approx(-0.075)
    assert "regime2_sell_dampen" in regime2.constraints_applied


def test_regime2_sell_allocation_multiplier_env_override(monkeypatch) -> None:
    monkeypatch.setenv("BOT_V2_REGIME2_SELL_ALLOCATION_MULT", "0.25")

    decision = allocate_signals(
        [_signal("ETHUSDT", "SELL", 0.80, uncertainty=0.0, regime=2)],
        total_risk_budget_frac=1.0,
        max_symbol_exposure_frac=0.15,
        min_confidence=0.65,
        enable_session_filter=False,
        enable_regime_bias=False,
        enable_symbol_accuracy=False,
        enable_event_gate=False,
        enable_model_agreement=False,
        enable_cost_gate=False,
    )

    assert decision.target_exposures["ETHUSDT"] == pytest.approx(-0.0375)
    assert "regime2_sell_dampen" in decision.constraints_applied


def test_regime2_sell_against_existing_long_is_flatten_only(monkeypatch) -> None:
    monkeypatch.delenv("BOT_V2_REGIME2_SELL_ALLOCATION_MULT", raising=False)

    decision = allocate_signals(
        [_signal("ETHUSDT", "SELL", 0.82, uncertainty=0.0, regime=2)],
        total_risk_budget_frac=1.0,
        max_symbol_exposure_frac=0.15,
        min_confidence=0.65,
        enable_session_filter=False,
        enable_regime_bias=False,
        enable_symbol_accuracy=False,
        enable_event_gate=False,
        enable_model_agreement=False,
        enable_cost_gate=False,
        current_positions={"ETHUSDT": 0.05},
    )

    assert decision.target_exposures == {}
    assert decision.skipped_symbols["ETHUSDT"] == "regime2_sell_flatten_only"
    assert "regime2_sell_flatten_only" in decision.constraints_applied


def test_chronos_agreement_allows_normal_entry(monkeypatch) -> None:
    monkeypatch.setenv("BOT_V2_REQUIRE_CHRONOS_AGREEMENT_FOR_ENTRY", "1")

    signal = StrategySignal(
        symbol="BTCUSDT",
        timeframe="1h",
        horizon_bars=4,
        signal="BUY",
        confidence=0.80,
        uncertainty=0.0,
        model_sources=_sources(lgbm=0.82, chronos=0.66, final=0.764, agreement=True),
    )

    decision = allocate_signals(
        [signal],
        min_confidence=0.65,
        enable_session_filter=False,
        enable_regime_bias=False,
        enable_symbol_accuracy=False,
        enable_event_gate=False,
        enable_model_agreement=False,
        enable_cost_gate=False,
    )

    assert decision.target_exposures["BTCUSDT"] > 0.0
    assert not any(item.startswith("chronos_") for item in decision.constraints_applied)


def test_chronos_disagreement_blocks_fresh_sell_by_default(monkeypatch) -> None:
    monkeypatch.setenv("BOT_V2_REQUIRE_CHRONOS_AGREEMENT_FOR_ENTRY", "1")

    signal = StrategySignal(
        symbol="ETHUSDT",
        timeframe="1h",
        horizon_bars=4,
        signal="SELL",
        confidence=0.72,
        uncertainty=0.0,
        model_sources=_sources(lgbm=0.28, chronos=0.62, final=0.399, agreement=False),
    )

    decision = allocate_signals(
        [signal],
        min_confidence=0.65,
        enable_session_filter=False,
        enable_regime_bias=False,
        enable_symbol_accuracy=False,
        enable_event_gate=False,
        enable_model_agreement=False,
        enable_cost_gate=False,
    )

    assert decision.target_exposures == {}
    assert decision.skipped_symbols["ETHUSDT"] == "chronos_disagreement_entry_block"
    assert "chronos_disagreement_entry_block" in decision.constraints_applied


def test_chronos_disagreement_blocks_fresh_buy_by_default(monkeypatch) -> None:
    monkeypatch.setenv("BOT_V2_REQUIRE_CHRONOS_AGREEMENT_FOR_ENTRY", "1")

    signal = StrategySignal(
        symbol="BTCUSDT",
        timeframe="1h",
        horizon_bars=4,
        signal="BUY",
        confidence=0.72,
        uncertainty=0.0,
        model_sources=_sources(lgbm=0.72, chronos=0.34, final=0.587, agreement=False),
    )

    decision = allocate_signals(
        [signal],
        min_confidence=0.65,
        enable_session_filter=False,
        enable_regime_bias=False,
        enable_symbol_accuracy=False,
        enable_event_gate=False,
        enable_model_agreement=False,
        enable_cost_gate=False,
    )

    assert decision.target_exposures == {}
    assert decision.skipped_symbols["BTCUSDT"] == "chronos_disagreement_entry_block"
    assert "chronos_disagreement_entry_block" in decision.constraints_applied


def test_chronos_disagreement_dampens_when_agreement_not_required(monkeypatch) -> None:
    monkeypatch.setenv("BOT_V2_REQUIRE_CHRONOS_AGREEMENT_FOR_ENTRY", "0")
    monkeypatch.setenv("BOT_V2_CHRONOS_DISAGREEMENT_MULT", "0.25")

    signal = StrategySignal(
        symbol="BTCUSDT",
        timeframe="1h",
        horizon_bars=4,
        signal="BUY",
        confidence=0.80,
        uncertainty=0.0,
        model_sources=_sources(lgbm=0.75, chronos=0.30, final=0.5925, agreement=False),
    )

    decision = allocate_signals(
        [signal],
        min_confidence=0.65,
        enable_session_filter=False,
        enable_regime_bias=False,
        enable_symbol_accuracy=False,
        enable_event_gate=False,
        enable_model_agreement=False,
        enable_cost_gate=False,
    )

    assert decision.target_exposures["BTCUSDT"] == pytest.approx(0.0375)
    assert "chronos_disagreement_dampen" in decision.constraints_applied


def test_chronos_disagreement_extreme_lgbm_confidence_overrides(monkeypatch) -> None:
    monkeypatch.setenv("BOT_V2_REQUIRE_CHRONOS_AGREEMENT_FOR_ENTRY", "1")
    monkeypatch.setenv("BOT_V2_CHRONOS_EXTREME_CONFIDENCE", "0.80")

    signal = StrategySignal(
        symbol="ETHUSDT",
        timeframe="1h",
        horizon_bars=4,
        signal="SELL",
        confidence=0.85,
        uncertainty=0.0,
        model_sources=_sources(lgbm=0.15, chronos=0.58, final=0.3005, agreement=False),
    )

    decision = allocate_signals(
        [signal],
        min_confidence=0.65,
        enable_session_filter=False,
        enable_regime_bias=False,
        enable_symbol_accuracy=False,
        enable_event_gate=False,
        enable_model_agreement=False,
        enable_cost_gate=False,
    )

    assert decision.target_exposures["ETHUSDT"] < 0.0
    assert "chronos_disagreement_extreme_override" in decision.constraints_applied


def test_chronos_disagreement_still_allows_flattening(monkeypatch) -> None:
    monkeypatch.setenv("BOT_V2_REQUIRE_CHRONOS_AGREEMENT_FOR_ENTRY", "1")

    signal = StrategySignal(
        symbol="ETHUSDT",
        timeframe="1h",
        horizon_bars=4,
        signal="SELL",
        confidence=0.72,
        uncertainty=0.0,
        model_sources=_sources(lgbm=0.28, chronos=0.62, final=0.399, agreement=False),
    )

    decision = allocate_signals(
        [signal],
        min_confidence=0.65,
        enable_session_filter=False,
        enable_regime_bias=False,
        enable_symbol_accuracy=False,
        enable_event_gate=False,
        enable_model_agreement=False,
        enable_cost_gate=False,
        current_positions={"ETHUSDT": 0.05},
    )

    assert decision.target_exposures == {}
    assert decision.skipped_symbols["ETHUSDT"] == "chronos_disagreement_flatten_only"
    assert "chronos_disagreement_flatten_only" in decision.constraints_applied


def test_chronos_unavailable_with_required_agreement_blocks_fresh_entry(monkeypatch) -> None:
    monkeypatch.setenv("BOT_V2_REQUIRE_CHRONOS_AGREEMENT_FOR_ENTRY", "1")

    signal = StrategySignal(
        symbol="BTCUSDT",
        timeframe="1h",
        horizon_bars=4,
        signal="BUY",
        confidence=0.80,
        uncertainty=0.0,
        model_sources=_sources(
            lgbm=0.80,
            chronos=None,
            final=0.80,
            agreement=None,
            chronos_enabled=True,
        ),
    )

    decision = allocate_signals(
        [signal],
        min_confidence=0.65,
        enable_session_filter=False,
        enable_regime_bias=False,
        enable_symbol_accuracy=False,
        enable_event_gate=False,
        enable_model_agreement=False,
        enable_cost_gate=False,
    )

    assert decision.target_exposures == {}
    assert decision.skipped_symbols["BTCUSDT"] == "chronos_unavailable_entry_block"
    assert "chronos_unavailable_entry_block" in decision.constraints_applied


def test_chronos_disabled_falls_back_to_normal_lgbm_path(monkeypatch) -> None:
    monkeypatch.setenv("BOT_V2_REQUIRE_CHRONOS_AGREEMENT_FOR_ENTRY", "1")

    signal = StrategySignal(
        symbol="BTCUSDT",
        timeframe="1h",
        horizon_bars=4,
        signal="BUY",
        confidence=0.80,
        uncertainty=0.0,
        model_sources=_sources(
            lgbm=0.80,
            chronos=None,
            final=0.80,
            agreement=None,
            chronos_enabled=False,
        ),
    )

    decision = allocate_signals(
        [signal],
        min_confidence=0.65,
        enable_session_filter=False,
        enable_regime_bias=False,
        enable_symbol_accuracy=False,
        enable_event_gate=False,
        enable_model_agreement=False,
        enable_cost_gate=False,
    )

    assert decision.target_exposures["BTCUSDT"] > 0.0
    assert not any(item.startswith("chronos_") for item in decision.constraints_applied)


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


def test_allocate_signals_event_gate_dampens_heavily() -> None:
    """event_gate_mult=0.10 should heavily dampen allocation."""
    normal_signal = StrategySignal(
        symbol="BTCUSDT", timeframe="1h", horizon_bars=4,
        signal="BUY", confidence=0.80, uncertainty=0.0,
        event_gate_mult=None,  # neutral 1.0×
    )
    dampened_signal = StrategySignal(
        symbol="ETHUSDT", timeframe="1h", horizon_bars=4,
        signal="BUY", confidence=0.80, uncertainty=0.0,
        event_gate_mult=0.10,  # near-veto
    )

    decision = allocate_signals(
        [normal_signal, dampened_signal],
        total_risk_budget_frac=1.0,
        max_symbol_exposure_frac=0.15,
        min_confidence=0.65,
        enable_session_filter=False,
        enable_regime_bias=False,
        enable_symbol_accuracy=False,
        enable_event_gate=True,
    )

    btc = abs(decision.target_exposures["BTCUSDT"])
    eth = abs(decision.target_exposures["ETHUSDT"])
    assert btc > eth
    assert eth == pytest.approx(btc * 0.10, rel=0.01)


def test_allocate_signals_event_gate_none_is_neutral() -> None:
    """event_gate_mult=None should not dampen (neutral 1.0×)."""
    sig_a = StrategySignal(
        symbol="BTCUSDT", timeframe="1h", horizon_bars=4,
        signal="BUY", confidence=0.80, uncertainty=0.0,
        event_gate_mult=None,
    )
    sig_b = StrategySignal(
        symbol="ETHUSDT", timeframe="1h", horizon_bars=4,
        signal="BUY", confidence=0.80, uncertainty=0.0,
        event_gate_mult=None,
    )

    decision = allocate_signals(
        [sig_a, sig_b],
        total_risk_budget_frac=1.0,
        max_symbol_exposure_frac=0.15,
        min_confidence=0.65,
        enable_session_filter=False,
        enable_regime_bias=False,
        enable_symbol_accuracy=False,
        enable_event_gate=True,
    )

    btc = abs(decision.target_exposures["BTCUSDT"])
    eth = abs(decision.target_exposures["ETHUSDT"])
    assert btc == pytest.approx(eth)


def test_allocate_signals_event_gate_disabled_ignores_field() -> None:
    """enable_event_gate=False should ignore event_gate_mult entirely."""
    sig_a = StrategySignal(
        symbol="BTCUSDT", timeframe="1h", horizon_bars=4,
        signal="BUY", confidence=0.80, uncertainty=0.0,
        event_gate_mult=None,
    )
    sig_b = StrategySignal(
        symbol="ETHUSDT", timeframe="1h", horizon_bars=4,
        signal="BUY", confidence=0.80, uncertainty=0.0,
        event_gate_mult=0.10,  # would dampen if enabled
    )

    decision = allocate_signals(
        [sig_a, sig_b],
        total_risk_budget_frac=1.0,
        max_symbol_exposure_frac=0.15,
        min_confidence=0.65,
        enable_session_filter=False,
        enable_regime_bias=False,
        enable_symbol_accuracy=False,
        enable_event_gate=False,  # disabled
    )

    btc = abs(decision.target_exposures["BTCUSDT"])
    eth = abs(decision.target_exposures["ETHUSDT"])
    # Both should be equal since event gate is disabled
    assert btc == pytest.approx(eth)


def test_allocate_signals_model_agreement_strong_is_full() -> None:
    """model_agreement >= 0.8 should give full allocation (1.0×)."""
    strong_agree = StrategySignal(
        symbol="BTCUSDT", timeframe="1h", horizon_bars=4,
        signal="BUY", confidence=0.80, uncertainty=0.0,
        model_agreement=0.9,  # strong → 1.0×
    )
    mild_agree = StrategySignal(
        symbol="ETHUSDT", timeframe="1h", horizon_bars=4,
        signal="BUY", confidence=0.80, uncertainty=0.0,
        model_agreement=0.6,  # mild → 0.85×
    )

    decision = allocate_signals(
        [strong_agree, mild_agree],
        total_risk_budget_frac=1.0,
        max_symbol_exposure_frac=0.15,
        min_confidence=0.65,
        enable_session_filter=False,
        enable_regime_bias=False,
        enable_symbol_accuracy=False,
        enable_event_gate=False,
        enable_model_agreement=True,
    )

    btc = abs(decision.target_exposures["BTCUSDT"])
    eth = abs(decision.target_exposures["ETHUSDT"])
    assert btc > eth
    assert eth == pytest.approx(btc * 0.85, rel=0.01)


def test_allocate_signals_model_agreement_disagree_dampens() -> None:
    """model_agreement < 0.5 should dampen to 0.60×."""
    full_signal = StrategySignal(
        symbol="BTCUSDT", timeframe="1h", horizon_bars=4,
        signal="BUY", confidence=0.80, uncertainty=0.0,
        model_agreement=0.9,  # 1.0×
    )
    disagree_signal = StrategySignal(
        symbol="ETHUSDT", timeframe="1h", horizon_bars=4,
        signal="BUY", confidence=0.80, uncertainty=0.0,
        model_agreement=0.2,  # < 0.5 → 0.60×
    )

    decision = allocate_signals(
        [full_signal, disagree_signal],
        total_risk_budget_frac=1.0,
        max_symbol_exposure_frac=0.15,
        min_confidence=0.65,
        enable_session_filter=False,
        enable_regime_bias=False,
        enable_symbol_accuracy=False,
        enable_event_gate=False,
        enable_model_agreement=True,
    )

    btc = abs(decision.target_exposures["BTCUSDT"])
    eth = abs(decision.target_exposures["ETHUSDT"])
    assert btc > eth
    assert eth == pytest.approx(btc * 0.60, rel=0.01)


def test_allocate_signals_model_agreement_none_is_neutral() -> None:
    """model_agreement=None should give 1.0× (no penalty — no ensemble data available)."""
    with_data = StrategySignal(
        symbol="BTCUSDT", timeframe="1h", horizon_bars=4,
        signal="BUY", confidence=0.80, uncertainty=0.0,
        model_agreement=0.9,  # strong → 1.0×
    )
    no_data = StrategySignal(
        symbol="ETHUSDT", timeframe="1h", horizon_bars=4,
        signal="BUY", confidence=0.80, uncertainty=0.0,
        model_agreement=None,  # no ensemble → 1.0× (no penalty)
    )

    decision = allocate_signals(
        [with_data, no_data],
        total_risk_budget_frac=1.0,
        max_symbol_exposure_frac=0.15,
        min_confidence=0.65,
        enable_session_filter=False,
        enable_regime_bias=False,
        enable_symbol_accuracy=False,
        enable_event_gate=False,
        enable_model_agreement=True,
    )

    btc = abs(decision.target_exposures["BTCUSDT"])
    eth = abs(decision.target_exposures["ETHUSDT"])
    # Both get 1.0× — no penalty for missing ensemble data
    assert eth == pytest.approx(btc, rel=0.01)


def test_allocate_signals_model_agreement_disabled_ignores_field() -> None:
    """enable_model_agreement=False should ignore model_agreement entirely."""
    sig_a = StrategySignal(
        symbol="BTCUSDT", timeframe="1h", horizon_bars=4,
        signal="BUY", confidence=0.80, uncertainty=0.0,
        model_agreement=0.9,
    )
    sig_b = StrategySignal(
        symbol="ETHUSDT", timeframe="1h", horizon_bars=4,
        signal="BUY", confidence=0.80, uncertainty=0.0,
        model_agreement=0.1,  # would dampen if enabled
    )

    decision = allocate_signals(
        [sig_a, sig_b],
        total_risk_budget_frac=1.0,
        max_symbol_exposure_frac=0.15,
        min_confidence=0.65,
        enable_session_filter=False,
        enable_regime_bias=False,
        enable_symbol_accuracy=False,
        enable_event_gate=False,
        enable_model_agreement=False,  # disabled
    )

    btc = abs(decision.target_exposures["BTCUSDT"])
    eth = abs(decision.target_exposures["ETHUSDT"])
    assert btc == pytest.approx(eth)


def test_portfolio_risk_policy_validate_args() -> None:
    with pytest.raises(ValueError):
        PortfolioRiskPolicy(max_symbol_exposure_frac=0.0)

    with pytest.raises(ValueError):
        PortfolioRiskPolicy(max_gross_exposure_frac=0.1, max_net_exposure_frac=0.2)
