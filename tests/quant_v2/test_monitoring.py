from __future__ import annotations

import pytest

from quant_v2.monitoring.kill_switch import (
    KillSwitchConfig,
    MonitoringSnapshot,
    evaluate_kill_switch,
)
from quant_v2.monitoring.shadow_drift import compute_shadow_live_drift


def test_evaluate_kill_switch_pauses_on_multiple_reasons() -> None:
    snapshot = MonitoringSnapshot(
        feature_drift_alert=True,
        confidence_collapse_alert=False,
        execution_anomaly_rate=0.15,
        connectivity_error_rate=0.05,
        hard_risk_breach=True,
    )
    config = KillSwitchConfig(max_execution_anomaly_rate=0.10, max_connectivity_error_rate=0.20)

    result = evaluate_kill_switch(snapshot, config=config)

    assert result.pause_trading is True
    assert "feature_drift" in result.reasons
    assert "execution_anomaly" in result.reasons
    assert "hard_risk_breach" in result.reasons


def test_evaluate_kill_switch_no_pause_when_all_clear() -> None:
    snapshot = MonitoringSnapshot(
        feature_drift_alert=False,
        confidence_collapse_alert=False,
        execution_anomaly_rate=0.01,
        connectivity_error_rate=0.02,
        hard_risk_breach=False,
    )

    result = evaluate_kill_switch(snapshot)

    assert result.pause_trading is False
    assert result.reasons == ()


def test_compute_shadow_live_drift_metrics_and_tolerance() -> None:
    stats = compute_shadow_live_drift(
        [0.62, 0.30, 0.58, 0.44],
        [0.60, 0.35, 0.55, 0.48],
        threshold=0.5,
    )

    assert stats.n_samples == 4
    assert stats.mean_abs_error == pytest.approx(0.035, abs=1e-6)
    assert stats.directional_agreement == 1.0
    assert stats.within_tolerance(max_mae=0.10, min_directional_agreement=0.75) is True


def test_compute_shadow_live_drift_validate_shapes() -> None:
    with pytest.raises(ValueError):
        compute_shadow_live_drift([0.5, 0.6], [0.5])
