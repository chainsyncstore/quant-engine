"""Runtime kill-switch evaluation for v2 production safety."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class MonitoringSnapshot:
    """Current runtime health indicators for a session/service."""

    feature_drift_alert: bool = False
    confidence_collapse_alert: bool = False
    execution_anomaly_rate: float = 0.0
    connectivity_error_rate: float = 0.0
    hard_risk_breach: bool = False

    def __post_init__(self) -> None:
        if not 0.0 <= self.execution_anomaly_rate <= 1.0:
            raise ValueError("execution_anomaly_rate must be in [0, 1]")
        if not 0.0 <= self.connectivity_error_rate <= 1.0:
            raise ValueError("connectivity_error_rate must be in [0, 1]")


@dataclass(frozen=True)
class KillSwitchConfig:
    """Thresholds controlling kill-switch activation."""

    max_execution_anomaly_rate: float = 0.10
    max_connectivity_error_rate: float = 0.20
    pause_on_feature_drift: bool = True
    pause_on_confidence_collapse: bool = True

    def __post_init__(self) -> None:
        if not 0.0 <= self.max_execution_anomaly_rate <= 1.0:
            raise ValueError("max_execution_anomaly_rate must be in [0, 1]")
        if not 0.0 <= self.max_connectivity_error_rate <= 1.0:
            raise ValueError("max_connectivity_error_rate must be in [0, 1]")


@dataclass(frozen=True)
class KillSwitchEvaluation:
    """Decision output from kill-switch checks."""

    pause_trading: bool
    reasons: tuple[str, ...] = field(default_factory=tuple)


def evaluate_kill_switch(
    snapshot: MonitoringSnapshot,
    *,
    config: KillSwitchConfig = KillSwitchConfig(),
) -> KillSwitchEvaluation:
    """Evaluate kill-switch conditions and return pause decision."""

    reasons: list[str] = []

    if config.pause_on_feature_drift and snapshot.feature_drift_alert:
        reasons.append("feature_drift")
    if config.pause_on_confidence_collapse and snapshot.confidence_collapse_alert:
        reasons.append("confidence_collapse")
    if snapshot.execution_anomaly_rate > config.max_execution_anomaly_rate:
        reasons.append("execution_anomaly")
    if snapshot.connectivity_error_rate > config.max_connectivity_error_rate:
        reasons.append("connectivity")
    if snapshot.hard_risk_breach:
        reasons.append("hard_risk_breach")

    return KillSwitchEvaluation(
        pause_trading=bool(reasons),
        reasons=tuple(reasons),
    )
