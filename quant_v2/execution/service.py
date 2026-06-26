"""Execution service boundary used by Telegram and other control-plane adapters."""

from __future__ import annotations

import asyncio
from collections.abc import Iterable
from dataclasses import replace
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
import functools
import logging
import os
from time import perf_counter
from typing import Any, Callable, Protocol

import numpy as np
import pandas as pd

from quant_v2.config import get_runtime_profile
from quant_v2.contracts import OrderPlan, PortfolioSnapshot, RiskSnapshot, StrategySignal
from quant_v2.execution.adapters import ExecutionResult
from quant_v2.execution.idempotency import build_idempotency_key
from quant_v2.execution.planner import PlannerConfig, build_execution_intents
from quant_v2.execution.reconciler import reconcile_target_exposures
from quant_v2.execution.state_wal import LifecycleStateRecord, validate_lifecycle_transition
from quant_v2.monitoring.kill_switch import (
    KillSwitchConfig,
    KillSwitchEvaluation,
    MonitoringSnapshot,
    evaluate_kill_switch,
)
from quant_v2.portfolio.optimizer import RiskParityOptimizer
from quant_v2.portfolio.risk_policy import (
    HardRiskLimits,
    OperatingRiskLimits,
    PortfolioRiskPolicy,
    build_dynamic_operating_limits,
)

logger = logging.getLogger(__name__)


def _default_dynamic_risk_policy() -> OperatingRiskLimits:
    return OperatingRiskLimits.from_hard_limits(HardRiskLimits())


def _default_operating_risk_policy() -> OperatingRiskLimits:
    dynamic = _default_dynamic_risk_policy()
    return dynamic.scaled(
        dynamic.effective_headroom_ratio,
        limit_source="operating_headroom",
    )


def _build_lifecycle_state_record(
    previous: LifecycleStateRecord | None,
    *,
    state: str,
    owner: str,
    reason: str,
    policy_version: str,
    retry_count: int | None = None,
) -> LifecycleStateRecord:
    validate_lifecycle_transition(previous.state if previous is not None else None, state)

    now = datetime.now(timezone.utc).isoformat()
    normalized_policy_version = str(policy_version or "").strip()
    if not normalized_policy_version:
        normalized_policy_version = (
            previous.policy_version
            if previous is not None and previous.policy_version
            else HardRiskLimits().policy_version
        )

    normalized_retry_count = 0 if retry_count is None else max(0, int(retry_count))
    if previous is not None and previous.state == str(state).strip().upper() and retry_count is None:
        normalized_retry_count = previous.retry_count + 1

    return LifecycleStateRecord(
        state=state,
        owner=owner,
        retry_count=normalized_retry_count,
        reason=reason,
        policy_version=normalized_policy_version,
        created_at=previous.created_at if previous is not None else now,
        transitioned_at=now if previous is None or previous.state != str(state).strip().upper() else previous.transitioned_at,
        updated_at=now,
    )


@dataclass(frozen=True)
class SessionRequest:
    """Start-session payload for execution backends."""

    user_id: int
    live: bool
    strategy_profile: str = "core_v2"
    universe: tuple[str, ...] = ()
    credentials: dict[str, str] = field(default_factory=dict)
    requested_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self) -> None:
        if self.user_id <= 0:
            raise ValueError("user_id must be positive")
        if not self.strategy_profile:
            raise ValueError("strategy_profile cannot be empty")


@dataclass(frozen=True)
class ExecutionDiagnostics:
    """Execution-quality telemetry tracked per routed session."""

    total_orders: int = 0
    accepted_orders: int = 0
    rejected_orders: int = 0
    reject_rate: float = 0.0
    slippage_sample_count: int = 0
    avg_adverse_slippage_bps: float = 0.0
    entry_orders: int = 0
    rebalance_orders: int = 0
    exit_orders: int = 0
    skipped_by_filter: int = 0
    skipped_by_deadband: int = 0
    paused_cycles: int = 0
    blocked_actionable_signals: int = 0
    routed_signals_total: int = 0
    routed_buy_signals: int = 0
    routed_sell_signals: int = 0
    routed_actionable_signals: int = 0
    live_go_no_go_passed: bool = True
    rollback_required: bool = False
    rollout_failure_streak: int = 0
    rollout_gate_reasons: tuple[str, ...] = field(default_factory=tuple)
    effective_symbol_cap_frac: float = 0.0
    effective_gross_cap_frac: float = 0.0
    effective_net_cap_frac: float = 0.0
    hard_symbol_cap_frac: float = 0.0
    hard_gross_cap_frac: float = 0.0
    hard_net_cap_frac: float = 0.0
    target_headroom_ratio: float = 0.85
    reserve_capacity_frac: float = 0.0
    adverse_mark_buffer_frac: float = 0.0
    risk_policy_version: str = ""


@dataclass(frozen=True)
class HardRiskPauseEvent:
    """Durable pause event emitted when a hard risk breach is confirmed."""

    user_id: int
    reason: str
    triggered_at: datetime
    breach_type: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RouteAuditEvent:
    """Structured execution-route decision for durable telemetry."""

    user_id: int
    created_at: datetime
    pause_state: str
    is_active: bool | None
    live_mode: bool
    symbol: str
    side: str
    quantity: float
    before_position: float | None
    after_position: float | None
    action_class: str
    reason: str
    accepted: bool | None = None
    status: str = ""
    order_id: str = ""
    idempotency_key: str = ""
    correlation_id: str = ""
    mark_price: float = 0.0
    risk_policy_version: str = ""
    hard_symbol_cap_frac: float = 0.0
    hard_gross_cap_frac: float = 0.0
    hard_net_cap_frac: float = 0.0
    dynamic_symbol_cap_frac: float = 0.0
    dynamic_gross_cap_frac: float = 0.0
    dynamic_net_cap_frac: float = 0.0
    operating_symbol_cap_frac: float = 0.0
    operating_gross_cap_frac: float = 0.0
    operating_net_cap_frac: float = 0.0
    target_headroom_ratio: float = 0.85
    reserve_capacity_frac: float = 0.0
    adverse_mark_buffer_frac: float = 0.0
    current_symbol_exposure_frac: float = 0.0
    projected_symbol_exposure_frac: float = 0.0
    current_gross_exposure_frac: float = 0.0
    projected_gross_exposure_frac: float = 0.0
    current_net_exposure_frac: float = 0.0
    projected_net_exposure_frac: float = 0.0
    symbol_headroom_frac: float = 0.0
    gross_headroom_frac: float = 0.0
    net_headroom_frac: float = 0.0


@dataclass(frozen=True)
class ExecutionStageTelemetry:
    """Structured timing event for routing and persistence stages."""

    user_id: int
    created_at: datetime
    stage: str
    duration_ms: float
    correlation_id: str = ""
    status: str = ""
    detail: str = ""


@dataclass(frozen=True)
class LifecycleRules:
    """Configurable position lifecycle controls for auto-exit behaviors."""

    auto_close_horizon_bars: int = 0
    stop_loss_pct: float = 0.0
    take_profit_atr_mult: float = 0.0

    def __post_init__(self) -> None:
        if self.auto_close_horizon_bars < 0:
            raise ValueError("auto_close_horizon_bars must be >= 0")
        if not 0.0 <= self.stop_loss_pct < 1.0:
            raise ValueError("stop_loss_pct must be within [0, 1)")
        if self.take_profit_atr_mult < 0.0:
            raise ValueError("take_profit_atr_mult must be >= 0")


class ExecutionService(Protocol):
    """Execution abstraction to decouple Telegram control-plane from runtime internals."""

    async def start_session(self, request: SessionRequest) -> bool:
        """Start a session. Returns False when session already exists."""

    async def stop_session(self, user_id: int) -> bool:
        """Stop session. Returns False when no running session exists."""

    def reset_session_state(self, user_id: int) -> bool:
        """Reset in-session paper state. Returns False when unsupported/unavailable."""

    def is_running(self, user_id: int) -> bool:
        """Return current running state for a user."""

    def get_portfolio_snapshot(self, user_id: int) -> PortfolioSnapshot | None:
        """Return latest portfolio snapshot for the user, if available."""

    def get_active_count(self) -> int:
        """Return total active sessions."""

    def get_session_mode(self, user_id: int) -> str | None:
        """Return backend session mode label for diagnostics."""

    def get_execution_diagnostics(self, user_id: int) -> ExecutionDiagnostics | None:
        """Return execution diagnostics for the session, if available."""

    def clear_execution_diagnostics(self, user_id: int) -> bool:
        """Reset execution diagnostics counters for the running session."""

    def restore_order_result(self, user_id: int, result: ExecutionResult) -> bool:
        """Restore a durable execution result for idempotent replay."""

    def set_monitoring_snapshot(
        self,
        user_id: int,
        snapshot: MonitoringSnapshot,
    ) -> KillSwitchEvaluation:
        """Update monitoring snapshot and return current kill-switch evaluation."""

    def get_kill_switch_evaluation(self, user_id: int) -> KillSwitchEvaluation | None:
        """Return latest kill-switch evaluation for a session."""

    def ingest_market_prices(self, user_id: int, prices: dict[str, float]) -> bool:
        """Merge externally-fetched market prices into runtime snapshot state."""

    async def route_signals(
        self,
        user_id: int,
        *,
        signals: Iterable[StrategySignal],
        prices: dict[str, float],
        monitoring_snapshot: MonitoringSnapshot | None = None,
        correlation_id: str = "",
    ) -> tuple[ExecutionResult, ...]:
        """Route strategy signals through execution planner and adapter."""


@dataclass
class _SessionState:
    """Internal runtime state for a started execution session."""

    request: SessionRequest
    adapter: object
    mode: str
    snapshot: PortfolioSnapshot
    hard_risk_policy: HardRiskLimits = field(default_factory=HardRiskLimits)
    dynamic_risk_policy: OperatingRiskLimits = field(default_factory=_default_dynamic_risk_policy)
    effective_risk_policy: OperatingRiskLimits = field(default_factory=_default_operating_risk_policy)
    equity_baseline_usd: float = 10_000.0
    last_prices: dict[str, float] = field(default_factory=dict)
    latest_signals: dict[str, StrategySignal] = field(default_factory=dict)
    previous_signals: dict[str, StrategySignal] = field(default_factory=dict)
    diagnostics: ExecutionDiagnostics = field(default_factory=ExecutionDiagnostics)
    paper_entry_price: dict[str, float] = field(default_factory=dict)
    accounted_order_keys: set[str] = field(default_factory=set)
    order_results_by_key: dict[str, ExecutionResult] = field(default_factory=dict)
    position_opened_at: dict[str, datetime] = field(default_factory=dict)
    position_entry_atr_pct: dict[str, float] = field(default_factory=dict)
    position_peak_pnl_pct: dict[str, float] = field(default_factory=dict)
    last_rebalance_at: dict[str, datetime] = field(default_factory=dict)
    external_execution_anomaly_rate: float = 0.0
    external_hard_risk_breach: bool = False
    lifecycle_rules: LifecycleRules = field(default_factory=LifecycleRules)
    monitoring_snapshot: MonitoringSnapshot = field(default_factory=MonitoringSnapshot)
    kill_switch: KillSwitchEvaluation = field(
        default_factory=lambda: KillSwitchEvaluation(pause_trading=False)
    )
    # --- Phase-3 redesign fields ---
    breach_history: list[datetime] = field(default_factory=list)
    soft_breach_active: bool = False
    equity_history: list[float] = field(default_factory=list)
    _equity_history_max: int = 120
    # --- Phase-4 limit order tracking ---
    open_limit_orders: dict[str, dict] = field(default_factory=dict)
    "Track resting limit orders: {order_id: {symbol, side, qty, filled, placed_at, is_partial}}"
    last_partial_fill_check: datetime | None = None
    # Rolling close prices for optimizer: symbol -> list of (timestamp, price)
    price_histories: dict[str, list[float]] = field(default_factory=dict)
    hard_risk_pause_persisted: bool = False


class InMemoryExecutionService:
    """Reference in-memory implementation for integration and migration testing."""

    def __init__(self) -> None:
        self._sessions: dict[int, SessionRequest] = {}
        self._snapshots: dict[int, PortfolioSnapshot] = {}
        self._monitoring: dict[int, MonitoringSnapshot] = {}
        self._kill_switch: dict[int, KillSwitchEvaluation] = {}
        self._lifecycle_states: dict[int, LifecycleStateRecord] = {}

    @staticmethod
    def _initial_snapshot() -> PortfolioSnapshot:
        return PortfolioSnapshot(
            timestamp=datetime.now(timezone.utc),
            equity_usd=10_000.0,
            risk=RiskSnapshot(
                gross_exposure_frac=0.0,
                net_exposure_frac=0.0,
                max_drawdown_frac=0.0,
                risk_budget_used_frac=0.0,
            ),
        )

    async def start_session(self, request: SessionRequest) -> bool:
        if request.user_id in self._sessions:
            return False

        self._sessions[request.user_id] = request
        self._snapshots[request.user_id] = self._initial_snapshot()
        self._monitoring[request.user_id] = MonitoringSnapshot()
        self._kill_switch[request.user_id] = KillSwitchEvaluation(pause_trading=False)
        self._lifecycle_states[request.user_id] = LifecycleStateRecord(
            state="ACTIVE",
            owner="alpha_session",
            retry_count=0,
            reason="session_started",
            policy_version=HardRiskLimits().policy_version,
        )
        return True

    async def stop_session(self, user_id: int) -> bool:
        existed = user_id in self._sessions
        if user_id in self._sessions or user_id in self._lifecycle_states:
            previous = self._lifecycle_states.get(user_id)
            if previous is None:
                previous = LifecycleStateRecord(
                    state="ACTIVE",
                    owner="alpha_session",
                    retry_count=0,
                    reason="session_started",
                    policy_version=HardRiskLimits().policy_version,
                )
            try:
                self._lifecycle_states[user_id] = _build_lifecycle_state_record(
                    previous,
                    state="PAUSED",
                    owner="control_plane",
                    reason="session_stopped",
                    policy_version=previous.policy_version,
                )
            except ValueError:
                self._lifecycle_states[user_id] = previous
        self._sessions.pop(user_id, None)
        self._snapshots.pop(user_id, None)
        self._monitoring.pop(user_id, None)
        self._kill_switch.pop(user_id, None)
        return existed

    def reset_session_state(self, user_id: int) -> bool:
        request = self._sessions.get(user_id)
        if request is None:
            return False
        if request.live:
            return False

        self._snapshots[user_id] = self._initial_snapshot()
        self._monitoring[user_id] = MonitoringSnapshot()
        self._kill_switch[user_id] = KillSwitchEvaluation(pause_trading=False)
        return True

    def is_running(self, user_id: int) -> bool:
        return user_id in self._sessions

    def get_active_count(self) -> int:
        return len(self._sessions)

    def get_session_mode(self, user_id: int) -> str | None:
        request = self._sessions.get(user_id)
        if request is None:
            return None
        return "live" if request.live else "paper"

    def get_execution_diagnostics(self, user_id: int) -> ExecutionDiagnostics | None:
        if user_id not in self._sessions:
            return None
        return ExecutionDiagnostics()

    def clear_execution_diagnostics(self, user_id: int) -> bool:
        return user_id in self._sessions

    def restore_order_result(self, user_id: int, result: ExecutionResult) -> bool:
        return user_id in self._sessions

    def set_monitoring_snapshot(
        self,
        user_id: int,
        snapshot: MonitoringSnapshot,
    ) -> KillSwitchEvaluation:
        if user_id not in self._sessions:
            raise KeyError(f"No active session for user {user_id}")

        self._monitoring[user_id] = snapshot
        evaluation = evaluate_kill_switch(snapshot)
        self._kill_switch[user_id] = evaluation
        return evaluation

    def get_kill_switch_evaluation(self, user_id: int) -> KillSwitchEvaluation | None:
        return self._kill_switch.get(user_id)

    def get_lifecycle_state(self, user_id: int) -> LifecycleStateRecord | None:
        return self._lifecycle_states.get(user_id)

    def set_lifecycle_state(
        self,
        user_id: int,
        *,
        state: str,
        owner: str,
        reason: str,
        policy_version: str = "",
        retry_count: int | None = None,
    ) -> LifecycleStateRecord:
        previous = self._lifecycle_states.get(user_id)
        if previous is None and user_id not in self._sessions:
            raise KeyError(f"No active session for user {user_id}")

        record = _build_lifecycle_state_record(
            previous,
            state=state,
            owner=owner,
            reason=reason,
            policy_version=policy_version,
            retry_count=retry_count,
        )
        self._lifecycle_states[user_id] = record
        return record

    def restore_lifecycle_state(
        self,
        user_id: int,
        record: LifecycleStateRecord,
    ) -> LifecycleStateRecord:
        previous = self._lifecycle_states.get(user_id)
        validate_lifecycle_transition(previous.state if previous is not None else None, record.state)
        self._lifecycle_states[user_id] = record
        return record

    def get_portfolio_snapshot(self, user_id: int) -> PortfolioSnapshot | None:
        return self._snapshots.get(user_id)

    def ingest_market_prices(self, user_id: int, prices: dict[str, float]) -> bool:
        if user_id not in self._sessions:
            return False

        normalized: dict[str, float] = {}
        for symbol, raw_price in prices.items():
            clean_symbol = str(symbol).strip().upper()
            if not clean_symbol:
                continue
            try:
                clean_price = float(raw_price)
            except (TypeError, ValueError):
                continue
            if clean_price > 0.0:
                normalized[clean_symbol] = clean_price

        if not normalized:
            return False

        snapshot = self._snapshots.get(user_id)
        if snapshot is not None:
            self._snapshots[user_id] = replace(
                snapshot,
                timestamp=datetime.now(timezone.utc),
            )
        return True

    async def route_signals(
        self,
        user_id: int,
        *,
        signals: Iterable[StrategySignal],
        prices: dict[str, float],
        monitoring_snapshot: MonitoringSnapshot | None = None,
        correlation_id: str = "",
    ) -> tuple[ExecutionResult, ...]:
        # Legacy placeholder backend; signal routing is implemented in RoutedExecutionService.
        _ = (user_id, tuple(signals), prices, monitoring_snapshot, correlation_id)
        return ()


class RoutedExecutionService:
    """Execution service that routes sessions to paper or live adapters."""

    def __init__(
        self,
        *,
        paper_adapter_factory: Callable[[], object] | None = None,
        live_adapter_factory: Callable[[SessionRequest], object] | None = None,
        initial_equity_usd: float = 10_000.0,
        risk_policy: PortfolioRiskPolicy | None = None,
        planner_config: PlannerConfig | None = None,
        kill_switch_config: KillSwitchConfig | None = None,
        allow_live_execution: bool = True,
        canary_live_risk_cap_frac: float | None = None,
        enforce_live_go_no_go: bool | None = None,
        live_go_no_go: bool | None = None,
        rollback_failure_threshold: int | None = None,
        min_rebalance_notional_usd: float | None = None,
        rebalance_cooldown_seconds: int | None = None,
        max_orders_per_cycle: int | None = None,
        hard_risk_pause_callback: Callable[[HardRiskPauseEvent], None] | None = None,
        route_audit_callback: Callable[[RouteAuditEvent], None] | None = None,
        stage_telemetry_callback: Callable[[ExecutionStageTelemetry], None] | None = None,
    ) -> None:
        self._sessions: dict[int, _SessionState] = {}
        self._route_locks: dict[int, asyncio.Lock] = {}
        self._lifecycle_states: dict[int, LifecycleStateRecord] = {}
        self._paper_adapter_factory = paper_adapter_factory or self._default_paper_adapter_factory
        self._live_adapter_factory = live_adapter_factory or self._default_live_adapter_factory
        self._initial_equity_usd = float(initial_equity_usd)
        base_policy = risk_policy or PortfolioRiskPolicy()
        if isinstance(base_policy, HardRiskLimits):
            self._risk_policy = base_policy
        else:
            self._risk_policy = HardRiskLimits(
                max_symbol_exposure_frac=float(base_policy.max_symbol_exposure_frac),
                max_gross_exposure_frac=float(base_policy.max_gross_exposure_frac),
                max_net_exposure_frac=float(base_policy.max_net_exposure_frac),
                correlation_bucket_caps={
                    bucket: float(limit)
                    for bucket, limit in base_policy.correlation_bucket_caps.items()
                },
            )
        self._planner_config = planner_config or PlannerConfig()
        self._optimizer = RiskParityOptimizer()
        self._kill_switch_config = kill_switch_config or KillSwitchConfig()
        self._allow_live_execution = bool(allow_live_execution)
        self._hard_risk_pause_callback = hard_risk_pause_callback
        self._route_audit_callback = route_audit_callback
        self._stage_telemetry_callback = stage_telemetry_callback
        if canary_live_risk_cap_frac is None:
            canary_live_risk_cap_frac = get_runtime_profile().deployment.canary_live_risk_cap_frac
        self._canary_live_risk_cap_frac = float(canary_live_risk_cap_frac)
        if not 0.0 < self._canary_live_risk_cap_frac <= 1.0:
            raise ValueError("canary_live_risk_cap_frac must be within (0, 1]")

        if rollback_failure_threshold is None:
            try:
                parsed_threshold = int(
                    (os.getenv("BOT_V2_ROLLBACK_FAILURE_THRESHOLD", "2").strip() or "2")
                )
            except ValueError:
                parsed_threshold = 2
        else:
            parsed_threshold = int(rollback_failure_threshold)
        self._rollback_failure_threshold = max(parsed_threshold, 1)

        if enforce_live_go_no_go is None:
            enforce_live_go_no_go = self._parse_bool_env("BOT_V2_ENFORCE_GO_NO_GO", True)
        if live_go_no_go is None:
            live_go_no_go = self._parse_bool_env("BOT_V2_LIVE_GO_NO_GO", True)

        self._enforce_live_go_no_go = bool(enforce_live_go_no_go)
        self._live_go_no_go_passed = bool(live_go_no_go)
        self._rollout_failure_streak = 0
        self._rollback_required = False
        self._rollback_reasons: tuple[str, ...] = ()
        self._refresh_runtime_rollout_controls()

        if min_rebalance_notional_usd is None:
            min_rebalance_weight_drift = self._parse_float_env(
                "BOT_V2_MIN_REBALANCE_WEIGHT_DRIFT",
                0.01,  # 1% default drift threshold
            )
        else:
            # Fallback legacy support if strictly passed
            min_rebalance_weight_drift = float(min_rebalance_notional_usd) / self._initial_equity_usd if self._initial_equity_usd > 0 else 0.01
            
        self._min_rebalance_weight_drift = max(float(min_rebalance_weight_drift), 0.0)

        if rebalance_cooldown_seconds is None:
            rebalance_cooldown_seconds = self._parse_int_env(
                "BOT_V2_REBALANCE_COOLDOWN_SECONDS",
                600,
            )
        self._rebalance_cooldown_seconds = max(int(rebalance_cooldown_seconds), 0)

        self._min_rebalance_confidence_delta = max(
            self._parse_float_env("BOT_V2_MIN_REBALANCE_CONFIDENCE_DELTA", 0.02),
            0.0,
        )
        self._min_rebalance_absolute_drift_usd = max(
            self._parse_float_env("BOT_V2_MIN_REBALANCE_ABSOLUTE_DRIFT_USD", 50.0),
            0.0,
        )
        self._rebalance_round_trip_cost_frac = max(
            self._parse_float_env("BOT_V2_REBALANCE_ROUND_TRIP_COST_FRAC", 0.0006),
            0.0,
        )

        if max_orders_per_cycle is None:
            max_orders_per_cycle = self._parse_int_env("BOT_V2_MAX_ORDERS_PER_CYCLE", 0)
        self._max_orders_per_cycle = max(int(max_orders_per_cycle), 0)

    async def start_session(self, request: SessionRequest) -> bool:
        if request.user_id in self._sessions:
            return False

        if request.live and self._allow_live_execution:
            self._enforce_live_start_rollout_gate()
            adapter = self._live_adapter_factory(request)
            mode = "live"
        else:
            adapter = self._paper_adapter_factory()
            mode = "paper_shadow" if request.live else "paper"

        session_policy = self._resolve_session_risk_policy(request)
        session_dynamic_policy = self._build_dynamic_operating_policy(
            session_policy,
            self._planner_config,
        )
        session_operating_policy = self._build_operating_policy(session_dynamic_policy)

        snapshot = self._build_snapshot(
            adapter,
            equity_usd=self._initial_equity_usd,
            risk_policy=session_policy,
            prices={},
        )
        diagnostics = self._build_initial_diagnostics(session_policy)
        self._sessions[request.user_id] = _SessionState(
            request=request,
            adapter=adapter,
            mode=mode,
            snapshot=snapshot,
            hard_risk_policy=session_policy,
            dynamic_risk_policy=session_dynamic_policy,
            effective_risk_policy=session_operating_policy,
            equity_baseline_usd=self._initial_equity_usd,
            last_prices={},
            diagnostics=diagnostics,
        )
        self._lifecycle_states[request.user_id] = _build_lifecycle_state_record(
            None,
            state="ACTIVE",
            owner="alpha_session",
            reason="session_started",
            policy_version=session_policy.policy_version,
            retry_count=0,
        )
        return True

    async def stop_session(self, user_id: int) -> bool:
        existed = user_id in self._sessions
        if existed or user_id in self._lifecycle_states:
            current = self._lifecycle_states.get(user_id)
            if current is None:
                current = LifecycleStateRecord(
                    state="ACTIVE",
                    owner="alpha_session",
                    retry_count=0,
                    reason="session_started",
                    policy_version=HardRiskLimits().policy_version,
                )
            try:
                self._lifecycle_states[user_id] = _build_lifecycle_state_record(
                    current,
                    state="PAUSED",
                    owner="control_plane",
                    reason="session_stopped",
                    policy_version=current.policy_version,
                )
            except ValueError:
                self._lifecycle_states[user_id] = current
        self._sessions.pop(user_id, None)
        self._route_locks.pop(user_id, None)
        return existed

    def reset_session_state(self, user_id: int) -> bool:
        """Reset paper-session adapter/snapshot/telemetry in-place when supported."""

        state = self._sessions.get(user_id)
        if state is None:
            return False
        if state.mode == "live":
            return False

        adapter = self._paper_adapter_factory()
        snapshot = self._build_snapshot(
            adapter,
            equity_usd=self._initial_equity_usd,
            risk_policy=state.hard_risk_policy,
            prices={},
        )
        diagnostics = self._build_initial_diagnostics(state.hard_risk_policy)
        session_dynamic_policy = self._build_dynamic_operating_policy(
            state.hard_risk_policy,
            self._planner_config,
        )
        self._sessions[user_id] = _SessionState(
            request=state.request,
            adapter=adapter,
            mode=state.mode,
            snapshot=snapshot,
            hard_risk_policy=state.hard_risk_policy,
            dynamic_risk_policy=session_dynamic_policy,
            effective_risk_policy=self._build_operating_policy(session_dynamic_policy),
            equity_baseline_usd=self._initial_equity_usd,
            last_prices={},
            diagnostics=diagnostics,
        )
        return True

    def get_paper_state(self, user_id: int) -> dict | None:
        """Return paper session state for WAL persistence.

        Returns dict with equity_baseline_usd, equity_usd (mark-to-market),
        open_positions, paper_entry_prices, and paper_entry_timestamps — or
        None if session doesn't exist.

        ``equity_usd`` is required by the signal-manager stranded-position
        flatten helper (audit_20260423 P0-4) to compute the optimizer's
        effective minimum-notional floor against current equity, not the
        baseline.
        """
        state = self._sessions.get(user_id)
        if state is None:
            return None
        positions = self._safe_get_positions(state.adapter)
        return {
            "equity_baseline_usd": float(state.equity_baseline_usd),
            "equity_usd": float(state.snapshot.equity_usd),
            "open_positions": {k: float(v) for k, v in positions.items() if abs(float(v)) > 1e-12},
            "paper_entry_prices": dict(state.paper_entry_price),
            "paper_entry_timestamps": {sym: ts.isoformat() for sym, ts in state.position_opened_at.items()},
        }

    async def restore_paper_state(
        self,
        user_id: int,
        *,
        equity_baseline_usd: float,
        open_positions: dict[str, float],
        paper_entry_prices: dict[str, float],
        paper_entry_timestamps: dict[str, str] | None = None,
    ) -> None:
        """Restore paper session equity, positions, and entry prices from WAL checkpoint."""
        state = self._sessions.get(user_id)
        if state is None:
            return
        if state.mode == "live":
            return

        state.equity_baseline_usd = max(1.0, float(equity_baseline_usd))
        state.accounted_order_keys.clear()
        state.order_results_by_key.clear()

        if open_positions:
            await self.sync_positions(
                user_id,
                target_positions=open_positions,
                prices={},
            )
            state.paper_entry_price.update(
                {k: float(v) for k, v in paper_entry_prices.items()}
            )
            # Restore entry timestamps with fallback to "now" for missing timestamps
            now_utc = datetime.now(timezone.utc)
            if paper_entry_timestamps:
                for sym, ts_str in paper_entry_timestamps.items():
                    try:
                        state.position_opened_at[sym] = datetime.fromisoformat(ts_str)
                    except Exception:
                        state.position_opened_at[sym] = now_utc

        logger.info(
            "Restored paper state for user %d: equity=$%.2f, %d positions",
            user_id,
            state.equity_baseline_usd,
            len(open_positions),
        )

    def is_running(self, user_id: int) -> bool:
        return user_id in self._sessions

    def get_active_count(self) -> int:
        """Return total active routed sessions."""

        return len(self._sessions)

    def get_portfolio_snapshot(self, user_id: int) -> PortfolioSnapshot | None:
        state = self._sessions.get(user_id)
        if state is None:
            return None

        try:
            current_positions = self._safe_get_positions(state.adapter)
            live_metrics: dict[str, dict[str, float]] | None = None
            if current_positions:
                live_metrics = self._safe_get_position_metrics(state.adapter)
                self._refresh_last_prices_from_adapter(
                    state,
                    open_positions=current_positions,
                    live_metrics=live_metrics,
                )

            self._refresh_snapshot_state(
                state,
                risk_policy=state.effective_risk_policy,
                prices=state.last_prices,
                open_positions=current_positions,
                live_metrics=live_metrics,
            )
        except Exception as exc:
            logger.warning("Failed refreshing v2 snapshot for user %s: %s", user_id, exc)

        return state.snapshot

    def ingest_market_prices(self, user_id: int, prices: dict[str, float]) -> bool:
        """Merge externally-fetched prices and refresh the in-memory snapshot."""

        state = self._sessions.get(user_id)
        if state is None:
            return False

        normalized: dict[str, float] = {}
        for symbol, raw_price in prices.items():
            clean_symbol = str(symbol).strip().upper()
            if not clean_symbol:
                continue
            try:
                clean_price = float(raw_price)
            except (TypeError, ValueError):
                continue
            if clean_price > 0.0:
                normalized[clean_symbol] = clean_price

        if not normalized:
            return False

        state.last_prices = {
            **state.last_prices,
            **normalized,
        }

        try:
            current_positions = self._safe_get_positions(state.adapter)
            live_metrics: dict[str, dict[str, float]] | None = None
            if current_positions and state.mode == "live":
                live_metrics = self._safe_get_position_metrics(state.adapter)
                self._refresh_last_prices_from_adapter(
                    state,
                    open_positions=current_positions,
                    live_metrics=live_metrics,
                )

            self._refresh_snapshot_state(
                state,
                risk_policy=state.effective_risk_policy,
                prices=state.last_prices,
                open_positions=current_positions,
                live_metrics=live_metrics,
            )
        except Exception as exc:
            logger.warning(
                "Failed ingesting external market prices for user %s: %s",
                user_id,
                exc,
            )

        return True

    async def route_signals(
        self,
        user_id: int,
        *,
        signals: Iterable[StrategySignal],
        prices: dict[str, float],
        monitoring_snapshot: MonitoringSnapshot | None = None,
        bucket_map: dict[str, str] | None = None,
        min_qty: float = 0.0,
        planner_config: PlannerConfig | None = None,
        risk_policy: PortfolioRiskPolicy | None = None,
        equity_usd: float | None = None,
        correlation_id: str = "",
    ) -> tuple[ExecutionResult, ...]:
        lock = self._route_locks.setdefault(int(user_id), asyncio.Lock())
        async with lock:
            return await self._route_signals_locked(
                user_id,
                signals=signals,
                prices=prices,
                monitoring_snapshot=monitoring_snapshot,
                bucket_map=bucket_map,
                min_qty=min_qty,
                planner_config=planner_config,
                risk_policy=risk_policy,
                equity_usd=equity_usd,
                correlation_id=correlation_id,
            )

    async def _route_signals_locked(
        self,
        user_id: int,
        *,
        signals: Iterable[StrategySignal],
        prices: dict[str, float],
        monitoring_snapshot: MonitoringSnapshot | None = None,
        bucket_map: dict[str, str] | None = None,
        min_qty: float = 0.0,
        planner_config: PlannerConfig | None = None,
        risk_policy: PortfolioRiskPolicy | None = None,
        equity_usd: float | None = None,
        correlation_id: str = "",
    ) -> tuple[ExecutionResult, ...]:
        state = self._sessions.get(user_id)
        if state is None:
            raise KeyError(f"No active session for user {user_id}")
        if min_qty < 0.0:
            raise ValueError("min_qty must be >= 0")

        route_started = perf_counter()
        signal_list = tuple(signals)

        _routed_buy = 0
        _routed_sell = 0
        _routed_actionable = 0
        for _sig in signal_list:
            _sig_type = str(getattr(_sig, "signal", "")).strip().upper()
            if _sig_type == "BUY":
                _routed_buy += 1
            elif _sig_type == "SELL":
                _routed_sell += 1
            if _sig.actionable:
                _routed_actionable += 1
        state.diagnostics = replace(
            state.diagnostics,
            routed_signals_total=state.diagnostics.routed_signals_total + len(signal_list),
            routed_buy_signals=state.diagnostics.routed_buy_signals + _routed_buy,
            routed_sell_signals=state.diagnostics.routed_sell_signals + _routed_sell,
            routed_actionable_signals=state.diagnostics.routed_actionable_signals + _routed_actionable,
        )

        incoming_prices: dict[str, float] = {}
        for symbol, raw_price in prices.items():
            clean_symbol = str(symbol).strip().upper()
            if not clean_symbol:
                continue
            try:
                clean_price = float(raw_price)
            except (TypeError, ValueError):
                continue
            if clean_price > 0.0:
                incoming_prices[clean_symbol] = clean_price
        if incoming_prices:
            state.last_prices = {
                **state.last_prices,
                **incoming_prices,
            }
            # Accumulate rolling price history for optimizer (keep last 200 bars)
            for sym, px in incoming_prices.items():
                hist = state.price_histories.setdefault(sym, [])
                hist.append(px)
                if len(hist) > 200:
                    state.price_histories[sym] = hist[-200:]

        if state.mode == "live":
            self._refresh_runtime_rollout_controls()
            gate_reasons = self._current_live_rollout_gate_reasons()
            if gate_reasons:
                blocked_actionable = sum(1 for signal in signal_list if signal.actionable)
                state.diagnostics = replace(
                    state.diagnostics,
                    paused_cycles=state.diagnostics.paused_cycles + 1,
                    blocked_actionable_signals=(
                        state.diagnostics.blocked_actionable_signals + blocked_actionable
                    ),
                )
                logger.error(
                    "Live routing blocked by rollout gates for user %s: %s",
                    user_id,
                    ",".join(gate_reasons),
                )
                state.kill_switch = KillSwitchEvaluation(pause_trading=True, reasons=gate_reasons)
                self._sync_rollout_diagnostics(state, gate_reasons=gate_reasons)
                return ()

        if monitoring_snapshot is not None:
            state.external_execution_anomaly_rate = monitoring_snapshot.execution_anomaly_rate
            state.external_hard_risk_breach = bool(monitoring_snapshot.hard_risk_breach)
            state.monitoring_snapshot = replace(
                monitoring_snapshot,
                execution_anomaly_rate=max(
                    state.external_execution_anomaly_rate,
                    state.diagnostics.reject_rate,
                ),
            )
        else:
            merged_anomaly = max(
                state.external_execution_anomaly_rate,
                state.diagnostics.reject_rate,
            )
            if merged_anomaly != state.monitoring_snapshot.execution_anomaly_rate:
                state.monitoring_snapshot = replace(
                    state.monitoring_snapshot,
                    execution_anomaly_rate=merged_anomaly,
                )

        precheck_hard_policy = state.hard_risk_policy
        if risk_policy is not None:
            explicit_hard_policy = self._coerce_hard_risk_policy(risk_policy)
            precheck_hard_policy = (
                self._apply_canary_risk_cap(explicit_hard_policy)
                if state.request.live
                else explicit_hard_policy
            )
        self._apply_snapshot_risk_monitoring(
            state,
            risk_policy=precheck_hard_policy,
            defer_internal_hard_pause=True,
        )
        internal_risk_only = False
        if state.kill_switch.pause_trading:
            internal_risk_only = (
                len(state.kill_switch.reasons) == 1
                and "hard_risk_breach" in state.kill_switch.reasons
                and not getattr(state, "external_hard_risk_breach", False)
            )
            if not internal_risk_only:
                blocked_actionable = sum(1 for signal in signal_list if signal.actionable)
                state.diagnostics = replace(
                    state.diagnostics,
                    paused_cycles=state.diagnostics.paused_cycles + 1,
                    blocked_actionable_signals=(
                        state.diagnostics.blocked_actionable_signals + blocked_actionable
                    ),
                )
                logger.warning(
                    "Kill-switch paused v2 execution for user %s; reasons=%s",
                    user_id,
                    ",".join(state.kill_switch.reasons),
                )
                if state.mode == "live":
                    self._record_rollout_observation(
                        failed=True,
                        reasons=state.kill_switch.reasons,
                    )
                    self._sync_rollout_diagnostics(state, gate_reasons=state.kill_switch.reasons)
                return ()
            else:
                logger.info(
                    "Internal hard_risk_breach active for user %s: bypassing pause to allow risk-reducing trades.",
                    user_id,
                )
                # Filter signals to ONLY those that reduce existing exposure.
                # A BUY is risk-reducing only when the symbol is currently short.
                # A SELL is risk-reducing only when the symbol is currently long.
                breach_positions = self._safe_get_positions(state.adapter)
                filtered: list = []
                for sig in signal_list:
                    if not sig.actionable:
                        filtered.append(sig)
                        continue
                    sym = str(sig.symbol).strip().upper()
                    current_qty = float(breach_positions.get(sym, 0.0))
                    direction = str(getattr(sig, "direction", getattr(sig, "signal", "HOLD"))).upper()
                    if direction == "BUY" and current_qty < -1e-12:
                        filtered.append(sig)  # reduces short
                    elif direction == "SELL" and current_qty > 1e-12:
                        filtered.append(sig)  # reduces long
                    elif direction not in ("BUY", "SELL"):
                        filtered.append(sig)  # HOLD / non-directional
                    else:
                        logger.info(
                            "Hard breach filter: blocked %s %s (pos=%.6f) — would increase exposure.",
                            direction, sym, current_qty,
                        )
                signal_list = filtered

        if not signal_list:
            return ()
        planning_prices = state.last_prices

        for signal in signal_list:
            symbol = str(signal.symbol).strip().upper()
            if not symbol:
                continue
            normalized_signal = signal if signal.symbol == symbol else replace(signal, symbol=symbol)
            if normalized_signal.actionable:
                previous_signal = state.latest_signals.get(symbol)
                if previous_signal is not None:
                    state.previous_signals[symbol] = previous_signal
                state.latest_signals[symbol] = normalized_signal

        current_positions = self._safe_get_positions(state.adapter)
        forced_exits = self._evaluate_lifecycle_forced_exits(
            state,
            current_positions=current_positions,
            prices=planning_prices,
        )
        if forced_exits:
            for symbol in forced_exits:
                state.latest_signals.pop(symbol, None)

        planning_signals = tuple(state.latest_signals.values())
        cfg = planner_config or self._planner_config

        if risk_policy is None:
            dynamic_policy = self._compute_dynamic_risk_policy(state)
            if dynamic_policy is None:
                dynamic_policy = self._build_dynamic_operating_policy(
                    state.hard_risk_policy,
                    cfg,
                )
        else:
            base_policy = self._coerce_hard_risk_policy(risk_policy)
            if state.request.live:
                base_policy = self._apply_canary_risk_cap(base_policy)
            state.hard_risk_policy = base_policy
            dynamic_policy = self._build_dynamic_operating_policy(
                base_policy,
                cfg,
            )

        # Apply soft-breach 90% cap reduction when active
        if getattr(state, "soft_breach_active", False):
            dynamic_policy = self._apply_soft_breach_caps(dynamic_policy)
        elif internal_risk_only:
            # 50% cap reduction during a hard breach to aggressively wind down
            dynamic_policy = dynamic_policy.scaled(0.5, limit_source="hard_breach_winddown")

        state.dynamic_risk_policy = dynamic_policy
        policy = self._build_operating_policy(dynamic_policy)
        state.effective_risk_policy = policy

        state.diagnostics = replace(
            state.diagnostics,
            effective_symbol_cap_frac=float(policy.max_symbol_exposure_frac),
            effective_gross_cap_frac=float(policy.max_gross_exposure_frac),
            effective_net_cap_frac=float(policy.max_net_exposure_frac),
            hard_symbol_cap_frac=float(state.hard_risk_policy.max_symbol_exposure_frac),
            hard_gross_cap_frac=float(state.hard_risk_policy.max_gross_exposure_frac),
            hard_net_cap_frac=float(state.hard_risk_policy.max_net_exposure_frac),
            target_headroom_ratio=float(getattr(policy, "target_headroom_ratio", cfg.target_headroom_ratio)),
            reserve_capacity_frac=float(getattr(policy, "reserve_capacity_frac", cfg.reserve_capacity_frac)),
            adverse_mark_buffer_frac=float(
                getattr(policy, "adverse_mark_buffer_frac", cfg.adverse_mark_buffer_frac)
            ),
            risk_policy_version=str(getattr(policy, "policy_version", "")),
        )

        if equity_usd is not None:
            if equity_usd <= 0.0:
                raise ValueError("equity_usd must be positive")
            state.snapshot = replace(state.snapshot, equity_usd=float(equity_usd))
            if state.mode != "live":
                state.equity_baseline_usd = float(equity_usd)

        planning_equity = self._resolve_snapshot_equity(
            state,
            open_positions=current_positions,
            prices=planning_prices,
        )

        # Convert rolling price lists to pd.Series for optimizer
        price_histories_series: dict[str, pd.Series] = {}
        for sym, hist_list in state.price_histories.items():
            if len(hist_list) >= 2:
                price_histories_series[sym] = pd.Series(hist_list)

        planning_started = perf_counter()
        intent_plan = build_execution_intents(
            planning_signals,
            policy=policy,
            config=cfg,
            bucket_map=bucket_map,
            optimizer=self._optimizer,
            price_histories=price_histories_series,
            current_positions=current_positions,
        )
        order_plans = reconcile_target_exposures(
            intent_plan.policy_result.exposures,
            current_positions_qty=current_positions,
            prices=planning_prices,
            equity_usd=planning_equity,
            min_qty=min_qty,
        )
        order_plans = tuple(
            sorted(
                order_plans,
                key=lambda plan: (
                    0
                    if self._is_risk_reducing_order(
                        current_qty=float(current_positions.get(plan.symbol, 0.0)),
                        side=plan.side,
                        quantity=float(plan.quantity),
                    )
                    else 1
                ),
            )
        )
        self._emit_stage_telemetry(
            user_id=user_id,
            stage="planning",
            started_at=planning_started,
            correlation_id=correlation_id,
            detail=f"signals={len(signal_list)} orders={len(order_plans)}",
        )

        now_utc = datetime.now(timezone.utc)
        epoch_minute = int(now_utc.timestamp() // 60)
        results: list[ExecutionResult] = []
        activity_by_key: dict[str, str] = {}
        residual_supervision_triggered = False
        projected_positions = {
            symbol: float(qty)
            for symbol, qty in current_positions.items()
        }
        if internal_risk_only:
            allowed_plans: list[OrderPlan] = []
            for plan in order_plans:
                idempotency_key = build_idempotency_key(
                    user_id=user_id,
                    plan=plan,
                    epoch_minute=epoch_minute,
                )
                mark_price = float(planning_prices.get(plan.symbol, 0.0) or 0.0)
                current_qty = float(current_positions.get(plan.symbol, 0.0))
                allowed, action_class, reason, after_qty = self._hard_pause_reduce_only_decision(
                    current_qty=current_qty,
                    side=plan.side,
                    quantity=float(plan.quantity),
                )
                if allowed:
                    allowed_plans.append(plan)
                    self._log_route_audit(
                        user_id=user_id,
                        correlation_id=correlation_id,
                        pause_state="internal_hard_risk",
                        is_active=None,
                        live_mode=(state.mode == "live"),
                        symbol=plan.symbol,
                        side=plan.side,
                        qty=float(plan.quantity),
                        before_position=current_qty,
                        after_position=after_qty,
                        action_class=action_class,
                        reason=reason,
                    )
                    continue

                self._log_route_audit(
                    user_id=user_id,
                    correlation_id=correlation_id,
                    pause_state="internal_hard_risk",
                    is_active=None,
                    live_mode=(state.mode == "live"),
                    symbol=plan.symbol,
                    side=plan.side,
                    qty=float(plan.quantity),
                    before_position=current_qty,
                    after_position=after_qty,
                    action_class=action_class,
                    reason=reason,
                )
                results.append(
                    self._build_skipped_result(
                        idempotency_key=idempotency_key,
                        plan=plan,
                        mark_price=mark_price,
                        reason=f"skipped_by_filter:{reason}",
                    )
                )
            order_plans = tuple(allowed_plans)

        attempted_orders = 0
        for plan in order_plans:
            order_started = perf_counter()
            idempotency_key = build_idempotency_key(
                user_id=user_id,
                plan=plan,
                epoch_minute=epoch_minute,
            )
            mark_price = float(planning_prices.get(plan.symbol, 0.0) or 0.0)
            current_qty = float(projected_positions.get(plan.symbol, 0.0))
            audit_after = self._project_after_position(
                current_qty=current_qty,
                side=plan.side,
                quantity=float(plan.quantity),
            )
            projected_candidate_positions = dict(projected_positions)
            projected_candidate_positions[plan.symbol] = audit_after
            telemetry = self._build_route_risk_telemetry(
                symbol=plan.symbol,
                current_positions=projected_positions,
                projected_positions=projected_candidate_positions,
                prices=planning_prices,
                equity_usd=planning_equity,
                bucket_map=bucket_map,
                hard_policy=state.hard_risk_policy,
                dynamic_policy=state.dynamic_risk_policy,
                operating_policy=policy,
            )
            previous_result = state.order_results_by_key.get(idempotency_key)
            if previous_result is not None:
                results.append(
                    previous_result.replay_copy()
                )
                audit_action = self._audit_action_class(
                    current_qty=current_qty,
                    side=plan.side,
                    quantity=float(plan.quantity),
                )
                self._log_route_audit(
                    user_id=user_id,
                    correlation_id=correlation_id,
                    pause_state="internal_hard_risk" if internal_risk_only else "none",
                    is_active=None,
                    live_mode=(state.mode == "live"),
                    symbol=plan.symbol,
                    side=plan.side,
                    qty=float(plan.quantity),
                    before_position=current_qty,
                    after_position=audit_after,
                    action_class=audit_action,
                    reason=f"idempotent_replay:{audit_action}",
                    **telemetry,
                )
                continue
            activity = self._classify_order_activity(
                current_qty=current_qty,
                side=plan.side,
                quantity=float(plan.quantity),
            )
            audit_action = self._audit_action_class(
                current_qty=current_qty,
                side=plan.side,
                quantity=float(plan.quantity),
            )
            risk_reducing_plan = self._is_risk_reducing_order(
                current_qty=current_qty,
                side=plan.side,
                quantity=float(plan.quantity),
            )
            bypass_alpha_filters = risk_reducing_plan

            if not risk_reducing_plan:
                limit_reason = self._projected_limit_breach_reason(
                    current_positions=projected_positions,
                    prices=planning_prices,
                    equity_usd=planning_equity,
                    hard_policy=state.hard_risk_policy,
                    bucket_map=bucket_map,
                    adverse_mark_buffer_frac=float(
                        getattr(policy, "adverse_mark_buffer_frac", cfg.adverse_mark_buffer_frac)
                    ),
                    symbol=plan.symbol,
                    side=plan.side,
                    quantity=float(plan.quantity),
                )
                if limit_reason is not None:
                    results.append(
                        self._build_skipped_result(
                            idempotency_key=idempotency_key,
                            plan=plan,
                            mark_price=mark_price,
                            reason=f"skipped_by_risk:{limit_reason}",
                        )
                    )
                    self._log_route_audit(
                        user_id=user_id,
                        correlation_id=correlation_id,
                        pause_state="internal_hard_risk" if internal_risk_only else "none",
                        is_active=None,
                        live_mode=(state.mode == "live"),
                        symbol=plan.symbol,
                        side=plan.side,
                        qty=float(plan.quantity),
                        before_position=current_qty,
                        after_position=audit_after,
                        action_class=audit_action,
                        reason=f"skipped_by_risk:{limit_reason}",
                        **telemetry,
                    )
                    continue

            if activity == "rebalance" and not bypass_alpha_filters:
                delta_notional_usd = abs(float(plan.quantity) * mark_price)
                weight_drift = 0.0
                if (
                    self._min_rebalance_weight_drift > 0.0
                    and mark_price > 0.0
                    and planning_equity > 0.0
                ):
                    weight_drift = delta_notional_usd / planning_equity
                    if (
                        weight_drift < self._min_rebalance_weight_drift
                        and delta_notional_usd < self._min_rebalance_absolute_drift_usd
                    ):
                        results.append(
                            self._build_skipped_result(
                                idempotency_key=idempotency_key,
                                plan=plan,
                                mark_price=mark_price,
                                reason="skipped_by_deadband:weight_drift_and_absolute_usd",
                            )
                        )
                        self._log_route_audit(
                            user_id=user_id,
                            correlation_id=correlation_id,
                            pause_state="internal_hard_risk" if internal_risk_only else "none",
                            is_active=None,
                            live_mode=(state.mode == "live"),
                            symbol=plan.symbol,
                            side=plan.side,
                            qty=float(plan.quantity),
                            before_position=current_qty,
                            after_position=audit_after,
                            action_class=audit_action,
                            reason="skipped_by_deadband:weight_drift_and_absolute_usd",
                            **telemetry,
                        )
                        continue

                current_signal = state.latest_signals.get(plan.symbol)
                previous_signal = state.previous_signals.get(plan.symbol)
                confidence_delta = self._compute_confidence_delta(
                    current_signal=current_signal,
                    previous_signal=previous_signal,
                )
                if confidence_delta < self._min_rebalance_confidence_delta:
                        results.append(
                            self._build_skipped_result(
                                idempotency_key=idempotency_key,
                                plan=plan,
                                mark_price=mark_price,
                                reason="skipped_by_deadband:confidence_delta",
                            )
                        )
                        self._log_route_audit(
                            user_id=user_id,
                            correlation_id=correlation_id,
                            pause_state="internal_hard_risk" if internal_risk_only else "none",
                            is_active=None,
                            live_mode=(state.mode == "live"),
                            symbol=plan.symbol,
                            side=plan.side,
                            qty=float(plan.quantity),
                            before_position=current_qty,
                            after_position=audit_after,
                            action_class=audit_action,
                            reason="skipped_by_deadband:confidence_delta",
                            **telemetry,
                        )
                        continue

                if not self._rebalance_edge_improves_enough(
                    current_signal=current_signal,
                    previous_signal=previous_signal,
                    delta_notional_usd=delta_notional_usd,
                ):
                        results.append(
                            self._build_skipped_result(
                                idempotency_key=idempotency_key,
                                plan=plan,
                                mark_price=mark_price,
                                reason="skipped_by_deadband:edge_improvement",
                            )
                        )
                        self._log_route_audit(
                            user_id=user_id,
                            correlation_id=correlation_id,
                            pause_state="internal_hard_risk" if internal_risk_only else "none",
                            is_active=None,
                            live_mode=(state.mode == "live"),
                            symbol=plan.symbol,
                            side=plan.side,
                            qty=float(plan.quantity),
                            before_position=current_qty,
                            after_position=audit_after,
                            action_class=audit_action,
                            reason="skipped_by_deadband:edge_improvement",
                            **telemetry,
                        )
                        continue

                if self._rebalance_cooldown_seconds > 0:
                    previous = state.last_rebalance_at.get(plan.symbol)
                    if previous is not None:
                        elapsed_seconds = (now_utc - previous).total_seconds()
                        if elapsed_seconds < self._rebalance_cooldown_seconds:
                            results.append(
                                self._build_skipped_result(
                                    idempotency_key=idempotency_key,
                                plan=plan,
                                mark_price=mark_price,
                                reason="skipped_by_deadband:cooldown",
                            )
                        )
                        self._log_route_audit(
                            user_id=user_id,
                            correlation_id=correlation_id,
                            pause_state="internal_hard_risk" if internal_risk_only else "none",
                            is_active=None,
                            live_mode=(state.mode == "live"),
                            symbol=plan.symbol,
                            side=plan.side,
                            qty=float(plan.quantity),
                            before_position=current_qty,
                            after_position=audit_after,
                            action_class=audit_action,
                            reason="skipped_by_deadband:cooldown",
                            **telemetry,
                        )
                        continue

            if (
                self._max_orders_per_cycle > 0
                and attempted_orders >= self._max_orders_per_cycle
                and not bypass_alpha_filters
            ):
                results.append(
                    self._build_skipped_result(
                        idempotency_key=idempotency_key,
                        plan=plan,
                        mark_price=mark_price,
                        reason="skipped_by_deadband:max_orders_per_cycle",
                    )
                )
                self._log_route_audit(
                    user_id=user_id,
                    correlation_id=correlation_id,
                    pause_state="internal_hard_risk" if internal_risk_only else "none",
                    is_active=None,
                    live_mode=(state.mode == "live"),
                    symbol=plan.symbol,
                    side=plan.side,
                    qty=float(plan.quantity),
                    before_position=current_qty,
                    after_position=audit_after,
                    action_class=audit_action,
                    reason="skipped_by_deadband:max_orders_per_cycle",
                    **telemetry,
                )
                continue

            attempted_orders += 1
            try:
                # We want to route a Limit order near the mark price
                # For a BUY, limit should be slightly above mark to ensure execution without reaching too far
                # For a SELL, limit should be slightly below mark

                # To simplify spread capture for this institutional upgrade, we submit post-only
                # limit orders exactly at the observed mark price (assumed mid or side appropriate)
                # and let it rest.
                limit_price = mark_price

                # FIX-1: Run the synchronous adapter call in a thread pool so the asyncio
                # event loop is never blocked by Binance API latency (can spike 500ms-2s).
                result = await asyncio.to_thread(
                    functools.partial(
                        state.adapter.place_order,
                        plan,
                        idempotency_key=idempotency_key,
                        mark_price=mark_price,
                        limit_price=limit_price,
                        post_only=True,
                    )
                )
            except Exception as exc:
                logger.error(
                    "Adapter order placement failed for user %s symbol=%s side=%s: %s",
                    user_id,
                    plan.symbol,
                    plan.side,
                    exc,
                )
                result = ExecutionResult(
                    accepted=False,
                    order_id="",
                    idempotency_key=idempotency_key,
                    symbol=plan.symbol,
                    side=plan.side,
                    requested_qty=float(plan.quantity),
                    filled_qty=0.0,
                    avg_price=mark_price,
                    status="error",
                    created_at=datetime.now(timezone.utc).isoformat(),
                    reason=f"adapter_exception:{exc.__class__.__name__}",
                )
            self._emit_stage_telemetry(
                user_id=user_id,
                stage="order_execution",
                started_at=order_started,
                correlation_id=correlation_id,
                status="accepted" if result.accepted else "rejected",
                detail=f"symbol={plan.symbol} side={plan.side}",
            )
            result = replace(
                result,
                risk_policy_version=str(telemetry.get("risk_policy_version", "")),
            )
            if self._requires_residual_supervision(result):
                residual_supervision_triggered = True
                residual_reason = "residual_position_supervision_required"
                state.kill_switch = KillSwitchEvaluation(
                    pause_trading=True,
                    reasons=tuple(
                        dict.fromkeys((*state.kill_switch.reasons, residual_reason))
                    ),
                )
                logger.warning(
                    "Residual-position supervision required for user %s symbol=%s reason=%s",
                    user_id,
                    plan.symbol,
                    result.reason,
            )
            results.append(result)
            activity_by_key[idempotency_key] = activity
            if idempotency_key:
                state.order_results_by_key[idempotency_key] = result
            filled_qty = float(result.economic_filled_qty if result.accepted else 0.0)
            actual_after = self._project_after_position(
                current_qty=current_qty,
                side=plan.side,
                quantity=filled_qty,
            )
            self._log_route_audit(
                user_id=user_id,
                correlation_id=correlation_id,
                pause_state="internal_hard_risk" if internal_risk_only else "none",
                is_active=None,
                live_mode=(state.mode == "live"),
                symbol=plan.symbol,
                side=plan.side,
                qty=float(plan.quantity),
                before_position=current_qty,
                after_position=actual_after,
                action_class=audit_action,
                reason=(result.reason or ("order_accepted" if result.accepted else "order_rejected")),
                accepted=bool(result.accepted),
                status=str(result.status or ""),
                order_id=str(result.order_id or ""),
                idempotency_key=str(result.idempotency_key or idempotency_key),
                mark_price=mark_price,
                **telemetry,
            )
            if result.accepted and filled_qty > 0.0:
                projected_positions[result.symbol] = actual_after

        state.diagnostics = self._update_execution_diagnostics(
            state.diagnostics,
            results=results,
            prices=planning_prices,
            activity_by_key=activity_by_key,
        )

        for result in results:
            activity = activity_by_key.get(result.idempotency_key)
            if activity == "rebalance" and result.accepted:
                state.last_rebalance_at[result.symbol] = now_utc

        if state.mode != "live":
            self._update_paper_entry_price(
                state,
                results=results,
                prices=planning_prices,
                starting_positions=current_positions,
            )
        merged_anomaly = max(
            state.external_execution_anomaly_rate,
            state.diagnostics.reject_rate,
        )
        if merged_anomaly != state.monitoring_snapshot.execution_anomaly_rate:
            state.monitoring_snapshot = replace(
                state.monitoring_snapshot,
                execution_anomaly_rate=merged_anomaly,
            )

        self._refresh_snapshot_state(
            state,
            risk_policy=policy,
            prices=planning_prices,
        )
        self._apply_snapshot_risk_monitoring(
            state,
            risk_policy=state.hard_risk_policy,
        )
        if residual_supervision_triggered:
            residual_reason = "residual_position_supervision_required"
            state.kill_switch = KillSwitchEvaluation(
                pause_trading=True,
                reasons=tuple(
                    dict.fromkeys((*state.kill_switch.reasons, residual_reason))
                ),
            )

        # Persist position snapshot to disk after every fill
        any_fill = any(r.accepted and float(getattr(r, "economic_filled_qty", r.filled_qty) or 0.0) > 0 for r in results)
        if any_fill:
            self._persist_position_snapshot(user_id, state, planning_prices)

        if state.kill_switch.pause_trading and "execution_anomaly" in state.kill_switch.reasons:
            logger.warning(
                "Kill-switch activated after execution telemetry update for user %s; reject_rate=%.3f",
                user_id,
                state.diagnostics.reject_rate,
            )

        if state.mode == "live":
            self._record_rollout_observation(
                failed=state.kill_switch.pause_trading,
                reasons=state.kill_switch.reasons if state.kill_switch.pause_trading else (),
            )
            self._sync_rollout_diagnostics(
                state,
                gate_reasons=state.kill_switch.reasons if state.kill_switch.pause_trading else (),
            )
        self._emit_stage_telemetry(
            user_id=user_id,
            stage="route_signals",
            started_at=route_started,
            correlation_id=correlation_id,
            status="ok",
            detail=f"results={len(results)} accepted={sum(1 for result in results if result.accepted)}",
        )
        return tuple(results)

    @staticmethod
    def _persist_position_snapshot(
        user_id: int,
        state: _SessionState,
        prices: dict[str, float],
    ) -> None:
        """Persist current positions to disk for crash recovery.

        Writes to /state/positions_snapshot.json (or BOT_SNAPSHOT_PATH).
        On startup, this snapshot can be compared with exchange positions
        to detect discrepancies.
        """
        import json as _json
        from pathlib import Path as _Path

        snapshot_path = _Path(os.getenv("BOT_SNAPSHOT_PATH", "/state/positions_snapshot.json"))
        try:
            snapshot_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "user_id": user_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "mode": state.mode,
                "equity_usd": state.snapshot.equity_usd,
                "positions": dict(state.snapshot.open_positions),
                "entry_prices": dict(state.paper_entry_price),
                "last_prices": {k: v for k, v in prices.items() if v > 0},
            }
            tmp = snapshot_path.with_suffix(".tmp")
            tmp.write_text(_json.dumps(data, indent=2), encoding="utf-8")
            tmp.replace(snapshot_path)
        except Exception as exc:
            logger.debug("Failed to persist position snapshot: %s", exc)

    @staticmethod
    def _signal_edge(signal: StrategySignal | None) -> float:
        if signal is None or not signal.actionable:
            return 0.0
        confidence = float(signal.confidence)
        return max(0.0, (2.0 * confidence) - 1.0)

    @staticmethod
    def _compute_confidence_delta(
        *,
        current_signal: StrategySignal | None,
        previous_signal: StrategySignal | None,
    ) -> float:
        if current_signal is None or previous_signal is None:
            return 1.0
        return abs(float(current_signal.confidence) - float(previous_signal.confidence))

    def _rebalance_edge_improves_enough(
        self,
        *,
        current_signal: StrategySignal | None,
        previous_signal: StrategySignal | None,
        delta_notional_usd: float,
    ) -> bool:
        if current_signal is None or previous_signal is None:
            return True
        new_edge = self._signal_edge(current_signal)
        old_edge = self._signal_edge(previous_signal)
        improvement = max(0.0, new_edge - old_edge) * max(float(delta_notional_usd), 0.0)
        trading_cost = self._rebalance_round_trip_cost_frac * max(float(delta_notional_usd), 0.0)
        return improvement >= trading_cost

    async def sync_positions(
        self,
        user_id: int,
        *,
        target_positions: dict[str, float],
        prices: dict[str, float] | None = None,
    ) -> tuple[ExecutionResult, ...]:
        """Place direct delta orders to align adapter positions with target quantities."""

        state = self._sessions.get(user_id)
        if state is None:
            raise KeyError(f"No active session for user {user_id}")

        incoming_prices = {
            str(symbol).strip().upper(): float(price)
            for symbol, price in (prices or {}).items()
            if str(symbol).strip() and float(price) > 0.0
        }
        if incoming_prices:
            state.last_prices = {
                **state.last_prices,
                **incoming_prices,
            }
        planning_prices = state.last_prices

        normalized_target: dict[str, float] = {}
        for symbol, qty in target_positions.items():
            clean_symbol = str(symbol).strip().upper()
            if not clean_symbol:
                continue
            clean_qty = float(qty)
            if abs(clean_qty) <= 1e-12:
                continue
            normalized_target[clean_symbol] = clean_qty

        if normalized_target:
            state.latest_signals = {
                symbol: StrategySignal(
                    symbol=symbol,
                    timeframe="1h",
                    horizon_bars=4,
                    signal="BUY" if qty > 0.0 else "SELL",
                    confidence=1.0,
                    uncertainty=0.0,
                    reason="maintenance_resume_seed",
                )
                for symbol, qty in normalized_target.items()
            }
        else:
            state.latest_signals.clear()

        current_positions = self._safe_get_positions(state.adapter)
        now_utc = datetime.now(timezone.utc)
        epoch_minute = int(now_utc.timestamp() // 60)
        results: list[ExecutionResult] = []
        activity_by_key: dict[str, str] = {}

        for symbol in sorted(set(current_positions) | set(normalized_target)):
            current_qty = float(current_positions.get(symbol, 0.0))
            target_qty = float(normalized_target.get(symbol, 0.0))
            delta_qty = target_qty - current_qty
            if abs(delta_qty) <= 1e-12:
                continue

            side = "BUY" if delta_qty > 0.0 else "SELL"
            quantity = abs(delta_qty)
            reduce_only = abs(target_qty) <= 1e-12
            plan = OrderPlan(
                symbol=symbol,
                side=side,
                quantity=quantity,
                reduce_only=reduce_only,
            )
            idempotency_key = build_idempotency_key(
                user_id=user_id,
                plan=plan,
                epoch_minute=epoch_minute,
            )
            mark_price = float(planning_prices.get(symbol, 0.0) or 0.0)
            activity = self._classify_order_activity(
                current_qty=current_qty,
                side=side,
                quantity=quantity,
            )

            try:
                # FIX-1: Non-blocking adapter call for manual sync_positions path.
                result = await asyncio.to_thread(
                    functools.partial(
                        state.adapter.place_order,
                        plan,
                        idempotency_key=idempotency_key,
                        mark_price=mark_price,
                    )
                )
            except Exception as exc:
                logger.error(
                    "Adapter manual sync placement failed for user %s symbol=%s side=%s: %s",
                    user_id,
                    symbol,
                    side,
                    exc,
                )
                result = ExecutionResult(
                    accepted=False,
                    order_id="",
                    idempotency_key=idempotency_key,
                    symbol=symbol,
                    side=side,
                    requested_qty=quantity,
                    filled_qty=0.0,
                    avg_price=mark_price,
                    status="error",
                    created_at=datetime.now(timezone.utc).isoformat(),
                    reason=f"adapter_exception:{exc.__class__.__name__}",
                )

            results.append(result)
            activity_by_key[idempotency_key] = activity

        state.diagnostics = self._update_execution_diagnostics(
            state.diagnostics,
            results=results,
            prices=planning_prices,
            activity_by_key=activity_by_key,
        )

        if state.mode != "live":
            self._update_paper_entry_price(
                state,
                results=results,
                prices=planning_prices,
                starting_positions=current_positions,
            )

        merged_anomaly = max(
            state.external_execution_anomaly_rate,
            state.diagnostics.reject_rate,
        )
        if merged_anomaly != state.monitoring_snapshot.execution_anomaly_rate:
            state.monitoring_snapshot = replace(
                state.monitoring_snapshot,
                execution_anomaly_rate=merged_anomaly,
            )

        self._refresh_snapshot_state(
            state,
            risk_policy=state.effective_risk_policy,
            prices=planning_prices,
        )

        return tuple(results)

    def set_monitoring_snapshot(
        self,
        user_id: int,
        snapshot: MonitoringSnapshot,
    ) -> KillSwitchEvaluation:
        """Update session monitoring snapshot and return latest kill-switch evaluation."""

        state = self._sessions.get(user_id)
        if state is None:
            raise KeyError(f"No active session for user {user_id}")

        state.external_execution_anomaly_rate = snapshot.execution_anomaly_rate
        state.external_hard_risk_breach = bool(snapshot.hard_risk_breach)
        merged_anomaly = max(
            state.external_execution_anomaly_rate,
            state.diagnostics.reject_rate,
        )
        state.monitoring_snapshot = replace(
            snapshot,
            execution_anomaly_rate=merged_anomaly,
            hard_risk_breach=state.external_hard_risk_breach,
        )
        self._apply_snapshot_risk_monitoring(
            state,
            risk_policy=state.hard_risk_policy,
        )
        if state.kill_switch.pause_trading:
            logger.warning(
                "Kill-switch activated for user %s; reasons=%s",
                user_id,
                ",".join(state.kill_switch.reasons),
            )
        return state.kill_switch

    def get_kill_switch_evaluation(self, user_id: int) -> KillSwitchEvaluation | None:
        """Return latest kill-switch state for diagnostics."""

        state = self._sessions.get(user_id)
        if state is None:
            return None
        return state.kill_switch

    def get_execution_diagnostics(self, user_id: int) -> ExecutionDiagnostics | None:
        """Return accumulated execution diagnostics for a session."""

        state = self._sessions.get(user_id)
        if state is None:
            return None
        return state.diagnostics

    def get_live_session_ids(self) -> list[int]:
        """Return user_ids of all sessions with mode == 'live'."""
        return [uid for uid, s in self._sessions.items() if s.mode == "live"]

    def get_session_adapter(self, user_id: int) -> object | None:
        """Return the raw adapter for a session (used by reconciliation loop)."""
        state = self._sessions.get(user_id)
        return state.adapter if state is not None else None

    def clear_execution_diagnostics(self, user_id: int) -> bool:
        """Reset execution telemetry counters while keeping the session and positions intact."""

        state = self._sessions.get(user_id)
        if state is None:
            return False

        state.diagnostics = self._build_initial_diagnostics(state.hard_risk_policy)
        merged_anomaly = max(
            state.external_execution_anomaly_rate,
            state.diagnostics.reject_rate,
        )
        if state.monitoring_snapshot.execution_anomaly_rate != merged_anomaly:
            state.monitoring_snapshot = replace(
                state.monitoring_snapshot,
                execution_anomaly_rate=merged_anomaly,
            )
        self._apply_snapshot_risk_monitoring(
            state,
            risk_policy=state.hard_risk_policy,
        )
        return True

    def restore_order_result(self, user_id: int, result: ExecutionResult) -> bool:
        """Restore a durable result for exact-once replay after a restart."""

        state = self._sessions.get(user_id)
        if state is None:
            return False
        key = str(getattr(result, "idempotency_key", "") or "").strip()
        if not key:
            return False
        state.order_results_by_key[key] = result
        if result.accepted and float(getattr(result, "economic_filled_qty", result.filled_qty) or 0.0) > 0.0:
            state.accounted_order_keys.add(key)
        return True

    def get_session_mode(self, user_id: int) -> str | None:
        """Return running mode label for diagnostics (live/paper)."""

        state = self._sessions.get(user_id)
        if state is None:
            return None
        return state.mode

    def get_last_prices(self, user_id: int) -> dict[str, float]:
        """Return latest merged mark-price cache for the session."""

        state = self._sessions.get(user_id)
        if state is None:
            return {}
        return dict(state.last_prices)

    def get_lifecycle_state(self, user_id: int) -> LifecycleStateRecord | None:
        """Return the durable lifecycle record for a session, if present."""

        return self._lifecycle_states.get(user_id)

    def set_lifecycle_state(
        self,
        user_id: int,
        *,
        state: str,
        owner: str,
        reason: str,
        policy_version: str = "",
        retry_count: int | None = None,
    ) -> LifecycleStateRecord:
        """Apply a monotonic lifecycle transition for a session."""

        current = self._lifecycle_states.get(user_id)
        if current is None and user_id not in self._sessions:
            raise KeyError(f"No active session for user {user_id}")

        record = _build_lifecycle_state_record(
            current,
            state=state,
            owner=owner,
            reason=reason,
            policy_version=policy_version,
            retry_count=retry_count,
        )
        self._lifecycle_states[user_id] = record
        return record

    def restore_lifecycle_state(
        self,
        user_id: int,
        record: LifecycleStateRecord,
    ) -> LifecycleStateRecord:
        """Restore a lifecycle record from durable replay without recomputing timestamps."""

        current = self._lifecycle_states.get(user_id)
        validate_lifecycle_transition(current.state if current is not None else None, record.state)
        self._lifecycle_states[user_id] = record
        return record

    def set_lifecycle_rules(
        self,
        user_id: int,
        *,
        auto_close_horizon_bars: int | None = None,
        stop_loss_pct: float | None = None,
    ) -> LifecycleRules:
        """Set per-session lifecycle rules for horizon and stop-loss auto exits."""

        state = self._sessions.get(user_id)
        if state is None:
            raise KeyError(f"No active session for user {user_id}")

        current = state.lifecycle_rules
        horizon = (
            current.auto_close_horizon_bars
            if auto_close_horizon_bars is None
            else int(auto_close_horizon_bars)
        )
        stop = current.stop_loss_pct if stop_loss_pct is None else float(stop_loss_pct)

        rules = LifecycleRules(
            auto_close_horizon_bars=horizon,
            stop_loss_pct=stop,
        )
        state.lifecycle_rules = rules
        return rules

    def get_lifecycle_rules(self, user_id: int) -> LifecycleRules | None:
        """Return configured lifecycle rules for a session."""

        state = self._sessions.get(user_id)
        if state is None:
            return None
        return state.lifecycle_rules

    def _refresh_snapshot_state(
        self,
        state: _SessionState,
        *,
        risk_policy: PortfolioRiskPolicy,
        prices: dict[str, float],
        open_positions: dict[str, float] | None = None,
        live_metrics: dict[str, dict[str, float]] | None = None,
    ) -> None:
        current_positions = (
            {symbol: float(qty) for symbol, qty in open_positions.items()}
            if open_positions is not None
            else self._safe_get_positions(state.adapter)
        )
        resolved_equity = self._resolve_snapshot_equity(
            state,
            open_positions=current_positions,
            prices=prices,
        )
        state.snapshot = self._build_snapshot(
            state.adapter,
            equity_usd=resolved_equity,
            risk_policy=risk_policy,
            prices=prices,
            open_positions=current_positions,
        )
        # Track equity history for dynamic volatility-scaled caps
        state.equity_history.append(resolved_equity)
        if len(state.equity_history) > state._equity_history_max:
            state.equity_history = state.equity_history[-state._equity_history_max:]
        if state.snapshot.open_positions:
            state.snapshot = replace(
                state.snapshot,
                symbol_pnl_usd=self._resolve_symbol_pnl(
                    state,
                    open_positions=state.snapshot.open_positions,
                    prices=prices,
                    live_metrics=live_metrics,
                ),
            )

        self._apply_snapshot_risk_monitoring(
            state,
            risk_policy=risk_policy,
        )

    @classmethod
    def _resolve_snapshot_equity(
        cls,
        state: _SessionState,
        *,
        open_positions: dict[str, float],
        prices: dict[str, float],
    ) -> float:
        if state.mode == "live":
            return max(float(state.snapshot.equity_usd), 1.0)

        baseline = max(float(state.equity_baseline_usd), 1.0)
        if not open_positions:
            return baseline

        unrealized = cls._resolve_symbol_pnl(
            state,
            open_positions=open_positions,
            prices=prices,
        )
        mark_to_market_equity = baseline + float(sum(unrealized.values()))
        return max(mark_to_market_equity, 1.0)

    @staticmethod
    def _compute_hard_risk_breach(
        snapshot: PortfolioSnapshot,
        *,
        risk_policy: PortfolioRiskPolicy,
    ) -> bool:
        risk = snapshot.risk
        if risk is None:
            return False

        eps = 1e-9
        if risk.gross_exposure_frac > float(risk_policy.max_gross_exposure_frac) + eps:
            return True
        if abs(risk.net_exposure_frac) > float(risk_policy.max_net_exposure_frac) + eps:
            return True

        equity = float(snapshot.equity_usd)
        if equity <= 0.0:
            return False
        symbol_cap = float(risk_policy.max_symbol_exposure_frac)
        for notional in (snapshot.symbol_notional_usd or {}).values():
            if (float(notional) / equity) > symbol_cap + eps:
                return True
        return False

    def _apply_snapshot_risk_monitoring(
        self,
        state: _SessionState,
        *,
        risk_policy: PortfolioRiskPolicy,
        defer_internal_hard_pause: bool = False,
    ) -> None:
        computed_breach = self._compute_hard_risk_breach(
            state.snapshot,
            risk_policy=risk_policy,
        )
        combined_hard_breach = bool(state.external_hard_risk_breach or computed_breach)

        # --- Soft-landing logic ---
        # Applies only to internally-computed portfolio breaches.
        # External hard_risk_breach (from monitoring snapshot) bypasses
        # soft-landing and triggers an immediate hard pause.
        if combined_hard_breach:
            if state.external_hard_risk_breach:
                # External kill-switch: respect immediately, no soft-landing
                state.soft_breach_active = False
            else:
                now = datetime.now(timezone.utc)
                if not state.breach_history or (now - state.breach_history[-1]).total_seconds() > 900:
                    state.breach_history.append(now)
                cutoff = now - timedelta(hours=4)
                state.breach_history = [t for t in state.breach_history if t >= cutoff]

                if len(state.breach_history) < 3:
                    combined_hard_breach = False
                    state.soft_breach_active = True
                    logger.info(
                        "Soft risk breach (%d/3 in 4h window) — reducing caps to 90%%",
                        len(state.breach_history),
                    )
                else:
                    state.soft_breach_active = False
                    logger.warning(
                        "Hard risk breach confirmed (%d breaches in 4h) — pausing execution",
                        len(state.breach_history),
                    )
        else:
            state.soft_breach_active = False

        if combined_hard_breach != state.monitoring_snapshot.hard_risk_breach:
            state.monitoring_snapshot = replace(
                state.monitoring_snapshot,
                hard_risk_breach=combined_hard_breach,
            )

        state.kill_switch = evaluate_kill_switch(
            state.monitoring_snapshot,
            config=self._kill_switch_config,
        )
        if combined_hard_breach or state.external_hard_risk_breach:
            try:
                self.set_lifecycle_state(
                    state.request.user_id,
                    state="REDUCE_ONLY",
                    owner="liquidation_supervisor",
                    reason="hard_risk_breach",
                    policy_version=str(getattr(risk_policy, "policy_version", "")),
                )
            except ValueError:
                pass
        if (
            state.kill_switch.pause_trading
            and "hard_risk_breach" in state.kill_switch.reasons
            and not state.hard_risk_pause_persisted
        ):
            if defer_internal_hard_pause and not state.external_hard_risk_breach:
                return
            self._record_hard_risk_pause(state, risk_policy=risk_policy)

    def _record_hard_risk_pause(
        self,
        state: _SessionState,
        *,
        risk_policy: PortfolioRiskPolicy,
    ) -> None:
        callback = self._hard_risk_pause_callback
        if callback is None:
            state.hard_risk_pause_persisted = True
            return

        snapshot = state.snapshot
        risk = snapshot.risk
        details: dict[str, Any] = {
            "mode": state.mode,
            "live": bool(state.request.live),
            "equity_usd": float(snapshot.equity_usd),
            "symbol_count": int(snapshot.symbol_count),
            "open_positions": {
                str(symbol): float(qty)
                for symbol, qty in (snapshot.open_positions or {}).items()
            },
            "symbol_notional_usd": {
                str(symbol): float(notional)
                for symbol, notional in (snapshot.symbol_notional_usd or {}).items()
            },
            "policy": {
                "max_symbol_exposure_frac": float(risk_policy.max_symbol_exposure_frac),
                "max_gross_exposure_frac": float(risk_policy.max_gross_exposure_frac),
                "max_net_exposure_frac": float(risk_policy.max_net_exposure_frac),
            },
        }
        if risk is not None:
            details["risk"] = {
                "gross_exposure_frac": float(risk.gross_exposure_frac),
                "net_exposure_frac": float(risk.net_exposure_frac),
                "max_drawdown_frac": float(risk.max_drawdown_frac),
                "risk_budget_used_frac": float(risk.risk_budget_used_frac),
            }

        event = HardRiskPauseEvent(
            user_id=state.request.user_id,
            reason="hard_risk_breach",
            triggered_at=datetime.now(timezone.utc),
            breach_type=(
                "external_monitoring"
                if state.external_hard_risk_breach
                else "portfolio_risk_policy"
            ),
            details=details,
        )

        try:
            callback(event)
            state.hard_risk_pause_persisted = True
        except Exception as exc:
            logger.error(
                "Failed persisting hard-risk pause for user %s: %s",
                state.request.user_id,
                exc,
                exc_info=True,
            )

    @classmethod
    def _evaluate_lifecycle_forced_exits(
        cls,
        state: _SessionState,
        *,
        current_positions: dict[str, float],
        prices: dict[str, float],
    ) -> set[str]:
        rules = state.lifecycle_rules
        has_any_rule = (
            rules.auto_close_horizon_bars > 0
            or rules.stop_loss_pct > 0.0
            or rules.take_profit_atr_mult > 0.0
        )
        if not has_any_rule:
            return set()

        now_utc = datetime.now(timezone.utc)
        forced: set[str] = set()
        active_symbols = {
            symbol
            for symbol, qty in current_positions.items()
            if abs(float(qty)) > 1e-12
        }
        for symbol in tuple(state.position_opened_at):
            if symbol not in active_symbols:
                state.position_opened_at.pop(symbol, None)
                state.position_entry_atr_pct.pop(symbol, None)
                state.position_peak_pnl_pct.pop(symbol, None)

        live_metrics: dict[str, dict[str, float]] = {}
        if state.mode == "live" and (rules.stop_loss_pct > 0.0 or rules.take_profit_atr_mult > 0.0):
            try:
                live_metrics = cls._safe_get_position_metrics(state.adapter)
            except Exception:
                live_metrics = {}

        for symbol, qty in current_positions.items():
            signed_qty = float(qty)
            if abs(signed_qty) <= 1e-12:
                continue

            opened_at = state.position_opened_at.get(symbol)
            if opened_at is None:
                opened_at = now_utc
                state.position_opened_at[symbol] = opened_at

            # --- Time-based auto-close ---
            if rules.auto_close_horizon_bars > 0:
                max_age = timedelta(hours=float(rules.auto_close_horizon_bars))
                if now_utc - opened_at >= max_age:
                    forced.add(symbol)
                    continue

            # --- Resolve entry and mark price ---
            if state.mode == "live":
                entry_price = float((live_metrics.get(symbol) or {}).get("entry_price", 0.0) or 0.0)
            else:
                entry_price = float(state.paper_entry_price.get(symbol, 0.0) or 0.0)
            mark_price = float(prices.get(symbol, 0.0) or 0.0)
            if entry_price <= 0.0 or mark_price <= 0.0:
                continue

            # Compute unrealized PnL %
            if signed_qty > 0.0:
                pnl_pct = (mark_price - entry_price) / entry_price
            else:
                pnl_pct = (entry_price - mark_price) / entry_price

            # --- Stop-loss check ---
            if rules.stop_loss_pct > 0.0 and pnl_pct <= -rules.stop_loss_pct:
                forced.add(symbol)
                continue

            # --- ATR-scaled take-profit with trailing stop ---
            if rules.take_profit_atr_mult > 0.0:
                entry_atr_pct = state.position_entry_atr_pct.get(symbol, 0.0)
                if entry_atr_pct > 0.0:
                    tp_target_pct = rules.take_profit_atr_mult * entry_atr_pct

                    # Update peak PnL tracking
                    prev_peak = state.position_peak_pnl_pct.get(symbol, 0.0)
                    if pnl_pct > prev_peak:
                        state.position_peak_pnl_pct[symbol] = pnl_pct
                        prev_peak = pnl_pct

                    # (a) Fixed target: close when profit hits target
                    if pnl_pct >= tp_target_pct:
                        logger.info(
                            "Take-profit triggered for %s: pnl=%.4f%% >= target=%.4f%%",
                            symbol, pnl_pct * 100, tp_target_pct * 100,
                        )
                        forced.add(symbol)
                        continue

                    # (b) Trailing stop: once profit exceeds 1.0 * ATR,
                    #     trail at 0.5 * ATR behind peak
                    trail_activation_pct = 1.0 * entry_atr_pct
                    trail_distance_pct = 0.5 * entry_atr_pct
                    if prev_peak >= trail_activation_pct:
                        trail_stop_pct = prev_peak - trail_distance_pct
                        if pnl_pct <= trail_stop_pct:
                            logger.info(
                                "Trailing take-profit triggered for %s: pnl=%.4f%% <= trail=%.4f%% (peak=%.4f%%)",
                                symbol, pnl_pct * 100, trail_stop_pct * 100, prev_peak * 100,
                            )
                            forced.add(symbol)
                            continue

        return forced

    def _enforce_live_start_rollout_gate(self) -> None:
        self._refresh_runtime_rollout_controls()
        gate_reasons = self._current_live_rollout_gate_reasons()
        if gate_reasons:
            raise RuntimeError(
                "Live rollout gates blocked session start: "
                + ",".join(gate_reasons)
            )

    def _current_live_rollout_gate_reasons(self) -> tuple[str, ...]:
        reasons: list[str] = []
        if self._enforce_live_go_no_go and not self._live_go_no_go_passed:
            reasons.append("go_no_go_failed")
        if self._rollback_required:
            reasons.append("rollback_required")
            reasons.extend(self._rollback_reasons)
        return tuple(dict.fromkeys(reasons))

    def _refresh_runtime_rollout_controls(self) -> None:
        self._live_go_no_go_passed = self._parse_bool_env(
            "BOT_V2_LIVE_GO_NO_GO",
            self._live_go_no_go_passed,
        )
        if self._parse_bool_env("BOT_V2_FORCE_ROLLBACK", False):
            self._rollback_required = True
            if "forced_rollback_env" not in self._rollback_reasons:
                self._rollback_reasons = (
                    "forced_rollback_env",
                    *self._rollback_reasons,
                )

    def _record_rollout_observation(self, *, failed: bool, reasons: tuple[str, ...]) -> None:
        if failed:
            self._rollout_failure_streak += 1
            if self._rollout_failure_streak >= self._rollback_failure_threshold:
                if not self._rollback_required:
                    logger.error(
                        "Rollback gate triggered after %d consecutive live failures; reasons=%s",
                        self._rollout_failure_streak,
                        ",".join(reasons) or "unspecified",
                    )
                self._rollback_required = True
                if reasons:
                    self._rollback_reasons = tuple(
                        dict.fromkeys(("rollback_threshold_exceeded", *reasons))
                    )
                elif not self._rollback_reasons:
                    self._rollback_reasons = ("rollback_threshold_exceeded",)
            return

        if not self._rollback_required:
            self._rollout_failure_streak = 0

    def _sync_rollout_diagnostics(
        self,
        state: _SessionState,
        *,
        gate_reasons: tuple[str, ...],
    ) -> None:
        merged_reasons = tuple(dict.fromkeys((*gate_reasons, *self._current_live_rollout_gate_reasons())))
        state.diagnostics = replace(
            state.diagnostics,
            live_go_no_go_passed=self._live_go_no_go_passed,
            rollback_required=self._rollback_required,
            rollout_failure_streak=self._rollout_failure_streak,
            rollout_gate_reasons=merged_reasons,
        )

    @staticmethod
    def _parse_bool_env(name: str, default: bool) -> bool:
        raw = os.getenv(name)
        if raw is None:
            return bool(default)
        clean = raw.strip().lower()
        if not clean:
            return bool(default)
        return clean in {"1", "true", "yes", "on"}

    @staticmethod
    def _parse_int_env(name: str, default: int) -> int:
        raw = os.getenv(name)
        if raw is None:
            return int(default)
        clean = raw.strip()
        if not clean:
            return int(default)
        try:
            return int(clean)
        except ValueError:
            return int(default)

    @staticmethod
    def _parse_float_env(name: str, default: float) -> float:
        raw = os.getenv(name)
        if raw is None:
            return float(default)
        clean = raw.strip()
        if not clean:
            return float(default)
        try:
            return float(clean)
        except ValueError:
            return float(default)

    def _build_initial_diagnostics(self, policy: HardRiskLimits) -> ExecutionDiagnostics:
        return ExecutionDiagnostics(
            live_go_no_go_passed=self._live_go_no_go_passed,
            rollback_required=self._rollback_required,
            rollout_failure_streak=self._rollout_failure_streak,
            rollout_gate_reasons=self._current_live_rollout_gate_reasons(),
            effective_symbol_cap_frac=float(policy.max_symbol_exposure_frac),
            effective_gross_cap_frac=float(policy.max_gross_exposure_frac),
            effective_net_cap_frac=float(policy.max_net_exposure_frac),
            hard_symbol_cap_frac=float(policy.max_symbol_exposure_frac),
            hard_gross_cap_frac=float(policy.max_gross_exposure_frac),
            hard_net_cap_frac=float(policy.max_net_exposure_frac),
            risk_policy_version=getattr(policy, "policy_version", ""),
        )

    @staticmethod
    def _coerce_hard_risk_policy(policy: PortfolioRiskPolicy) -> HardRiskLimits:
        if isinstance(policy, HardRiskLimits):
            return policy
        return HardRiskLimits(
            max_symbol_exposure_frac=float(policy.max_symbol_exposure_frac),
            max_gross_exposure_frac=float(policy.max_gross_exposure_frac),
            max_net_exposure_frac=float(policy.max_net_exposure_frac),
            correlation_bucket_caps={
                bucket: float(limit)
                for bucket, limit in policy.correlation_bucket_caps.items()
            },
            policy_version=str(getattr(policy, "policy_version", "")) or HardRiskLimits().policy_version,
        )

    @staticmethod
    def _build_dynamic_operating_policy(
        hard_policy: HardRiskLimits,
        cfg: PlannerConfig,
    ) -> OperatingRiskLimits:
        return OperatingRiskLimits.from_hard_limits(
            hard_policy,
            target_headroom_ratio=cfg.target_headroom_ratio,
            fee_reserve_frac=cfg.fee_reserve_frac,
            slippage_reserve_frac=cfg.slippage_reserve_frac,
            rounding_reserve_frac=cfg.rounding_reserve_frac,
            min_quantity_reserve_frac=cfg.min_quantity_reserve_frac,
            adverse_mark_buffer_frac=cfg.adverse_mark_buffer_frac,
        )

    @staticmethod
    def _build_operating_policy(policy: OperatingRiskLimits) -> OperatingRiskLimits:
        return policy.scaled(
            policy.effective_headroom_ratio,
            limit_source="operating_headroom",
        )

    @staticmethod
    def _requires_residual_supervision(result: ExecutionResult) -> bool:
        reason = str(result.reason or "")
        status = str(result.status or "")
        return (
            "supervised_residual_position_required" in reason
            or status == "residual_supervision_required"
        )

    def _resolve_session_risk_policy(self, request: SessionRequest) -> HardRiskLimits:
        if not request.live:
            return self._risk_policy
        return self._apply_canary_risk_cap(self._risk_policy)

    def _apply_canary_risk_cap(self, policy: HardRiskLimits) -> HardRiskLimits:
        cap = self._canary_live_risk_cap_frac
        return HardRiskLimits(
            max_symbol_exposure_frac=min(policy.max_symbol_exposure_frac, cap),
            max_gross_exposure_frac=min(policy.max_gross_exposure_frac, cap),
            max_net_exposure_frac=min(policy.max_net_exposure_frac, cap),
            correlation_bucket_caps={
                bucket: min(float(limit), cap)
                for bucket, limit in policy.correlation_bucket_caps.items()
            },
            policy_version=policy.policy_version,
        )

    @staticmethod
    def _compute_dynamic_risk_policy(
        state: _SessionState,
    ) -> OperatingRiskLimits | None:
        """Compute volatility-scaled risk caps from equity history.

        Returns ``None`` when history is too short to estimate volatility,
        signalling the caller to fall back to the static policy.

        Formulae (Phase-3 redesign):
            sigma_60 = 60-bar realised volatility of portfolio equity
            gross_cap = min(1.20 / sigma_60, 0.85)
            net_cap   = 0.45 * gross_cap
            symbol_cap = 0.30 * gross_cap
        """
        history = state.equity_history
        if len(history) < 30:
            return None

        equity_arr = np.array(history[-60:], dtype=float)
        if len(equity_arr) < 2:
            return None
        returns = np.diff(equity_arr) / np.maximum(equity_arr[:-1], 1e-9)
        sigma_60 = float(np.std(returns, ddof=1)) if len(returns) > 1 else 0.0

        if sigma_60 <= 1e-9:
            return None

        return build_dynamic_operating_limits(
            hard_limits=state.hard_risk_policy,
            sigma_60=sigma_60,
            target_headroom_ratio=float(
                getattr(state.effective_risk_policy, "target_headroom_ratio", 1.0)
            ),
            fee_reserve_frac=float(
                getattr(state.effective_risk_policy, "fee_reserve_frac", 0.0)
            ),
            slippage_reserve_frac=float(
                getattr(state.effective_risk_policy, "slippage_reserve_frac", 0.0)
            ),
            rounding_reserve_frac=float(
                getattr(state.effective_risk_policy, "rounding_reserve_frac", 0.0)
            ),
            min_quantity_reserve_frac=float(
                getattr(state.effective_risk_policy, "min_quantity_reserve_frac", 0.0)
            ),
            adverse_mark_buffer_frac=float(
                getattr(state.effective_risk_policy, "adverse_mark_buffer_frac", 0.0)
            ),
        )

    @staticmethod
    def _apply_soft_breach_caps(policy: OperatingRiskLimits) -> OperatingRiskLimits:
        """Scale all risk caps to 90% during a soft-breach window."""
        return policy.scaled(0.90, limit_source="soft_breach")

    async def check_and_handle_partial_fills(self, user_id: int) -> list[dict]:
        """Check for partially filled limit orders and cancel remaining quantity.

        Phase 4: After 1 cycle (1 hour), check if any limit orders are partially filled.
        If partially filled: cancel the remaining quantity and log the event.

        Returns list of handled partial fill events for diagnostics.
        """
        state = self._sessions.get(user_id)
        if state is None:
            return []

        # Only applicable for live mode with an adapter that supports get_open_orders
        if state.mode != "live":
            return []

        now = datetime.now(timezone.utc)

        # Check if at least 1 hour has passed since last check
        if state.last_partial_fill_check is not None:
            elapsed_hours = (now - state.last_partial_fill_check).total_seconds() / 3600
            if elapsed_hours < 1.0:
                return []

        state.last_partial_fill_check = now

        handled_events: list[dict] = []
        get_open_orders = getattr(state.adapter, "get_open_orders", None)
        cancel_order = getattr(state.adapter, "cancel_order", None)

        if not callable(get_open_orders) or not callable(cancel_order):
            return []

        try:
            open_orders = get_open_orders()
        except Exception as e:
            logger.warning("Failed to fetch open orders for partial fill check: %s", e)
            return []

        for order in open_orders:
            order_id = str(order.get("orderId", ""))
            symbol = str(order.get("symbol", ""))
            status = str(order.get("status", "")).upper()

            # Check if order is partially filled
            if status == "PARTIALLY_FILLED":
                executed_qty = float(order.get("executedQty", 0.0))
                orig_qty = float(order.get("origQty", order.get("quantity", 0.0)))
                remaining_qty = orig_qty - executed_qty

                if remaining_qty > 0:
                    try:
                        cancel_result = cancel_order(symbol, order_id)
                        event = {
                            "order_id": order_id,
                            "symbol": symbol,
                            "side": str(order.get("side", "")),
                            "executed_qty": executed_qty,
                            "remaining_qty": remaining_qty,
                            "cancelled_at": now.isoformat(),
                            "cancel_result": str(cancel_result.get("status", "unknown")),
                        }
                        handled_events.append(event)
                        logger.info(
                            "Partial fill handled for %s order %s: executed=%.4f, remaining=%.4f cancelled",
                            symbol, order_id, executed_qty, remaining_qty
                        )
                    except Exception as e:
                        logger.error("Failed to cancel partially filled order %s: %s", order_id, e)

        return handled_events

    @staticmethod
    def _default_paper_adapter_factory() -> object:
        from quant_v2.execution.adapters import InMemoryPaperAdapter

        return InMemoryPaperAdapter()

    @staticmethod
    def _default_live_adapter_factory(request: SessionRequest) -> object:
        from quant.config import BinanceAPIConfig
        from quant.data.binance_client import BinanceClient
        from quant_v2.execution.binance_adapter import BinanceExecutionAdapter

        api_key = request.credentials.get("binance_api_key", "").strip()
        api_secret = request.credentials.get("binance_api_secret", "").strip()
        if not api_key or not api_secret:
            raise RuntimeError("Live session requires Binance API credentials")

        cfg = BinanceAPIConfig(api_key=api_key, api_secret=api_secret)
        client = BinanceClient(config=cfg)
        return BinanceExecutionAdapter(client)

    @staticmethod
    def _safe_get_positions(adapter: object) -> dict[str, float]:
        get_positions = getattr(adapter, "get_positions", None)
        if not callable(get_positions):
            raise TypeError("Adapter must provide get_positions()")
        return dict(get_positions())

    @staticmethod
    def _safe_get_position_metrics(adapter: object) -> dict[str, dict[str, float]]:
        getter = getattr(adapter, "get_position_metrics", None)
        if not callable(getter):
            return {}

        raw = getter()
        if not isinstance(raw, dict):
            return {}

        metrics: dict[str, dict[str, float]] = {}
        for symbol, payload in raw.items():
            if not isinstance(payload, dict):
                continue
            entry_price = float(payload.get("entry_price", 0.0) or 0.0)
            unrealized = float(payload.get("unrealized_pnl_usd", 0.0) or 0.0)
            mark_price = float(payload.get("mark_price", 0.0) or 0.0)
            metrics[str(symbol)] = {
                "entry_price": entry_price,
                "unrealized_pnl_usd": unrealized,
                "mark_price": mark_price,
            }
        return metrics

    @classmethod
    def _refresh_last_prices_from_adapter(
        cls,
        state: _SessionState,
        *,
        open_positions: dict[str, float],
        live_metrics: dict[str, dict[str, float]] | None = None,
    ) -> None:
        if not open_positions:
            return

        metrics = live_metrics if live_metrics is not None else cls._safe_get_position_metrics(state.adapter)
        if not metrics:
            return

        refreshed_prices: dict[str, float] = {}
        for symbol, qty in open_positions.items():
            signed_qty = float(qty)
            if abs(signed_qty) <= 1e-12:
                continue

            payload = metrics.get(symbol, {})
            mark_price = float(payload.get("mark_price", 0.0) or 0.0)
            if mark_price <= 0.0:
                entry_price = float(payload.get("entry_price", 0.0) or 0.0)
                unrealized = float(payload.get("unrealized_pnl_usd", 0.0) or 0.0)
                if entry_price > 0.0:
                    derived_mark = entry_price + (unrealized / signed_qty)
                    if derived_mark > 0.0:
                        mark_price = float(derived_mark)

            if mark_price > 0.0:
                refreshed_prices[symbol] = mark_price

        if refreshed_prices:
            state.last_prices = {
                **state.last_prices,
                **refreshed_prices,
            }

    @staticmethod
    def _classify_order_activity(*, current_qty: float, side: str, quantity: float) -> str:
        eps = 1e-12
        side_sign = 1.0 if side == "BUY" else -1.0
        next_qty = float(current_qty) + (side_sign * float(quantity))

        if abs(current_qty) <= eps and abs(next_qty) > eps:
            return "entry"
        if abs(next_qty) <= eps and abs(current_qty) > eps:
            return "exit"
        return "rebalance"

    @staticmethod
    def _project_after_position(*, current_qty: float, side: str, quantity: float) -> float:
        side_sign = 1.0 if side == "BUY" else -1.0
        return float(current_qty) + (side_sign * float(quantity))

    @staticmethod
    def _conservative_mark(*, price: float, adverse_mark_buffer_frac: float) -> float:
        if price <= 0.0:
            return 0.0
        return float(price) * (1.0 + max(float(adverse_mark_buffer_frac), 0.0))

    @classmethod
    def _exposure_metrics(
        cls,
        *,
        positions: dict[str, float],
        prices: dict[str, float],
        equity_usd: float,
        bucket_map: dict[str, str] | None = None,
        adverse_mark_buffer_frac: float = 0.0,
    ) -> dict[str, Any]:
        symbol_exposure_fracs: dict[str, float] = {}
        bucket_gross_exposure: dict[str, float] = {}
        gross = 0.0
        net = 0.0
        if equity_usd <= 0.0:
            return {
                "symbol_exposure_fracs": symbol_exposure_fracs,
                "bucket_gross_exposure": bucket_gross_exposure,
                "gross_exposure_frac": 0.0,
                "net_exposure_frac": 0.0,
            }

        for sym, qty in positions.items():
            base_price = float(prices.get(sym, 0.0) or 0.0)
            price = cls._conservative_mark(
                price=base_price,
                adverse_mark_buffer_frac=adverse_mark_buffer_frac,
            )
            if price <= 0.0 or abs(qty) <= 1e-12:
                continue
            exposure = (float(qty) * price) / equity_usd
            symbol_exposure_fracs[sym] = float(exposure)
            gross += abs(exposure)
            net += exposure
            if bucket_map:
                bucket = bucket_map.get(sym, "unmapped")
                bucket_gross_exposure[bucket] = bucket_gross_exposure.get(bucket, 0.0) + abs(exposure)

        return {
            "symbol_exposure_fracs": symbol_exposure_fracs,
            "bucket_gross_exposure": bucket_gross_exposure,
            "gross_exposure_frac": float(gross),
            "net_exposure_frac": float(net),
        }

    @classmethod
    def _projected_limit_breach_reason(
        cls,
        *,
        current_positions: dict[str, float],
        prices: dict[str, float],
        equity_usd: float,
        hard_policy: PortfolioRiskPolicy,
        bucket_map: dict[str, str] | None = None,
        adverse_mark_buffer_frac: float = 0.0,
        symbol: str,
        side: str,
        quantity: float,
    ) -> str | None:
        if equity_usd <= 0.0:
            return "non_positive_equity"

        projected_positions = {
            sym: float(qty)
            for sym, qty in current_positions.items()
        }
        current_qty = float(projected_positions.get(symbol, 0.0))
        projected_positions[symbol] = cls._project_after_position(
            current_qty=current_qty,
            side=side,
            quantity=quantity,
        )

        metrics = cls._exposure_metrics(
            positions=projected_positions,
            prices=prices,
            equity_usd=equity_usd,
            bucket_map=bucket_map,
            adverse_mark_buffer_frac=adverse_mark_buffer_frac,
        )
        for exposure in metrics["symbol_exposure_fracs"].values():
            if abs(float(exposure)) > float(hard_policy.max_symbol_exposure_frac) + 1e-9:
                return "projected_symbol_cap"
        for bucket, limit in hard_policy.correlation_bucket_caps.items():
            if float(metrics["bucket_gross_exposure"].get(bucket, 0.0)) > float(limit) + 1e-9:
                return "projected_bucket_cap"
        if float(metrics["gross_exposure_frac"]) > float(hard_policy.max_gross_exposure_frac) + 1e-9:
            return "projected_gross_cap"
        if abs(float(metrics["net_exposure_frac"])) > float(hard_policy.max_net_exposure_frac) + 1e-9:
            return "projected_net_cap"
        return None

    @classmethod
    def _portfolio_limit_breach_reason(
        cls,
        *,
        positions: dict[str, float],
        prices: dict[str, float],
        equity_usd: float,
        policy: PortfolioRiskPolicy,
        bucket_map: dict[str, str] | None = None,
        adverse_mark_buffer_frac: float = 0.0,
    ) -> str | None:
        if equity_usd <= 0.0:
            return "non_positive_equity"

        metrics = cls._exposure_metrics(
            positions=positions,
            prices=prices,
            equity_usd=equity_usd,
            bucket_map=bucket_map,
            adverse_mark_buffer_frac=adverse_mark_buffer_frac,
        )
        for exposure in metrics["symbol_exposure_fracs"].values():
            if abs(float(exposure)) > float(policy.max_symbol_exposure_frac) + 1e-9:
                return "current_symbol_cap"
        for bucket, limit in policy.correlation_bucket_caps.items():
            if float(metrics["bucket_gross_exposure"].get(bucket, 0.0)) > float(limit) + 1e-9:
                return "current_bucket_cap"
        if float(metrics["gross_exposure_frac"]) > float(policy.max_gross_exposure_frac) + 1e-9:
            return "current_gross_cap"
        if abs(float(metrics["net_exposure_frac"])) > float(policy.max_net_exposure_frac) + 1e-9:
            return "current_net_cap"
        return None

    @staticmethod
    def _headroom_value(limit: float, exposure: float) -> float:
        return max(float(limit) - abs(float(exposure)), 0.0)

    @classmethod
    def _build_route_risk_telemetry(
        cls,
        *,
        symbol: str,
        current_positions: dict[str, float],
        projected_positions: dict[str, float],
        prices: dict[str, float],
        equity_usd: float,
        bucket_map: dict[str, str] | None,
        hard_policy: HardRiskLimits,
        dynamic_policy: OperatingRiskLimits,
        operating_policy: OperatingRiskLimits,
    ) -> dict[str, float | str]:
        current_metrics = cls._exposure_metrics(
            positions=current_positions,
            prices=prices,
            equity_usd=equity_usd,
            bucket_map=bucket_map,
            adverse_mark_buffer_frac=float(dynamic_policy.adverse_mark_buffer_frac),
        )
        projected_metrics = cls._exposure_metrics(
            positions=projected_positions,
            prices=prices,
            equity_usd=equity_usd,
            bucket_map=bucket_map,
            adverse_mark_buffer_frac=float(operating_policy.adverse_mark_buffer_frac),
        )
        current_symbol = float(current_metrics["symbol_exposure_fracs"].get(symbol, 0.0))
        projected_symbol = float(projected_metrics["symbol_exposure_fracs"].get(symbol, 0.0))
        return {
            "risk_policy_version": str(hard_policy.policy_version),
            "hard_symbol_cap_frac": float(hard_policy.max_symbol_exposure_frac),
            "hard_gross_cap_frac": float(hard_policy.max_gross_exposure_frac),
            "hard_net_cap_frac": float(hard_policy.max_net_exposure_frac),
            "dynamic_symbol_cap_frac": float(dynamic_policy.max_symbol_exposure_frac),
            "dynamic_gross_cap_frac": float(dynamic_policy.max_gross_exposure_frac),
            "dynamic_net_cap_frac": float(dynamic_policy.max_net_exposure_frac),
            "operating_symbol_cap_frac": float(operating_policy.max_symbol_exposure_frac),
            "operating_gross_cap_frac": float(operating_policy.max_gross_exposure_frac),
            "operating_net_cap_frac": float(operating_policy.max_net_exposure_frac),
            "target_headroom_ratio": float(operating_policy.target_headroom_ratio),
            "reserve_capacity_frac": float(operating_policy.reserve_capacity_frac),
            "adverse_mark_buffer_frac": float(operating_policy.adverse_mark_buffer_frac),
            "current_symbol_exposure_frac": current_symbol,
            "projected_symbol_exposure_frac": projected_symbol,
            "current_gross_exposure_frac": float(current_metrics["gross_exposure_frac"]),
            "projected_gross_exposure_frac": float(projected_metrics["gross_exposure_frac"]),
            "current_net_exposure_frac": float(current_metrics["net_exposure_frac"]),
            "projected_net_exposure_frac": float(projected_metrics["net_exposure_frac"]),
            "symbol_headroom_frac": cls._headroom_value(
                operating_policy.max_symbol_exposure_frac,
                projected_symbol,
            ),
            "gross_headroom_frac": cls._headroom_value(
                operating_policy.max_gross_exposure_frac,
                float(projected_metrics["gross_exposure_frac"]),
            ),
            "net_headroom_frac": cls._headroom_value(
                operating_policy.max_net_exposure_frac,
                float(projected_metrics["net_exposure_frac"]),
            ),
        }

    @classmethod
    def _audit_action_class(cls, *, current_qty: float, side: str, quantity: float) -> str:
        eps = 1e-12
        next_qty = cls._project_after_position(
            current_qty=current_qty,
            side=side,
            quantity=quantity,
        )
        if abs(current_qty) <= eps and abs(next_qty) > eps:
            return "INCREASE"
        if current_qty * next_qty < -eps:
            return "FLIP"
        if abs(next_qty) <= eps and abs(current_qty) > eps:
            return "FLATTEN"
        if abs(next_qty) < abs(current_qty) - eps:
            return "REDUCE"
        return "INCREASE"

    @classmethod
    def _is_risk_reducing_order(cls, *, current_qty: float, side: str, quantity: float) -> bool:
        return cls._audit_action_class(
            current_qty=current_qty,
            side=side,
            quantity=quantity,
        ) in {"FLATTEN", "REDUCE"}

    @classmethod
    def _hard_pause_reduce_only_decision(
        cls,
        *,
        current_qty: float,
        side: str,
        quantity: float,
    ) -> tuple[bool, str, str, float]:
        eps = 1e-12
        after_qty = cls._project_after_position(
            current_qty=current_qty,
            side=side,
            quantity=quantity,
        )
        action_class = cls._audit_action_class(
            current_qty=current_qty,
            side=side,
            quantity=quantity,
        )
        if abs(current_qty) <= eps:
            return False, action_class, "hard_pause_new_symbol_exposure", after_qty
        if current_qty * after_qty < -eps:
            return False, "FLIP", "hard_pause_position_flip_blocked", after_qty
        if abs(after_qty) > abs(current_qty) + eps:
            return False, action_class, "hard_pause_increased_abs_exposure_blocked", after_qty
        if abs(after_qty) <= eps:
            return True, "FLATTEN", "hard_pause_reduce_only_flatten_allowed", 0.0
        return True, "REDUCE", "hard_pause_reduce_only_allowed", after_qty

    def _log_route_audit(
        self,
        *,
        user_id: int,
        pause_state: str,
        is_active: bool | None,
        live_mode: bool,
        symbol: str,
        side: str,
        qty: float,
        before_position: float | None,
        after_position: float | None,
        action_class: str,
        reason: str,
        accepted: bool | None = None,
        status: str = "",
        order_id: str = "",
        idempotency_key: str = "",
        mark_price: float = 0.0,
        risk_policy_version: str = "",
        hard_symbol_cap_frac: float = 0.0,
        hard_gross_cap_frac: float = 0.0,
        hard_net_cap_frac: float = 0.0,
        dynamic_symbol_cap_frac: float = 0.0,
        dynamic_gross_cap_frac: float = 0.0,
        dynamic_net_cap_frac: float = 0.0,
        operating_symbol_cap_frac: float = 0.0,
        operating_gross_cap_frac: float = 0.0,
        operating_net_cap_frac: float = 0.0,
        target_headroom_ratio: float = 0.85,
        reserve_capacity_frac: float = 0.0,
        adverse_mark_buffer_frac: float = 0.0,
        current_symbol_exposure_frac: float = 0.0,
        projected_symbol_exposure_frac: float = 0.0,
        current_gross_exposure_frac: float = 0.0,
        projected_gross_exposure_frac: float = 0.0,
        current_net_exposure_frac: float = 0.0,
        projected_net_exposure_frac: float = 0.0,
        symbol_headroom_frac: float = 0.0,
        gross_headroom_frac: float = 0.0,
        net_headroom_frac: float = 0.0,
        correlation_id: str = "",
    ) -> None:
        logger.info(
            "V2 route audit user_id=%s pause_state=%s is_active=%s live_mode=%s "
            "symbol=%s side=%s qty=%.12g before_position=%s after_position=%s "
            "action_class=%s reason=%s risk_policy_version=%s "
            "hard_caps=(%.4f,%.4f,%.4f) dynamic_caps=(%.4f,%.4f,%.4f) "
            "operating_caps=(%.4f,%.4f,%.4f) current_exp=(%.4f,%.4f,%.4f) "
            "projected_exp=(%.4f,%.4f,%.4f) headroom=(%.4f,%.4f,%.4f)",
            user_id,
            pause_state,
            "unknown" if is_active is None else bool(is_active),
            bool(live_mode),
            symbol,
            side,
            float(qty or 0.0),
            "unknown" if before_position is None else f"{float(before_position):.12g}",
            "unknown" if after_position is None else f"{float(after_position):.12g}",
            action_class,
            reason,
            risk_policy_version,
            float(hard_symbol_cap_frac),
            float(hard_gross_cap_frac),
            float(hard_net_cap_frac),
            float(dynamic_symbol_cap_frac),
            float(dynamic_gross_cap_frac),
            float(dynamic_net_cap_frac),
            float(operating_symbol_cap_frac),
            float(operating_gross_cap_frac),
            float(operating_net_cap_frac),
            float(current_symbol_exposure_frac),
            float(current_gross_exposure_frac),
            float(current_net_exposure_frac),
            float(projected_symbol_exposure_frac),
            float(projected_gross_exposure_frac),
            float(projected_net_exposure_frac),
            float(symbol_headroom_frac),
            float(gross_headroom_frac),
            float(net_headroom_frac),
        )
        callback = self._route_audit_callback
        if callback is None:
            return
        event = RouteAuditEvent(
            user_id=user_id,
            created_at=datetime.now(timezone.utc),
            correlation_id=str(correlation_id or ""),
            pause_state=pause_state,
            is_active=is_active,
            live_mode=bool(live_mode),
            symbol=str(symbol or ""),
            side=str(side or ""),
            quantity=float(qty or 0.0),
            before_position=before_position,
            after_position=after_position,
            action_class=str(action_class or ""),
            reason=str(reason or ""),
            accepted=accepted,
            status=str(status or ""),
            order_id=str(order_id or ""),
            idempotency_key=str(idempotency_key or ""),
            mark_price=float(mark_price or 0.0),
            risk_policy_version=str(risk_policy_version or ""),
            hard_symbol_cap_frac=float(hard_symbol_cap_frac or 0.0),
            hard_gross_cap_frac=float(hard_gross_cap_frac or 0.0),
            hard_net_cap_frac=float(hard_net_cap_frac or 0.0),
            dynamic_symbol_cap_frac=float(dynamic_symbol_cap_frac or 0.0),
            dynamic_gross_cap_frac=float(dynamic_gross_cap_frac or 0.0),
            dynamic_net_cap_frac=float(dynamic_net_cap_frac or 0.0),
            operating_symbol_cap_frac=float(operating_symbol_cap_frac or 0.0),
            operating_gross_cap_frac=float(operating_gross_cap_frac or 0.0),
            operating_net_cap_frac=float(operating_net_cap_frac or 0.0),
            target_headroom_ratio=float(target_headroom_ratio or 0.0),
            reserve_capacity_frac=float(reserve_capacity_frac or 0.0),
            adverse_mark_buffer_frac=float(adverse_mark_buffer_frac or 0.0),
            current_symbol_exposure_frac=float(current_symbol_exposure_frac or 0.0),
            projected_symbol_exposure_frac=float(projected_symbol_exposure_frac or 0.0),
            current_gross_exposure_frac=float(current_gross_exposure_frac or 0.0),
            projected_gross_exposure_frac=float(projected_gross_exposure_frac or 0.0),
            current_net_exposure_frac=float(current_net_exposure_frac or 0.0),
            projected_net_exposure_frac=float(projected_net_exposure_frac or 0.0),
            symbol_headroom_frac=float(symbol_headroom_frac or 0.0),
            gross_headroom_frac=float(gross_headroom_frac or 0.0),
            net_headroom_frac=float(net_headroom_frac or 0.0),
        )
        try:
            callback(event)
        except Exception as exc:
            logger.warning(
                "Route-audit callback failed for user %s symbol=%s: %s",
                user_id,
                symbol,
                exc,
            )

    def _emit_stage_telemetry(
        self,
        *,
        user_id: int,
        stage: str,
        started_at: float,
        correlation_id: str = "",
        status: str = "",
        detail: str = "",
    ) -> None:
        callback = self._stage_telemetry_callback
        if callback is None:
            return
        event = ExecutionStageTelemetry(
            user_id=user_id,
            created_at=datetime.now(timezone.utc),
            stage=str(stage or ""),
            duration_ms=max((perf_counter() - started_at) * 1000.0, 0.0),
            correlation_id=str(correlation_id or ""),
            status=str(status or ""),
            detail=str(detail or ""),
        )
        try:
            callback(event)
        except Exception as exc:
            logger.warning(
                "Stage-telemetry callback failed for user %s stage=%s: %s",
                user_id,
                stage,
                exc,
            )

    @staticmethod
    def _build_skipped_result(
        *,
        idempotency_key: str,
        plan: object,
        mark_price: float,
        reason: str,
    ) -> ExecutionResult:
        return ExecutionResult(
            accepted=False,
            order_id="",
            idempotency_key=idempotency_key,
            symbol=str(getattr(plan, "symbol", "")),
            side=str(getattr(plan, "side", "BUY")),
            requested_qty=float(getattr(plan, "quantity", 0.0) or 0.0),
            filled_qty=0.0,
            avg_price=float(mark_price or 0.0),
            status="skipped",
            created_at=datetime.now(timezone.utc).isoformat(),
            reason=reason,
        )

    @classmethod
    def _update_execution_diagnostics(
        cls,
        current: ExecutionDiagnostics,
        *,
        results: Iterable[ExecutionResult],
        prices: dict[str, float],
        activity_by_key: dict[str, str] | None = None,
    ) -> ExecutionDiagnostics:
        total_orders = current.total_orders
        accepted_orders = current.accepted_orders
        rejected_orders = current.rejected_orders
        slippage_sum = current.avg_adverse_slippage_bps * current.slippage_sample_count
        slippage_samples = current.slippage_sample_count
        entry_orders = current.entry_orders
        rebalance_orders = current.rebalance_orders
        exit_orders = current.exit_orders
        skipped_by_filter = current.skipped_by_filter
        skipped_by_deadband = current.skipped_by_deadband

        activity_lookup = activity_by_key or {}

        for result in results:
            reason = (result.reason or "").strip().lower()
            if reason.startswith("skipped_by_filter"):
                skipped_by_filter += 1
                continue
            if reason.startswith("skipped_by_deadband"):
                skipped_by_deadband += 1
                continue

            total_orders += 1
            if result.accepted:
                accepted_orders += 1
                activity = activity_lookup.get(result.idempotency_key)
                if activity == "entry":
                    entry_orders += 1
                elif activity == "exit":
                    exit_orders += 1
                elif activity == "rebalance":
                    rebalance_orders += 1
            else:
                rejected_orders += 1

            mark_price = float(prices.get(result.symbol, 0.0))
            adverse_slippage_bps = cls._adverse_slippage_bps(
                result.side,
                fill_price=result.avg_price,
                mark_price=mark_price,
            )
            if adverse_slippage_bps is not None:
                slippage_sum += adverse_slippage_bps
                slippage_samples += 1

        reject_rate = (rejected_orders / total_orders) if total_orders else 0.0
        avg_adverse_slippage_bps = (slippage_sum / slippage_samples) if slippage_samples else 0.0

        return ExecutionDiagnostics(
            total_orders=total_orders,
            accepted_orders=accepted_orders,
            rejected_orders=rejected_orders,
            reject_rate=float(reject_rate),
            slippage_sample_count=slippage_samples,
            avg_adverse_slippage_bps=float(avg_adverse_slippage_bps),
            entry_orders=entry_orders,
            rebalance_orders=rebalance_orders,
            exit_orders=exit_orders,
            skipped_by_filter=skipped_by_filter,
            skipped_by_deadband=skipped_by_deadband,
            paused_cycles=current.paused_cycles,
            blocked_actionable_signals=current.blocked_actionable_signals,
            routed_signals_total=current.routed_signals_total,
            routed_buy_signals=current.routed_buy_signals,
            routed_sell_signals=current.routed_sell_signals,
            routed_actionable_signals=current.routed_actionable_signals,
            live_go_no_go_passed=current.live_go_no_go_passed,
            rollback_required=current.rollback_required,
            rollout_failure_streak=current.rollout_failure_streak,
            rollout_gate_reasons=current.rollout_gate_reasons,
            effective_symbol_cap_frac=current.effective_symbol_cap_frac,
            effective_gross_cap_frac=current.effective_gross_cap_frac,
            effective_net_cap_frac=current.effective_net_cap_frac,
            hard_symbol_cap_frac=current.hard_symbol_cap_frac,
            hard_gross_cap_frac=current.hard_gross_cap_frac,
            hard_net_cap_frac=current.hard_net_cap_frac,
            target_headroom_ratio=current.target_headroom_ratio,
            reserve_capacity_frac=current.reserve_capacity_frac,
            adverse_mark_buffer_frac=current.adverse_mark_buffer_frac,
            risk_policy_version=current.risk_policy_version,
        )

    @staticmethod
    def _update_paper_entry_price(
        state: _SessionState,
        *,
        results: Iterable[ExecutionResult],
        prices: dict[str, float],
        starting_positions: dict[str, float],
    ) -> None:
        running_positions: dict[str, float] = {
            symbol: float(qty)
            for symbol, qty in starting_positions.items()
        }
        now_utc = datetime.now(timezone.utc)

        for result in results:
            fill_qty = float(result.economic_filled_qty if result.accepted else 0.0)
            if not result.accepted or fill_qty <= 0.0:
                continue
            idempotency_key = str(result.idempotency_key or "")

            symbol = result.symbol
            current_qty = float(running_positions.get(symbol, 0.0))
            entry_price = float(state.paper_entry_price.get(symbol, 0.0) or 0.0)
            previous_opened_at = state.position_opened_at.get(symbol)

            fill_price = float(result.avg_price)
            if fill_price <= 0.0:
                fill_price = float(prices.get(symbol, 0.0) or 0.0)
            if fill_price <= 0.0:
                continue

            next_qty, next_entry, realized_pnl = RoutedExecutionService._apply_paper_fill(
                current_qty=current_qty,
                current_entry_price=entry_price,
                side=result.side,
                fill_qty=fill_qty,
                fill_price=fill_price,
            )
            if realized_pnl != 0.0:
                state.equity_baseline_usd = max(1.0, float(state.equity_baseline_usd + realized_pnl))
            if idempotency_key:
                state.accounted_order_keys.add(idempotency_key)

            if abs(next_qty) <= 1e-12:
                state.paper_entry_price.pop(symbol, None)
                running_positions.pop(symbol, None)
                state.position_opened_at.pop(symbol, None)
                state.position_entry_atr_pct.pop(symbol, None)
                state.position_peak_pnl_pct.pop(symbol, None)
            else:
                state.paper_entry_price[symbol] = next_entry
                running_positions[symbol] = next_qty
                flipped_direction = (current_qty > 0.0 > next_qty) or (current_qty < 0.0 < next_qty)
                if abs(current_qty) <= 1e-12 or flipped_direction:
                    state.position_opened_at[symbol] = now_utc
                    state.position_peak_pnl_pct.pop(symbol, None)
                    # Store ATR% from the latest signal for take-profit scaling
                    latest_signal = state.latest_signals.get(symbol)
                    if latest_signal is not None and latest_signal.atr_pct is not None:
                        state.position_entry_atr_pct[symbol] = float(latest_signal.atr_pct)
                    else:
                        state.position_entry_atr_pct.pop(symbol, None)
                elif previous_opened_at is not None:
                    state.position_opened_at[symbol] = previous_opened_at
                else:
                    state.position_opened_at[symbol] = now_utc

    @staticmethod
    def _apply_paper_fill(
        *,
        current_qty: float,
        current_entry_price: float,
        side: str,
        fill_qty: float,
        fill_price: float,
    ) -> tuple[float, float, float]:
        eps = 1e-12
        if fill_qty <= 0.0 or fill_price <= 0.0:
            return float(current_qty), float(current_entry_price), 0.0

        side_sign = 1.0 if side == "BUY" else -1.0
        next_qty = float(current_qty) + (side_sign * float(fill_qty))
        realized_pnl = 0.0

        if abs(current_qty) > eps:
            reducing_long = current_qty > 0.0 and side_sign < 0.0
            reducing_short = current_qty < 0.0 and side_sign > 0.0
            if reducing_long or reducing_short:
                closed_qty = min(abs(current_qty), float(fill_qty))
                if reducing_long:
                    realized_pnl = (float(fill_price) - float(current_entry_price)) * closed_qty
                else:
                    realized_pnl = (float(current_entry_price) - float(fill_price)) * closed_qty

        if abs(current_qty) <= eps:
            return next_qty, float(fill_price), realized_pnl

        same_direction = (current_qty > 0 and side_sign > 0) or (current_qty < 0 and side_sign < 0)
        if same_direction:
            total_abs = abs(current_qty) + float(fill_qty)
            if total_abs <= eps:
                return 0.0, 0.0, realized_pnl
            weighted = (abs(current_qty) * float(current_entry_price)) + (float(fill_qty) * float(fill_price))
            return next_qty, (weighted / total_abs), realized_pnl

        remaining = abs(current_qty) - float(fill_qty)
        if remaining > eps:
            return next_qty, float(current_entry_price), realized_pnl
        if abs(remaining) <= eps:
            return 0.0, 0.0, realized_pnl
        return next_qty, float(fill_price), realized_pnl

    @classmethod
    def _resolve_symbol_pnl(
        cls,
        state: _SessionState,
        *,
        open_positions: dict[str, float],
        prices: dict[str, float],
        live_metrics: dict[str, dict[str, float]] | None = None,
    ) -> dict[str, float]:
        symbol_pnl: dict[str, float] = {}

        resolved_live_metrics: dict[str, dict[str, float]] = {}
        if state.mode == "live":
            if live_metrics is not None:
                resolved_live_metrics = live_metrics
            else:
                try:
                    resolved_live_metrics = cls._safe_get_position_metrics(state.adapter)
                except Exception:
                    resolved_live_metrics = {}

        for symbol, qty in open_positions.items():
            signed_qty = float(qty)
            if signed_qty == 0.0:
                continue

            if state.mode == "live":
                metrics = resolved_live_metrics.get(symbol, {})
                unrealized = float(metrics.get("unrealized_pnl_usd", 0.0) or 0.0)
                if unrealized != 0.0:
                    symbol_pnl[symbol] = unrealized
                    continue
                entry_price = float(metrics.get("entry_price", 0.0) or 0.0)
            else:
                entry_price = float(state.paper_entry_price.get(symbol, 0.0) or 0.0)

            mark_price = float(prices.get(symbol, 0.0) or 0.0)
            if entry_price <= 0.0 or mark_price <= 0.0:
                continue

            symbol_pnl[symbol] = (mark_price - entry_price) * signed_qty
        return symbol_pnl

    @staticmethod
    def _adverse_slippage_bps(
        side: str,
        *,
        fill_price: float,
        mark_price: float,
    ) -> float | None:
        if mark_price <= 0.0 or fill_price <= 0.0:
            return None

        if side == "BUY":
            raw_bps = ((fill_price - mark_price) / mark_price) * 10_000.0
        elif side == "SELL":
            raw_bps = ((mark_price - fill_price) / mark_price) * 10_000.0
        else:
            return None
        return max(float(raw_bps), 0.0)

    @classmethod
    def _build_snapshot(
        cls,
        adapter: object,
        *,
        equity_usd: float,
        risk_policy: PortfolioRiskPolicy,
        prices: dict[str, float],
        open_positions: dict[str, float] | None = None,
    ) -> PortfolioSnapshot:
        resolved_positions = (
            {symbol: float(qty) for symbol, qty in open_positions.items()}
            if open_positions is not None
            else cls._safe_get_positions(adapter)
        )

        gross = 0.0
        net = 0.0
        symbol_notionals: dict[str, float] = {}
        if equity_usd > 0.0 and prices:
            for symbol, qty in resolved_positions.items():
                mark_price = float(prices.get(symbol, 0.0))
                if mark_price <= 0.0:
                    continue
                notional = float(qty) * mark_price
                symbol_notionals[symbol] = abs(notional)
                gross += abs(notional) / equity_usd
                net += notional / equity_usd

        risk_budget = 0.0
        if risk_policy.max_gross_exposure_frac > 0.0:
            risk_budget = gross / risk_policy.max_gross_exposure_frac

        return PortfolioSnapshot(
            timestamp=datetime.now(timezone.utc),
            equity_usd=equity_usd,
            open_positions=resolved_positions,
            symbol_notional_usd=symbol_notionals,
            risk=RiskSnapshot(
                gross_exposure_frac=float(gross),
                net_exposure_frac=float(net),
                max_drawdown_frac=0.0,
                risk_budget_used_frac=float(risk_budget),
            ),
        )
