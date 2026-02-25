"""Execution service boundary used by Telegram and other control-plane adapters."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
import logging
import os
from typing import Callable, Protocol

from quant_v2.config import get_runtime_profile
from quant_v2.contracts import OrderPlan, PortfolioSnapshot, RiskSnapshot, StrategySignal
from quant_v2.execution.adapters import ExecutionResult
from quant_v2.execution.idempotency import build_idempotency_key
from quant_v2.execution.planner import PlannerConfig, build_execution_intents
from quant_v2.execution.reconciler import reconcile_target_exposures
from quant_v2.monitoring.kill_switch import (
    KillSwitchConfig,
    KillSwitchEvaluation,
    MonitoringSnapshot,
    evaluate_kill_switch,
)
from quant_v2.portfolio.risk_policy import PortfolioRiskPolicy

logger = logging.getLogger(__name__)


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
    live_go_no_go_passed: bool = True
    rollback_required: bool = False
    rollout_failure_streak: int = 0
    rollout_gate_reasons: tuple[str, ...] = field(default_factory=tuple)
    effective_symbol_cap_frac: float = 0.0
    effective_gross_cap_frac: float = 0.0
    effective_net_cap_frac: float = 0.0


@dataclass(frozen=True)
class LifecycleRules:
    """Configurable position lifecycle controls for auto-exit behaviors."""

    auto_close_horizon_bars: int = 0
    stop_loss_pct: float = 0.0

    def __post_init__(self) -> None:
        if self.auto_close_horizon_bars < 0:
            raise ValueError("auto_close_horizon_bars must be >= 0")
        if not 0.0 <= self.stop_loss_pct < 1.0:
            raise ValueError("stop_loss_pct must be within [0, 1)")


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

    def set_monitoring_snapshot(
        self,
        user_id: int,
        snapshot: MonitoringSnapshot,
    ) -> KillSwitchEvaluation:
        """Update monitoring snapshot and return current kill-switch evaluation."""

    def get_kill_switch_evaluation(self, user_id: int) -> KillSwitchEvaluation | None:
        """Return latest kill-switch evaluation for a session."""

    async def route_signals(
        self,
        user_id: int,
        *,
        signals: Iterable[StrategySignal],
        prices: dict[str, float],
        monitoring_snapshot: MonitoringSnapshot | None = None,
    ) -> tuple[ExecutionResult, ...]:
        """Route strategy signals through execution planner and adapter."""


@dataclass
class _SessionState:
    """Internal runtime state for a started execution session."""

    request: SessionRequest
    adapter: object
    mode: str
    snapshot: PortfolioSnapshot
    effective_risk_policy: PortfolioRiskPolicy
    equity_baseline_usd: float = 10_000.0
    last_prices: dict[str, float] = field(default_factory=dict)
    latest_signals: dict[str, StrategySignal] = field(default_factory=dict)
    diagnostics: ExecutionDiagnostics = field(default_factory=ExecutionDiagnostics)
    paper_entry_price: dict[str, float] = field(default_factory=dict)
    position_opened_at: dict[str, datetime] = field(default_factory=dict)
    last_rebalance_at: dict[str, datetime] = field(default_factory=dict)
    external_execution_anomaly_rate: float = 0.0
    external_hard_risk_breach: bool = False
    lifecycle_rules: LifecycleRules = field(default_factory=LifecycleRules)
    monitoring_snapshot: MonitoringSnapshot = field(default_factory=MonitoringSnapshot)
    kill_switch: KillSwitchEvaluation = field(
        default_factory=lambda: KillSwitchEvaluation(pause_trading=False)
    )


class InMemoryExecutionService:
    """Reference in-memory implementation for integration and migration testing."""

    def __init__(self) -> None:
        self._sessions: dict[int, SessionRequest] = {}
        self._snapshots: dict[int, PortfolioSnapshot] = {}
        self._monitoring: dict[int, MonitoringSnapshot] = {}
        self._kill_switch: dict[int, KillSwitchEvaluation] = {}

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
        return True

    async def stop_session(self, user_id: int) -> bool:
        existed = user_id in self._sessions
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

    def get_portfolio_snapshot(self, user_id: int) -> PortfolioSnapshot | None:
        return self._snapshots.get(user_id)

    async def route_signals(
        self,
        user_id: int,
        *,
        signals: Iterable[StrategySignal],
        prices: dict[str, float],
        monitoring_snapshot: MonitoringSnapshot | None = None,
    ) -> tuple[ExecutionResult, ...]:
        # Legacy placeholder backend; signal routing is implemented in RoutedExecutionService.
        _ = (user_id, tuple(signals), prices, monitoring_snapshot)
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
    ) -> None:
        self._sessions: dict[int, _SessionState] = {}
        self._paper_adapter_factory = paper_adapter_factory or self._default_paper_adapter_factory
        self._live_adapter_factory = live_adapter_factory or self._default_live_adapter_factory
        self._initial_equity_usd = float(initial_equity_usd)
        self._risk_policy = risk_policy or PortfolioRiskPolicy()
        self._planner_config = planner_config or PlannerConfig()
        self._kill_switch_config = kill_switch_config or KillSwitchConfig()
        self._allow_live_execution = bool(allow_live_execution)
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
                0,
            )
        self._rebalance_cooldown_seconds = max(int(rebalance_cooldown_seconds), 0)

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
            effective_risk_policy=session_policy,
            equity_baseline_usd=self._initial_equity_usd,
            last_prices={},
            diagnostics=diagnostics,
        )
        return True

    async def stop_session(self, user_id: int) -> bool:
        existed = user_id in self._sessions
        self._sessions.pop(user_id, None)
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
            risk_policy=state.effective_risk_policy,
            prices={},
        )
        diagnostics = self._build_initial_diagnostics(state.effective_risk_policy)
        self._sessions[user_id] = _SessionState(
            request=state.request,
            adapter=adapter,
            mode=state.mode,
            snapshot=snapshot,
            effective_risk_policy=state.effective_risk_policy,
            equity_baseline_usd=self._initial_equity_usd,
            last_prices={},
            diagnostics=diagnostics,
        )
        return True

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
    ) -> tuple[ExecutionResult, ...]:
        state = self._sessions.get(user_id)
        if state is None:
            raise KeyError(f"No active session for user {user_id}")
        if min_qty < 0.0:
            raise ValueError("min_qty must be >= 0")

        signal_list = tuple(signals)

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

        precheck_policy = (
            self._apply_canary_risk_cap(risk_policy)
            if (risk_policy is not None and state.request.live)
            else (risk_policy or state.effective_risk_policy)
        )
        self._apply_snapshot_risk_monitoring(
            state,
            risk_policy=precheck_policy,
        )
        if state.kill_switch.pause_trading:
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

        if not signal_list:
            return ()

        incoming_prices = {
            symbol: float(price)
            for symbol, price in prices.items()
            if float(price) > 0.0
        }
        if incoming_prices:
            state.last_prices = {
                **state.last_prices,
                **incoming_prices,
            }
        planning_prices = state.last_prices

        for signal in signal_list:
            symbol = str(signal.symbol).strip().upper()
            if not symbol:
                continue
            normalized_signal = signal if signal.symbol == symbol else replace(signal, symbol=symbol)
            if normalized_signal.actionable:
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

        if risk_policy is None:
            policy = state.effective_risk_policy
        else:
            policy = self._apply_canary_risk_cap(risk_policy) if state.request.live else risk_policy
        state.diagnostics = replace(
            state.diagnostics,
            effective_symbol_cap_frac=float(policy.max_symbol_exposure_frac),
            effective_gross_cap_frac=float(policy.max_gross_exposure_frac),
            effective_net_cap_frac=float(policy.max_net_exposure_frac),
        )
        cfg = planner_config or self._planner_config

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

        intent_plan = build_execution_intents(
            planning_signals,
            policy=policy,
            config=cfg,
            bucket_map=bucket_map,
        )
        order_plans = reconcile_target_exposures(
            intent_plan.policy_result.exposures,
            current_positions_qty=current_positions,
            prices=planning_prices,
            equity_usd=planning_equity,
            min_qty=min_qty,
        )

        now_utc = datetime.now(timezone.utc)
        epoch_minute = int(now_utc.timestamp() // 60)
        results: list[ExecutionResult] = []
        activity_by_key: dict[str, str] = {}
        attempted_orders = 0
        for plan in order_plans:
            idempotency_key = build_idempotency_key(
                user_id=user_id,
                plan=plan,
                epoch_minute=epoch_minute,
            )
            mark_price = float(planning_prices.get(plan.symbol, 0.0) or 0.0)
            current_qty = float(current_positions.get(plan.symbol, 0.0))
            activity = self._classify_order_activity(
                current_qty=current_qty,
                side=plan.side,
                quantity=float(plan.quantity),
            )

            if activity == "rebalance":
                delta_notional_usd = abs(float(plan.quantity) * mark_price)
                if (
                    self._min_rebalance_weight_drift > 0.0
                    and mark_price > 0.0
                    and planning_equity > 0.0
                ):
                    weight_drift = delta_notional_usd / planning_equity
                    # A strict $50.0 minimum absolute USD drift deadband prevents dust rebalances on small portfolios
                    # Additionally, we check the weight drift against _min_rebalance_weight_drift (e.g. 1%)
                    min_absolute_drift_usd = 50.0 
                    
                    if weight_drift < self._min_rebalance_weight_drift and delta_notional_usd < min_absolute_drift_usd:
                        results.append(
                            self._build_skipped_result(
                                idempotency_key=idempotency_key,
                                plan=plan,
                                mark_price=mark_price,
                                reason="skipped_by_deadband:min_weight_drift_and_absolute_usd",
                            )
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
                            continue

            if self._max_orders_per_cycle > 0 and attempted_orders >= self._max_orders_per_cycle:
                results.append(
                    self._build_skipped_result(
                        idempotency_key=idempotency_key,
                        plan=plan,
                        mark_price=mark_price,
                        reason="skipped_by_deadband:max_orders_per_cycle",
                    )
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
                
                result = state.adapter.place_order(
                    plan,
                    idempotency_key=idempotency_key,
                    mark_price=mark_price,
                    limit_price=limit_price,
                    post_only=True,
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
            results.append(result)
            activity_by_key[idempotency_key] = activity

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
        return tuple(results)

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
            plan = OrderPlan(
                symbol=symbol,
                side=side,
                quantity=quantity,
                reduce_only=False,
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
                result = state.adapter.place_order(
                    plan,
                    idempotency_key=idempotency_key,
                    mark_price=mark_price,
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
            risk_policy=state.effective_risk_policy,
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

    def clear_execution_diagnostics(self, user_id: int) -> bool:
        """Reset execution telemetry counters while keeping the session and positions intact."""

        state = self._sessions.get(user_id)
        if state is None:
            return False

        state.diagnostics = self._build_initial_diagnostics(state.effective_risk_policy)
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
            risk_policy=state.effective_risk_policy,
        )
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
    ) -> None:
        computed_breach = self._compute_hard_risk_breach(
            state.snapshot,
            risk_policy=risk_policy,
        )
        combined_hard_breach = bool(state.external_hard_risk_breach or computed_breach)
        if combined_hard_breach != state.monitoring_snapshot.hard_risk_breach:
            state.monitoring_snapshot = replace(
                state.monitoring_snapshot,
                hard_risk_breach=combined_hard_breach,
            )

        state.kill_switch = evaluate_kill_switch(
            state.monitoring_snapshot,
            config=self._kill_switch_config,
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
        if rules.auto_close_horizon_bars <= 0 and rules.stop_loss_pct <= 0.0:
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

        live_metrics: dict[str, dict[str, float]] = {}
        if state.mode == "live" and rules.stop_loss_pct > 0.0:
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

            if rules.auto_close_horizon_bars > 0:
                max_age = timedelta(hours=float(rules.auto_close_horizon_bars))
                if now_utc - opened_at >= max_age:
                    forced.add(symbol)
                    continue

            if rules.stop_loss_pct <= 0.0:
                continue

            if state.mode == "live":
                entry_price = float((live_metrics.get(symbol) or {}).get("entry_price", 0.0) or 0.0)
            else:
                entry_price = float(state.paper_entry_price.get(symbol, 0.0) or 0.0)
            mark_price = float(prices.get(symbol, 0.0) or 0.0)
            if entry_price <= 0.0 or mark_price <= 0.0:
                continue

            if signed_qty > 0.0:
                stop_level = entry_price * (1.0 - rules.stop_loss_pct)
                if mark_price <= stop_level:
                    forced.add(symbol)
            else:
                stop_level = entry_price * (1.0 + rules.stop_loss_pct)
                if mark_price >= stop_level:
                    forced.add(symbol)

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

    def _build_initial_diagnostics(self, policy: PortfolioRiskPolicy) -> ExecutionDiagnostics:
        return ExecutionDiagnostics(
            live_go_no_go_passed=self._live_go_no_go_passed,
            rollback_required=self._rollback_required,
            rollout_failure_streak=self._rollout_failure_streak,
            rollout_gate_reasons=self._current_live_rollout_gate_reasons(),
            effective_symbol_cap_frac=float(policy.max_symbol_exposure_frac),
            effective_gross_cap_frac=float(policy.max_gross_exposure_frac),
            effective_net_cap_frac=float(policy.max_net_exposure_frac),
        )

    def _resolve_session_risk_policy(self, request: SessionRequest) -> PortfolioRiskPolicy:
        if not request.live:
            return self._risk_policy
        return self._apply_canary_risk_cap(self._risk_policy)

    def _apply_canary_risk_cap(self, policy: PortfolioRiskPolicy) -> PortfolioRiskPolicy:
        cap = self._canary_live_risk_cap_frac
        return PortfolioRiskPolicy(
            max_symbol_exposure_frac=min(policy.max_symbol_exposure_frac, cap),
            max_gross_exposure_frac=min(policy.max_gross_exposure_frac, cap),
            max_net_exposure_frac=min(policy.max_net_exposure_frac, cap),
            correlation_bucket_caps={
                bucket: min(float(limit), cap)
                for bucket, limit in policy.correlation_bucket_caps.items()
            },
        )

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
            live_go_no_go_passed=current.live_go_no_go_passed,
            rollback_required=current.rollback_required,
            rollout_failure_streak=current.rollout_failure_streak,
            rollout_gate_reasons=current.rollout_gate_reasons,
            effective_symbol_cap_frac=current.effective_symbol_cap_frac,
            effective_gross_cap_frac=current.effective_gross_cap_frac,
            effective_net_cap_frac=current.effective_net_cap_frac,
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
            if not result.accepted or result.filled_qty <= 0.0:
                continue

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
                fill_qty=float(result.filled_qty),
                fill_price=fill_price,
            )
            if realized_pnl != 0.0:
                state.equity_baseline_usd = max(1.0, float(state.equity_baseline_usd + realized_pnl))

            if abs(next_qty) <= 1e-12:
                state.paper_entry_price.pop(symbol, None)
                running_positions.pop(symbol, None)
                state.position_opened_at.pop(symbol, None)
            else:
                state.paper_entry_price[symbol] = next_entry
                running_positions[symbol] = next_qty
                flipped_direction = (current_qty > 0.0 > next_qty) or (current_qty < 0.0 < next_qty)
                if abs(current_qty) <= 1e-12 or flipped_direction:
                    state.position_opened_at[symbol] = now_utc
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
