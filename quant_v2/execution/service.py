"""Execution service boundary used by Telegram and other control-plane adapters."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace
from dataclasses import dataclass, field
from datetime import datetime, timezone
import logging
import os
from typing import Callable, Protocol

from quant_v2.config import get_runtime_profile
from quant_v2.contracts import PortfolioSnapshot, RiskSnapshot, StrategySignal
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
    live_go_no_go_passed: bool = True
    rollback_required: bool = False
    rollout_failure_streak: int = 0
    rollout_gate_reasons: tuple[str, ...] = field(default_factory=tuple)
    effective_symbol_cap_frac: float = 0.0
    effective_gross_cap_frac: float = 0.0
    effective_net_cap_frac: float = 0.0


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
    last_prices: dict[str, float] = field(default_factory=dict)
    diagnostics: ExecutionDiagnostics = field(default_factory=ExecutionDiagnostics)
    paper_entry_price: dict[str, float] = field(default_factory=dict)
    last_rebalance_at: dict[str, datetime] = field(default_factory=dict)
    external_execution_anomaly_rate: float = 0.0
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
            min_rebalance_notional_usd = self._parse_float_env(
                "BOT_V2_MIN_REBALANCE_NOTIONAL_USD",
                10.0,
            )
        self._min_rebalance_notional_usd = max(float(min_rebalance_notional_usd), 0.0)

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
            state.snapshot = self._build_snapshot(
                state.adapter,
                equity_usd=state.snapshot.equity_usd,
                risk_policy=state.effective_risk_policy,
                prices=state.last_prices,
            )
            if state.snapshot.open_positions:
                state.snapshot = replace(
                    state.snapshot,
                    symbol_pnl_usd=self._resolve_symbol_pnl(
                        state,
                        open_positions=state.snapshot.open_positions,
                        prices=state.last_prices,
                    ),
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

        if state.mode == "live":
            self._refresh_runtime_rollout_controls()
            gate_reasons = self._current_live_rollout_gate_reasons()
            if gate_reasons:
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

        state.kill_switch = evaluate_kill_switch(
            state.monitoring_snapshot,
            config=self._kill_switch_config,
        )
        if state.kill_switch.pause_trading:
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

        signal_list = tuple(signals)
        if not signal_list:
            return ()
        if not any(signal.actionable for signal in signal_list):
            return ()

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

        intent_plan = build_execution_intents(
            signal_list,
            policy=policy,
            config=cfg,
            bucket_map=bucket_map,
        )
        current_positions = self._safe_get_positions(state.adapter)
        order_plans = reconcile_target_exposures(
            intent_plan.policy_result.exposures,
            current_positions_qty=current_positions,
            prices=prices,
            equity_usd=state.snapshot.equity_usd,
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
            mark_price = float(prices.get(plan.symbol, 0.0) or 0.0)
            current_qty = float(current_positions.get(plan.symbol, 0.0))
            activity = self._classify_order_activity(
                current_qty=current_qty,
                side=plan.side,
                quantity=float(plan.quantity),
            )

            if activity == "rebalance":
                delta_notional_usd = abs(float(plan.quantity) * mark_price)
                if (
                    self._min_rebalance_notional_usd > 0.0
                    and mark_price > 0.0
                    and delta_notional_usd < self._min_rebalance_notional_usd
                ):
                    results.append(
                        self._build_skipped_result(
                            idempotency_key=idempotency_key,
                            plan=plan,
                            mark_price=mark_price,
                            reason="skipped_by_deadband:min_notional_delta",
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
                result = state.adapter.place_order(
                    plan,
                    idempotency_key=idempotency_key,
                    mark_price=mark_price,
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
            prices=prices,
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
                prices=prices,
                starting_positions=current_positions,
            )

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
        merged_anomaly = max(
            state.external_execution_anomaly_rate,
            state.diagnostics.reject_rate,
        )
        if merged_anomaly != state.monitoring_snapshot.execution_anomaly_rate:
            state.monitoring_snapshot = replace(
                state.monitoring_snapshot,
                execution_anomaly_rate=merged_anomaly,
            )

        state.kill_switch = evaluate_kill_switch(
            state.monitoring_snapshot,
            config=self._kill_switch_config,
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

        state.snapshot = self._build_snapshot(
            state.adapter,
            equity_usd=state.snapshot.equity_usd,
            risk_policy=policy,
            prices=state.last_prices,
        )
        if state.snapshot.open_positions:
            state.snapshot = replace(
                state.snapshot,
                symbol_pnl_usd=self._resolve_symbol_pnl(
                    state,
                    open_positions=state.snapshot.open_positions,
                    prices=state.last_prices,
                ),
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
        merged_anomaly = max(
            state.external_execution_anomaly_rate,
            state.diagnostics.reject_rate,
        )
        state.monitoring_snapshot = replace(
            snapshot,
            execution_anomaly_rate=merged_anomaly,
        )
        state.kill_switch = evaluate_kill_switch(
            state.monitoring_snapshot,
            config=self._kill_switch_config,
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

    def get_session_mode(self, user_id: int) -> str | None:
        """Return running mode label for diagnostics (live/paper)."""

        state = self._sessions.get(user_id)
        if state is None:
            return None
        return state.mode

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
            metrics[str(symbol)] = {
                "entry_price": entry_price,
                "unrealized_pnl_usd": unrealized,
            }
        return metrics

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

        for result in results:
            if not result.accepted or result.filled_qty <= 0.0:
                continue

            symbol = result.symbol
            current_qty = float(running_positions.get(symbol, 0.0))
            entry_price = float(state.paper_entry_price.get(symbol, 0.0) or 0.0)

            fill_price = float(result.avg_price)
            if fill_price <= 0.0:
                fill_price = float(prices.get(symbol, 0.0) or 0.0)
            if fill_price <= 0.0:
                continue

            next_qty, next_entry = RoutedExecutionService._apply_paper_fill(
                current_qty=current_qty,
                current_entry_price=entry_price,
                side=result.side,
                fill_qty=float(result.filled_qty),
                fill_price=fill_price,
            )
            if abs(next_qty) <= 1e-12:
                state.paper_entry_price.pop(symbol, None)
                running_positions.pop(symbol, None)
            else:
                state.paper_entry_price[symbol] = next_entry
                running_positions[symbol] = next_qty

    @staticmethod
    def _apply_paper_fill(
        *,
        current_qty: float,
        current_entry_price: float,
        side: str,
        fill_qty: float,
        fill_price: float,
    ) -> tuple[float, float]:
        eps = 1e-12
        if fill_qty <= 0.0 or fill_price <= 0.0:
            return float(current_qty), float(current_entry_price)

        side_sign = 1.0 if side == "BUY" else -1.0
        next_qty = float(current_qty) + (side_sign * float(fill_qty))

        if abs(current_qty) <= eps:
            return next_qty, float(fill_price)

        same_direction = (current_qty > 0 and side_sign > 0) or (current_qty < 0 and side_sign < 0)
        if same_direction:
            total_abs = abs(current_qty) + float(fill_qty)
            if total_abs <= eps:
                return 0.0, 0.0
            weighted = (abs(current_qty) * float(current_entry_price)) + (float(fill_qty) * float(fill_price))
            return next_qty, (weighted / total_abs)

        remaining = abs(current_qty) - float(fill_qty)
        if remaining > eps:
            return next_qty, float(current_entry_price)
        if abs(remaining) <= eps:
            return 0.0, 0.0
        return next_qty, float(fill_price)

    @classmethod
    def _resolve_symbol_pnl(
        cls,
        state: _SessionState,
        *,
        open_positions: dict[str, float],
        prices: dict[str, float],
    ) -> dict[str, float]:
        symbol_pnl: dict[str, float] = {}

        live_metrics: dict[str, dict[str, float]] = {}
        if state.mode == "live":
            try:
                live_metrics = cls._safe_get_position_metrics(state.adapter)
            except Exception:
                live_metrics = {}

        for symbol, qty in open_positions.items():
            signed_qty = float(qty)
            if signed_qty == 0.0:
                continue

            if state.mode == "live":
                metrics = live_metrics.get(symbol, {})
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
    ) -> PortfolioSnapshot:
        open_positions = cls._safe_get_positions(adapter)

        gross = 0.0
        net = 0.0
        symbol_notionals: dict[str, float] = {}
        if equity_usd > 0.0 and prices:
            for symbol, qty in open_positions.items():
                mark_price = float(prices.get(symbol, 0.0))
                if mark_price <= 0.0:
                    continue
                notional = float(qty) * mark_price
                symbol_notionals[symbol] = abs(notional)
                gross += abs(notional) / equity_usd
                net += notional / equity_usd

        risk_budget = 0.0
        if risk_policy.max_gross_exposure_frac > 0.0:
            risk_budget = min(1.0, gross / risk_policy.max_gross_exposure_frac)

        return PortfolioSnapshot(
            timestamp=datetime.now(timezone.utc),
            equity_usd=equity_usd,
            open_positions=open_positions,
            symbol_notional_usd=symbol_notionals,
            risk=RiskSnapshot(
                gross_exposure_frac=float(gross),
                net_exposure_frac=float(net),
                max_drawdown_frac=0.0,
                risk_budget_used_frac=float(risk_budget),
            ),
        )
