"""Standalone entry point for the v2 execution engine container.

FIX-2: Commands arrive via RedisStreamCommandBus (XREADGROUP) for at-least-once
        guaranteed delivery.  The fire-and-forget Pub/Sub cmd:exec channel is retired.
FIX-3: WAL is replayed on boot to reconstruct all active sessions before the
        command bus is opened, so a container OOM-kill loses no quant state.
FIX-4: heartbeat_stale watchdog alerts now engage the kill-switch to pause new
        order entries when the market data feed goes dark.

Usage:
    python -m quant_v2.execution.main
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
from time import perf_counter
from datetime import datetime, timezone
from typing import Callable

from quant_v2.execution.redis_bus import (
    BusMessage,
    RedisCommandBus,
    RedisStreamCommandBus,
)
from quant_v2.execution.adapters import ExecutionResult
from quant_v2.execution.outcomes import ExecutionOutcome
from quant_v2.execution.service import (
    ExecutionStageTelemetry,
    RoutedExecutionService,
    SessionRequest,
)
from quant_v2.execution.state_wal import LifecycleStateRecord, RedisWAL, WALEntry
from quant_v2.execution.watchdog import LifecycleWatchdog, WatchdogAlert
from quant_v2.monitoring.kill_switch import MonitoringSnapshot
from quant_v2.monitoring.health_dashboard import build_runtime_resource_health
from quant_v2.monitoring.runtime_probes import record_runtime_boot_marker
from quant_v2.portfolio.risk_policy import PortfolioRiskPolicy

logger = logging.getLogger(__name__)


class ExecutionEngineServer:
    """Standalone execution engine that communicates via Redis.

    Command ingestion uses Redis Streams (at-least-once, survives restarts).
    Event publishing (replies to Telegram) uses Pub/Sub (ephemeral, best-effort).
    """

    DEFAULT_STALE_HEARTBEAT_SECONDS = 120.0

    def __init__(
        self,
        redis_url: str,
        *,
        risk_policy: PortfolioRiskPolicy | None = None,
        stage_telemetry_callback: Callable[[ExecutionStageTelemetry], None] | None = None,
    ) -> None:
        self._redis_url = redis_url
        self._stage_telemetry_callback = stage_telemetry_callback

        # FIX-2: Guaranteed command bus replaces fire-and-forget Pub/Sub for incoming cmds.
        self._stream_bus = RedisStreamCommandBus(redis_url)

        # Keep the PubSub bus solely for Telegram event replies (non-safety-critical).
        self._event_bus = RedisCommandBus(redis_url)

        # FIX-3: WAL connected at boot; replayed before accepting commands.
        self._wal = RedisWAL(redis_url)

        self._stale_heartbeat_seconds = self._resolve_stale_heartbeat_seconds()

        self._watchdog = LifecycleWatchdog(
            check_interval_seconds=5.0,
            on_alert=self._handle_watchdog_alert,
            stale_heartbeat_seconds=self._stale_heartbeat_seconds,
        )

        self._service = RoutedExecutionService(
            risk_policy=risk_policy,
            allow_live_execution=True,
            stage_telemetry_callback=stage_telemetry_callback,
        )

        # Session-state mutex: serialises concurrent mutations from the
        # command handler, watchdog alerts, and reconciliation loop to
        # prevent double-execution / WAL corruption race conditions.
        self._session_lock = asyncio.Lock()

        self._shutting_down = False
        self._reconciliation_task: asyncio.Task | None = None
        self._last_reconciliation_started_at: datetime | None = None
        self._last_reconciliation_completed_at: datetime | None = None
        self._last_reconciliation_error_at: datetime | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Initialize connections, replay WAL, then start processing."""
        try:
            record_runtime_boot_marker()
        except Exception as exc:
            logger.warning("Failed recording runtime boot marker: %s", exc)
        # 1. Connect infrastructure
        await self._stream_bus.connect()
        await self._event_bus.connect()
        await self._wal.connect()

        # FIX-3: Replay WAL to rebuild all active sessions BEFORE opening the
        # command bus. This means a crash-recovered container is fully warm
        # and honours lifecycle rules / risk limits immediately.
        entries = await self._wal.replay("0-0")
        rebuilt = await self._rebuild_state_from_wal(entries)
        logger.info(
            "WAL replay complete: %d entries processed, %d sessions rebuilt",
            len(entries),
            rebuilt,
        )

        # 2. Start watchdog
        await self._watchdog.start()

        # 3. Start consuming commands from the durable stream
        await self._stream_bus.start_consuming(self._handle_command)

        # 4. Start background ledger reconciliation for live sessions
        self._reconciliation_task = asyncio.create_task(
            self._reconciliation_loop(interval_seconds=30),
            name="ledger_reconciliation",
        )

        logger.info("Execution engine server started")

    async def stop(self) -> None:
        """Gracefully shut down all components."""
        self._shutting_down = True
        if self._reconciliation_task and not self._reconciliation_task.done():
            self._reconciliation_task.cancel()
            try:
                await self._reconciliation_task
            except asyncio.CancelledError:
                pass
        await self._watchdog.stop()
        await self._stream_bus.disconnect()
        await self._event_bus.disconnect()
        await self._wal.disconnect()
        logger.info("Execution engine server stopped")

    @classmethod
    def _resolve_stale_heartbeat_seconds(cls) -> float:
        """Resolve stale-feed threshold from env with safe defaults.

        BOT_V2_STALE_HEARTBEAT_MS takes precedence over
        BOT_V2_STALE_HEARTBEAT_SECONDS.
        """

        raw_ms = os.getenv("BOT_V2_STALE_HEARTBEAT_MS", "").strip()
        if raw_ms:
            try:
                parsed_ms = float(raw_ms)
            except ValueError:
                parsed_ms = cls.DEFAULT_STALE_HEARTBEAT_SECONDS * 1000.0
            return max(parsed_ms / 1000.0, 1.0)

        raw_seconds = (
            os.getenv("BOT_V2_STALE_HEARTBEAT_SECONDS", str(cls.DEFAULT_STALE_HEARTBEAT_SECONDS)).strip()
            or str(cls.DEFAULT_STALE_HEARTBEAT_SECONDS)
        )
        try:
            parsed_seconds = float(raw_seconds)
        except ValueError:
            parsed_seconds = cls.DEFAULT_STALE_HEARTBEAT_SECONDS
        return max(parsed_seconds, 1.0)

    @staticmethod
    def _bounded_rate(value: object) -> float:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return 0.0
        if parsed < 0.0:
            return 0.0
        if parsed > 1.0:
            return 1.0
        return parsed

    @classmethod
    def _parse_monitoring_snapshot(cls, payload: dict) -> MonitoringSnapshot | None:
        raw = payload.get("monitoring_snapshot")
        if raw is None:
            return None
        if isinstance(raw, MonitoringSnapshot):
            return raw
        if not isinstance(raw, dict):
            return None

        return MonitoringSnapshot(
            feature_drift_alert=bool(raw.get("feature_drift_alert", False)),
            confidence_collapse_alert=bool(raw.get("confidence_collapse_alert", False)),
            execution_anomaly_rate=cls._bounded_rate(raw.get("execution_anomaly_rate", 0.0)),
            connectivity_error_rate=cls._bounded_rate(raw.get("connectivity_error_rate", 0.0)),
            hard_risk_breach=bool(raw.get("hard_risk_breach", False)),
        )

    async def _persist_lifecycle_state(
        self,
        user_id: int,
        *,
        previous_state: LifecycleStateRecord | None,
    ) -> LifecycleStateRecord | None:
        """Persist the current lifecycle state when the service has advanced."""

        current_state = None
        get_lifecycle_state = getattr(self._service, "get_lifecycle_state", None)
        if callable(get_lifecycle_state):
            current_state = get_lifecycle_state(user_id)
        if current_state is None or current_state == previous_state:
            return current_state
        await self._wal.log_lifecycle_transition(user_id, record=current_state)
        return current_state

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
        callback = getattr(self, "_stage_telemetry_callback", None)
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

    async def _execute_stale_feed_circuit_breaker(
        self,
        user_id: int,
        *,
        reason: str,
    ) -> dict:
        """Cancel resting orders and flatten all exposure for a session."""

        get_adapter = getattr(self._service, "get_session_adapter", None)
        adapter = get_adapter(user_id) if callable(get_adapter) else None
        if adapter is None:
            return {
                "user_id": user_id,
                "reason": reason,
                "flattened": False,
                "position_symbols": [],
                "canceled_symbols": [],
                "error": "no_session_adapter",
            }

        get_positions = getattr(adapter, "get_positions", None)
        positions: dict[str, float] = {}
        if callable(get_positions):
            try:
                raw_positions = await asyncio.to_thread(get_positions)
                if isinstance(raw_positions, dict):
                    for symbol, qty in raw_positions.items():
                        clean_symbol = str(symbol).strip().upper()
                        clean_qty = float(qty)
                        if clean_symbol and abs(clean_qty) > 1e-12:
                            positions[clean_symbol] = clean_qty
            except Exception as exc:
                logger.error(
                    "Failed to read positions during circuit breaker for user %s: %s",
                    user_id,
                    exc,
                )

        open_order_symbols: set[str] = set()
        get_open_orders = getattr(adapter, "get_open_orders", None)
        if callable(get_open_orders):
            try:
                raw_orders = await asyncio.to_thread(get_open_orders, None)
            except TypeError:
                raw_orders = await asyncio.to_thread(get_open_orders)
            except Exception as exc:
                logger.warning(
                    "Failed to read open orders during circuit breaker for user %s: %s",
                    user_id,
                    exc,
                )
                raw_orders = []

            if isinstance(raw_orders, list):
                for order in raw_orders:
                    if not isinstance(order, dict):
                        continue
                    clean_symbol = str(order.get("symbol", "")).strip().upper()
                    if clean_symbol:
                        open_order_symbols.add(clean_symbol)

        symbols_to_cancel = sorted(set(positions) | open_order_symbols)
        canceled_symbols: list[str] = []
        cancel_all_orders = getattr(adapter, "cancel_all_orders", None)
        if callable(cancel_all_orders):
            for symbol in symbols_to_cancel:
                try:
                    await asyncio.to_thread(cancel_all_orders, symbol)
                    canceled_symbols.append(symbol)
                except Exception as exc:
                    logger.warning(
                        "Circuit breaker failed cancelling open orders for user %s symbol=%s: %s",
                        user_id,
                        symbol,
                        exc,
                    )

        flatten_error = ""
        flatten_results = ()
        sync_positions = getattr(self._service, "sync_positions", None)
        if positions and callable(sync_positions):
            get_last_prices = getattr(self._service, "get_last_prices", None)
            price_hints = get_last_prices(user_id) if callable(get_last_prices) else None
            normalized_prices = (
                {
                    str(symbol).strip().upper(): float(price)
                    for symbol, price in (price_hints or {}).items()
                    if str(symbol).strip() and float(price) > 0.0
                }
                if isinstance(price_hints, dict)
                else None
            )
            try:
                flatten_results = await sync_positions(
                    user_id,
                    target_positions={},
                    prices=normalized_prices or None,
                )
            except Exception as exc:
                flatten_error = f"sync_positions_failed:{exc.__class__.__name__}"
                logger.error(
                    "Circuit breaker flatten failed for user %s: %s",
                    user_id,
                    exc,
                )

        accepted = sum(1 for result in flatten_results if getattr(result, "accepted", False))
        rejected = sum(1 for result in flatten_results if not getattr(result, "accepted", False))

        set_lifecycle_state = getattr(self._service, "set_lifecycle_state", None)
        if callable(set_lifecycle_state):
            try:
                current_positions = {}
                if callable(get_positions):
                    refreshed_positions = await asyncio.to_thread(get_positions)
                    if isinstance(refreshed_positions, dict):
                        for symbol, qty in refreshed_positions.items():
                            clean_symbol = str(symbol).strip().upper()
                            clean_qty = float(qty)
                            if clean_symbol and abs(clean_qty) > 1e-12:
                                current_positions[clean_symbol] = clean_qty

                current_lifecycle = None
                get_lifecycle_state = getattr(self._service, "get_lifecycle_state", None)
                if callable(get_lifecycle_state):
                    current_lifecycle = get_lifecycle_state(user_id)

                if flatten_error:
                    next_state = "INCIDENT"
                    reason_text = f"stale_feed_circuit_breaker:{flatten_error}"
                elif current_positions:
                    next_state = "FLATTENING"
                    reason_text = f"stale_feed_circuit_breaker:{reason}"
                else:
                    next_state = "FLAT_CONFIRMED"
                    reason_text = f"stale_feed_circuit_breaker:{reason}"

                try:
                    set_lifecycle_state(
                        user_id,
                        state=next_state,
                        owner="liquidation_supervisor",
                        reason=reason_text,
                        policy_version=(
                            current_lifecycle.policy_version
                            if current_lifecycle is not None
                            else ""
                        ),
                    )
                except ValueError:
                    pass
            except Exception as exc:
                logger.debug(
                    "Lifecycle state update during stale-feed flatten failed for user %s: %s",
                    user_id,
                    exc,
                )

        return {
            "user_id": user_id,
            "reason": reason,
            "flattened": bool(not positions or accepted > 0),
            "position_symbols": sorted(positions),
            "canceled_symbols": canceled_symbols,
            "flatten_orders_accepted": accepted,
            "flatten_orders_rejected": rejected,
            "error": flatten_error,
        }

    async def _circuit_breaker_flatten(
        self,
        user_id: int,
        *,
        reason: str,
    ) -> dict:
        """Compatibility alias for the shutdown flatten path."""

        return await self._execute_stale_feed_circuit_breaker(user_id, reason=reason)

    # ------------------------------------------------------------------
    # FIX-3: WAL Replay  --  reconstruct session state on boot
    # ------------------------------------------------------------------

    async def _rebuild_state_from_wal(self, entries: list[WALEntry]) -> int:
        """Replay WAL entries to rebuild the in-memory session map.

        Pass 1: replay session_started / session_stopped to determine active sessions.
        Also track the latest state_checkpoint per user for equity/position restore.
        Pass 2: apply the latest checkpoint to each rebuilt session so that paper
        balance and positions survive container restarts.
        """
        rebuilt = 0
        latest_checkpoints: dict[int, dict] = {}
        pending_lifecycle: dict[int, list[LifecycleStateRecord]] = {}

        for entry in entries:
            if entry.event_type == "session_started":
                user_id = entry.user_id
                payload = entry.payload
                req = SessionRequest(
                    user_id=user_id,
                    live=bool(payload.get("live", False)),
                    strategy_profile=payload.get("strategy_profile", "core_v2"),
                )
                started = await self._service.start_session(req)
                if started:
                    lifecycle_payload = payload.get("lifecycle_state")
                    if isinstance(lifecycle_payload, dict):
                        record = LifecycleStateRecord.from_payload(lifecycle_payload)
                        restore_lifecycle_state = getattr(self._service, "restore_lifecycle_state", None)
                        if callable(restore_lifecycle_state):
                            restore_lifecycle_state(user_id, record)
                    for record in pending_lifecycle.pop(user_id, []):
                        restore_lifecycle_state = getattr(self._service, "restore_lifecycle_state", None)
                        if callable(restore_lifecycle_state):
                            restore_lifecycle_state(user_id, record)
                    self._watchdog.register_session(
                        user_id,
                        is_live=req.live,
                        initial_equity_usd=10_000.0,
                    )
                    rebuilt += 1
                    logger.info(
                        "WAL rebuild: restored session user=%d (live=%s, profile=%s)",
                        user_id,
                        req.live,
                        req.strategy_profile,
                    )

            elif entry.event_type == "session_stopped":
                user_id = entry.user_id
                stopped = await self._service.stop_session(user_id)
                if stopped:
                    self._watchdog.deregister_session(user_id)
                    rebuilt = max(0, rebuilt - 1)
                latest_checkpoints.pop(user_id, None)

            elif entry.event_type == "lifecycle_changed":
                record = LifecycleStateRecord.from_payload(entry.payload)
                restore_lifecycle_state = getattr(self._service, "restore_lifecycle_state", None)
                if callable(restore_lifecycle_state):
                    if self._service.is_running(entry.user_id) or self._service.get_lifecycle_state(entry.user_id) is not None:
                        try:
                            restore_lifecycle_state(entry.user_id, record)
                        except KeyError:
                            pending_lifecycle.setdefault(entry.user_id, []).append(record)
                    else:
                        pending_lifecycle.setdefault(entry.user_id, []).append(record)

            elif entry.event_type == "order_executed":
                payload = entry.payload
                outcome_raw = str(payload.get("outcome", "") or "").strip()
                try:
                    outcome = ExecutionOutcome(outcome_raw) if outcome_raw else ExecutionOutcome.NEW_FILL
                except ValueError:
                    outcome = ExecutionOutcome.UNKNOWN_REQUIRES_RECONCILIATION
                try:
                    result = ExecutionResult(
                        accepted=True,
                        order_id=str(payload.get("venue_order_id") or payload.get("order_id") or ""),
                        idempotency_key=str(payload.get("request_id") or payload.get("idempotency_key") or ""),
                        symbol=str(payload.get("symbol", "")),
                        side=str(payload.get("side", "BUY")),
                        requested_qty=float(payload.get("quantity", 0.0) or 0.0),
                        filled_qty=float(payload.get("quantity", 0.0) or 0.0),
                        avg_price=float(payload.get("avg_price", 0.0) or 0.0),
                        status=str(payload.get("status", "filled")),
                        created_at=str(payload.get("created_at") or entry.timestamp or ""),
                        reason="replayed_from_wal",
                        risk_policy_version=str(payload.get("risk_policy_version", "") or ""),
                        request_id=str(payload.get("request_id") or payload.get("idempotency_key") or ""),
                        venue_order_id=str(payload.get("venue_order_id") or payload.get("order_id") or ""),
                        fill_id=str(payload.get("fill_id") or payload.get("venue_order_id") or payload.get("order_id") or ""),
                        accounting_transaction_id=str(payload.get("accounting_transaction_id") or ""),
                        original_order_id=str(payload.get("original_order_id") or payload.get("venue_order_id") or payload.get("order_id") or ""),
                        original_fill_id=str(payload.get("original_fill_id") or payload.get("fill_id") or payload.get("venue_order_id") or payload.get("order_id") or ""),
                        outcome=outcome,
                        newly_filled_qty=float(payload.get("newly_filled_qty", payload.get("quantity", 0.0)) or 0.0),
                        replayed_at=str(payload.get("replayed_at") or ""),
                    )
                except Exception as exc:
                    logger.warning("WAL replay: failed restoring order result for user %s: %s", entry.user_id, exc)
                else:
                    restore_order_result = getattr(self._service, "restore_order_result", None)
                    if callable(restore_order_result):
                        restore_order_result(entry.user_id, result)

            elif entry.event_type == "state_checkpoint":
                latest_checkpoints[entry.user_id] = entry.payload

        # Pass 2: restore paper state from latest checkpoint per active session
        for user_id, cp in latest_checkpoints.items():
            if not self._service.is_running(user_id):
                continue
            try:
                await self._service.restore_paper_state(
                    user_id,
                    equity_baseline_usd=float(cp.get("equity_baseline_usd", 10_000.0)),
                    open_positions={k: float(v) for k, v in cp.get("open_positions", {}).items()},
                    paper_entry_prices={k: float(v) for k, v in cp.get("paper_entry_prices", {}).items()},
                )
                equity = float(cp.get("equity_baseline_usd", 10_000.0))
                self._watchdog.update_mtm_equity(user_id, equity)
                logger.info(
                    "WAL rebuild: restored paper state user=%d equity=$%.2f positions=%d",
                    user_id,
                    equity,
                    len(cp.get("open_positions", {})),
                )
            except Exception as exc:
                logger.warning("WAL rebuild: failed restoring state for user %d: %s", user_id, exc)

        return rebuilt

    # ------------------------------------------------------------------
    # Background Ledger Reconciliation
    # ------------------------------------------------------------------

    async def _reconciliation_loop(self, interval_seconds: int = 30) -> None:
        """Continuously verify exchange positions match local state.

        Detects phantom positions (exchange-side but not tracked locally)
        and ghost positions (tracked locally but gone from exchange).
        On drift, triggers sync_positions to re-align.
        """
        logger.info("Reconciliation loop started (interval=%ds)", interval_seconds)
        try:
            while not self._shutting_down:
                await asyncio.sleep(interval_seconds)
                self._last_reconciliation_started_at = datetime.now(timezone.utc)
                live_ids = self._service.get_live_session_ids()
                for user_id in live_ids:
                    try:
                        await self._reconcile_session(user_id)
                    except Exception as exc:
                        logger.error(
                            "Reconciliation failed for user %d: %s", user_id, exc
                        )
                self._last_reconciliation_completed_at = datetime.now(timezone.utc)
        except asyncio.CancelledError:
            logger.debug("Reconciliation loop cancelled")
        except Exception:
            self._last_reconciliation_error_at = datetime.now(timezone.utc)
            logger.exception("Reconciliation loop crashed")

    async def get_operational_health(self) -> dict:
        """Return live execution backlog and reconciliation freshness."""

        captured_at = datetime.now(timezone.utc)

        stream_health: dict = {}
        try:
            stream_health = await self._stream_bus.get_queue_health()
        except Exception as exc:
            stream_health = {
                "status": "unknown",
                "error": str(exc),
            }

        wal_health: dict = {}
        try:
            wal_health = await self._wal.get_stream_health()
        except Exception as exc:
            wal_health = {
                "status": "unknown",
                "error": str(exc),
            }

        runtime_health = build_runtime_resource_health()

        redis_memory: dict = {}
        redis_clients = (
            getattr(self._stream_bus, "_redis", None),
            getattr(self._wal, "_redis", None),
        )
        for redis_client in redis_clients:
            info = getattr(redis_client, "info", None)
            if not callable(info):
                continue
            try:
                raw_info = await info("memory")
            except Exception:
                continue
            if isinstance(raw_info, dict):
                redis_memory = raw_info
                break

        redis_used_memory_bytes = 0
        redis_maxmemory_bytes = 0
        try:
            redis_used_memory_bytes = int(redis_memory.get("used_memory", 0) or 0)
        except (TypeError, ValueError):
            redis_used_memory_bytes = 0
        try:
            redis_maxmemory_bytes = int(redis_memory.get("maxmemory", 0) or 0)
        except (TypeError, ValueError):
            redis_maxmemory_bytes = 0

        redis_memory_used_mb = max(redis_used_memory_bytes / (1024.0 * 1024.0), 0.0) if redis_used_memory_bytes else 0.0
        redis_memory_max_mb = max(redis_maxmemory_bytes / (1024.0 * 1024.0), 0.0) if redis_maxmemory_bytes else 0.0
        redis_memory_fraction = (
            (redis_used_memory_bytes / redis_maxmemory_bytes)
            if redis_used_memory_bytes > 0 and redis_maxmemory_bytes > 0
            else None
        )
        redis_memory_status = "unknown"
        if redis_used_memory_bytes > 0:
            redis_memory_status = "healthy"
            if redis_memory_fraction is not None and redis_memory_fraction >= 0.90:
                redis_memory_status = "degraded"
            elif redis_memory_fraction is not None and redis_memory_fraction >= 0.75:
                redis_memory_status = "warning"

        live_session_ids = ()
        get_live_session_ids = getattr(self._service, "get_live_session_ids", None)
        if callable(get_live_session_ids):
            try:
                live_session_ids = tuple(get_live_session_ids())
            except Exception:
                live_session_ids = ()

        recon_reference = self._last_reconciliation_completed_at or self._last_reconciliation_started_at
        reconciliation_lag_seconds = None
        if recon_reference is not None:
            reconciliation_lag_seconds = max(
                (captured_at - recon_reference).total_seconds(),
                0.0,
            )

        reconciliation_status = "unknown"
        if reconciliation_lag_seconds is not None:
            reconciliation_status = "healthy"
            if reconciliation_lag_seconds >= 90.0:
                reconciliation_status = "warning"
            if reconciliation_lag_seconds >= 180.0:
                reconciliation_status = "degraded"
        if self._last_reconciliation_error_at is not None:
            error_age = max((captured_at - self._last_reconciliation_error_at).total_seconds(), 0.0)
            if error_age <= 300.0:
                reconciliation_status = "degraded"

        statuses = [
            stream_health.get("status"),
            wal_health.get("status"),
            redis_memory_status,
            runtime_health.get("status"),
            reconciliation_status,
        ]
        overall_status = "unknown"
        if any(status == "degraded" for status in statuses):
            overall_status = "degraded"
        elif any(status == "warning" for status in statuses):
            overall_status = "warning"
        elif any(status == "healthy" for status in statuses):
            overall_status = "healthy"

        return {
            "status": overall_status,
            "captured_at": captured_at.isoformat(),
            "live_session_count": len(live_session_ids),
            "stream_bus": stream_health,
            "wal": wal_health,
            "redis_memory": {
                "status": redis_memory_status,
                "used_memory_mb": redis_memory_used_mb,
                "maxmemory_mb": redis_memory_max_mb,
                "used_memory_fraction": redis_memory_fraction,
            },
            "runtime_health": runtime_health,
            "reconciliation": {
                "status": reconciliation_status,
                "lag_seconds": reconciliation_lag_seconds,
                "last_started_at": (
                    self._last_reconciliation_started_at.isoformat()
                    if self._last_reconciliation_started_at is not None
                    else ""
                ),
                "last_completed_at": (
                    self._last_reconciliation_completed_at.isoformat()
                    if self._last_reconciliation_completed_at is not None
                    else ""
                ),
                "last_error_at": (
                    self._last_reconciliation_error_at.isoformat()
                    if self._last_reconciliation_error_at is not None
                    else ""
                ),
            },
        }

    async def _reconcile_session(self, user_id: int) -> None:
        """Compare exchange vs local positions for a single session."""
        async with self._session_lock:
            adapter = self._service.get_session_adapter(user_id)
            if adapter is None:
                return

            get_positions = getattr(adapter, "get_positions", None)
            if not callable(get_positions):
                return

            exchange_positions: dict[str, float] = await asyncio.to_thread(get_positions)

            snap = self._service.get_portfolio_snapshot(user_id)
            local_positions: dict[str, float] = (
                dict(snap.open_positions) if snap and snap.open_positions else {}
            )

            exchange_syms = {s for s, q in exchange_positions.items() if q != 0.0}
            local_syms = {s for s, q in local_positions.items() if q != 0.0}

            phantom = exchange_syms - local_syms  # On exchange, not tracked locally
            ghost = local_syms - exchange_syms  # Tracked locally, gone from exchange

            if phantom or ghost:
                logger.critical(
                    "STATE DRIFT user=%d: phantom=%s ghost=%s — triggering sync",
                    user_id,
                    phantom or "none",
                    ghost or "none",
                )
                await self._wal.append(
                    WALEntry(
                        event_type="reconciliation_drift",
                        user_id=user_id,
                        payload={
                            "phantom": sorted(phantom),
                            "ghost": sorted(ghost),
                        },
                    )
                )

                # FIX-5: Broadcast drift to Telegram so the human admin is alerted
                try:
                    await self._event_bus.send_event(
                        action="reconciliation_drift_alert",
                        payload={
                            "user_id": user_id,
                            "phantom": sorted(phantom),
                            "ghost": sorted(ghost),
                        },
                    )
                except Exception as alert_exc:
                    logger.error(
                        "Failed to send drift alert for user %d: %s", user_id, alert_exc
                    )

                # Re-align local state to exchange state
                prices_snap = {}
                if snap:
                    prices_snap = {
                        s: float(snap.open_positions.get(s, 0.0))
                        for s in exchange_syms | local_syms
                    }
                try:
                    await self._service.sync_positions(
                        user_id,
                        target_positions=exchange_positions,
                        prices=prices_snap or None,
                    )
                except Exception as sync_exc:
                    logger.error(
                        "sync_positions failed for user %d after drift: %s",
                        user_id, sync_exc,
                    )
                    try:
                        await self._event_bus.send_event(
                            action="reconciliation_sync_failed",
                            payload={
                                "user_id": user_id,
                                "error": str(sync_exc),
                            },
                        )
                    except Exception:
                        pass
            else:
                logger.debug("Reconciliation OK for user %d", user_id)


    async def _handle_command(self, msg: BusMessage) -> None:
        """Dispatch a command from the Telegram container."""
        action = msg.action
        payload = msg.payload
        correlation_id = msg.correlation_id

        async with self._session_lock:
            try:
                if action == "start_session":
                    result = await self._cmd_start_session(payload)
                elif action == "stop_session":
                    result = await self._cmd_stop_session(payload)
                elif action == "route_signals":
                    result = await self._cmd_route_signals(payload, correlation_id=correlation_id)
                elif action == "get_snapshot":
                    result = await self._cmd_get_snapshot(payload)
                elif action == "set_lifecycle":
                    result = await self._cmd_set_lifecycle(payload)
                elif action == "ping":
                    result = {"status": "pong"}
                else:
                    result = {"error": f"unknown_action: {action}"}

                await self._event_bus.send_event(
                    action=f"{action}_result",
                    payload=result,
                    correlation_id=correlation_id,
                )

            except Exception as e:
                logger.exception("Command handler error for action=%s", action)
                await self._event_bus.send_event(
                    action=f"{action}_error",
                    payload={"error": str(e)},
                    correlation_id=correlation_id,
                )

    async def _cmd_start_session(self, payload: dict) -> dict:
        user_id = int(payload["user_id"])
        live = bool(payload.get("live", False))
        req = SessionRequest(
            user_id=user_id,
            live=live,
            strategy_profile=payload.get("strategy_profile", "core_v2"),
            credentials=payload.get("credentials", {}),
        )
        success = await self._service.start_session(req)

        if success:
            # OpSec: WAL must NEVER contain API keys.
            lifecycle_state = None
            get_lifecycle_state = getattr(self._service, "get_lifecycle_state", None)
            if callable(get_lifecycle_state):
                lifecycle_state = get_lifecycle_state(user_id)
            await self._wal.log_session_started(
                user_id,
                live=live,
                strategy_profile=req.strategy_profile,
                lifecycle_state=lifecycle_state,
            )
            self._watchdog.register_session(
                user_id, is_live=live, initial_equity_usd=10_000.0
            )

            # Zero out credential strings in-place after use (best-effort memory hygiene).
            creds = req.credentials
            for key in list(creds.keys()):
                creds[key] = ""

        return {"success": success, "user_id": user_id}

    async def _cmd_stop_session(self, payload: dict) -> dict:
        user_id = int(payload["user_id"])
        previous_lifecycle = None
        get_lifecycle_state = getattr(self._service, "get_lifecycle_state", None)
        if callable(get_lifecycle_state):
            previous_lifecycle = get_lifecycle_state(user_id)
        success = await self._service.stop_session(user_id)

        if success:
            await self._persist_lifecycle_state(
                user_id,
                previous_state=previous_lifecycle,
            )
            await self._wal.log_session_stopped(user_id)
            self._watchdog.deregister_session(user_id)

        return {"success": success, "user_id": user_id}

    async def _cmd_route_signals(self, payload: dict, *, correlation_id: str = "") -> dict:
        from quant_v2.contracts import StrategySignal

        command_started = perf_counter()
        user_id = int(payload["user_id"])
        prices = payload.get("prices", {})
        normalized_prices: dict[str, float] = {}
        if isinstance(prices, dict):
            for symbol, raw_price in prices.items():
                clean_symbol = str(symbol).strip().upper()
                if not clean_symbol:
                    continue
                try:
                    clean_price = float(raw_price)
                except (TypeError, ValueError):
                    continue
                if clean_price > 0.0:
                    normalized_prices[clean_symbol] = clean_price
        raw_signals = payload.get("signals", [])
        monitoring_snapshot = self._parse_monitoring_snapshot(payload)

        signals = tuple(
            StrategySignal(
                symbol=s["symbol"],
                timeframe=s.get("timeframe", "1h"),
                horizon_bars=int(s.get("horizon_bars", 4)),
                signal=s["signal"],
                confidence=float(s.get("confidence", 0.5)),
                uncertainty=s.get("uncertainty"),
            )
            for s in raw_signals
        )

        # Fresh non-zero prices imply a successful market-data pull.
        if normalized_prices:
            self._watchdog.record_tick(user_id)
        self._emit_stage_telemetry(
            user_id=user_id,
            stage="command_parse",
            started_at=command_started,
            correlation_id=correlation_id,
            detail=f"signals={len(signals)} prices={len(normalized_prices)}",
        )

        previous_lifecycle = None
        get_lifecycle_state = getattr(self._service, "get_lifecycle_state", None)
        if callable(get_lifecycle_state):
            previous_lifecycle = get_lifecycle_state(user_id)
        routing_started = perf_counter()
        results = await self._service.route_signals(
            user_id,
            signals=signals,
            prices=normalized_prices,
            monitoring_snapshot=monitoring_snapshot,
            correlation_id=correlation_id,
        )
        self._emit_stage_telemetry(
            user_id=user_id,
            stage="routing_call",
            started_at=routing_started,
            correlation_id=correlation_id,
            status="ok",
            detail=f"results={len(results)}",
        )
        await self._persist_lifecycle_state(
            user_id,
            previous_state=previous_lifecycle,
        )

        # Log position updates to WAL
        ledger_started = perf_counter()
        for r in results:
            if r.accepted:
                await self._wal.log_order_executed(
                    user_id,
                    symbol=r.symbol,
                    side=r.side,
                    quantity=r.filled_qty,
                    avg_price=r.avg_price,
                    status=r.status,
                    risk_policy_version=r.risk_policy_version,
                    outcome=getattr(r.outcome, "value", str(r.outcome)),
                    newly_filled_qty=float(getattr(r, "newly_filled_qty", r.filled_qty) or 0.0),
                    request_id=str(getattr(r, "request_id", "") or ""),
                    venue_order_id=str(getattr(r, "venue_order_id", "") or ""),
                    fill_id=str(getattr(r, "fill_id", "") or ""),
                    accounting_transaction_id=str(getattr(r, "accounting_transaction_id", "") or ""),
                    original_order_id=str(getattr(r, "original_order_id", "") or ""),
                    original_fill_id=str(getattr(r, "original_fill_id", "") or ""),
                    replayed_at=str(getattr(r, "replayed_at", "") or ""),
                    correlation_id=correlation_id,
                )
        self._emit_stage_telemetry(
            user_id=user_id,
            stage="ledger_commit",
            started_at=ledger_started,
            correlation_id=correlation_id,
            detail=f"accepted={sum(1 for result in results if result.accepted)}",
        )

        # Update watchdog with latest equity
        refresh_started = perf_counter()
        snap = self._service.get_portfolio_snapshot(user_id)
        if snap:
            self._watchdog.update_mtm_equity(user_id, snap.equity_usd)
            self._watchdog.record_tick(user_id)

        # Persist state checkpoint so balance/positions survive restarts
        paper_state = self._service.get_paper_state(user_id)
        if paper_state is not None:
            try:
                await self._wal.log_state_checkpoint(
                    user_id,
                    equity_baseline_usd=paper_state["equity_baseline_usd"],
                    open_positions=paper_state["open_positions"],
                    paper_entry_prices=paper_state["paper_entry_prices"],
                )
            except Exception as exc:
                logger.warning("WAL state checkpoint failed for user %d: %s", user_id, exc)
        self._emit_stage_telemetry(
            user_id=user_id,
            stage="post_fill_refresh",
            started_at=refresh_started,
            correlation_id=correlation_id,
            detail=f"snapshot={bool(snap)} checkpoint={bool(paper_state is not None)}",
        )

        return {
            "user_id": user_id,
            "results": [
                {
                    "symbol": r.symbol,
                    "side": r.side,
                    "accepted": r.accepted,
                    "filled_qty": r.filled_qty,
                    "avg_price": r.avg_price,
                    "status": r.status,
                    "reason": r.reason,
                }
                for r in results
            ],
        }

    async def _cmd_get_snapshot(self, payload: dict) -> dict:
        user_id = int(payload["user_id"])
        snap = self._service.get_portfolio_snapshot(user_id)
        if snap is None:
            return {"user_id": user_id, "snapshot": None}

        return {
            "user_id": user_id,
            "snapshot": {
                "equity_usd": snap.equity_usd,
                "open_positions": dict(snap.open_positions),
                "symbol_count": snap.symbol_count,
                "timestamp": snap.timestamp,
            },
        }

    async def _cmd_set_lifecycle(self, payload: dict) -> dict:
        user_id = int(payload["user_id"])
        horizon_hours = payload.get("horizon_hours")
        stop_loss_pct = payload.get("stop_loss_pct")

        if horizon_hours is not None:
            from datetime import timedelta
            from datetime import datetime, timezone as tz

            deadline = datetime.now(tz.utc) + timedelta(hours=float(horizon_hours))
            self._watchdog.update_horizon(user_id, deadline)

        if stop_loss_pct is not None:
            snap = self._service.get_portfolio_snapshot(user_id)
            if snap:
                stop_equity = snap.equity_usd * (1.0 - float(stop_loss_pct) / 100.0)
                self._watchdog.update_stop_loss(user_id, stop_equity)

        return {"user_id": user_id, "status": "lifecycle_updated"}

    # ------------------------------------------------------------------
    # FIX-4: Watchdog alert handler — heartbeat_stale now engages kill-switch
    # ------------------------------------------------------------------

    async def _handle_watchdog_alert(self, alert: WatchdogAlert) -> None:
        """Handle alerts from the lifecycle watchdog.

        FIX-4: heartbeat_stale (market data feed dropped) now engages the
        kill-switch and emergency flatten path to prevent trading on stale
        prices during a data outage.

        P0: stale-feed alerts now trigger a hard circuit breaker that cancels
        resting orders and flattens all open exposure.
        """
        async with self._session_lock:
            logger.warning(
                "Watchdog alert for user=%d: %s - %s",
                alert.user_id,
                alert.alert_type,
                alert.reason,
            )

            try:
                await self._event_bus.send_event(
                    action="watchdog_alert",
                    payload={
                        "user_id": alert.user_id,
                        "alert_type": alert.alert_type,
                        "reason": alert.reason,
                        "triggered_at": alert.triggered_at,
                    },
                )
            except Exception as exc:
                logger.warning(
                    "Failed to publish watchdog_alert event for user %s: %s",
                    alert.user_id,
                    exc,
                )

            previous_lifecycle = None
            get_lifecycle_state = getattr(self._service, "get_lifecycle_state", None)
            if callable(get_lifecycle_state):
                previous_lifecycle = get_lifecycle_state(alert.user_id)

            reasons: tuple[str, ...] = (alert.alert_type,)
            monitoring_snapshot: MonitoringSnapshot | None = None

            if alert.alert_type in ("stop_loss_triggered", "horizon_expired"):
                logger.info(
                    "Auto-flattening positions for user=%d due to %s",
                    alert.user_id,
                    alert.alert_type,
                )
                monitoring_snapshot = MonitoringSnapshot(
                    hard_risk_breach=True,
                )
            elif alert.alert_type == "heartbeat_stale":
                logger.warning(
                    "Market data feed stale for user=%d — triggering stale-feed circuit breaker.",
                    alert.user_id,
                )
                reasons = ("heartbeat_stale", "stale_feed_circuit_breaker")
                monitoring_snapshot = MonitoringSnapshot(
                    connectivity_error_rate=1.0,
                )

            await self._wal.log_kill_switch(
                alert.user_id,
                triggered=True,
                reasons=reasons,
            )

            if monitoring_snapshot is not None:
                set_snapshot = getattr(self._service, "set_monitoring_snapshot", None)
                if callable(set_snapshot):
                    try:
                        set_snapshot(alert.user_id, monitoring_snapshot)
                    except Exception as exc:
                        logger.warning(
                            "Failed applying kill-switch monitoring snapshot for user %s: %s",
                            alert.user_id,
                            exc,
                        )
                await self._persist_lifecycle_state(
                    alert.user_id,
                    previous_state=previous_lifecycle,
                )

            flatten_result = await self._execute_stale_feed_circuit_breaker(
                alert.user_id,
                reason=alert.alert_type,
            )
            await self._persist_lifecycle_state(
                alert.user_id,
                previous_state=previous_lifecycle,
            )
            try:
                await self._event_bus.send_event(
                    action="watchdog_flatten_result",
                    payload={
                        "user_id": alert.user_id,
                        "alert_type": alert.alert_type,
                        "reasons": list(reasons),
                        **flatten_result,
                    },
                )
            except Exception as exc:
                logger.warning(
                    "Failed to publish watchdog_flatten_result for user %s: %s",
                    alert.user_id,
                    exc,
                )


# ------------------------------------------------------------------
# Entrypoint
# ------------------------------------------------------------------

async def run_server() -> None:
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
    server = ExecutionEngineServer(redis_url)

    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()

    def _signal_handler():
        logger.info("Received shutdown signal")
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _signal_handler)
        except NotImplementedError:
            pass  # Windows doesn't support add_signal_handler

    await server.start()
    logger.info("Execution engine running. Press Ctrl+C to stop.")

    await stop_event.wait()

    # Attempt to flatten all live positions before shutting down
    try:
        live_ids = server._service.get_live_session_ids()
        if live_ids:
            logger.info("Shutdown: flattening %d live session(s)...", len(live_ids))
            for uid in live_ids:
                try:
                    result = await server._circuit_breaker_flatten(
                        uid, reason="graceful_shutdown",
                    )
                    logger.info("Shutdown flatten user=%s result=%s", uid, result)
                except Exception as flatten_exc:
                    logger.error("Shutdown flatten failed for user=%s: %s", uid, flatten_exc)
        else:
            logger.info("Shutdown: no live sessions to flatten")
    except Exception as exc:
        logger.error("Shutdown flatten sweep failed: %s", exc)

    await server.stop()
    logger.info("positions_flat")  # sentinel for flatten_on_shutdown.sh


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    asyncio.run(run_server())


if __name__ == "__main__":
    main()
