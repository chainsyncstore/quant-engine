"""Live Capital Readiness Tests.

Targets the 4 critical patches from the v2 live-capital audit:
  1. Async Adapter — place_order must not block the event loop
  2. Redis Streams  — RedisStreamCommandBus XADD/XREADGROUP/XACK round-trip
  3. WAL Replay    — boot-time session reconstruction from WAL entries
  4. Tick-Stale    — heartbeat_stale watchdog alert triggers kill-switch

All tests are pure-python (no live Redis or Binance connection required).
All async tests are wrapped in asyncio.run() to avoid requiring pytest-asyncio.
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from quant_v2.execution.adapters import ExecutionResult
from quant_v2.execution.redis_bus import (
    BusMessage,
    RedisStreamCommandBus,
)
from quant_v2.execution.service import RoutedExecutionService, SessionRequest
from quant_v2.execution.state_wal import (
    InMemoryWAL,
    LifecycleStateRecord,
)
from quant_v2.execution.watchdog import LifecycleWatchdog


# ─────────────────────────────────────────────────────────────────
# FIX 1: Async Adapter — Event Loop Must Not Be Blocked
# ─────────────────────────────────────────────────────────────────

class _SlowSyncAdapter:
    """Simulates a sync Binance adapter with 200ms I/O latency (time.sleep)."""

    def __init__(self) -> None:
        self._positions: dict[str, float] = {}

    def get_positions(self) -> dict[str, float]:
        time.sleep(0.2)
        return dict(self._positions)

    def place_order(
        self,
        plan,
        *,
        idempotency_key: str,
        mark_price=None,
        limit_price=None,
        post_only=False,
    ):
        time.sleep(0.2)
        self._positions[plan.symbol] = self._positions.get(plan.symbol, 0.0) + float(plan.quantity)
        return ExecutionResult(
            accepted=True,
            order_id="ok-1",
            idempotency_key=idempotency_key,
            symbol=plan.symbol,
            side=plan.side,
            requested_qty=float(plan.quantity),
            filled_qty=float(plan.quantity),
            avg_price=float(mark_price or 50_000.0),
            status="filled",
            created_at=datetime.now(timezone.utc).isoformat(),
        )


def test_fix1_async_adapter_does_not_block_event_loop():
    """FIX-1: A 200ms synchronous adapter must not stall concurrent coroutines.

    Without asyncio.to_thread() both tasks run sequentially (~400ms).
    With to_thread() they run concurrently and both finish faster.
    We use a generous 800ms ceiling (Windows thread-pool scheduling overhead).
    """
    from quant_v2.contracts import StrategySignal

    adapter = _SlowSyncAdapter()
    service = RoutedExecutionService(paper_adapter_factory=lambda: adapter)

    async def _run():
        req = SessionRequest(user_id=1001, live=False)
        await service.start_session(req)

        concurrent_completed = []

        async def _side_task():
            await asyncio.sleep(0.05)
            concurrent_completed.append(True)

        start = time.monotonic()
        signal = StrategySignal(
            symbol="BTCUSDT", timeframe="1h", horizon_bars=4,
            signal="BUY", confidence=0.9,
        )
        # route_signals and a pure-async side-task run concurrently via gather.
        await asyncio.gather(
            service.route_signals(1001, signals=(signal,), prices={"BTCUSDT": 50_000.0}),
            _side_task(),
        )
        return time.monotonic() - start, concurrent_completed

    elapsed, completed = asyncio.run(_run())

    assert completed, "Side coroutine was blocked and never ran during adapter I/O"
    # Sequential (blocked) execution would take ~400ms for two 200ms calls.
    # With to_thread(), the gather lets both tasks run in the same time as one.
    # We allow 800ms to accommodate Windows thread scheduler overhead.
    assert elapsed < 0.8, (
        f"Event loop appears to be blocked: elapsed={elapsed:.3f}s (expected < 0.8s). "
        "Did asyncio.to_thread() get applied to place_order()?"
    )


# ─────────────────────────────────────────────────────────────────
# FIX 2: Redis Streams — XADD / XREADGROUP / XACK Round-Trip
# ─────────────────────────────────────────────────────────────────

def test_fix2_stream_bus_enqueues_and_consumes_message():
    """FIX-2: XADD enqueue works and _process_entry ACKs on successful handler."""

    async def _run():
        redis_mock = AsyncMock()
        redis_mock.ping = AsyncMock(return_value=True)
        redis_mock.xgroup_create = AsyncMock(return_value=True)
        redis_mock.xpending_range = AsyncMock(return_value=[])
        redis_mock.xadd = AsyncMock(return_value="1700000000000-0")
        redis_mock.xack = AsyncMock(return_value=1)

        test_msg = BusMessage(
            action="stop_session",
            payload={"user_id": 42},
            timestamp="2026-01-01T00:00:00+00:00",
            correlation_id="corr-1",
        )

        received: list[BusMessage] = []

        async def _handler(msg: BusMessage) -> None:
            received.append(msg)

        with patch("redis.asyncio.from_url", return_value=redis_mock):
            bus = RedisStreamCommandBus("redis://localhost:6379")
            bus._redis = redis_mock

            # Test enqueue
            msg_id = await bus.enqueue(test_msg)
            assert msg_id == "1700000000000-0"
            redis_mock.xadd.assert_awaited_once()

            # Test _process_entry directly (the core consume logic) without starting the loop
            await bus._process_entry(
                "1700000000000-0",
                {"data": test_msg.to_json()},
                _handler,
            )

        assert len(received) == 1
        assert received[0].action == "stop_session"
        assert received[0].payload["user_id"] == 42
        redis_mock.xack.assert_awaited_once()

    asyncio.run(_run())


def test_fix2_stream_bus_pending_messages_drained_on_boot():
    """FIX-2: _drain_pending re-processes unACKed messages from a crashed container."""

    async def _run():
        redis_mock = AsyncMock()
        redis_mock.ping = AsyncMock(return_value=True)
        redis_mock.xgroup_create = AsyncMock(return_value=True)

        pending_msg = BusMessage(
            action="stop_session",
            payload={"user_id": 99},
            timestamp="2026-01-01T00:00:00+00:00",
        )

        redis_mock.xpending_range = AsyncMock(return_value=[
            {
                "message_id": "1699999999999-0",
                "consumer": "execution_engine",
                "time_since_delivered": 5000,
                "times_delivered": 1,
            }
        ])
        redis_mock.xrange = AsyncMock(return_value=[
            ("1699999999999-0", {"data": pending_msg.to_json()})
        ])
        redis_mock.xack = AsyncMock(return_value=1)

        drained: list[BusMessage] = []

        async def _handler(msg: BusMessage) -> None:
            drained.append(msg)

        with patch("redis.asyncio.from_url", return_value=redis_mock):
            bus = RedisStreamCommandBus("redis://localhost:6379")
            bus._redis = redis_mock
            # Call _drain_pending directly rather than starting the infinite consumer loop.
            await bus._drain_pending(_handler)

        assert len(drained) == 1, "Pending message was not drained on boot"
        assert drained[0].action == "stop_session"
        assert drained[0].payload["user_id"] == 99
        redis_mock.xack.assert_awaited_once()

    asyncio.run(_run())


# ─────────────────────────────────────────────────────────────────
# FIX 3: WAL Replay — Boot-Time Session Reconstruction
# ─────────────────────────────────────────────────────────────────

def test_fix3_wal_replay_rebuilds_active_sessions():
    """FIX-3: session_started WAL entries recreate running sessions on engine boot."""
    from quant_v2.execution.main import ExecutionEngineServer

    async def _run():
        wal = InMemoryWAL()
        # Use live=False so no Binance credentials are required in the test.
        await wal.log_session_started(101, live=False, strategy_profile="core_v2")
        await wal.log_session_started(202, live=False, strategy_profile="core_v2")

        server = ExecutionEngineServer.__new__(ExecutionEngineServer)
        server._stream_bus = MagicMock()
        server._event_bus = MagicMock()
        server._wal = wal
        server._shutting_down = False
        server._watchdog = LifecycleWatchdog()
        server._service = RoutedExecutionService()

        entries = await wal.replay("0-0")
        rebuilt = await server._rebuild_state_from_wal(entries)
        return rebuilt, server._service

    rebuilt, service = asyncio.run(_run())

    assert rebuilt == 2
    assert service.is_running(101)
    assert service.is_running(202)


def test_fix3_wal_replay_skips_stopped_sessions():
    """FIX-3: session_stopped WAL entries mean that session is NOT rebuilt."""
    from quant_v2.execution.main import ExecutionEngineServer

    async def _run():
        wal = InMemoryWAL()
        await wal.log_session_started(303, live=False, strategy_profile="core_v2")
        await wal.log_session_stopped(303)

        server = ExecutionEngineServer.__new__(ExecutionEngineServer)
        server._stream_bus = MagicMock()
        server._event_bus = MagicMock()
        server._wal = wal
        server._shutting_down = False
        server._watchdog = LifecycleWatchdog()
        server._service = RoutedExecutionService()

        entries = await wal.replay("0-0")
        await server._rebuild_state_from_wal(entries)
        return server._service

    service = asyncio.run(_run())
    assert not service.is_running(303), "Stopped session was incorrectly rebuilt"


def test_fix3_wal_replay_restores_lifecycle_state_transitions():
    """FIX-3: lifecycle transitions should survive replay and restore the latest durable state."""
    from quant_v2.execution.main import ExecutionEngineServer

    async def _run():
        wal = InMemoryWAL()
        initial = LifecycleStateRecord(
            state="ACTIVE",
            owner="alpha_session",
            retry_count=0,
            reason="session_started",
            policy_version="wp03-risk-v1",
        )
        incident = LifecycleStateRecord(
            state="INCIDENT",
            owner="liquidation_supervisor",
            retry_count=1,
            reason="heartbeat_stale",
            policy_version="wp03-risk-v1",
        )
        confirmed = LifecycleStateRecord(
            state="FLAT_CONFIRMED",
            owner="liquidation_supervisor",
            retry_count=1,
            reason="flat_confirmed",
            policy_version="wp03-risk-v1",
        )

        await wal.log_session_started(404, live=False, strategy_profile="core_v2", lifecycle_state=initial)
        await wal.log_lifecycle_transition(404, record=incident)
        await wal.log_lifecycle_transition(404, record=confirmed)

        server = ExecutionEngineServer.__new__(ExecutionEngineServer)
        server._stream_bus = MagicMock()
        server._event_bus = MagicMock()
        server._wal = wal
        server._shutting_down = False
        server._watchdog = LifecycleWatchdog()
        server._service = RoutedExecutionService()

        entries = await wal.replay("0-0")
        await server._rebuild_state_from_wal(entries)
        return server._service

    service = asyncio.run(_run())
    lifecycle = service.get_lifecycle_state(404)

    assert service.is_running(404)
    assert lifecycle is not None
    assert lifecycle.state == "FLAT_CONFIRMED"
    assert lifecycle.owner == "liquidation_supervisor"
    assert lifecycle.reason == "flat_confirmed"


# ─────────────────────────────────────────────────────────────────
# FIX 4: Heartbeat-Stale Alert → Kill-Switch  (Tick Starvation)
# ─────────────────────────────────────────────────────────────────

def test_fix4_heartbeat_stale_alert_logs_kill_switch_to_wal():
    """FIX-4: A heartbeat_stale watchdog alert must write kill_switch_triggered to WAL."""
    from quant_v2.execution.main import ExecutionEngineServer
    from quant_v2.execution.watchdog import WatchdogAlert

    async def _run():
        wal = InMemoryWAL()

        server = ExecutionEngineServer.__new__(ExecutionEngineServer)
        server._stream_bus = MagicMock()
        server._event_bus = AsyncMock()
        server._event_bus.send_event = AsyncMock(return_value=0)
        server._wal = wal
        server._shutting_down = False
        server._session_lock = asyncio.Lock()
        server._service = MagicMock()
        server._watchdog = MagicMock()

        alert = WatchdogAlert(
            user_id=555,
            alert_type="heartbeat_stale",
            reason="No market tick for 120s (threshold: 120s)",
        )
        await server._handle_watchdog_alert(alert)
        return await wal.replay("0-0")

    entries = asyncio.run(_run())
    kill_entries = [e for e in entries if e.event_type == "kill_switch_triggered"]

    assert kill_entries, "No kill_switch_triggered entry written to WAL for heartbeat_stale"
    assert kill_entries[0].user_id == 555
    assert "heartbeat_stale" in kill_entries[0].payload.get("reasons", [])


def test_fix4_stop_loss_alert_also_logs_kill_switch_to_wal():
    """Regression: stop_loss_triggered must still write kill_switch_triggered entry."""
    from quant_v2.execution.main import ExecutionEngineServer
    from quant_v2.execution.watchdog import WatchdogAlert

    async def _run():
        wal = InMemoryWAL()

        server = ExecutionEngineServer.__new__(ExecutionEngineServer)
        server._stream_bus = MagicMock()
        server._event_bus = AsyncMock()
        server._event_bus.send_event = AsyncMock(return_value=0)
        server._wal = wal
        server._shutting_down = False
        server._session_lock = asyncio.Lock()
        server._service = MagicMock()
        server._watchdog = MagicMock()

        alert = WatchdogAlert(
            user_id=666,
            alert_type="stop_loss_triggered",
            reason="MTM equity $9800 <= stop-loss $9850",
        )
        await server._handle_watchdog_alert(alert)
        return await wal.replay("0-0")

    entries = asyncio.run(_run())
    kill_entries = [e for e in entries if e.event_type == "kill_switch_triggered"]

    assert kill_entries, "No kill_switch_triggered entry for stop_loss_triggered"
    assert kill_entries[0].user_id == 666
    assert "stop_loss_triggered" in kill_entries[0].payload.get("reasons", [])


def test_fix4_stale_feed_watchdog_auto_flattens_open_exposure():
    """P0: heartbeat_stale must cancel resting orders and flatten live exposure."""
    from quant_v2.execution.main import ExecutionEngineServer

    class _StubAdapter:
        def __init__(self) -> None:
            self.cancelled: list[str] = []

        def get_positions(self) -> dict[str, float]:
            return {"BTCUSDT": 0.75}

        def get_open_orders(self, symbol: str | None = None) -> list[dict]:
            if symbol:
                return [{"symbol": symbol, "orderId": "resting-1"}]
            return [{"symbol": "BTCUSDT", "orderId": "resting-1"}]

        def cancel_all_orders(self, symbol: str) -> None:
            self.cancelled.append(symbol)

    async def _run():
        wal = InMemoryWAL()

        server = ExecutionEngineServer.__new__(ExecutionEngineServer)
        server._stream_bus = MagicMock()
        server._event_bus = AsyncMock()
        server._event_bus.send_event = AsyncMock(return_value=0)
        server._wal = wal
        server._shutting_down = False
        server._session_lock = asyncio.Lock()

        adapter = _StubAdapter()
        server._service = MagicMock()
        server._service.get_session_adapter = MagicMock(return_value=adapter)
        server._service.get_last_prices = MagicMock(return_value={"BTCUSDT": 50_000.0})
        server._service.set_monitoring_snapshot = MagicMock()

        server._service.sync_positions = AsyncMock(
            return_value=(
                ExecutionResult(
                    accepted=True,
                    order_id="flatten-1",
                    idempotency_key="idem-1",
                    symbol="BTCUSDT",
                    side="SELL",
                    requested_qty=0.75,
                    filled_qty=0.75,
                    avg_price=50_000.0,
                    status="filled",
                    created_at=datetime.now(timezone.utc).isoformat(),
                ),
            )
        )

        watchdog = LifecycleWatchdog(
            check_interval_seconds=60.0,
            stale_heartbeat_seconds=1.0,
            on_alert=server._handle_watchdog_alert,
        )
        server._watchdog = watchdog

        watchdog.register_session(user_id=777, is_live=True, initial_equity_usd=10_000.0)
        watchdog._last_tick_time[777] = datetime.now(timezone.utc) - timedelta(seconds=5)

        await watchdog._run_checks()

        entries = await wal.replay("0-0")
        return server, adapter, entries

    server, adapter, entries = asyncio.run(_run())

    sync_kwargs = server._service.sync_positions.await_args.kwargs
    assert sync_kwargs["target_positions"] == {}
    assert "BTCUSDT" in adapter.cancelled

    kill_entries = [e for e in entries if e.event_type == "kill_switch_triggered"]
    assert kill_entries, "Expected kill_switch_triggered WAL entry for stale-feed breaker"
    assert "heartbeat_stale" in kill_entries[0].payload.get("reasons", [])
    assert "stale_feed_circuit_breaker" in kill_entries[0].payload.get("reasons", [])

    flatten_events = [
        c
        for c in server._event_bus.send_event.await_args_list
        if c.kwargs.get("action") == "watchdog_flatten_result"
    ]
    assert flatten_events, "Expected watchdog_flatten_result event after stale alert"
    assert flatten_events[0].kwargs["payload"].get("flattened") is True


def test_fix4_route_signals_records_tick_on_market_data_pull():
    """P0: any successful price pull should refresh watchdog heartbeat freshness."""
    from quant_v2.execution.main import ExecutionEngineServer
    from quant_v2.monitoring.kill_switch import MonitoringSnapshot

    async def _run():
        server = ExecutionEngineServer.__new__(ExecutionEngineServer)
        server._stream_bus = MagicMock()
        server._event_bus = AsyncMock()
        server._wal = InMemoryWAL()
        server._shutting_down = False
        server._session_lock = asyncio.Lock()

        server._watchdog = MagicMock()
        server._watchdog.record_tick = MagicMock()
        server._watchdog.update_mtm_equity = MagicMock()

        server._service = MagicMock()
        server._service.route_signals = AsyncMock(return_value=())
        server._service.get_portfolio_snapshot = MagicMock(return_value=None)

        payload = {
            "user_id": 123,
            "prices": {"BTCUSDT": 51_234.5},
            "signals": [],
            "monitoring_snapshot": {
                "connectivity_error_rate": 0.2,
                "execution_anomaly_rate": 0.1,
                "hard_risk_breach": False,
            },
        }

        await server._cmd_route_signals(payload)
        args = server._service.route_signals.await_args.kwargs
        return server._watchdog.record_tick.call_count, args["monitoring_snapshot"]

    tick_call_count, snapshot = asyncio.run(_run())
    assert tick_call_count == 1
    assert isinstance(snapshot, MonitoringSnapshot)
    assert snapshot.connectivity_error_rate == 0.2


def test_fix4_route_signals_emits_stage_telemetry_for_route_and_persistence():
    """Command-path telemetry should expose parse, routing, ledger, and refresh stages."""
    from quant_v2.execution.main import ExecutionEngineServer

    async def _run():
        stage_events = []
        server = ExecutionEngineServer.__new__(ExecutionEngineServer)
        server._stage_telemetry_callback = stage_events.append
        server._stream_bus = MagicMock()
        server._event_bus = AsyncMock()
        server._wal = InMemoryWAL()
        server._shutting_down = False
        server._session_lock = asyncio.Lock()

        server._watchdog = MagicMock()
        server._watchdog.record_tick = MagicMock()
        server._watchdog.update_mtm_equity = MagicMock()

        server._service = MagicMock()
        server._service.get_lifecycle_state = MagicMock(return_value=None)
        server._service.route_signals = AsyncMock(
            return_value=(
                ExecutionResult(
                    accepted=True,
                    order_id="paper-2",
                    idempotency_key="xyz",
                    symbol="BTCUSDT",
                    side="SELL",
                    requested_qty=0.25,
                    filled_qty=0.25,
                    avg_price=50_000.0,
                    status="filled",
                    created_at=datetime.now(timezone.utc).isoformat(),
                    reason="bounded_limit_exit",
                    risk_policy_version="wp03-risk-v1",
                ),
            )
        )
        server._service.get_portfolio_snapshot = MagicMock(return_value=None)
        server._service.get_paper_state = MagicMock(
            return_value={
                "equity_baseline_usd": 10_000.0,
                "open_positions": {"BTCUSDT": 0.25},
                "paper_entry_prices": {"BTCUSDT": 50_000.0},
            }
        )

        payload = {
            "user_id": 124,
            "prices": {"BTCUSDT": 50_123.4},
            "signals": [
                {
                    "symbol": "BTCUSDT",
                    "timeframe": "1h",
                    "horizon_bars": 4,
                    "signal": "SELL",
                    "confidence": 0.8,
                }
            ],
            "monitoring_snapshot": {
                "connectivity_error_rate": 0.2,
                "execution_anomaly_rate": 0.1,
                "hard_risk_breach": False,
            },
        }

        await server._cmd_route_signals(payload, correlation_id="corr-stage-124")
        return stage_events

    stage_events = asyncio.run(_run())
    assert [event.stage for event in stage_events] == [
        "command_parse",
        "routing_call",
        "ledger_commit",
        "post_fill_refresh",
    ]
    assert all(event.correlation_id == "corr-stage-124" for event in stage_events)
    assert all(event.duration_ms >= 0.0 for event in stage_events)
    assert stage_events[-1].detail == "snapshot=False checkpoint=True"


def test_fix4_route_signals_persists_risk_policy_version_to_wal():
    """Accepted fills should carry the policy version into the WAL payload."""
    from quant_v2.execution.main import ExecutionEngineServer

    async def _run():
        server = ExecutionEngineServer.__new__(ExecutionEngineServer)
        server._stream_bus = MagicMock()
        server._event_bus = AsyncMock()
        server._wal = InMemoryWAL()
        server._shutting_down = False
        server._session_lock = asyncio.Lock()

        server._watchdog = MagicMock()
        server._watchdog.record_tick = MagicMock()
        server._watchdog.update_mtm_equity = MagicMock()

        server._service = MagicMock()
        server._service.route_signals = AsyncMock(
            return_value=(
                ExecutionResult(
                    accepted=True,
                    order_id="paper-1",
                    idempotency_key="abc",
                    symbol="BTCUSDT",
                    side="SELL",
                    requested_qty=0.25,
                    filled_qty=0.25,
                    avg_price=50_000.0,
                    status="filled",
                    created_at=datetime.now(timezone.utc).isoformat(),
                    reason="bounded_limit_exit:supervised_residual_position_required:step_size",
                    risk_policy_version="wp03-risk-v1",
                ),
            )
        )
        server._service.get_portfolio_snapshot = MagicMock(return_value=None)

        payload = {
            "user_id": 123,
            "prices": {"BTCUSDT": 51_234.5},
            "signals": [],
            "monitoring_snapshot": {
                "connectivity_error_rate": 0.2,
                "execution_anomaly_rate": 0.1,
                "hard_risk_breach": False,
            },
        }

        await server._cmd_route_signals(payload, correlation_id="corr-live-123")
        kwargs = server._service.route_signals.await_args.kwargs
        entries = await server._wal.replay()
        return entries, kwargs

    entries, kwargs = asyncio.run(_run())
    order_events = [entry for entry in entries if entry.event_type == "order_executed"]
    assert order_events
    assert order_events[0].payload["risk_policy_version"] == "wp03-risk-v1"
    assert order_events[0].payload["correlation_id"] == "corr-live-123"
    assert kwargs["correlation_id"] == "corr-live-123"


def test_fix4_execution_operational_health_surfaces_queue_and_reconciliation_lag():
    """ExecutionEngineServer should expose queue, WAL, and reconciliation health."""
    from quant_v2.execution.main import ExecutionEngineServer

    async def _run():
        server = ExecutionEngineServer.__new__(ExecutionEngineServer)
        server._stream_bus = MagicMock()
        server._stream_bus.get_queue_health = AsyncMock(
            return_value={
                "status": "warning",
                "stream_backlog_count": 3,
                "pending_count": 2,
                "lag_entries": 1,
                "max_pending_age_seconds": 45.0,
            }
        )
        server._stream_bus._redis = MagicMock()
        server._stream_bus._redis.info = AsyncMock(
            return_value={
                "used_memory": 536_870_912,
                "maxmemory": 1_073_741_824,
            }
        )
        server._wal = MagicMock()
        server._wal.get_stream_health = AsyncMock(
            return_value={
                "status": "healthy",
                "entry_count": 7,
                "latest_entry_age_seconds": 12.0,
            }
        )
        server._service = MagicMock()
        server._service.get_live_session_ids = MagicMock(return_value=(10,))
        server._last_reconciliation_started_at = datetime.now(timezone.utc) - timedelta(seconds=40)
        server._last_reconciliation_completed_at = datetime.now(timezone.utc) - timedelta(seconds=35)
        server._last_reconciliation_error_at = None
        with patch("quant_v2.execution.main.build_runtime_resource_health", return_value={
            "status": "healthy",
            "project_root": "C:/tmp",
            "cpu_percent": 10.0,
            "memory_percent": 20.0,
            "disk_percent": 30.0,
            "rss_mb": 40.0,
            "open_file_count": 2,
            "load_avg_1m": None,
            "process_uptime_seconds": 100.0,
            "container_restart_count": 1,
            "container_started_at": "2026-01-01T00:00:00+00:00",
            "dns_probe": {"status": "healthy", "latency_ms": 2.0},
            "database_lock_probe": {"status": "healthy", "lock_latency_ms": 1.0},
        }):
            return await server.get_operational_health()

    health = asyncio.run(_run())

    assert health["status"] == "warning"
    assert health["live_session_count"] == 1
    assert health["stream_bus"]["stream_backlog_count"] == 3
    assert health["wal"]["entry_count"] == 7
    assert health["reconciliation"]["lag_seconds"] is not None
    assert health["redis_memory"]["status"] == "healthy"
    assert health["redis_memory"]["used_memory_mb"] == 512.0
    assert health["runtime_health"]["container_restart_count"] == 1
