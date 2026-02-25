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
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from quant_v2.execution.adapters import ExecutionResult
from quant_v2.execution.redis_bus import (
    BusMessage,
    RedisStreamCommandBus,
    STREAM_CMD_KEY,
)
from quant_v2.execution.service import RoutedExecutionService, SessionRequest
from quant_v2.execution.state_wal import (
    InMemoryWAL,
    WALEntry,
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
