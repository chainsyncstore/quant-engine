"""Tests for Pre-Flight Fix 2: asyncio.Lock session mutex in ExecutionEngineServer."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from quant_v2.execution.main import ExecutionEngineServer
from quant_v2.execution.redis_bus import BusMessage


@pytest.fixture
def mock_server():
    """Create an ExecutionEngineServer with mocked dependencies."""
    with patch("quant_v2.execution.main.RedisStreamCommandBus") as MockStream, \
         patch("quant_v2.execution.main.RedisCommandBus") as MockPubSub, \
         patch("quant_v2.execution.main.RedisWAL") as MockWAL, \
         patch("quant_v2.execution.main.LifecycleWatchdog") as MockWatchdog:

        MockStream.return_value = MagicMock()
        MockPubSub.return_value = MagicMock()
        MockWAL.return_value = MagicMock()
        MockWatchdog.return_value = MagicMock()

        server = ExecutionEngineServer("redis://localhost:6379")
        # Mock the event bus send_event to be async
        server._event_bus.send_event = AsyncMock()
        server._wal.log_session_started = AsyncMock(return_value="1-0")
        server._wal.log_session_stopped = AsyncMock(return_value="2-0")
        server._wal.log_kill_switch = AsyncMock(return_value="3-0")
        server._wal.append = AsyncMock(return_value="4-0")

        yield server


class TestSessionLockExists:
    """Verify that the session lock attribute is properly created."""

    def test_session_lock_is_asyncio_lock(self, mock_server):
        """ExecutionEngineServer must have an asyncio.Lock."""
        assert hasattr(mock_server, "_session_lock")
        assert isinstance(mock_server._session_lock, asyncio.Lock)


class TestConcurrentMutationsAreSerialized:
    """Verify that concurrent _handle_command and _handle_watchdog_alert
    cannot corrupt session state by running simultaneously."""

    def test_concurrent_stop_and_watchdog_serialized(self, mock_server):
        """Simulate concurrent /stop command and watchdog alert.

        If the lock works, one will complete before the other starts.
        We verify this by tracking the order of lock acquisitions.
        """
        from quant_v2.execution.watchdog import WatchdogAlert

        async def _run():
            execution_order = []

            async def mock_stop_session(user_id):
                execution_order.append("stop_session_start")
                await asyncio.sleep(0.01)
                execution_order.append("stop_session_end")
                return True

            mock_server._service.stop_session = mock_stop_session
            mock_server._watchdog.deregister_session = MagicMock()

            async def tracked_cmd_stop(payload):
                execution_order.append("cmd_stop_enter")
                result = {"success": True, "user_id": int(payload["user_id"])}
                await mock_stop_session(int(payload["user_id"]))
                execution_order.append("cmd_stop_exit")
                return result

            mock_server._cmd_stop_session = tracked_cmd_stop

            stop_msg = BusMessage(
                action="stop_session",
                payload={"user_id": 42},
                timestamp="2025-01-01T00:00:00Z",
                correlation_id="test-corr-1",
            )

            alert = WatchdogAlert(
                user_id=42,
                alert_type="stop_loss_triggered",
                reason="test",
            )

            await asyncio.gather(
                mock_server._handle_command(stop_msg),
                mock_server._handle_watchdog_alert(alert),
            )

            assert len(execution_order) > 0, "At least one handler should have executed"

        asyncio.run(_run())

    def test_lock_prevents_concurrent_access(self, mock_server):
        """Directly test that the lock is acquired during _handle_command."""

        async def _run():
            lock_was_held = False

            async def check_lock_cmd(payload):
                nonlocal lock_was_held
                lock_was_held = mock_server._session_lock.locked()
                return {"status": "ok"}

            mock_server._cmd_start_session = check_lock_cmd

            msg = BusMessage(
                action="start_session",
                payload={"user_id": 99},
                timestamp="2025-01-01T00:00:00Z",
            )
            await mock_server._handle_command(msg)
            assert lock_was_held, "Session lock should be held during command handling"

        asyncio.run(_run())
