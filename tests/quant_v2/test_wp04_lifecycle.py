from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from quant_v2.execution.main import ExecutionEngineServer
from quant_v2.execution.service import RoutedExecutionService, SessionRequest
from quant_v2.execution.state_wal import InMemoryWAL
from quant_v2.execution.watchdog import LifecycleWatchdog
from quant_v2.monitoring.kill_switch import MonitoringSnapshot


class _FakePaperAdapter:
    def __init__(self) -> None:
        self._positions: dict[str, float] = {}

    def get_positions(self) -> dict[str, float]:
        return dict(self._positions)


def test_lifecycle_state_transitions_are_monotonic_and_retry_counts_increment() -> None:
    service = RoutedExecutionService(
        paper_adapter_factory=_FakePaperAdapter,
        live_adapter_factory=lambda request: _FakePaperAdapter(),
    )

    assert asyncio.run(service.start_session(SessionRequest(user_id=901, live=False))) is True

    active = service.get_lifecycle_state(901)
    assert active is not None
    assert active.state == "ACTIVE"
    assert active.owner == "alpha_session"
    assert active.retry_count == 0

    pause_eval = service.set_monitoring_snapshot(
        901,
        MonitoringSnapshot(hard_risk_breach=True),
    )
    assert pause_eval.pause_trading is True

    reduce_only = service.get_lifecycle_state(901)
    assert reduce_only is not None
    assert reduce_only.state == "REDUCE_ONLY"

    retried = service.set_lifecycle_state(
        901,
        state="REDUCE_ONLY",
        owner="liquidation_supervisor",
        reason="hard_risk_breach",
        policy_version=reduce_only.policy_version,
    )
    assert retried.retry_count == reduce_only.retry_count + 1

    flattening = service.set_lifecycle_state(
        901,
        state="FLATTENING",
        owner="liquidation_supervisor",
        reason="supervised_flatten",
        policy_version=reduce_only.policy_version,
    )
    assert flattening.state == "FLATTENING"

    confirmed = service.set_lifecycle_state(
        901,
        state="FLAT_CONFIRMED",
        owner="liquidation_supervisor",
        reason="positions_flat",
        policy_version=reduce_only.policy_version,
    )
    assert confirmed.state == "FLAT_CONFIRMED"

    with pytest.raises(ValueError):
        service.set_lifecycle_state(
            901,
            state="REDUCE_ONLY",
            owner="liquidation_supervisor",
            reason="backwards_transition",
            policy_version=reduce_only.policy_version,
        )

    assert asyncio.run(service.stop_session(901)) is True
    paused = service.get_lifecycle_state(901)
    assert paused is not None
    assert paused.state == "PAUSED"
    assert paused.owner == "control_plane"


def test_lifecycle_state_replays_across_start_stop_boundaries() -> None:
    async def _run() -> tuple[list, ExecutionEngineServer]:
        wal = InMemoryWAL()
        server = ExecutionEngineServer.__new__(ExecutionEngineServer)
        server._stream_bus = MagicMock()
        server._event_bus = AsyncMock()
        server._event_bus.send_event = AsyncMock(return_value=0)
        server._wal = wal
        server._shutting_down = False
        server._session_lock = asyncio.Lock()
        server._service = RoutedExecutionService(
            paper_adapter_factory=_FakePaperAdapter,
            live_adapter_factory=lambda request: _FakePaperAdapter(),
        )
        server._watchdog = LifecycleWatchdog()

        await server._cmd_start_session(
            {
                "user_id": 321,
                "live": False,
                "strategy_profile": "core_v2",
            }
        )

        active = server._service.get_lifecycle_state(321)
        assert active is not None

        previous = active
        server._service.set_lifecycle_state(
            321,
            state="REDUCE_ONLY",
            owner="liquidation_supervisor",
            reason="hard_risk_breach",
            policy_version=active.policy_version,
        )
        await server._persist_lifecycle_state(321, previous_state=previous)

        previous = server._service.get_lifecycle_state(321)
        assert previous is not None
        server._service.set_lifecycle_state(
            321,
            state="FLATTENING",
            owner="liquidation_supervisor",
            reason="stale_feed_circuit_breaker",
            policy_version=previous.policy_version,
        )
        await server._persist_lifecycle_state(321, previous_state=previous)

        previous = server._service.get_lifecycle_state(321)
        assert previous is not None
        server._service.set_lifecycle_state(
            321,
            state="FLAT_CONFIRMED",
            owner="liquidation_supervisor",
            reason="positions_flat",
            policy_version=previous.policy_version,
        )
        await server._persist_lifecycle_state(321, previous_state=previous)

        await server._cmd_stop_session({"user_id": 321})
        return await wal.replay("0-0"), server

    entries, server = asyncio.run(_run())
    lifecycle_entries = [entry for entry in entries if entry.event_type == "lifecycle_changed"]
    assert [entry.payload["state"] for entry in lifecycle_entries] == [
        "REDUCE_ONLY",
        "FLATTENING",
        "FLAT_CONFIRMED",
        "PAUSED",
    ]

    start_entries = [entry for entry in entries if entry.event_type == "session_started"]
    assert start_entries
    assert start_entries[0].payload["lifecycle_state"]["state"] == "ACTIVE"

    assert server._service.get_lifecycle_state(321) is not None
    assert server._service.get_lifecycle_state(321).state == "PAUSED"

    rebuilt_server = ExecutionEngineServer.__new__(ExecutionEngineServer)
    rebuilt_server._stream_bus = MagicMock()
    rebuilt_server._event_bus = AsyncMock()
    rebuilt_server._wal = InMemoryWAL()
    rebuilt_server._shutting_down = False
    rebuilt_server._session_lock = asyncio.Lock()
    rebuilt_server._service = RoutedExecutionService(
        paper_adapter_factory=_FakePaperAdapter,
        live_adapter_factory=lambda request: _FakePaperAdapter(),
    )
    rebuilt_server._watchdog = LifecycleWatchdog()

    async def _replay() -> int:
        rebuilt_server._wal._entries = list(entries)
        replayed = await rebuilt_server._wal.replay("0-0")
        return await rebuilt_server._rebuild_state_from_wal(replayed)

    rebuilt = asyncio.run(_replay())
    assert rebuilt == 0
    assert rebuilt_server._service.is_running(321) is False
    restored = rebuilt_server._service.get_lifecycle_state(321)
    assert restored is not None
    assert restored.state == "PAUSED"
    assert restored.owner == "control_plane"
