"""Tests for Fix 3: Reconciliation drift alerts to Telegram.

Validates that _reconcile_session broadcasts reconciliation_drift_alert
via _event_bus when phantom or ghost positions are detected.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from quant_v2.execution.main import ExecutionEngineServer
from quant_v2.execution.service import RoutedExecutionService, SessionRequest
from quant_v2.execution.state_wal import InMemoryWAL


class DriftingAdapter:
    """Adapter that returns different positions to simulate drift.

    get_positions() returns exchange positions (used by reconciliation).
    The snapshot will start empty, so any positions from get_positions
    will appear as phantom drift.
    """

    def __init__(self, exchange_positions: dict[str, float]) -> None:
        self._positions = exchange_positions

    def get_positions(self) -> dict[str, float]:
        return dict(self._positions)

    def place_order(self, plan, *, idempotency_key, mark_price=None, limit_price=None, post_only=False):
        from quant_v2.execution.adapters import ExecutionResult
        from datetime import datetime, timezone
        return ExecutionResult(
            accepted=True, order_id="1", idempotency_key=idempotency_key,
            symbol=plan.symbol, side=plan.side, requested_qty=float(plan.quantity),
            filled_qty=float(plan.quantity), avg_price=float(mark_price or 0),
            status="filled", created_at=datetime.now(timezone.utc).isoformat(), reason="",
        )


def test_drift_sends_telegram_event():
    """When phantom/ghost drift is detected, a reconciliation_drift_alert event is sent."""

    async def _run():
        # Adapter returns ETHUSDT which the service doesn't track yet
        drifting = DriftingAdapter({"BTCUSDT": 0.5, "ETHUSDT": 2.0})
        service = RoutedExecutionService(
            paper_adapter_factory=lambda: DriftingAdapter({}),
            live_adapter_factory=lambda req: drifting,
            allow_live_execution=True,
        )

        req = SessionRequest(user_id=1, live=True)
        await service.start_session(req)

        wal = InMemoryWAL()

        server = ExecutionEngineServer.__new__(ExecutionEngineServer)
        server._service = service
        server._wal = wal
        server._shutting_down = False
        server._session_lock = asyncio.Lock()
        server._event_bus = AsyncMock()

        # Force the snapshot's open_positions to differ from the adapter
        # by patching get_portfolio_snapshot to return an empty snapshot
        from quant_v2.contracts import PortfolioSnapshot, RiskSnapshot
        from datetime import datetime, timezone
        empty_snap = PortfolioSnapshot(
            timestamp=datetime.now(timezone.utc),
            equity_usd=10_000.0,
            risk=RiskSnapshot(
                gross_exposure_frac=0.0, net_exposure_frac=0.0,
                max_drawdown_frac=0.0, risk_budget_used_frac=0.0,
            ),
            open_positions={},
        )
        original_get_snap = service.get_portfolio_snapshot
        service.get_portfolio_snapshot = lambda uid: empty_snap

        await server._reconcile_session(1)

        calls = server._event_bus.send_event.call_args_list
        drift_calls = [
            c for c in calls
            if c.kwargs.get("action") == "reconciliation_drift_alert"
        ]
        assert len(drift_calls) >= 1, (
            f"Expected reconciliation_drift_alert event, got actions: "
            f"{[c.kwargs.get('action') for c in calls]}"
        )

    asyncio.run(_run())


def test_sync_failure_sends_alert():
    """If sync_positions raises, a reconciliation_sync_failed alert is sent."""

    async def _run():
        drifting = DriftingAdapter({"BTCUSDT": 0.5, "ETHUSDT": 2.0})
        service = RoutedExecutionService(
            paper_adapter_factory=lambda: DriftingAdapter({}),
            live_adapter_factory=lambda req: drifting,
            allow_live_execution=True,
        )

        req = SessionRequest(user_id=1, live=True)
        await service.start_session(req)

        # Force drift detection by patching snapshot
        from quant_v2.contracts import PortfolioSnapshot, RiskSnapshot
        from datetime import datetime, timezone
        empty_snap = PortfolioSnapshot(
            timestamp=datetime.now(timezone.utc),
            equity_usd=10_000.0,
            risk=RiskSnapshot(
                gross_exposure_frac=0.0, net_exposure_frac=0.0,
                max_drawdown_frac=0.0, risk_budget_used_frac=0.0,
            ),
            open_positions={},
        )
        service.get_portfolio_snapshot = lambda uid: empty_snap

        # Make sync_positions raise
        async def failing_sync(*a, **kw):
            raise RuntimeError("Binance down")
        service.sync_positions = failing_sync

        wal = InMemoryWAL()

        server = ExecutionEngineServer.__new__(ExecutionEngineServer)
        server._service = service
        server._wal = wal
        server._shutting_down = False
        server._session_lock = asyncio.Lock()
        server._event_bus = AsyncMock()

        # Should NOT raise
        await server._reconcile_session(1)

        calls = server._event_bus.send_event.call_args_list
        sync_fail_calls = [
            c for c in calls
            if c.kwargs.get("action") == "reconciliation_sync_failed"
        ]
        assert len(sync_fail_calls) >= 1, (
            f"Expected reconciliation_sync_failed event, got: "
            f"{[c.kwargs.get('action') for c in calls]}"
        )

    asyncio.run(_run())
