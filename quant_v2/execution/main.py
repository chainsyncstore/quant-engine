"""Standalone entry point for the v2 execution engine container.

This module subscribes to the Redis command bus, processes execution
commands via RoutedExecutionService, and publishes results back to
the Telegram container via Redis Pub/Sub.

Usage:
    python -m quant_v2.execution.main
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys

from quant_v2.execution.redis_bus import (
    CMD_EXEC_CHANNEL,
    BusMessage,
    RedisCommandBus,
)
from quant_v2.execution.service import RoutedExecutionService, SessionRequest
from quant_v2.execution.state_wal import RedisWAL
from quant_v2.execution.watchdog import LifecycleWatchdog, WatchdogAlert
from quant_v2.portfolio.risk_policy import PortfolioRiskPolicy

logger = logging.getLogger(__name__)


class ExecutionEngineServer:
    """Standalone execution engine that communicates via Redis."""

    def __init__(
        self,
        redis_url: str,
        *,
        risk_policy: PortfolioRiskPolicy | None = None,
    ) -> None:
        self._redis_url = redis_url
        self._bus = RedisCommandBus(redis_url)
        self._wal = RedisWAL(redis_url)

        self._watchdog = LifecycleWatchdog(
            check_interval_seconds=5.0,
            on_alert=self._handle_watchdog_alert,
            stale_heartbeat_seconds=120.0,
        )

        self._service = RoutedExecutionService(
            risk_policy=risk_policy,
            allow_live_execution=True,
        )

        self._shutting_down = False

    async def start(self) -> None:
        """Initialize connections and start processing."""
        await self._bus.connect()
        await self._wal.connect()
        await self._bus.subscribe(CMD_EXEC_CHANNEL, self._handle_command)
        await self._watchdog.start()
        logger.info("Execution engine server started")

    async def stop(self) -> None:
        """Gracefully shut down all components."""
        self._shutting_down = True
        await self._watchdog.stop()
        await self._bus.disconnect()
        await self._wal.disconnect()
        logger.info("Execution engine server stopped")

    async def _handle_command(self, msg: BusMessage) -> None:
        """Dispatch a command from the Telegram container."""
        action = msg.action
        payload = msg.payload
        correlation_id = msg.correlation_id

        try:
            if action == "start_session":
                result = await self._cmd_start_session(payload)
            elif action == "stop_session":
                result = await self._cmd_stop_session(payload)
            elif action == "route_signals":
                result = await self._cmd_route_signals(payload)
            elif action == "get_snapshot":
                result = await self._cmd_get_snapshot(payload)
            elif action == "set_lifecycle":
                result = await self._cmd_set_lifecycle(payload)
            elif action == "ping":
                result = {"status": "pong"}
            else:
                result = {"error": f"unknown_action: {action}"}

            await self._bus.send_event(
                action=f"{action}_result",
                payload=result,
                correlation_id=correlation_id,
            )

        except Exception as e:
            logger.exception("Command handler error for action=%s", action)
            await self._bus.send_event(
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
            await self._wal.log_session_started(
                user_id, live=live, strategy_profile=req.strategy_profile
            )
            self._watchdog.register_session(
                user_id, is_live=live, initial_equity_usd=10_000.0
            )

        return {"success": success, "user_id": user_id}

    async def _cmd_stop_session(self, payload: dict) -> dict:
        user_id = int(payload["user_id"])
        success = await self._service.stop_session(user_id)

        if success:
            await self._wal.log_session_stopped(user_id)
            self._watchdog.deregister_session(user_id)

        return {"success": success, "user_id": user_id}

    async def _cmd_route_signals(self, payload: dict) -> dict:
        from quant_v2.contracts import StrategySignal

        user_id = int(payload["user_id"])
        prices = payload.get("prices", {})
        raw_signals = payload.get("signals", [])

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

        results = await self._service.route_signals(
            user_id, signals=signals, prices=prices
        )

        # Log position updates to WAL
        for r in results:
            if r.accepted:
                await self._wal.log_order_executed(
                    user_id,
                    symbol=r.symbol,
                    side=r.side,
                    quantity=r.filled_qty,
                    avg_price=r.avg_price,
                    status=r.status,
                )

        # Update watchdog with latest equity
        snap = self._service.get_portfolio_snapshot(user_id)
        if snap:
            self._watchdog.update_mtm_equity(user_id, snap.equity_usd)
            self._watchdog.record_tick(user_id)

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

    async def _handle_watchdog_alert(self, alert: WatchdogAlert) -> None:
        """Handle alerts from the lifecycle watchdog."""
        logger.warning(
            "Watchdog alert for user=%d: %s - %s",
            alert.user_id,
            alert.alert_type,
            alert.reason,
        )

        await self._bus.send_event(
            action="watchdog_alert",
            payload={
                "user_id": alert.user_id,
                "alert_type": alert.alert_type,
                "reason": alert.reason,
                "triggered_at": alert.triggered_at,
            },
        )

        # If stop-loss or horizon, auto-flatten
        if alert.alert_type in ("stop_loss_triggered", "horizon_expired"):
            logger.info(
                "Auto-flattening positions for user=%d due to %s",
                alert.user_id,
                alert.alert_type,
            )
            await self._wal.log_kill_switch(
                alert.user_id, triggered=True, reasons=(alert.alert_type,)
            )


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
    await server.stop()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    asyncio.run(run_server())


if __name__ == "__main__":
    main()
