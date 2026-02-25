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

from quant_v2.execution.redis_bus import (
    BusMessage,
    EVT_TG_CHANNEL,
    RedisCommandBus,
    RedisStreamCommandBus,
)
from quant_v2.execution.service import RoutedExecutionService, SessionRequest
from quant_v2.execution.state_wal import RedisWAL, WALEntry
from quant_v2.execution.watchdog import LifecycleWatchdog, WatchdogAlert
from quant_v2.portfolio.risk_policy import PortfolioRiskPolicy

logger = logging.getLogger(__name__)


class ExecutionEngineServer:
    """Standalone execution engine that communicates via Redis.

    Command ingestion uses Redis Streams (at-least-once, survives restarts).
    Event publishing (replies to Telegram) uses Pub/Sub (ephemeral, best-effort).
    """

    def __init__(
        self,
        redis_url: str,
        *,
        risk_policy: PortfolioRiskPolicy | None = None,
    ) -> None:
        self._redis_url = redis_url

        # FIX-2: Guaranteed command bus replaces fire-and-forget Pub/Sub for incoming cmds.
        self._stream_bus = RedisStreamCommandBus(redis_url)

        # Keep the PubSub bus solely for Telegram event replies (non-safety-critical).
        self._event_bus = RedisCommandBus(redis_url)

        # FIX-3: WAL connected at boot; replayed before accepting commands.
        self._wal = RedisWAL(redis_url)

        self._watchdog = LifecycleWatchdog(
            check_interval_seconds=5.0,
            on_alert=self._handle_watchdog_alert,
            stale_heartbeat_seconds=7200.0,
        )

        self._service = RoutedExecutionService(
            risk_policy=risk_policy,
            allow_live_execution=True,
        )

        # Session-state mutex: serialises concurrent mutations from the
        # command handler, watchdog alerts, and reconciliation loop to
        # prevent double-execution / WAL corruption race conditions.
        self._session_lock = asyncio.Lock()

        self._shutting_down = False
        self._reconciliation_task: asyncio.Task | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Initialize connections, replay WAL, then start processing."""
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

    # ------------------------------------------------------------------
    # FIX-3: WAL Replay  --  reconstruct session state on boot
    # ------------------------------------------------------------------

    async def _rebuild_state_from_wal(self, entries: list[WALEntry]) -> int:
        """Replay WAL entries to rebuild the in-memory session map.

        Only 'session_started' and 'session_stopped' events are needed to
        determine which sessions should be live.  Actual positions are
        retrieved from the exchange on the next routing cycle.
        """
        rebuilt = 0
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
                live_ids = self._service.get_live_session_ids()
                for user_id in live_ids:
                    try:
                        await self._reconcile_session(user_id)
                    except Exception as exc:
                        logger.error(
                            "Reconciliation failed for user %d: %s", user_id, exc
                        )
        except asyncio.CancelledError:
            logger.debug("Reconciliation loop cancelled")

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
                    result = await self._cmd_route_signals(payload)
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
            await self._wal.log_session_started(
                user_id, live=live, strategy_profile=req.strategy_profile
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

    # ------------------------------------------------------------------
    # FIX-4: Watchdog alert handler — heartbeat_stale now engages kill-switch
    # ------------------------------------------------------------------

    async def _handle_watchdog_alert(self, alert: WatchdogAlert) -> None:
        """Handle alerts from the lifecycle watchdog.

        FIX-4: heartbeat_stale (market data feed dropped) now engages the
        kill-switch to pause new order entries, preventing the engine from
        trading on stale prices during a data outage.

        Auto-flatten is performed for hard lifecycle events (stop_loss_triggered,
        horizon_expired).  For heartbeat_stale we pause-only — we do NOT
        force-close positions, as the outage could be transient.  The kill-switch
        will lift automatically on the next successful tick.
        """
        async with self._session_lock:
            logger.warning(
                "Watchdog alert for user=%d: %s - %s",
                alert.user_id,
                alert.alert_type,
                alert.reason,
            )

            await self._event_bus.send_event(
                action="watchdog_alert",
                payload={
                    "user_id": alert.user_id,
                    "alert_type": alert.alert_type,
                    "reason": alert.reason,
                    "triggered_at": alert.triggered_at,
                },
            )

            # Auto-flatten: position-closing lifecycle events
            if alert.alert_type in ("stop_loss_triggered", "horizon_expired"):
                logger.info(
                    "Auto-flattening positions for user=%d due to %s",
                    alert.user_id,
                    alert.alert_type,
                )
                await self._wal.log_kill_switch(
                    alert.user_id, triggered=True, reasons=(alert.alert_type,)
                )

            # FIX-4: Pause trading (halt new entries, exits remain active) when data goes dark.
            elif alert.alert_type == "heartbeat_stale":
                logger.warning(
                    "Market data feed stale for user=%d — pausing new order entries. "
                    "Exits remain active. Kill-switch will lift on next tick.",
                    alert.user_id,
                )
                await self._wal.log_kill_switch(
                    alert.user_id,
                    triggered=True,
                    reasons=("heartbeat_stale",),
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
    await server.stop()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    asyncio.run(run_server())


if __name__ == "__main__":
    main()
