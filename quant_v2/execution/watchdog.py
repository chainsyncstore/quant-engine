"""Independent lifecycle watchdog for tick-starvation-immune enforcement.

Runs as an independent asyncio task that checks session horizons and
stop-losses against UTC wall-clock time, regardless of whether market
data ticks are arriving.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Awaitable

logger = logging.getLogger(__name__)


@dataclass
class WatchedSession:
    """A session being monitored by the watchdog."""

    user_id: int
    started_at: datetime
    horizon_deadline_utc: datetime | None = None
    stop_loss_equity_usd: float | None = None
    latest_mtm_equity_usd: float = 0.0
    is_live: bool = False
    flatten_requested: bool = False


@dataclass(frozen=True)
class WatchdogAlert:
    """An alert emitted by the watchdog when a lifecycle condition triggers."""

    user_id: int
    alert_type: str  # "horizon_expired" | "stop_loss_triggered" | "heartbeat_stale"
    reason: str
    triggered_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class LifecycleWatchdog:
    """Independent async watchdog that enforces lifecycle constraints via wall-clock UTC.

    This watchdog runs every `check_interval_seconds` and is completely
    decoupled from market data ticks. Even if the websocket feed drops,
    the watchdog will still fire horizon expiry and stop-loss alerts.
    """

    def __init__(
        self,
        *,
        check_interval_seconds: float = 5.0,
        on_alert: Callable[[WatchdogAlert], Awaitable[None]] | None = None,
        stale_heartbeat_seconds: float = 120.0,
    ) -> None:
        self._check_interval = check_interval_seconds
        self._on_alert = on_alert
        self._stale_heartbeat_seconds = stale_heartbeat_seconds
        self._sessions: dict[int, WatchedSession] = {}
        self._last_tick_time: dict[int, datetime] = {}
        self._task: asyncio.Task | None = None
        self._running = False

    def register_session(
        self,
        user_id: int,
        *,
        is_live: bool = False,
        horizon_hours: float | None = None,
        stop_loss_equity_usd: float | None = None,
        initial_equity_usd: float = 10_000.0,
    ) -> WatchedSession:
        """Register a session for lifecycle monitoring."""
        now = datetime.now(timezone.utc)

        deadline = None
        if horizon_hours is not None and horizon_hours > 0:
            deadline = now + timedelta(hours=horizon_hours)

        session = WatchedSession(
            user_id=user_id,
            started_at=now,
            horizon_deadline_utc=deadline,
            stop_loss_equity_usd=stop_loss_equity_usd,
            latest_mtm_equity_usd=initial_equity_usd,
            is_live=is_live,
        )
        self._sessions[user_id] = session
        self._last_tick_time[user_id] = now
        logger.info(
            "Watchdog registered session user=%d (horizon=%s, stop_loss=$%.2f)",
            user_id,
            deadline.isoformat() if deadline else "none",
            stop_loss_equity_usd or 0.0,
        )
        return session

    def deregister_session(self, user_id: int) -> bool:
        """Remove a session from watchdog monitoring."""
        removed = self._sessions.pop(user_id, None)
        self._last_tick_time.pop(user_id, None)
        if removed:
            logger.info("Watchdog deregistered session user=%d", user_id)
        return removed is not None

    def update_mtm_equity(self, user_id: int, equity_usd: float) -> None:
        """Update the latest MTM equity for a watched session."""
        session = self._sessions.get(user_id)
        if session:
            session.latest_mtm_equity_usd = equity_usd

    def update_horizon(self, user_id: int, deadline_utc: datetime | None) -> None:
        """Update the horizon deadline for a watched session."""
        session = self._sessions.get(user_id)
        if session:
            session.horizon_deadline_utc = deadline_utc

    def update_stop_loss(self, user_id: int, stop_loss_equity_usd: float | None) -> None:
        """Update the stop-loss equity threshold for a watched session."""
        session = self._sessions.get(user_id)
        if session:
            session.stop_loss_equity_usd = stop_loss_equity_usd

    def record_tick(self, user_id: int) -> None:
        """Record that a market data tick was received for freshness tracking."""
        if user_id in self._sessions:
            self._last_tick_time[user_id] = datetime.now(timezone.utc)

    def get_watched_sessions(self) -> dict[int, WatchedSession]:
        """Return a copy of all watched sessions."""
        return dict(self._sessions)

    async def start(self) -> None:
        """Start the watchdog background loop."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._check_loop())
        logger.info(
            "Watchdog started (interval=%.1fs, stale_threshold=%.1fs)",
            self._check_interval,
            self._stale_heartbeat_seconds,
        )

    async def stop(self) -> None:
        """Stop the watchdog background loop."""
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._task = None
        logger.info("Watchdog stopped")

    async def _check_loop(self) -> None:
        """Main watchdog loop that runs on wall-clock time."""
        try:
            while self._running:
                await self._run_checks()
                await asyncio.sleep(self._check_interval)
        except asyncio.CancelledError:
            logger.debug("Watchdog check loop cancelled")

    async def _run_checks(self) -> None:
        """Evaluate all watched sessions against current UTC time."""
        now = datetime.now(timezone.utc)

        for user_id, session in list(self._sessions.items()):
            if session.flatten_requested:
                continue

            # 1. Horizon expiry check
            if (
                session.horizon_deadline_utc is not None
                and now >= session.horizon_deadline_utc
            ):
                session.flatten_requested = True
                alert = WatchdogAlert(
                    user_id=user_id,
                    alert_type="horizon_expired",
                    reason=(
                        f"Horizon deadline reached: {session.horizon_deadline_utc.isoformat()}"
                    ),
                )
                await self._emit_alert(alert)
                continue

            # 2. Stop-loss check
            if (
                session.stop_loss_equity_usd is not None
                and session.latest_mtm_equity_usd <= session.stop_loss_equity_usd
            ):
                session.flatten_requested = True
                alert = WatchdogAlert(
                    user_id=user_id,
                    alert_type="stop_loss_triggered",
                    reason=(
                        f"MTM equity ${session.latest_mtm_equity_usd:,.2f} "
                        f"<= stop-loss ${session.stop_loss_equity_usd:,.2f}"
                    ),
                )
                await self._emit_alert(alert)
                continue

            # 3. Heartbeat staleness check (tick starvation detection)
            last_tick = self._last_tick_time.get(user_id)
            if last_tick is not None:
                staleness = (now - last_tick).total_seconds()
                if staleness > self._stale_heartbeat_seconds:
                    alert = WatchdogAlert(
                        user_id=user_id,
                        alert_type="heartbeat_stale",
                        reason=(
                            f"No market tick for {staleness:.0f}s "
                            f"(threshold: {self._stale_heartbeat_seconds:.0f}s)"
                        ),
                    )
                    await self._emit_alert(alert)
                    # Reset timer to avoid flooding
                    self._last_tick_time[user_id] = now

    async def _emit_alert(self, alert: WatchdogAlert) -> None:
        """Dispatch an alert to the registered handler."""
        logger.warning(
            "Watchdog alert: user=%d type=%s reason=%s",
            alert.user_id,
            alert.alert_type,
            alert.reason,
        )
        if self._on_alert:
            try:
                await self._on_alert(alert)
            except Exception:
                logger.exception(
                    "Watchdog alert handler failed for user=%d", alert.user_id
                )
