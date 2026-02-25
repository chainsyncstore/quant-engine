"""Tests for Fix 4: Watchdog flatten_requested premature commit.

Validates that flatten_requested is NOT set if _emit_alert fails,
allowing retry on the next watchdog tick cycle.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

from quant_v2.execution.watchdog import LifecycleWatchdog, WatchdogAlert


class TestFlattenNotSetBeforeAlertSuccess:
    """flatten_requested must stay False if the alert handler raises."""

    def test_flatten_not_set_on_stop_loss_when_alert_fails(self):
        alert_calls: list[WatchdogAlert] = []

        async def failing_handler(alert: WatchdogAlert) -> None:
            alert_calls.append(alert)
            raise ConnectionError("Telegram API down")

        async def _run():
            watchdog = LifecycleWatchdog(
                check_interval_seconds=60.0,
                on_alert=failing_handler,
            )

            session = watchdog.register_session(
                user_id=1,
                is_live=True,
                stop_loss_equity_usd=8000.0,
                initial_equity_usd=10_000.0,
            )

            # Trigger stop-loss by dropping equity
            watchdog.update_mtm_equity(1, 7500.0)

            await watchdog._run_checks()

            # Alert handler was called
            assert len(alert_calls) == 1
            assert alert_calls[0].alert_type == "stop_loss_triggered"

            # But flatten_requested should NOT be set because handler raised
            assert session.flatten_requested is False

        asyncio.run(_run())

    def test_flatten_not_set_on_horizon_when_alert_fails(self):
        async def failing_handler(alert: WatchdogAlert) -> None:
            raise RuntimeError("Handler crashed")

        async def _run():
            watchdog = LifecycleWatchdog(
                check_interval_seconds=60.0,
                on_alert=failing_handler,
            )

            session = watchdog.register_session(
                user_id=1,
                is_live=True,
                horizon_hours=0.001,  # ~3.6 seconds
                initial_equity_usd=10_000.0,
            )

            # Force horizon to be expired
            session.horizon_deadline_utc = datetime.now(timezone.utc) - timedelta(
                seconds=10
            )

            await watchdog._run_checks()

            # flatten_requested should NOT be set
            assert session.flatten_requested is False

        asyncio.run(_run())


class TestFlattenSetAfterAlertSuccess:
    """flatten_requested must be True once _emit_alert completes without error."""

    def test_flatten_set_after_successful_stop_loss_alert(self):
        alert_calls: list[WatchdogAlert] = []

        async def success_handler(alert: WatchdogAlert) -> None:
            alert_calls.append(alert)

        async def _run():
            watchdog = LifecycleWatchdog(
                check_interval_seconds=60.0,
                on_alert=success_handler,
            )

            session = watchdog.register_session(
                user_id=1,
                is_live=True,
                stop_loss_equity_usd=8000.0,
                initial_equity_usd=10_000.0,
            )

            watchdog.update_mtm_equity(1, 7500.0)

            await watchdog._run_checks()

            assert len(alert_calls) == 1
            assert session.flatten_requested is True

        asyncio.run(_run())


class TestWatchdogRetriesOnNextTick:
    """After a failed alert, the next _run_checks cycle should re-attempt."""

    def test_watchdog_retries_on_next_tick(self):
        call_count = 0

        async def intermittent_handler(alert: WatchdogAlert) -> None:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ConnectionError("First attempt fails")
            # Second attempt succeeds

        async def _run():
            watchdog = LifecycleWatchdog(
                check_interval_seconds=60.0,
                on_alert=intermittent_handler,
            )

            session = watchdog.register_session(
                user_id=1,
                is_live=True,
                stop_loss_equity_usd=8000.0,
                initial_equity_usd=10_000.0,
            )

            watchdog.update_mtm_equity(1, 7500.0)

            # First tick: alert fails, flatten_requested stays False
            await watchdog._run_checks()
            assert session.flatten_requested is False
            assert call_count == 1

            # Second tick: alert succeeds, flatten_requested becomes True
            await watchdog._run_checks()
            assert session.flatten_requested is True
            assert call_count == 2

        asyncio.run(_run())
