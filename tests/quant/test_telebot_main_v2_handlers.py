from __future__ import annotations

import asyncio
from datetime import datetime
from dataclasses import dataclass, field
from types import SimpleNamespace

from quant.telebot import main as telebot_main
from quant_v2.execution.service import ExecutionDiagnostics
from quant_v2.monitoring.kill_switch import KillSwitchEvaluation


class _FakeMessage:
    def __init__(self) -> None:
        self.replies: list[str] = []

    async def reply_text(self, text: str) -> None:
        self.replies.append(text)


class _FakeUpdate:
    def __init__(self, user_id: int) -> None:
        self.effective_user = SimpleNamespace(id=user_id)
        self.message = _FakeMessage()


class _FakeContext:
    def __init__(self) -> None:
        self.bot = None
        self.args: list[str] = []


@dataclass
class _FakeSourceManager:
    running: bool
    mode: str = "paper"
    reset_result: bool = True
    stop_result: bool = True
    stats: dict | None = None
    recent: tuple[dict, ...] = ()
    stop_calls: list[int] = field(default_factory=list)

    def is_running(self, user_id: int) -> bool:
        _ = user_id
        return self.running

    def get_session_mode(self, user_id: int) -> str:
        _ = user_id
        return self.mode

    def reset_session_state(self, user_id: int) -> bool:
        _ = user_id
        return self.reset_result

    async def stop_session(self, user_id: int) -> bool:
        self.stop_calls.append(user_id)
        return self.stop_result

    def get_signal_stats(self, user_id: int) -> dict:
        _ = user_id
        return self.stats or {}

    def get_recent_signals(self, user_id: int, *, limit: int = 5) -> tuple[dict, ...]:
        _ = user_id
        return self.recent[:limit]


@dataclass
class _FakeBridge:
    running: bool
    mode: str = "paper"
    reset_result: bool = True
    stop_result: bool = True
    stop_calls: list[int] = field(default_factory=list)

    def is_running(self, user_id: int) -> bool:
        _ = user_id
        return self.running

    def get_session_mode(self, user_id: int) -> str:
        _ = user_id
        return self.mode

    def reset_session_state(self, user_id: int) -> bool:
        _ = user_id
        return self.reset_result

    async def stop_session(self, user_id: int) -> bool:
        self.stop_calls.append(user_id)
        return self.stop_result


def test_reset_demo_v2_marks_degraded_alert_when_pair_mismatched(monkeypatch) -> None:
    user_id = 801
    update = _FakeUpdate(user_id)
    context = _FakeContext()

    source_manager = _FakeSourceManager(running=True, mode="paper", reset_result=True)
    bridge = _FakeBridge(running=False, mode="paper", reset_result=False)

    monkeypatch.setattr(telebot_main, "EXECUTION_BACKEND", "v2_memory")
    monkeypatch.setattr(telebot_main, "_get_signal_source_manager", lambda *args, **kwargs: source_manager)
    monkeypatch.setattr(telebot_main, "_get_v2_bridge", lambda: bridge)

    telebot_main.V2_DEGRADED_ALERTED_USERS.clear()
    asyncio.run(telebot_main.reset_demo(update, context))

    assert user_id in telebot_main.V2_DEGRADED_ALERTED_USERS
    assert update.message.replies
    msg = update.message.replies[-1]
    assert "Session remains degraded" in msg
    assert "Reset: signal source paper state." in msg


def test_reset_demo_v2_clears_degraded_alert_when_pair_healthy(monkeypatch) -> None:
    user_id = 802
    update = _FakeUpdate(user_id)
    context = _FakeContext()

    source_manager = _FakeSourceManager(running=True, mode="paper", reset_result=True)
    bridge = _FakeBridge(running=True, mode="paper", reset_result=True)

    monkeypatch.setattr(telebot_main, "EXECUTION_BACKEND", "v2_memory")
    monkeypatch.setattr(telebot_main, "_get_signal_source_manager", lambda *args, **kwargs: source_manager)
    monkeypatch.setattr(telebot_main, "_get_v2_bridge", lambda: bridge)

    telebot_main.V2_DEGRADED_ALERTED_USERS.clear()
    telebot_main.V2_DEGRADED_ALERTED_USERS.add(user_id)

    asyncio.run(telebot_main.reset_demo(update, context))

    assert user_id not in telebot_main.V2_DEGRADED_ALERTED_USERS
    assert update.message.replies
    msg = update.message.replies[-1]
    assert "Reset: signal source and execution paper state." in msg
    assert "fresh state" in msg


def test_stats_v2_degraded_includes_source_diagnostics(monkeypatch) -> None:
    user_id = 803
    update = _FakeUpdate(user_id)
    context = _FakeContext()

    source_manager = _FakeSourceManager(
        running=True,
        stats={
            "total_signals": 4,
            "buys": 2,
            "sells": 1,
            "holds": 1,
            "drift_alerts": 0,
            "symbols": 1,
        },
        recent=(
            {
                "symbol": "btcusdt",
                "signal": "buy",
                "close_price": object(),
                "probability": "oops",
            },
        ),
    )
    bridge = _FakeBridge(running=False, mode="paper")

    monkeypatch.setattr(telebot_main, "EXECUTION_BACKEND", "v2_memory")
    monkeypatch.setattr(telebot_main, "_get_signal_source_manager", lambda *args, **kwargs: source_manager)
    monkeypatch.setattr(telebot_main, "_get_v2_bridge", lambda: bridge)

    asyncio.run(telebot_main.stats(update, context))

    assert update.message.replies
    msg = update.message.replies[-1]
    assert "Session Degraded" in msg
    assert "Signal Source:" in msg
    assert "BTCUSDT BUY @ 0.00 (P=0.000)" in msg


def test_start_demo_delegates_to_start_engine_with_demo_mode(monkeypatch) -> None:
    update = _FakeUpdate(804)
    context = _FakeContext()

    captured: dict[str, object] = {}

    async def _fake_start_engine(update_arg, context_arg, live: bool) -> None:
        captured["update"] = update_arg
        captured["context"] = context_arg
        captured["live"] = live

    monkeypatch.setattr(telebot_main, "_start_engine", _fake_start_engine)

    asyncio.run(telebot_main.start_demo(update, context))

    assert captured["update"] is update
    assert captured["context"] is context
    assert captured["live"] is False


def test_continue_demo_delegates_to_continue_helper(monkeypatch) -> None:
    update = _FakeUpdate(8041)
    context = _FakeContext()

    captured: dict[str, object] = {}

    async def _fake_continue(update_arg, context_arg, *, live: bool) -> None:
        captured["update"] = update_arg
        captured["context"] = context_arg
        captured["live"] = live

    monkeypatch.setattr(telebot_main, "_continue_from_maintenance", _fake_continue)

    asyncio.run(telebot_main.continue_demo(update, context))

    assert captured["update"] is update
    assert captured["context"] is context
    assert captured["live"] is False


def test_continue_live_delegates_to_continue_helper(monkeypatch) -> None:
    update = _FakeUpdate(8042)
    context = _FakeContext()

    captured: dict[str, object] = {}

    async def _fake_continue(update_arg, context_arg, *, live: bool) -> None:
        captured["update"] = update_arg
        captured["context"] = context_arg
        captured["live"] = live

    monkeypatch.setattr(telebot_main, "_continue_from_maintenance", _fake_continue)

    asyncio.run(telebot_main.continue_live(update, context))

    assert captured["update"] is update
    assert captured["context"] is context
    assert captured["live"] is True


def test_stop_trading_v2_stops_source_and_bridge_and_clears_alert(monkeypatch) -> None:
    user_id = 805
    update = _FakeUpdate(user_id)
    context = _FakeContext()

    source_manager = _FakeSourceManager(running=True, stop_result=True)
    bridge = _FakeBridge(running=True, stop_result=False)

    persisted: dict[str, object] = {}

    def _fake_persist(*args, **kwargs):
        persisted["args"] = args
        persisted["kwargs"] = kwargs

    monkeypatch.setattr(telebot_main, "_get_signal_source_manager", lambda *args, **kwargs: source_manager)
    monkeypatch.setattr(telebot_main, "_get_v2_bridge", lambda: bridge)
    monkeypatch.setattr(telebot_main, "_persist_user_session_flags", _fake_persist)

    telebot_main.V2_DEGRADED_ALERTED_USERS.clear()
    telebot_main.V2_DEGRADED_ALERTED_USERS.add(user_id)

    asyncio.run(telebot_main.stop_trading(update, context))

    assert source_manager.stop_calls == [user_id]
    assert bridge.stop_calls == [user_id]
    assert user_id not in telebot_main.V2_DEGRADED_ALERTED_USERS
    assert persisted["args"] == (user_id,)
    assert persisted["kwargs"] == {"is_active": False}
    assert update.message.replies
    assert "Engine STOPPED" in update.message.replies[-1]


def test_status_v2_marks_degraded_when_session_pair_mismatched(monkeypatch) -> None:
    user_id = 806
    update = _FakeUpdate(user_id)
    context = _FakeContext()

    source_manager = _FakeSourceManager(running=True)
    bridge = _FakeBridge(running=False)

    monkeypatch.setattr(telebot_main, "EXECUTION_BACKEND", "v2_memory")
    monkeypatch.setattr(telebot_main, "_get_signal_source_manager", lambda *args, **kwargs: source_manager)
    monkeypatch.setattr(telebot_main, "_get_v2_bridge", lambda: bridge)

    telebot_main.V2_DEGRADED_ALERTED_USERS.clear()
    asyncio.run(telebot_main.status(update, context))

    assert user_id in telebot_main.V2_DEGRADED_ALERTED_USERS
    assert update.message.replies
    msg = update.message.replies[-1]
    assert "Session Degraded" in msg
    assert "Signal source running: `True`" in msg
    assert "Execution bridge running: `False`" in msg


def test_status_v2_clears_degraded_alert_when_pair_healthy(monkeypatch) -> None:
    user_id = 807
    update = _FakeUpdate(user_id)
    context = _FakeContext()

    source_manager = _FakeSourceManager(running=True)
    bridge = _FakeBridge(running=True)

    monkeypatch.setattr(telebot_main, "EXECUTION_BACKEND", "v2_memory")
    monkeypatch.setattr(telebot_main, "_get_signal_source_manager", lambda *args, **kwargs: source_manager)
    monkeypatch.setattr(telebot_main, "_get_v2_bridge", lambda: bridge)

    telebot_main.V2_DEGRADED_ALERTED_USERS.clear()
    telebot_main.V2_DEGRADED_ALERTED_USERS.add(user_id)

    asyncio.run(telebot_main.status(update, context))

    assert user_id not in telebot_main.V2_DEGRADED_ALERTED_USERS
    assert update.message.replies
    assert "Engine Running" in update.message.replies[-1]


def test_help_command_v2_clarifies_rebalancer_behavior(monkeypatch) -> None:
    update = _FakeUpdate(808)
    context = _FakeContext()

    monkeypatch.setattr(telebot_main, "_get_v2_bridge", lambda: object())
    monkeypatch.setattr(telebot_main, "_using_shadow_backend", lambda: False)
    monkeypatch.setattr(telebot_main, "_is_admin_user", lambda user_id: False)

    asyncio.run(telebot_main.help_command(update, context))

    assert update.message.replies
    msg = update.message.replies[-1]
    assert "/start_demo - Start PAPER trading (v2 multi-symbol)" in msg
    assert "/continue_demo - Resume after maintenance from snapshot" in msg
    assert "practice balance" in msg
    assert "/lifecycle - Show your auto-close safety settings" in msg
    assert "/set_horizon <hours|off> - Auto-close open trades after N hours" in msg
    assert "/set_stoploss <percent|off> - Auto-close a trade at your max loss %" in msg
    assert "/lifetime_stats - View lifetime demo/live equity, notional, and PnL" in msg


def test_lifecycle_command_reports_persisted_and_runtime_rules(monkeypatch) -> None:
    update = _FakeUpdate(8081)
    context = _FakeContext()

    class _LifecycleBridge:
        def is_running(self, user_id: int) -> bool:
            _ = user_id
            return True

        def get_lifecycle_rules(self, user_id: int):
            _ = user_id
            return SimpleNamespace(auto_close_horizon_bars=4, stop_loss_pct=0.02)

    monkeypatch.setattr(telebot_main, "_get_v2_bridge", lambda: _LifecycleBridge())
    monkeypatch.setattr(
        telebot_main,
        "_load_persisted_lifecycle_preferences",
        lambda user_id: (8, 0.03),
    )

    asyncio.run(telebot_main.lifecycle(update, context))

    assert update.message.replies
    msg = update.message.replies[-1]
    assert "Saved time limit" in msg
    assert "8 hour(s)" in msg
    assert "Saved loss limit" in msg
    assert "Close trade if loss reaches 3.00%" in msg
    assert "Active time limit" in msg
    assert "4 hour(s)" in msg
    assert "Active loss limit" in msg
    assert "Close trade if loss reaches 2.00%" in msg
    assert "Applied right now to your active session." in msg


def test_set_horizon_persists_and_applies_to_active_session(monkeypatch) -> None:
    update = _FakeUpdate(8082)
    context = _FakeContext()
    context.args = ["4"]

    class _LifecycleBridge:
        def is_running(self, user_id: int) -> bool:
            _ = user_id
            return True

    persisted: dict[str, object] = {}

    def _fake_persist(user_id: int, *, auto_close_horizon_bars=None, stop_loss_pct=None):
        persisted["user_id"] = user_id
        persisted["horizon"] = auto_close_horizon_bars
        persisted["stop_loss"] = stop_loss_pct
        return 4, 0.02

    monkeypatch.setattr(telebot_main, "_get_v2_bridge", lambda: _LifecycleBridge())
    monkeypatch.setattr(telebot_main, "_persist_lifecycle_preferences", _fake_persist)
    monkeypatch.setattr(
        telebot_main,
        "_apply_lifecycle_preferences_to_running_session",
        lambda *args, **kwargs: True,
    )

    asyncio.run(telebot_main.set_horizon(update, context))

    assert persisted == {
        "user_id": 8082,
        "horizon": 4,
        "stop_loss": None,
    }
    assert update.message.replies
    msg = update.message.replies[-1]
    assert "Trade Safety Updated" in msg
    assert "4 hour(s)" in msg
    assert "Close trade if loss reaches 2.00%" in msg
    assert "Applied now to your active session." in msg


def test_set_stoploss_accepts_percent_and_persists_fraction(monkeypatch) -> None:
    update = _FakeUpdate(8083)
    context = _FakeContext()
    context.args = ["2"]

    class _LifecycleBridge:
        def is_running(self, user_id: int) -> bool:
            _ = user_id
            return False

    persisted: dict[str, object] = {}

    def _fake_persist(user_id: int, *, auto_close_horizon_bars=None, stop_loss_pct=None):
        persisted["user_id"] = user_id
        persisted["horizon"] = auto_close_horizon_bars
        persisted["stop_loss"] = stop_loss_pct
        return 6, float(stop_loss_pct)

    monkeypatch.setattr(telebot_main, "_get_v2_bridge", lambda: _LifecycleBridge())
    monkeypatch.setattr(telebot_main, "_persist_lifecycle_preferences", _fake_persist)
    monkeypatch.setattr(
        telebot_main,
        "_apply_lifecycle_preferences_to_running_session",
        lambda *args, **kwargs: False,
    )

    asyncio.run(telebot_main.set_stoploss(update, context))

    assert persisted["user_id"] == 8083
    assert persisted["horizon"] is None
    assert persisted["stop_loss"] == 0.02
    assert update.message.replies
    msg = update.message.replies[-1]
    assert "Trade Safety Updated" in msg
    assert "6 hour(s)" in msg
    assert "Close trade if loss reaches 2.00%" in msg
    assert "Saved for your next `/start_demo` or `/start_live`." in msg


def test_help_command_admin_lists_prepare_update(monkeypatch) -> None:
    update = _FakeUpdate(809)
    context = _FakeContext()

    monkeypatch.setattr(telebot_main, "_get_v2_bridge", lambda: object())
    monkeypatch.setattr(telebot_main, "_using_shadow_backend", lambda: False)
    monkeypatch.setattr(telebot_main, "_is_admin_user", lambda user_id: True)

    asyncio.run(telebot_main.help_command(update, context))

    assert update.message.replies
    msg = update.message.replies[-1]
    assert "/prepare_update - Save user snapshots and send pre-update notice" in msg
    assert "/update_complete - Notify users deploy is done and share recovery steps" in msg


def test_lifetime_stats_renders_summary(monkeypatch) -> None:
    user_id = 8091
    update = _FakeUpdate(user_id)
    context = _FakeContext()

    class _Bridge:
        def is_running(self, requested_user_id: int) -> bool:
            _ = requested_user_id
            return True

    class _SourceManager:
        def is_running(self, requested_user_id: int) -> bool:
            _ = requested_user_id
            return False

    monkeypatch.setattr(telebot_main, "_get_v2_bridge", lambda: _Bridge())
    monkeypatch.setattr(telebot_main, "_get_signal_source_manager", lambda: _SourceManager())
    monkeypatch.setattr(
        telebot_main,
        "_refresh_lifetime_stats_from_runtime",
        lambda requested_user_id, *, bridge: None,
    )
    monkeypatch.setattr(
        telebot_main,
        "_load_lifetime_stats_summary",
        lambda requested_user_id: {
            "current_demo_equity_usd": 10_250.0,
            "current_live_equity_usd": 9_700.0,
            "current_demo_notional_usd": 1_000.0,
            "current_live_notional_usd": 2_200.0,
            "current_demo_symbols": 2,
            "current_live_symbols": 3,
            "lifetime_demo_pnl_usd": 250.0,
            "lifetime_live_pnl_usd": -300.0,
            "active_mode": "LIVE",
            "created_at": datetime(2026, 1, 1, 12, 0, 0),
            "last_updated_at": datetime(2026, 1, 2, 13, 30, 0),
            "strategy_profile": "core_v2",
        },
    )

    asyncio.run(telebot_main.lifetime_stats(update, context))

    assert update.message.replies
    msg = update.message.replies[-1]
    assert "Lifetime Trading Stats" in msg
    assert "Demo equity" in msg
    assert "$10,250.00" in msg
    assert "Live active notional" in msg
    assert "$2,200.00" in msg
    assert "Demo total" in msg
    assert "$+250.00" in msg
    assert "Live total" in msg
    assert "$-300.00" in msg
    assert "Combined total" in msg
    assert "$-50.00" in msg
    assert "Engine status: `RUNNING`" in msg


def test_lifetime_stats_returns_not_found_for_missing_user(monkeypatch) -> None:
    update = _FakeUpdate(8092)
    context = _FakeContext()

    monkeypatch.setattr(telebot_main, "_get_v2_bridge", lambda: None)
    monkeypatch.setattr(telebot_main, "_get_signal_source_manager", lambda: None)
    monkeypatch.setattr(
        telebot_main,
        "_refresh_lifetime_stats_from_runtime",
        lambda requested_user_id, *, bridge: None,
    )
    monkeypatch.setattr(telebot_main, "_load_lifetime_stats_summary", lambda requested_user_id: None)

    asyncio.run(telebot_main.lifetime_stats(update, context))

    assert update.message.replies
    assert "Account not found" in update.message.replies[-1]


def test_execution_diagnostics_text_includes_activity_and_caps() -> None:
    class _DiagBridge:
        def get_execution_diagnostics(self, user_id: int):
            _ = user_id
            return ExecutionDiagnostics(
                total_orders=5,
                accepted_orders=4,
                rejected_orders=1,
                reject_rate=0.2,
                slippage_sample_count=3,
                avg_adverse_slippage_bps=7.5,
                entry_orders=2,
                rebalance_orders=2,
                exit_orders=0,
                skipped_by_filter=1,
                skipped_by_deadband=2,
                paused_cycles=3,
                blocked_actionable_signals=7,
                effective_symbol_cap_frac=0.05,
                effective_gross_cap_frac=0.15,
                effective_net_cap_frac=0.10,
            )

        def get_kill_switch_evaluation(self, user_id: int):
            _ = user_id
            return KillSwitchEvaluation(pause_trading=False)

    text = telebot_main._build_execution_diagnostics_text(_DiagBridge(), 808)
    assert "Order Activity" in text
    assert "entries=2, rebalances=2, exits=0" in text
    assert "Skipped: filter=1, deadband=2" in text
    assert "Kill-switch blocks: cycles=3, actionable_signals=7" in text
    assert "Effective caps" in text
