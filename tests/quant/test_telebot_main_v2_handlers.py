from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from types import SimpleNamespace

from quant.telebot import main as telebot_main


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
