from __future__ import annotations

import asyncio
from datetime import datetime
from dataclasses import dataclass, field
from types import SimpleNamespace

from quant.telebot import main as telebot_main
from quant_v2.execution.service import ExecutionDiagnostics
from quant_v2.model_registry import ModelRegistry
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
    traded_stats: dict | None = None
    recent_traded: tuple[dict, ...] = ()
    scorecard: dict | None = None
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

    def get_traded_signal_stats(self, user_id: int) -> dict:
        _ = user_id
        return self.traded_stats or {"total_trades": 0, "buys": 0, "sells": 0, "symbols": 0, "per_symbol": {}}

    def get_recent_traded_signals(self, user_id: int, *, limit: int = 8) -> tuple[dict, ...]:
        _ = user_id
        return self.recent_traded[:limit]

    def get_scorecard_summary(self) -> dict:
        return self.scorecard or {}


@dataclass
class _FakeBridge:
    running: bool
    mode: str = "paper"
    reset_result: bool = True
    stop_result: bool = True
    service: object | None = None
    execution_diagnostics: object | None = None
    stop_calls: list[int] = field(default_factory=list)
    flatten_calls: list[tuple[int, dict[str, float] | None]] = field(default_factory=list)

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

    def get_execution_diagnostics(self, user_id: int) -> object | None:
        _ = user_id
        return self.execution_diagnostics

    async def flatten_session(self, user_id: int, *, prices: dict[str, float] | None = None):
        self.flatten_calls.append((user_id, prices))
        return (
            SimpleNamespace(
                accepted=True,
                symbol="BTCUSDT",
            ),
        )


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


def test_flatten_demo_v2_flattens_running_sessions(monkeypatch) -> None:
    user_id = 804
    update = _FakeUpdate(user_id)
    context = _FakeContext()

    source_manager = _FakeSourceManager(running=True, mode="paper")
    bridge = _FakeBridge(
        running=True,
        mode="paper",
        service=SimpleNamespace(
            get_portfolio_snapshot=lambda uid: SimpleNamespace(
                open_positions={"BTCUSDT": 0.015, "ETHUSDT": -0.02}
            ),
            get_last_prices=lambda uid: {"BTCUSDT": 63000.0, "ETHUSDT": 3200.0},
        ),
    )

    monkeypatch.setattr(telebot_main, "EXECUTION_BACKEND", "v2_memory")
    monkeypatch.setattr(telebot_main, "_get_signal_source_manager", lambda *args, **kwargs: source_manager)
    monkeypatch.setattr(telebot_main, "_get_v2_bridge", lambda: bridge)
    monkeypatch.setattr(telebot_main, "_load_running_demo_session_ids", lambda **kwargs: [804, 805])
    monkeypatch.setattr(telebot_main, "_is_admin_user", lambda user_id: True)

    asyncio.run(telebot_main.flatten_demo(update, context))

    assert bridge.flatten_calls == [
        (804, {"BTCUSDT": 63000.0, "ETHUSDT": 3200.0}),
        (805, {"BTCUSDT": 63000.0, "ETHUSDT": 3200.0}),
    ]
    assert update.message.replies
    msg = update.message.replies[-1]
    assert "Demo flatten complete" in msg
    assert "Sessions targeted: `2`" in msg
    assert "Accepted orders: `2`" in msg
    assert "flattened `1` order(s) across `2` symbol(s)" in msg


def test_stats_v2_degraded_includes_source_diagnostics(monkeypatch) -> None:
    user_id = 803
    update = _FakeUpdate(user_id)
    context = _FakeContext()

    source_manager = _FakeSourceManager(
        running=True,
        traded_stats={
            "total_trades": 3,
            "buys": 2,
            "sells": 1,
            "symbols": 1,
            "per_symbol": {"BTCUSDT": {"buys": 2, "sells": 1}},
        },
        recent_traded=(
            {
                "symbol": "btcusdt",
                "signal": "buy",
                "close_price": 50000.0,
                "probability": 0.72,
                "regime": 3,
            },
        ),
    )
    bridge = _FakeBridge(
        running=False,
        mode="paper",
        service=SimpleNamespace(
            get_portfolio_snapshot=lambda uid: SimpleNamespace(
                open_positions={"btcusdt": 0.015},
                symbol_notional_usd={"btcusdt": 945.0},
                symbol_pnl_usd={"btcusdt": 12.34},
            ),
            get_last_prices=lambda uid: {"btcusdt": 63000.0},
        ),
    )

    monkeypatch.setattr(telebot_main, "EXECUTION_BACKEND", "v2_memory")
    monkeypatch.setattr(telebot_main, "_get_signal_source_manager", lambda *args, **kwargs: source_manager)
    monkeypatch.setattr(telebot_main, "_get_v2_bridge", lambda: bridge)

    asyncio.run(telebot_main.stats(update, context))

    assert update.message.replies
    msg = update.message.replies[-1]
    assert "Session Degraded" in msg
    assert "Model Trade Picks (session):" in msg
    assert "- Held positions:" in msg
    assert "BTCUSDT: LONG 0.015000" in msg
    assert "mark=63000.00" in msg
    assert "notional=$945.00" in msg
    assert "pnl=$+12.34" in msg
    assert "BTCUSDT BUY @ 50000.00 (P=0.720, R=3)" in msg


def test_model_trade_picks_derives_symbols_from_positions(monkeypatch) -> None:
    """When signal_log is empty (restart) but bridge has positions, active symbols
    and trade counts should still reflect reality."""

    source_manager = _FakeSourceManager(
        running=True,
        traded_stats={
            "total_trades": 0, "buys": 0, "sells": 0,
            "symbols": 0, "per_symbol": {},
        },
    )
    bridge = _FakeBridge(
        running=False,
        mode="paper",
        service=SimpleNamespace(
            get_portfolio_snapshot=lambda uid: SimpleNamespace(
                open_positions={"AVAXUSDT": 74.39, "ETHUSDT": 1.5},
                symbol_notional_usd={"AVAXUSDT": 697.0, "ETHUSDT": 3500.0},
                symbol_pnl_usd={"AVAXUSDT": -2.0, "ETHUSDT": 15.0},
            ),
            get_last_prices=lambda uid: {"AVAXUSDT": 9.38, "ETHUSDT": 2333.0},
        ),
        execution_diagnostics=ExecutionDiagnostics(
            routed_buy_signals=5, routed_sell_signals=2,
            routed_actionable_signals=7,
        ),
    )

    text = telebot_main._build_source_signal_diagnostics_text(
        source_manager, 999, bridge=bridge,
    )

    assert "Active symbols: 2" in text
    assert "Trades: 7 (BUY=5, SELL=2)" in text
    assert "AVAXUSDT: LONG" in text
    assert "ETHUSDT: LONG" in text


def test_refresh_v2_stats_market_snapshot_ingests_normalized_prices() -> None:
    class _Bridge:
        def __init__(self) -> None:
            self.ingested: list[dict[str, float]] = []

        def is_running(self, user_id: int) -> bool:
            _ = user_id
            return True

        def ingest_market_prices(self, user_id: int, prices: dict[str, float]) -> bool:
            _ = user_id
            self.ingested.append(prices)
            return True

    class _Source:
        async def get_realtime_prices(self, user_id: int) -> dict[str, object]:
            _ = user_id
            return {
                "btcusdt": "101.5",
                "ETHUSDT": 0.0,
                "": 99.0,
                "XRPUSDT": "bad",
            }

    bridge = _Bridge()
    source = _Source()

    asyncio.run(
        telebot_main._refresh_v2_stats_market_snapshot(
            999,
            bridge=bridge,
            source_manager=source,
        )
    )

    assert bridge.ingested == [{"BTCUSDT": 101.5}]


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

    requested: dict[str, object] = {}

    def _fake_request(user_id_arg, *, state, owner, reason, evidence_ref, policy_version):
        requested["user_id"] = user_id_arg
        requested["state"] = state
        requested["owner"] = owner
        requested["reason"] = reason
        requested["evidence_ref"] = evidence_ref
        requested["policy_version"] = policy_version
        return {
            "state": state,
            "owner": owner,
            "retry_count": 1,
            "policy_version": policy_version,
            "reason": reason,
            "evidence_ref": evidence_ref,
        }

    monkeypatch.setattr(telebot_main, "_get_signal_source_manager", lambda *args, **kwargs: source_manager)
    monkeypatch.setattr(telebot_main, "_get_v2_bridge", lambda: bridge)
    monkeypatch.setattr(telebot_main, "_request_reviewed_lifecycle_transition", _fake_request)

    telebot_main.V2_DEGRADED_ALERTED_USERS.clear()
    telebot_main.V2_DEGRADED_ALERTED_USERS.add(user_id)

    asyncio.run(telebot_main.stop_trading(update, context))

    assert source_manager.stop_calls == [user_id]
    assert bridge.stop_calls == [user_id]
    assert user_id not in telebot_main.V2_DEGRADED_ALERTED_USERS
    assert requested == {
        "user_id": user_id,
        "state": "review_requested",
        "owner": "telegram_operator",
        "reason": "operator_stop",
        "evidence_ref": "telegram:/stop",
        "policy_version": telebot_main.LIFECYCLE_TRANSITION_POLICY_VERSION,
    }
    assert update.message.replies
    assert "Engine STOPPED" in update.message.replies[-1]
    assert "Pause request recorded for review." in update.message.replies[-1]


def test_start_demo_blocks_when_reviewed_lifecycle_transition_is_pending(monkeypatch) -> None:
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from quant.telebot.models import Base, User, UserContext

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    user = User(telegram_id=901, username="user901", status="active")
    user.context = UserContext(
        telegram_id=901,
        is_active=True,
        lifecycle_transition_state="review_requested",
        lifecycle_transition_owner="telegram_operator",
        lifecycle_transition_retry_count=1,
        lifecycle_transition_policy_version=telebot_main.LIFECYCLE_TRANSITION_POLICY_VERSION,
        lifecycle_transition_reason="operator_stop",
        lifecycle_transition_evidence_ref="telegram:/stop",
    )
    session.add(user)
    session.commit()
    session.close()

    class _StartSourceManager:
        def __init__(self) -> None:
            self.start_calls: list[int] = []

        def is_running(self, user_id: int) -> bool:
            _ = user_id
            return False

        async def start_session(self, user_id: int, creds, *, on_signal, execute_orders: bool) -> bool:
            _ = creds, on_signal, execute_orders
            self.start_calls.append(user_id)
            return True

    class _StartBridge:
        def __init__(self) -> None:
            self.start_calls: list[int] = []

        async def start_session(self, user_id: int, *, live: bool, credentials) -> bool:
            _ = live, credentials
            self.start_calls.append(user_id)
            return True

    update = _FakeUpdate(901)
    context = _FakeContext()
    source_manager = _StartSourceManager()
    bridge = _StartBridge()

    monkeypatch.setattr(telebot_main, "SessionLocal", Session)
    monkeypatch.setattr(telebot_main, "_get_signal_source_manager", lambda *args, **kwargs: source_manager)
    monkeypatch.setattr(telebot_main, "_get_v2_bridge", lambda: bridge)
    monkeypatch.setattr(telebot_main, "_using_v2_backend", lambda: True)
    monkeypatch.setattr(telebot_main, "_using_v2_primary_backend", lambda: False)

    asyncio.run(telebot_main.start_demo(update, context))

    assert source_manager.start_calls == []
    assert bridge.start_calls == []
    assert update.message.replies
    assert "reviewed lifecycle transition" in update.message.replies[-1]


def test_review_transition_command_clears_pending_request_after_two_live_approvals(monkeypatch) -> None:
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from quant.telebot.models import Base, User, UserContext

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    user = User(telegram_id=902, username="user902", status="active")
    user.context = UserContext(
        telegram_id=902,
        is_active=True,
        live_mode=True,
        lifecycle_transition_state="review_requested",
        lifecycle_transition_owner="telegram_operator",
        lifecycle_transition_retry_count=1,
        lifecycle_transition_policy_version=telebot_main.LIFECYCLE_TRANSITION_POLICY_VERSION,
        lifecycle_transition_reason="operator_stop",
        lifecycle_transition_evidence_ref="telegram:/stop",
    )
    session.add(user)
    session.commit()
    session.close()

    update = _FakeUpdate(1000)
    context = _FakeContext()
    context.args = ["902", "approve", "telegram:/stop", "recon:902", "true"]

    monkeypatch.setattr(telebot_main, "SessionLocal", Session)
    monkeypatch.setattr(telebot_main, "_is_admin_user", lambda user_id: True)
    monkeypatch.setattr(telebot_main, "_get_v2_bridge", lambda: None)

    asyncio.run(telebot_main.review_transition(update, context))

    session = Session()
    db_user = session.query(User).filter_by(telegram_id=902).first()
    assert db_user is not None and db_user.context is not None
    assert db_user.context.lifecycle_transition_state == "review_requested"
    assert db_user.context.lifecycle_transition_approval_count == 1
    assert update.message.replies
    assert "second authorized approval" in update.message.replies[-1]

    update_second = _FakeUpdate(1001)
    context_second = _FakeContext()
    context_second.args = ["902", "approve", "telegram:/stop", "recon:902", "true"]
    asyncio.run(telebot_main.review_transition(update_second, context_second))

    session.expire_all()
    db_user = session.query(User).filter_by(telegram_id=902).first()
    assert db_user is not None and db_user.context is not None
    assert db_user.context.lifecycle_transition_state == "cleared"
    assert db_user.context.lifecycle_transition_reviewed_at is not None
    assert db_user.context.lifecycle_transition_approval_count == 2
    assert "Lifecycle transition cleared" in update_second.message.replies[-1]
    session.close()


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
    assert "/review_transition <user_id> approve <evidence_ref> <reconciliation_ref> [flat_confirmed]" in msg
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
    monkeypatch.setattr(
        telebot_main,
        "_load_active_lifecycle_transition",
        lambda user_id: {
            "state": "review_requested",
            "owner": "telegram_operator",
            "retry_count": 2,
            "policy_version": telebot_main.LIFECYCLE_TRANSITION_POLICY_VERSION,
            "requested_at": datetime(2026, 5, 28, 13, 30),
            "reviewed_at": None,
            "reason": "operator_stop",
            "evidence_ref": "telegram:/stop",
        },
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
    assert "Lifecycle transition review" in msg
    assert "review_requested" in msg
    assert "telegram_operator" in msg


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
    assert "/flatten_demo - Flatten open positions for running demo sessions" in msg
    assert "/model_candidates - List retrain candidates" in msg
    assert "/model_eval - Show quarantine forward-evaluation summary" in msg
    assert "/model_auto_promote on|off - Toggle gated automatic promotion" in msg
    assert "/model_approve <version_id> <evidence_digest> [--expires-at=<iso>] [reason...] - Record a promotion approval" in msg
    assert "/model_events [count] - Show registry transition history" in msg
    assert "/model_promote <version_id> [evidence_digest] [reason...] - Manually activate a candidate" in msg
    assert "/model_reject <version_id> [reason...] - Mark candidate as rejected" in msg
    assert "/model_expire <version_id> [reason...] - Mark candidate as expired" in msg


def test_model_active_shows_manifest_contract_summary(tmp_path, monkeypatch) -> None:
    registry = ModelRegistry(tmp_path / "registry")
    artifact = tmp_path / "candidate"
    artifact.mkdir(parents=True)

    from datetime import timezone

    import numpy as np
    import pandas as pd

    from quant_v2.models.trainer import save_model_bundle, train

    monkeypatch.setenv("QUANT_IMAGE", "registry.example/quant-bot@sha256:" + "a" * 64)
    idx = pd.date_range(end=datetime.now(timezone.utc), periods=180, freq="h", tz="UTC")
    X = pd.DataFrame(
        {
            "f1": np.linspace(-2.0, 2.0, len(idx)),
            "f2": np.sin(np.linspace(0.0, 8.0, len(idx))),
            "f3": np.cos(np.linspace(0.0, 6.0, len(idx))),
        },
        index=idx,
    )
    y = pd.Series((np.sin(np.linspace(0.0, 12.0, len(idx))) > 0.0).astype(int), index=idx)
    trained = train(X, y, horizon=4, calibration_frac=0.2)
    save_model_bundle(
        trained,
        artifact / "model_4m.pkl",
        metadata={
            "dataset_digest": "digest-123",
            "threshold": 0.60,
            "threshold_policy": {
                "source": "oof_dev_predictions",
                "selected_threshold": 0.60,
                "selected_accuracy": 0.63,
            },
        },
    )
    registry.register_version("candidate_contract", artifact, metrics={"promotion_eligible": True})
    registry.set_active_version("candidate_contract")

    update = _FakeUpdate(80908)
    context = _FakeContext()

    monkeypatch.setattr(telebot_main, "MODEL_REGISTRY", registry)
    monkeypatch.setattr(telebot_main, "_is_admin_user", lambda user_id: True)
    monkeypatch.setattr(telebot_main, "_resolve_runtime_model_dir", lambda: artifact)
    monkeypatch.setattr(telebot_main, "_get_v2_bridge", lambda: None)
    monkeypatch.setattr(telebot_main, "_get_signal_source_manager", lambda **kwargs: None)
    monkeypatch.setattr(telebot_main, "_using_v2_primary_backend", lambda: True)

    asyncio.run(telebot_main.model_active(update, context))

    assert update.message.replies
    msg = update.message.replies[-1]
    assert "Pointer manifest" in msg
    assert "Pointer threshold" in msg
    assert "Manifest image" in msg
    assert "Manifest feature schema" in msg
    assert "Manifest dataset digest" in msg
    assert "Manifest threshold" in msg
    assert "Manifest threshold policy" in msg


def test_model_candidates_shows_contract_summary(tmp_path, monkeypatch) -> None:
    registry = ModelRegistry(tmp_path / "registry")
    artifact = tmp_path / "candidate"
    artifact.mkdir(parents=True)

    from datetime import timezone

    import numpy as np
    import pandas as pd

    from quant_v2.models.trainer import save_model_bundle, train

    monkeypatch.setenv("QUANT_IMAGE", "registry.example/quant-bot@sha256:" + "a" * 64)
    idx = pd.date_range(end=datetime.now(timezone.utc), periods=180, freq="h", tz="UTC")
    X = pd.DataFrame(
        {
            "f1": np.linspace(-2.0, 2.0, len(idx)),
            "f2": np.sin(np.linspace(0.0, 8.0, len(idx))),
            "f3": np.cos(np.linspace(0.0, 6.0, len(idx))),
        },
        index=idx,
    )
    y = pd.Series((np.sin(np.linspace(0.0, 12.0, len(idx))) > 0.0).astype(int), index=idx)
    trained = train(X, y, horizon=4, calibration_frac=0.2)
    save_model_bundle(
        trained,
        artifact / "model_4m.pkl",
        metadata={
            "dataset_digest": "digest-456",
            "threshold": 0.60,
            "threshold_policy": {
                "source": "oof_dev_predictions",
                "selected_threshold": 0.60,
                "selected_accuracy": 0.63,
            },
        },
    )
    registry.register_version("candidate_contract", artifact, metrics={"promotion_eligible": True})
    registry.mark_paper_quarantine("candidate_contract", notes="paper run")

    update = _FakeUpdate(80909)
    context = _FakeContext()

    monkeypatch.setattr(telebot_main, "MODEL_REGISTRY", registry)
    monkeypatch.setattr(telebot_main, "_is_admin_user", lambda user_id: True)

    asyncio.run(telebot_main.model_candidates(update, context))

    assert update.message.replies
    msg = update.message.replies[-1]
    assert "contract=image=registry.example/quant-bot@sha256:" in msg
    assert "schema=" in msg
    assert "dataset=digest-456" in msg
    assert "threshold=0.6" in msg
    assert "policy=oof_dev_predictions" in msg


def test_model_versions_shows_contract_summary(tmp_path, monkeypatch) -> None:
    registry = ModelRegistry(tmp_path / "registry")
    artifact = tmp_path / "candidate"
    artifact.mkdir(parents=True)

    from datetime import timezone

    import numpy as np
    import pandas as pd

    from quant_v2.models.trainer import save_model_bundle, train

    monkeypatch.setenv("QUANT_IMAGE", "registry.example/quant-bot@sha256:" + "a" * 64)
    idx = pd.date_range(end=datetime.now(timezone.utc), periods=180, freq="h", tz="UTC")
    X = pd.DataFrame(
        {
            "f1": np.linspace(-2.0, 2.0, len(idx)),
            "f2": np.sin(np.linspace(0.0, 8.0, len(idx))),
            "f3": np.cos(np.linspace(0.0, 6.0, len(idx))),
        },
        index=idx,
    )
    y = pd.Series((np.sin(np.linspace(0.0, 12.0, len(idx))) > 0.0).astype(int), index=idx)
    trained = train(X, y, horizon=4, calibration_frac=0.2)
    save_model_bundle(
        trained,
        artifact / "model_4m.pkl",
        metadata={
            "dataset_digest": "digest-789",
            "threshold": 0.60,
            "threshold_policy": {
                "source": "oof_dev_predictions",
                "selected_threshold": 0.60,
                "selected_accuracy": 0.63,
            },
        },
    )
    registry.register_version("candidate_contract", artifact, metrics={"promotion_eligible": True})
    registry.mark_paper_quarantine("candidate_contract", notes="paper run")
    registry.set_active_version("candidate_contract")

    update = _FakeUpdate(80910)
    context = _FakeContext()

    monkeypatch.setattr(telebot_main, "MODEL_REGISTRY", registry)
    monkeypatch.setattr(telebot_main, "_is_admin_user", lambda user_id: True)

    asyncio.run(telebot_main.model_versions(update, context))

    assert update.message.replies
    msg = update.message.replies[-1]
    assert "contract=image=registry.example/quant-bot@sha256:" in msg
    assert "schema=" in msg
    assert "dataset=digest-789" in msg
    assert "threshold=0.6" in msg
    assert "policy=oof_dev_predictions" in msg


def test_model_promote_admin_updates_active_pointer(tmp_path, monkeypatch) -> None:
    registry = ModelRegistry(tmp_path / "registry")
    artifact = tmp_path / "candidate"
    artifact.mkdir(parents=True)
    from datetime import timezone

    import numpy as np
    import pandas as pd

    from quant_v2.models.trainer import save_model_bundle, train

    monkeypatch.setenv("QUANT_IMAGE", "registry.example/quant-bot@sha256:" + "a" * 64)
    idx = pd.date_range(end=datetime.now(timezone.utc), periods=180, freq="h", tz="UTC")
    X = pd.DataFrame(
        {
            "f1": np.linspace(-2.0, 2.0, len(idx)),
            "f2": np.sin(np.linspace(0.0, 8.0, len(idx))),
            "f3": np.cos(np.linspace(0.0, 6.0, len(idx))),
        },
        index=idx,
    )
    y = pd.Series((np.sin(np.linspace(0.0, 12.0, len(idx))) > 0.0).astype(int), index=idx)
    trained = train(X, y, horizon=4, calibration_frac=0.2)
    save_model_bundle(trained, artifact / "model_4m.pkl")
    registry.register_version(
        "candidate_a",
        artifact,
        metrics={"promotion_eligible": True},
    )

    update = _FakeUpdate(8092)
    context = _FakeContext()
    context.args = ["candidate_a"]

    class _Source:
        model_dir = artifact

        def get_active_count(self) -> int:
            return 0

    class _Bridge:
        def get_active_count(self) -> int:
            return 0

    monkeypatch.setattr(telebot_main, "MODEL_REGISTRY", registry)
    monkeypatch.setattr(telebot_main, "_is_admin_user", lambda user_id: True)
    monkeypatch.setattr(telebot_main, "_get_v2_bridge", lambda: _Bridge())
    monkeypatch.setattr(telebot_main, "_get_signal_source_manager", lambda **kwargs: _Source())

    asyncio.run(telebot_main.model_promote(update, context))

    active = registry.get_active_version()
    assert active is not None
    assert active.version_id == "candidate_a"
    assert active.promoted_by == "telegram:8092"
    assert update.message.replies
    assert "Model Promotion Applied" in update.message.replies[-1]


def test_model_approve_records_registry_event(tmp_path, monkeypatch) -> None:
    registry = ModelRegistry(tmp_path / "registry")
    artifact = tmp_path / "candidate"
    artifact.mkdir(parents=True)
    registry.register_version("candidate_a", artifact, metrics={"promotion_eligible": True})

    update = _FakeUpdate(8093)
    context = _FakeContext()
    context.args = ["candidate_a", "sha256:" + "3" * 64, "reviewed", "by", "ops"]

    monkeypatch.setattr(telebot_main, "MODEL_REGISTRY", registry)
    monkeypatch.setattr(telebot_main, "_is_admin_user", lambda user_id: True)

    asyncio.run(telebot_main.model_approve(update, context))

    events = registry.list_registry_events()
    assert any(event.event_type == "promotion_approval_recorded" for event in events)
    assert update.message.replies
    assert "Promotion Approval Recorded" in update.message.replies[-1]


def test_model_approve_records_expiry_metadata(tmp_path, monkeypatch) -> None:
    registry = ModelRegistry(tmp_path / "registry")
    artifact = tmp_path / "candidate"
    artifact.mkdir(parents=True)
    registry.register_version("candidate_a", artifact, metrics={"promotion_eligible": True})

    update = _FakeUpdate(80931)
    context = _FakeContext()
    context.args = [
        "candidate_a",
        "sha256:" + "5" * 64,
        "--expires-at=2026-12-31T23:59:59+00:00",
        "reviewed",
        "by",
        "ops",
    ]

    monkeypatch.setattr(telebot_main, "MODEL_REGISTRY", registry)
    monkeypatch.setattr(telebot_main, "_is_admin_user", lambda user_id: True)

    asyncio.run(telebot_main.model_approve(update, context))

    event = registry.list_registry_events()[-1]
    assert event.payload["expires_at"] == "2026-12-31T23:59:59+00:00"
    assert "Expires at" in update.message.replies[-1]


def test_model_events_lists_registry_history(tmp_path, monkeypatch) -> None:
    registry = ModelRegistry(tmp_path / "registry")
    artifact = tmp_path / "candidate"
    artifact.mkdir(parents=True)
    registry.register_version("candidate_a", artifact, metrics={"promotion_eligible": True})
    registry.set_active_version("candidate_a")
    registry.record_promotion_approval(
        "candidate_a",
        approved_by="telegram:9001",
        evidence_digest="sha256:" + "4" * 64,
        reason="reviewed",
    )

    update = _FakeUpdate(8094)
    context = _FakeContext()
    context.args = ["5"]

    monkeypatch.setattr(telebot_main, "MODEL_REGISTRY", registry)
    monkeypatch.setattr(telebot_main, "_is_admin_user", lambda user_id: True)

    asyncio.run(telebot_main.model_events(update, context))

    assert update.message.replies
    msg = update.message.replies[-1]
    assert "Registry Event History" in msg
    assert "promotion_approval_recorded" in msg
    assert "active_pointer_set" in msg
    assert "candidate_a" in msg
    assert "telegram:9001" in msg


def test_model_reject_marks_terminal_status(tmp_path, monkeypatch) -> None:
    registry = ModelRegistry(tmp_path / "registry")
    artifact = tmp_path / "candidate"
    artifact.mkdir(parents=True)
    registry.register_version("candidate_reject", artifact, metrics={"promotion_eligible": True})

    update = _FakeUpdate(8098)
    context = _FakeContext()
    context.args = ["candidate_reject", "manual", "rejection"]

    monkeypatch.setattr(telebot_main, "MODEL_REGISTRY", registry)
    monkeypatch.setattr(telebot_main, "_is_admin_user", lambda user_id: True)

    asyncio.run(telebot_main.model_reject(update, context))

    record = registry.get_version("candidate_reject")
    assert record is not None
    assert record.status == "rejected"
    assert update.message.replies
    assert "Model Rejected" in update.message.replies[-1]


def test_model_expire_marks_terminal_status(tmp_path, monkeypatch) -> None:
    registry = ModelRegistry(tmp_path / "registry")
    artifact = tmp_path / "candidate"
    artifact.mkdir(parents=True)
    registry.register_version("candidate_expire", artifact, metrics={"promotion_eligible": True})

    update = _FakeUpdate(8099)
    context = _FakeContext()
    context.args = ["candidate_expire", "manual", "expiry"]

    monkeypatch.setattr(telebot_main, "MODEL_REGISTRY", registry)
    monkeypatch.setattr(telebot_main, "_is_admin_user", lambda user_id: True)

    asyncio.run(telebot_main.model_expire(update, context))

    record = registry.get_version("candidate_expire")
    assert record is not None
    assert record.status == "expired"
    assert update.message.replies
    assert "Model Expired" in update.message.replies[-1]


def test_model_promote_requires_evidence_digest_when_approval_gate_enabled(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("MODEL_REGISTRY_REQUIRE_TWO_PERSON_APPROVAL", "1")
    registry = ModelRegistry(tmp_path / "registry")
    artifact = tmp_path / "candidate"
    artifact.mkdir(parents=True)

    from datetime import timezone

    import numpy as np
    import pandas as pd

    from quant_v2.models.trainer import save_model_bundle, train

    monkeypatch.setenv("QUANT_IMAGE", "registry.example/quant-bot@sha256:" + "a" * 64)
    idx = pd.date_range(end=datetime.now(timezone.utc), periods=180, freq="h", tz="UTC")
    X = pd.DataFrame(
        {
            "f1": np.linspace(-2.0, 2.0, len(idx)),
            "f2": np.sin(np.linspace(0.0, 8.0, len(idx))),
            "f3": np.cos(np.linspace(0.0, 6.0, len(idx))),
        },
        index=idx,
    )
    y = pd.Series((np.sin(np.linspace(0.0, 12.0, len(idx))) > 0.0).astype(int), index=idx)
    trained = train(X, y, horizon=4, calibration_frac=0.2)
    save_model_bundle(trained, artifact / "model_4m.pkl")
    registry.register_version("candidate_gate", artifact, metrics={"promotion_eligible": True})
    registry.record_promotion_approval(
        "candidate_gate",
        approved_by="reviewer_a",
        evidence_digest="sha256:" + "6" * 64,
        reason="reviewed",
    )
    registry.record_promotion_approval(
        "candidate_gate",
        approved_by="reviewer_b",
        evidence_digest="sha256:" + "6" * 64,
        reason="reviewed",
    )

    update = _FakeUpdate(8096)
    context = _FakeContext()
    context.args = ["candidate_gate"]

    class _Source:
        model_dir = artifact

        def get_active_count(self) -> int:
            return 0

    class _Bridge:
        def get_active_count(self) -> int:
            return 0

    monkeypatch.setattr(telebot_main, "MODEL_REGISTRY", registry)
    monkeypatch.setattr(telebot_main, "_is_admin_user", lambda user_id: True)
    monkeypatch.setattr(telebot_main, "_get_v2_bridge", lambda: _Bridge())
    monkeypatch.setattr(telebot_main, "_get_signal_source_manager", lambda **kwargs: _Source())

    asyncio.run(telebot_main.model_promote(update, context))

    assert registry.get_active_version() is None
    assert update.message.replies
    assert "requires an evidence digest" in update.message.replies[-1]


def test_model_promote_admin_updates_active_pointer_when_evidence_digest_matches(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("MODEL_REGISTRY_REQUIRE_TWO_PERSON_APPROVAL", "1")
    registry = ModelRegistry(tmp_path / "registry")
    artifact = tmp_path / "candidate"
    artifact.mkdir(parents=True)

    from datetime import timezone

    import numpy as np
    import pandas as pd

    from quant_v2.models.trainer import save_model_bundle, train

    monkeypatch.setenv("QUANT_IMAGE", "registry.example/quant-bot@sha256:" + "a" * 64)
    idx = pd.date_range(end=datetime.now(timezone.utc), periods=180, freq="h", tz="UTC")
    X = pd.DataFrame(
        {
            "f1": np.linspace(-2.0, 2.0, len(idx)),
            "f2": np.sin(np.linspace(0.0, 8.0, len(idx))),
            "f3": np.cos(np.linspace(0.0, 6.0, len(idx))),
        },
        index=idx,
    )
    y = pd.Series((np.sin(np.linspace(0.0, 12.0, len(idx))) > 0.0).astype(int), index=idx)
    trained = train(X, y, horizon=4, calibration_frac=0.2)
    save_model_bundle(trained, artifact / "model_4m.pkl")
    registry.register_version("candidate_gate", artifact, metrics={"promotion_eligible": True})
    digest = "sha256:" + "6" * 64
    registry.record_promotion_approval(
        "candidate_gate",
        approved_by="reviewer_a",
        evidence_digest=digest,
        reason="reviewed",
    )
    registry.record_promotion_approval(
        "candidate_gate",
        approved_by="reviewer_b",
        evidence_digest=digest,
        reason="reviewed",
    )

    update = _FakeUpdate(8097)
    context = _FakeContext()
    context.args = ["candidate_gate", digest, "reviewed", "by", "ops"]

    class _Source:
        model_dir = artifact

        def get_active_count(self) -> int:
            return 0

    class _Bridge:
        def get_active_count(self) -> int:
            return 0

    monkeypatch.setattr(telebot_main, "MODEL_REGISTRY", registry)
    monkeypatch.setattr(telebot_main, "_is_admin_user", lambda user_id: True)
    monkeypatch.setattr(telebot_main, "_get_v2_bridge", lambda: _Bridge())
    monkeypatch.setattr(telebot_main, "_get_signal_source_manager", lambda **kwargs: _Source())

    asyncio.run(telebot_main.model_promote(update, context))

    active = registry.get_active_version()
    assert active is not None
    assert active.version_id == "candidate_gate"
    assert update.message.replies
    assert "Model Promotion Applied" in update.message.replies[-1]


def test_model_rollback_rejects_terminal_target(tmp_path, monkeypatch) -> None:
    registry = ModelRegistry(tmp_path / "registry")
    active_artifact = tmp_path / "active"
    active_artifact.mkdir(parents=True)

    from datetime import timezone

    import numpy as np
    import pandas as pd

    from quant_v2.models.trainer import save_model_bundle, train

    monkeypatch.setenv("QUANT_IMAGE", "registry.example/quant-bot@sha256:" + "a" * 64)
    idx = pd.date_range(end=datetime.now(timezone.utc), periods=180, freq="h", tz="UTC")
    X = pd.DataFrame(
        {
            "f1": np.linspace(-2.0, 2.0, len(idx)),
            "f2": np.sin(np.linspace(0.0, 8.0, len(idx))),
            "f3": np.cos(np.linspace(0.0, 6.0, len(idx))),
        },
        index=idx,
    )
    y = pd.Series((np.sin(np.linspace(0.0, 12.0, len(idx))) > 0.0).astype(int), index=idx)
    trained = train(X, y, horizon=4, calibration_frac=0.2)
    save_model_bundle(trained, active_artifact / "model_4m.pkl")
    registry.register_version("active_a", active_artifact, metrics={"promotion_eligible": True})
    registry.set_active_version("active_a")

    rejected_artifact = tmp_path / "rejected"
    rejected_artifact.mkdir(parents=True)
    registry.register_version("terminal_rejected", rejected_artifact, metrics={"promotion_eligible": True})
    registry.reject_version("terminal_rejected", notes="manual rejection")

    update = _FakeUpdate(8100)
    context = _FakeContext()
    context.args = ["terminal_rejected"]

    class _Source:
        model_dir = active_artifact

        def get_active_count(self) -> int:
            return 0

    class _Bridge:
        def get_active_count(self) -> int:
            return 0

    monkeypatch.setattr(telebot_main, "MODEL_REGISTRY", registry)
    monkeypatch.setattr(telebot_main, "_is_admin_user", lambda user_id: True)
    monkeypatch.setattr(telebot_main, "_get_v2_bridge", lambda: _Bridge())
    monkeypatch.setattr(telebot_main, "_get_signal_source_manager", lambda **kwargs: _Source())

    asyncio.run(telebot_main.model_rollback(update, context))

    active = registry.get_active_version()
    assert active is not None
    assert active.version_id == "active_a"
    assert update.message.replies
    assert "not reactivatable" in update.message.replies[-1]


def test_model_events_rejects_tampered_registry_history(tmp_path, monkeypatch) -> None:
    registry = ModelRegistry(tmp_path / "registry")
    artifact = tmp_path / "candidate"
    artifact.mkdir(parents=True)
    registry.register_version("candidate_a", artifact, metrics={"promotion_eligible": True})
    registry.set_active_version("candidate_a")

    events_path = registry.root / "registry_events.jsonl"
    events_path.write_text(
        events_path.read_text(encoding="utf-8").replace("candidate_a", "candidate_x", 1),
        encoding="utf-8",
    )

    update = _FakeUpdate(8095)
    context = _FakeContext()

    monkeypatch.setattr(telebot_main, "MODEL_REGISTRY", registry)
    monkeypatch.setattr(telebot_main, "_is_admin_user", lambda user_id: True)

    asyncio.run(telebot_main.model_events(update, context))

    assert update.message.replies
    assert "Unable to read registry events" in update.message.replies[-1]


def test_model_rollback_rejects_without_previous_active_target(tmp_path, monkeypatch) -> None:
    registry = ModelRegistry(tmp_path / "registry")
    active_artifact = tmp_path / "active"
    active_artifact.mkdir(parents=True)
    from datetime import timezone

    import numpy as np
    import pandas as pd

    from quant_v2.models.trainer import save_model_bundle, train

    monkeypatch.setenv("QUANT_IMAGE", "registry.example/quant-bot@sha256:" + "a" * 64)
    idx = pd.date_range(end=datetime.now(timezone.utc), periods=180, freq="h", tz="UTC")
    X = pd.DataFrame(
        {
            "f1": np.linspace(-2.0, 2.0, len(idx)),
            "f2": np.sin(np.linspace(0.0, 8.0, len(idx))),
            "f3": np.cos(np.linspace(0.0, 6.0, len(idx))),
        },
        index=idx,
    )
    y = pd.Series((np.sin(np.linspace(0.0, 12.0, len(idx))) > 0.0).astype(int), index=idx)
    trained = train(X, y, horizon=4, calibration_frac=0.2)
    save_model_bundle(trained, active_artifact / "model_4m.pkl")
    registry.register_version(
        "active_a",
        active_artifact,
        metrics={"promotion_eligible": True},
    )
    registry.set_active_version("active_a")

    target_artifact = tmp_path / "target"
    target_artifact.mkdir(parents=True)
    save_model_bundle(trained, target_artifact / "model_4m.pkl")
    registry.register_version(
        "candidate_bad",
        target_artifact,
        metrics={"promotion_eligible": True},
    )

    update = _FakeUpdate(8094)
    context = _FakeContext()
    context.args = ["candidate_bad"]

    class _Source:
        model_dir = active_artifact

        def get_active_count(self) -> int:
            return 0

    class _Bridge:
        def get_active_count(self) -> int:
            return 0

    monkeypatch.setattr(telebot_main, "MODEL_REGISTRY", registry)
    monkeypatch.setattr(telebot_main, "_is_admin_user", lambda user_id: True)
    monkeypatch.setattr(telebot_main, "_get_v2_bridge", lambda: _Bridge())
    monkeypatch.setattr(telebot_main, "_get_signal_source_manager", lambda **kwargs: _Source())

    asyncio.run(telebot_main.model_rollback(update, context))

    active = registry.get_active_version()
    assert active is not None
    assert active.version_id == "active_a"
    assert update.message.replies
    assert "No previous active version available for rollback" in update.message.replies[-1]


def test_model_rollback_rejects_ready_but_non_previous_target(tmp_path, monkeypatch) -> None:
    registry = ModelRegistry(tmp_path / "registry")
    active_artifact = tmp_path / "active"
    active_artifact.mkdir(parents=True)
    previous_artifact = tmp_path / "previous"
    previous_artifact.mkdir(parents=True)
    target_artifact = tmp_path / "target"
    target_artifact.mkdir(parents=True)

    from datetime import timezone

    import numpy as np
    import pandas as pd

    from quant_v2.models.trainer import save_model_bundle, train

    monkeypatch.setenv("QUANT_IMAGE", "registry.example/quant-bot@sha256:" + "a" * 64)
    idx = pd.date_range(end=datetime.now(timezone.utc), periods=180, freq="h", tz="UTC")
    X = pd.DataFrame(
        {
            "f1": np.linspace(-2.0, 2.0, len(idx)),
            "f2": np.sin(np.linspace(0.0, 8.0, len(idx))),
            "f3": np.cos(np.linspace(0.0, 6.0, len(idx))),
        },
        index=idx,
    )
    y = pd.Series((np.sin(np.linspace(0.0, 12.0, len(idx))) > 0.0).astype(int), index=idx)
    trained = train(X, y, horizon=4, calibration_frac=0.2)
    save_model_bundle(trained, active_artifact / "model_4m.pkl")
    save_model_bundle(trained, previous_artifact / "model_4m.pkl")
    save_model_bundle(trained, target_artifact / "model_4m.pkl")
    registry.register_version(
        "active_a",
        active_artifact,
        metrics={"promotion_eligible": True},
    )
    registry.register_version(
        "previous_b",
        previous_artifact,
        metrics={"promotion_eligible": True},
    )
    registry.register_version(
        "candidate_c",
        target_artifact,
        metrics={"promotion_eligible": True},
    )
    registry.set_active_version("active_a")
    registry.set_active_version("previous_b")

    update = _FakeUpdate(8093)
    context = _FakeContext()
    context.args = ["candidate_c"]

    class _Source:
        model_dir = previous_artifact

        def get_active_count(self) -> int:
            return 0

    class _Bridge:
        def get_active_count(self) -> int:
            return 0

    monkeypatch.setattr(telebot_main, "MODEL_REGISTRY", registry)
    monkeypatch.setattr(telebot_main, "_is_admin_user", lambda user_id: True)
    monkeypatch.setattr(telebot_main, "_get_v2_bridge", lambda: _Bridge())
    monkeypatch.setattr(telebot_main, "_get_signal_source_manager", lambda **kwargs: _Source())

    asyncio.run(telebot_main.model_rollback(update, context))

    active = registry.get_active_version()
    assert active is not None
    assert active.version_id == "previous_b"
    assert update.message.replies
    assert "previous active version" in update.message.replies[-1]


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
                routed_signals_total=12,
                routed_buy_signals=5,
                routed_sell_signals=4,
                routed_actionable_signals=9,
                effective_symbol_cap_frac=0.05,
                effective_gross_cap_frac=0.15,
                effective_net_cap_frac=0.10,
            )

        def get_kill_switch_evaluation(self, user_id: int):
            _ = user_id
            return KillSwitchEvaluation(pause_trading=False)

    text = telebot_main._build_execution_diagnostics_text(_DiagBridge(), 808)
    assert "Execution Telemetry (session cumulative):" in text
    assert "Routed signals: 12" in text
    assert "BUY=5, SELL=4, actionable=9" in text
    assert "Kill-switch blocks: cycles=3, actionable_blocked=7" in text
    assert "Orders attempted: 5" in text
    assert "Order breakdown: entries=2, rebalances=2, exits=0" in text
    assert "Skipped: filter=1, deadband=2" in text
    assert "Effective caps" in text
