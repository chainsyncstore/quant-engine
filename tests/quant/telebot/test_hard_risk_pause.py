from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from quant.telebot import main as telebot_main
from quant.telebot.models import Base, ExecutionRouteEvent, User, UserContext
from quant_v2.execution.adapters import ExecutionResult
from quant_v2.execution.service import HardRiskPauseEvent, RouteAuditEvent


@pytest.fixture
def temp_db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    yield Session


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
        self.bot = SimpleNamespace()


class _FakeApp:
    def __init__(self) -> None:
        self.sent: list[tuple[int, str]] = []
        self.bot = SimpleNamespace(send_message=self._send_message)

    async def _send_message(self, *, chat_id: int, text: str) -> None:
        self.sent.append((chat_id, text))


class _RestoreSource:
    def __init__(self) -> None:
        self.running: set[int] = set()
        self.start_calls: list[int] = []
        self.stop_calls: list[int] = []

    async def start_session(self, user_id: int, creds, *, on_signal, execute_orders: bool) -> bool:
        _ = creds, on_signal, execute_orders
        self.start_calls.append(user_id)
        self.running.add(user_id)
        return True

    async def stop_session(self, user_id: int) -> bool:
        self.stop_calls.append(user_id)
        self.running.discard(user_id)
        return True

    def is_running(self, user_id: int) -> bool:
        return user_id in self.running


class _RestoreBridge:
    def __init__(self) -> None:
        self.running: set[int] = set()
        self.start_calls: list[int] = []
        self.stop_calls: list[int] = []
        self.sync_calls: list[int] = []

    async def start_session(self, user_id: int, *, live: bool, credentials) -> bool:
        _ = live, credentials
        self.start_calls.append(user_id)
        self.running.add(user_id)
        return True

    async def stop_session(self, user_id: int) -> bool:
        self.stop_calls.append(user_id)
        self.running.discard(user_id)
        return True

    def is_running(self, user_id: int) -> bool:
        return user_id in self.running

    def get_session_mode(self, user_id: int) -> str:
        _ = user_id
        return "paper"

    def get_active_count(self) -> int:
        return len(self.running)

    async def sync_positions(self, user_id: int, *, target_positions, prices=None):
        _ = target_positions, prices
        self.sync_calls.append(user_id)
        return (
            ExecutionResult(
                accepted=True,
                order_id="restore-1",
                idempotency_key="restore-1",
                symbol="BTCUSDT",
                side="BUY",
                requested_qty=0.0,
                filled_qty=0.0,
                avg_price=0.0,
                status="accepted",
                created_at="2026-05-28T12:00:00+00:00",
            ),
        )

    def clear_execution_diagnostics(self, user_id: int) -> bool:
        _ = user_id
        return True


class _RoutingBridge(_RestoreBridge):
    def __init__(self) -> None:
        super().__init__()
        self.route_calls: list[int] = []
        self.monitoring_calls: list[int] = []
        self.price_calls: list[tuple[int, dict[str, float]]] = []

    def set_monitoring_snapshot(self, user_id: int, snapshot) -> None:
        _ = snapshot
        self.monitoring_calls.append(user_id)

    def ingest_market_prices(self, user_id: int, prices: dict[str, float]) -> None:
        self.price_calls.append((user_id, dict(prices)))

    async def route_signals(self, user_id: int, *, signals, prices, monitoring_snapshot=None):
        _ = signals, prices, monitoring_snapshot
        self.route_calls.append(user_id)
        return ()


def _add_user(
    Session,
    user_id: int,
    *,
    is_active: bool,
    hard_paused: bool = False,
    maintenance_pending: bool = False,
) -> None:
    session = Session()
    user = User(telegram_id=user_id, username=f"user{user_id}", status="active")
    user.context = UserContext(
        telegram_id=user_id,
        is_active=is_active,
        live_mode=False,
        hard_risk_paused=hard_paused,
        hard_risk_pause_reason="hard_risk_breach" if hard_paused else None,
        hard_risk_pause_triggered_at=datetime(2026, 5, 28, 12, 0, 0),
        hard_risk_pause_breach_type="external_monitoring" if hard_paused else None,
        hard_risk_pause_details='{"mode":"paper"}' if hard_paused else None,
        maintenance_resume_pending=maintenance_pending,
        maintenance_resume_payload=(
            '{"captured_at":"2026-05-28T12:00:00+00:00","mode":"demo",'
            '"positions":{"BTCUSDT":0.01},"prices":{"BTCUSDT":50000.0}}'
            if maintenance_pending
            else None
        ),
    )
    session.add(user)
    session.commit()
    session.close()


def _set_paper_state(Session, user_id: int, paper_state: dict) -> None:
    session = Session()
    db_user = session.query(User).filter_by(telegram_id=user_id).first()
    assert db_user is not None and db_user.context is not None
    db_user.context.paper_state_json = telebot_main.json.dumps(paper_state)
    db_user.context.current_demo_notional_usd = 0.0
    db_user.context.current_demo_symbols = 0
    session.commit()
    session.close()


def test_persist_hard_risk_pause_event_writes_user_context(temp_db) -> None:
    Session = temp_db
    _add_user(Session, 9001, is_active=True)

    event = HardRiskPauseEvent(
        user_id=9001,
        reason="hard_risk_breach",
        triggered_at=datetime(2026, 5, 28, 13, 30, tzinfo=timezone.utc),
        breach_type="portfolio_risk_policy",
        details={"mode": "paper", "risk": {"gross_exposure_frac": 1.25}},
    )

    with patch.object(telebot_main, "SessionLocal", Session):
        telebot_main._persist_hard_risk_pause_event(event)

    session = Session()
    db_user = session.query(User).filter_by(telegram_id=9001).first()
    assert db_user is not None
    assert db_user.context is not None
    assert db_user.context.hard_risk_paused is True
    assert db_user.context.hard_risk_pause_reason == "hard_risk_breach"
    assert db_user.context.hard_risk_pause_breach_type == "portfolio_risk_policy"
    assert db_user.context.is_active is False
    assert "gross_exposure_frac" in (db_user.context.hard_risk_pause_details or "")
    session.close()


def test_persist_route_audit_event_writes_execution_route_event(temp_db) -> None:
    Session = temp_db

    event = RouteAuditEvent(
        user_id=9009,
        created_at=datetime(2026, 5, 28, 13, 35, tzinfo=timezone.utc),
        pause_state="none",
        is_active=True,
        live_mode=False,
        symbol="BTCUSDT",
        side="BUY",
        quantity=0.1,
        before_position=0.0,
        after_position=0.1,
        action_class="entry",
        reason="order_accepted",
        accepted=True,
        status="filled",
        order_id="paper-1",
        idempotency_key="abc123",
        mark_price=50_000.0,
    )

    with patch.object(telebot_main, "SessionLocal", Session):
        telebot_main._persist_route_audit_event(event)

    session = Session()
    row = session.query(ExecutionRouteEvent).filter_by(telegram_id=9009).first()
    assert row is not None
    assert row.symbol == "BTCUSDT"
    assert row.side == "BUY"
    assert row.accepted is True
    assert row.action_class == "entry"
    assert row.reason == "order_accepted"
    assert row.mark_price == pytest.approx(50_000.0)
    session.close()


def test_route_audit_event_evaluates_prior_blocked_shadow_return(temp_db) -> None:
    Session = temp_db
    blocked = RouteAuditEvent(
        user_id=9015,
        created_at=datetime(2026, 5, 28, 13, 35, tzinfo=timezone.utc),
        pause_state="none",
        is_active=True,
        live_mode=False,
        symbol="BTCUSDT",
        side="BUY",
        quantity=0.1,
        before_position=0.0,
        after_position=0.0,
        action_class="blocked",
        reason="skipped_by_deadband:cooldown",
        accepted=False,
        status="skipped",
        mark_price=100.0,
    )
    later_mark = RouteAuditEvent(
        user_id=9015,
        created_at=datetime(2026, 5, 28, 14, 35, tzinfo=timezone.utc),
        pause_state="none",
        is_active=True,
        live_mode=False,
        symbol="BTCUSDT",
        side="BUY",
        quantity=0.1,
        before_position=0.0,
        after_position=0.1,
        action_class="entry",
        reason="order_accepted",
        accepted=True,
        status="filled",
        mark_price=103.0,
    )

    with patch.object(telebot_main, "SessionLocal", Session):
        telebot_main._persist_route_audit_event(blocked)
        telebot_main._persist_route_audit_event(later_mark)

    session = Session()
    row = (
        session.query(ExecutionRouteEvent)
        .filter_by(telegram_id=9015, action_class="blocked")
        .first()
    )
    assert row is not None
    assert row.future_mark_price == pytest.approx(103.0)
    assert row.future_return_bps == pytest.approx(300.0)
    assert row.shadow_evaluated_at is not None
    session.close()


def test_startup_health_report_surfaces_hard_pause_and_route_shadow_counts(
    temp_db,
    monkeypatch,
) -> None:
    Session = temp_db
    _add_user(Session, 9016, is_active=True, hard_paused=True)
    event = RouteAuditEvent(
        user_id=9016,
        created_at=datetime(2026, 5, 28, 13, 35, tzinfo=timezone.utc),
        pause_state="none",
        is_active=True,
        live_mode=False,
        symbol="BTCUSDT",
        side="BUY",
        quantity=0.1,
        before_position=0.0,
        after_position=0.0,
        action_class="blocked",
        reason="skipped_by_deadband:cooldown",
        accepted=False,
        status="skipped",
        mark_price=100.0,
    )

    monkeypatch.setattr(telebot_main, "SessionLocal", Session)
    telebot_main._persist_route_audit_event(event)

    report = telebot_main._build_startup_health_report()

    assert "Startup Health Report:" in report
    assert "hard_risk_paused=1" in report
    assert "route_events=1" in report
    assert "hard_risk_paused_users=1" in report


def test_persist_hard_risk_pause_event_stops_running_sessions(temp_db, monkeypatch) -> None:
    Session = temp_db
    _add_user(Session, 9010, is_active=True)
    source = _RestoreSource()
    bridge = _RestoreBridge()
    source.running.add(9010)
    bridge.running.add(9010)

    monkeypatch.setattr(telebot_main, "SessionLocal", Session)
    monkeypatch.setattr(telebot_main, "V2_SIGNAL_MANAGER", source)
    monkeypatch.setattr(telebot_main, "V2_BRIDGE", bridge)
    monkeypatch.setattr(telebot_main, "_using_v2_primary_backend", lambda: True)
    monkeypatch.setattr(telebot_main, "_using_manager_signal_source", lambda: False)

    event = HardRiskPauseEvent(
        user_id=9010,
        reason="hard_risk_breach",
        triggered_at=datetime(2026, 5, 28, 13, 30, tzinfo=timezone.utc),
        breach_type="portfolio_risk_policy",
        details={"mode": "paper"},
    )

    telebot_main._persist_hard_risk_pause_event(event)

    assert source.stop_calls == [9010]
    assert bridge.stop_calls == [9010]
    assert not source.is_running(9010)
    assert not bridge.is_running(9010)


def test_persisted_hard_risk_pause_blocks_routing_with_live_session(temp_db, monkeypatch) -> None:
    Session = temp_db
    _add_user(Session, 9011, is_active=True, hard_paused=True)
    bridge = _RoutingBridge()
    bridge.running.add(9011)

    monkeypatch.setattr(telebot_main, "SessionLocal", Session)
    monkeypatch.setattr(telebot_main, "_using_v2_backend", lambda: True)
    monkeypatch.setattr(telebot_main, "_using_v2_primary_backend", lambda: True)
    monkeypatch.setattr(telebot_main, "_get_v2_bridge", lambda: bridge)

    notifier = telebot_main._build_signal_notifier(SimpleNamespace(), 9011)
    signal = telebot_main.StrategySignal(
        symbol="BTCUSDT",
        timeframe="1h",
        horizon_bars=4,
        signal="BUY",
        confidence=0.90,
    )

    asyncio.run(
        notifier(
            {
                "signal": "BUY",
                "symbol": "BTCUSDT",
                "close_price": 50_000.0,
                "v2_signal": signal,
                "v2_prices": {"BTCUSDT": 50_000.0},
            }
        )
    )

    assert bridge.route_calls == []


def test_hold_signal_routes_quietly_for_reconciliation(temp_db, monkeypatch) -> None:
    Session = temp_db
    _add_user(Session, 9014, is_active=True, hard_paused=False)
    bridge = _RoutingBridge()
    bridge.running.add(9014)
    app = _FakeApp()

    monkeypatch.setattr(telebot_main, "SessionLocal", Session)
    monkeypatch.setattr(telebot_main, "_using_v2_backend", lambda: True)
    monkeypatch.setattr(telebot_main, "_using_v2_primary_backend", lambda: True)
    monkeypatch.setattr(telebot_main, "_get_v2_bridge", lambda: bridge)

    notifier = telebot_main._build_signal_notifier(app, 9014)
    signal = telebot_main.StrategySignal(
        symbol="BTCUSDT",
        timeframe="1h",
        horizon_bars=4,
        signal="HOLD",
        confidence=0.50,
    )

    asyncio.run(
        notifier(
            {
                "signal": "HOLD",
                "symbol": "BTCUSDT",
                "close_price": 50_000.0,
                "v2_signal": signal,
                "v2_prices": {"BTCUSDT": 50_000.0},
            }
        )
    )

    assert bridge.route_calls == [9014]
    assert app.sent == []


def test_inactive_user_blocks_routing_with_live_session(temp_db, monkeypatch) -> None:
    Session = temp_db
    _add_user(Session, 9012, is_active=False, hard_paused=False)
    bridge = _RoutingBridge()
    bridge.running.add(9012)

    monkeypatch.setattr(telebot_main, "SessionLocal", Session)
    monkeypatch.setattr(telebot_main, "_using_v2_backend", lambda: True)
    monkeypatch.setattr(telebot_main, "_using_v2_primary_backend", lambda: True)
    monkeypatch.setattr(telebot_main, "_get_v2_bridge", lambda: bridge)

    notifier = telebot_main._build_signal_notifier(SimpleNamespace(), 9012)
    signal = telebot_main.StrategySignal(
        symbol="ETHUSDT",
        timeframe="1h",
        horizon_bars=4,
        signal="SELL",
        confidence=0.90,
    )

    asyncio.run(
        notifier(
            {
                "signal": "SELL",
                "symbol": "ETHUSDT",
                "close_price": 2_000.0,
                "v2_signal": signal,
                "v2_prices": {"ETHUSDT": 2_000.0},
            }
        )
    )

    assert bridge.route_calls == []


def test_lifetime_stats_reconciles_paper_state_positions(temp_db, monkeypatch) -> None:
    Session = temp_db
    _add_user(Session, 9013, is_active=False)
    _set_paper_state(
        Session,
        9013,
        {
            "equity_usd": 10_050.0,
            "open_positions": {"BTCUSDT": 0.1, "ETHUSDT": -2.0},
            "paper_entry_prices": {"BTCUSDT": 50_000.0, "ETHUSDT": 2_000.0},
        },
    )

    monkeypatch.setattr(telebot_main, "SessionLocal", Session)

    summary = telebot_main._load_lifetime_stats_summary(9013)

    assert summary is not None
    assert summary["current_demo_symbols"] == 2
    assert summary["current_demo_notional_usd"] == pytest.approx(9_000.0)


def test_restore_active_sessions_skips_paused_user_and_restores_unpaused(temp_db, monkeypatch) -> None:
    Session = temp_db
    _add_user(Session, 9002, is_active=True, hard_paused=True)
    _add_user(Session, 9003, is_active=True, hard_paused=False)

    source = _RestoreSource()
    bridge = _RestoreBridge()
    app = _FakeApp()

    monkeypatch.setattr(telebot_main, "SessionLocal", Session)
    monkeypatch.setattr(telebot_main, "_using_v2_primary_backend", lambda: True)
    monkeypatch.setattr(telebot_main, "_using_v2_backend", lambda: True)
    monkeypatch.setattr(telebot_main, "_get_signal_source_manager", lambda *args, **kwargs: source)
    monkeypatch.setattr(telebot_main, "_get_v2_bridge", lambda: bridge)
    monkeypatch.setattr(
        telebot_main,
        "_resolve_runtime_metadata",
        lambda *, bridge: ("core_v2", "model_a", "registry_active:model_a"),
    )
    monkeypatch.setattr(telebot_main, "_apply_saved_lifecycle_preferences", lambda user_id, bridge: None)
    monkeypatch.setattr(telebot_main, "_load_paper_state", lambda user_id: None)
    monkeypatch.setattr(telebot_main, "_reset_last_equity_anchor", lambda user_id, *, equity, live: None)

    asyncio.run(telebot_main._restore_active_sessions(app))

    assert bridge.start_calls == [9003]
    assert source.start_calls == [9003]
    assert app.sent and app.sent[0][0] == 9002
    assert "Trading Paused: Hard Risk Breach" in app.sent[0][1]

    session = Session()
    paused_user = session.query(User).filter_by(telegram_id=9002).first()
    resumed_user = session.query(User).filter_by(telegram_id=9003).first()
    assert paused_user.context.is_active is False
    assert resumed_user.context.is_active is True
    session.close()


def test_continue_from_maintenance_blocks_active_hard_risk_pause(temp_db, monkeypatch) -> None:
    Session = temp_db
    _add_user(Session, 9004, is_active=False, hard_paused=True, maintenance_pending=True)
    bridge = _RestoreBridge()
    update = _FakeUpdate(9004)

    monkeypatch.setattr(telebot_main, "SessionLocal", Session)
    monkeypatch.setattr(telebot_main, "_get_v2_bridge", lambda: bridge)
    monkeypatch.setattr(telebot_main, "_get_signal_source_manager", lambda *args, **kwargs: None)
    monkeypatch.setattr(telebot_main, "_using_v2_primary_backend", lambda: False)

    asyncio.run(telebot_main._continue_from_maintenance(update, _FakeContext(), live=False))

    assert bridge.start_calls == []
    assert update.message.replies
    assert "Trading Paused: Hard Risk Breach" in update.message.replies[-1]


def test_status_surfaces_persisted_hard_risk_pause(temp_db, monkeypatch) -> None:
    Session = temp_db
    _add_user(Session, 9006, is_active=False, hard_paused=True)
    bridge = _RestoreBridge()
    update = _FakeUpdate(9006)

    monkeypatch.setattr(telebot_main, "SessionLocal", Session)
    monkeypatch.setattr(telebot_main, "_get_v2_bridge", lambda: bridge)
    monkeypatch.setattr(telebot_main, "_get_signal_source_manager", lambda *args, **kwargs: None)
    monkeypatch.setattr(telebot_main, "_using_v2_primary_backend", lambda: False)

    asyncio.run(telebot_main.status(update, _FakeContext()))

    assert update.message.replies
    msg = update.message.replies[-1]
    assert "Trading Paused: Hard Risk Breach" in msg
    assert "external_monitoring" in msg
    assert "Runtime engine: `STOPPED`" in msg


def test_continue_from_maintenance_allows_manual_db_clear(temp_db, monkeypatch) -> None:
    Session = temp_db
    _add_user(Session, 9005, is_active=False, hard_paused=False, maintenance_pending=True)
    bridge = _RestoreBridge()
    update = _FakeUpdate(9005)

    monkeypatch.setattr(telebot_main, "SessionLocal", Session)
    monkeypatch.setattr(telebot_main, "_get_v2_bridge", lambda: bridge)
    monkeypatch.setattr(telebot_main, "_get_signal_source_manager", lambda *args, **kwargs: None)
    monkeypatch.setattr(telebot_main, "_using_v2_primary_backend", lambda: False)
    monkeypatch.setattr(
        telebot_main,
        "_resolve_runtime_metadata",
        lambda *, bridge: ("core_v2", "model_a", "registry_active:model_a"),
    )
    monkeypatch.setattr(telebot_main, "_apply_saved_lifecycle_preferences", lambda user_id, bridge: None)

    asyncio.run(telebot_main._continue_from_maintenance(update, _FakeContext(), live=False))

    assert bridge.start_calls == [9005]
    assert bridge.sync_calls == [9005]
    assert update.message.replies
    assert "Maintenance Continue Complete" in update.message.replies[-1]
