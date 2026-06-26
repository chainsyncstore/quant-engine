from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from quant.telebot import main as telebot_main
from quant.telebot.models import Base, User, UserContext
from quant.telebot.redaction import redact, safe_exception_message


class _FakeMessage:
    def __init__(self) -> None:
        self.replies: list[str] = []

    async def reply_text(self, text: str) -> None:
        self.replies.append(text)


class _FakeUpdate:
    def __init__(self, user_id: int) -> None:
        self.effective_user = SimpleNamespace(id=user_id)
        self.message = _FakeMessage()


class _FakeBot:
    def __init__(self) -> None:
        self.sent: list[tuple[int, str]] = []

    async def send_message(self, chat_id: int, text: str) -> None:
        self.sent.append((int(chat_id), text))


class _FakeContext:
    def __init__(self, args: list[str] | None = None) -> None:
        self.args = args or []
        self.bot = _FakeBot()


def _session_factory():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)


def test_redaction_removes_secret_like_values() -> None:
    raw = "api_key=ABCDEF1234567890ABCDEF1234567890 token=12345:abcdefghijklmnopqrstuvwxyz"

    cleaned = redact(raw)

    assert "ABCDEF1234567890ABCDEF1234567890" not in cleaned
    assert "12345:abcdefghijklmnopqrstuvwxyz" not in cleaned
    assert "[REDACTED" in cleaned
    assert "RuntimeError" in safe_exception_message(RuntimeError(raw))
    assert "ABCDEF1234567890ABCDEF1234567890" not in safe_exception_message(RuntimeError(raw))


def test_http_client_logs_do_not_emit_token_bearing_urls() -> None:
    assert logging.getLogger("httpx").level >= logging.WARNING
    assert logging.getLogger("httpcore").level >= logging.WARNING


def test_setup_rejects_pending_and_banned_users_without_persisting(monkeypatch) -> None:
    Session = _session_factory()
    db = Session()
    pending = User(telegram_id=1001, username="pending", status="pending")
    pending.context = UserContext(telegram_id=1001)
    banned = User(telegram_id=1002, username="banned", status="banned")
    banned.context = UserContext(telegram_id=1002)
    db.add_all([pending, banned])
    db.commit()
    db.close()

    monkeypatch.setattr(telebot_main, "SessionLocal", Session)

    for user_id in (1001, 1002):
        update = _FakeUpdate(user_id)
        context = _FakeContext(args=["SENTINEL_API_KEY", "SENTINEL_SECRET"])
        asyncio.run(telebot_main.setup(update, context))
        assert "Account not approved" in update.message.replies[-1]

    db = Session()
    for user_id in (1001, 1002):
        ctx = db.query(UserContext).filter_by(telegram_id=user_id).first()
        assert ctx is not None
        assert ctx.binance_api_key is None
        assert ctx.binance_api_secret is None
    db.close()


def test_setup_allows_active_user_and_encrypts_credentials(monkeypatch) -> None:
    Session = _session_factory()
    db = Session()
    user = User(telegram_id=1003, username="active", status="active")
    user.context = UserContext(telegram_id=1003)
    db.add(user)
    db.commit()
    db.close()

    monkeypatch.setattr(telebot_main, "SessionLocal", Session)
    update = _FakeUpdate(1003)
    context = _FakeContext(args=["SENTINEL_API_KEY", "SENTINEL_SECRET"])

    asyncio.run(telebot_main.setup(update, context))

    db = Session()
    ctx = db.query(UserContext).filter_by(telegram_id=1003).first()
    assert ctx is not None
    assert ctx.binance_api_key
    assert ctx.binance_api_secret
    assert "SENTINEL_API_KEY" not in ctx.binance_api_key
    assert "SENTINEL_SECRET" not in ctx.binance_api_secret
    db.close()


def test_revoke_clears_credentials_and_live_flags(monkeypatch) -> None:
    Session = _session_factory()
    db = Session()
    user = User(telegram_id=1004, username="active", status="active")
    user.context = UserContext(
        telegram_id=1004,
        binance_api_key="encrypted-key",
        binance_api_secret="encrypted-secret",
        live_mode=True,
        is_active=True,
    )
    db.add(user)
    db.commit()
    db.close()

    monkeypatch.setenv("ADMIN_ID", "999")
    monkeypatch.setattr(telebot_main, "SessionLocal", Session)
    monkeypatch.setattr(telebot_main, "_get_signal_source_manager", lambda *args, **kwargs: None)
    monkeypatch.setattr(telebot_main, "_get_v2_bridge", lambda: None)

    update = _FakeUpdate(999)
    context = _FakeContext(args=["1004"])
    asyncio.run(telebot_main.revoke(update, context))

    db = Session()
    revoked = db.query(User).filter_by(telegram_id=1004).first()
    assert revoked is not None
    assert revoked.status == "banned"
    assert revoked.context.binance_api_key is None
    assert revoked.context.binance_api_secret is None
    assert revoked.context.live_mode is False
    assert revoked.context.is_active is False
    db.close()


def test_main_does_not_register_raw_debug_handler_by_default(monkeypatch) -> None:
    class FakeApplication:
        def __init__(self) -> None:
            self.handlers: list[object] = []

        def add_handler(self, handler, *args, **kwargs) -> None:
            _ = (args, kwargs)
            self.handlers.append(handler)

        def add_error_handler(self, handler) -> None:
            self.error_handler = handler

        def run_polling(self, *args, **kwargs) -> None:
            _ = (args, kwargs)

    class FakeBuilder:
        def __init__(self) -> None:
            self.app = FakeApplication()

        def token(self, token: str) -> "FakeBuilder":
            return self

        def post_init(self, callback) -> "FakeBuilder":
            self.post_init_callback = callback
            return self

        def build(self) -> FakeApplication:
            return self.app

    builder = FakeBuilder()
    monkeypatch.setenv("TELEGRAM_TOKEN", "test-token")
    monkeypatch.setenv("RETRAIN_ENABLED", "0")
    monkeypatch.delenv("BOT_TELEGRAM_DEBUG_UPDATES", raising=False)
    monkeypatch.setattr(telebot_main, "ApplicationBuilder", lambda: builder)

    telebot_main.main()

    handler_names = {handler.__class__.__name__ for handler in builder.app.handlers}
    assert "MessageHandler" not in handler_names
