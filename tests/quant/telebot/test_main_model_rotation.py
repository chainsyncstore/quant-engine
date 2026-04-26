"""Integration tests for model rotation persistence wiring.

Refs: audit_20260423 task P3-1
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from quant.telebot.models import Base, User, UserContext
from quant.telebot import main as telebot_main


@pytest.fixture
def temp_db():
    """Create an in-memory SQLite database for testing."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    yield Session


def test_model_rotation_persists_to_user_context(temp_db: Any) -> None:
    """Full wiring: hook fires -> _persist_user_session_flags updates DB."""
    Session = temp_db

    # Create a user with active session
    session = Session()
    user = User(telegram_id=12345, username="testuser")
    user.context = UserContext(
        telegram_id=12345,
        is_active=True,
        live_mode=False,
        active_model_version="model_old",
        active_model_source="registry_active:model_old",
    )
    session.add(user)
    session.commit()
    session.close()

    # Create a fake V2_SIGNAL_MANAGER with session for user 12345
    fake_manager = MagicMock()
    fake_manager.sessions = {12345: MagicMock()}

    # Patch the main module's SessionLocal to use our test DB
    with patch.object(telebot_main, "V2_SIGNAL_MANAGER", fake_manager):
        with patch.object(telebot_main, "SessionLocal", Session):
            # Simulate the _on_v2_model_rotated callback being invoked
            telebot_main._on_v2_model_rotated("model_20260421_192947", "registry_active:model_20260421_192947")

    # Verify the DB was updated
    session = Session()
    db_user = session.query(User).filter_by(telegram_id=12345).first()
    assert db_user is not None
    assert db_user.context is not None
    assert db_user.context.active_model_version == "model_20260421_192947"
    assert db_user.context.active_model_source == "registry_active:model_20260421_192947"
    session.close()


def test_model_rotation_handles_missing_user_gracefully(temp_db: Any) -> None:
    """Hook should not fail when user_id doesn't exist in DB."""
    Session = temp_db

    # No users in DB
    with patch.object(telebot_main, "SessionLocal", Session):
        # Should not raise
        telebot_main._on_v2_model_rotated("model_20260421_192947", "registry_active:model_20260421_192947")


def test_model_rotation_persists_for_multiple_users(temp_db: Any) -> None:
    """Hook updates all active sessions when model rotates."""
    Session = temp_db

    # Create multiple users with active sessions
    session = Session()
    for uid in [111, 222, 333]:
        user = User(telegram_id=uid, username=f"user{uid}")
        user.context = UserContext(
            telegram_id=uid,
            is_active=True,
            live_mode=False,
            active_model_version="model_old",
            active_model_source="registry_active:model_old",
        )
        session.add(user)
    session.commit()
    session.close()

    # Create a fake V2_SIGNAL_MANAGER with sessions
    fake_manager = MagicMock()
    fake_manager.sessions = {111: MagicMock(), 222: MagicMock(), 333: MagicMock()}

    with patch.object(telebot_main, "V2_SIGNAL_MANAGER", fake_manager):
        with patch.object(telebot_main, "SessionLocal", Session):
            telebot_main._on_v2_model_rotated("model_new", "registry_active:model_new")

    # Verify all users were updated
    session = Session()
    for uid in [111, 222, 333]:
        db_user = session.query(User).filter_by(telegram_id=uid).first()
        assert db_user is not None
        assert db_user.context is not None
        assert db_user.context.active_model_version == "model_new"
        assert db_user.context.active_model_source == "registry_active:model_new"
    session.close()


def test_model_rotation_skips_when_manager_none() -> None:
    """Hook returns early when V2_SIGNAL_MANAGER is None."""
    with patch.object(telebot_main, "V2_SIGNAL_MANAGER", None):
        # Should not raise
        telebot_main._on_v2_model_rotated("model_new", "registry_active:model_new")
