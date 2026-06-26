from __future__ import annotations

from sqlalchemy import create_engine, text

from quant.telebot import main as telebot_main
from quant.telebot.models import ExecutionRouteEvent


def test_execution_route_event_schema_backfills_current_model_columns(tmp_path) -> None:
    db_path = tmp_path / "stale-route-events.db"
    engine = create_engine(f"sqlite:///{db_path}")
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                CREATE TABLE execution_route_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    telegram_id INTEGER NOT NULL,
                    created_at DATETIME,
                    symbol VARCHAR,
                    action_class VARCHAR
                )
                """
            )
        )

    previous_engine = telebot_main.ENGINE
    telebot_main.ENGINE = engine
    try:
        telebot_main._ensure_execution_route_event_schema()
        with engine.connect() as conn:
            existing = {
                str(row[1])
                for row in conn.execute(text("PRAGMA table_info(execution_route_events)"))
            }
    finally:
        telebot_main.ENGINE = previous_engine
        engine.dispose()

    expected = {
        column.name
        for column in ExecutionRouteEvent.__table__.columns
        if not column.primary_key
    }
    assert expected <= existing
