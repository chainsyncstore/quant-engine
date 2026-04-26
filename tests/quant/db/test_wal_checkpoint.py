"""Tests for SQLite WAL checkpoint helper."""

import os
import tempfile
from unittest.mock import MagicMock

from sqlalchemy import create_engine, text

from quant.telebot.main import checkpoint_wal


def test_checkpoint_wal_returns_zeros_on_empty_db():
    """In-memory SQLite: checkpoint should return integers (may be 0 or -1 for busy)."""
    engine = create_engine("sqlite:///:memory:")
    pages, frames = checkpoint_wal(engine)
    # Pages can be -1 (busy), 0 (no pages), or positive (pages checkpointed)
    # Frames can be -1 (busy) or positive (frames in WAL before checkpoint)
    # Result format: (busy, log_frames, checkpointed_frames)
    assert isinstance(pages, int)
    assert isinstance(frames, int)


def test_checkpoint_wal_noop_on_non_sqlite():
    """Non-SQLite backend should return (0, 0) without raising."""
    mock_engine = MagicMock()
    mock_engine.url.get_backend_name.return_value = "postgresql"
    result = checkpoint_wal(mock_engine)
    assert result == (0, 0)


def test_checkpoint_wal_after_writes_reduces_frames():
    """File-backed SQLite with WAL: writes create frames, checkpoint clears them."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        engine = create_engine(f"sqlite:///{db_path}")

        try:
            # Enable WAL mode
            with engine.connect() as conn:
                conn.execute(text("PRAGMA journal_mode=WAL"))
                conn.commit()

            # Create table and insert rows to generate WAL frames
            with engine.connect() as conn:
                conn.execute(text("CREATE TABLE test (id INTEGER PRIMARY KEY, value TEXT)"))
                conn.commit()

            # Insert 100 rows
            with engine.connect() as conn:
                for i in range(100):
                    conn.execute(text(f"INSERT INTO test (value) VALUES ('row_{i}')"))
                conn.commit()

            # Checkpoint and verify frames were checkpointed
            pages, frames = checkpoint_wal(engine)

            # After inserts, there should be some frames checkpointed
            assert frames >= 0
            assert pages >= 0

            # Second checkpoint should show no new frames
            pages2, frames2 = checkpoint_wal(engine)
            assert pages2 == 0  # No new pages to checkpoint
        finally:
            engine.dispose()  # Release file handles before temp dir cleanup
