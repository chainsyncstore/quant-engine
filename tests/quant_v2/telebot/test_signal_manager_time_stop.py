"""Tests for time-stop safety feature in V2SignalManager."""

from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from quant_v2.telebot.signal_manager import V2SignalManager, _SignalSession


@pytest.fixture
def manager(tmp_path: Path) -> V2SignalManager:
    """Create a V2SignalManager with default test params."""
    return V2SignalManager(
        model_dir=tmp_path,
        symbols=("BTCUSDT",),
        loop_interval_seconds=1,
        max_hold_hours=12,
    )


@pytest.fixture
def mock_session() -> _SignalSession:
    """Create a mock _SignalSession for testing."""
    return _SignalSession(
        user_id=12345,
        live=False,
        client=MagicMock(),
        on_signal=MagicMock(),
    )


class TestTimeStop:
    """Tests for the time-stop safety feature."""

    def test_time_stop_upgrades_hold_to_sell_when_aged(
        self, manager: V2SignalManager, mock_session: _SignalSession
    ) -> None:
        """Verify HOLD is upgraded to SELL when position exceeds max_hold_hours."""
        # Set up a 13-hour old position
        aged_ts = datetime.now(timezone.utc) - timedelta(hours=13)
        mock_session.paper_entry_timestamps["BNBUSDT"] = aged_ts

        payload = {
            "symbol": "BNBUSDT",
            "signal": "HOLD",
            "reason": "model_hold",
            "close_price": 640.12,
        }

        manager._apply_time_stop(mock_session, payload)

        assert payload["signal"] == "SELL"
        assert payload["time_stop"] is True
        assert "time_stop=13." in payload["reason"]

    def test_time_stop_noop_for_fresh_position(
        self, manager: V2SignalManager, mock_session: _SignalSession
    ) -> None:
        """Verify HOLD is preserved for fresh positions under max_hold_hours."""
        # Set up a 1-hour old position
        fresh_ts = datetime.now(timezone.utc) - timedelta(hours=1)
        mock_session.paper_entry_timestamps["BNBUSDT"] = fresh_ts

        payload = {
            "symbol": "BNBUSDT",
            "signal": "HOLD",
            "reason": "model_hold",
            "close_price": 640.12,
        }

        manager._apply_time_stop(mock_session, payload)

        assert payload["signal"] == "HOLD"
        assert "time_stop" not in payload

    def test_time_stop_noop_for_symbol_without_entry(
        self, manager: V2SignalManager, mock_session: _SignalSession
    ) -> None:
        """Verify HOLD is preserved when symbol has no entry timestamp."""
        # No entry timestamp for BNBUSDT
        payload = {
            "symbol": "BNBUSDT",
            "signal": "HOLD",
            "reason": "model_hold",
            "close_price": 640.12,
        }

        manager._apply_time_stop(mock_session, payload)

        assert payload["signal"] == "HOLD"
        assert "time_stop" not in payload

    def test_time_stop_noop_for_buy_and_sell_signals(
        self, manager: V2SignalManager, mock_session: _SignalSession
    ) -> None:
        """Verify BUY and SELL signals are never overridden, even when aged."""
        # Set up a 20-hour old position
        aged_ts = datetime.now(timezone.utc) - timedelta(hours=20)
        mock_session.paper_entry_timestamps["BNBUSDT"] = aged_ts

        buy_payload = {
            "symbol": "BNBUSDT",
            "signal": "BUY",
            "reason": "model_buy",
            "close_price": 640.12,
        }
        sell_payload = {
            "symbol": "BNBUSDT",
            "signal": "SELL",
            "reason": "model_sell",
            "close_price": 640.12,
        }

        manager._apply_time_stop(mock_session, buy_payload)
        manager._apply_time_stop(mock_session, sell_payload)

        assert buy_payload["signal"] == "BUY"
        assert sell_payload["signal"] == "SELL"
        assert "time_stop" not in buy_payload
        assert "time_stop" not in sell_payload

    def test_time_stop_noop_for_live_session(
        self, manager: V2SignalManager, mock_session: _SignalSession
    ) -> None:
        """Verify time-stop is disabled for live sessions."""
        mock_session.live = True
        # Set up a 20-hour old position
        aged_ts = datetime.now(timezone.utc) - timedelta(hours=20)
        mock_session.paper_entry_timestamps["BNBUSDT"] = aged_ts

        payload = {
            "symbol": "BNBUSDT",
            "signal": "HOLD",
            "reason": "model_hold",
            "close_price": 640.12,
        }

        manager._apply_time_stop(mock_session, payload)

        assert payload["signal"] == "HOLD"
        assert "time_stop" not in payload


def test_time_stop_env_var_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify env var BOT_V2_MAX_HOLD_HOURS overrides default."""
    monkeypatch.setenv("BOT_V2_MAX_HOLD_HOURS", "4")

    manager = V2SignalManager(
        model_dir=tmp_path,
        symbols=("BTCUSDT",),
        loop_interval_seconds=1,
    )

    assert manager.max_hold_hours == 4
