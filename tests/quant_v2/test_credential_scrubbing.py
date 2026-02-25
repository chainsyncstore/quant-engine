"""Tests for Fix 2: Credential scrubbing in WAL entries."""

from __future__ import annotations

import json

from quant_v2.execution.state_wal import WALEntry, _scrub_payload


class TestScrubPayload:
    """Unit tests for the _scrub_payload utility."""

    def test_strips_api_key_and_api_secret(self):
        payload = {
            "live": True,
            "api_key": "abc123secret",
            "api_secret": "xyz789secret",
            "strategy_profile": "core_v2",
        }
        result = _scrub_payload(payload)
        assert result["api_key"] == "***REDACTED***"
        assert result["api_secret"] == "***REDACTED***"
        assert result["live"] is True
        assert result["strategy_profile"] == "core_v2"

    def test_strips_nested_credentials(self):
        payload = {
            "user_id": 42,
            "credentials": {"api_key": "k", "api_secret": "s"},
        }
        result = _scrub_payload(payload)
        assert result["credentials"] == "***REDACTED***"

    def test_leaves_clean_payload_unchanged(self):
        payload = {"live": True, "universe": ["BTCUSDT"]}
        result = _scrub_payload(payload)
        assert result == payload


class TestWALEntryNeverContainsKeys:
    """Ensure WALEntry.to_json() redacts credential material."""

    def test_wal_entry_never_contains_api_keys(self):
        entry = WALEntry(
            event_type="session_started",
            user_id=42,
            payload={
                "live": True,
                "strategy_profile": "core_v2",
                "credentials": {"api_key": "REAL_KEY", "api_secret": "REAL_SECRET"},
            },
        )
        serialized = entry.to_json()
        parsed = json.loads(serialized)

        # credentials key itself should be redacted
        assert "REAL_KEY" not in serialized
        assert "REAL_SECRET" not in serialized
        assert parsed["payload"]["credentials"] == "***REDACTED***"

    def test_wal_entry_with_no_sensitive_data_unchanged(self):
        entry = WALEntry(
            event_type="equity_updated",
            user_id=42,
            payload={"equity_usd": 10500.0},
        )
        serialized = entry.to_json()
        parsed = json.loads(serialized)
        assert parsed["payload"]["equity_usd"] == 10500.0
