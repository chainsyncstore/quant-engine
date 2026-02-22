"""Tests for session filter."""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone

from quant.data.session_filter import filter_sessions
from quant.config import ResearchConfig


class TestSessionFilter:
    def test_keeps_weekday_session_bars(self, synthetic_ohlcv):
        result = filter_sessions(synthetic_ohlcv)
        # All bars should already be in session (our fixture generates session-only data)
        assert len(result) == len(synthetic_ohlcv)

    def test_drops_weekend_bars(self, monkeypatch):
        monkeypatch.setattr(
            "quant.data.session_filter.get_research_config",
            lambda: ResearchConfig(mode="fx"),
        )
        # Create data with a Saturday bar
        timestamps = [
            datetime(2025, 12, 6, 10, 0, tzinfo=timezone.utc),  # Saturday
            datetime(2025, 12, 8, 10, 0, tzinfo=timezone.utc),  # Monday
        ]
        df = pd.DataFrame(
            {"open": [1.0, 1.0], "high": [1.01, 1.01], "low": [0.99, 0.99],
             "close": [1.005, 1.005], "volume": [100, 100]},
            index=pd.DatetimeIndex(timestamps, tz="UTC"),
        )
        result = filter_sessions(df)
        assert len(result) == 1  # Only Monday kept
        assert result.index[0].weekday() == 0  # Monday

    def test_drops_out_of_session_bars(self, monkeypatch):
        monkeypatch.setattr(
            "quant.data.session_filter.get_research_config",
            lambda: ResearchConfig(mode="fx"),
        )
        # Create data at 03:00 UTC (before London open)
        timestamps = [
            datetime(2025, 12, 8, 3, 0, tzinfo=timezone.utc),   # Too early
            datetime(2025, 12, 8, 10, 0, tzinfo=timezone.utc),  # In session
            datetime(2025, 12, 8, 22, 0, tzinfo=timezone.utc),  # Too late
        ]
        df = pd.DataFrame(
            {"open": [1.0] * 3, "high": [1.01] * 3, "low": [0.99] * 3,
             "close": [1.005] * 3, "volume": [100] * 3},
            index=pd.DatetimeIndex(timestamps, tz="UTC"),
        )
        result = filter_sessions(df)
        assert len(result) == 1
        assert result.index[0].hour == 10

    def test_empty_dataframe(self):
        df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        df.index = pd.DatetimeIndex([], tz="UTC")
        result = filter_sessions(df)
        assert len(result) == 0
