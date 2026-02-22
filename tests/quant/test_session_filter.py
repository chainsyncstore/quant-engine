"""Tests for session filter."""

import pandas as pd
from datetime import datetime, timezone

from quant.data.session_filter import filter_sessions


class TestSessionFilter:
    def test_keeps_weekday_session_bars(self, synthetic_ohlcv):
        result = filter_sessions(synthetic_ohlcv)
        # Crypto mode keeps all bars.
        assert len(result) == len(synthetic_ohlcv)

    def test_keeps_weekend_bars(self):
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
        assert len(result) == 2
        assert result.index[0].weekday() == 5  # Saturday
        assert result.index[1].weekday() == 0  # Monday

    def test_keeps_all_hours(self):
        # Create data across arbitrary hours
        timestamps = [
            datetime(2025, 12, 8, 3, 0, tzinfo=timezone.utc),
            datetime(2025, 12, 8, 10, 0, tzinfo=timezone.utc),
            datetime(2025, 12, 8, 22, 0, tzinfo=timezone.utc),
        ]
        df = pd.DataFrame(
            {"open": [1.0] * 3, "high": [1.01] * 3, "low": [0.99] * 3,
             "close": [1.005] * 3, "volume": [100] * 3},
            index=pd.DatetimeIndex(timestamps, tz="UTC"),
        )
        result = filter_sessions(df)
        assert len(result) == 3
        assert list(result.index.hour) == [3, 10, 22]

    def test_empty_dataframe(self):
        df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        df.index = pd.DatetimeIndex([], tz="UTC")
        result = filter_sessions(df)
        assert len(result) == 0
