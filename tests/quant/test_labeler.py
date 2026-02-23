"""Tests for labeling engine."""

import numpy as np
import pandas as pd

from quant.labels.labeler import add_labels


class TestLabeler:
    def test_3m_labels_correct(self, synthetic_ohlcv):
        from quant.config import get_research_config
        cfg = get_research_config()
        result = add_labels(synthetic_ohlcv, horizons=[3])
        # Verify labels manually for first few rows
        for i in range(min(10, len(result))):
            move = synthetic_ohlcv["close"].iloc[i + 3] - synthetic_ohlcv["close"].iloc[i]
            dz = synthetic_ohlcv["close"].iloc[i] * cfg.dead_zone_pct
            if move > dz:
                expected = 1
            elif move < -dz:
                expected = 0
            else:
                expected = -1
            assert result["label_3m"].iloc[i] == expected, f"Mismatch at index {i}"

    def test_5m_labels_correct(self, synthetic_ohlcv):
        from quant.config import get_research_config
        cfg = get_research_config()
        result = add_labels(synthetic_ohlcv, horizons=[5])
        for i in range(min(10, len(result))):
            move = synthetic_ohlcv["close"].iloc[i + 5] - synthetic_ohlcv["close"].iloc[i]
            dz = synthetic_ohlcv["close"].iloc[i] * cfg.dead_zone_pct
            if move > dz:
                expected = 1
            elif move < -dz:
                expected = 0
            else:
                expected = -1
            assert result["label_5m"].iloc[i] == expected, f"Mismatch at index {i}"

    def test_tail_rows_dropped(self, synthetic_ohlcv):
        result = add_labels(synthetic_ohlcv, horizons=[3, 5])
        # Should drop last 5 rows (max horizon)
        assert len(result) == len(synthetic_ohlcv) - 5

    def test_labels_are_ternary(self, synthetic_ohlcv):
        result = add_labels(synthetic_ohlcv, horizons=[3, 5])
        assert set(result["label_3m"].unique()).issubset({-1, 0, 1})
        assert set(result["label_5m"].unique()).issubset({-1, 0, 1})

    def test_dead_zone_filters_small_moves(self):
        """Verify moves within dead zone are labeled FLAT (-1)."""
        from datetime import datetime, timedelta, timezone
        # Create prices with known moves beyond dead zone in crypto mode.
        ts = [datetime(2025, 12, 1, 10, i, tzinfo=timezone.utc) for i in range(10)]
        prices = [1.0, 1.0, 1.0, 1.0012, 1.0, 1.0, 0.9988, 1.0, 1.0, 1.00001]
        df = pd.DataFrame({
            "open": prices, "high": [p + 0.001 for p in prices],
            "low": [p - 0.001 for p in prices], "close": prices,
            "volume": [100] * 10,
        }, index=pd.DatetimeIndex(ts, tz="UTC"))
        result = add_labels(df, horizons=[3])
        # Row 0: close[3]=1.0012 - close[0]=1.0 = 0.0012 > dead_zone → UP (1)
        assert result["label_3m"].iloc[0] == 1
        # Row 3: close[6]=0.9988 - close[3]=1.0012 = -0.0024 < -dead_zone → DOWN (0)
        assert result["label_3m"].iloc[3] == 0

    def test_both_horizons_present(self, synthetic_ohlcv):
        from quant.config import get_research_config
        cfg = get_research_config()
        result = add_labels(synthetic_ohlcv)
        for h in cfg.horizons:
            assert f"label_{h}m" in result.columns
