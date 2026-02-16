"""Tests for labeling engine."""

import numpy as np
import pandas as pd

from quant.labels.labeler import add_labels


class TestLabeler:
    def test_3m_labels_correct(self, synthetic_ohlcv):
        result = add_labels(synthetic_ohlcv, horizons=[3])
        # Verify labels manually for first few rows
        for i in range(min(10, len(result))):
            expected = 1 if synthetic_ohlcv["close"].iloc[i + 3] > synthetic_ohlcv["close"].iloc[i] else 0
            assert result["label_3m"].iloc[i] == expected, f"Mismatch at index {i}"

    def test_5m_labels_correct(self, synthetic_ohlcv):
        result = add_labels(synthetic_ohlcv, horizons=[5])
        for i in range(min(10, len(result))):
            expected = 1 if synthetic_ohlcv["close"].iloc[i + 5] > synthetic_ohlcv["close"].iloc[i] else 0
            assert result["label_5m"].iloc[i] == expected, f"Mismatch at index {i}"

    def test_tail_rows_dropped(self, synthetic_ohlcv):
        result = add_labels(synthetic_ohlcv, horizons=[3, 5])
        # Should drop last 5 rows (max horizon)
        assert len(result) == len(synthetic_ohlcv) - 5

    def test_labels_are_binary(self, synthetic_ohlcv):
        result = add_labels(synthetic_ohlcv, horizons=[3, 5])
        assert set(result["label_3m"].unique()).issubset({0, 1})
        assert set(result["label_5m"].unique()).issubset({0, 1})

    def test_both_horizons_present(self, synthetic_ohlcv):
        result = add_labels(synthetic_ohlcv)
        assert "label_3m" in result.columns
        assert "label_5m" in result.columns
