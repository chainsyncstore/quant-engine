from __future__ import annotations

import numpy as np
import pandas as pd

from quant_v2.validation.temporal_validation import (
    build_temporal_validation_plan,
    compute_recency_weights,
    effective_sample_size,
)


def _make_multi_symbol_frame(months: int = 8) -> pd.DataFrame:
    ts = pd.date_range("2024-01-01", periods=months * 30, freq="D", tz="UTC")
    rows = []
    for symbol_idx, symbol in enumerate(("BTCUSDT", "ETHUSDT")):
        base = 100.0 + symbol_idx * 10.0
        for i, timestamp in enumerate(ts):
            close = base + i
            rows.append(
                {
                    "timestamp": timestamp,
                    "symbol": symbol,
                    "open": close - 0.2,
                    "high": close + 0.4,
                    "low": close - 0.5,
                    "close": close,
                    "volume": 1000.0 + i,
                }
            )
    return pd.DataFrame(rows).set_index(["timestamp", "symbol"]).sort_index()


def test_temporal_plan_reserves_final_holdout_and_purges_train_overlap() -> None:
    df = _make_multi_symbol_frame()
    plan = build_temporal_validation_plan(
        df,
        training_windows_months=(3, 6),
        expanding_included=True,
        test_window_months=1,
        holdout_months=2,
        purge_bars=2,
        min_train_rows=10,
    )

    assert plan.holdout_start is not None
    assert plan.holdout_end is not None
    assert len(plan.holdout_indices) > 0
    assert len(plan.folds) > 0
    assert plan.total_trials >= len(plan.folds)

    holdout_rows = set(plan.holdout_indices.tolist())
    for fold in plan.folds:
        assert fold.train_end < fold.test_start
        assert fold.train_start <= fold.train_end
        assert fold.purge_start <= fold.test_start
        assert not holdout_rows.intersection(fold.train_indices.tolist())
        assert not holdout_rows.intersection(fold.test_indices.tolist())

    variants = {fold.variant for fold in plan.folds}
    assert "rolling_3m" in variants
    assert "expanding" in variants


def test_recency_weights_are_normalized_and_reduce_effective_sample_size() -> None:
    timestamps = pd.date_range("2024-01-01", periods=10, freq="D", tz="UTC")
    weights = compute_recency_weights(timestamps, half_life_days=5.0)

    assert len(weights) == len(timestamps)
    assert np.isclose(float(weights.mean()), 1.0)
    assert weights[-1] > weights[0]
    assert effective_sample_size(weights) < len(weights)
