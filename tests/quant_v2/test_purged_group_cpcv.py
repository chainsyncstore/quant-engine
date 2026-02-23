from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_v2.validation.purged_group_cpcv import (
    build_symbol_clusters,
    iter_purged_group_splits,
    summarize_split_coverage,
)


def _build_df() -> pd.DataFrame:
    ts = pd.date_range("2025-01-01", periods=60, freq="1h", tz="UTC")
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]
    idx = pd.MultiIndex.from_product([ts, symbols], names=["timestamp", "symbol"])

    values = np.linspace(100.0, 130.0, num=len(idx))
    return pd.DataFrame(
        {
            "open": values,
            "high": values + 1.0,
            "low": values - 1.0,
            "close": values + 0.3,
            "volume": np.full(len(idx), 1000.0),
        },
        index=idx,
    )


def test_build_symbol_clusters() -> None:
    clusters = build_symbol_clusters(["ETHUSDT", "BTCUSDT", "SOLUSDT"], cluster_size=2)
    assert clusters == [("BTCUSDT", "ETHUSDT"), ("SOLUSDT",)]


def test_iter_purged_group_splits_no_train_test_overlap() -> None:
    df = _build_df()
    splits = iter_purged_group_splits(
        df,
        n_time_splits=5,
        symbol_cluster_size=2,
        embargo_bars=2,
        min_train_rows=10,
    )

    assert splits, "Expected at least one split"

    for split in splits:
        assert len(np.intersect1d(split.train_indices, split.test_indices)) == 0


def test_iter_purged_group_splits_hold_out_symbols_and_purge_window() -> None:
    df = _build_df()
    splits = iter_purged_group_splits(
        df,
        n_time_splits=4,
        symbol_cluster_size=2,
        embargo_bars=1,
        min_train_rows=10,
    )

    assert splits

    for split in splits:
        train_index = df.iloc[split.train_indices].index
        test_index = df.iloc[split.test_indices].index

        train_symbols = set(train_index.get_level_values("symbol"))
        test_symbols = set(test_index.get_level_values("symbol"))

        assert test_symbols.issubset(set(split.test_symbols))
        assert train_symbols.isdisjoint(test_symbols)

        train_ts = train_index.get_level_values("timestamp")
        assert not ((train_ts >= split.purge_time_start) & (train_ts <= split.purge_time_end)).any()


def test_summarize_split_coverage() -> None:
    df = _build_df()
    splits = iter_purged_group_splits(df, n_time_splits=3, symbol_cluster_size=2, embargo_bars=1)
    summary = summarize_split_coverage(splits)

    assert summary["n_splits"] == len(splits)
    assert summary["n_unique_test_symbols"] > 0
    assert summary["n_total_test_rows"] > 0


def test_iter_purged_group_splits_validate_args() -> None:
    df = _build_df()

    with pytest.raises(ValueError):
        iter_purged_group_splits(df, n_time_splits=1)

    with pytest.raises(ValueError):
        iter_purged_group_splits(df, symbol_cluster_size=0)

    with pytest.raises(ValueError):
        iter_purged_group_splits(df, embargo_bars=-1)
