"""Group-aware purged CV splitter for multi-symbol validation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PurgedGroupSplit:
    """One train/test split with symbol-group holdout and time embargo."""

    split_id: str
    train_indices: np.ndarray
    test_indices: np.ndarray
    test_symbols: tuple[str, ...]
    test_time_start: pd.Timestamp
    test_time_end: pd.Timestamp
    purge_time_start: pd.Timestamp
    purge_time_end: pd.Timestamp

    @property
    def n_train(self) -> int:
        return int(len(self.train_indices))

    @property
    def n_test(self) -> int:
        return int(len(self.test_indices))


def build_symbol_clusters(symbols: list[str] | tuple[str, ...], cluster_size: int) -> list[tuple[str, ...]]:
    """Build contiguous symbol clusters for group holdout."""

    if cluster_size <= 0:
        raise ValueError("cluster_size must be > 0")

    clean = sorted({s.strip().upper() for s in symbols if s and s.strip()})
    if not clean:
        raise ValueError("No symbols available for clustering")

    clusters: list[tuple[str, ...]] = []
    for i in range(0, len(clean), cluster_size):
        clusters.append(tuple(clean[i : i + cluster_size]))
    return clusters


def iter_purged_group_splits(
    df: pd.DataFrame,
    *,
    n_time_splits: int = 5,
    symbol_cluster_size: int = 2,
    embargo_bars: int = 24,
    min_train_rows: int = 1,
) -> list[PurgedGroupSplit]:
    """Return strict splits with symbol-group holdout and purged time windows."""

    if n_time_splits < 2:
        raise ValueError("n_time_splits must be >= 2")
    if symbol_cluster_size <= 0:
        raise ValueError("symbol_cluster_size must be > 0")
    if embargo_bars < 0:
        raise ValueError("embargo_bars must be >= 0")
    if min_train_rows < 1:
        raise ValueError("min_train_rows must be >= 1")

    if not isinstance(df.index, pd.MultiIndex) or list(df.index.names) != ["timestamp", "symbol"]:
        raise ValueError("df must be MultiIndex with levels ['timestamp', 'symbol']")

    ts = pd.DatetimeIndex(df.index.get_level_values("timestamp"))
    symbols = df.index.get_level_values("symbol").astype(str)

    unique_ts = pd.DatetimeIndex(sorted(ts.unique()))
    unique_symbols = sorted(set(symbols))

    if len(unique_ts) < n_time_splits:
        raise ValueError("Not enough unique timestamps for requested n_time_splits")

    time_folds = [pd.DatetimeIndex(chunk) for chunk in np.array_split(unique_ts, n_time_splits) if len(chunk) > 0]
    symbol_clusters = build_symbol_clusters(unique_symbols, cluster_size=symbol_cluster_size)
    ts_to_pos = {timestamp: i for i, timestamp in enumerate(unique_ts)}

    splits: list[PurgedGroupSplit] = []

    for time_fold_idx, fold_ts in enumerate(time_folds):
        test_start = fold_ts[0]
        test_end = fold_ts[-1]

        left_pos = max(0, ts_to_pos[test_start] - embargo_bars)
        right_pos = min(len(unique_ts) - 1, ts_to_pos[test_end] + embargo_bars)
        purge_start = unique_ts[left_pos]
        purge_end = unique_ts[right_pos]

        test_time_mask = (ts >= test_start) & (ts <= test_end)
        purge_mask = (ts >= purge_start) & (ts <= purge_end)

        for cluster_idx, cluster in enumerate(symbol_clusters):
            cluster_mask = symbols.isin(cluster)
            test_mask = test_time_mask & cluster_mask
            train_mask = (~cluster_mask) & (~purge_mask)

            train_indices = np.flatnonzero(train_mask)
            test_indices = np.flatnonzero(test_mask)

            if len(test_indices) == 0 or len(train_indices) < min_train_rows:
                continue

            split = PurgedGroupSplit(
                split_id=f"t{time_fold_idx:02d}_g{cluster_idx:02d}",
                train_indices=train_indices,
                test_indices=test_indices,
                test_symbols=cluster,
                test_time_start=test_start,
                test_time_end=test_end,
                purge_time_start=purge_start,
                purge_time_end=purge_end,
            )
            splits.append(split)

    return splits


def summarize_split_coverage(splits: list[PurgedGroupSplit]) -> dict[str, int]:
    """Return compact summary for diagnostics and report logs."""

    n_splits = len(splits)
    symbols = sorted({symbol for split in splits for symbol in split.test_symbols})

    return {
        "n_splits": n_splits,
        "n_test_symbols": len(symbols),
        "n_unique_test_symbols": len(symbols),
        "n_total_test_rows": int(sum(split.n_test for split in splits)),
        "n_total_train_rows": int(sum(split.n_train for split in splits)),
    }
