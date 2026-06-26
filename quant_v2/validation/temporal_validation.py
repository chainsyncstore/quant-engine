"""Timestamp-native temporal validation helpers for v2 research."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TemporalFold:
    """One timestamp-native train/test fold."""

    fold_id: str
    variant: str
    train_indices: np.ndarray
    test_indices: np.ndarray
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    purge_start: pd.Timestamp
    purge_end: pd.Timestamp
    holdout: bool = False

    @property
    def n_train(self) -> int:
        return int(len(self.train_indices))

    @property
    def n_test(self) -> int:
        return int(len(self.test_indices))


@dataclass(frozen=True)
class TemporalValidationPlan:
    """Canonical temporal folds plus a final untouched holdout."""

    folds: list[TemporalFold]
    holdout_indices: np.ndarray
    holdout_start: pd.Timestamp | None
    holdout_end: pd.Timestamp | None
    training_windows_months: tuple[int, ...]
    expanding_included: bool
    test_window_months: int
    holdout_months: int
    purge_bars: int
    total_trials: int
    metadata: dict[str, Any] = field(default_factory=dict)


def _require_multi_index(df: pd.DataFrame) -> tuple[pd.DatetimeIndex, pd.Index]:
    if not isinstance(df.index, pd.MultiIndex) or list(df.index.names) != ["timestamp", "symbol"]:
        raise ValueError("df must be MultiIndex with levels ['timestamp', 'symbol']")
    timestamps = pd.DatetimeIndex(df.index.get_level_values("timestamp"))
    if timestamps.tz is None:
        raise ValueError("timestamp level must be timezone-aware")
    if str(timestamps.tz) not in {"UTC", "UTC+00:00", "UTC-00:00"}:
        raise ValueError("timestamp level must be UTC")
    return timestamps, df.index.get_level_values("symbol")


def compute_recency_weights(
    timestamps: pd.DatetimeIndex,
    *,
    half_life_days: float,
) -> np.ndarray:
    """Return normalized exponential recency weights and preserve effective history."""

    if len(timestamps) == 0:
        return np.array([], dtype=float)

    ts = pd.DatetimeIndex(timestamps)
    anchor = ts.max()
    delta_days = (anchor - ts).total_seconds() / 86400.0
    decay = np.log(2.0) / max(float(half_life_days), 1e-9)
    weights = np.exp(-decay * np.asarray(delta_days, dtype=float))
    mean = float(weights.mean()) if weights.size else 1.0
    if mean > 0.0:
        weights = weights / mean
    return weights.astype(float, copy=False)


def effective_sample_size(weights: np.ndarray) -> float:
    """Return the Kish effective sample size for a weight vector."""

    if weights.size == 0:
        return 0.0
    w = np.asarray(weights, dtype=float)
    denom = float(np.sum(w * w))
    if denom <= 0.0:
        return 0.0
    return float((np.sum(w) ** 2) / denom)


def build_temporal_validation_plan(
    df: pd.DataFrame,
    *,
    training_windows_months: tuple[int, ...] = (3, 6, 9, 12),
    expanding_included: bool = True,
    test_window_months: int = 1,
    holdout_months: int = 3,
    purge_bars: int = 0,
    min_train_rows: int = 1,
) -> TemporalValidationPlan:
    """Build explicit monthly timestamp folds plus a final untouched holdout."""

    if test_window_months < 1:
        raise ValueError("test_window_months must be >= 1")
    if holdout_months < 1:
        raise ValueError("holdout_months must be >= 1")
    if purge_bars < 0:
        raise ValueError("purge_bars must be >= 0")
    if min_train_rows < 1:
        raise ValueError("min_train_rows must be >= 1")

    timestamps, _ = _require_multi_index(df)
    unique_ts = pd.DatetimeIndex(sorted(timestamps.unique()))
    if len(unique_ts) < 2:
        raise ValueError("Not enough timestamps for temporal validation")

    periods = timestamps.to_period("M")
    unique_periods = pd.PeriodIndex(sorted(periods.unique()))
    if len(unique_periods) <= holdout_months:
        raise ValueError("Not enough monthly periods for a final holdout")

    holdout_periods = unique_periods[-holdout_months:]
    holdout_mask = periods.isin(holdout_periods)
    holdout_indices = np.flatnonzero(np.asarray(holdout_mask))
    holdout_timestamps = timestamps[holdout_mask]
    holdout_start = holdout_timestamps.min() if holdout_indices.size else None
    holdout_end = holdout_timestamps.max() if holdout_indices.size else None

    development_periods = unique_periods[:-holdout_months]
    if len(development_periods) < test_window_months + 1:
        raise ValueError("Not enough development periods for the requested test window")

    folds: list[TemporalFold] = []
    variant_specs: list[tuple[str, int | None]] = [
        (f"rolling_{months}m", months) for months in training_windows_months
    ]
    if expanding_included:
        variant_specs.append(("expanding", None))

    dev_start_period = development_periods[0]
    test_periods = development_periods[test_window_months:]

    for test_idx, test_period in enumerate(test_periods, start=1):
        test_mask = periods == test_period
        test_indices = np.flatnonzero(np.asarray(test_mask))
        if len(test_indices) == 0:
            continue

        test_timestamps = timestamps[test_mask]
        test_start = test_timestamps.min()
        test_end = test_timestamps.max()
        purge_start = test_start - pd.Timedelta(hours=purge_bars)
        purge_end = test_end

        for variant_name, window_months in variant_specs:
            if window_months is None:
                train_periods = development_periods[development_periods < test_period]
            else:
                eligible_periods = development_periods[
                    (development_periods < test_period)
                    & (development_periods >= test_period - window_months)
                ]
                train_periods = eligible_periods

            train_mask = periods.isin(train_periods)
            if purge_bars > 0:
                train_mask = train_mask & (timestamps < purge_start)

            train_indices = np.flatnonzero(np.asarray(train_mask))
            if len(train_indices) < min_train_rows:
                continue
            if not len(train_indices) or not len(test_indices):
                continue

            train_timestamps = timestamps[train_mask]
            train_start = train_timestamps.min()
            train_end = train_timestamps.max()
            folds.append(
                TemporalFold(
                    fold_id=f"{variant_name}_t{test_idx:02d}",
                    variant=variant_name,
                    train_indices=train_indices,
                    test_indices=test_indices,
                    train_start=train_start,
                    train_end=train_end,
                    test_start=test_start,
                    test_end=test_end,
                    purge_start=purge_start,
                    purge_end=purge_end,
                )
            )

    if not folds:
        raise ValueError("No temporal folds produced for the requested configuration")

    total_trials = len(folds) * max(len(training_windows_months) + int(expanding_included), 1)
    metadata = {
        "development_periods": [str(period) for period in development_periods],
        "holdout_periods": [str(period) for period in holdout_periods],
        "dev_start_period": str(dev_start_period),
    }
    return TemporalValidationPlan(
        folds=folds,
        holdout_indices=holdout_indices,
        holdout_start=holdout_start,
        holdout_end=holdout_end,
        training_windows_months=training_windows_months,
        expanding_included=expanding_included,
        test_window_months=test_window_months,
        holdout_months=holdout_months,
        purge_bars=purge_bars,
        total_trials=total_trials,
        metadata=metadata,
    )
