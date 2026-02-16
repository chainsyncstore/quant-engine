"""
Parquet-based local data storage with versioned snapshots.

Handles save/load, data validation, and snapshot management.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from quant.config import get_path_config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data validation
# ---------------------------------------------------------------------------
class DataQualityError(Exception):
    """Raised when data fails quality checks."""


def validate_ohlcv(df: pd.DataFrame) -> None:
    """
    Validate OHLCV data quality.

    Checks:
        - Required columns present
        - DatetimeIndex sorted ascending
        - No duplicate timestamps
        - OHLC relationships (low <= open/close <= high)
        - Volume >= 0

    Raises:
        DataQualityError: If any check fails.
    """
    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise DataQualityError(f"Missing columns: {missing}")

    if not isinstance(df.index, pd.DatetimeIndex):
        raise DataQualityError("Index must be DatetimeIndex")

    if not df.index.is_monotonic_increasing:
        raise DataQualityError("Index must be sorted ascending")

    dupes = df.index.duplicated().sum()
    if dupes > 0:
        raise DataQualityError(f"Found {dupes} duplicate timestamps")

    # OHLC relationships
    bad_low = (df["low"] > df["open"]).any() or (df["low"] > df["close"]).any()
    bad_high = (df["high"] < df["open"]).any() or (df["high"] < df["close"]).any()
    if bad_low or bad_high:
        raise DataQualityError("OHLC relationship violated: low <= open/close <= high")

    if (df["volume"] < 0).any():
        raise DataQualityError("Negative volume detected")


def report_gaps(df: pd.DataFrame, max_gap_minutes: int = 5) -> list[dict]:
    """
    Find gaps in 1-minute data during session hours.

    Args:
        df: Session-filtered OHLCV data.
        max_gap_minutes: Threshold for reporting gaps.

    Returns:
        List of gap dicts with 'start', 'end', 'gap_minutes'.
    """
    if len(df) < 2:
        return []

    diffs = pd.Series(df.index[1:] - df.index[:-1])
    gap_mask = diffs > pd.Timedelta(minutes=max_gap_minutes)

    gaps = []
    for i in np.where(gap_mask)[0]:
        gaps.append(
            {
                "start": df.index[i],
                "end": df.index[i + 1],
                "gap_minutes": int(diffs.iloc[i].total_seconds() / 60),
            }
        )

    if gaps:
        logger.warning("Found %d gaps > %d minutes in data", len(gaps), max_gap_minutes)
    return gaps


# ---------------------------------------------------------------------------
# Save / Load
# ---------------------------------------------------------------------------
def save_raw(df: pd.DataFrame, label: str) -> Path:
    """Save raw OHLCV data to parquet.

    Args:
        df: Validated OHLCV DataFrame.
        label: Descriptive label (e.g., 'EURUSD_1m_20260101_20260301').

    Returns:
        Path to saved file.
    """
    paths = get_path_config()
    fpath = paths.datasets_raw / f"{label}.parquet"
    df.to_parquet(fpath, engine="pyarrow", compression="snappy")
    logger.info("Saved raw data: %s (%d rows)", fpath, len(df))
    return fpath


def load_raw(label: str) -> pd.DataFrame:
    """Load raw OHLCV data from parquet.

    Args:
        label: Same label used in save_raw.

    Returns:
        DataFrame with UTC DatetimeIndex.
    """
    paths = get_path_config()
    fpath = paths.datasets_raw / f"{label}.parquet"
    if not fpath.exists():
        raise FileNotFoundError(f"No raw data file: {fpath}")
    df = pd.read_parquet(fpath, engine="pyarrow")
    logger.info("Loaded raw data: %s (%d rows)", fpath, len(df))
    return df


def snapshot(df: pd.DataFrame, tag: Optional[str] = None) -> Path:
    """Create a versioned snapshot of the dataset.

    Args:
        df: OHLCV DataFrame to snapshot.
        tag: Optional descriptive tag.

    Returns:
        Path to snapshot file.
    """
    paths = get_path_config()
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    name = f"EURUSD_1m_snap_{ts}"
    if tag:
        name += f"_{tag}"
    fpath = paths.datasets_snapshots / f"{name}.parquet"
    df.to_parquet(fpath, engine="pyarrow", compression="snappy")
    logger.info("Snapshot saved: %s (%d rows)", fpath, len(df))
    return fpath


def load_latest_snapshot() -> Optional[pd.DataFrame]:
    """Load the most recent dataset snapshot.

    Returns:
        DataFrame or None if no snapshots exist.
    """
    paths = get_path_config()
    snaps = sorted(paths.datasets_snapshots.glob("*.parquet"))
    if not snaps:
        return None
    fpath = snaps[-1]
    df = pd.read_parquet(fpath, engine="pyarrow")
    logger.info("Loaded snapshot: %s (%d rows)", fpath, len(df))
    return df
