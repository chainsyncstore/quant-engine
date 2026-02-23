"""Snapshot storage and QA validation for v2 multi-symbol datasets."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from quant_v2.config import get_runtime_profile

MAX_CLOSE_SPIKE_FRAC = 0.60
FUNDING_STALE_MAX_STREAK = 72
OPEN_INTEREST_STALE_MAX_STREAK = 24


class DataQualityError(Exception):
    """Raised when a multi-symbol dataset fails validation checks."""


@dataclass(frozen=True)
class MultiSymbolSnapshot:
    """Saved snapshot metadata."""

    parquet_path: Path
    manifest_path: Path
    manifest: dict[str, Any]


def _max_constant_streak(series: pd.Series) -> int:
    """Return longest run of unchanged non-null values in a series."""

    clean = series.dropna()
    if clean.empty:
        return 0

    groups = (clean != clean.shift()).cumsum()
    return int(clean.groupby(groups).size().max())


def _infer_expected_step(timestamps: pd.DatetimeIndex) -> pd.Timedelta | None:
    """Infer dominant bar step from a sorted timestamp index."""

    if len(timestamps) < 3:
        return None

    deltas = timestamps.to_series().diff().dropna()
    if deltas.empty:
        return None

    mode = deltas.mode()
    step = mode.iloc[0] if not mode.empty else deltas.median()
    if pd.isna(step) or step <= pd.Timedelta(0):
        return None
    return pd.Timedelta(step)


def validate_multi_symbol_ohlcv(
    df: pd.DataFrame,
    *,
    expected_symbols: tuple[str, ...] | None = None,
) -> None:
    """Validate canonical v2 multi-symbol OHLCV layout."""

    if not isinstance(df.index, pd.MultiIndex) or df.index.nlevels != 2:
        raise DataQualityError("Index must be MultiIndex with levels [timestamp, symbol]")

    if list(df.index.names) != ["timestamp", "symbol"]:
        raise DataQualityError("MultiIndex levels must be named ['timestamp', 'symbol']")

    required_cols = {"open", "high", "low", "close", "volume"}
    missing = required_cols - set(df.columns)
    if missing:
        raise DataQualityError(f"Missing required columns: {sorted(missing)}")

    ts = df.index.get_level_values("timestamp")
    if not isinstance(ts, pd.DatetimeIndex):
        raise DataQualityError("timestamp level must be DatetimeIndex")
    if ts.tz is None:
        raise DataQualityError("timestamp level must be timezone-aware")

    dupes = int(df.index.duplicated().sum())
    if dupes > 0:
        raise DataQualityError(f"Found {dupes} duplicate (timestamp, symbol) rows")

    symbols_present = tuple(sorted(str(s) for s in df.index.get_level_values("symbol").unique()))
    if expected_symbols:
        missing_symbols = sorted(set(expected_symbols) - set(symbols_present))
        if missing_symbols:
            raise DataQualityError(f"Missing expected symbols: {missing_symbols}")

    for symbol, sym_df in df.groupby(level="symbol", sort=False):
        sym_ts = sym_df.index.get_level_values("timestamp")
        if not sym_ts.is_monotonic_increasing:
            raise DataQualityError(f"Symbol {symbol} timestamps are not sorted ascending")

        expected_step = _infer_expected_step(pd.DatetimeIndex(sym_ts))
        if expected_step is not None:
            deltas = pd.DatetimeIndex(sym_ts).to_series().diff().dropna()
            if not deltas.empty and deltas.max() > expected_step * 1.5:
                raise DataQualityError(
                    f"Symbol {symbol} has continuity gap larger than expected step "
                    f"({deltas.max()} > {expected_step})"
                )

        for col in required_cols:
            null_ratio = float(sym_df[col].isna().mean())
            if null_ratio > 0.0:
                raise DataQualityError(
                    f"Null ratio for required column {col} on symbol {symbol} exceeds 0 "
                    f"({null_ratio:.4f})"
                )

        bad_low = (sym_df["low"] > sym_df["open"]).any() or (sym_df["low"] > sym_df["close"]).any()
        bad_high = (sym_df["high"] < sym_df["open"]).any() or (sym_df["high"] < sym_df["close"]).any()
        if bad_low or bad_high:
            raise DataQualityError(f"OHLC relationship violated for symbol {symbol}")

        if (sym_df["volume"] < 0).any():
            raise DataQualityError(f"Negative volume detected for symbol {symbol}")

        close_returns = (
            sym_df["close"]
            .pct_change()
            .abs()
            .replace([float("inf"), float("-inf")], pd.NA)
            .dropna()
        )
        if not close_returns.empty and float(close_returns.max()) > MAX_CLOSE_SPIKE_FRAC:
            raise DataQualityError(
                f"Symbol {symbol} has close-price spike above {MAX_CLOSE_SPIKE_FRAC:.2f} "
                f"(max={float(close_returns.max()):.4f})"
            )

        stale_rules = (
            ("funding_rate_raw", FUNDING_STALE_MAX_STREAK),
            ("open_interest", OPEN_INTEREST_STALE_MAX_STREAK),
            ("open_interest_value", OPEN_INTEREST_STALE_MAX_STREAK),
        )
        for field, max_streak in stale_rules:
            if field not in sym_df.columns:
                continue
            streak = _max_constant_streak(sym_df[field])
            if streak >= max_streak:
                raise DataQualityError(
                    f"Symbol {symbol} has stale {field} series "
                    f"(constant streak={streak}, limit={max_streak})"
                )


def build_snapshot_manifest(
    df: pd.DataFrame,
    *,
    dataset_name: str,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a compact manifest describing a multi-symbol snapshot."""

    symbols = sorted(str(s) for s in df.index.get_level_values("symbol").unique())
    ts = df.index.get_level_values("timestamp")

    rows_per_symbol = {
        symbol: int((df.index.get_level_values("symbol") == symbol).sum())
        for symbol in symbols
    }

    manifest: dict[str, Any] = {
        "dataset_name": dataset_name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "n_rows": int(len(df)),
        "n_symbols": int(len(symbols)),
        "symbols": symbols,
        "rows_per_symbol": rows_per_symbol,
        "columns": list(df.columns),
        "time_range": {
            "start": str(ts.min()) if len(ts) else None,
            "end": str(ts.max()) if len(ts) else None,
        },
        "null_ratio_by_column": {
            col: float(df[col].isna().mean())
            for col in df.columns
        },
        "metadata": metadata or {},
    }
    return manifest


def save_multi_symbol_snapshot(
    df: pd.DataFrame,
    *,
    dataset_name: str,
    metadata: dict[str, Any] | None = None,
    root_dir: Path | None = None,
) -> MultiSymbolSnapshot:
    """Persist dataset to parquet and write a sidecar manifest."""

    clean_name = dataset_name.strip()
    if not clean_name:
        raise ValueError("dataset_name cannot be empty")

    validate_multi_symbol_ohlcv(df)

    if root_dir is None:
        root_dir = get_runtime_profile().project_root / "datasets" / "v2" / "snapshots"
    root_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    base_name = f"{clean_name}_{ts}"

    parquet_path = root_dir / f"{base_name}.parquet"
    manifest_path = root_dir / f"{base_name}.manifest.json"

    df.to_parquet(parquet_path, engine="pyarrow", compression="snappy")
    manifest = build_snapshot_manifest(df, dataset_name=clean_name, metadata=metadata)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    return MultiSymbolSnapshot(
        parquet_path=parquet_path,
        manifest_path=manifest_path,
        manifest=manifest,
    )


def load_multi_symbol_snapshot(parquet_path: Path | str) -> tuple[pd.DataFrame, dict[str, Any] | None]:
    """Load dataset parquet and optional sidecar manifest."""

    path = Path(parquet_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Snapshot not found: {path}")

    df = pd.read_parquet(path, engine="pyarrow")
    if isinstance(df.index, pd.MultiIndex):
        df.index = df.index.set_names(["timestamp", "symbol"])

    manifest_path = path.with_suffix("").with_suffix(".manifest.json")
    manifest = None
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    return df, manifest
