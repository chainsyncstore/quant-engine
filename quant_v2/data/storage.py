"""Snapshot storage and QA validation for v2 multi-symbol datasets."""

from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from quant_v2.config import get_runtime_profile

MAX_CLOSE_SPIKE_FRAC = 0.60
FUNDING_STALE_MAX_STREAK = 72
OPEN_INTEREST_STALE_MAX_STREAK = 24
DATASET_RAW_SCHEMA_VERSION = "wp08-multi-symbol-raw-v1"
DATASET_TRANSFORMED_SCHEMA_VERSION = "wp08-multi-symbol-transformed-v1"


class DataQualityError(Exception):
    """Raised when a multi-symbol dataset fails validation checks."""


@dataclass(frozen=True)
class MultiSymbolSnapshot:
    """Saved snapshot metadata."""

    parquet_path: Path
    manifest_path: Path
    manifest: dict[str, Any]


@dataclass(frozen=True)
class DatasetManifest:
    """Canonical reproducibility manifest for a multi-symbol dataset."""

    dataset_name: str
    created_at: str
    source_retrieved_at: str
    raw_schema_version: str
    transformed_schema_version: str
    requested_symbols: list[str]
    fetched_symbols: list[str]
    failed_symbols: dict[str, str]
    requested_time_range: dict[str, str | None]
    actual_time_range: dict[str, str | None]
    n_rows: int
    n_symbols: int
    rows_per_symbol: dict[str, int]
    unique_timestamps_per_symbol: dict[str, int]
    duplicate_rows_per_symbol: dict[str, int]
    gap_stats_per_symbol: dict[str, dict[str, Any]]
    coverage_by_symbol: dict[str, dict[str, float]]
    content_digests: dict[str, str]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "created_at": self.created_at,
            "source_retrieved_at": self.source_retrieved_at,
            "raw_schema_version": self.raw_schema_version,
            "transformed_schema_version": self.transformed_schema_version,
            "requested_symbols": list(self.requested_symbols),
            "fetched_symbols": list(self.fetched_symbols),
            "failed_symbols": dict(self.failed_symbols),
            "requested_time_range": dict(self.requested_time_range),
            "actual_time_range": dict(self.actual_time_range),
            "n_rows": self.n_rows,
            "n_symbols": self.n_symbols,
            "rows_per_symbol": dict(self.rows_per_symbol),
            "unique_timestamps_per_symbol": dict(self.unique_timestamps_per_symbol),
            "duplicate_rows_per_symbol": dict(self.duplicate_rows_per_symbol),
            "gap_stats_per_symbol": dict(self.gap_stats_per_symbol),
            "coverage_by_symbol": dict(self.coverage_by_symbol),
            "content_digests": dict(self.content_digests),
            "metadata": dict(self.metadata),
        }


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


def _canonical_digest(df: pd.DataFrame) -> str:
    """Return a stable digest for a canonical multi-symbol frame."""

    canonical = df.sort_index()
    digest = hashlib.sha256()
    digest.update("|".join(map(str, canonical.columns)).encode("utf-8"))
    digest.update("|".join(str(dtype) for dtype in canonical.dtypes).encode("utf-8"))
    hashed = pd.util.hash_pandas_object(canonical, index=True).to_numpy(dtype="uint64", copy=False)
    digest.update(hashed.tobytes())
    return digest.hexdigest()


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
    if str(ts.tz) not in {"UTC", "UTC+00:00", "UTC-00:00"}:
        raise DataQualityError("timestamp level must be UTC")

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

        if "funding_rate_raw" in sym_df.columns:
            funding_null_ratio = float(sym_df["funding_rate_raw"].isna().mean())
            if funding_null_ratio > 0.02:
                raise DataQualityError(
                    f"Null ratio for funding_rate_raw on symbol {symbol} exceeds 0.02 "
                    f"({funding_null_ratio:.4f})"
                )

        stale_rules = (
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


def build_dataset_manifest(
    df: pd.DataFrame,
    *,
    dataset_name: str,
    metadata: dict[str, Any] | None = None,
) -> DatasetManifest:
    """Build a reproducibility manifest from a canonical multi-symbol frame."""

    validate_multi_symbol_ohlcv(df)
    metadata = dict(metadata or {})

    symbols = sorted(str(s) for s in df.index.get_level_values("symbol").unique())
    ts = df.index.get_level_values("timestamp")
    requested_symbols = [str(s) for s in metadata.get("requested_symbols", symbols)]
    fetched_symbols = [str(s) for s in metadata.get("fetched_symbols", symbols)]
    failed_symbols = {str(k): str(v) for k, v in metadata.get("failed_symbols", {}).items()}
    requested_time_range = {
        "start": metadata.get("requested_time_range_start"),
        "end": metadata.get("requested_time_range_end"),
    }
    actual_time_range = {
        "start": str(ts.min()) if len(ts) else None,
        "end": str(ts.max()) if len(ts) else None,
    }

    rows_per_symbol: dict[str, int] = {}
    unique_timestamps_per_symbol: dict[str, int] = {}
    duplicate_rows_per_symbol: dict[str, int] = {}
    gap_stats_per_symbol: dict[str, dict[str, Any]] = {}
    coverage_by_symbol: dict[str, dict[str, float]] = {}

    for symbol, sym_df in df.groupby(level="symbol", sort=False):
        sym_ts = pd.DatetimeIndex(sym_df.index.get_level_values("timestamp"))
        rows_per_symbol[str(symbol)] = int(len(sym_df))
        unique_timestamps_per_symbol[str(symbol)] = int(sym_ts.nunique())
        duplicate_rows_per_symbol[str(symbol)] = int(sym_df.index.duplicated().sum())

        expected_step = _infer_expected_step(sym_ts)
        if expected_step is None or len(sym_ts) < 2:
            gap_stats = {"expected_step": None, "gap_count": 0, "max_gap": None}
        else:
            deltas = sym_ts.to_series().diff().dropna()
            gap_mask = deltas > expected_step * 1.5
            gap_stats = {
                "expected_step": str(expected_step),
                "gap_count": int(gap_mask.sum()),
                "max_gap": str(deltas.max()) if not deltas.empty else None,
            }
        gap_stats_per_symbol[str(symbol)] = gap_stats

        coverage_by_symbol[str(symbol)] = {
            column: float(1.0 - sym_df[column].isna().mean())
            for column in df.columns
        }

    return DatasetManifest(
        dataset_name=dataset_name,
        created_at=datetime.now(timezone.utc).isoformat(),
        source_retrieved_at=str(metadata.get("source_retrieved_at", datetime.now(timezone.utc).isoformat())),
        raw_schema_version=str(metadata.get("raw_schema_version", DATASET_RAW_SCHEMA_VERSION)),
        transformed_schema_version=str(metadata.get("transformed_schema_version", DATASET_TRANSFORMED_SCHEMA_VERSION)),
        requested_symbols=requested_symbols,
        fetched_symbols=fetched_symbols,
        failed_symbols=failed_symbols,
        requested_time_range=requested_time_range,
        actual_time_range=actual_time_range,
        n_rows=int(len(df)),
        n_symbols=int(len(symbols)),
        rows_per_symbol=rows_per_symbol,
        unique_timestamps_per_symbol=unique_timestamps_per_symbol,
        duplicate_rows_per_symbol=duplicate_rows_per_symbol,
        gap_stats_per_symbol=gap_stats_per_symbol,
        coverage_by_symbol=coverage_by_symbol,
        content_digests={
            "frame_sha256": _canonical_digest(df),
        },
        metadata=metadata,
    )


def build_snapshot_manifest(
    df: pd.DataFrame,
    *,
    dataset_name: str,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a compact manifest describing a multi-symbol snapshot."""
    manifest = build_dataset_manifest(df, dataset_name=dataset_name, metadata=metadata)
    output = manifest.to_dict()
    output["columns"] = list(df.columns)
    output["null_ratio_by_column"] = {col: float(df[col].isna().mean()) for col in df.columns}
    output["symbols"] = list(manifest.fetched_symbols)
    output["time_range"] = output.pop("actual_time_range")
    return output


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
