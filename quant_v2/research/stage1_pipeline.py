"""Stage-1 v2 pipeline: dataset snapshot + group-purged split planning."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from quant_v2.config import get_runtime_profile
from quant_v2.data.multi_symbol_dataset import fetch_universe_dataset
from quant_v2.data.storage import (
    MultiSymbolSnapshot,
    load_multi_symbol_snapshot,
    save_multi_symbol_snapshot,
    validate_multi_symbol_ohlcv,
)
from quant_v2.validation.purged_group_cpcv import (
    PurgedGroupSplit,
    iter_purged_group_splits,
    summarize_split_coverage,
)


@dataclass(frozen=True)
class Stage1Result:
    """Output bundle for stage-1 preparation and split planning."""

    dataset: pd.DataFrame
    snapshot: MultiSymbolSnapshot
    splits: list[PurgedGroupSplit]
    split_summary: dict[str, int]


def load_or_build_dataset(
    *,
    snapshot_path: str | None = None,
    months: int = 18,
    interval: str | None = None,
    symbols: tuple[str, ...] | None = None,
    fail_fast: bool = False,
    root_dir: Path | None = None,
    client=None,
) -> tuple[pd.DataFrame, MultiSymbolSnapshot]:
    """Load existing snapshot or fetch/build a fresh universe snapshot."""

    if months <= 0:
        raise ValueError("months must be > 0")

    profile = get_runtime_profile()

    if snapshot_path:
        dataset, manifest = load_multi_symbol_snapshot(snapshot_path)
        validate_multi_symbol_ohlcv(dataset)
        snap_path = Path(snapshot_path).expanduser()
        manifest_path = snap_path.with_suffix("").with_suffix(".manifest.json")
        snapshot = MultiSymbolSnapshot(
            parquet_path=snap_path,
            manifest_path=manifest_path,
            manifest=manifest or {},
        )
        return dataset, snapshot

    selected_symbols = symbols or profile.universe.symbols
    selected_interval = interval or profile.universe.anchor_interval

    date_to = datetime.now(timezone.utc)
    date_from = date_to - timedelta(days=months * 30)

    dataset = fetch_universe_dataset(
        selected_symbols,
        date_from=date_from,
        date_to=date_to,
        interval=selected_interval,
        fail_fast=fail_fast,
        client=client,
    )
    if dataset.empty:
        raise RuntimeError("Fetched universe dataset is empty")
    validate_multi_symbol_ohlcv(dataset, expected_symbols=selected_symbols if fail_fast else None)

    snapshot = save_multi_symbol_snapshot(
        dataset,
        dataset_name=f"stage1_{selected_interval}",
        metadata={
            "symbols": list(selected_symbols),
            "interval": selected_interval,
            "months": months,
            "fail_fast": fail_fast,
        },
        root_dir=root_dir,
    )
    return dataset, snapshot


def build_stage1_result(
    dataset: pd.DataFrame,
    snapshot: MultiSymbolSnapshot,
    *,
    n_time_splits: int = 5,
    symbol_cluster_size: int = 2,
    embargo_bars: int = 24,
    min_train_rows: int = 100,
) -> Stage1Result:
    """Build group-purged validation splits for stage-1 multi-symbol setup."""

    validate_multi_symbol_ohlcv(dataset)

    splits = iter_purged_group_splits(
        dataset,
        n_time_splits=n_time_splits,
        symbol_cluster_size=symbol_cluster_size,
        embargo_bars=embargo_bars,
        min_train_rows=min_train_rows,
    )
    summary = summarize_split_coverage(splits)

    return Stage1Result(
        dataset=dataset,
        snapshot=snapshot,
        splits=splits,
        split_summary=summary,
    )
