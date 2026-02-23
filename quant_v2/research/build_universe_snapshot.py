"""Build and persist a multi-symbol v2 dataset snapshot."""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from quant_v2.config import default_universe_symbols, get_runtime_profile
from quant_v2.data.multi_symbol_dataset import fetch_universe_dataset
from quant_v2.data.storage import (
    MultiSymbolSnapshot,
    save_multi_symbol_snapshot,
    validate_multi_symbol_ohlcv,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BuildResult:
    """Output from a universe snapshot build run."""

    dataset: pd.DataFrame
    snapshot: MultiSymbolSnapshot


def resolve_symbols(symbols_csv: str | None) -> tuple[str, ...]:
    """Resolve symbol list from CLI input or v2 defaults."""

    if symbols_csv is None or not symbols_csv.strip():
        return default_universe_symbols()

    symbols = tuple(s.strip().upper() for s in symbols_csv.split(",") if s.strip())
    if not symbols:
        raise ValueError("No symbols resolved from --symbols input")
    return symbols


def build_universe_snapshot(
    *,
    date_from: datetime,
    date_to: datetime,
    symbols: tuple[str, ...] | None = None,
    interval: str | None = None,
    dataset_name: str = "universe_1h",
    include_funding: bool = True,
    include_open_interest: bool = True,
    fail_fast: bool = False,
    root_dir: Path | None = None,
    client=None,
) -> BuildResult:
    """Fetch, validate, and persist a v2 multi-symbol snapshot."""

    profile = get_runtime_profile()
    selected_symbols = symbols or profile.universe.symbols
    selected_interval = interval or profile.universe.anchor_interval

    dataset = fetch_universe_dataset(
        selected_symbols,
        date_from=date_from,
        date_to=date_to,
        interval=selected_interval,
        include_funding=include_funding,
        include_open_interest=include_open_interest,
        fail_fast=fail_fast,
        client=client,
    )
    if dataset.empty:
        raise RuntimeError("Fetched universe dataset is empty")

    validate_multi_symbol_ohlcv(dataset, expected_symbols=selected_symbols if fail_fast else None)

    snapshot = save_multi_symbol_snapshot(
        dataset,
        dataset_name=dataset_name,
        metadata={
            "interval": selected_interval,
            "symbols": list(selected_symbols),
            "include_funding": include_funding,
            "include_open_interest": include_open_interest,
            "fail_fast": fail_fast,
        },
        root_dir=root_dir,
    )

    logger.info(
        "Saved v2 snapshot %s (%d rows, %d symbols)",
        snapshot.parquet_path,
        len(dataset),
        snapshot.manifest.get("n_symbols", 0),
    )
    return BuildResult(dataset=dataset, snapshot=snapshot)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build v2 multi-symbol dataset snapshot")
    parser.add_argument("--months", type=int, default=18, help="History length in months")
    parser.add_argument("--symbols", type=str, default="", help="Comma-separated symbols")
    parser.add_argument("--interval", type=str, default="", help="Binance interval (default: profile anchor)")
    parser.add_argument("--dataset-name", type=str, default="universe_1h", help="Snapshot dataset name prefix")
    parser.add_argument("--no-funding", action="store_true", help="Skip funding-rate fetch")
    parser.add_argument("--no-open-interest", action="store_true", help="Skip open-interest fetch")
    parser.add_argument("--fail-fast", action="store_true", help="Fail on first symbol fetch error")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    date_to = datetime.now(timezone.utc)
    date_from = date_to - timedelta(days=args.months * 30)

    symbols = resolve_symbols(args.symbols)
    interval = args.interval.strip() or None

    result = build_universe_snapshot(
        date_from=date_from,
        date_to=date_to,
        symbols=symbols,
        interval=interval,
        dataset_name=args.dataset_name,
        include_funding=not args.no_funding,
        include_open_interest=not args.no_open_interest,
        fail_fast=args.fail_fast,
    )

    logger.info(
        "Build complete: %s (manifest=%s)",
        result.snapshot.parquet_path,
        result.snapshot.manifest_path,
    )


if __name__ == "__main__":
    main()
