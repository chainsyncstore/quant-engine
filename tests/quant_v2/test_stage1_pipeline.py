from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import pytest

from quant_v2.research.stage1_pipeline import build_stage1_result, load_or_build_dataset


class FakeClient:
    def fetch_historical(self, date_from, date_to, symbol=None, interval=None):
        idx = pd.date_range(date_from, periods=36, freq="1h", tz="UTC")
        base = 100.0 if symbol == "BTCUSDT" else 80.0
        return pd.DataFrame(
            {
                "open": [base + i for i in range(36)],
                "high": [base + i + 1.0 for i in range(36)],
                "low": [base + i - 1.0 for i in range(36)],
                "close": [base + i + 0.4 for i in range(36)],
                "volume": [1000.0] * 36,
                "taker_buy_volume": [500.0] * 36,
                "taker_sell_volume": [500.0] * 36,
            },
            index=idx,
        )

    def fetch_funding_rates(self, date_from, date_to, symbol=None):
        idx = pd.date_range(date_from, periods=6, freq="6h", tz="UTC")
        return pd.DataFrame({"funding_rate_raw": [0.0001] * 6}, index=idx)

    def fetch_open_interest(self, date_from, date_to, symbol=None, period="1h"):
        idx = pd.date_range(date_from, periods=36, freq="1h", tz="UTC")
        return pd.DataFrame(
            {
                "open_interest": [1_000_000.0 + i * 1000 for i in range(36)],
                "open_interest_value": [10_000_000.0 + i * 5000 for i in range(36)],
            },
            index=idx,
        )


def test_load_or_build_dataset_validate_months() -> None:
    with pytest.raises(ValueError):
        load_or_build_dataset(months=0, client=FakeClient())


def test_stage1_pipeline_builds_snapshot_and_splits(tmp_path) -> None:
    dataset, snapshot = load_or_build_dataset(
        months=1,
        symbols=("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"),
        interval="1h",
        fail_fast=True,
        root_dir=tmp_path,
        client=FakeClient(),
    )

    result = build_stage1_result(
        dataset,
        snapshot,
        n_time_splits=3,
        symbol_cluster_size=2,
        embargo_bars=1,
        min_train_rows=10,
    )

    assert not result.dataset.empty
    assert result.snapshot.parquet_path.exists()
    assert result.split_summary["n_splits"] == len(result.splits)
    assert result.split_summary["n_total_test_rows"] > 0


def test_load_or_build_dataset_from_snapshot_path(tmp_path) -> None:
    # First, build and persist a snapshot
    dataset, snapshot = load_or_build_dataset(
        months=1,
        symbols=("BTCUSDT", "ETHUSDT"),
        interval="1h",
        fail_fast=True,
        root_dir=tmp_path,
        client=FakeClient(),
    )

    loaded, loaded_snapshot = load_or_build_dataset(snapshot_path=str(snapshot.parquet_path), months=1)

    pd.testing.assert_frame_equal(dataset, loaded)
    assert loaded_snapshot.parquet_path == snapshot.parquet_path
