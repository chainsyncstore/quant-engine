from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import pytest

from quant_v2.data.multi_symbol_dataset import fetch_symbol_dataset, fetch_universe_dataset
from quant_v2.data.storage import (
    DataQualityError,
    load_multi_symbol_snapshot,
    save_multi_symbol_snapshot,
    validate_multi_symbol_ohlcv,
)


class FakeClient:
    def __init__(self, failing_symbol: str | None = None) -> None:
        self.failing_symbol = failing_symbol

    def fetch_historical(self, date_from, date_to, symbol=None, interval=None):
        if symbol == self.failing_symbol:
            raise RuntimeError(f"forced error for {symbol}")

        idx = pd.date_range(date_from, periods=4, freq="1h", tz="UTC")
        base = 100.0 if symbol == "BTCUSDT" else 50.0
        return pd.DataFrame(
            {
                "open": [base, base + 1, base + 2, base + 3],
                "high": [base + 1, base + 2, base + 3, base + 4],
                "low": [base - 1, base, base + 1, base + 2],
                "close": [base + 0.5, base + 1.5, base + 2.5, base + 3.5],
                "volume": [1000.0, 1100.0, 900.0, 950.0],
                "taker_buy_volume": [500.0, 600.0, 450.0, 470.0],
                "taker_sell_volume": [500.0, 500.0, 450.0, 480.0],
            },
            index=idx,
        )

    def fetch_funding_rates(self, date_from, date_to, symbol=None):
        idx = pd.date_range(date_from, periods=2, freq="2h", tz="UTC")
        return pd.DataFrame({"funding_rate_raw": [0.0001, 0.0002]}, index=idx)

    def fetch_open_interest(self, date_from, date_to, symbol=None, period="1h"):
        idx = pd.date_range(date_from, periods=4, freq="1h", tz="UTC")
        return pd.DataFrame(
            {
                "open_interest": [100000.0, 101000.0, 102000.0, 103000.0],
                "open_interest_value": [1.0e7, 1.01e7, 1.02e7, 1.03e7],
            },
            index=idx,
        )


def _date_bounds() -> tuple[datetime, datetime]:
    date_from = datetime(2025, 1, 1, tzinfo=timezone.utc)
    date_to = datetime(2025, 1, 3, tzinfo=timezone.utc)
    return date_from, date_to


def test_fetch_symbol_dataset_merges_supplementary_columns() -> None:
    date_from, date_to = _date_bounds()
    df = fetch_symbol_dataset(
        "BTCUSDT",
        date_from=date_from,
        date_to=date_to,
        client=FakeClient(),
    )

    assert not df.empty
    assert "funding_rate_raw" in df.columns
    assert "open_interest" in df.columns
    assert df.index.name == "timestamp"


def test_fetch_universe_dataset_builds_multi_index() -> None:
    date_from, date_to = _date_bounds()
    df = fetch_universe_dataset(
        ["BTCUSDT", "ETHUSDT"],
        date_from=date_from,
        date_to=date_to,
        client=FakeClient(),
    )

    assert isinstance(df.index, pd.MultiIndex)
    assert list(df.index.names) == ["timestamp", "symbol"]
    symbols = sorted(set(df.index.get_level_values("symbol")))
    assert symbols == ["BTCUSDT", "ETHUSDT"]


def test_fetch_universe_dataset_skips_failed_symbol_when_not_fail_fast() -> None:
    date_from, date_to = _date_bounds()
    df = fetch_universe_dataset(
        ["BTCUSDT", "ETHUSDT"],
        date_from=date_from,
        date_to=date_to,
        fail_fast=False,
        client=FakeClient(failing_symbol="ETHUSDT"),
    )

    symbols = sorted(set(df.index.get_level_values("symbol")))
    assert symbols == ["BTCUSDT"]


def test_fetch_universe_dataset_raises_when_fail_fast() -> None:
    date_from, date_to = _date_bounds()

    with pytest.raises(RuntimeError):
        fetch_universe_dataset(
            ["BTCUSDT", "ETHUSDT"],
            date_from=date_from,
            date_to=date_to,
            fail_fast=True,
            client=FakeClient(failing_symbol="ETHUSDT"),
        )


def test_validate_and_save_load_multi_symbol_snapshot(tmp_path) -> None:
    date_from, date_to = _date_bounds()
    df = fetch_universe_dataset(
        ["BTCUSDT", "ETHUSDT"],
        date_from=date_from,
        date_to=date_to,
        client=FakeClient(),
    )

    validate_multi_symbol_ohlcv(df, expected_symbols=("BTCUSDT", "ETHUSDT"))

    snap = save_multi_symbol_snapshot(
        df,
        dataset_name="universe_1h",
        metadata={"source": "unit_test"},
        root_dir=tmp_path,
    )

    loaded_df, manifest = load_multi_symbol_snapshot(snap.parquet_path)

    pd.testing.assert_frame_equal(df, loaded_df)
    assert manifest is not None
    assert manifest["dataset_name"] == "universe_1h"
    assert manifest["n_symbols"] == 2
    assert manifest["metadata"]["source"] == "unit_test"


def test_validate_multi_symbol_ohlcv_rejects_missing_columns() -> None:
    idx = pd.MultiIndex.from_product(
        [pd.date_range("2025-01-01", periods=2, freq="1h", tz="UTC"), ["BTCUSDT"]],
        names=["timestamp", "symbol"],
    )
    df = pd.DataFrame({"open": [1.0, 2.0], "close": [1.1, 2.1]}, index=idx)

    with pytest.raises(DataQualityError):
        validate_multi_symbol_ohlcv(df)


def test_validate_multi_symbol_ohlcv_rejects_continuity_gap() -> None:
    ts = pd.DatetimeIndex(
        [
            pd.Timestamp("2025-01-01T00:00:00Z"),
            pd.Timestamp("2025-01-01T01:00:00Z"),
            pd.Timestamp("2025-01-01T03:00:00Z"),
        ]
    )
    idx = pd.MultiIndex.from_arrays([ts, ["BTCUSDT", "BTCUSDT", "BTCUSDT"]], names=["timestamp", "symbol"])
    df = pd.DataFrame(
        {
            "open": [100.0, 101.0, 102.0],
            "high": [101.0, 102.0, 103.0],
            "low": [99.0, 100.0, 101.0],
            "close": [100.5, 101.5, 102.5],
            "volume": [1000.0, 1001.0, 1002.0],
        },
        index=idx,
    )

    with pytest.raises(DataQualityError):
        validate_multi_symbol_ohlcv(df)


def test_validate_multi_symbol_ohlcv_rejects_close_spike_outlier() -> None:
    idx = pd.MultiIndex.from_product(
        [pd.date_range("2025-01-01", periods=4, freq="1h", tz="UTC"), ["BTCUSDT"]],
        names=["timestamp", "symbol"],
    )
    df = pd.DataFrame(
        {
            "open": [100.0, 101.0, 250.0, 251.0],
            "high": [101.0, 102.0, 251.0, 252.0],
            "low": [99.0, 100.0, 249.0, 250.0],
            "close": [100.5, 101.5, 250.5, 251.5],
            "volume": [1000.0, 1100.0, 1200.0, 1300.0],
        },
        index=idx,
    )

    with pytest.raises(DataQualityError):
        validate_multi_symbol_ohlcv(df)


def test_validate_multi_symbol_ohlcv_rejects_stale_open_interest_series() -> None:
    idx = pd.MultiIndex.from_product(
        [pd.date_range("2025-01-01", periods=30, freq="1h", tz="UTC"), ["BTCUSDT"]],
        names=["timestamp", "symbol"],
    )
    df = pd.DataFrame(
        {
            "open": [100.0 + i for i in range(30)],
            "high": [101.0 + i for i in range(30)],
            "low": [99.0 + i for i in range(30)],
            "close": [100.5 + i for i in range(30)],
            "volume": [1000.0 + i for i in range(30)],
            "open_interest": [100_000.0] * 30,
            "open_interest_value": [10_000_000.0] * 30,
        },
        index=idx,
    )

    with pytest.raises(DataQualityError):
        validate_multi_symbol_ohlcv(df)
