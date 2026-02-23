from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from quant_v2.research.build_universe_snapshot import build_universe_snapshot, resolve_symbols


class FakeClient:
    def fetch_historical(self, date_from, date_to, symbol=None, interval=None):
        idx = pd.date_range(date_from, periods=6, freq="1h", tz="UTC")
        base = 100.0 if symbol == "BTCUSDT" else 80.0
        return pd.DataFrame(
            {
                "open": [base + i for i in range(6)],
                "high": [base + i + 1.0 for i in range(6)],
                "low": [base + i - 1.0 for i in range(6)],
                "close": [base + i + 0.4 for i in range(6)],
                "volume": [1000.0] * 6,
                "taker_buy_volume": [500.0] * 6,
                "taker_sell_volume": [500.0] * 6,
            },
            index=idx,
        )

    def fetch_funding_rates(self, date_from, date_to, symbol=None):
        idx = pd.date_range(date_from, periods=2, freq="4h", tz="UTC")
        return pd.DataFrame({"funding_rate_raw": [0.0001, 0.00015]}, index=idx)

    def fetch_open_interest(self, date_from, date_to, symbol=None, period="1h"):
        idx = pd.date_range(date_from, periods=6, freq="1h", tz="UTC")
        return pd.DataFrame(
            {
                "open_interest": [1_000_000.0 + i * 1000 for i in range(6)],
                "open_interest_value": [10_000_000.0 + i * 5000 for i in range(6)],
            },
            index=idx,
        )


def test_resolve_symbols() -> None:
    assert resolve_symbols("btcusdt, ethusdt") == ("BTCUSDT", "ETHUSDT")


def test_build_universe_snapshot(tmp_path) -> None:
    date_from = datetime(2025, 1, 1, tzinfo=timezone.utc)
    date_to = datetime(2025, 1, 2, tzinfo=timezone.utc)

    result = build_universe_snapshot(
        date_from=date_from,
        date_to=date_to,
        symbols=("BTCUSDT", "ETHUSDT"),
        dataset_name="unit_universe",
        root_dir=tmp_path,
        fail_fast=True,
        client=FakeClient(),
    )

    assert not result.dataset.empty
    assert result.snapshot.parquet_path.exists()
    assert result.snapshot.manifest_path.exists()

    manifest = result.snapshot.manifest
    assert manifest["dataset_name"] == "unit_universe"
    assert manifest["n_symbols"] == 2
    assert sorted(manifest["symbols"]) == ["BTCUSDT", "ETHUSDT"]
