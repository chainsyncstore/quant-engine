"""
Shared test fixtures — synthetic market data generators.
"""

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic_ohlcv() -> pd.DataFrame:
    """Generate 1000 bars of synthetic OHLCV data for feature/label tests."""
    rng = np.random.default_rng(42)
    n = 1000
    timestamps = []
    current = datetime(2025, 12, 1, 8, 0, tzinfo=timezone.utc)

    while len(timestamps) < n:
        if current.weekday() < 5 and 8 <= current.hour < 21:
            timestamps.append(current)
        current += timedelta(minutes=1)

    price = 1.0850
    records = []
    for ts in timestamps:
        vol = 0.00015
        ret = rng.normal(0, vol)
        o = price
        c = o + ret
        h = max(o, c) + abs(ret) * rng.uniform(0.2, 1.0)
        l = min(o, c) - abs(ret) * rng.uniform(0.2, 1.0)
        volume = max(1.0, rng.normal(100, 30))
        taker_buy = volume * rng.uniform(0.35, 0.65)
        taker_sell = volume - taker_buy
        records.append(
            {
                "timestamp": ts,
                "open": o,
                "high": h,
                "low": l,
                "close": c,
                "volume": volume,
                "taker_buy_volume": taker_buy,
                "taker_sell_volume": taker_sell,
            }
        )
        price = c

    df = pd.DataFrame(records).set_index("timestamp")
    df.index = pd.DatetimeIndex(df.index, tz="UTC")
    return df


@pytest.fixture
def large_synthetic_ohlcv() -> pd.DataFrame:
    """Generate 35000 bars — enough for walk-forward validation."""
    rng = np.random.default_rng(42)
    n = 35000
    timestamps = []
    current = datetime(2025, 9, 1, 8, 0, tzinfo=timezone.utc)

    while len(timestamps) < n:
        if current.weekday() < 5 and 8 <= current.hour < 21:
            timestamps.append(current)
        current += timedelta(minutes=1)

    price = 1.0850
    records = []
    for ts in timestamps:
        regime_cycle = np.sin(2 * np.pi * len(records) / 5000) * 0.5 + 0.5
        vol = 0.0001 + regime_cycle * 0.0003
        ret = rng.normal(0, vol)
        o = price
        c = o + ret
        h = max(o, c) + abs(ret) * rng.uniform(0.1, 1.0)
        l = min(o, c) - abs(ret) * rng.uniform(0.1, 1.0)
        volume = max(1.0, rng.normal(100, 30))
        taker_buy = volume * rng.uniform(0.35, 0.65)
        taker_sell = volume - taker_buy
        records.append(
            {
                "timestamp": ts,
                "open": o,
                "high": h,
                "low": l,
                "close": c,
                "volume": volume,
                "taker_buy_volume": taker_buy,
                "taker_sell_volume": taker_sell,
            }
        )
        price = c

    df = pd.DataFrame(records).set_index("timestamp")
    df.index = pd.DatetimeIndex(df.index, tz="UTC")
    return df
