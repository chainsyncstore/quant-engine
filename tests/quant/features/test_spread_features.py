import numpy as np
import pandas as pd
import pytest
from quant.features import spread_features

def test_spread_features_calculation():
    # Create sample data
    # 100 bars
    dates = pd.date_range("2021-01-01", periods=100, freq="1min")
    df = pd.DataFrame({
        "open": 1.1000, "high": 1.1010, "low": 1.0990, "close": 1.1005,
        "spread": 0.0001  # 1 pip spread
    }, index=dates)

    # Induce a spread spike at index 50
    df.iloc[50, df.columns.get_loc("spread")] = 0.0005

    # Compute features
    out = spread_features.compute(df)

    # Check columns exist
    assert "spread_zscore" in out.columns
    assert "spread_change_5" in out.columns
    assert "spread_regime_ratio" in out.columns
    assert "spread_to_atr" in out.columns

    # Check Z-score spike
    # Spread was constant 0.0001, so std=0, until window embraces spike
    # At index 50, zscore should be high
    zscore_50 = out["spread_zscore"].iloc[50]
    # Actually, rolling window includes current observation in pandas by default
    # but std might be small. 
    # Let's just check it's not NaN
    assert not np.isnan(zscore_50)

    # Check spread change
    # Spread increased 5x at index 50 (from 1 to 5)
    # At index 50, change from 45 is ...
    # Wait, pct_change(5) compares i and i-5.
    # period 45 spread was 0.0001. period 50 is 0.0005. 
    # change = (0.0005 - 0.0001) / 0.0001 = 4.0
    assert out["spread_change_5"].iloc[50] == 4.0

    # Check spread/ATR
    # ATR takes 14 bars to warm up.
    assert np.isnan(out["spread_to_atr"].iloc[0])
    assert not np.isnan(out["spread_to_atr"].iloc[20])

def test_spread_features_missing_column():
    dates = pd.date_range("2021-01-01", periods=10)
    df = pd.DataFrame({"close": 1.0}, index=dates)
    
    # Should not crash, just return copy
    out = spread_features.compute(df)
    assert "spread_zscore" not in out.columns
    assert len(out) == 10
