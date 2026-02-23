from __future__ import annotations

import pandas as pd

from quant_v2.research.regime_context import add_regime_context_features


def test_add_regime_context_features_appends_expected_columns() -> None:
    idx = pd.MultiIndex.from_product(
        [pd.date_range("2025-01-01", periods=30, freq="1h", tz="UTC"), ["BTCUSDT", "ETHUSDT"]],
        names=["timestamp", "symbol"],
    )
    df = pd.DataFrame(
        {
            "open": [100.0 + i * 0.1 for i in range(len(idx))],
            "high": [101.0 + i * 0.1 for i in range(len(idx))],
            "low": [99.0 + i * 0.1 for i in range(len(idx))],
            "close": [100.0 + i * 0.1 for i in range(len(idx))],
            "volume": [1000.0 + (i % 7) * 10 for i in range(len(idx))],
        },
        index=idx,
    )

    out = add_regime_context_features(df)

    assert "regime_trend_24h" in out.columns
    assert "regime_volatility_24h" in out.columns
    assert "regime_stress_24h" in out.columns
    assert "regime_high_vol_flag" in out.columns
    assert out["regime_high_vol_flag"].between(0.0, 1.0).all()
