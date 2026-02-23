from __future__ import annotations

import pandas as pd

from quant_v2.research.event_labels import apply_event_aware_label_filters


def test_apply_event_aware_label_filters_masks_funding_and_volatility_events() -> None:
    idx = pd.date_range("2025-01-01", periods=8, freq="1h", tz="UTC")
    df = pd.DataFrame(
        {
            "open": [100, 101, 102, 103, 130, 131, 132, 133],
            "high": [101, 102, 103, 104, 131, 132, 133, 134],
            "low": [99, 100, 101, 102, 129, 130, 131, 132],
            "close": [100, 101, 102, 103, 130, 131, 132, 133],
            "volume": [1000] * 8,
            "funding_rate_raw": [0.0, 0.0, 0.002, 0.0, 0.0, 0.0, 0.0, 0.0],
            "label_1m": [1, 1, 1, 1, 1, 1, 1, -1],
        },
        index=idx,
    )

    out = apply_event_aware_label_filters(df, horizons=[1])

    assert "event_exclusion_flag" in out.columns
    assert "event_funding_window_flag" in out.columns
    assert "event_volatility_shock_flag" in out.columns
    assert (out["event_exclusion_flag"] >= 0).all()
    assert out.loc[idx[2], "label_1m"] == -1
    assert (out["label_1m"] == -1).sum() >= 2


def test_apply_event_aware_label_filters_noop_when_no_event_columns() -> None:
    idx = pd.date_range("2025-01-01", periods=4, freq="1h", tz="UTC")
    df = pd.DataFrame(
        {
            "close": [100.0, 101.0, 102.0, 103.0],
            "volume": [10.0, 11.0, 12.0, 13.0],
            "label_1m": [1, 0, 1, -1],
        },
        index=idx,
    )

    out = apply_event_aware_label_filters(df, horizons=[1])
    assert out["label_1m"].tolist()[-1] == -1
    assert "event_exclusion_flag" in out.columns
