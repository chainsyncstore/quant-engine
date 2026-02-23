from __future__ import annotations

import pandas as pd
import pytest

from quant_v2.research.cross_sectional_features import add_cross_sectional_features


def _sample_df() -> pd.DataFrame:
    idx = pd.MultiIndex.from_product(
        [
            pd.date_range("2025-01-01", periods=4, freq="1h", tz="UTC"),
            ["BTCUSDT", "ETHUSDT", "SOLUSDT"],
        ],
        names=["timestamp", "symbol"],
    )
    return pd.DataFrame(
        {
            "open": [100, 80, 60, 101, 81, 59, 102, 79, 58, 103, 78, 57],
            "high": [101, 81, 61, 102, 82, 60, 103, 80, 59, 104, 79, 58],
            "low": [99, 79, 59, 100, 80, 58, 101, 78, 57, 102, 77, 56],
            "close": [100.5, 80.5, 60.5, 101.5, 81.0, 59.5, 102.2, 79.2, 58.3, 103.0, 78.0, 57.0],
            "volume": [1000, 900, 800, 1100, 920, 780, 1200, 940, 760, 1300, 960, 740],
        },
        index=idx,
    )


def test_add_cross_sectional_features_appends_expected_columns() -> None:
    df = _sample_df()

    out = add_cross_sectional_features(df)

    expected = {
        "xs_ret_1h_z",
        "xs_ret_1h_rank",
        "xs_volume_z",
        "xs_volume_rank",
        "xs_dispersion_ret_1h",
        "xs_breadth_up_ret_1h",
    }
    assert expected.issubset(set(out.columns))
    assert out["xs_ret_1h_z"].isna().sum() == 0
    assert out["xs_volume_z"].isna().sum() == 0
    assert ((out["xs_ret_1h_rank"] >= 0.0) & (out["xs_ret_1h_rank"] <= 1.0)).all()
    assert ((out["xs_breadth_up_ret_1h"] >= 0.0) & (out["xs_breadth_up_ret_1h"] <= 1.0)).all()


def test_add_cross_sectional_features_validates_index_shape() -> None:
    df = _sample_df().reset_index()

    with pytest.raises(ValueError):
        add_cross_sectional_features(df)
