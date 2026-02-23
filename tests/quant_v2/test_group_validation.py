from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import quant_v2.research.group_validation as gv


def _build_raw_df(n_ts: int = 48, symbols: tuple[str, ...] = ("BTCUSDT", "ETHUSDT")) -> pd.DataFrame:
    ts = pd.date_range("2025-01-01", periods=n_ts, freq="1h", tz="UTC")
    rows = []
    for symbol_idx, symbol in enumerate(symbols):
        base = 100.0 + symbol_idx * 20.0
        for i, t in enumerate(ts):
            close = base + i
            rows.append(
                {
                    "timestamp": t,
                    "symbol": symbol,
                    "open": close - 0.2,
                    "high": close + 0.4,
                    "low": close - 0.5,
                    "close": close,
                    "volume": 1000.0 + i,
                    "taker_buy_volume": 500.0 + (i % 5),
                    "taker_sell_volume": 500.0 - (i % 5),
                }
            )

    df = pd.DataFrame(rows).set_index(["timestamp", "symbol"]).sort_index()
    return df


def _build_prepared_df(n_ts: int = 72, symbols: tuple[str, ...] = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT")) -> pd.DataFrame:
    ts = pd.date_range("2025-01-01", periods=n_ts, freq="1h", tz="UTC")
    frames = []
    for symbol_idx, symbol in enumerate(symbols):
        base = 100.0 + symbol_idx * 10.0
        closes = np.array([base + i for i in range(n_ts)], dtype=float)
        future = np.roll(closes, -1)

        label = np.where(future > closes, 1, 0).astype(int)
        label[-1] = -1

        feat1 = label.astype(float)
        feat1[label == -1] = 0.5

        df_symbol = pd.DataFrame(
            {
                "open": closes - 0.2,
                "high": closes + 0.4,
                "low": closes - 0.5,
                "close": closes,
                "volume": np.full(n_ts, 1000.0 + symbol_idx),
                "taker_buy_volume": np.full(n_ts, 500.0),
                "taker_sell_volume": np.full(n_ts, 500.0),
                "f1": feat1,
                "label_1m": label,
            },
            index=ts,
        )
        df_symbol["symbol"] = symbol
        df_symbol = df_symbol.reset_index().set_index(["index", "symbol"]).rename_axis(["timestamp", "symbol"])
        frames.append(df_symbol)

    return pd.concat(frames).sort_index()


def test_prepare_multi_symbol_dataset(monkeypatch) -> None:
    raw = _build_raw_df()

    def fake_build_features(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["f1"] = np.linspace(0.0, 1.0, len(out))
        return out

    def fake_add_labels(df: pd.DataFrame, horizons):
        out = df.copy()
        for h in horizons:
            out[f"label_{h}m"] = 1
        return out

    monkeypatch.setattr(gv, "build_features", fake_build_features)
    monkeypatch.setattr(gv, "add_labels", fake_add_labels)

    prepared = gv.prepare_multi_symbol_dataset(raw, horizons=[1, 4])

    assert isinstance(prepared.index, pd.MultiIndex)
    assert list(prepared.index.names) == ["timestamp", "symbol"]
    assert "f1" in prepared.columns
    assert "label_1m" in prepared.columns
    assert "label_4m" in prepared.columns
    assert "xs_ret_1h_z" in prepared.columns
    assert "xs_volume_rank" in prepared.columns
    assert "xs_dispersion_ret_1h" in prepared.columns


def test_run_group_purged_validation(monkeypatch) -> None:
    df = _build_prepared_df()

    class DummyModel:
        feature_names = ["f1"]
        feature_importance = {"f1": 1.0}

    def fake_train(X_train, y_train, horizon, params_override=None):
        return DummyModel()

    def fake_predict(model, X):
        return np.where(X["f1"].to_numpy() > 0.5, 0.8, 0.2)

    monkeypatch.setattr(gv.model_trainer, "train", fake_train)
    monkeypatch.setattr(gv, "predict_proba", fake_predict)

    result = gv.run_group_purged_validation(
        df,
        horizon=1,
        n_time_splits=4,
        symbol_cluster_size=2,
        embargo_bars=1,
        min_train_rows=20,
    )

    assert result.validation_mode == "group_purged_cpcv"
    assert len(result.folds) > 0
    assert result.overall["n_trades"] > 0
    assert "deflated_sharpe_ratio" in result.robustness
    assert result.split_summary["n_splits"] >= len(result.folds)


def test_run_group_purged_validation_uses_injected_train_predict() -> None:
    df = _build_prepared_df()
    calls = {"train": 0, "predict": 0}

    class DummyModel:
        pass

    def injected_train(X_train, y_train, horizon, params_override=None):
        calls["train"] += 1
        return DummyModel()

    def injected_predict(model, X):
        calls["predict"] += 1
        return np.where(X["f1"].to_numpy() > 0.5, 0.8, 0.2)

    result = gv.run_group_purged_validation(
        df,
        horizon=1,
        n_time_splits=4,
        symbol_cluster_size=2,
        embargo_bars=1,
        min_train_rows=20,
        train_fn=injected_train,
        predict_fn=injected_predict,
    )

    assert len(result.folds) > 0
    assert calls["train"] > 0
    assert calls["predict"] > 0


def test_run_group_purged_validation_with_precomputed_splits(monkeypatch) -> None:
    df = _build_prepared_df()

    class DummyModel:
        feature_names = ["f1"]
        feature_importance = {"f1": 1.0}

    def fake_train(X_train, y_train, horizon, params_override=None):
        return DummyModel()

    def fake_predict(model, X):
        return np.where(X["f1"].to_numpy() > 0.5, 0.8, 0.2)

    monkeypatch.setattr(gv.model_trainer, "train", fake_train)
    monkeypatch.setattr(gv, "predict_proba", fake_predict)

    splits = gv.iter_purged_group_splits(
        df,
        n_time_splits=4,
        symbol_cluster_size=2,
        embargo_bars=1,
        min_train_rows=20,
    )
    split_summary = gv.summarize_split_coverage(splits)

    result = gv.run_group_purged_validation(
        df,
        horizon=1,
        min_train_rows=20,
        precomputed_splits=splits,
        split_summary=split_summary,
    )

    assert result.validation_mode == "group_purged_cpcv"
    assert result.split_summary == split_summary
    assert len(result.folds) > 0


def test_run_group_purged_validation_missing_label_raises() -> None:
    df = _build_prepared_df().drop(columns=["label_1m"])

    with pytest.raises(ValueError):
        gv.run_group_purged_validation(df, horizon=1)
