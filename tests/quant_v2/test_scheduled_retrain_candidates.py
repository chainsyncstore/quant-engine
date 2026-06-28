from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import json

import pandas as pd

from quant_v2.model_registry import ModelRegistry
from quant_v2.data import multi_symbol_dataset
from quant_v2.models.trainer import save_model_bundle as real_save_model_bundle, train as real_train
from quant_v2.research import scheduled_retrain


def _featured_frame(rows: int = 720) -> pd.DataFrame:
    timestamps = pd.date_range(
        end=datetime.now(timezone.utc),
        periods=rows,
        freq="h",
        tz="UTC",
    )
    close = [100.0 + idx * 0.2 for idx in range(rows)]
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "close": close,
            "feature_a": pd.Series(close).pct_change().fillna(0.0).to_list(),
        }
    )


def _patch_fast_retrain(monkeypatch, tmp_path: Path) -> None:
    featured = _featured_frame()

    class _FakeClient:
        def fetch_historical(self, date_from, date_to, symbol=None, interval=None):
            dates = pd.DatetimeIndex(featured["timestamp"])
            return pd.DataFrame(
                {
                    "open": featured["close"].to_numpy(),
                    "high": featured["close"].to_numpy() + 0.5,
                    "low": featured["close"].to_numpy() - 0.5,
                    "close": featured["close"].to_numpy(),
                    "volume": [1000.0] * len(featured),
                    "taker_buy_volume": [500.0] * len(featured),
                    "taker_sell_volume": [500.0] * len(featured),
                },
                index=dates,
            )

        def fetch_funding_rates(self, date_from, date_to, symbol=None):
            return pd.DataFrame(
                {
                    "funding_rate_raw": [0.0] * len(featured),
                },
                index=pd.DatetimeIndex(featured["timestamp"]),
            )

        def fetch_open_interest(self, date_from, date_to, symbol=None, period="1h"):
            return pd.DataFrame(
                {
                    "open_interest": [1_000.0] * len(featured),
                    "open_interest_value": [100_000.0] * len(featured),
                },
                index=pd.DatetimeIndex(featured["timestamp"]),
            )

    def _universe_dataset(*args, **kwargs):
        dates = pd.DatetimeIndex(featured["timestamp"])
        frame = pd.DataFrame(
            {
                "open": featured["close"].to_numpy(),
                "high": featured["close"].to_numpy() + 0.5,
                "low": featured["close"].to_numpy() - 0.5,
                "close": featured["close"].to_numpy(),
                "volume": [1000.0] * len(featured),
                "taker_buy_volume": [500.0] * len(featured),
                "taker_sell_volume": [500.0] * len(featured),
            },
            index=dates,
        )
        frame.index.name = "timestamp"
        frame["symbol"] = "BTCUSDT"
        frame = frame.reset_index().set_index(["timestamp", "symbol"]).sort_index()
        return frame

    def _symbol_dataset(*args, **kwargs):
        return _universe_dataset(*args, **kwargs)

    monkeypatch.setenv("RETRAIN_MIN_TRAIN_ROWS", "100")
    monkeypatch.setattr(scheduled_retrain, "BinanceClient", lambda: _FakeClient())
    monkeypatch.setattr(scheduled_retrain, "fetch_universe_dataset", _universe_dataset)
    monkeypatch.setattr(scheduled_retrain, "fetch_symbol_dataset", _symbol_dataset)
    monkeypatch.setattr(multi_symbol_dataset, "fetch_universe_dataset", _universe_dataset)
    monkeypatch.setattr(multi_symbol_dataset, "fetch_symbol_dataset", _symbol_dataset)
    monkeypatch.setattr(scheduled_retrain, "build_features", lambda raw: featured.copy())
    monkeypatch.setattr(
        scheduled_retrain,
        "get_feature_columns",
        lambda frame: ["close", "feature_a"],
    )
    monkeypatch.setattr(
        scheduled_retrain,
        "get_research_config",
        lambda: SimpleNamespace(wf_calibration_frac=0.2),
    )
    bundle_X = pd.DataFrame(
        {
            "close": featured["close"].to_numpy(),
            "feature_a": featured["feature_a"].to_numpy(),
        },
        index=pd.DatetimeIndex(featured["timestamp"]),
    )
    bundle_y = pd.Series(
        (pd.Series(featured["close"]).pct_change().fillna(0.0).to_numpy() > 0.0).astype(int),
        index=bundle_X.index,
    )
    prebuilt_models = {
        horizon: real_train(bundle_X, bundle_y, horizon=horizon, calibration_frac=0.2)
        for horizon in (2, 4, 8)
    }

    def _train(*args, **kwargs):
        horizon = int(kwargs.get("horizon", 4))
        return prebuilt_models[horizon]

    monkeypatch.setattr(scheduled_retrain, "train", _train)
    monkeypatch.setattr(scheduled_retrain, "_walk_forward_cv", lambda *args, **kwargs: 0.61)
    monkeypatch.setattr(scheduled_retrain, "_validate_model_single", lambda *args, **kwargs: 0.60)
    monkeypatch.setattr(
        scheduled_retrain,
        "_consume_temporal_validation_summary",
        lambda: {
            "selected": {
                "half_life_days": 60.0,
                "mean_accuracy": 0.61,
                "holdout_rows": 120,
                "threshold_policy": {
                    "source": "oof_dev_predictions",
                    "selected_threshold": 0.60,
                    "selected_accuracy": 0.63,
                    "threshold_min": 0.50,
                    "threshold_max": 0.80,
                    "threshold_step": 0.05,
                    "samples": 480,
                },
            },
            "holdout_rows": 120,
            "holdout_start": "2025-01-01T00:00:00+00:00",
            "holdout_end": "2025-01-06T00:00:00+00:00",
            "training_windows_months": [3, 6, 9, 12],
            "holdout_months": 3,
            "purge_bars": 100,
            "actual_trial_count": 4,
        },
    )

    monkeypatch.setattr(scheduled_retrain, "save_model", real_save_model_bundle)


def test_scheduled_retrain_registers_candidate_without_promoting_by_default(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.delenv("BOT_RETRAIN_AUTO_PROMOTE", raising=False)
    _patch_fast_retrain(monkeypatch, tmp_path)

    version_id = scheduled_retrain.retrain_and_promote(
        model_root=tmp_path / "models",
        registry_root=tmp_path / "registry",
        train_months=1,
        min_accuracy=0.55,
        extra_symbols=[],
    )

    registry = ModelRegistry(tmp_path / "registry")
    assert version_id is not None
    assert registry.get_active_version() is None
    record = registry.get_version(version_id)
    manifest = registry.get_artifact_manifest(version_id)
    assert record is not None
    assert record.status == "paper_quarantine"
    assert record.metrics["promotion_eligible"] is True
    assert record.metrics["paper_quarantine_required"] is True
    assert record.metrics["train_rows"] >= 700
    assert record.metrics["horizons_trained"] == [2, 4, 8]
    assert record.metrics["symbols_fetched"] == ["BTCUSDT"]
    assert manifest["training"]["threshold_policy"]["source"] == "oof_dev_predictions"
    assert manifest["training"]["threshold"] == 0.60
    assert manifest["training"]["calibration_policy"]["strategy"] == "fold_local_sigmoid_calibration"


def test_scheduled_retrain_auto_promote_requires_explicit_env(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("BOT_RETRAIN_AUTO_PROMOTE", "1")
    _patch_fast_retrain(monkeypatch, tmp_path)

    version_id = scheduled_retrain.retrain_and_promote(
        model_root=tmp_path / "models",
        registry_root=tmp_path / "registry",
        train_months=1,
        min_accuracy=0.55,
        extra_symbols=[],
    )

    registry = ModelRegistry(tmp_path / "registry")
    active = registry.get_active_version()
    assert version_id is not None
    assert active is not None
    assert active.version_id == version_id
    assert active.status == "active"
    assert active.promoted_by == "scheduled_retrain:auto_promote"


def test_scheduled_retrain_requires_dev_evidence_even_if_holdout_is_positive(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.delenv("BOT_RETRAIN_AUTO_PROMOTE", raising=False)
    _patch_fast_retrain(monkeypatch, tmp_path)
    monkeypatch.setattr(scheduled_retrain, "_walk_forward_cv", lambda *args, **kwargs: 0.0)
    monkeypatch.setattr(
        scheduled_retrain,
        "_consume_temporal_validation_summary",
        lambda: {
            "selected": {
                "half_life_days": 60.0,
                "mean_accuracy": 0.0,
                "holdout_accuracy": 0.96,
                "holdout_rows": 120,
            },
            "holdout_rows": 120,
            "holdout_start": "2025-01-01T00:00:00+00:00",
            "holdout_end": "2025-01-06T00:00:00+00:00",
            "training_windows_months": [3, 6, 9, 12],
            "holdout_months": 3,
            "purge_bars": 100,
            "actual_trial_count": 4,
        },
    )

    version_id = scheduled_retrain.retrain_and_promote(
        model_root=tmp_path / "models",
        registry_root=tmp_path / "registry",
        train_months=1,
        min_accuracy=0.55,
        extra_symbols=[],
    )

    registry = ModelRegistry(tmp_path / "registry")
    assert version_id is None
    assert registry.get_active_version() is None
    assert registry.list_versions() == []


def test_select_threshold_from_oof_predictions_prefers_best_accuracy() -> None:
    result = scheduled_retrain._select_threshold_from_oof_predictions(
        predictions=[0.2, 0.3, 0.6, 0.7, 0.9, 0.95],
        labels=[0, 0, 1, 1, 1, 1],
    )

    assert result["source"] == "oof_dev_predictions"
    assert result["selected_threshold"] >= 0.5
    assert result["selected_accuracy"] >= 0.8


def test_scheduled_retrain_validation_failure_leaves_no_visible_production_artifact(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.delenv("BOT_RETRAIN_AUTO_PROMOTE", raising=False)
    _patch_fast_retrain(monkeypatch, tmp_path)
    monkeypatch.setattr(
        scheduled_retrain,
        "validate_multi_symbol_ohlcv",
        lambda frame: (_ for _ in ()).throw(ValueError("stale ADAUSDT funding data")),
    )

    version_id = scheduled_retrain.retrain_and_promote(
        model_root=tmp_path / "models",
        registry_root=tmp_path / "registry",
        train_months=1,
        min_accuracy=0.55,
        extra_symbols=["ADAUSDT"],
    )

    assert version_id is None
    assert list((tmp_path / "models").glob("model_*")) == []
    failed_records = list((tmp_path / "models" / ".failed").glob("model_*.json"))
    assert len(failed_records) == 1
    payload = json.loads(failed_records[0].read_text(encoding="utf-8"))
    assert payload["reason"] == "dataset_validation_failed"
    assert "stale ADAUSDT funding data" in payload["details"]["error"]
