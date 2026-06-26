from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quant_v2.research.model_quality_recovery import (
    QUALITY_RECOVERY_POLICY_VERSION,
    VALIDATION_POLICY_VERSION,
    build_benchmark_replay_report,
    build_failed_retrain_diagnostic,
    build_label_audit_report,
)


def _quality_dataset(rows: int = 120) -> pd.DataFrame:
    timestamps = pd.date_range("2026-01-01", periods=rows, freq="h", tz="UTC")
    symbols = ["BTCUSDT", "ETHUSDT"]
    frames = []
    for idx, symbol in enumerate(symbols):
        base = 100.0 + idx * 20.0
        phase = np.linspace(0.0, 8.0, rows)
        close = base + np.sin(phase) * 2.0 + np.linspace(0.0, 3.0, rows)
        volume = 1_000.0 + (np.cos(phase) * 50.0)
        funding = np.zeros(rows)
        funding[::8] = 0.0001 if idx == 0 else -0.00005
        frame = pd.DataFrame(
            {
                "open": close - 0.25,
                "high": close + 0.50,
                "low": close - 0.75,
                "close": close,
                "volume": volume,
                "funding_rate_raw": funding,
            },
            index=timestamps,
        )
        frame["symbol"] = symbol
        frames.append(frame.reset_index().set_index(["index", "symbol"]).rename_axis(["timestamp", "symbol"]))
    return pd.concat(frames).sort_index()


def test_failed_retrain_diagnostic_reports_latest_failure(tmp_path: Path) -> None:
    model_root = tmp_path / "models" / "production"
    registry_root = model_root / "registry"
    (model_root / ".failed").mkdir(parents=True, exist_ok=True)
    registry_root.mkdir(parents=True, exist_ok=True)
    (registry_root / "active.json").write_text(
        json.dumps({"version_id": "model_1", "updated_at": datetime.now(timezone.utc).isoformat()}),
        encoding="utf-8",
    )
    (registry_root / "registry_events.jsonl").write_text("event-1\n", encoding="utf-8")
    (model_root / ".failed" / "model_20260624_220737.json").write_text(
        json.dumps({"reason": "old_reason", "details": {"foo": "bar"}}),
        encoding="utf-8",
    )
    latest = model_root / ".failed" / "model_20260625_090308.json"
    latest.write_text(
        json.dumps({"reason": "no_valid_horizon_models", "details": {"missing_horizons": [2, 4, 8]}}),
        encoding="utf-8",
    )

    report = build_failed_retrain_diagnostic(model_root, registry_root)

    assert report["policy_version"] == QUALITY_RECOVERY_POLICY_VERSION
    assert report["failure_record"]["reason"] == "no_valid_horizon_models"
    assert report["failure_record"]["details"]["missing_horizons"] == [2, 4, 8]
    assert report["registry"]["registry_event_count"] == 1
    assert report["filesystem"]["failed_record_count"] == 2


def test_label_audit_reports_dead_zone_balance() -> None:
    dataset = _quality_dataset(rows=96)
    report = build_label_audit_report(dataset, horizons=(2,), dead_zones=(0.001, 0.005))

    horizon = report["horizons"]["2"]
    narrow = horizon["dead_zone_summaries"]["0.0010"]
    wide = horizon["dead_zone_summaries"]["0.0050"]

    assert report["policy_version"] == QUALITY_RECOVERY_POLICY_VERSION
    assert report["grid"]["training_windows_months"] == [3, 6, 9, 12]
    assert narrow["ambiguous_ratio"] <= wide["ambiguous_ratio"]
    assert "BTCUSDT" in narrow["by_symbol"]
    assert "BTCUSDT:2026-01" in narrow["by_symbol_month"]


def test_benchmark_replay_report_generates_actor_summaries() -> None:
    dataset = _quality_dataset(rows=120)
    report = build_benchmark_replay_report(dataset)

    assert report["policy_version"] == QUALITY_RECOVERY_POLICY_VERSION
    assert report["actor_summaries"].keys() == {
        "flat",
        "momentum",
        "mean_reversion",
        "volatility_filtered",
    }
    assert report["comparisons"]["best_actor"] in report["actor_summaries"]
    assert report["actor_summaries"]["flat"]["fill_count"] == 0
    assert "exposure_by_symbol_usd" in report["actor_summaries"]["momentum"]
    assert report["comparisons"]["best_minus_flat_cost_adjusted_net_pnl_usd"] == pytest.approx(
        report["comparisons"]["best_cost_adjusted_net_pnl_usd"] - report["comparisons"]["flat_cost_adjusted_net_pnl_usd"]
    )


def test_scheduled_retrain_manifest_versions_are_persisted(tmp_path: Path, monkeypatch) -> None:
    from quant_v2.research import scheduled_retrain

    featured = _quality_dataset(rows=220).reset_index()
    stronger_close = 100.0 + np.linspace(0.0, 80.0, len(featured))
    feature_frame = pd.DataFrame(
        {
            "close": stronger_close,
            "feature_a": np.linspace(-1.0, 1.0, len(featured)),
        },
        index=pd.DatetimeIndex(featured["timestamp"]),
    )
    labels = (feature_frame["close"].pct_change().fillna(0.0) > 0.0).astype(int)
    prebuilt = {
        horizon: scheduled_retrain.train(feature_frame, labels, horizon=horizon, sample_weight=None)
        for horizon in (2, 4, 8)
    }

    monkeypatch.setattr(scheduled_retrain, "BinanceClient", lambda: object())
    monkeypatch.setattr(scheduled_retrain, "fetch_universe_dataset", lambda *args, **kwargs: _quality_dataset(rows=220))
    monkeypatch.setattr(scheduled_retrain, "build_features", lambda raw: feature_frame.copy())
    monkeypatch.setattr(scheduled_retrain, "get_feature_columns", lambda frame: ["close", "feature_a"])
    monkeypatch.setattr(scheduled_retrain, "get_research_config", lambda: type("Cfg", (), {"wf_calibration_frac": 0.2})())
    monkeypatch.setattr(scheduled_retrain, "train", lambda *args, **kwargs: prebuilt[int(kwargs.get("horizon", 4))])
    monkeypatch.setattr(scheduled_retrain, "_walk_forward_cv", lambda *args, **kwargs: 0.61)
    monkeypatch.setattr(scheduled_retrain, "_validate_model_single", lambda *args, **kwargs: 0.60)
    monkeypatch.setenv("RETRAIN_MIN_TRAIN_ROWS", "100")
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
    monkeypatch.setenv("BOT_RETRAIN_AUTO_PROMOTE", "0")

    version_id = scheduled_retrain.retrain_and_promote(
        model_root=tmp_path / "models",
        registry_root=tmp_path / "registry",
        train_months=1,
        min_accuracy=0.55,
        extra_symbols=[],
    )

    assert version_id is not None
    registry = scheduled_retrain.ModelRegistry(tmp_path / "registry")
    manifest = registry.get_artifact_manifest(version_id)
    assert manifest["training"]["validation_policy"]["validation_policy_version"] == VALIDATION_POLICY_VERSION
    assert manifest["training"]["validation_policy"]["quality_recovery_policy_version"] == QUALITY_RECOVERY_POLICY_VERSION
