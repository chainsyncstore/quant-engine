from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from quant_v2.model_registry import ModelRegistry
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

    monkeypatch.setenv("RETRAIN_MIN_TRAIN_ROWS", "100")
    monkeypatch.setattr(scheduled_retrain, "BinanceClient", lambda: object())
    monkeypatch.setattr(
        scheduled_retrain,
        "fetch_symbol_dataset",
        lambda *args, **kwargs: pd.DataFrame({"close": featured["close"]}),
    )
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
    monkeypatch.setattr(scheduled_retrain, "train", lambda *args, **kwargs: object())
    monkeypatch.setattr(scheduled_retrain, "_walk_forward_cv", lambda *args, **kwargs: 0.61)
    monkeypatch.setattr(scheduled_retrain, "_validate_model_single", lambda *args, **kwargs: 0.60)

    def _save_model(model, path: Path) -> None:  # noqa: ANN001
        _ = model
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("placeholder", encoding="utf-8")

    monkeypatch.setattr(scheduled_retrain, "save_model", _save_model)


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
    assert record is not None
    assert record.status == "paper_quarantine"
    assert record.metrics["promotion_eligible"] is True
    assert record.metrics["paper_quarantine_required"] is True
    assert record.metrics["train_rows"] >= 700
    assert record.metrics["horizons_trained"] == [2, 4, 8]
    assert record.metrics["symbols_fetched"] == ["BTCUSDT"]


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
