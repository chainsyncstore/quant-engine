from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import joblib
import pandas as pd
import pytest

from quant_v2.model_registry import ModelRegistry, write_model_manifest


torch = pytest.importorskip("torch")
from quant_v2.research import confirmation_shadow_export


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


def _write_active_artifact(model_root: Path, registry_root: Path) -> str:
    version_id = "model_active"
    artifact_dir = model_root / version_id
    artifact_dir.mkdir(parents=True)
    for horizon in (2, 4, 8):
        joblib.dump(
            SimpleNamespace(horizon=horizon, feature_names=[]),
            artifact_dir / f"model_{horizon}m.pkl",
        )
    metrics = {
        "promotion_eligible": True,
        "validation_scores": {"2": 0.61, "4": 0.61, "8": 0.61},
        "horizons_trained": [2, 4, 8],
        "required_horizons": [2, 4, 8],
    }
    write_model_manifest(
        artifact_dir,
        version_id=version_id,
        required_horizons=(2, 4, 8),
        metrics=metrics,
        source="test",
    )
    registry = ModelRegistry(registry_root, model_root=model_root)
    registry.register_version(
        version_id,
        artifact_dir,
        metrics=metrics,
        tags={"promotion_eligible": "true"},
    )
    registry.promote_version(version_id, promoted_by="test")
    return version_id


def test_export_active_confirmation_shadow_updates_active_manifest_and_metrics(
    tmp_path: Path,
    monkeypatch,
) -> None:
    model_root = tmp_path / "models"
    registry_root = tmp_path / "registry"
    version_id = _write_active_artifact(model_root, registry_root)
    featured = _featured_frame()

    monkeypatch.setenv("RETRAIN_CONFIRMATION_HORIZON", "4")
    monkeypatch.setenv("RETRAIN_CONFIRMATION_MIN_PROFIT_SAMPLES", "20")
    monkeypatch.setattr(confirmation_shadow_export, "BinanceClient", lambda: object())
    monkeypatch.setattr(
        confirmation_shadow_export,
        "fetch_symbol_dataset",
        lambda *args, **kwargs: pd.DataFrame({"close": featured["close"]}),
    )
    monkeypatch.setattr(
        confirmation_shadow_export,
        "build_features",
        lambda raw: featured.copy(),
    )
    monkeypatch.setattr(
        confirmation_shadow_export,
        "get_feature_columns",
        lambda frame: ["close", "feature_a"],
    )
    monkeypatch.setattr(confirmation_shadow_export, "load_model", lambda path: object())
    monkeypatch.setattr(
        confirmation_shadow_export,
        "_predict_model_proba",
        lambda model, X_test: pd.Series(
            [0.9 if idx % 2 == 0 else 0.1 for idx in range(len(X_test))],
            index=X_test.index,
        ),
    )

    result = confirmation_shadow_export.export_active_confirmation_shadow(
        model_root=model_root,
        registry_root=registry_root,
        train_months=1,
        extra_symbols=[],
    )

    artifact_dir = model_root / version_id
    assert result["version_id"] == version_id
    assert (artifact_dir / "confirmation" / "config.json").is_file()
    assert (artifact_dir / "confirmation" / "state_dict.pt").is_file()

    registry = ModelRegistry(registry_root, model_root=model_root)
    manifest = registry.validate_artifact_manifest(artifact_dir, smoke_load=True)
    kinds = {entry["kind"] for entry in manifest["files"]}
    assert "confirmation_config" in kinds
    assert "confirmation_state_dict" in kinds

    active = registry.get_active_version()
    assert active is not None
    assert active.version_id == version_id
    assert active.metrics["confirmation"]["available"] is True
    assert active.metrics["confirmation"]["runtime_mode"] == "shadow_only"
    assert active.metrics["confirmation"]["profitability"]["sample_count"] > 0
