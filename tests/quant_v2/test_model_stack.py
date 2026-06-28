from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_v2.models.predictor import predict_proba, predict_proba_with_uncertainty
from quant_v2.models.trainer import load_model, save_model_bundle, train
from quant.features.schema import FEATURE_CATALOG_SHA256, FEATURE_CATALOG_VERSION


def _sample_training_data(rows: int = 200) -> tuple[pd.DataFrame, pd.Series]:
    idx = pd.date_range("2025-01-01", periods=rows, freq="1h", tz="UTC")
    x1 = np.linspace(-2.0, 2.0, rows)
    x2 = np.sin(np.linspace(0.0, 10.0, rows))
    x3 = np.cos(np.linspace(0.0, 8.0, rows))

    logits = 0.9 * np.sin(np.linspace(0.0, 18.0, rows)) + 0.5 * x2 - 0.2 * x3
    y = (logits > 0.0).astype(int)

    X = pd.DataFrame({"f1": x1, "f2": x2, "f3": x3}, index=idx)
    return X, pd.Series(y, index=idx)


def test_v2_model_stack_train_exposes_primary_calibration_meta_layers() -> None:
    X, y = _sample_training_data(rows=220)

    trained = train(
        X,
        y,
        horizon=1,
        calibration_frac=0.2,
    )

    assert trained.horizon == 1
    assert trained.primary_model is not None
    assert trained.calibrated_model is not None
    assert trained.meta_model is not None
    assert trained.fit_samples > 0
    assert trained.calibration_samples > 0
    assert set(trained.feature_names) == {"f1", "f2", "f3"}
    assert trained.feature_importance


def test_v2_model_stack_predicts_probability_and_uncertainty() -> None:
    X, y = _sample_training_data(rows=180)
    trained = train(X, y, horizon=4, calibration_frac=0.2)

    batch = X.tail(25)
    proba = predict_proba(trained, batch)
    assert proba.shape == (25,)
    assert np.all(np.isfinite(proba))
    assert np.all((proba >= 0.0) & (proba <= 1.0))

    proba2, uncertainty = predict_proba_with_uncertainty(trained, batch)
    assert proba2.shape == (25,)
    assert uncertainty.shape == (25,)
    assert np.all((proba2 >= 0.0) & (proba2 <= 1.0))
    assert np.all((uncertainty >= 0.0) & (uncertainty <= 1.0))
    assert np.allclose(proba, proba2)


def test_v2_predictor_rejects_unexpected_features() -> None:
    X, y = _sample_training_data(rows=180)
    trained = train(X, y, horizon=4, calibration_frac=0.2)

    with pytest.raises(ValueError, match="Unexpected feature columns"):
        predict_proba(trained, X.assign(extra_feature=1.0))


def test_v2_model_stack_manifest_roundtrip_and_runtime_validation(tmp_path, monkeypatch) -> None:
    X, y = _sample_training_data(rows=220)
    monkeypatch.setenv("QUANT_IMAGE", "registry.example/quant-bot@sha256:" + "a" * 64)
    X.attrs.update(
        {
            "feature_catalog_version": FEATURE_CATALOG_VERSION,
            "feature_catalog_sha256": FEATURE_CATALOG_SHA256,
            "feature_missing_data_policy": "drop-core-and-derived-missing-v1",
        }
    )

    trained = train(X, y, horizon=4, calibration_frac=0.2)
    artifact = tmp_path / "model_4m.pkl"
    manifest_path = save_model_bundle(
        trained,
        artifact,
        metadata={
            "dataset_digest": "abc123",
            "symbols": ["BTCUSDT"],
            "horizons": [4],
            "validation_mode": "purged_kfold",
            "validation_cost_policy_version": "wp07-execution-cost-v1",
        },
    )

    assert artifact.exists()
    assert manifest_path.exists()

    loaded = load_model(artifact)
    assert loaded.horizon == trained.horizon
    assert loaded.artifact_manifest["schema_version"] == "wp10-model-artifact-v1"
    assert loaded.artifact_manifest["training"]["dataset_digest"] == "abc123"
    assert loaded.artifact_manifest["model"]["feature_catalog_version"] == FEATURE_CATALOG_VERSION
    assert loaded.artifact_manifest["model"]["feature_catalog_sha256"] == FEATURE_CATALOG_SHA256
    assert loaded.artifact_manifest["model"]["feature_missing_data_policy"] == "drop-core-and-derived-missing-v1"


def test_v2_model_stack_golden_row_parity_after_save_load(tmp_path, monkeypatch) -> None:
    X, y = _sample_training_data(rows=220)
    monkeypatch.setenv("QUANT_IMAGE", "registry.example/quant-bot@sha256:" + "a" * 64)

    trained = train(X, y, horizon=4, calibration_frac=0.2)
    artifact = tmp_path / "model_4m.pkl"
    save_model_bundle(trained, artifact)

    golden_row = X.tail(1)
    before = predict_proba(trained, golden_row)
    loaded = load_model(artifact)
    after = predict_proba(loaded, golden_row)

    assert np.allclose(before, after, atol=1e-12, rtol=0.0)


def test_v2_model_stack_manifest_rejects_checksum_and_image_mismatch(tmp_path, monkeypatch) -> None:
    X, y = _sample_training_data(rows=220)
    monkeypatch.setenv("QUANT_IMAGE", "registry.example/quant-bot@sha256:" + "a" * 64)
    trained = train(X, y, horizon=4, calibration_frac=0.2)
    artifact = tmp_path / "model_4m.pkl"
    save_model_bundle(trained, artifact)

    # Runtime image drift should fail closed before the model is used.
    monkeypatch.setenv("QUANT_IMAGE", "registry.example/quant-bot@sha256:" + "b" * 64)
    with pytest.raises(ValueError, match="Artifact image mismatch"):
        load_model(artifact)

    # Restore matching runtime identity, then tamper with the artifact bytes.
    monkeypatch.setenv("QUANT_IMAGE", "registry.example/quant-bot@sha256:" + "a" * 64)
    artifact.write_bytes(artifact.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="Artifact checksum mismatch"):
        load_model(artifact)


def test_v2_model_stack_load_rejects_missing_manifest(tmp_path, monkeypatch) -> None:
    X, y = _sample_training_data(rows=220)
    monkeypatch.setenv("QUANT_IMAGE", "registry.example/quant-bot@sha256:" + "a" * 64)
    trained = train(X, y, horizon=4, calibration_frac=0.2)
    artifact = tmp_path / "model_4m.pkl"
    from quant_v2.models.trainer import save_model

    save_model(trained, artifact)

    with pytest.raises(FileNotFoundError, match="Missing artifact manifest"):
        load_model(artifact)
