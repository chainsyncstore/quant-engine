"""v2 ensemble-oriented trainer with fold-local calibration support."""

from __future__ import annotations

import logging
import hashlib
import json
import os
import platform
import subprocess
from dataclasses import dataclass, field
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from lightgbm import early_stopping as lgb_early_stopping
from lightgbm import log_evaluation as lgb_log_evaluation
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import PredefinedSplit

from quant.config import get_research_config

logger = logging.getLogger(__name__)


@dataclass
class TrainedModel:
    """Container for the v2 trained stack (primary + optional calibration/meta)."""

    horizon: int
    feature_names: list[str]
    feature_importance: dict[str, float]
    primary_model: LGBMClassifier
    calibrated_model: CalibratedClassifierCV | None = None
    meta_model: LogisticRegression | None = None
    calibration_method: str = "sigmoid"
    fit_samples: int = 0
    calibration_samples: int = 0
    feature_dtypes: dict[str, str] = field(default_factory=dict)
    feature_catalog_version: str | None = None
    feature_catalog_sha256: str | None = None
    feature_missing_data_policy: str | None = None
    artifact_manifest: dict[str, Any] = field(default_factory=dict)


def train(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    horizon: int,
    calibration_frac: float | None = None,
    params_override: dict[str, Any] | None = None,
    sample_weight: np.ndarray | None = None,
) -> TrainedModel:
    """Train v2 model stack with fold-local calibration and meta refinement."""

    if X_train.empty:
        raise ValueError("X_train cannot be empty")
    if len(X_train) != len(y_train):
        raise ValueError("X_train and y_train must have same length")

    y = pd.Series(y_train).astype(int)
    unique_labels = set(int(v) for v in np.unique(y.to_numpy()))
    if not unique_labels.issubset({0, 1}):
        raise ValueError("y_train must be binary labels in {0, 1}")

    cfg = get_research_config()
    cal_frac = float(calibration_frac) if calibration_frac is not None else float(cfg.wf_calibration_frac)
    cal_frac = min(max(cal_frac, 0.05), 0.5)

    n_total = len(X_train)
    n_cal = int(round(n_total * cal_frac))
    n_cal = max(1, min(n_cal, max(n_total - 5, 1)))
    n_fit = max(n_total - n_cal, 1)

    lgbm_params = {
        "n_estimators": cfg.lgbm_n_estimators,
        "max_depth": cfg.lgbm_max_depth,
        "learning_rate": cfg.lgbm_learning_rate,
        "subsample": cfg.lgbm_subsample,
        "colsample_bytree": cfg.lgbm_colsample_bytree,
        "min_child_samples": cfg.lgbm_min_child_samples,
        "reg_alpha": cfg.lgbm_reg_alpha,
        "reg_lambda": cfg.lgbm_reg_lambda,
        "random_state": 42,
        "verbose": -1,
        "importance_type": "gain",
    }
    if params_override:
        lgbm_params.update(params_override)

    X_fit = X_train.iloc[:n_fit]
    y_fit = y.iloc[:n_fit]

    primary = LGBMClassifier(**lgbm_params)
    if sample_weight is not None:
        # Slice sample_weight to match fit fold
        sw_fit = sample_weight[:n_fit] if len(sample_weight) >= n_fit else None
    else:
        sw_fit = None

    # Early stopping: use calibration set as eval set to prevent over-training
    early_stop_rounds = getattr(cfg, 'lgbm_early_stopping_rounds', 0)
    X_cal_es = X_train.iloc[n_fit:]
    y_cal_es = y.iloc[n_fit:]
    fit_kwargs: dict[str, Any] = {"sample_weight": sw_fit}
    if early_stop_rounds > 0 and len(X_cal_es) >= 10:
        fit_kwargs["eval_set"] = [(X_cal_es.values, y_cal_es.values)]
        fit_kwargs["callbacks"] = [
            lgb_early_stopping(early_stop_rounds, verbose=False),
            lgb_log_evaluation(-1),
        ]
    primary.fit(X_fit.values, y_fit.values, **fit_kwargs)

    importances = np.asarray(primary.feature_importances_, dtype=float)
    if importances.size == 0:
        fi = {}
    else:
        norm = float(importances.sum())
        if norm <= 0.0:
            normed = np.zeros_like(importances)
        else:
            normed = importances / norm
        fi = dict(zip(X_train.columns, normed.tolist(), strict=False))
        fi = dict(sorted(fi.items(), key=lambda kv: kv[1], reverse=True))

    calibrated_model: CalibratedClassifierCV | None = None
    meta_model: LogisticRegression | None = None

    y_cal = y.iloc[n_fit:]
    use_fold_local_calibration = (
        n_fit >= 10
        and n_cal >= 10
        and len(set(int(v) for v in y_fit.to_numpy())) == 2
        and len(set(int(v) for v in y_cal.to_numpy())) == 2
    )

    if use_fold_local_calibration:
        split_index = np.array([-1] * n_fit + [0] * n_cal)
        split = PredefinedSplit(test_fold=split_index)

        calibrated_model = CalibratedClassifierCV(
            estimator=LGBMClassifier(**lgbm_params),
            method="sigmoid",
            cv=split,
        )
        calibrated_model.fit(X_train.values, y.values)

        # Meta-refinement model over primary probabilities.
        primary_cal_proba = primary.predict_proba(X_train.iloc[n_fit:].values)[:, 1]
        meta_features = primary_cal_proba.reshape(-1, 1)
        meta_model = LogisticRegression(max_iter=300, random_state=42)
        meta_model.fit(meta_features, y_cal.values)
    else:
        logger.info(
            "Skipping fold-local calibration/meta for horizon=%s due to insufficient split diversity "
            "(fit=%s, cal=%s)",
            horizon,
            n_fit,
            n_cal,
        )

    logger.info(
        "Trained v2 model_%sm: fit=%s cal=%s features=%s calibrated=%s meta=%s",
        horizon,
        n_fit,
        n_cal,
        len(X_train.columns),
        calibrated_model is not None,
        meta_model is not None,
    )

    feature_attrs = dict(getattr(X_train, "attrs", {}) or {})

    return TrainedModel(
        horizon=int(horizon),
        feature_names=list(X_train.columns),
        feature_importance=fi,
        primary_model=primary,
        calibrated_model=calibrated_model,
        meta_model=meta_model,
        fit_samples=n_fit,
        calibration_samples=n_cal,
        feature_dtypes={str(col): str(dtype) for col, dtype in X_train.dtypes.items()},
        feature_catalog_version=str(feature_attrs.get("feature_catalog_version") or "") or None,
        feature_catalog_sha256=str(feature_attrs.get("feature_catalog_sha256") or "") or None,
        feature_missing_data_policy=str(feature_attrs.get("feature_missing_data_policy") or "") or None,
    )


def save_model(trained: TrainedModel, path: Path) -> None:
    """Persist trained v2 model stack."""

    joblib.dump(trained, path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _git_sha() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=_repo_root(),
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return result.stdout.strip() or None


def _dependency_versions() -> dict[str, str]:
    versions: dict[str, str] = {"python": platform.python_version()}
    for package, key in (
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("scikit-learn", "scikit_learn"),
        ("lightgbm", "lightgbm"),
        ("joblib", "joblib"),
    ):
        try:
            versions[key] = importlib_metadata.version(package)
        except importlib_metadata.PackageNotFoundError:
            versions[key] = "unavailable"
    return versions


def _current_image_reference() -> str | None:
    for env_name in ("QUANT_IMAGE", "QUANT_IMAGE_DIGEST", "CODEX_IMAGE_DIGEST"):
        value = os.getenv(env_name)
        if value:
            return value.strip()
    return None


def build_model_artifact_manifest(
    trained: TrainedModel,
    path: Path,
    *,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a strict runtime manifest for a saved v2 model artifact."""

    path = Path(path).expanduser()
    payload = dict(metadata or {})
    checksum = _sha256_file(path)
    model_schema = {
        "feature_names": list(trained.feature_names),
        "feature_dtypes": dict(trained.feature_dtypes),
    }
    schema_digest = hashlib.sha256(
        json.dumps(model_schema, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    manifest = {
        "schema_version": "wp10-model-artifact-v1",
        "saved_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "artifact_path": str(path.resolve()),
        "artifact_name": path.name,
        "model": {
            "horizon": int(trained.horizon),
            "feature_names": list(trained.feature_names),
            "feature_schema_sha256": schema_digest,
            "feature_dtypes": dict(trained.feature_dtypes),
            "feature_catalog_version": trained.feature_catalog_version,
            "feature_catalog_sha256": trained.feature_catalog_sha256,
            "feature_missing_data_policy": trained.feature_missing_data_policy,
            "feature_importance": dict(trained.feature_importance),
            "calibration_method": trained.calibration_method,
            "fit_samples": int(trained.fit_samples),
            "calibration_samples": int(trained.calibration_samples),
            "has_calibrated_model": bool(trained.calibrated_model is not None),
            "has_meta_model": bool(trained.meta_model is not None),
        },
        "runtime": {
            "git_sha": _git_sha(),
            "image_reference": _current_image_reference(),
            "dependencies": _dependency_versions(),
        },
        "training": payload,
        "checksums": {
            "model_pickle_sha256": checksum,
        },
    }
    manifest["checksums"]["artifact_manifest_sha256"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return manifest


def save_model_bundle(
    trained: TrainedModel,
    path: Path,
    *,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Persist model and its strict manifest sidecar."""

    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    save_model(trained, path)
    manifest = build_model_artifact_manifest(trained, path, metadata=metadata)
    manifest_path = Path(path).with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    trained.artifact_manifest = manifest
    return manifest_path


def _manifest_path_for_model(path: Path) -> Path:
    return Path(path).with_suffix(".manifest.json")


def _validate_manifest_against_runtime(manifest: dict[str, Any], path: Path) -> None:
    runtime = manifest.get("runtime") or {}
    manifest_git_sha = runtime.get("git_sha")
    current_git_sha = _git_sha()
    if manifest_git_sha and current_git_sha and str(manifest_git_sha) != current_git_sha:
        raise ValueError(
            f"Artifact code mismatch for {path.name}: manifest git_sha={manifest_git_sha} current={current_git_sha}"
        )

    manifest_image = runtime.get("image_reference")
    current_image = _current_image_reference()
    if manifest_image and current_image and str(manifest_image) != current_image:
        raise ValueError(
            f"Artifact image mismatch for {path.name}: manifest image={manifest_image} current={current_image}"
        )

    manifest_deps = runtime.get("dependencies") or {}
    current_deps = _dependency_versions()
    for key, current_value in current_deps.items():
        manifest_value = manifest_deps.get(key)
        if manifest_value and str(manifest_value) != current_value:
            raise ValueError(
                f"Artifact dependency mismatch for {path.name}: {key} manifest={manifest_value} current={current_value}"
            )


def _manifest_checksum(manifest: dict[str, Any]) -> str:
    payload = json.loads(json.dumps(manifest, sort_keys=True, separators=(",", ":")))
    checksums = payload.get("checksums") or {}
    if isinstance(checksums, dict):
        checksums = dict(checksums)
        checksums.pop("artifact_manifest_sha256", None)
        payload["checksums"] = checksums
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def load_model(path: Path) -> TrainedModel:
    """Load trained v2 model stack from disk with strict manifest validation."""

    path = Path(path).expanduser()
    manifest_path = _manifest_path_for_model(path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing artifact manifest: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "wp10-model-artifact-v1":
        raise ValueError(f"Unsupported artifact manifest schema for {path.name}")

    expected_manifest_checksum = (manifest.get("checksums") or {}).get("artifact_manifest_sha256")
    actual_manifest_checksum = _manifest_checksum(manifest)
    if expected_manifest_checksum and str(expected_manifest_checksum) != actual_manifest_checksum:
        raise ValueError(
            f"Artifact manifest checksum mismatch for {path.name}: manifest={expected_manifest_checksum} actual={actual_manifest_checksum}"
        )

    checksums = manifest.get("checksums") or {}
    expected_checksum = checksums.get("model_pickle_sha256")
    actual_checksum = _sha256_file(path)
    if expected_checksum and str(expected_checksum) != actual_checksum:
        raise ValueError(
            f"Artifact checksum mismatch for {path.name}: manifest={expected_checksum} actual={actual_checksum}"
        )

    _validate_manifest_against_runtime(manifest, path)

    trained = joblib.load(path)
    if not isinstance(trained, TrainedModel):
        raise TypeError(f"Unexpected artifact type for {path.name}: {type(trained)!r}")

    model_manifest = manifest.get("model") or {}
    manifest_feature_names = list(model_manifest.get("feature_names") or [])
    if manifest_feature_names and list(trained.feature_names) != manifest_feature_names:
        raise ValueError(f"Artifact feature schema mismatch for {path.name}")

    manifest_feature_dtypes = dict(model_manifest.get("feature_dtypes") or {})
    if manifest_feature_dtypes and dict(trained.feature_dtypes) != manifest_feature_dtypes:
        raise ValueError(f"Artifact feature dtype mismatch for {path.name}")

    manifest_feature_catalog_version = model_manifest.get("feature_catalog_version")
    if manifest_feature_catalog_version and trained.feature_catalog_version:
        if str(manifest_feature_catalog_version) != str(trained.feature_catalog_version):
            raise ValueError(f"Artifact feature catalog version mismatch for {path.name}")

    manifest_feature_catalog_sha256 = model_manifest.get("feature_catalog_sha256")
    if manifest_feature_catalog_sha256 and trained.feature_catalog_sha256:
        if str(manifest_feature_catalog_sha256) != str(trained.feature_catalog_sha256):
            raise ValueError(f"Artifact feature catalog mismatch for {path.name}")

    manifest_feature_missing_data_policy = model_manifest.get("feature_missing_data_policy")
    if manifest_feature_missing_data_policy and trained.feature_missing_data_policy:
        if str(manifest_feature_missing_data_policy) != str(trained.feature_missing_data_policy):
            raise ValueError(f"Artifact feature missing-data policy mismatch for {path.name}")

    manifest_horizon = model_manifest.get("horizon")
    if manifest_horizon is not None and int(trained.horizon) != int(manifest_horizon):
        raise ValueError(f"Artifact horizon mismatch for {path.name}")

    trained.artifact_manifest = manifest
    return trained
