"""Scheduled weekly retrain: trains multi-horizon v2 models and registers candidates.

Designed to run inside the Docker container on a schedule (cron or asyncio timer).
The signal manager only hot-swaps after explicit manual promotion.

Usage (standalone):
    python -m quant_v2.research.scheduled_retrain

Environment:
    BOT_MODEL_ROOT        – model artifact root (default: /app/models/production)
    BOT_MODEL_REGISTRY_ROOT – registry root (default: {MODEL_ROOT}/registry)
    RETRAIN_INTERVAL_HOURS – hours between retrain runs (default: 168 = 7 days)
    RETRAIN_TRAIN_MONTHS   – months of training data (default: 12)
    RETRAIN_MIN_ACCURACY   – minimum accuracy to promote a model (default: 0.525)
    RETRAIN_TRAIN_SYMBOLS  – comma-separated extra symbols to include in training (default: full universe from default_universe_symbols() minus BTCUSDT)
    RETRAIN_REQUIRE_ALL_SYMBOLS - require every requested symbol before promotion (default: true)
    RETRAIN_MIN_TRAIN_ROWS - minimum featured rows required before promotion (default: 80% of expected hourly rows)
    RETRAIN_REQUIRE_ALL_HORIZONS - require every horizon before promotion (default: true)
    BOT_RETRAIN_AUTO_PROMOTE - emergency legacy auto-promotion override (default: 0)
"""

from __future__ import annotations

import logging
import json
import os
import shutil
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from quant.config import get_research_config
from quant.data.binance_client import BinanceClient
from quant.features.pipeline import build_features, get_feature_columns
from quant_v2.config import default_universe_symbols
from quant_v2.data.multi_symbol_dataset import fetch_universe_dataset
from quant_v2.data.storage import build_dataset_manifest, validate_multi_symbol_ohlcv
from quant_v2.model_registry import ModelRegistry
from quant_v2.research.model_quality_recovery import (
    QUALITY_RECOVERY_POLICY_VERSION,
    VALIDATION_POLICY_VERSION,
)
from quant_v2.models.trainer import (
    TrainedModel,
    load_model,
    save_model_bundle as save_model,
    train,
)
from quant_v2.validation.temporal_validation import (
    build_temporal_validation_plan,
    compute_recency_weights,
    effective_sample_size,
)

logger = logging.getLogger(__name__)

HORIZONS = (2, 4, 8)
_SENTINEL_FILE = "/tmp/.retrain_last_run"
_LAST_TEMPORAL_VALIDATION_SUMMARY: dict[str, object] | None = None
_BUILDING_DIRNAME = ".building"
_FAILED_DIRNAME = ".failed"
_ARCHIVE_DIRNAME = "archive"


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("Invalid %s=%r; using default %d", name, raw, default)
        return default


def _build_labels(
    df: pd.DataFrame, horizon: int, dead_zone: float = 0.002
) -> pd.Series:
    """Ternary label with dead zone: only label moves > dead_zone as directional.

    Micro-moves smaller than dead_zone (e.g., 0.2%) are treated as ambiguous
    and dropped during training. This prevents the model from wasting capacity
    predicting noise near transaction costs.

    Returns NaN for ambiguous bars, which should be filtered out before fitting.
    """
    if isinstance(df.index, pd.MultiIndex) and "symbol" in df.index.names:
        pieces: list[pd.Series] = []
        grouped = df.groupby(level="symbol", sort=False)
        for _, sym_df in grouped:
            close = pd.to_numeric(sym_df["close"], errors="coerce")
            future_return = close.shift(-horizon) / close - 1.0
            sym_labels = pd.Series(np.nan, index=sym_df.index)
            sym_labels[future_return > dead_zone] = 1
            sym_labels[future_return < -dead_zone] = 0
            pieces.append(sym_labels)
        if not pieces:
            return pd.Series(np.nan, index=df.index)
        return pd.concat(pieces).sort_index()

    labels = pd.Series(np.nan, index=df.index)
    close = pd.to_numeric(df["close"], errors="coerce")
    future_return = close.shift(-horizon) / close - 1.0
    labels[future_return > dead_zone] = 1    # profitable up move
    labels[future_return < -dead_zone] = 0   # profitable down move
    # values in [-dead_zone, dead_zone] remain NaN → filtered out
    return labels


def _validate_model_single(model: TrainedModel, X_test: pd.DataFrame, y_test: pd.Series) -> float:
    """Single-fold validation using predictor.py path (matches inference)."""
    if X_test.empty:
        return 0.0
    from quant_v2.models.predictor import predict_proba
    proba = predict_proba(model, X_test)
    preds = (proba > 0.5).astype(int)
    accuracy = float(np.mean(preds == y_test.values))
    return accuracy


def _inject_reference_returns(frame: pd.DataFrame, btc_returns: pd.Series | None) -> pd.DataFrame:
    """Inject BTC reference returns for cross-pair features on one symbol frame."""

    if btc_returns is None or btc_returns.empty:
        return frame

    out = frame.copy()
    if "_btc_returns" not in out.columns:
        out["_btc_returns"] = btc_returns.reindex(out.index, method="ffill").fillna(0.0)
    return out


def _consume_temporal_validation_summary() -> dict[str, object] | None:
    global _LAST_TEMPORAL_VALIDATION_SUMMARY
    summary = _LAST_TEMPORAL_VALIDATION_SUMMARY
    _LAST_TEMPORAL_VALIDATION_SUMMARY = None
    return summary


def _select_threshold_from_oof_predictions(
    predictions: np.ndarray,
    labels: np.ndarray,
) -> dict[str, object]:
    """Select a probability threshold from out-of-fold development evidence."""

    cfg = get_research_config()
    threshold_min = float(cfg.threshold_min)
    threshold_max = float(cfg.threshold_max)
    threshold_step = float(cfg.threshold_step)
    thresholds = np.round(np.arange(threshold_min, threshold_max + threshold_step / 2, threshold_step), 2)

    probs = np.asarray(predictions, dtype=float)
    truth = np.asarray(labels, dtype=int)
    if probs.size == 0 or truth.size == 0:
        return {
            "source": "insufficient_oof_predictions",
            "selected_threshold": 0.5,
            "selected_accuracy": 0.0,
            "threshold_min": threshold_min,
            "threshold_max": threshold_max,
            "threshold_step": threshold_step,
            "samples": int(len(truth)),
        }

    if len(np.unique(truth)) < 2:
        return {
            "source": "class_collapse",
            "selected_threshold": 0.5,
            "selected_accuracy": float((truth == truth[0]).mean()) if truth.size else 0.0,
            "threshold_min": threshold_min,
            "threshold_max": threshold_max,
            "threshold_step": threshold_step,
            "samples": int(len(truth)),
        }

    best_threshold = 0.5
    best_accuracy = -1.0
    for threshold in thresholds:
        predicted = (probs >= float(threshold)).astype(int)
        accuracy = float((predicted == truth).mean())
        if accuracy > best_accuracy or (accuracy == best_accuracy and float(threshold) < best_threshold):
            best_accuracy = accuracy
            best_threshold = float(threshold)

    return {
        "source": "oof_dev_predictions",
        "selected_threshold": float(best_threshold),
        "selected_accuracy": float(best_accuracy),
        "threshold_min": threshold_min,
        "threshold_max": threshold_max,
        "threshold_step": threshold_step,
        "samples": int(len(truth)),
    }


def _summarize_selection_risk(ledger: dict[str, object]) -> dict[str, float]:
    attempts = [a for a in ledger.get("attempts", []) if isinstance(a, dict)]
    scored = [
        float(attempt.get("holdout_accuracy", attempt.get("mean_accuracy", 0.0)))
        for attempt in attempts
        if float(attempt.get("fold_count", 0)) > 0
    ]
    if not scored:
        return {"pbo_equivalent": 0.0, "selected_holdout_rank_percentile": 1.0}

    selected = ledger.get("selected") or {}
    selected_score = float(selected.get("holdout_accuracy", selected.get("mean_accuracy", 0.0)))
    sorted_scores = sorted(scored)
    rank = sum(score <= selected_score for score in sorted_scores)
    percentile = rank / max(len(sorted_scores), 1)
    return {
        "pbo_equivalent": float(max(0.0, 1.0 - percentile)),
        "selected_holdout_rank_percentile": float(min(max(percentile, 0.0), 1.0)),
    }


def _walk_forward_cv(
    X: pd.DataFrame,
    y: pd.Series,
    horizon: int,
    n_splits: int = 5,
    embargo_bars: int = 100,
    cal_frac: float = 0.20,
    timestamps: pd.Index | pd.Series | pd.DatetimeIndex | None = None,
) -> float:
    """Temporal validation with explicit timestamp folds and a frozen holdout."""

    _ = (n_splits, cal_frac)
    total_len = len(X)
    if total_len < 1000:
        return 0.0
    if timestamps is None:
        return 0.0

    ts_index = pd.DatetimeIndex(timestamps)
    if len(ts_index) != total_len:
        raise ValueError("timestamps length must match X/y length")

    period_count = len(pd.PeriodIndex(ts_index, freq="M").unique())
    holdout_months = 3 if period_count >= 6 else max(1, period_count // 4)

    plan = build_temporal_validation_plan(
        X,
        training_windows_months=(3, 6, 9, 12),
        expanding_included=True,
        test_window_months=1,
        holdout_months=holdout_months,
        purge_bars=max(int(embargo_bars), int(horizon)),
        min_train_rows=500,
    )

    def _fit_and_score(
        train_idx: np.ndarray,
        test_idx: np.ndarray,
        *,
        half_life_days: float,
    ) -> tuple[float, float, np.ndarray, np.ndarray]:
        X_train_fold = X.iloc[train_idx]
        y_train_fold = y.iloc[train_idx]
        X_test_fold = X.iloc[test_idx]
        y_test_fold = y.iloc[test_idx]
        if len(X_train_fold) < 500 or len(X_test_fold) < 100:
            return 0.0, 0.0, np.asarray([], dtype=float), np.asarray([], dtype=int)
        weights = compute_recency_weights(ts_index[train_idx], half_life_days=half_life_days)
        ess = effective_sample_size(weights)
        model = train(X_train_fold, y_train_fold, horizon=horizon, sample_weight=weights)
        acc = _validate_model_single(model, X_test_fold, y_test_fold)
        from quant_v2.models.predictor import predict_proba
        fold_probs = predict_proba(model, X_test_fold)
        return acc, ess, np.asarray(fold_probs, dtype=float), np.asarray(y_test_fold.to_numpy(), dtype=int)

    half_life_candidates = (30.0, 60.0, 90.0)
    attempts: list[dict[str, object]] = []
    best_attempt: dict[str, object] | None = None

    for half_life_days in half_life_candidates:
        fold_scores: list[float] = []
        ess_values: list[float] = []
        oof_predictions: list[float] = []
        oof_truths: list[int] = []
        for fold in plan.folds:
            if fold.holdout:
                continue
            if fold.n_train < 500 or fold.n_test < 100:
                continue
            try:
                acc, ess, fold_probs, fold_truths = _fit_and_score(
                    fold.train_indices,
                    fold.test_indices,
                    half_life_days=half_life_days,
                )
            except Exception as exc:
                logger.warning(
                    "Temporal CV fold %s failed for horizon=%dh half_life=%.0f: %s",
                    fold.fold_id,
                    horizon,
                    half_life_days,
                    exc,
                )
                continue
            if acc <= 0.0:
                continue
            fold_scores.append(acc)
            ess_values.append(ess)
            oof_predictions.extend(float(v) for v in fold_probs)
            oof_truths.extend(int(v) for v in fold_truths)

        mean_accuracy = float(np.mean(fold_scores)) if fold_scores else 0.0
        attempt = {
            "half_life_days": float(half_life_days),
            "fold_count": int(len(fold_scores)),
            "mean_accuracy": mean_accuracy,
            "effective_sample_size_mean": float(np.mean(ess_values)) if ess_values else 0.0,
            "training_windows_months": list(plan.training_windows_months),
            "expanding_included": bool(plan.expanding_included),
            "test_window_months": int(plan.test_window_months),
            "holdout_months": int(plan.holdout_months),
            "trial_count": int(len(plan.folds)),
            "threshold_policy": _select_threshold_from_oof_predictions(
                np.asarray(oof_predictions, dtype=float),
                np.asarray(oof_truths, dtype=int),
            ),
        }
        attempts.append(attempt)
        if best_attempt is None or mean_accuracy > float(best_attempt["mean_accuracy"]):
            best_attempt = attempt

    if best_attempt is None:
        return 0.0

    holdout_rows = int(len(plan.holdout_indices))
    if holdout_rows >= 100:
        dev_mask = np.ones(total_len, dtype=bool)
        dev_mask[plan.holdout_indices] = False
        dev_idx = np.flatnonzero(dev_mask)
        holdout_weights = compute_recency_weights(ts_index[dev_idx], half_life_days=float(best_attempt["half_life_days"]))
        holdout_ess = effective_sample_size(holdout_weights)
        holdout_model = train(X.iloc[dev_idx], y.iloc[dev_idx], horizon=horizon, sample_weight=holdout_weights)
        best_attempt["holdout_accuracy"] = float(
            _validate_model_single(
                holdout_model,
                X.iloc[plan.holdout_indices],
                y.iloc[plan.holdout_indices],
            )
        )
        best_attempt["holdout_effective_sample_size"] = float(holdout_ess)
        best_attempt["holdout_rows"] = holdout_rows

    global _LAST_TEMPORAL_VALIDATION_SUMMARY
    _LAST_TEMPORAL_VALIDATION_SUMMARY = {
        "horizon": int(horizon),
        "attempts": attempts,
        "selected": best_attempt,
        "actual_trial_count": int(len(attempts)),
        "holdout_start": str(plan.holdout_start) if plan.holdout_start is not None else None,
        "holdout_end": str(plan.holdout_end) if plan.holdout_end is not None else None,
        "fold_count": int(len(plan.folds)),
        "purge_bars": int(plan.purge_bars),
        "holdout_months": int(plan.holdout_months),
        "holdout_rows": int(holdout_rows),
    }
    return float(best_attempt["mean_accuracy"])


def _compute_sample_weights(timestamps: pd.DatetimeIndex, half_life_days: float = 60) -> np.ndarray:
    """Backward-compatible wrapper around the temporal validation weight helper."""

    return compute_recency_weights(pd.DatetimeIndex(timestamps), half_life_days=half_life_days)


def _staging_artifact_dir(model_root: Path, version_id: str) -> Path:
    return Path(model_root).expanduser() / _BUILDING_DIRNAME / version_id


def _failed_artifact_record_path(model_root: Path, version_id: str) -> Path:
    return Path(model_root).expanduser() / _FAILED_DIRNAME / f"{version_id}.json"


def _archive_artifact_dir(model_root: Path, version_id: str) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return Path(model_root).expanduser() / _ARCHIVE_DIRNAME / f"failed_retrain_{timestamp}" / version_id


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    tmp_path.replace(path)


def _cleanup_tree(path: Path) -> None:
    try:
        if path.exists():
            shutil.rmtree(path)
    except Exception:
        logger.exception("Failed to clean up staging path: %s", path)


def _write_failure_record(
    model_root: Path,
    version_id: str,
    *,
    reason: str,
    details: dict[str, object] | None = None,
) -> None:
    payload: dict[str, object] = {
        "version_id": version_id,
        "failed_at": datetime.now(timezone.utc).isoformat(),
        "reason": reason,
    }
    if details:
        payload["details"] = details
    _write_json_atomic(_failed_artifact_record_path(model_root, version_id), payload)


def _persist_staged_model_bundle(
    trained: TrainedModel,
    *,
    staging_path: Path,
    metadata: dict[str, object],
) -> None:
    save_model(trained, staging_path, metadata=metadata)


def _validate_artifact_bundle(path: Path) -> None:
    load_model(path)


def retrain_and_promote(
    model_root: Path,
    registry_root: Path,
    train_months: int = 12,
    min_accuracy: float = 0.525,
    extra_symbols: list[str] | None = None,
) -> str | None:
    """Run full retrain → validate → register → promote cycle.

    Returns the new version_id if promoted, or None if validation failed.
    """
    model_root = Path(model_root).expanduser()
    registry_root = Path(registry_root).expanduser()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    version_id = f"model_{timestamp}"
    staging_dir = _staging_artifact_dir(model_root, version_id)
    final_dir = model_root / version_id
    staging_dir.mkdir(parents=True, exist_ok=True)
    if final_dir.exists():
        logger.error("Retrain: target artifact directory already exists: %s", final_dir)
        _write_failure_record(
            model_root,
            version_id,
            reason="target_artifact_already_exists",
            details={"final_dir": str(final_dir)},
        )
        _cleanup_tree(staging_dir)
        return None

    def _abort(reason: str, *, details: dict[str, object] | None = None, archive_source: Path | None = None) -> None:
        payload = {
            "model_root": str(model_root),
            "registry_root": str(registry_root),
            "version_id": version_id,
            "staging_dir": str(staging_dir),
            "final_dir": str(final_dir),
            "train_months": int(train_months),
            "min_accuracy": float(min_accuracy),
            "extra_symbols": list(extra_symbols or []),
        }
        if details:
            payload.update(details)
        _write_failure_record(model_root, version_id, reason=reason, details=payload)
        if archive_source is not None and archive_source.exists():
            archive_dir = _archive_artifact_dir(model_root, version_id)
            archive_dir.parent.mkdir(parents=True, exist_ok=True)
            try:
                if archive_dir.exists():
                    shutil.rmtree(archive_dir)
                archive_source.replace(archive_dir)
                logger.warning("Retrain: archived failed artifact to %s", archive_dir)
            except Exception as exc:
                logger.exception("Retrain: failed to archive artifact %s: %s", archive_source, exc)
        _cleanup_tree(staging_dir)

    client = BinanceClient()
    date_to = datetime.now(timezone.utc)
    date_from = date_to - timedelta(days=train_months * 30)

    primary_symbol = "BTCUSDT"
    symbols_to_fetch = list(dict.fromkeys([primary_symbol] + (extra_symbols or [])))
    logger.info("Retrain: fetching %d months of data for symbols=%s...", train_months, symbols_to_fetch)
    raw_universe = fetch_universe_dataset(
        symbols_to_fetch,
        date_from=date_from,
        date_to=date_to,
        interval="1h",
        include_funding=True,
        include_open_interest=True,
        fail_fast=False,
        client=client,
    )
    if raw_universe.empty:
        logger.error("Retrain: no data fetched for any symbol. Aborting.")
        _abort("no_data_fetched", details={"requested_symbols": symbols_to_fetch})
        return None

    try:
        validate_multi_symbol_ohlcv(raw_universe)
    except Exception as exc:
        logger.error("Retrain: dataset validation failed: %s", exc)
        _abort(
            "dataset_validation_failed",
            details={"requested_symbols": symbols_to_fetch, "error": str(exc)},
        )
        return None

    fetched_symbols = sorted(str(symbol) for symbol in raw_universe.index.get_level_values("symbol").unique())
    failed_symbols = {symbol: "missing from fetched universe" for symbol in symbols_to_fetch if symbol not in fetched_symbols}
    raw_manifest = build_dataset_manifest(
        raw_universe,
        dataset_name=f"scheduled_retrain_{timestamp}",
        metadata={
            "requested_symbols": symbols_to_fetch,
            "fetched_symbols": fetched_symbols,
            "failed_symbols": failed_symbols,
            "requested_time_range_start": date_from.isoformat(),
            "requested_time_range_end": date_to.isoformat(),
            "source_retrieved_at": date_to.isoformat(),
        },
    )

    all_featured_frames: list[pd.DataFrame] = []
    btc_returns: pd.Series | None = None

    if primary_symbol in raw_universe.index.get_level_values("symbol"):
        btc_frame = raw_universe.xs(primary_symbol, level="symbol").copy()
        btc_close = pd.to_numeric(btc_frame["close"], errors="coerce").dropna()
        btc_returns = btc_close.pct_change()

    for symbol, sym_raw in raw_universe.groupby(level="symbol", sort=False):
        per_symbol = sym_raw.droplevel("symbol").copy()
        per_symbol = _inject_reference_returns(per_symbol, btc_returns if str(symbol) != primary_symbol else None)
        sym_featured = build_features(per_symbol)
        if sym_featured.empty:
            failed_symbols[str(symbol)] = "no feature rows"
            logger.warning("Retrain: %s produced no feature rows", symbol)
            continue
        sym_featured = sym_featured.copy()
        sym_featured["symbol"] = str(symbol)
        sym_featured = sym_featured.reset_index().set_index(["timestamp", "symbol"]).sort_index()
        all_featured_frames.append(sym_featured)
        logger.info("Retrain: %s -> %d feature rows", symbol, len(sym_featured))

    if not all_featured_frames:
        logger.error("Retrain: no feature rows produced for any symbol. Aborting.")
        _abort("no_feature_rows", details={"symbols_requested": symbols_to_fetch, "symbols_fetched": fetched_symbols})
        return None

    require_all_symbols = _env_flag("RETRAIN_REQUIRE_ALL_SYMBOLS", True)
    require_all_horizons = _env_flag("RETRAIN_REQUIRE_ALL_HORIZONS", True)
    auto_promote = _env_flag("BOT_RETRAIN_AUTO_PROMOTE", False)
    required_symbols = set(symbols_to_fetch)
    fetched_symbol_set = set(fetched_symbols)
    missing_symbols = sorted(required_symbols - fetched_symbol_set)
    default_min_symbols = len(symbols_to_fetch) if require_all_symbols else 1
    min_symbols = _env_int("RETRAIN_MIN_SYMBOLS", default_min_symbols)
    if require_all_symbols and missing_symbols:
        logger.error(
            "Retrain: missing required symbols %s; fetched=%s failed=%s. Aborting promotion.",
            missing_symbols,
            fetched_symbols,
            failed_symbols,
        )
        _abort(
            "missing_required_symbols",
            details={
                "missing_symbols": missing_symbols,
                "fetched_symbols": fetched_symbols,
                "failed_symbols": failed_symbols,
            },
        )
        return None
    if len(fetched_symbols) < min_symbols:
        logger.error(
            "Retrain: fetched %d symbols, below required minimum %d. fetched=%s failed=%s. Aborting promotion.",
            len(fetched_symbols),
            min_symbols,
            fetched_symbols,
            failed_symbols,
        )
        _abort(
            "insufficient_symbol_coverage",
            details={
                "minimum_symbols": int(min_symbols),
                "fetched_symbols": fetched_symbols,
                "failed_symbols": failed_symbols,
            },
        )
        return None

    featured = pd.concat(all_featured_frames, axis=0).sort_index()
    if isinstance(featured.index, pd.MultiIndex):
        featured.index = featured.index.set_names(["timestamp", "symbol"])
    featured.attrs.update(dict(all_featured_frames[0].attrs))
    logger.info("Retrain: combined dataset = %d rows across %d symbols", len(featured), len(all_featured_frames))
    feature_cols = get_feature_columns(featured)
    logger.info("Retrain: featured dataset = %d rows, %d features", len(featured), len(feature_cols))

    expected_hourly_rows = len(symbols_to_fetch) * train_months * 30 * 24
    default_min_train_rows = max(1000, int(expected_hourly_rows * 0.8))
    min_train_rows = _env_int("RETRAIN_MIN_TRAIN_ROWS", default_min_train_rows)
    if len(featured) < min_train_rows:
        logger.error(
            "Retrain: insufficient data after feature engineering (%d rows < %d required). "
            "fetched=%s failed=%s",
            len(featured),
            min_train_rows,
            fetched_symbols,
            failed_symbols,
        )
        _abort(
            "insufficient_feature_rows",
            details={
                "min_train_rows": int(min_train_rows),
                "featured_rows": int(len(featured)),
                "fetched_symbols": fetched_symbols,
                "failed_symbols": failed_symbols,
            },
        )
        return None

    cfg = get_research_config()

    model_paths: dict[int, Path] = {}
    validation_scores: dict[int, float] = {}
    temporal_validation_ledger: dict[str, object] = {}
    all_passed = True

    for horizon in HORIZONS:
        labels = _build_labels(featured, horizon)
        mask = labels.notna()
        X_all = featured.loc[mask, feature_cols]
        y_all = labels.loc[mask]
        timestamps = pd.DatetimeIndex(X_all.index.get_level_values("timestamp")) if isinstance(X_all.index, pd.MultiIndex) else pd.DatetimeIndex([])

        if len(X_all) < 500:
            logger.warning("Retrain: insufficient data for horizon=%dh (%d rows), skipping", horizon, len(X_all))
            continue

        logger.info("Retrain horizon=%dh: evaluating explicit temporal folds", horizon)

        cv_accuracy = _walk_forward_cv(
            X_all,
            y_all,
            horizon=horizon,
            n_splits=5,
            embargo_bars=100,
            cal_frac=cfg.wf_calibration_frac,
            timestamps=timestamps,
        )
        temporal_summary = _consume_temporal_validation_summary()
        if temporal_summary is not None:
            temporal_validation_ledger[str(horizon)] = temporal_summary

        dev_accuracy = float(cv_accuracy)
        logger.info(
            "Retrain horizon=%dh: dev_accuracy=%.4f (threshold=%.4f)",
            horizon,
            dev_accuracy,
            min_accuracy,
        )

        if dev_accuracy <= 0.0:
            logger.warning("Retrain horizon=%dh: failed to produce usable development evidence", horizon)
            all_passed = False
            continue

        if dev_accuracy < min_accuracy:
            logger.warning("Retrain horizon=%dh: FAILED development validation (%.4f < %.4f)", horizon, dev_accuracy, min_accuracy)
            all_passed = False
            continue

        # Refit on the approved development window using the best dev half-life.
        best_half_life = 60.0
        if temporal_summary is not None:
            selected = temporal_summary.get("selected") or {}
            best_half_life = float(selected.get("half_life_days", 60.0))
        holdout_rows = int((temporal_summary or {}).get("holdout_rows") or 0)
        if holdout_rows <= 0 or holdout_rows >= len(X_all):
            logger.warning(
                "Retrain horizon=%dh: invalid untouched holdout window (%d rows); refusing promotion",
                horizon,
                holdout_rows,
            )
            all_passed = False
            continue
        dev_mask = np.ones(len(X_all), dtype=bool)
        if holdout_rows > 0:
            dev_mask[-holdout_rows:] = False
        X_train = X_all.iloc[np.flatnonzero(dev_mask)]
        y_train = y_all.iloc[np.flatnonzero(dev_mask)]
        X_holdout = X_all.iloc[-holdout_rows:]
        y_holdout = y_all.iloc[-holdout_rows:]
        sample_weights = _compute_sample_weights(pd.DatetimeIndex(X_train.index.get_level_values("timestamp")), half_life_days=best_half_life)
        logger.info(
            "Retrain horizon=%dh: refit on %d dev rows with half_life=%.0f ess=%.1f",
            horizon,
            len(X_train),
            best_half_life,
            effective_sample_size(sample_weights),
        )
        try:
            model = train(X_train, y_train, horizon=horizon, sample_weight=sample_weights)
        except Exception as e:
            logger.error("Retrain horizon=%dh refit failed: %s", horizon, e)
            all_passed = False
            continue

        holdout_accuracy = float(_validate_model_single(model, X_holdout, y_holdout))
        validation_scores[horizon] = holdout_accuracy
        logger.info(
            "Retrain horizon=%dh: final_holdout_accuracy=%.4f (threshold=%.4f)",
            horizon,
            holdout_accuracy,
            min_accuracy,
        )
        if holdout_accuracy < min_accuracy:
            logger.warning(
                "Retrain horizon=%dh: FAILED untouched holdout validation (%.4f < %.4f)",
                horizon,
                holdout_accuracy,
                min_accuracy,
            )
            all_passed = False
            continue

        path = staging_dir / f"model_{horizon}m.pkl"
        artifact_metadata = {
            "dataset_manifest": raw_manifest.to_dict(),
            "validation_summary": temporal_summary,
            "validation_score": float(holdout_accuracy),
            "freeze_summary": {
                "configuration_frozen": True,
                "development_accuracy": float(dev_accuracy),
                "holdout_accuracy": float(holdout_accuracy),
                "selected_half_life_days": float(best_half_life),
                "holdout_rows": int(holdout_rows),
                "holdout_start": str((temporal_summary or {}).get("holdout_start") or ""),
                "holdout_end": str((temporal_summary or {}).get("holdout_end") or ""),
            },
            "fit_window": {
                "rows": int(len(X_train)),
                "start": str(pd.DatetimeIndex(X_train.index.get_level_values("timestamp")).min()),
                "end": str(pd.DatetimeIndex(X_train.index.get_level_values("timestamp")).max()),
            },
            "holdout_window": {
                "rows": int(len(X_holdout)),
                "start": str(pd.DatetimeIndex(X_holdout.index.get_level_values("timestamp")).min()),
                "end": str(pd.DatetimeIndex(X_holdout.index.get_level_values("timestamp")).max()),
            },
            "selected_half_life_days": float(best_half_life),
            "sample_weight_policy": {
                "half_life_days": float(best_half_life),
                "effective_sample_size": float(effective_sample_size(sample_weights)),
            },
            "validation_policy": {
                "validation_policy_version": VALIDATION_POLICY_VERSION,
                "quality_recovery_policy_version": QUALITY_RECOVERY_POLICY_VERSION,
                "training_windows_months": list((temporal_summary or {}).get("training_windows_months", [])),
                "holdout_months": int((temporal_summary or {}).get("holdout_months", 0) or 0),
                "purge_bars": int((temporal_summary or {}).get("purge_bars", 0) or 0),
                "trial_count": int((temporal_summary or {}).get("actual_trial_count", 0) or 0),
            },
            "calibration_policy": {
                "strategy": "fold_local_sigmoid_calibration",
                "sample_fraction": float(get_research_config().wf_calibration_frac),
                "selection_source": str(
                    ((temporal_summary or {}).get("selected") or {}).get("threshold_policy", {}).get("source", "default")
                ),
            },
            "runtime_policy": {
                "min_accuracy": min_accuracy,
                "require_all_symbols": require_all_symbols,
                "require_all_horizons": require_all_horizons,
                "auto_promote": auto_promote,
            },
            "estimator": {
                "name": "LGBMClassifier",
                "calibration_method": getattr(model, "calibration_method", "sigmoid"),
                "calibration_samples": int(getattr(model, "calibration_samples", 0)),
                "fit_samples": int(getattr(model, "fit_samples", 0)),
            },
            "threshold": float(
                ((temporal_summary or {}).get("selected") or {}).get("threshold_policy", {}).get("selected_threshold", 0.5)
                or 0.5
            ),
            "threshold_policy": dict(
                ((temporal_summary or {}).get("selected") or {}).get("threshold_policy")
                or {
                    "source": "default_fixed",
                    "selected_threshold": 0.5,
                    "selected_accuracy": None,
                }
            ),
        }
        _persist_staged_model_bundle(
            model,
            staging_path=path,
            metadata=artifact_metadata,
        )
        _validate_artifact_bundle(path)
        model_paths[horizon] = path
        logger.info("Retrain horizon=%dh: saved to %s", horizon, path)

    if not model_paths:
        logger.error("Retrain: no models passed validation. Aborting promotion.")
        _abort(
            "no_valid_horizon_models",
            details={"missing_horizons": list(HORIZONS)},
        )
        return None

    missing_horizons = sorted(set(HORIZONS) - set(model_paths))
    if require_all_horizons and (not all_passed or missing_horizons):
        logger.error(
            "Retrain: refusing to promote partial horizon set. trained=%s missing=%s all_passed=%s",
            sorted(model_paths),
            missing_horizons,
            all_passed,
        )
        _abort(
            "partial_horizon_set_rejected",
            details={
                "trained_horizons": sorted(model_paths),
                "missing_horizons": missing_horizons,
                "all_passed": bool(all_passed),
            },
        )
        return None

    if not all_passed:
        logger.warning("Retrain: some horizons failed but %d passed. Registering partial candidate.", len(model_paths))

    try:
        staging_dir.replace(final_dir)
    except Exception as exc:
        logger.error("Retrain: failed to publish staged artifact %s -> %s: %s", staging_dir, final_dir, exc)
        _abort(
            "staged_publish_failed",
            details={"staging_dir": str(staging_dir), "final_dir": str(final_dir), "error": str(exc)},
        )
        return None

    try:
        for horizon in sorted(model_paths):
            _validate_artifact_bundle(final_dir / f"model_{horizon}m.pkl")
    except Exception as exc:
        logger.error("Retrain: published artifact failed runtime validation: %s", exc)
        _abort(
            "published_artifact_validation_failed",
            archive_source=final_dir,
            details={"error": str(exc), "trained_horizons": sorted(model_paths)},
        )
        return None

    # Register as paper-quarantine by default. Activation is manual after
    # forward paper/shadow evaluation, unless the emergency override is set.
    registry = ModelRegistry(registry_root)
    registered_status = "candidate" if auto_promote else "paper_quarantine"

    metrics = {
        "validation_scores": {str(h): round(s, 4) for h, s in validation_scores.items()},
        "train_months": train_months,
        "train_rows": len(featured),
        "horizons_trained": list(model_paths.keys()),
        "symbols_requested": symbols_to_fetch,
        "symbols_fetched": fetched_symbols,
        "symbols_failed": failed_symbols,
        "raw_dataset_manifest": raw_manifest.to_dict(),
        "temporal_validation_ledger": temporal_validation_ledger,
        "selection_risk": {
            horizon: _summarize_selection_risk(ledger)
            for horizon, ledger in temporal_validation_ledger.items()
            if isinstance(ledger, dict)
        },
        "trial_count": int(sum(int((entry.get("selected") or {}).get("trial_count", 0)) for entry in temporal_validation_ledger.values())) or len(HORIZONS),
        "min_train_rows": min_train_rows,
        "trained_at": timestamp,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "promotion_eligible": bool(model_paths) and (all_passed or not require_all_horizons),
        "promotion_eligibility_notes": (
            "passed validation checks"
            if bool(model_paths) and (all_passed or not require_all_horizons)
            else "failed validation checks"
        ),
        "min_accuracy": min_accuracy,
        "require_all_symbols": require_all_symbols,
        "require_all_horizons": require_all_horizons,
        "paper_quarantine_required": not auto_promote,
    }

    try:
        registry.register_version(
            version_id=version_id,
            artifact_dir=final_dir,
            metrics=metrics,
            tags={
                "source": "scheduled_retrain",
                "horizons": str(list(model_paths.keys())),
                "status": registered_status,
                "promotion_eligible": "true" if metrics["promotion_eligible"] else "false",
            },
            description=f"Scheduled retrain {timestamp}: {len(model_paths)} horizons, accuracies={validation_scores}",
            status=registered_status,
        )
        logger.info(
            "Retrain: registered %s %s (%d horizons). Auto-promote=%s",
            registered_status,
            version_id,
            len(model_paths),
            auto_promote,
        )
        if auto_promote:
            registry.promote_version(
                version_id,
                promoted_by="scheduled_retrain:auto_promote",
                notes="BOT_RETRAIN_AUTO_PROMOTE enabled",
            )
            logger.warning("Retrain: auto-promoted %s as active model", version_id)
    except Exception as e:
        logger.error("Retrain: registry candidate registration failed: %s", e)
        _abort(
            "registry_registration_failed",
            archive_source=final_dir,
            details={"error": str(e)},
        )
        return None

    return version_id


def run_scheduler_loop() -> None:
    """Blocking loop: retrain immediately, then every RETRAIN_INTERVAL_HOURS."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    model_root = Path(os.getenv("BOT_MODEL_ROOT", "/app/models/production")).expanduser()
    registry_root = Path(os.getenv("BOT_MODEL_REGISTRY_ROOT", str(model_root / "registry"))).expanduser()
    interval_hours = int(os.getenv("RETRAIN_INTERVAL_HOURS", "168"))
    train_months = int(os.getenv("RETRAIN_TRAIN_MONTHS", "12"))
    min_accuracy = float(os.getenv("RETRAIN_MIN_ACCURACY", "0.525"))
    # Default extra symbols track the live signal universe minus the anchor BTCUSDT.
    # Override via RETRAIN_TRAIN_SYMBOLS=<comma-separated>.
    _universe = [s for s in default_universe_symbols() if s != "BTCUSDT"]
    _default_extra_syms = ",".join(_universe) if _universe else "ETHUSDT,BNBUSDT"
    _extra_sym_raw = os.getenv("RETRAIN_TRAIN_SYMBOLS", _default_extra_syms).strip()
    extra_symbols = [s.strip() for s in _extra_sym_raw.split(",") if s.strip()] if _extra_sym_raw else []

    logger.info(
        "Retrain scheduler started: interval=%dh, train_months=%d, min_accuracy=%.2f, model_root=%s",
        interval_hours, train_months, min_accuracy, model_root,
    )

    # Wait for signal manager to finish its first cycle before competing for API
    startup_delay = int(os.getenv("RETRAIN_STARTUP_DELAY_SECONDS", "90"))
    logger.info("Retrain: waiting %ds before first run (let signal manager settle)...", startup_delay)
    time.sleep(startup_delay)

    while True:
        result = None
        for attempt in range(3):
            try:
                result = retrain_and_promote(
                    model_root=model_root,
                    registry_root=registry_root,
                    train_months=train_months,
                    min_accuracy=min_accuracy,
                    extra_symbols=extra_symbols,
                )
                if result:
                    logger.info("Retrain cycle complete: registered candidate %s", result)
                    break
                else:
                    logger.warning("Retrain attempt %d/3: no candidate registered", attempt + 1)
                    if attempt < 2:
                        logger.info("Retrying in 300s...")
                        time.sleep(300)
            except Exception as e:
                logger.exception("Retrain attempt %d/3 failed: %s", attempt + 1, e)
                if attempt < 2:
                    logger.info("Retrying in 300s...")
                    time.sleep(300)

        if result is None:
            logger.error("All 3 retrain attempts failed this cycle")

        # Write sentinel so we can track last run
        try:
            Path(_SENTINEL_FILE).write_text(
                datetime.now(timezone.utc).isoformat(), encoding="utf-8",
            )
        except Exception:
            pass

        logger.info("Next retrain in %d hours. Sleeping...", interval_hours)
        time.sleep(interval_hours * 3600)


if __name__ == "__main__":
    run_scheduler_loop()
