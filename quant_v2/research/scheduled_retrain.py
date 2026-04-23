"""Scheduled weekly retrain: trains multi-horizon v2 models, validates, registers, and promotes.

Designed to run inside the Docker container on a schedule (cron or asyncio timer).
The signal manager hot-swaps to the new model on its next cycle.

Usage (standalone):
    python -m quant_v2.research.scheduled_retrain

Environment:
    BOT_MODEL_ROOT        – model artifact root (default: /app/models/production)
    BOT_MODEL_REGISTRY_ROOT – registry root (default: {MODEL_ROOT}/registry)
    RETRAIN_INTERVAL_HOURS – hours between retrain runs (default: 168 = 7 days)
    RETRAIN_TRAIN_MONTHS   – months of training data (default: 12)
    RETRAIN_MIN_ACCURACY   – minimum accuracy to promote a model (default: 0.525)
    RETRAIN_TRAIN_SYMBOLS  – comma-separated extra symbols to include in training (default: full universe from default_universe_symbols() minus BTCUSDT)
"""

from __future__ import annotations

import logging
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from quant.config import get_research_config
from quant.data.binance_client import BinanceClient
from quant.features.pipeline import build_features, get_feature_columns
from quant_v2.config import default_universe_symbols
from quant_v2.data.multi_symbol_dataset import fetch_symbol_dataset
from quant_v2.model_registry import ModelRegistry
from quant_v2.models.trainer import TrainedModel, train, save_model, load_model

logger = logging.getLogger(__name__)

HORIZONS = (2, 4, 8)
_SENTINEL_FILE = "/tmp/.retrain_last_run"


def _build_labels(
    df: pd.DataFrame, horizon: int, dead_zone: float = 0.002
) -> pd.Series:
    """Ternary label with dead zone: only label moves > dead_zone as directional.

    Micro-moves smaller than dead_zone (e.g., 0.2%) are treated as ambiguous
    and dropped during training. This prevents the model from wasting capacity
    predicting noise near transaction costs.

    Returns NaN for ambiguous bars, which should be filtered out before fitting.
    """
    close = pd.to_numeric(df["close"], errors="coerce")
    future_return = close.shift(-horizon) / close - 1.0

    labels = pd.Series(np.nan, index=df.index)
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


def _walk_forward_cv(
    X: pd.DataFrame,
    y: pd.Series,
    horizon: int,
    n_splits: int = 5,
    embargo_bars: int = 100,
    cal_frac: float = 0.20,
) -> float:
    """Purged walk-forward cross-validation with embargo gaps.

    Returns mean accuracy across all folds. Prevents leakage between train/test.
    """
    from sklearn.model_selection import TimeSeriesSplit

    total_len = len(X)
    if total_len < 1000:
        return 0.0

    # Use TimeSeriesSplit for expanding windows
    tscv = TimeSeriesSplit(n_splits=n_splits)
    accuracies: list[float] = []

    for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(X)):
        # Apply embargo: remove last `embargo_bars` from train
        # to prevent leakage from train into test
        if len(train_idx) > embargo_bars:
            train_idx = train_idx[:-embargo_bars]

        X_train_fold = X.iloc[train_idx]
        y_train_fold = y.iloc[train_idx]
        X_test_fold = X.iloc[test_idx]
        y_test_fold = y.iloc[test_idx]

        if len(X_train_fold) < 500 or len(X_test_fold) < 100:
            continue

        try:
            model = train(X_train_fold, y_train_fold, horizon=horizon)
            acc = _validate_model_single(model, X_test_fold, y_test_fold)
            accuracies.append(acc)
        except Exception as e:
            logger.warning("CV fold %d failed for horizon=%dh: %s", fold_idx, horizon, e)
            continue

    if not accuracies:
        return 0.0
    return float(np.mean(accuracies))


def _compute_sample_weights(timestamps: pd.DatetimeIndex, half_life_days: float = 60) -> np.ndarray:
    """Exponential decay weights: recent data gets more weight.

    half_life_days: number of days for weight to decay to 50%
    """
    if len(timestamps) == 0:
        return np.ones(len(timestamps))

    now = timestamps[-1]  # most recent
    days_ago = (now - timestamps).total_seconds() / 86400.0
    decay_rate = np.log(2) / half_life_days
    weights = np.exp(-decay_rate * days_ago)
    return weights / weights.mean()  # normalize to mean=1


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
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    version_id = f"model_{timestamp}"
    artifact_dir = model_root / version_id
    artifact_dir.mkdir(parents=True, exist_ok=True)

    client = BinanceClient()
    date_to = datetime.now(timezone.utc)
    date_from = date_to - timedelta(days=train_months * 30)

    # Fetch primary symbol (BTCUSDT) plus optional extra symbols for richer training
    primary_symbol = "BTCUSDT"
    symbols_to_fetch = [primary_symbol] + (extra_symbols or [])
    logger.info("Retrain: fetching %d months of data for symbols=%s...", train_months, symbols_to_fetch)

    all_featured_frames: list[pd.DataFrame] = []
    btc_returns: pd.Series | None = None  # cached for cross-pair feature injection

    for symbol in symbols_to_fetch:
        try:
            raw = fetch_symbol_dataset(
                symbol,
                date_from=date_from,
                date_to=date_to,
                client=client,
                include_funding=True,
                include_open_interest=True,
            )

            # Compute & cache BTC returns from the primary symbol
            if symbol == primary_symbol and btc_returns is None:
                btc_close = pd.to_numeric(raw["close"], errors="coerce").dropna()
                btc_returns = btc_close.pct_change()

            # Inject BTC returns so cross-pair features match inference
            if btc_returns is not None and "_btc_returns" not in raw.columns:
                raw["_btc_returns"] = btc_returns.reindex(raw.index, method="ffill").fillna(0.0)

            sym_featured = build_features(raw)
            if not sym_featured.empty:
                all_featured_frames.append(sym_featured)
                logger.info("Retrain: %s -> %d feature rows", symbol, len(sym_featured))
        except Exception as e:
            logger.warning("Retrain: data fetch failed for %s: %s", symbol, e)

    if not all_featured_frames:
        logger.error("Retrain: no data fetched for any symbol. Aborting.")
        _cleanup_artifact_dir(artifact_dir)
        return None

    featured = pd.concat(all_featured_frames, ignore_index=True)
    featured = featured.sort_values("timestamp") if "timestamp" in featured.columns else featured
    logger.info("Retrain: combined dataset = %d rows across %d symbols", len(featured), len(all_featured_frames))
    feature_cols = get_feature_columns(featured)
    logger.info("Retrain: featured dataset = %d rows, %d features", len(featured), len(feature_cols))

    # Defensive: fill any remaining NaN (shouldn't happen after pipeline fix, but safety first)
    nan_count = featured[feature_cols].isna().sum().sum()
    if nan_count > 0:
        logger.warning("Retrain: filling %d NaN values in feature columns", nan_count)
        featured[feature_cols] = featured[feature_cols].fillna(0.0)

    if len(featured) < 1000:
        logger.error("Retrain: insufficient data after feature engineering (%d rows)", len(featured))
        return None

    cfg = get_research_config()

    model_paths: dict[int, Path] = {}
    validation_scores: dict[int, float] = {}
    all_passed = True

    for horizon in HORIZONS:
        labels = _build_labels(featured, horizon)
        mask = labels.notna()
        X_all = featured.loc[mask, feature_cols]
        y_all = labels.loc[mask]

        if len(X_all) < 500:
            logger.warning("Retrain: insufficient data for horizon=%dh (%d rows), skipping", horizon, len(X_all))
            continue

        # Split on the FILTERED data, not the raw featured frame
        split_idx = int(len(X_all) * 0.8)
        X_train = X_all.iloc[:split_idx]
        y_train = y_all.iloc[:split_idx]
        X_test = X_all.iloc[split_idx:]
        y_test = y_all.iloc[split_idx:]

        logger.info("Retrain horizon=%dh: train=%d test=%d", horizon, len(X_train), len(X_test))

        # Compute sample weights for recency bias (Phase 4)
        if "timestamp" in featured.columns:
            timestamps = pd.to_datetime(featured.loc[mask, "timestamp"])
            sample_weights = _compute_sample_weights(timestamps, half_life_days=60)
        else:
            sample_weights = None

        try:
            sw_train = sample_weights[:split_idx] if sample_weights is not None else None
            model = train(X_train, y_train, horizon=horizon, sample_weight=sw_train)
        except Exception as e:
            logger.error("Retrain horizon=%dh training failed: %s", horizon, e)
            all_passed = False
            continue

        # Phase 2: Walk-forward CV validation (more robust than single split)
        cv_accuracy = _walk_forward_cv(
            X_all, y_all, horizon=horizon,
            n_splits=5, embargo_bars=100, cal_frac=cfg.wf_calibration_frac
        )
        # Also get single-split accuracy for comparison
        single_accuracy = _validate_model_single(model, X_test, y_test)

        # Use CV accuracy for promotion gate, log both
        accuracy = cv_accuracy if cv_accuracy > 0 else single_accuracy
        validation_scores[horizon] = accuracy
        logger.info("Retrain horizon=%dh: CV accuracy=%.4f, single-split=%.4f (threshold=%.4f)",
                    horizon, cv_accuracy, single_accuracy, min_accuracy)

        if accuracy < min_accuracy:
            logger.warning("Retrain horizon=%dh: FAILED validation (%.4f < %.4f)", horizon, accuracy, min_accuracy)
            all_passed = False
            continue

        path = artifact_dir / f"model_{horizon}m.pkl"
        save_model(model, path)
        model_paths[horizon] = path
        logger.info("Retrain horizon=%dh: saved to %s", horizon, path)

    if not model_paths:
        logger.error("Retrain: no models passed validation. Aborting promotion.")
        _cleanup_artifact_dir(artifact_dir)
        return None

    if not all_passed:
        logger.warning("Retrain: some horizons failed but %d passed. Promoting partial ensemble.", len(model_paths))

    # Register and promote
    registry = ModelRegistry(registry_root)

    metrics = {
        "validation_scores": {str(h): round(s, 4) for h, s in validation_scores.items()},
        "train_months": train_months,
        "train_rows": len(featured),
        "horizons_trained": list(model_paths.keys()),
        "trained_at": timestamp,
    }

    try:
        registry.register_version(
            version_id=version_id,
            artifact_dir=artifact_dir,
            metrics=metrics,
            tags={"source": "scheduled_retrain", "horizons": str(list(model_paths.keys()))},
            description=f"Scheduled retrain {timestamp}: {len(model_paths)} horizons, accuracies={validation_scores}",
        )
        registry.set_active_version(version_id)
        logger.info("Retrain: promoted %s as active model (%d horizons)", version_id, len(model_paths))
    except Exception as e:
        logger.error("Retrain: registry promotion failed: %s", e)
        return None

    return version_id


def _cleanup_artifact_dir(artifact_dir: Path) -> None:
    """Remove empty artifact directory on failed retrain."""
    try:
        for f in artifact_dir.iterdir():
            f.unlink()
        artifact_dir.rmdir()
    except Exception:
        pass


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
                    logger.info("Retrain cycle complete: promoted %s", result)
                    break
                else:
                    logger.warning("Retrain attempt %d/3: no model promoted", attempt + 1)
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
