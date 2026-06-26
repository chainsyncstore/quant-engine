"""Automated walk-forward retrain pipeline for multi-horizon ensemble."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

from quant.data.binance_client import BinanceClient
from quant.features.pipeline import build_features, get_feature_columns
from quant_v2.data.multi_symbol_dataset import fetch_symbol_dataset
from quant_v2.research.model_quality_recovery import QUALITY_RECOVERY_POLICY_VERSION, VALIDATION_POLICY_VERSION
from quant_v2.models.trainer import save_model_bundle, train

logger = logging.getLogger(__name__)

HORIZONS = (2, 4, 8)
TRAIN_MONTHS = 6
LABEL_COL_TEMPLATE = "label_{horizon}h"


def _build_labels(df, horizon: int):
    """Binary label: did price go up over the next `horizon` bars?"""
    import pandas as pd
    close = pd.to_numeric(df["close"], errors="coerce")
    future_return = close.shift(-horizon) / close - 1.0
    return (future_return > 0).astype(int)


def run_retrain(
    output_dir: Path,
    symbol: str = "BTCUSDT",
    train_months: int = TRAIN_MONTHS,
) -> dict[int, Path]:
    """Retrain models for all horizons. Returns {horizon: model_path}."""

    client = BinanceClient()
    date_to = datetime.now(timezone.utc)
    date_from = date_to - timedelta(days=train_months * 30)

    logger.info("Fetching training data for %s: %s to %s", symbol, date_from, date_to)
    raw = fetch_symbol_dataset(
        symbol, date_from=date_from, date_to=date_to, client=client,
    )
    featured = build_features(raw)
    feature_cols = get_feature_columns(featured)

    model_paths: dict[int, Path] = {}
    for horizon in HORIZONS:
        labels = _build_labels(featured, horizon)
        # Drop rows without labels (last `horizon` rows)
        mask = labels.notna()
        X = featured.loc[mask, feature_cols]
        y = labels.loc[mask]

        if len(X) < 500:
            logger.warning("Insufficient data for horizon=%dh (%d rows), skipping", horizon, len(X))
            continue

        model = train(X, y, horizon=horizon)
        path = output_dir / f"model_{horizon}m.pkl"
        save_model_bundle(
            model,
            path,
            metadata={
                "pipeline": "retrain_pipeline",
                "symbol": symbol,
                "train_months": train_months,
                "horizon": horizon,
                "validation_policy_version": VALIDATION_POLICY_VERSION,
                "quality_recovery_policy_version": QUALITY_RECOVERY_POLICY_VERSION,
            },
        )
        model_paths[horizon] = path
        logger.info("Trained horizon=%dh: %d samples, saved to %s", horizon, len(X), path)

    return model_paths
