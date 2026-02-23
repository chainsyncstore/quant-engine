"""
Train production models and save artifacts for live signal generation.

Trains LightGBM + GMM regime model on the full available dataset,
runs walk-forward to discover regime thresholds, and saves everything
needed for live prediction.

Usage:
    python -m quant.live.train_models [--months 6]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from quant.config import get_research_config, get_path_config
from quant.data.binance_client import BinanceClient
from quant.data.session_filter import filter_sessions
from quant.data.storage import snapshot, validate_ohlcv, load_latest_snapshot
from quant.features.pipeline import build_features, get_feature_columns
from quant.labels.labeler import add_labels
from quant.models import trainer as model_trainer
from quant.models.predictor import predict_proba
from quant.regime import gmm_regime
from quant.validation.walk_forward import run_walk_forward

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Output directory for live model artifacts
MODELS_DIR = Path("models/production")


def train_production_models(
    df: pd.DataFrame,
    params_override: Dict | None = None,
) -> Path:
    """
    Train production models for ALL horizons and save artifacts.

    Returns:
        Path to the saved model directory.
    """
    cfg = get_research_config()
    start_time = time.time()

    # --- Feature engineering ---
    logger.info("=" * 60)
    logger.info("STEP 1: Feature Engineering")
    logger.info("=" * 60)
    # Ensure fresh config is loaded (with new max_features)
    df_features = build_features(filter_sessions(df))
    feature_cols = get_feature_columns(df_features)
    logger.info("Features (%d): %s", len(feature_cols), feature_cols)

    # --- Labels ---
    logger.info("=" * 60)
    logger.info("STEP 2: Labeling")
    logger.info("=" * 60)
    # Ensure 10m label is created
    df_labeled = add_labels(df_features, horizons=cfg.horizons)

    # --- Walk-Forward Validation ---
    logger.info("=" * 60)
    logger.info("STEP 3: Walk-Forward Validation (Threshold Calibration)")
    logger.info("=" * 60)
    wf_result = run_walk_forward(df_labeled, params_override=params_override)

    # Log WF results summary
    for h, report in wf_result.reports.items():
        logger.info(
            "  WF %dm: EV=%.6f, WR=%.1f%%, Sharpe=%.2f, Trades=%d",
            h, report.overall_ev, report.overall_win_rate * 100,
            report.overall_sharpe, report.overall_n_trades,
        )
        for rm in report.per_regime:
            status = "TRADE" if rm.ev > 0 else "SKIP"
            logger.info(
                "    Regime %d: %s | thresh=%.2f | EV=%.6f | WR=%.1f%% | n=%d",
                rm.regime, status, rm.optimal_threshold, rm.ev,
                rm.win_rate * 100, rm.n_trades,
            )

    # Prepare artifact directory
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = MODELS_DIR / f"model_{ts}"
    model_dir.mkdir(parents=True, exist_ok=True)

    # Collect config for all horizons
    full_config = {
        "mode": cfg.mode,
        "horizons": cfg.horizons,
        "feature_cols": feature_cols,
        "spread": cfg.spread_price,
        "taker_fee_rate": cfg.taker_fee_rate,
        "trained_at": ts,
        "training_bars": len(df_labeled),
        "params_override": params_override,
        "regime_thresholds": {},
        "regime_config": {},
    }

    # --- Train Final Models for Each Horizon ---
    logger.info("=" * 60)
    logger.info("STEP 4: Train Final Production Models")
    logger.info("=" * 60)

    for h in cfg.horizons:
        logger.info("Training horizon: %dm", h)

        # 4a. Use walk-forward derived thresholds and regime config
        wf_thresholds = wf_result.thresholds.get(h, {})
        wf_regime_metrics = {
            rm.regime: rm for rm in wf_result.reports[h].per_regime
        } if h in wf_result.reports else {}

        full_config["regime_thresholds"][str(h)] = {
            str(r): wf_thresholds.get(r, 0.65) for r in range(cfg.n_regimes)
        }
        full_config["regime_config"][str(h)] = {
            str(r): {
                "ev": wf_regime_metrics[r].ev if r in wf_regime_metrics else 0.0,
                "win_rate": wf_regime_metrics[r].win_rate if r in wf_regime_metrics else 0.5,
                "n_trades": wf_regime_metrics[r].n_trades if r in wf_regime_metrics else 0,
                "threshold": wf_thresholds.get(r, 0.65),
                "tradeable": wf_regime_metrics[r].ev > 0 if r in wf_regime_metrics else False,
            } for r in range(cfg.n_regimes)
        }

        # 4b. Train Model on ALL data
        label_col = f"label_{h}m"
        X_all = df_labeled[feature_cols]
        y_all = df_labeled[label_col]

        # Filter FLAT
        mask = y_all != -1
        trained = model_trainer.train(
            X_all[mask], y_all[mask], horizon=h, params_override=params_override
        )

        # 4c. Save Model
        model_path = model_dir / f"model_{h}m.joblib"
        model_trainer.save_model(trained, model_path)

    # --- Train final regime model on all data ---
    regime_model = gmm_regime.fit(df_labeled)
    regime_path = model_dir / "regime_model.joblib"
    gmm_regime.save_model(regime_model, regime_path)

    # --- Save full config ---
    config_path = model_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(full_config, f, indent=2, default=str)

    duration = time.time() - start_time
    logger.info("All production models saved to: %s (%.1fs)", model_dir, duration)

    return model_dir


def fetch_binance_data(months: int = 36) -> pd.DataFrame:
    """Fetch BTC 1H data with funding rates and open interest from Binance."""
    client = BinanceClient()
    date_to = datetime.now(timezone.utc)
    date_from = date_to - timedelta(days=months * 30)

    logger.info("Fetching BTCUSDT 1H klines: %s -> %s", date_from, date_to)
    ohlcv = client.fetch_historical(date_from, date_to)
    logger.info("OHLCV: %d bars", len(ohlcv))

    logger.info("Fetching funding rates...")
    funding = client.fetch_funding_rates(date_from, date_to)
    logger.info("Funding rates: %d entries", len(funding))

    logger.info("Fetching open interest...")
    oi = client.fetch_open_interest(date_from, date_to)
    logger.info("Open interest: %d entries", len(oi))

    df = BinanceClient.merge_supplementary(ohlcv, funding, oi)
    logger.info("Merged dataset: %d bars, columns: %s", len(df), list(df.columns))

    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Train production models")
    parser.add_argument("--months", type=int, default=36, help="Months of history")
    parser.add_argument("--fetch", action="store_true", help="Fetch fresh data")
    parser.add_argument("--horizon", type=int, default=3, help="Prediction horizon")
    args = parser.parse_args()

    get_path_config()

    if args.fetch:
        df = fetch_binance_data(months=args.months)

        if df.empty:
            logger.error("No data received")
            sys.exit(1)

        # Save snapshot for future use
        snapshot(df)
    else:
        df = load_latest_snapshot()
        if df is None:
            logger.error("No data found. Use --fetch")
            sys.exit(1)

    train_production_models(df)


if __name__ == "__main__":
    main()
