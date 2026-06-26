"""One-off active-model native Torch confirmation shadow export.

This command trains a confirmation artifact for the currently active model
version and records offline agreement/disagreement profitability evidence. It
does not promote a new baseline model and runtime loading remains shadow-only.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from quant.data.binance_client import BinanceClient
from quant.features.pipeline import build_features, get_feature_columns
from quant_v2.config import default_universe_symbols
from quant_v2.data.multi_symbol_dataset import fetch_symbol_dataset
from quant_v2.model_registry import DEFAULT_REQUIRED_HORIZONS, ModelRegistry, write_model_manifest
from quant_v2.models.confirmation_trainer import train_and_export_confirmation_model
from quant_v2.models.trainer import load_model
from quant_v2.research.scheduled_retrain import (
    _build_forward_returns,
    _build_labels,
    _env_float,
    _env_int,
    _predict_model_proba,
)

logger = logging.getLogger(__name__)


def export_active_confirmation_shadow(
    *,
    model_root: Path,
    registry_root: Path,
    train_months: int = 6,
    extra_symbols: list[str] | None = None,
) -> dict[str, Any]:
    """Train confirmation for the active version and update its evidence."""

    registry = ModelRegistry(registry_root, model_root=model_root)
    active = registry.get_active_version()
    if active is None:
        raise RuntimeError("No active model version is registered")

    artifact_dir = registry.validate_artifact_dir(active.artifact_dir)
    horizon = _env_int("RETRAIN_CONFIRMATION_HORIZON", 4)
    model_path = _find_horizon_model(artifact_dir, horizon)
    if model_path is None:
        raise RuntimeError(f"Active artifact has no {horizon}m model: {artifact_dir}")
    primary_model = load_model(registry.validate_model_file(model_path))

    featured, fetched_symbols, failed_symbols, symbols_requested = _fetch_featured_dataset(
        train_months=train_months,
        extra_symbols=extra_symbols,
    )
    feature_cols = get_feature_columns(featured)
    if not feature_cols:
        raise RuntimeError("No feature columns available for confirmation training")
    nan_count = featured[feature_cols].isna().sum().sum()
    if nan_count > 0:
        logger.warning("Confirmation export: filling %d NaN feature values", nan_count)
        featured[feature_cols] = featured[feature_cols].fillna(0.0)

    labels = _build_labels(featured, horizon)
    forward_returns = _build_forward_returns(featured, horizon)
    mask = labels.notna()
    X_all = featured.loc[mask, feature_cols]
    y_all = labels.loc[mask]
    returns_all = forward_returns.loc[mask]
    min_rows = _env_int("RETRAIN_CONFIRMATION_MIN_PROFIT_SAMPLES", 200)
    if len(X_all) < max(200, min_rows):
        raise RuntimeError(
            f"Insufficient rows for confirmation shadow export: {len(X_all)}"
        )

    split_idx = int(len(X_all) * 0.8)
    X_train = X_all.iloc[:split_idx]
    y_train = y_all.iloc[:split_idx]
    X_validation = X_all.iloc[split_idx:]
    y_validation = y_all.iloc[split_idx:]
    returns_validation = returns_all.iloc[split_idx:]
    primary_probs = _predict_model_proba(primary_model, X_validation)

    result = train_and_export_confirmation_model(
        X_train,
        y_train,
        X_validation,
        y_validation,
        artifact_dir,
        version_id=active.version_id,
        horizon=horizon,
        primary_validation_probabilities=[float(value) for value in primary_probs],
        validation_forward_returns=[float(value) for value in returns_validation],
        round_trip_cost_bps=_env_float("RETRAIN_CONFIRMATION_ROUND_TRIP_COST_BPS", 20.0),
        min_profitability_samples=min_rows,
        min_agreement_edge_bps=_env_float(
            "RETRAIN_CONFIRMATION_MIN_AGREEMENT_EDGE_BPS",
            2.0,
        ),
        min_agreement_win_rate=_env_float(
            "RETRAIN_CONFIRMATION_MIN_AGREEMENT_WIN_RATE",
            0.52,
        ),
        min_agreement_coverage=_env_float(
            "RETRAIN_CONFIRMATION_MIN_AGREEMENT_COVERAGE",
            0.10,
        ),
    ).as_dict()
    result["runtime_mode"] = "shadow_only"
    result["symbols_requested"] = symbols_requested
    result["symbols_fetched"] = fetched_symbols
    result["symbols_failed"] = failed_symbols
    result["exported_at"] = datetime.now(timezone.utc).isoformat()
    result["active_version_id"] = active.version_id

    metrics = dict(active.metrics)
    metrics["confirmation"] = result
    metrics["confirmation_shadow_updated_at"] = result["exported_at"]
    required_horizons = tuple(
        int(horizon_value)
        for horizon_value in metrics.get("required_horizons", DEFAULT_REQUIRED_HORIZONS)
    )
    manifest = write_model_manifest(
        artifact_dir,
        version_id=active.version_id,
        required_horizons=required_horizons,
        metrics=metrics,
        source="active_confirmation_shadow_export",
    )
    registry.validate_artifact_manifest(
        artifact_dir,
        required_horizons=required_horizons,
        smoke_load=True,
    )
    registry.update_version_metrics(active.version_id, metrics=metrics)

    return {
        "version_id": active.version_id,
        "artifact_dir": str(artifact_dir),
        "confirmation": result,
        "manifest_files": [entry["path"] for entry in manifest["files"]],
    }


def _fetch_featured_dataset(
    *,
    train_months: int,
    extra_symbols: list[str] | None,
) -> tuple[pd.DataFrame, list[str], dict[str, str], list[str]]:
    client = BinanceClient()
    date_to = datetime.now(timezone.utc)
    date_from = date_to - timedelta(days=int(train_months) * 30)
    primary_symbol = "BTCUSDT"
    symbols_requested = list(dict.fromkeys([primary_symbol] + (extra_symbols or [])))
    all_featured_frames: list[pd.DataFrame] = []
    fetched_symbols: list[str] = []
    failed_symbols: dict[str, str] = {}
    btc_returns: pd.Series | None = None

    for symbol in symbols_requested:
        try:
            raw = fetch_symbol_dataset(
                symbol,
                date_from=date_from,
                date_to=date_to,
                client=client,
                include_funding=True,
                include_open_interest=True,
            )
            if symbol == primary_symbol and btc_returns is None:
                btc_close = pd.to_numeric(raw["close"], errors="coerce").dropna()
                btc_returns = btc_close.pct_change()
            if btc_returns is not None and "_btc_returns" not in raw.columns:
                raw["_btc_returns"] = btc_returns.reindex(raw.index, method="ffill").fillna(0.0)
            featured = build_features(raw)
            if featured.empty:
                failed_symbols[symbol] = "no feature rows"
                continue
            all_featured_frames.append(featured)
            fetched_symbols.append(symbol)
        except Exception as exc:
            failed_symbols[symbol] = str(exc)
            logger.warning("Confirmation export: data fetch failed for %s: %s", symbol, exc)

    if not all_featured_frames:
        raise RuntimeError(f"No data fetched for confirmation export: {failed_symbols}")
    combined = pd.concat(all_featured_frames, ignore_index=True)
    if "timestamp" in combined.columns:
        combined = combined.sort_values("timestamp")
    return combined, fetched_symbols, failed_symbols, symbols_requested


def _find_horizon_model(artifact_dir: Path, horizon: int) -> Path | None:
    for suffix in ("pkl", "joblib"):
        candidate = artifact_dir / f"model_{int(horizon)}m.{suffix}"
        if candidate.is_file():
            return candidate
    return None


def _default_extra_symbols() -> list[str]:
    raw = os.getenv("RETRAIN_TRAIN_SYMBOLS", "").strip()
    if raw:
        return [symbol.strip() for symbol in raw.split(",") if symbol.strip()]
    return [symbol for symbol in default_universe_symbols() if symbol != "BTCUSDT"]


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    model_root = Path(os.getenv("BOT_MODEL_ROOT", "/app/models/production")).expanduser()
    registry_root = Path(
        os.getenv("BOT_MODEL_REGISTRY_ROOT", str(model_root / "registry"))
    ).expanduser()
    train_months = _env_int("RETRAIN_TRAIN_MONTHS", 6)
    result = export_active_confirmation_shadow(
        model_root=model_root,
        registry_root=registry_root,
        train_months=train_months,
        extra_symbols=_default_extra_symbols(),
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
