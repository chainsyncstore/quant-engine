"""Model recovery experiment runner.

This module turns the recovery spec into an executable research workflow:

1. load a multi-symbol snapshot;
2. build per-symbol features;
3. audit labels and benchmark baselines;
4. evaluate a bounded grid of candidate configurations; and
5. write experiment-only artifacts without touching production registry state.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import argparse
from collections import Counter
import hashlib
import json
import logging
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Literal, Sequence

import numpy as np
import pandas as pd

from quant.features.pipeline import build_features, get_feature_columns
from quant.features.schema import FEATURE_SPECS
from quant_v2.contracts import StrategySignal
from quant_v2.data.storage import build_dataset_manifest, load_multi_symbol_snapshot, validate_multi_symbol_ohlcv
from quant_v2.execution.cost_policy import ExecutionCostPolicy
from quant_v2.models.predictor import predict_proba
from quant_v2.models.trainer import TrainedModel, save_model_bundle, train
from quant_v2.portfolio.cost_model import BinanceCostModel
from quant_v2.research.economic_thresholds import EconomicThresholdConfig, select_threshold_by_utility
from quant_v2.research.candidate_quality import (
    CandidateQualityDecision,
    default_candidate_quality_rules,
    evaluate_candidate_quality,
)
from quant_v2.research.selection_risk import build_selection_risk_report
from quant_v2.research.trade_outcome_labels import (
    TradeOutcomeLabelConfig,
    build_trade_outcome_labels,
    build_trade_outcome_report,
)
from quant_v2.research.regime_context import add_regime_context_features
from quant_v2.research.portfolio_replay import ReplayActorConfig, ReplayScenario, run_portfolio_replay
from quant_v2.validation.temporal_validation import (
    build_temporal_validation_plan,
    compute_recency_weights,
    effective_sample_size,
)

logger = logging.getLogger(__name__)

POLICY_VERSION = "model_recovery_experiment_runner_v1"
LABEL_POLICY_VERSION = "model_recovery_label_audit_v1"
BENCHMARK_POLICY_VERSION = "model_recovery_benchmark_replay_v1"
DEFAULT_HORIZONS: tuple[int, ...] = (2, 4, 8)
DEFAULT_DEAD_ZONES: tuple[float, ...] = (0.001, 0.0015, 0.002, 0.003, 0.005)
DEFAULT_TRAINING_WINDOWS_MONTHS: tuple[int, ...] = (3, 6, 9, 12)
DEFAULT_RECENCY_HALF_LIFES_DAYS: tuple[int, ...] = (30, 60, 90)
DEFAULT_FEATURE_SETS: tuple[str, ...] = (
    "full",
    "price_volume_funding",
    "no_open_interest",
    "no_orderbook_placeholders",
)
DEFAULT_MIN_ACCURACY = 0.60
DEFAULT_MIN_ACTIONABLE_DECISIONS = 100
DEFAULT_MAX_DRAWDOWN_FRAC = 0.25
DEFAULT_MAX_CANDIDATES = 144
DEFAULT_SEED = 1337
DEFAULT_MAX_VALIDATION_FOLDS = 3
DEFAULT_REPORT_FILL_LIMIT = 200
DEFAULT_REPLAY_MAX_BARS_PER_SYMBOL = 240
DEFAULT_MIN_SYMBOL_TAKE_COUNT = int(os.getenv("MODEL_RECOVERY_MIN_SYMBOL_TAKE_COUNT", "10"))
DEFAULT_MAX_SYMBOL_DRAWDOWN_BPS = float(os.getenv("MODEL_RECOVERY_MAX_SYMBOL_DRAWDOWN_BPS", "100.0"))
DEFAULT_MAX_REPLAY_GAP_BPS = 25.0
DEFAULT_MIN_BENCHMARK_MARGIN_BPS = float(os.getenv("MODEL_RECOVERY_MIN_BENCHMARK_MARGIN_BPS", "5.0"))
EXPERIMENT_TRAIN_PARAMS: dict[str, Any] = {
    "n_estimators": 120,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.7,
    "colsample_bytree": 0.7,
    "min_child_samples": 25,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "n_jobs": 1,
}

_FEATURE_GROUPS: dict[str, list[str]] = {}
for spec in FEATURE_SPECS:
    _FEATURE_GROUPS.setdefault(spec.group, []).append(spec.name)
_ALL_FEATURE_COLUMNS: tuple[str, ...] = tuple(spec.name for spec in FEATURE_SPECS)

_FEATURE_SET_GROUPS: dict[str, tuple[str, ...]] = {
    "full": tuple(_FEATURE_GROUPS.keys()),
    "price_volume_funding": (
        "momentum",
        "volatility",
        "candle_geometry",
        "trend",
        "volume",
        "time_encoding",
        "microstructure",
        "cross_timeframe",
        "funding_rate",
        "crypto_session",
    ),
    "no_open_interest": tuple(group for group in _FEATURE_GROUPS.keys() if group not in {"open_interest", "liquidation_proximity"}),
    "no_orderbook_placeholders": tuple(group for group in _FEATURE_GROUPS.keys() if group != "order_book"),
}

_FEATURE_SET_PRIORITY = {name: idx for idx, name in enumerate(DEFAULT_FEATURE_SETS)}


def _normalize_trade_outcome_side(side: str) -> str:
    clean = str(side).strip().lower()
    if clean in {"long", "short"}:
        return clean
    return clean


@dataclass(frozen=True)
class RecoveryCandidateConfig:
    horizon: int
    training_window_months: int
    recency_half_life_days: int
    dead_zone_bps: float
    feature_set: str
    label_mode: str = "directional_return"
    trade_outcome_side: str = "long"
    trade_outcome_profit_target_bps: float = 20.0
    trade_outcome_stop_loss_bps: float = 30.0
    trade_outcome_round_trip_cost_bps: float = 8.0

    def __post_init__(self) -> None:
        normalized_side = _normalize_trade_outcome_side(self.trade_outcome_side)
        object.__setattr__(self, "trade_outcome_side", normalized_side)
        if self.label_mode == "trade_outcome" and normalized_side not in {"long", "short"}:
            raise ValueError(f"unsupported trade_outcome_side={self.trade_outcome_side!r}")

    def candidate_id(self) -> str:
        dead = f"{float(self.dead_zone_bps):.4f}".replace(".", "p")
        base = (
            f"h{int(self.horizon)}"
            f"_tw{int(self.training_window_months)}m"
            f"_hl{int(self.recency_half_life_days)}d"
            f"_dz{dead}"
            f"_fs{self.feature_set}"
        )
        if self.label_mode == "trade_outcome":
            return f"{base}_side{self.trade_outcome_side}"
        return base


@dataclass(frozen=True)
class CandidateEvaluation:
    config: RecoveryCandidateConfig
    candidate_id: str
    passed: bool
    score: float
    failure_reasons: tuple[str, ...]
    dataset_manifest: dict[str, Any]
    feature_manifest: dict[str, Any]
    label_audit: dict[str, Any]
    fold_ledger: dict[str, Any]
    threshold_policy: dict[str, Any]
    holdout_report: dict[str, Any]
    replay_report: dict[str, Any]
    selection_risk_summary: dict[str, Any]
    accuracy_threshold_policy: dict[str, Any] | None = None
    model_artifact_path: str | None = None
    variant_id: str = ""
    benchmark_delta_report: dict[str, Any] = field(default_factory=dict)
    candidate_quality_report: dict[str, Any] = field(default_factory=dict)
    replay_gap_diagnostics: dict[str, Any] = field(default_factory=dict)
    symbol_pruning_report: dict[str, Any] = field(default_factory=dict)
    maintenance_report: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Phase4VariantSpec:
    variant_id: str
    variant_kind: str
    config: RecoveryCandidateConfig


@dataclass(frozen=True)
class Phase4RepairRunResult:
    run_id: str
    output_dir: Path
    docs_output_dir: Path
    snapshot_path: Path
    diagnostics_report: dict[str, Any]
    benchmark_report: dict[str, Any]
    label_audit_report: dict[str, Any]
    variant_evaluations: list[CandidateEvaluation]
    summary: dict[str, Any]


@dataclass(frozen=True)
class ExperimentRunResult:
    run_id: str
    output_dir: Path
    docs_output_dir: Path
    snapshot_path: Path
    evaluated_candidates: list[CandidateEvaluation]
    passed_candidates: list[CandidateEvaluation]
    benchmark_report: dict[str, Any]
    label_audit_report: dict[str, Any]
    summary: dict[str, Any]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def _sha256(payload: Any) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _git_short_sha() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            cwd=Path(__file__).resolve().parents[2],
        )
    except Exception:
        return "unknown"
    sha = result.stdout.strip()
    return sha or "unknown"


def _run_id() -> str:
    return f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{_git_short_sha()}"


def _frame_digest(frame: pd.DataFrame) -> str:
    ordered = frame.sort_index().sort_index(axis=1)
    return _sha256(
        {
            "columns": list(ordered.columns),
            "dtypes": {str(column): str(dtype) for column, dtype in ordered.dtypes.items()},
            "hash": hashlib.sha256(
                pd.util.hash_pandas_object(ordered, index=True).values.tobytes()
            ).hexdigest(),
        }
    )


def _ensure_multiindex_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(frame.index, pd.MultiIndex) or list(frame.index.names) != ["timestamp", "symbol"]:
        raise ValueError("dataset must be MultiIndex with levels ['timestamp', 'symbol']")
    validate_multi_symbol_ohlcv(frame)
    ordered = frame.sort_index().copy()
    ordered.index = ordered.index.set_names(["timestamp", "symbol"])
    return ordered


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _to_jsonable(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, (pd.Timestamp, datetime)):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_to_jsonable(payload), indent=2, sort_keys=True, default=str), encoding="utf-8")


def _markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    if not rows:
        return "_No rows_"
    widths = [len(str(header)) for header in headers]
    for row in rows:
        for idx, item in enumerate(row):
            widths[idx] = max(widths[idx], len(str(item)))
    header_line = "| " + " | ".join(str(h).ljust(widths[i]) for i, h in enumerate(headers)) + " |"
    separator = "| " + " | ".join("-" * widths[i] for i in range(len(headers))) + " |"
    body = [
        "| " + " | ".join(str(item).ljust(widths[i]) for i, item in enumerate(row)) + " |"
        for row in rows
    ]
    return "\n".join([header_line, separator, *body])


def _normalize_feature_set(feature_set: str) -> str:
    clean = str(feature_set).strip().lower()
    if clean not in _FEATURE_SET_GROUPS:
        raise ValueError(f"unknown feature set: {feature_set}")
    return clean


def _feature_columns_for_set(feature_set: str, available_columns: Iterable[str]) -> list[str]:
    clean = _normalize_feature_set(feature_set)
    allowed_groups = set(_FEATURE_SET_GROUPS[clean])
    allowed_names = {
        name
        for group in allowed_groups
        for name in _FEATURE_GROUPS.get(group, [])
    }
    ordered_available = [column for column in _ALL_FEATURE_COLUMNS if column in set(available_columns)]
    selected = [column for column in ordered_available if column in allowed_names]
    if not selected:
        raise ValueError(f"feature set {clean} produced no usable feature columns")
    return selected


def _window_frame(frame: pd.DataFrame, months: int) -> pd.DataFrame:
    if months < 1:
        raise ValueError("months must be positive")
    timestamps = pd.DatetimeIndex(frame.index.get_level_values("timestamp"))
    cutoff = timestamps.max() - pd.DateOffset(months=int(months))
    return frame.loc[timestamps >= cutoff].copy()


def _pre_holdout_training_mask(
    index: pd.MultiIndex,
    *,
    holdout_start: pd.Timestamp | None,
    training_window_months: int,
    exclude_indices: np.ndarray,
) -> np.ndarray:
    timestamps = pd.DatetimeIndex(index.get_level_values("timestamp"))
    mask = np.ones(len(index), dtype=bool)
    mask[np.asarray(exclude_indices, dtype=int)] = False
    if holdout_start is not None:
        cutoff = pd.Timestamp(holdout_start) - pd.DateOffset(months=int(training_window_months))
        mask = mask & (timestamps >= cutoff) & (timestamps < pd.Timestamp(holdout_start))
    return mask


def _select_validation_folds(
    folds: Iterable[Any],
    *,
    preferred_variant: str,
    max_folds: int = DEFAULT_MAX_VALIDATION_FOLDS,
) -> list[Any]:
    candidates = [fold for fold in folds if not getattr(fold, "holdout", False)]
    preferred = [fold for fold in candidates if getattr(fold, "variant", "") == preferred_variant]
    selected = preferred[-max_folds:] if len(preferred) >= max_folds else preferred
    if len(selected) < max_folds:
        selected_ids = {getattr(fold, "fold_id", id(fold)) for fold in selected}
        for fold in candidates:
            fold_id = getattr(fold, "fold_id", id(fold))
            if fold_id in selected_ids:
                continue
            selected.append(fold)
            selected_ids.add(fold_id)
            if len(selected) >= max_folds:
                break
    return list(selected)


def _build_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Build features symbol by symbol so cross-pair helpers receive BTC context."""

    raw = _ensure_multiindex_frame(frame)
    symbols = sorted(str(symbol) for symbol in raw.index.get_level_values("symbol").unique())
    btc_returns: pd.Series | None = None
    if "BTCUSDT" in symbols:
        btc_frame = raw.xs("BTCUSDT", level="symbol").sort_index()
        btc_close = pd.to_numeric(btc_frame["close"], errors="coerce")
        btc_returns = btc_close.pct_change()

    featured_frames: list[pd.DataFrame] = []
    for symbol, sym_frame in raw.groupby(level="symbol", sort=False):
        base = sym_frame.droplevel("symbol").copy()
        if btc_returns is not None and str(symbol).upper() != "BTCUSDT":
            base["_btc_returns"] = btc_returns.reindex(base.index, method="ffill").fillna(0.0)
        try:
            featured = build_features(base)
        except Exception as exc:
            logger.warning("Feature build failed for %s: %s", symbol, exc)
            continue
        if featured.empty:
            continue
        featured = featured.copy()
        featured["symbol"] = str(symbol)
        featured = featured.reset_index().set_index(["timestamp", "symbol"]).sort_index()
        featured_frames.append(featured)

    if not featured_frames:
        empty_index = pd.MultiIndex(levels=[[], []], codes=[[], []], names=["timestamp", "symbol"])
        return pd.DataFrame(index=empty_index)

    combined = pd.concat(featured_frames).sort_index()
    combined.index = combined.index.set_names(["timestamp", "symbol"])
    return combined


def _expected_cost_floor_bps(frame: pd.DataFrame) -> dict[str, Any]:
    cost_model = BinanceCostModel()
    symbol_costs: dict[str, float] = {}
    for symbol, sym_frame in frame.groupby(level="symbol", sort=True):
        close = pd.to_numeric(sym_frame["close"], errors="coerce")
        volume = pd.to_numeric(sym_frame.get("volume"), errors="coerce") if "volume" in sym_frame.columns else None
        quote_volume = pd.to_numeric(sym_frame.get("quote_volume"), errors="coerce") if "quote_volume" in sym_frame.columns else None
        if quote_volume is not None and quote_volume.notna().any():
            adv_usd = float(quote_volume.dropna().tail(24).mean())
        elif volume is not None and close.notna().any():
            adv_usd = float((volume * close).dropna().tail(24).mean())
        else:
            adv_usd = 10_000_000.0
        estimate = cost_model.estimate(str(symbol), 10_000.0, adv_usd=adv_usd)
        symbol_costs[str(symbol)] = float(estimate.round_trip_cost_bps)
    values = list(symbol_costs.values())
    median = float(np.median(values)) if values else 0.0
    return {
        "assumed_notional_usd": 10_000.0,
        "per_symbol_round_trip_cost_bps": symbol_costs,
        "median_round_trip_cost_bps": median,
        "recommended_dead_zone_bps_floor": median,
    }


def _build_labels(frame: pd.DataFrame, horizon: int, dead_zone: float) -> pd.Series:
    labels = pd.Series(np.nan, index=frame.index)
    grouped = frame.groupby(level="symbol", sort=False)
    for _, sym_frame in grouped:
        close = pd.to_numeric(sym_frame["close"], errors="coerce")
        future_return = close.shift(-horizon) / close - 1.0
        sym_labels = pd.Series(np.nan, index=sym_frame.index)
        sym_labels[future_return > dead_zone] = 1
        sym_labels[future_return < -dead_zone] = 0
        labels.loc[sym_frame.index] = sym_labels
    return labels


def _label_stats(frame: pd.DataFrame, labels: pd.Series, *, horizon: int, dead_zone: float) -> dict[str, Any]:
    future_return = pd.Series(np.nan, index=frame.index, dtype=float)
    for _, sym_frame in frame.groupby(level="symbol", sort=False):
        close = pd.to_numeric(sym_frame["close"], errors="coerce")
        future_return.loc[sym_frame.index] = (close.shift(-horizon) / close - 1.0) * 10_000.0
    valid = labels.notna()
    labelled = labels[valid]
    total = int(len(frame))
    up_count = int((labelled == 1).sum())
    down_count = int((labelled == 0).sum())
    ambiguous_count = int((~valid).sum())
    labelled_count = int(valid.sum())
    positive_count = up_count + down_count
    label_pct = {
        "up": float(up_count / total) if total else 0.0,
        "down": float(down_count / total) if total else 0.0,
        "ambiguous": float(ambiguous_count / total) if total else 0.0,
    }
    by_symbol: dict[str, Any] = {}
    by_month: dict[str, Any] = {}
    timestamps = pd.DatetimeIndex(frame.index.get_level_values("timestamp"))
    months = timestamps.to_period("M")
    symbols = frame.index.get_level_values("symbol").astype(str)
    tmp = pd.DataFrame({"label": labels.to_numpy(), "symbol": symbols.to_numpy(), "month": months.to_numpy()})
    for symbol, sym_frame in frame.groupby(level="symbol", sort=True):
        sym_labels = labels.loc[sym_frame.index]
        by_symbol[str(symbol)] = {
            "rows": int(len(sym_frame)),
            "labelled_rows": int(sym_labels.notna().sum()),
            "up_count": int((sym_labels == 1).sum()),
            "down_count": int((sym_labels == 0).sum()),
            "ambiguous_count": int(sym_labels.isna().sum()),
        }
    for (symbol, month), group in tmp.groupby(["symbol", "month"], sort=True):
        key = f"{symbol}:{month}"
        by_month[key] = {
            "rows": int(len(group)),
            "labelled_rows": int(group["label"].notna().sum()),
            "up_count": int((group["label"] == 1).sum()),
            "down_count": int((group["label"] == 0).sum()),
            "ambiguous_count": int(group["label"].isna().sum()),
        }
    forward = future_return.dropna()
    return {
        "sample_count": total,
        "symbol_count": int(frame.index.get_level_values("symbol").nunique()),
        "label_counts": {
            "up": up_count,
            "down": down_count,
            "ambiguous": ambiguous_count,
            "labelled": labelled_count,
        },
        "label_percentages": label_pct,
        "ambiguous_rate": float(ambiguous_count / total) if total else 0.0,
        "forward_return_bps": {
            "mean": float(forward.mean()) if not forward.empty else 0.0,
            "median": float(forward.median()) if not forward.empty else 0.0,
            "std": float(forward.std(ddof=0)) if len(forward) > 1 else 0.0,
            "p25": float(forward.quantile(0.25)) if not forward.empty else 0.0,
            "p75": float(forward.quantile(0.75)) if not forward.empty else 0.0,
        },
        "estimated_cost_floor": _expected_cost_floor_bps(frame),
        "earliest_timestamp": str(timestamps.min()) if len(timestamps) else None,
        "latest_timestamp": str(timestamps.max()) if len(timestamps) else None,
        "by_symbol_label_counts": by_symbol,
        "by_month_label_counts": by_month,
        "label_balance_score": float(
            1.0 - abs((up_count / max(positive_count, 1)) - (down_count / max(positive_count, 1)))
            if positive_count
            else 0.0
        ),
        "horizon": int(horizon),
        "dead_zone_bps": float(dead_zone),
    }


def _build_label_audit_report(frame: pd.DataFrame) -> dict[str, Any]:
    cells: dict[str, Any] = {}
    horizon_summary: dict[str, Any] = {}
    for horizon in DEFAULT_HORIZONS:
        horizon_key = str(horizon)
        horizon_summary[horizon_key] = {}
        for training_window_months in DEFAULT_TRAINING_WINDOWS_MONTHS:
            window_frame = _window_frame(frame, training_window_months)
            if window_frame.empty:
                continue
            horizon_summary[horizon_key][str(training_window_months)] = {}
            for recency_half_life_days in DEFAULT_RECENCY_HALF_LIFES_DAYS:
                half_life_key = str(recency_half_life_days)
                horizon_summary[horizon_key][str(training_window_months)][half_life_key] = {}
                for dead_zone in DEFAULT_DEAD_ZONES:
                    labels = _build_labels(window_frame, horizon, dead_zone)
                    stats = _label_stats(window_frame, labels, horizon=horizon, dead_zone=dead_zone)
                    stats.update(
                        {
                            "training_window_months": int(training_window_months),
                            "recency_half_life_days": int(recency_half_life_days),
                        }
                    )
                    cell_id = (
                        f"h{horizon}_tw{training_window_months}m_hl{recency_half_life_days}d_"
                        f"dz{str(round(float(dead_zone), 4)).replace('.', 'p')}"
                    )
                    cells[cell_id] = stats
                    horizon_summary[horizon_key][str(training_window_months)][half_life_key][f"{dead_zone:.4f}"] = stats

    ranked = sorted(
        cells.items(),
        key=lambda item: (
            -float(item[1].get("label_balance_score", 0.0)),
            float(item[1].get("ambiguous_rate", 1.0)),
            -int(item[1].get("sample_count", 0)),
            int(item[1].get("recency_half_life_days", 0)),
            int(item[1].get("training_window_months", 0)),
            int(item[1].get("horizon", 0)),
            item[0],
        ),
    )

    report = {
        "policy_version": LABEL_POLICY_VERSION,
        "generated_at": _utc_now(),
        "dataset": {
            "rows": int(len(frame)),
            "symbols": sorted(str(symbol) for symbol in frame.index.get_level_values("symbol").unique()),
            "start": str(pd.DatetimeIndex(frame.index.get_level_values("timestamp")).min()),
            "end": str(pd.DatetimeIndex(frame.index.get_level_values("timestamp")).max()),
            "frame_digest": _frame_digest(frame),
        },
        "grid": {
            "horizons": list(DEFAULT_HORIZONS),
            "dead_zones": list(DEFAULT_DEAD_ZONES),
            "training_windows_months": list(DEFAULT_TRAINING_WINDOWS_MONTHS),
            "recency_half_life_days": list(DEFAULT_RECENCY_HALF_LIFES_DAYS),
        },
        "cells": cells,
        "horizons": horizon_summary,
        "summary": {
            "cell_count": len(cells),
            "ranked_cell_ids": [cell_id for cell_id, _ in ranked[:20]],
            "best_cell": ranked[0][0] if ranked else None,
            "best_cell_stats": ranked[0][1] if ranked else None,
            "recommended_dead_zone_floor_bps": _expected_cost_floor_bps(frame)["recommended_dead_zone_bps_floor"],
        },
    }
    return report


def _trade_outcome_label_config(
    *,
    horizon: int,
    dead_zone_fraction: float,
    profit_target_bps: float,
    stop_loss_bps: float,
    round_trip_cost_bps: float,
) -> TradeOutcomeLabelConfig:
    return TradeOutcomeLabelConfig(
        horizon_bars=int(horizon),
        profit_target_bps=float(profit_target_bps),
        stop_loss_bps=float(stop_loss_bps),
        dead_zone_bps=float(dead_zone_fraction) * 10_000.0,
        round_trip_cost_bps=float(round_trip_cost_bps),
    )


def _trade_outcome_label_stats(
    frame: pd.DataFrame,
    labels: pd.Series,
    *,
    horizon: int,
    dead_zone_fraction: float,
    trade_cfg: TradeOutcomeLabelConfig,
) -> dict[str, Any]:
    valid = labels.notna()
    take_count = int((labels == 1).sum())
    skip_count = int((labels == 0).sum())
    ambiguous_count = int((~valid).sum())
    labelled_count = int(valid.sum())
    total = int(len(frame))
    positive_count = take_count + skip_count
    label_pct = {
        "take": float(take_count / total) if total else 0.0,
        "skip": float(skip_count / total) if total else 0.0,
        "ambiguous": float(ambiguous_count / total) if total else 0.0,
    }
    by_symbol: dict[str, Any] = {}
    by_month: dict[str, Any] = {}
    timestamps = pd.DatetimeIndex(frame.index.get_level_values("timestamp"))
    months = timestamps.to_period("M")
    symbols = frame.index.get_level_values("symbol").astype(str)
    tmp = pd.DataFrame({"label": labels.to_numpy(), "symbol": symbols.to_numpy(), "month": months.to_numpy()})
    for symbol, sym_frame in frame.groupby(level="symbol", sort=True):
        sym_labels = labels.loc[sym_frame.index]
        by_symbol[str(symbol)] = {
            "rows": int(len(sym_frame)),
            "labelled_rows": int(sym_labels.notna().sum()),
            "take_count": int((sym_labels == 1).sum()),
            "skip_count": int((sym_labels == 0).sum()),
            "ambiguous_count": int(sym_labels.isna().sum()),
        }
    for (symbol, month), group in tmp.groupby(["symbol", "month"], sort=True):
        key = f"{symbol}:{month}"
        by_month[key] = {
            "rows": int(len(group)),
            "labelled_rows": int(group["label"].notna().sum()),
            "take_count": int((group["label"] == 1).sum()),
            "skip_count": int((group["label"] == 0).sum()),
            "ambiguous_count": int(group["label"].isna().sum()),
        }
    cost_floor = _expected_cost_floor_bps(frame)
    return {
        "sample_count": total,
        "symbol_count": int(frame.index.get_level_values("symbol").nunique()),
        "label_counts": {
            "take": take_count,
            "skip": skip_count,
            "ambiguous": ambiguous_count,
            "labelled": labelled_count,
        },
        "label_percentages": label_pct,
        "ambiguous_rate": float(ambiguous_count / total) if total else 0.0,
        "estimated_cost_floor": cost_floor,
        "earliest_timestamp": str(timestamps.min()) if len(timestamps) else None,
        "latest_timestamp": str(timestamps.max()) if len(timestamps) else None,
        "by_symbol_label_counts": by_symbol,
        "by_month_label_counts": by_month,
        "label_balance_score": float(
            1.0 - abs((take_count / max(positive_count, 1)) - (skip_count / max(positive_count, 1)))
            if positive_count
            else 0.0
        ),
        "horizon": int(horizon),
        "dead_zone_bps": float(dead_zone_fraction),
        "trade_outcome_config": asdict(trade_cfg),
    }


def _build_trade_outcome_label_audit_report(
    frame: pd.DataFrame,
    *,
    profit_target_bps: float,
    stop_loss_bps: float,
    round_trip_cost_bps: float,
) -> dict[str, Any]:
    cells: dict[str, Any] = {}
    horizon_summary: dict[str, Any] = {}
    for horizon in DEFAULT_HORIZONS:
        horizon_key = str(horizon)
        horizon_summary[horizon_key] = {}
        for training_window_months in DEFAULT_TRAINING_WINDOWS_MONTHS:
            window_frame = _window_frame(frame, training_window_months)
            if window_frame.empty:
                continue
            horizon_summary[horizon_key][str(training_window_months)] = {}
            for recency_half_life_days in DEFAULT_RECENCY_HALF_LIFES_DAYS:
                half_life_key = str(recency_half_life_days)
                horizon_summary[horizon_key][str(training_window_months)][half_life_key] = {}
                for dead_zone in DEFAULT_DEAD_ZONES:
                    trade_cfg = _trade_outcome_label_config(
                        horizon=horizon,
                        dead_zone_fraction=dead_zone,
                        profit_target_bps=profit_target_bps,
                        stop_loss_bps=stop_loss_bps,
                        round_trip_cost_bps=round_trip_cost_bps,
                    )
                    labels = build_trade_outcome_labels(window_frame, config=trade_cfg, side="long")
                    stats = _trade_outcome_label_stats(
                        window_frame,
                        labels,
                        horizon=horizon,
                        dead_zone_fraction=dead_zone,
                        trade_cfg=trade_cfg,
                    )
                    stats.update(
                        {
                            "training_window_months": int(training_window_months),
                            "recency_half_life_days": int(recency_half_life_days),
                        }
                    )
                    cell_id = (
                        f"h{horizon}_tw{training_window_months}m_hl{recency_half_life_days}d_"
                        f"dz{str(round(float(dead_zone), 4)).replace('.', 'p')}"
                    )
                    cells[cell_id] = stats
                    horizon_summary[horizon_key][str(training_window_months)][half_life_key][f"{dead_zone:.4f}"] = stats

    ranked = sorted(
        cells.items(),
        key=lambda item: (
            -float(item[1].get("label_balance_score", 0.0)),
            float(item[1].get("ambiguous_rate", 1.0)),
            -int(item[1].get("sample_count", 0)),
            int(item[1].get("recency_half_life_days", 0)),
            int(item[1].get("training_window_months", 0)),
            int(item[1].get("horizon", 0)),
            item[0],
        ),
    )

    report = {
        "policy_version": "trade_outcome_label_audit_v1",
        "generated_at": _utc_now(),
        "label_mode": "trade_outcome",
        "dataset": {
            "rows": int(len(frame)),
            "symbols": sorted(str(symbol) for symbol in frame.index.get_level_values("symbol").unique()),
            "start": str(pd.DatetimeIndex(frame.index.get_level_values("timestamp")).min()),
            "end": str(pd.DatetimeIndex(frame.index.get_level_values("timestamp")).max()),
            "frame_digest": _frame_digest(frame),
        },
        "grid": {
            "horizons": list(DEFAULT_HORIZONS),
            "dead_zones": list(DEFAULT_DEAD_ZONES),
            "training_windows_months": list(DEFAULT_TRAINING_WINDOWS_MONTHS),
            "recency_half_life_days": list(DEFAULT_RECENCY_HALF_LIFES_DAYS),
        },
        "cells": cells,
        "horizons": horizon_summary,
        "summary": {
            "cell_count": len(cells),
            "ranked_cell_ids": [cell_id for cell_id, _ in ranked[:20]],
            "best_cell": ranked[0][0] if ranked else None,
            "best_cell_stats": ranked[0][1] if ranked else None,
            "recommended_dead_zone_floor_bps": _expected_cost_floor_bps(frame)["recommended_dead_zone_bps_floor"],
        },
    }
    return report


def _summarize_prediction_audit(index: pd.MultiIndex, probabilities: np.ndarray, predictions: np.ndarray) -> dict[str, Any]:
    probs = np.asarray(probabilities, dtype=float)
    preds = np.asarray(predictions, dtype=int)
    timestamps = pd.DatetimeIndex(index.get_level_values("timestamp"))
    symbols = index.get_level_values("symbol").astype(str)
    if probs.size == 0 or preds.size == 0:
        return {
            "sample_count": 0,
            "predicted_buy_count": 0,
            "predicted_sell_count": 0,
            "one_sided_collapse": False,
            "by_symbol": {},
            "by_month": {},
            "probability_summary": {"mean": 0.0, "median": 0.0, "p05": 0.0, "p95": 0.0},
        }

    def _summary(mask: np.ndarray) -> dict[str, Any]:
        mask = np.asarray(mask, dtype=bool)
        masked_preds = preds[mask]
        masked_probs = probs[mask]
        buy = int((masked_preds == 1).sum())
        sell = int((masked_preds == 0).sum())
        total = int(len(masked_preds))
        collapse_share = max(buy, sell) / total if total else 0.0
        return {
            "sample_count": total,
            "predicted_buy_count": buy,
            "predicted_sell_count": sell,
            "buy_share": float(buy / total) if total else 0.0,
            "sell_share": float(sell / total) if total else 0.0,
            "one_sided_collapse": bool(total and (buy == 0 or sell == 0 or collapse_share >= 0.95)),
            "probability_summary": {
                "mean": float(masked_probs.mean()) if total else 0.0,
                "median": float(np.median(masked_probs)) if total else 0.0,
                "p05": float(np.quantile(masked_probs, 0.05)) if total else 0.0,
                "p95": float(np.quantile(masked_probs, 0.95)) if total else 0.0,
            },
        }

    by_symbol: dict[str, Any] = {}
    for symbol in sorted(set(symbols)):
        by_symbol[symbol] = _summary(symbols == symbol)

    by_month: dict[str, Any] = {}
    for month in sorted(set(timestamps.to_period("M"))):
        by_month[str(month)] = _summary(timestamps.to_period("M") == month)

    overall = _summary(np.ones(len(preds), dtype=bool))
    overall["by_symbol"] = by_symbol
    overall["by_month"] = by_month
    return overall


def _feature_health_report(feature_frame: pd.DataFrame, feature_columns: list[str]) -> dict[str, Any]:
    selected = [str(column) for column in feature_columns]
    available = [column for column in selected if column in feature_frame.columns]
    selected_frame = feature_frame.loc[:, available].copy() if available else feature_frame.iloc[:, 0:0].copy()
    missingness = {
        column: float(feature_frame[column].isna().mean()) if column in feature_frame.columns else 1.0
        for column in selected
    }
    constant_features = [
        column
        for column in available
        if selected_frame[column].nunique(dropna=True) <= 1
    ]
    top_missing = sorted(missingness.items(), key=lambda item: (item[1], item[0]), reverse=True)[:20]
    return {
        "selected_feature_count": int(len(selected)),
        "available_feature_count": int(len(available)),
        "selected_features": selected,
        "available_features": available,
        "missingness_by_feature": missingness,
        "top_missing_features": [{"feature": name, "missingness": value} for name, value in top_missing],
        "constant_feature_count": int(len(constant_features)),
        "constant_features": constant_features,
        "feature_missingness_max": float(max(missingness.values())) if missingness else 0.0,
        "feature_missingness_avg": float(sum(missingness.values()) / len(missingness)) if missingness else 0.0,
    }


def _feature_importance_report(model: Any, *, limit: int = 20) -> dict[str, Any]:
    raw = getattr(model, "feature_importance", {}) or {}
    if not isinstance(raw, dict):
        raw = {}
    ranked = sorted(((str(name), float(value)) for name, value in raw.items()), key=lambda item: (item[1], item[0]), reverse=True)
    return {
        "feature_count": int(len(raw)),
        "top_features": [{"feature": name, "importance": value} for name, value in ranked[:limit]],
    }


def _build_timestamp_context(frame: pd.DataFrame) -> pd.DataFrame:
    enriched = add_regime_context_features(frame)
    context_columns = [column for column in enriched.columns if column.startswith("regime_")]
    if "funding_rate_raw" in enriched.columns:
        context_columns.append("funding_rate_raw")
    if "close" in enriched.columns:
        context_columns.append("close")
    timestamp_context = enriched.loc[:, list(dict.fromkeys(context_columns))].groupby(level="timestamp", sort=True).first().sort_index()
    btc_frame = frame.xs("BTCUSDT", level="symbol").sort_index() if "BTCUSDT" in frame.index.get_level_values("symbol") else None
    if btc_frame is not None and not btc_frame.empty:
        btc_close = pd.to_numeric(btc_frame["close"], errors="coerce")
        btc_ret_24h = btc_close.pct_change(24)
        btc_drawdown_24h = btc_close / btc_close.cummax() - 1.0
        btc_context = pd.DataFrame(
            {
                "btc_return_24h": btc_ret_24h,
                "btc_drawdown_24h": btc_drawdown_24h,
            }
        )
        timestamp_context = timestamp_context.join(btc_context, how="left")
    else:
        timestamp_context["btc_return_24h"] = np.nan
        timestamp_context["btc_drawdown_24h"] = np.nan
    return timestamp_context


def _slice_frame_by_timestamps(frame: pd.DataFrame, timestamps: pd.Index) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    ts_index = pd.DatetimeIndex(timestamps)
    mask = frame.index.get_level_values("timestamp").isin(ts_index)
    return frame.loc[mask].copy()


def _benchmark_regime_comparison(frame: pd.DataFrame) -> dict[str, Any]:
    if not isinstance(frame.index, pd.MultiIndex) or list(frame.index.names) != ["timestamp", "symbol"]:
        return {}
    context = _build_timestamp_context(frame)
    if context.empty:
        return {}

    regimes: dict[str, pd.Index] = {}
    regimes["high_vol"] = context.index[context["regime_high_vol_flag"].fillna(0.0) >= 0.5]
    regimes["low_vol"] = context.index[context["regime_high_vol_flag"].fillna(0.0) < 0.5]
    regimes["btc_trend_up"] = context.index[context["btc_return_24h"].fillna(0.0) >= 0.0]
    regimes["btc_trend_down"] = context.index[context["btc_return_24h"].fillna(0.0) < 0.0]
    regimes["funding_positive"] = context.index[context.get("funding_rate_raw", pd.Series(index=context.index, dtype=float)).fillna(0.0) >= 0.0]
    regimes["funding_negative"] = context.index[context.get("funding_rate_raw", pd.Series(index=context.index, dtype=float)).fillna(0.0) < 0.0]
    regimes["drawdown_window"] = context.index[context["btc_drawdown_24h"].fillna(0.0) <= -0.05]
    regimes["recovery_window"] = context.index[(context["btc_drawdown_24h"].fillna(0.0) > -0.05) & (context["btc_return_24h"].fillna(0.0) >= 0.0)]

    report: dict[str, Any] = {}
    for name, ts_index in regimes.items():
        sliced = _slice_frame_by_timestamps(frame, ts_index)
        if sliced.empty or sliced.index.get_level_values("symbol").nunique() < 2:
            report[name] = {"available": False, "rows": int(len(sliced))}
            continue
        benchmark = _benchmark_replay_report_core(sliced, include_cross_sections=False)
        report[name] = {
            "available": True,
            "rows": int(len(sliced)),
            "symbols": sorted(str(symbol) for symbol in sliced.index.get_level_values("symbol").unique()),
            "best_actor": benchmark.get("comparisons", {}).get("best_actor"),
            "best_minus_flat_cost_adjusted_net_pnl_usd": float(
                benchmark.get("comparisons", {}).get("best_minus_flat_cost_adjusted_net_pnl_usd", 0.0)
            ),
            "flat_cost_adjusted_net_pnl_usd": float(
                benchmark.get("comparisons", {}).get("flat_cost_adjusted_net_pnl_usd", 0.0)
            ),
        }
    return report


def _momentum_signal(history: pd.DataFrame, *, horizon_bars: int, invert: bool = False, vol_filter_bps: float | None = None) -> dict[str, Any]:
    close = pd.to_numeric(history.get("close"), errors="coerce").dropna()
    if len(close) <= 4:
        return {"signal": "HOLD", "confidence": 0.5, "uncertainty": 0.5, "reason": "insufficient_history"}
    if vol_filter_bps is not None and len(close) >= 20:
        vol = float(pd.Series(close.pct_change().dropna()).tail(20).std() or 0.0) * 10_000.0
        if vol > vol_filter_bps:
            return {"signal": "HOLD", "confidence": 0.5, "uncertainty": 0.5, "reason": "vol_filter_hold"}
    recent_return = float(close.iloc[-1] / close.iloc[-5] - 1.0)
    if abs(recent_return) < 0.003:
        return {"signal": "HOLD", "confidence": 0.5, "uncertainty": 0.5, "reason": "deadband"}
    if recent_return > 0.0:
        signal = "SELL" if invert else "BUY"
    else:
        signal = "BUY" if invert else "SELL"
    confidence = min(0.95, 0.55 + abs(recent_return) * 30.0)
    return {
        "signal": signal,
        "confidence": float(confidence),
        "uncertainty": float(max(0.0, 1.0 - confidence)),
        "reason": "benchmark_momentum" if not invert else "benchmark_mean_reversion",
    }


def _always_long_signal(history: pd.DataFrame) -> dict[str, Any]:
    if len(history) < 2:
        return {"signal": "HOLD", "confidence": 0.5, "uncertainty": 0.5, "reason": "insufficient_history"}
    return {"signal": "BUY", "confidence": 0.58, "uncertainty": 0.42, "reason": "long_only_baseline"}


def _always_short_signal(history: pd.DataFrame) -> dict[str, Any]:
    if len(history) < 2:
        return {"signal": "HOLD", "confidence": 0.5, "uncertainty": 0.5, "reason": "insufficient_history"}
    return {"signal": "SELL", "confidence": 0.58, "uncertainty": 0.42, "reason": "short_only_baseline"}


def _moving_average_signal(history: pd.DataFrame, *, short_window: int = 8, long_window: int = 24) -> dict[str, Any]:
    close = pd.to_numeric(history.get("close"), errors="coerce").dropna()
    if len(close) < max(short_window, long_window, 5):
        return {"signal": "HOLD", "confidence": 0.5, "uncertainty": 0.5, "reason": "insufficient_history"}
    short_ma = float(close.tail(short_window).mean())
    long_ma = float(close.tail(long_window).mean())
    if abs(short_ma - long_ma) / max(long_ma, 1e-9) < 0.002:
        return {"signal": "HOLD", "confidence": 0.5, "uncertainty": 0.5, "reason": "ma_deadband"}
    if short_ma > long_ma:
        signal = "BUY"
    else:
        signal = "SELL"
    confidence = min(0.95, 0.55 + abs(short_ma - long_ma) / max(long_ma, 1e-9) * 20.0)
    return {
        "signal": signal,
        "confidence": float(confidence),
        "uncertainty": float(max(0.0, 1.0 - confidence)),
        "reason": "moving_average_trend",
    }


def _volatility_breakout_signal(history: pd.DataFrame) -> dict[str, Any]:
    close = pd.to_numeric(history.get("close"), errors="coerce").dropna()
    if len(close) < 20:
        return {"signal": "HOLD", "confidence": 0.5, "uncertainty": 0.5, "reason": "insufficient_history"}
    recent = close.tail(20)
    latest = float(recent.iloc[-1])
    upper = float(recent.iloc[:-1].max()) if len(recent) > 1 else latest
    lower = float(recent.iloc[:-1].min()) if len(recent) > 1 else latest
    if latest >= upper * 1.01:
        return {"signal": "BUY", "confidence": 0.76, "uncertainty": 0.24, "reason": "volatility_breakout_up"}
    if latest <= lower * 0.99:
        return {"signal": "SELL", "confidence": 0.76, "uncertainty": 0.24, "reason": "volatility_breakout_down"}
    return {"signal": "HOLD", "confidence": 0.5, "uncertainty": 0.5, "reason": "volatility_breakout_hold"}


def _funding_aware_abstain_signal(history: pd.DataFrame) -> dict[str, Any]:
    funding_col = None
    for column in ("funding_rate_raw", "funding_rate", "funding_bps"):
        if column in history.columns:
            funding_col = column
            break
    if funding_col is None:
        return {"signal": "HOLD", "confidence": 0.5, "uncertainty": 0.5, "reason": "funding_unavailable"}
    funding = pd.to_numeric(history[funding_col], errors="coerce").fillna(0.0)
    if len(funding) == 0:
        return {"signal": "HOLD", "confidence": 0.5, "uncertainty": 0.5, "reason": "insufficient_history"}
    latest = float(funding.iloc[-1])
    if abs(latest) > 0.00025:
        return {"signal": "HOLD", "confidence": 0.55, "uncertainty": 0.45, "reason": "funding_abstain"}
    return _momentum_signal(history, horizon_bars=4)


def _adverse_excursion_exit_signal(history: pd.DataFrame) -> dict[str, Any]:
    close = pd.to_numeric(history.get("close"), errors="coerce").dropna()
    if len(close) < 12:
        return {"signal": "HOLD", "confidence": 0.5, "uncertainty": 0.5, "reason": "insufficient_history"}
    drawdown = float(close.iloc[-1] / close.cummax().iloc[-1] - 1.0)
    if drawdown <= -0.05:
        return {"signal": "HOLD", "confidence": 0.55, "uncertainty": 0.45, "reason": "adverse_excursion_exit"}
    return _momentum_signal(history, horizon_bars=4, invert=True, vol_filter_bps=140.0)


def _benchmark_signal_resolver(actor: ReplayActorConfig, symbol: str, history: pd.DataFrame, timestamp: pd.Timestamp, market_risk) -> Any:
    name = str(actor.metadata.get("benchmark_name") or actor.name).strip().lower()
    if name == "flat":
        return StrategySignal(
            symbol=symbol,
            timeframe="1h",
            horizon_bars=actor.horizon_bars,
            signal="HOLD",
            confidence=0.5,
            uncertainty=0.5,
            reason="flat_benchmark",
            market_risk=market_risk,
        )
    if name == "momentum":
        payload = _momentum_signal(history, horizon_bars=actor.horizon_bars)
    elif name == "mean_reversion":
        payload = _momentum_signal(history, horizon_bars=actor.horizon_bars, invert=True)
    elif name == "volatility_filtered":
        payload = _momentum_signal(history, horizon_bars=actor.horizon_bars, vol_filter_bps=120.0)
    elif name == "long_only":
        payload = _always_long_signal(history)
    elif name == "short_only":
        payload = _always_short_signal(history)
    elif name == "moving_average_trend":
        payload = _moving_average_signal(history)
    elif name == "volatility_breakout":
        payload = _volatility_breakout_signal(history)
    elif name == "adverse_excursion_exit":
        payload = _adverse_excursion_exit_signal(history)
    elif name == "funding_aware_abstain":
        payload = _funding_aware_abstain_signal(history)
    else:
        return None
    return StrategySignal(
        symbol=symbol,
        timeframe="1h",
        horizon_bars=actor.horizon_bars,
        signal=payload["signal"],
        confidence=payload["confidence"],
        uncertainty=payload["uncertainty"],
        reason=payload["reason"],
        market_risk=market_risk,
    )


def _benchmark_actor_availability(frame: pd.DataFrame) -> dict[str, dict[str, Any]]:
    has_funding = any(column in frame.columns for column in ("funding_rate_raw", "funding_rate", "funding_bps"))
    return {
        "flat": {"available": True, "reason": "baseline"},
        "momentum": {"available": True, "reason": "baseline"},
        "mean_reversion": {"available": True, "reason": "baseline"},
        "volatility_filtered": {"available": True, "reason": "baseline"},
        "long_only": {"available": True, "reason": "baseline"},
        "short_only": {"available": True, "reason": "baseline"},
        "moving_average_trend": {"available": True, "reason": "baseline"},
        "volatility_breakout": {"available": True, "reason": "baseline"},
        "adverse_excursion_exit": {"available": True, "reason": "baseline"},
        "funding_aware_abstain": {"available": bool(has_funding), "reason": "requires_funding_columns" if not has_funding else "baseline"},
    }


def _candidate_signal_resolver(
    actor: ReplayActorConfig,
    symbol: str,
    history: pd.DataFrame,
    timestamp: pd.Timestamp,
    market_risk,
    *,
    feature_frame: pd.DataFrame | None = None,
) -> Any:
    feature_columns = list(actor.metadata.get("feature_columns") or [])
    model = actor.model
    if model is None or not feature_columns:
        return None
    row: pd.DataFrame
    if feature_frame is not None and not feature_frame.empty:
        available = [column for column in feature_columns if column in feature_frame.columns]
        if len(available) != len(feature_columns):
            return None
        key = (pd.Timestamp(timestamp), symbol)
        if key not in feature_frame.index:
            return None
        row = feature_frame.loc[[key], available]
    else:
        featured = build_features(history.copy())
        if featured.empty:
            return None
        available = [column for column in feature_columns if column in featured.columns]
        if len(available) != len(feature_columns):
            return None
        row = featured.iloc[[-1]][available]
    if len(available) != len(feature_columns):
        return None
    if bool(row.isna().any(axis=None)):
        return None
    proba = float(predict_proba(model, row)[0])
    threshold = float(actor.threshold)
    label_mode = str(actor.metadata.get("label_mode", "directional_return"))
    trade_outcome_side = _normalize_trade_outcome_side(str(actor.metadata.get("trade_outcome_side", "long")))
    if label_mode == "trade_outcome":
        if proba >= threshold and trade_outcome_side == "long":
            signal = "BUY"
            confidence = proba
            reason = f"trade_outcome_long_take>={threshold:.2f}"
        elif proba >= threshold and trade_outcome_side == "short":
            signal = "SELL"
            confidence = proba
            reason = f"trade_outcome_short_take>={threshold:.2f}"
        else:
            signal = "HOLD"
            confidence = max(proba, 1.0 - proba)
            reason = "trade_outcome_skip"
    elif label_mode == "directional_return":
        if proba >= threshold:
            signal = "BUY"
            confidence = proba
            reason = f"candidate_proba_up>={threshold:.2f}"
        elif proba <= 1.0 - threshold:
            signal = "SELL"
            confidence = 1.0 - proba
            reason = f"candidate_proba_down>={threshold:.2f}"
        else:
            signal = "HOLD"
            confidence = max(proba, 1.0 - proba)
            reason = "candidate_deadband"
    else:
        signal = "HOLD"
        confidence = max(proba, 1.0 - proba)
        reason = "unsupported_label_mode"
    return StrategySignal(
        symbol=symbol,
        timeframe="1h",
        horizon_bars=actor.horizon_bars,
        signal=signal,
        confidence=float(confidence),
        uncertainty=float(max(0.0, 1.0 - confidence)),
        reason=reason,
        market_risk=market_risk,
    )


def _close_lookup_by_symbol(dataset: pd.DataFrame) -> dict[str, tuple[pd.DatetimeIndex, np.ndarray]]:
    lookup: dict[str, tuple[pd.DatetimeIndex, np.ndarray]] = {}
    for symbol, symbol_frame in dataset.groupby(level="symbol", sort=False):
        ordered = symbol_frame.droplevel("symbol").sort_index()
        close = pd.to_numeric(ordered["close"], errors="coerce").dropna()
        lookup[str(symbol)] = (pd.DatetimeIndex(close.index), close.to_numpy(dtype=float))
    return lookup


def _tail_bars_per_symbol(frame: pd.DataFrame, max_bars_per_symbol: int) -> pd.DataFrame:
    max_bars_per_symbol = int(max_bars_per_symbol)
    if max_bars_per_symbol <= 0 or frame.empty:
        return frame.sort_index().copy()
    pieces = [
        symbol_frame.tail(max_bars_per_symbol)
        for _, symbol_frame in frame.sort_index().groupby(level="symbol", sort=False)
    ]
    if not pieces:
        return frame.iloc[0:0].copy()
    return pd.concat(pieces, axis=0).sort_index()


def _replay_manifest(replay) -> dict[str, Any]:
    actors: dict[str, Any] = {}
    for name, result in replay.actors.items():
        fills_payload = [asdict(fill) for fill in result.fills]
        actors[str(name)] = {
            "actor": result.actor,
            "metrics": dict(result.metrics),
            "state_digest": result.state_digest,
            "manifest": dict(result.manifest),
            "equity_curve_points": int(len(result.equity_curve)),
            "equity_curve_digest": _sha256(result.equity_curve),
            "fill_count": int(len(result.fills)),
            "fills_digest": _sha256(fills_payload),
            "blocked_intent_count": int(len(result.blocked_intents)),
            "blocked_intents_digest": _sha256(result.blocked_intents),
            "risk_transition_count": int(len(result.risk_transitions)),
            "risk_transitions_digest": _sha256(result.risk_transitions),
        }
    return {
        "replay_digest": replay.replay_digest,
        "manifest": dict(replay.manifest),
        "timestamp_count": int(replay.timestamp_count),
        "event_count": int(replay.event_count),
        "actors": actors,
    }


def _summarize_replay_actor(
    result,
    dataset: pd.DataFrame,
    *,
    close_lookup: dict[str, tuple[pd.DatetimeIndex, np.ndarray]] | None = None,
    max_report_fills: int = DEFAULT_REPORT_FILL_LIMIT,
) -> dict[str, Any]:
    fills = list(result.fills)
    max_report_fills = max(0, int(max_report_fills))
    fills_payload = [asdict(fill) for fill in fills[:max_report_fills]]
    total_trade_notional = sum(float(fill.filled_qty) * float(fill.price) for fill in fills)
    total_fees = float(result.metrics.get("total_fees_usd", 0.0))
    total_slippage = float(result.metrics.get("total_slippage_usd", 0.0))
    net_pnl = float(result.metrics.get("net_pnl_usd", 0.0))
    max_drawdown_frac = float(result.metrics.get("max_drawdown_frac", 0.0))
    equity_curve = list(result.equity_curve)
    exposure_by_symbol = {symbol: 0.0 for symbol in dataset.index.get_level_values("symbol").unique()}
    close_lookup = close_lookup or _close_lookup_by_symbol(dataset)
    for point in equity_curve:
        positions = point.get("positions") or {}
        timestamp = pd.Timestamp(point["timestamp"])
        for symbol, qty in positions.items():
            lookup = close_lookup.get(str(symbol))
            if lookup is None:
                continue
            index, closes = lookup
            if len(index) == 0:
                continue
            close_idx = int(index.searchsorted(timestamp, side="right")) - 1
            if close_idx < 0:
                continue
            close = float(closes[close_idx])
            exposure_by_symbol[str(symbol)] = max(exposure_by_symbol.get(str(symbol), 0.0), abs(float(qty)) * close)
    return {
        "actor": result.actor,
        "metrics": dict(result.metrics),
        "net_pnl_usd": net_pnl,
        "cost_adjusted_net_pnl_usd": net_pnl,
        "total_fees_usd": total_fees,
        "total_slippage_usd": total_slippage,
        "max_drawdown_frac": max_drawdown_frac,
        "fill_count": int(result.metrics.get("fill_count", 0) or 0),
        "closed_trade_count": int(sum(1 for fill in fills if fill.outcome in {"NEW_FILL", "PARTIAL_FILL"})),
        "blocked_intents": int(result.metrics.get("blocked_intents", 0) or 0),
        "turnover": float(result.metrics.get("turnover", 0.0)),
        "average_trade_usd": float(total_trade_notional / max(len(fills), 1)),
        "exposure_by_symbol_usd": {symbol: float(value) for symbol, value in sorted(exposure_by_symbol.items())},
        "state_digest": result.state_digest,
        "manifest": dict(result.manifest),
        "fills": fills_payload,
        "fills_payload_count": int(len(fills_payload)),
        "fills_truncated": bool(len(fills_payload) < len(fills)),
    }


def _benchmark_actor_config(name: str) -> ReplayActorConfig:
    return ReplayActorConfig(
        name=name,
        kind="fixed",
        min_confidence=0.55,
        horizon_bars=4,
        baseline_lookback=4,
        baseline_deadband=0.003,
        metadata={"benchmark_name": name},
    )


def _candidate_replay_report(
    frame: pd.DataFrame,
    *,
    model: TrainedModel,
    feature_columns: list[str],
    threshold: float,
    horizon: int,
    label_mode: str = "directional_return",
    trade_outcome_side: str = "long",
    replay_feature_frame: pd.DataFrame | None = None,
    max_bars_per_symbol: int = DEFAULT_REPLAY_MAX_BARS_PER_SYMBOL,
    allowed_symbols: Sequence[str] | None = None,
    signal_resolver: Any | None = None,
) -> dict[str, Any]:
    source_frame = frame
    if allowed_symbols is not None:
        frame = _slice_frame_by_symbols(frame, allowed_symbols)
        if replay_feature_frame is not None:
            replay_feature_frame = _slice_frame_by_symbols(replay_feature_frame, allowed_symbols)
    frame = _tail_bars_per_symbol(frame, max_bars_per_symbol)
    if frame.empty:
        empty_actor = {
            "actor": "candidate",
            "metrics": {"net_pnl_usd": 0.0, "fill_count": 0, "total_fees_usd": 0.0, "total_slippage_usd": 0.0, "max_drawdown_frac": 0.0},
            "net_pnl_usd": 0.0,
            "cost_adjusted_net_pnl_usd": 0.0,
            "total_fees_usd": 0.0,
            "total_slippage_usd": 0.0,
            "max_drawdown_frac": 0.0,
            "fill_count": 0,
            "closed_trade_count": 0,
            "blocked_intents": 0,
            "turnover": 0.0,
            "average_trade_usd": 0.0,
            "exposure_by_symbol_usd": {},
            "state_digest": _sha256({"actor": "candidate", "frame_digest": None, "empty": True}),
            "manifest": {},
            "fills": [],
            "fills_payload_count": 0,
            "fills_truncated": False,
        }
        empty_flat = dict(empty_actor, actor="flat", state_digest=_sha256({"actor": "flat", "frame_digest": None, "empty": True}))
        return {
            "policy_version": POLICY_VERSION,
            "generated_at": _utc_now(),
            "dataset": {
                "rows": 0,
                "source_rows": int(len(source_frame)),
                "max_bars_per_symbol": int(max_bars_per_symbol),
                "truncated_for_replay": False,
                "symbols": [],
                "start": None,
                "end": None,
                "allowed_symbols": list(allowed_symbols) if allowed_symbols is not None else [],
            },
            "replay_digest": _sha256({"empty_candidate_replay": True, "allowed_symbols": list(allowed_symbols) if allowed_symbols is not None else []}),
            "replay_artifact_policy": {
                "full_replay_embedded": False,
                "max_report_fills_per_actor": int(DEFAULT_REPORT_FILL_LIMIT),
                "reason": "empty replay frame after symbol pruning",
            },
            "actor_summaries": {
                "candidate": empty_actor,
                "flat": empty_flat,
            },
            "comparisons": {
                "candidate_minus_flat_cost_adjusted_net_pnl_usd": 0.0,
                "candidate_minus_flat_cost_adjusted_net_pnl_bps": 0.0,
                "best_actor": "flat",
            },
            "replay": {
                "replay_digest": _sha256({"empty_candidate_replay": True}),
                "manifest": {"empty": True},
                "timestamp_count": 0,
                "event_count": 0,
                "actors": {},
            },
            "warnings": ["empty_replay_frame_after_symbol_pruning"],
        }
    if replay_feature_frame is None:
        replay_feature_frame = _build_feature_frame(frame)

    if signal_resolver is None:
        def _precomputed_candidate_resolver(
            actor: ReplayActorConfig,
            symbol: str,
            history: pd.DataFrame,
            timestamp: pd.Timestamp,
            market_risk,
        ) -> Any:
            return _candidate_signal_resolver(
                actor,
                symbol,
                history,
                timestamp,
                market_risk,
                feature_frame=replay_feature_frame,
            )

        signal_resolver = _precomputed_candidate_resolver

    actors = {
        "candidate": ReplayActorConfig(
            name="candidate",
            kind="model",
            model=model,
            threshold=float(threshold),
            min_confidence=0.55,
            horizon_bars=int(horizon),
            metadata={
                "feature_columns": list(feature_columns),
                "benchmark_name": "candidate",
                "label_mode": str(label_mode),
                "trade_outcome_side": _normalize_trade_outcome_side(trade_outcome_side),
            },
        ),
        "flat": _benchmark_actor_config("flat"),
    }
    replay = run_portfolio_replay(
        frame,
        actors,
        initial_equity=1_000.0,
        scenario=ReplayScenario(name="candidate_replay"),
        dataset_manifest={
            "policy_version": POLICY_VERSION,
            "frame_digest": _frame_digest(frame),
        },
        cost_policy=ExecutionCostPolicy(),
        signal_resolver=signal_resolver,
        throttle_allocation_logs=True,
        research_mode=True,
    )
    close_lookup = _close_lookup_by_symbol(frame)
    actor_summaries = {
        name: _summarize_replay_actor(result, frame, close_lookup=close_lookup)
        for name, result in replay.actors.items()
    }
    flat = actor_summaries["flat"]
    candidate = actor_summaries["candidate"]
    return {
        "policy_version": POLICY_VERSION,
        "generated_at": _utc_now(),
        "dataset": {
            "rows": int(len(frame)),
            "source_rows": int(len(source_frame)),
            "max_bars_per_symbol": int(max_bars_per_symbol),
            "truncated_for_replay": bool(len(frame) < len(source_frame)),
            "symbols": sorted(str(symbol) for symbol in frame.index.get_level_values("symbol").unique()),
            "start": str(pd.DatetimeIndex(frame.index.get_level_values("timestamp")).min()),
            "end": str(pd.DatetimeIndex(frame.index.get_level_values("timestamp")).max()),
            "allowed_symbols": list(allowed_symbols) if allowed_symbols is not None else sorted(str(symbol) for symbol in frame.index.get_level_values("symbol").unique()),
        },
        "replay_digest": replay.replay_digest,
        "replay_artifact_policy": {
            "full_replay_embedded": False,
            "max_report_fills_per_actor": int(DEFAULT_REPORT_FILL_LIMIT),
            "reason": "compact reports keep real-data recovery runs bounded while preserving replay digests and actor metrics",
        },
        "actor_summaries": actor_summaries,
        "comparisons": {
            "candidate_minus_flat_cost_adjusted_net_pnl_usd": float(candidate["cost_adjusted_net_pnl_usd"] - flat["cost_adjusted_net_pnl_usd"]),
            "candidate_minus_flat_cost_adjusted_net_pnl_bps": float(
                (candidate["cost_adjusted_net_pnl_usd"] - flat["cost_adjusted_net_pnl_usd"]) / 1_000.0 * 10_000.0
            ),
            "best_actor": "candidate" if candidate["cost_adjusted_net_pnl_usd"] >= flat["cost_adjusted_net_pnl_usd"] else "flat",
        },
        "replay": _replay_manifest(replay),
    }


def _benchmark_replay_report_core(
    frame: pd.DataFrame,
    *,
    max_bars_per_symbol: int = DEFAULT_REPLAY_MAX_BARS_PER_SYMBOL,
    include_cross_sections: bool = True,
) -> dict[str, Any]:
    source_frame = frame
    frame = _tail_bars_per_symbol(frame, max_bars_per_symbol)
    actor_names = (
        "flat",
        "momentum",
        "mean_reversion",
        "volatility_filtered",
        "long_only",
        "short_only",
        "moving_average_trend",
        "volatility_breakout",
        "adverse_excursion_exit",
        "funding_aware_abstain",
    )
    if frame.empty:
        actor_summaries = {
            name: {
                "actor": name,
                "metrics": {"net_pnl_usd": 0.0, "fill_count": 0, "max_drawdown_frac": 0.0},
                "net_pnl_usd": 0.0,
                "cost_adjusted_net_pnl_usd": 0.0,
                "total_fees_usd": 0.0,
                "total_slippage_usd": 0.0,
                "max_drawdown_frac": 0.0,
                "fill_count": 0,
                "closed_trade_count": 0,
                "blocked_intents": 0,
                "turnover": 0.0,
                "average_trade_usd": 0.0,
                "exposure_by_symbol_usd": {},
                "state_digest": _sha256({"actor": name, "empty": True}),
                "manifest": {},
                "fills": [],
                "fills_payload_count": 0,
                "fills_truncated": False,
            }
            for name in actor_names
        }
        return {
            "policy_version": BENCHMARK_POLICY_VERSION,
            "generated_at": _utc_now(),
            "dataset": {
                "rows": 0,
                "source_rows": int(len(source_frame)),
                "max_bars_per_symbol": int(max_bars_per_symbol),
                "truncated_for_replay": False,
                "symbols": [],
                "start": None,
                "end": None,
            },
            "replay_digest": _sha256({"empty_benchmark_replay": True}),
            "replay_artifact_policy": {
                "full_replay_embedded": False,
                "max_report_fills_per_actor": int(DEFAULT_REPORT_FILL_LIMIT),
                "reason": "empty replay frame after symbol pruning",
            },
            "actor_summaries": actor_summaries,
            "availability": {},
            "comparisons": {
                "best_actor": "flat",
                "best_nonflat_actor": "flat",
                "flat_cost_adjusted_net_pnl_usd": 0.0,
                "best_nonflat_cost_adjusted_net_pnl_usd": 0.0,
                "best_minus_flat_cost_adjusted_net_pnl_usd": 0.0,
                "best_minus_flat_cost_adjusted_net_pnl_bps": 0.0,
            },
            "by_symbol_best_actor": {},
            "by_regime_best_actor": {},
            "replay": {"replay_digest": _sha256({"empty_benchmark_replay": True}), "manifest": {"empty": True}, "timestamp_count": 0, "event_count": 0, "actors": {}},
        }
    actors = {
        name: _benchmark_actor_config(name)
        for name in actor_names
    }
    replay = run_portfolio_replay(
        frame,
        actors,
        initial_equity=1_000.0,
        scenario=ReplayScenario(name="benchmark_replay"),
        dataset_manifest={
            "policy_version": POLICY_VERSION,
            "frame_digest": _frame_digest(frame),
        },
        cost_policy=ExecutionCostPolicy(),
        signal_resolver=_benchmark_signal_resolver,
        throttle_allocation_logs=True,
        research_mode=True,
    )
    close_lookup = _close_lookup_by_symbol(frame)
    actor_summaries = {
        name: _summarize_replay_actor(result, frame, close_lookup=close_lookup)
        for name, result in replay.actors.items()
    }
    nonflat = {name: summary for name, summary in actor_summaries.items() if name != "flat"}
    best_nonflat_name = max(nonflat, key=lambda name: float(nonflat[name]["cost_adjusted_net_pnl_usd"]))
    best_name = max(actor_summaries, key=lambda name: float(actor_summaries[name]["cost_adjusted_net_pnl_usd"]))
    flat_value = float(actor_summaries["flat"]["cost_adjusted_net_pnl_usd"])
    best_nonflat = actor_summaries[best_nonflat_name]
    by_symbol_best_actor: dict[str, Any] = {}
    by_regime_best_actor: dict[str, Any] = {}
    if include_cross_sections:
        for symbol in sorted(str(symbol) for symbol in frame.index.get_level_values("symbol").unique()):
            symbol_frame = frame.xs(symbol, level="symbol", drop_level=False)
            if symbol_frame.empty:
                continue
            symbol_report = _benchmark_replay_report(
                symbol_frame,
                max_bars_per_symbol=max_bars_per_symbol,
                include_cross_sections=False,
            )
            by_symbol_best_actor[symbol] = {
                "available": True,
                "rows": int(len(symbol_frame)),
                "best_actor": symbol_report["comparisons"]["best_actor"],
                "best_minus_flat_cost_adjusted_net_pnl_usd": float(
                    symbol_report["comparisons"]["best_minus_flat_cost_adjusted_net_pnl_usd"]
                ),
            }
        by_regime_best_actor = _benchmark_regime_comparison(frame)
    return {
        "policy_version": BENCHMARK_POLICY_VERSION,
        "generated_at": _utc_now(),
        "dataset": {
            "rows": int(len(frame)),
            "source_rows": int(len(source_frame)),
            "max_bars_per_symbol": int(max_bars_per_symbol),
            "truncated_for_replay": bool(len(frame) < len(source_frame)),
            "symbols": sorted(str(symbol) for symbol in frame.index.get_level_values("symbol").unique()),
            "start": str(pd.DatetimeIndex(frame.index.get_level_values("timestamp")).min()),
            "end": str(pd.DatetimeIndex(frame.index.get_level_values("timestamp")).max()),
        },
        "replay_digest": replay.replay_digest,
        "replay_artifact_policy": {
            "full_replay_embedded": False,
            "max_report_fills_per_actor": int(DEFAULT_REPORT_FILL_LIMIT),
            "reason": "compact reports keep real-data recovery runs bounded while preserving replay digests and actor metrics",
        },
        "actor_summaries": actor_summaries,
        "availability": _benchmark_actor_availability(frame),
        "comparisons": {
            "best_actor": best_name,
            "best_nonflat_actor": best_nonflat_name,
            "flat_cost_adjusted_net_pnl_usd": flat_value,
            "best_nonflat_cost_adjusted_net_pnl_usd": float(best_nonflat["cost_adjusted_net_pnl_usd"]),
            "best_minus_flat_cost_adjusted_net_pnl_usd": float(actor_summaries[best_name]["cost_adjusted_net_pnl_usd"] - flat_value),
            "best_minus_flat_cost_adjusted_net_pnl_bps": float((actor_summaries[best_name]["cost_adjusted_net_pnl_usd"] - flat_value) / 1_000.0 * 10_000.0),
        },
        "by_symbol_best_actor": by_symbol_best_actor,
        "by_regime_best_actor": by_regime_best_actor,
        "replay": _replay_manifest(replay),
    }


def _benchmark_replay_report(
    frame: pd.DataFrame,
    *,
    max_bars_per_symbol: int = DEFAULT_REPLAY_MAX_BARS_PER_SYMBOL,
    include_cross_sections: bool = True,
) -> dict[str, Any]:
    return _benchmark_replay_report_core(
        frame,
        max_bars_per_symbol=max_bars_per_symbol,
        include_cross_sections=include_cross_sections,
    )


def _candidate_side_label(config: RecoveryCandidateConfig, selection_risk_summary: dict[str, Any]) -> str:
    if config.label_mode == "trade_outcome":
        return _normalize_trade_outcome_side(config.trade_outcome_side)
    benchmark_like = str(selection_risk_summary.get("benchmark_like_behavior") or "").strip().lower()
    if benchmark_like == "short_only":
        return "short"
    if benchmark_like == "long_only":
        return "long"
    predicted_buy_count = int(selection_risk_summary.get("predicted_buy_count", 0) or 0)
    predicted_sell_count = int(selection_risk_summary.get("predicted_sell_count", 0) or 0)
    return "short" if predicted_sell_count > predicted_buy_count else "long"


def _same_side_actor_name(candidate_side: str) -> str:
    side = _normalize_trade_outcome_side(candidate_side)
    return "short_only" if side == "short" else "long_only"


def _slice_frame_by_symbols(frame: pd.DataFrame, symbols: Sequence[str]) -> pd.DataFrame:
    wanted = {str(symbol).upper() for symbol in symbols}
    if not wanted:
        return frame.iloc[0:0].copy()
    symbol_index = frame.index.get_level_values("symbol").astype(str).str.upper()
    return frame.loc[symbol_index.isin(wanted)].copy()


def _safe_series_max(series: pd.Series) -> float:
    if series.empty:
        return 0.0
    return float(pd.to_numeric(series, errors="coerce").max())


def _build_quality_regime_context(frame: pd.DataFrame) -> pd.DataFrame:
    context = _build_timestamp_context(frame)
    out = context.copy()
    out["trend_bucket"] = "unknown"
    out["volatility_bucket"] = "unknown"
    out["drawdown_24h_bucket"] = "unknown"
    out["drawdown_7d_bucket"] = "unknown"
    out["funding_bucket"] = "unknown"

    trend = pd.to_numeric(out.get("regime_trend_24h"), errors="coerce")
    out.loc[trend >= 0.002, "trend_bucket"] = "bullish"
    out.loc[trend <= -0.002, "trend_bucket"] = "bearish"
    out.loc[trend.between(-0.002, 0.002, inclusive="both"), "trend_bucket"] = "sideways"

    vol = pd.to_numeric(out.get("regime_volatility_24h"), errors="coerce")
    vol_low = vol.rolling(96, min_periods=10).quantile(0.20)
    vol_high = vol.rolling(96, min_periods=10).quantile(0.80)
    out.loc[vol.notna() & vol_low.notna() & (vol <= vol_low), "volatility_bucket"] = "low"
    out.loc[vol.notna() & vol_high.notna() & (vol >= vol_high), "volatility_bucket"] = "high"
    mid_mask = vol.notna() & vol_low.notna() & vol_high.notna() & (vol > vol_low) & (vol < vol_high)
    out.loc[mid_mask, "volatility_bucket"] = "normal"

    dd24 = pd.to_numeric(out.get("btc_drawdown_24h"), errors="coerce")
    out.loc[dd24 >= -0.01, "drawdown_24h_bucket"] = "none"
    out.loc[(dd24 < -0.01) & (dd24 >= -0.03), "drawdown_24h_bucket"] = "mild"
    out.loc[(dd24 < -0.03) & (dd24 >= -0.07), "drawdown_24h_bucket"] = "moderate"
    out.loc[dd24 < -0.07, "drawdown_24h_bucket"] = "severe"

    btc_close: pd.Series | None = None
    if "BTCUSDT" in frame.index.get_level_values("symbol"):
        btc_frame = frame.xs("BTCUSDT", level="symbol").sort_index()
        btc_close = pd.to_numeric(btc_frame["close"], errors="coerce")
    if btc_close is not None and not btc_close.empty:
        rolling_max_7d = btc_close.rolling(168, min_periods=24).max()
        dd7 = btc_close / rolling_max_7d - 1.0
        dd7 = dd7.reindex(out.index, method="ffill")
        out.loc[dd7 >= -0.01, "drawdown_7d_bucket"] = "none"
        out.loc[(dd7 < -0.01) & (dd7 >= -0.03), "drawdown_7d_bucket"] = "mild"
        out.loc[(dd7 < -0.03) & (dd7 >= -0.07), "drawdown_7d_bucket"] = "moderate"
        out.loc[dd7 < -0.07, "drawdown_7d_bucket"] = "severe"

    funding = pd.to_numeric(out.get("funding_rate_raw"), errors="coerce")
    out.loc[funding >= 0.00005, "funding_bucket"] = "positive"
    out.loc[funding <= -0.00005, "funding_bucket"] = "negative"
    out.loc[funding.between(-0.00005, 0.00005, inclusive="both"), "funding_bucket"] = "neutral"
    return out


def _compare_candidate_slice(
    frame: pd.DataFrame,
    *,
    model: TrainedModel,
    feature_columns: list[str],
    threshold: float,
    horizon: int,
    label_mode: str,
    trade_outcome_side: str,
    feature_frame: pd.DataFrame,
    signal_resolver: Any | None,
    same_side_actor: str,
    max_bars_per_symbol: int = DEFAULT_REPLAY_MAX_BARS_PER_SYMBOL,
) -> dict[str, Any]:
    if frame.empty:
        return {
            "available": False,
            "rows": 0,
            "symbols": [],
            "candidate_pnl_usd": 0.0,
            "flat_pnl_usd": 0.0,
            "best_nonflat_actor": None,
            "best_nonflat_pnl_usd": 0.0,
            "same_side_actor": same_side_actor,
            "same_side_pnl_usd": 0.0,
            "candidate_minus_flat_pnl_usd": 0.0,
            "candidate_minus_best_nonflat_pnl_usd": 0.0,
            "candidate_minus_same_side_pnl_usd": 0.0,
            "candidate_minus_best_nonflat_bps": 0.0,
            "candidate_minus_same_side_bps": 0.0,
            "take_count": 0,
            "fill_count": 0,
            "max_drawdown_bps": 0.0,
            "decision": "reject",
            "reason": "empty_slice",
        }

    candidate_replay = _candidate_replay_report(
        frame,
        model=model,
        feature_columns=feature_columns,
        threshold=threshold,
        horizon=horizon,
        replay_feature_frame=feature_frame,
        signal_resolver=signal_resolver,
        label_mode=label_mode,
        trade_outcome_side=trade_outcome_side,
        max_bars_per_symbol=max_bars_per_symbol,
    )
    benchmark_replay = _benchmark_replay_report(frame)
    candidate = candidate_replay["actor_summaries"]["candidate"]
    flat = benchmark_replay["actor_summaries"]["flat"]
    nonflat = {name: summary for name, summary in benchmark_replay["actor_summaries"].items() if name != "flat"}
    best_nonflat_actor = None
    best_nonflat_summary: dict[str, Any] | None = None
    if nonflat:
        best_nonflat_actor = max(nonflat, key=lambda name: float(nonflat[name]["cost_adjusted_net_pnl_usd"]))
        best_nonflat_summary = nonflat[best_nonflat_actor]
    same_side_summary = benchmark_replay["actor_summaries"].get(same_side_actor) or flat
    candidate_pnl = float(candidate["cost_adjusted_net_pnl_usd"])
    flat_pnl = float(flat["cost_adjusted_net_pnl_usd"])
    best_nonflat_pnl = float(best_nonflat_summary["cost_adjusted_net_pnl_usd"]) if best_nonflat_summary is not None else flat_pnl
    same_side_pnl = float(same_side_summary["cost_adjusted_net_pnl_usd"])
    candidate_minus_flat = candidate_pnl - flat_pnl
    candidate_minus_best_nonflat = candidate_pnl - best_nonflat_pnl
    candidate_minus_same_side = candidate_pnl - same_side_pnl
    candidate_minus_best_nonflat_bps = candidate_minus_best_nonflat / 1_000.0 * 10_000.0
    candidate_minus_same_side_bps = candidate_minus_same_side / 1_000.0 * 10_000.0
    take_count = int(candidate.get("fill_count", 0) or 0)
    max_drawdown_bps = float(candidate.get("max_drawdown_frac", 0.0)) * 10_000.0
    passed = candidate_minus_flat > 0.0 and candidate_minus_best_nonflat > 0.0 and candidate_minus_same_side > 0.0
    passed = passed and candidate_minus_best_nonflat_bps >= DEFAULT_MIN_BENCHMARK_MARGIN_BPS
    passed = passed and candidate_minus_same_side_bps >= DEFAULT_MIN_BENCHMARK_MARGIN_BPS
    passed = passed and take_count >= DEFAULT_MIN_SYMBOL_TAKE_COUNT
    passed = passed and max_drawdown_bps <= DEFAULT_MAX_SYMBOL_DRAWDOWN_BPS
    reason_parts = []
    if candidate_minus_flat <= 0.0:
        reason_parts.append("does_not_beat_flat")
    if candidate_minus_best_nonflat <= 0.0:
        reason_parts.append("does_not_beat_best_nonflat")
    if candidate_minus_same_side <= 0.0:
        reason_parts.append("does_not_beat_same_side")
    if candidate_minus_best_nonflat_bps < DEFAULT_MIN_BENCHMARK_MARGIN_BPS:
        reason_parts.append("insufficient_best_nonflat_margin")
    if candidate_minus_same_side_bps < DEFAULT_MIN_BENCHMARK_MARGIN_BPS:
        reason_parts.append("insufficient_same_side_margin")
    if take_count < DEFAULT_MIN_SYMBOL_TAKE_COUNT:
        reason_parts.append("insufficient_take_count")
    if max_drawdown_bps > DEFAULT_MAX_SYMBOL_DRAWDOWN_BPS:
        reason_parts.append("drawdown_too_high")
    return {
        "available": True,
        "rows": int(len(frame)),
        "symbols": sorted(str(symbol) for symbol in frame.index.get_level_values("symbol").unique()),
        "candidate_pnl_usd": candidate_pnl,
        "flat_pnl_usd": flat_pnl,
        "best_nonflat_actor": best_nonflat_actor,
        "best_nonflat_pnl_usd": best_nonflat_pnl,
        "same_side_actor": same_side_actor,
        "same_side_pnl_usd": same_side_pnl,
        "candidate_minus_flat_pnl_usd": candidate_minus_flat,
        "candidate_minus_best_nonflat_pnl_usd": candidate_minus_best_nonflat,
        "candidate_minus_same_side_pnl_usd": candidate_minus_same_side,
        "candidate_minus_best_nonflat_bps": candidate_minus_best_nonflat_bps,
        "candidate_minus_same_side_bps": candidate_minus_same_side_bps,
        "take_count": take_count,
        "fill_count": int(candidate.get("fill_count", 0) or 0),
        "max_drawdown_bps": max_drawdown_bps,
        "decision": "allow" if passed else "reject",
        "reason": "ok" if passed else ",".join(reason_parts) or "unknown_error",
    }


def _build_regime_section_report(
    frame: pd.DataFrame,
    *,
    model: TrainedModel,
    feature_columns: list[str],
    threshold: float,
    horizon: int,
    label_mode: str,
    trade_outcome_side: str,
    feature_frame: pd.DataFrame,
    signal_resolver: Any | None,
    same_side_actor: str,
) -> dict[str, Any]:
    if frame.empty:
        return {
            "available": False,
            "by_dimension": {},
            "coverage": {"available_bucket_count": 0, "unknown_bucket_count": 0, "total_bucket_count": 0},
        }
    context = _build_quality_regime_context(frame)
    column_by_dimension = {
        "trend": "trend_bucket",
        "volatility": "volatility_bucket",
        "btc_drawdown_24h": "drawdown_24h_bucket",
        "btc_drawdown_7d": "drawdown_7d_bucket",
        "funding": "funding_bucket",
    }
    dimensions = {
        "trend": ["bullish", "bearish", "sideways", "unknown"],
        "volatility": ["low", "normal", "high", "unknown"],
        "btc_drawdown_24h": ["none", "mild", "moderate", "severe", "unknown"],
        "btc_drawdown_7d": ["none", "mild", "moderate", "severe", "unknown"],
        "funding": ["positive", "neutral", "negative", "unknown"],
    }
    by_dimension: dict[str, Any] = {}
    available_bucket_count = 0
    unknown_bucket_count = 0
    total_bucket_count = 0
    for dimension, buckets in dimensions.items():
        dim_entries: dict[str, Any] = {}
        column_name = column_by_dimension[dimension]
        for bucket in buckets:
            total_bucket_count += 1
            if bucket == "unknown":
                unknown_bucket_count += 1
            timestamps = context.index[context[column_name] == bucket]
            slice_frame = _slice_frame_by_timestamps(frame, timestamps)
            entry = _compare_candidate_slice(
                slice_frame,
                model=model,
                feature_columns=feature_columns,
                threshold=threshold,
                horizon=horizon,
                label_mode=label_mode,
                trade_outcome_side=trade_outcome_side,
                feature_frame=feature_frame,
                signal_resolver=signal_resolver,
                same_side_actor=same_side_actor,
            )
            entry["bucket"] = bucket
            entry["dimension"] = dimension
            dim_entries[bucket] = entry
            if entry.get("available"):
                available_bucket_count += 1
        by_dimension[dimension] = dim_entries
    return {
        "available": True,
        "by_dimension": by_dimension,
        "coverage": {
            "available_bucket_count": int(available_bucket_count),
            "unknown_bucket_count": int(unknown_bucket_count),
            "total_bucket_count": int(total_bucket_count),
        },
    }


def _build_symbol_pruning_report(
    frame: pd.DataFrame,
    *,
    model: TrainedModel,
    feature_columns: list[str],
    threshold: float,
    horizon: int,
    label_mode: str,
    trade_outcome_side: str,
    feature_frame: pd.DataFrame,
    signal_resolver: Any | None,
    same_side_actor: str,
) -> dict[str, Any]:
    by_symbol: dict[str, Any] = {}
    allowed_symbols: list[str] = []
    rejected_symbols: list[dict[str, Any]] = []
    for symbol in sorted(str(symbol) for symbol in frame.index.get_level_values("symbol").unique()):
        symbol_frame = frame.xs(symbol, level="symbol", drop_level=False)
        entry = _compare_candidate_slice(
            symbol_frame,
            model=model,
            feature_columns=feature_columns,
            threshold=threshold,
            horizon=horizon,
            label_mode=label_mode,
            trade_outcome_side=trade_outcome_side,
            feature_frame=feature_frame,
            signal_resolver=signal_resolver,
            same_side_actor=same_side_actor,
        )
        by_symbol[symbol] = entry
        if entry.get("decision") == "allow":
            allowed_symbols.append(symbol)
        else:
            rejected_symbols.append({"symbol": symbol, "reason": entry.get("reason", "unknown_error")})
    return {
        "available": bool(by_symbol),
        "by_symbol": by_symbol,
        "allowed_symbols": allowed_symbols,
        "allowed_symbol_count": int(len(allowed_symbols)),
        "rejected_symbols": rejected_symbols,
        "rejected_symbol_count": int(len(rejected_symbols)),
    }


def _build_replay_gap_diagnostics(
    holdout_report: dict[str, Any],
    replay_report: dict[str, Any],
    *,
    benchmark_delta_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    replay_candidate = replay_report.get("actor_summaries", {}).get("candidate", {}) if isinstance(replay_report, dict) else {}
    row_level_expectancy_bps = float(holdout_report.get("cost_adjusted_expectancy_bps", 0.0))
    replay_net_return_bps = float(replay_candidate.get("cost_adjusted_net_pnl_usd", 0.0)) / 1_000.0 * 10_000.0
    gap_bps = replay_net_return_bps - row_level_expectancy_bps
    return {
        "policy_version": "replay_gap_v1",
        "row_level_expectancy_bps": row_level_expectancy_bps,
        "replay_net_return_bps": replay_net_return_bps,
        "gap_bps": gap_bps,
        "gap_tolerance_bps": DEFAULT_MAX_REPLAY_GAP_BPS,
        "cost_drag_bps": abs(gap_bps),
        "hold_duration_summary": {
            "predicted_take_count": int(holdout_report.get("predicted_take_count", 0)),
            "predicted_skip_count": int(holdout_report.get("predicted_skip_count", 0)),
            "predicted_hold_count": int(holdout_report.get("predicted_hold_count", 0)),
        },
        "adverse_excursion_summary": {
            "max_drawdown_frac": float(replay_candidate.get("max_drawdown_frac", 0.0)),
        },
        "symbol_concentration": dict(replay_candidate.get("exposure_by_symbol_usd", {}) or {}),
        "top_gap_reasons": (
            ["row_level_expectancy_disagrees_with_replay"]
            if abs(gap_bps) > DEFAULT_MAX_REPLAY_GAP_BPS
            else ["gap_within_tolerance"]
        ),
        "benchmark_delta_hint": {
            "candidate_minus_best_nonflat_bps": float((benchmark_delta_report or {}).get("overall", {}).get("candidate_minus_best_nonflat_bps", 0.0)),
            "candidate_minus_same_side_bps": float((benchmark_delta_report or {}).get("overall", {}).get("candidate_minus_same_side_bps", 0.0)),
        },
    }


def _candidate_quality_report_from_decision(decision: CandidateQualityDecision) -> dict[str, Any]:
    return {
        "candidate_id": decision.candidate_id,
        "overall_decision": decision.overall_decision,
        "passed": bool(decision.passed),
        "evidence_digest": decision.evidence_digest,
        "summary": dict(decision.summary),
        "rule_results": [asdict(result) for result in decision.rule_results],
    }


def _build_candidate_quality_summary(results: Sequence[CandidateQualityDecision | dict[str, Any]]) -> dict[str, Any]:
    failure_reasons: Counter[str] = Counter()
    for decision in results:
        if isinstance(decision, CandidateQualityDecision):
            rule_results = decision.rule_results
            for result in rule_results:
                if not result.passed:
                    failure_reasons[result.reason] += 1
            continue
        if isinstance(decision, dict):
            for result in decision.get("rule_results", []) or []:
                if isinstance(result, dict) and not bool(result.get("passed", False)):
                    failure_reasons[str(result.get("reason", "unknown"))] += 1
            continue
    return {
        "evaluated_candidates": int(len(results)),
        "passed_quality": int(sum(1 for decision in results if (decision.passed if isinstance(decision, CandidateQualityDecision) else bool(decision.get("passed", False))))),
        "watch_quality": int(sum(1 for decision in results if (decision.overall_decision if isinstance(decision, CandidateQualityDecision) else str(decision.get("overall_decision", "")).lower()) == "watch")),
        "failed_quality": int(sum(1 for decision in results if (decision.overall_decision if isinstance(decision, CandidateQualityDecision) else str(decision.get("overall_decision", "")).lower()) == "fail")),
        "top_failure_reasons": [
            {"reason": reason, "count": count}
            for reason, count in failure_reasons.most_common(10)
        ],
    }


def _neutral_maintenance_report(
    *,
    candidate_id: str,
    recent_net_return_bps: float,
    recent_drawdown_bps: float,
    recent_benchmark_delta_bps: float,
    recent_actionable_decisions: int,
) -> dict[str, Any]:
    metrics = {
        "recent_net_return_bps": float(recent_net_return_bps),
        "recent_drawdown_bps": float(recent_drawdown_bps),
        "recent_benchmark_delta_bps": float(recent_benchmark_delta_bps),
        "shadow_drift_mae": 0.0,
        "recent_actionable_decisions": int(recent_actionable_decisions),
        "candidate_count": 0,
        "hard_risk_pauses": 0,
        "approval_required": True,
    }
    return {
        "active_version_id": candidate_id,
        "decayed": False,
        "no_trade_required": False,
        "proven_shadow_version_id": None,
        "recommended_action": "continue",
        "blockers": [],
        "evidence_digest": _sha256({"candidate_id": candidate_id, "metrics": metrics, "decayed": False}),
        "metrics": metrics,
    }


def _empty_candidate_replay_report(frame: pd.DataFrame) -> dict[str, Any]:
    symbols = sorted(str(symbol) for symbol in frame.index.get_level_values("symbol").unique()) if not frame.empty else []
    return {
        "policy_version": POLICY_VERSION,
        "generated_at": _utc_now(),
        "dataset": {
            "rows": int(len(frame)),
            "source_rows": int(len(frame)),
            "max_bars_per_symbol": int(DEFAULT_REPLAY_MAX_BARS_PER_SYMBOL),
            "truncated_for_replay": False,
            "symbols": symbols,
            "start": None if frame.empty else str(pd.DatetimeIndex(frame.index.get_level_values("timestamp")).min()),
            "end": None if frame.empty else str(pd.DatetimeIndex(frame.index.get_level_values("timestamp")).max()),
        },
        "replay_digest": _sha256({"policy_version": POLICY_VERSION, "dataset_rows": int(len(frame)), "empty": True}),
        "replay_artifact_policy": {
            "full_replay_embedded": False,
            "max_report_fills_per_actor": int(DEFAULT_REPORT_FILL_LIMIT),
            "reason": "empty_final_frame_after_symbol_pruning",
        },
        "actor_summaries": {
            "candidate": {
                "cost_adjusted_net_pnl_usd": 0.0,
                "max_drawdown_frac": 0.0,
                "fill_count": 0,
                "exposure_by_symbol_usd": {},
            },
            "flat": {
                "cost_adjusted_net_pnl_usd": 0.0,
                "max_drawdown_frac": 0.0,
                "fill_count": 0,
                "exposure_by_symbol_usd": {},
            },
        },
        "comparisons": {
            "candidate_minus_flat_cost_adjusted_net_pnl_usd": 0.0,
            "candidate_minus_flat_cost_adjusted_net_pnl_bps": 0.0,
            "best_actor": "flat",
        },
        "replay": {
            "policy_version": "replay_empty_v1",
            "actors": {},
        },
    }


def _empty_benchmark_replay_report(frame: pd.DataFrame) -> dict[str, Any]:
    symbols = sorted(str(symbol) for symbol in frame.index.get_level_values("symbol").unique()) if not frame.empty else []
    return {
        "policy_version": BENCHMARK_POLICY_VERSION,
        "generated_at": _utc_now(),
        "dataset": {
            "rows": int(len(frame)),
            "source_rows": int(len(frame)),
            "max_bars_per_symbol": int(DEFAULT_REPLAY_MAX_BARS_PER_SYMBOL),
            "truncated_for_replay": False,
            "symbols": symbols,
            "start": None if frame.empty else str(pd.DatetimeIndex(frame.index.get_level_values("timestamp")).min()),
            "end": None if frame.empty else str(pd.DatetimeIndex(frame.index.get_level_values("timestamp")).max()),
        },
        "replay_digest": _sha256({"policy_version": BENCHMARK_POLICY_VERSION, "dataset_rows": int(len(frame)), "empty": True}),
        "replay_artifact_policy": {
            "full_replay_embedded": False,
            "max_report_fills_per_actor": int(DEFAULT_REPORT_FILL_LIMIT),
            "reason": "empty_final_frame_after_symbol_pruning",
        },
        "actor_summaries": {
            "flat": {
                "cost_adjusted_net_pnl_usd": 0.0,
                "max_drawdown_frac": 0.0,
                "fill_count": 0,
                "exposure_by_symbol_usd": {},
            }
        },
        "availability": {"flat": {"available": True, "reason": "baseline"}},
        "comparisons": {
            "best_actor": "flat",
            "best_nonflat_actor": None,
            "flat_cost_adjusted_net_pnl_usd": 0.0,
            "best_nonflat_cost_adjusted_net_pnl_usd": 0.0,
            "best_minus_flat_cost_adjusted_net_pnl_usd": 0.0,
            "best_minus_flat_cost_adjusted_net_pnl_bps": 0.0,
        },
        "by_symbol_best_actor": {},
        "by_regime_best_actor": {},
        "replay": {
            "policy_version": "benchmark_empty_v1",
            "actors": {},
        },
    }


def _build_benchmark_delta_report(
    frame: pd.DataFrame,
    *,
    model: TrainedModel,
    feature_columns: list[str],
    threshold: float,
    horizon: int,
    label_mode: str,
    trade_outcome_side: str,
    feature_frame: pd.DataFrame,
    signal_resolver: Any | None,
    symbol_pruning_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    same_side_actor = _same_side_actor_name(trade_outcome_side)
    comparison = _compare_candidate_slice(
        frame,
        model=model,
        feature_columns=feature_columns,
        threshold=threshold,
        horizon=horizon,
        label_mode=label_mode,
        trade_outcome_side=trade_outcome_side,
        feature_frame=feature_frame,
        signal_resolver=signal_resolver,
        same_side_actor=same_side_actor,
    )
    by_symbol: dict[str, Any] = {}
    if not frame.empty:
        for symbol in sorted(str(symbol) for symbol in frame.index.get_level_values("symbol").unique()):
            symbol_frame = frame.xs(symbol, level="symbol", drop_level=False)
            by_symbol[symbol] = _compare_candidate_slice(
                symbol_frame,
                model=model,
                feature_columns=feature_columns,
                threshold=threshold,
                horizon=horizon,
                label_mode=label_mode,
                trade_outcome_side=trade_outcome_side,
                feature_frame=feature_frame,
                signal_resolver=signal_resolver,
                same_side_actor=same_side_actor,
            )
    by_regime = _build_regime_section_report(
        frame,
        model=model,
        feature_columns=feature_columns,
        threshold=threshold,
        horizon=horizon,
        label_mode=label_mode,
        trade_outcome_side=trade_outcome_side,
        feature_frame=feature_frame,
        signal_resolver=signal_resolver,
        same_side_actor=same_side_actor,
    )
    overall = {
        "candidate_pnl_usd": float(comparison["candidate_pnl_usd"]),
        "candidate_return_bps": float(comparison["candidate_pnl_usd"] / 1_000.0 * 10_000.0),
        "flat_pnl_usd": float(comparison["flat_pnl_usd"]),
        "best_nonflat_actor": comparison["best_nonflat_actor"],
        "best_nonflat_pnl_usd": float(comparison["best_nonflat_pnl_usd"]),
        "same_side_actor": comparison["same_side_actor"],
        "same_side_pnl_usd": float(comparison["same_side_pnl_usd"]),
        "candidate_minus_flat_pnl_usd": float(comparison["candidate_minus_flat_pnl_usd"]),
        "candidate_minus_best_nonflat_pnl_usd": float(comparison["candidate_minus_best_nonflat_pnl_usd"]),
        "candidate_minus_same_side_pnl_usd": float(comparison["candidate_minus_same_side_pnl_usd"]),
        "candidate_minus_best_nonflat_bps": float(comparison["candidate_minus_best_nonflat_bps"]),
        "candidate_minus_same_side_bps": float(comparison["candidate_minus_same_side_bps"]),
        "take_count": int(comparison["take_count"]),
        "fill_count": int(comparison["fill_count"]),
        "max_drawdown_bps": float(comparison["max_drawdown_bps"]),
        "decision": comparison["decision"],
        "reason": comparison["reason"],
    }
    warnings: list[str] = []
    if symbol_pruning_report and not symbol_pruning_report.get("available", False):
        warnings.append("symbol_pruning_unavailable")
    if overall["decision"] == "reject" and overall["reason"]:
        warnings.extend(part for part in str(overall["reason"]).split(",") if part)
    candidate_side = "directional" if label_mode == "directional_return" else _normalize_trade_outcome_side(trade_outcome_side)
    return {
        "policy_version": BENCHMARK_POLICY_VERSION,
        "candidate_id": comparison.get("candidate_id", ""),
        "candidate_side": candidate_side,
        "threshold_source": "selected_threshold_policy",
        "overall": overall,
        "by_symbol": by_symbol,
        "by_regime": by_regime,
        "symbol_pruning": symbol_pruning_report or {},
        "rejected_symbols": list((symbol_pruning_report or {}).get("rejected_symbols", [])),
        "warnings": warnings,
    }


def _phase4_base_config(label_audit_report: dict[str, Any]) -> RecoveryCandidateConfig:
    summary = label_audit_report.get("summary", {}) or {}
    best_cell = summary.get("best_cell_stats", {}) or {}
    recommended_dead_zone_bps = float(summary.get("recommended_dead_zone_floor_bps", 0.001) or 0.001)
    recommended_dead_zone = recommended_dead_zone_bps / 10_000.0
    horizon = int(best_cell.get("horizon", DEFAULT_HORIZONS[0]) or DEFAULT_HORIZONS[0])
    training_window_months = int(best_cell.get("training_window_months", DEFAULT_TRAINING_WINDOWS_MONTHS[1]) or DEFAULT_TRAINING_WINDOWS_MONTHS[1])
    recency_half_life_days = int(best_cell.get("recency_half_life_days", DEFAULT_RECENCY_HALF_LIFES_DAYS[1]) or DEFAULT_RECENCY_HALF_LIFES_DAYS[1])
    dead_zone_bps = float(best_cell.get("dead_zone_bps", recommended_dead_zone) or recommended_dead_zone)
    return RecoveryCandidateConfig(
        horizon=horizon,
        training_window_months=training_window_months,
        recency_half_life_days=recency_half_life_days,
        dead_zone_bps=max(dead_zone_bps, recommended_dead_zone),
        feature_set="price_volume_funding",
    )


def build_phase4_variant_specs(
    label_audit_report: dict[str, Any],
    *,
    label_mode: str = "directional_return",
    trade_outcome_profit_target_bps: float = 20.0,
    trade_outcome_stop_loss_bps: float = 30.0,
    trade_outcome_round_trip_cost_bps: float = 8.0,
) -> list[Phase4VariantSpec]:
    """Return a bounded repair suite that targets the observed failure mode."""

    base = _phase4_base_config(label_audit_report)
    sides = ("long", "short") if label_mode == "trade_outcome" else ("long",)
    variant_templates = (
        ("cost_aware_ternary", "cost_aware_ternary", "price_volume_funding"),
        ("horizon_specific_features", "horizon_specific_features", "no_open_interest"),
        ("symbol_group_calibration", "symbol_group_calibration", "full"),
        ("regime_gated_abstain", "regime_gated_abstain", "no_orderbook_placeholders"),
    )
    specs: list[Phase4VariantSpec] = []
    for variant_id, variant_kind, feature_set in variant_templates:
        for side in sides:
            side_suffix = f"_{side}" if label_mode == "trade_outcome" else ""
            specs.append(
                Phase4VariantSpec(
                    variant_id=f"{variant_id}{side_suffix}",
                    variant_kind=variant_kind,
                    config=RecoveryCandidateConfig(
                        horizon=base.horizon,
                        training_window_months=base.training_window_months,
                        recency_half_life_days=base.recency_half_life_days,
                        dead_zone_bps=base.dead_zone_bps,
                        feature_set=feature_set,
                        label_mode=label_mode,
                        trade_outcome_side=side,
                        trade_outcome_profit_target_bps=trade_outcome_profit_target_bps,
                        trade_outcome_stop_loss_bps=trade_outcome_stop_loss_bps,
                        trade_outcome_round_trip_cost_bps=trade_outcome_round_trip_cost_bps,
                    ),
                )
            )
    return specs


def build_research_input_diagnostics_report(
    frame: pd.DataFrame,
    *,
    label_audit_report: dict[str, Any],
    benchmark_report: dict[str, Any],
    variant_evaluations: list[CandidateEvaluation],
) -> dict[str, Any]:
    feature_frame = _build_feature_frame(frame)
    variant_reports: dict[str, Any] = {}
    failure_tally: Counter[str] = Counter()
    passed_variants = 0

    for variant in variant_evaluations:
        variant_key = variant.variant_id or variant.candidate_id
        failure_tally.update(variant.failure_reasons)
        if variant.passed:
            passed_variants += 1
        feature_manifest = dict(variant.feature_manifest or {})
        feature_columns = list(feature_manifest.get("selected_feature_columns") or [])
        variant_reports[variant_key] = {
            "variant_id": variant.variant_id,
            "candidate_id": variant.candidate_id,
            "variant_kind": variant.selection_risk_summary.get("variant_kind"),
            "passed": bool(variant.passed),
            "score": float(variant.score),
            "failure_reasons": list(variant.failure_reasons),
            "prediction_audit": dict(variant.holdout_report.get("prediction_audit", {}) or {}),
            "feature_importance": dict(variant.holdout_report.get("feature_importance", {}) or {}),
            "feature_health": _feature_health_report(feature_frame, feature_columns),
            "holdout_accuracy": float(variant.holdout_report.get("holdout_accuracy", 0.0)),
            "cost_adjusted_expectancy_bps": float(variant.holdout_report.get("cost_adjusted_expectancy_bps", 0.0)),
            "replay_cost_adjusted_net_pnl_usd": float(
                variant.replay_report.get("actor_summaries", {}).get("candidate", {}).get("cost_adjusted_net_pnl_usd", 0.0)
            ),
        }

    failed_variants = [variant for variant in variant_evaluations if not variant.passed]
    best_failed = None
    if failed_variants:
        best_failed = max(
            failed_variants,
            key=lambda candidate: (
                float(candidate.score),
                float(candidate.holdout_report.get("holdout_accuracy", 0.0)),
                float(candidate.replay_report.get("actor_summaries", {}).get("candidate", {}).get("cost_adjusted_net_pnl_usd", 0.0)),
                candidate.candidate_id,
            ),
        )

    regime_comparison = _benchmark_regime_comparison(frame)
    prediction_collapse = {
        variant_key: bool(report.get("prediction_audit", {}).get("one_sided_collapse", False))
        for variant_key, report in variant_reports.items()
    }

    return {
        "policy_version": POLICY_VERSION,
        "generated_at": _utc_now(),
        "dataset": {
            "rows": int(len(frame)),
            "symbols": sorted(str(s) for s in frame.index.get_level_values("symbol").unique()),
            "start": str(pd.DatetimeIndex(frame.index.get_level_values("timestamp")).min()),
            "end": str(pd.DatetimeIndex(frame.index.get_level_values("timestamp")).max()),
            "frame_digest": _frame_digest(frame),
        },
        "label_audit": label_audit_report,
        "benchmark_replay": benchmark_report,
        "variant_reports": variant_reports,
        "summary": {
            "variant_count": int(len(variant_evaluations)),
            "passed_variant_count": int(passed_variants),
            "failed_variant_count": int(len(failed_variants)),
            "best_failed_variant_id": best_failed.variant_id if best_failed is not None else None,
            "best_failed_candidate_id": best_failed.candidate_id if best_failed is not None else None,
            "best_failed_score": float(best_failed.score) if best_failed is not None else None,
            "prediction_collapse_observed": bool(any(prediction_collapse.values())),
            "failure_reason_counts": dict(failure_tally),
            "recommended_next_step": "paper_quarantine_candidate" if passed_variants else "remain_no_trade",
        },
        "regime_comparison": regime_comparison,
        "diagnostic_notes": [
            "Use the best failed variant as the next repair target.",
            "Do not resume trading until a variant passes offline gates and paper soak.",
        ],
    }


def _candidate_grid(
    label_audit_report: dict[str, Any],
    *,
    label_mode: str = "directional_return",
    trade_outcome_profit_target_bps: float = 20.0,
    trade_outcome_stop_loss_bps: float = 30.0,
    trade_outcome_round_trip_cost_bps: float = 8.0,
) -> list[RecoveryCandidateConfig]:
    cells = label_audit_report.get("cells", {}) or {}
    configs: list[RecoveryCandidateConfig] = []
    for horizon in DEFAULT_HORIZONS:
        for training_window_months in DEFAULT_TRAINING_WINDOWS_MONTHS:
            for recency_half_life_days in DEFAULT_RECENCY_HALF_LIFES_DAYS:
                for dead_zone in DEFAULT_DEAD_ZONES:
                    for feature_set in DEFAULT_FEATURE_SETS:
                        configs.append(
                            RecoveryCandidateConfig(
                                horizon=int(horizon),
                                training_window_months=int(training_window_months),
                                recency_half_life_days=int(recency_half_life_days),
                                dead_zone_bps=float(dead_zone),
                                feature_set=str(feature_set),
                                label_mode=label_mode,
                                trade_outcome_side="long",
                                trade_outcome_profit_target_bps=float(trade_outcome_profit_target_bps),
                                trade_outcome_stop_loss_bps=float(trade_outcome_stop_loss_bps),
                                trade_outcome_round_trip_cost_bps=float(trade_outcome_round_trip_cost_bps),
                            )
                        )

    def _score(config: RecoveryCandidateConfig) -> tuple[float, float, int, int, int, int, int, int, str]:
        cell_key = (
            f"h{config.horizon}_tw{config.training_window_months}m_"
            f"hl{config.recency_half_life_days}d_dz{str(round(float(config.dead_zone_bps), 4)).replace('.', 'p')}"
        )
        cell = cells.get(cell_key, {})
        side_priority = 0 if config.trade_outcome_side == "long" else 1
        return (
            float(cell.get("label_balance_score", 0.0)),
            -float(cell.get("ambiguous_rate", 1.0)),
            int(cell.get("sample_count", 0)),
            -int(config.recency_half_life_days),
            -int(config.training_window_months),
            -int(config.horizon),
            -_FEATURE_SET_PRIORITY.get(config.feature_set, 999),
            -side_priority,
            config.candidate_id(),
        )

    sorted_configs = sorted(configs, key=_score, reverse=True)
    if label_mode != "trade_outcome":
        return sorted_configs

    expanded: list[RecoveryCandidateConfig] = []
    for config in sorted_configs:
        expanded.append(replace(config, trade_outcome_side="long"))
        expanded.append(replace(config, trade_outcome_side="short"))
    return expanded


def _build_candidate_labels(frame: pd.DataFrame, config: RecoveryCandidateConfig) -> tuple[pd.Series, dict[str, Any]]:
    if config.label_mode == "directional_return":
        labels = _build_labels(frame, config.horizon, config.dead_zone_bps)
        report = _label_stats(frame, labels, horizon=config.horizon, dead_zone=config.dead_zone_bps)
        report.update(
            {
                "label_mode": "directional_return",
                "policy_version": LABEL_POLICY_VERSION,
            }
        )
        return labels, report

    if config.label_mode == "trade_outcome":
        side = _normalize_trade_outcome_side(config.trade_outcome_side)
        trade_cfg = _trade_outcome_label_config(
            horizon=config.horizon,
            dead_zone_fraction=config.dead_zone_bps,
            profit_target_bps=config.trade_outcome_profit_target_bps,
            stop_loss_bps=config.trade_outcome_stop_loss_bps,
            round_trip_cost_bps=config.trade_outcome_round_trip_cost_bps,
        )
        labels = build_trade_outcome_labels(frame, config=trade_cfg, side=side)
        report = build_trade_outcome_report(frame, config=trade_cfg)
        selected_report = dict(report.get(side, {}))
        selected_report.update(
            {
                "label_mode": "trade_outcome",
                "trade_outcome_side": side,
                "policy_version": report.get("policy_version"),
                "label_counts": dict(selected_report.get("label_counts", {})),
                "barrier_counts": dict(selected_report.get("barrier_counts", {})),
                "trade_outcome_report": report,
            }
        )
        return labels, selected_report

    raise ValueError(f"unsupported label_mode={config.label_mode!r}")


def _limit_candidates(configs: list[RecoveryCandidateConfig], max_candidates: int) -> list[RecoveryCandidateConfig]:
    if max_candidates <= 0:
        raise ValueError("max_candidates must be positive")
    if len(configs) <= max_candidates:
        return list(configs)
    selected: list[RecoveryCandidateConfig] = []
    seen: set[str] = set()
    for horizon in DEFAULT_HORIZONS:
        for config in configs:
            if config.horizon == horizon:
                selected.append(config)
                seen.add(config.candidate_id())
                break
        if len(selected) >= max_candidates:
            return selected[:max_candidates]
    for config in configs:
        if config.candidate_id() in seen:
            continue
        selected.append(config)
        seen.add(config.candidate_id())
        if len(selected) >= max_candidates:
            break
    return selected


def _select_threshold_from_oof_predictions(predictions: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    cfg_min = 0.50
    cfg_max = 0.80
    cfg_step = 0.05
    thresholds = np.round(np.arange(cfg_min, cfg_max + cfg_step / 2.0, cfg_step), 2)
    probs = np.asarray(predictions, dtype=float)
    truth = np.asarray(labels, dtype=int)
    if probs.size == 0 or truth.size == 0:
        return {
            "source": "insufficient_oof_predictions",
            "selected_threshold": 0.5,
            "selected_accuracy": 0.0,
            "threshold_min": cfg_min,
            "threshold_max": cfg_max,
            "threshold_step": cfg_step,
            "samples": int(len(truth)),
        }
    if len(np.unique(truth)) < 2:
        return {
            "source": "class_collapse",
            "selected_threshold": 0.5,
            "selected_accuracy": float((truth == truth[0]).mean()) if truth.size else 0.0,
            "threshold_min": cfg_min,
            "threshold_max": cfg_max,
            "threshold_step": cfg_step,
            "samples": int(len(truth)),
        }
    best_threshold = 0.5
    best_accuracy = -1.0
    for threshold in thresholds:
        predicted = (probs >= float(threshold)).astype(int)
        accuracy = float((predicted == truth).mean())
        if accuracy > best_accuracy or (accuracy == best_accuracy and float(threshold) < best_threshold):
            best_threshold = float(threshold)
            best_accuracy = accuracy
    return {
        "source": "oof_dev_predictions",
        "selected_threshold": float(best_threshold),
        "selected_accuracy": float(best_accuracy),
        "threshold_min": cfg_min,
        "threshold_max": cfg_max,
        "threshold_step": cfg_step,
        "samples": int(len(truth)),
    }


def _candidate_decision_returns_bps(
    *,
    label_mode: str,
    trade_outcome_side: str,
    predictions: np.ndarray,
    forward_return_bps: np.ndarray,
    cost_bps: float,
) -> np.ndarray:
    preds = np.asarray(predictions, dtype=int).reshape(-1)
    forward = np.asarray(forward_return_bps, dtype=float).reshape(-1)
    if preds.size != forward.size:
        raise ValueError("predictions and forward_return_bps must have the same length")
    side = _normalize_trade_outcome_side(trade_outcome_side)
    if label_mode == "directional_return":
        return np.where(
            preds == 1,
            forward - float(cost_bps),
            -forward - float(cost_bps),
        ).astype(float)
    if label_mode == "trade_outcome":
        if side == "long":
            return np.where(preds == 1, forward - float(cost_bps), 0.0).astype(float)
        return np.where(preds == 1, -forward - float(cost_bps), 0.0).astype(float)
    raise ValueError(f"unsupported label_mode={label_mode!r}")


def _evaluate_candidate(
    frame: pd.DataFrame,
    *,
    config: RecoveryCandidateConfig,
    benchmark_report: dict[str, Any],
    run_dir: Path,
    min_accuracy: float,
    min_actionable_decisions: int,
    max_drawdown_frac: float,
    variant_id: str = "",
    variant_kind: str = "baseline",
) -> CandidateEvaluation:
    artifact_id = config.candidate_id() if not variant_id else f"{config.candidate_id()}__{variant_id}"
    candidate_dir = run_dir / "candidates" / artifact_id
    candidate_dir.mkdir(parents=True, exist_ok=True)
    validation_frame = frame.copy()
    trade_outcome_side = _normalize_trade_outcome_side(config.trade_outcome_side)
    dataset_manifest = build_dataset_manifest(
        validation_frame,
        dataset_name=f"model_recovery_{config.candidate_id()}",
        metadata={
            "candidate_id": config.candidate_id(),
            "feature_set": config.feature_set,
            "horizon": config.horizon,
            "training_window_months": config.training_window_months,
            "training_window_applied_by_temporal_folds": True,
            "recency_half_life_days": config.recency_half_life_days,
            "dead_zone_bps": config.dead_zone_bps,
            "label_mode": config.label_mode,
            "trade_outcome_side": trade_outcome_side,
            "trade_outcome_profit_target_bps": config.trade_outcome_profit_target_bps,
            "trade_outcome_stop_loss_bps": config.trade_outcome_stop_loss_bps,
            "trade_outcome_round_trip_cost_bps": config.trade_outcome_round_trip_cost_bps,
        },
    ).to_dict()

    feature_frame = _build_feature_frame(validation_frame)
    feature_cols = _feature_columns_for_set(config.feature_set, feature_frame.columns)
    selected_frame = feature_frame.loc[:, feature_cols].copy()
    labels, candidate_label_report = _build_candidate_labels(validation_frame, config)
    valid_mask = labels.notna() & selected_frame.notna().all(axis=1)
    selected_frame = selected_frame.loc[valid_mask].copy()
    labels = labels.loc[valid_mask].astype(int)
    feature_manifest = {
        "feature_set": config.feature_set,
        "included_groups": list(_FEATURE_SET_GROUPS[_normalize_feature_set(config.feature_set)]),
        "excluded_groups": sorted(set(_FEATURE_GROUPS) - set(_FEATURE_SET_GROUPS[_normalize_feature_set(config.feature_set)])),
        "selected_feature_columns": list(feature_cols),
        "selected_feature_count": len(feature_cols),
        "available_feature_count": len(get_feature_columns(feature_frame)),
        "feature_catalog_version": feature_frame.attrs.get("feature_catalog_version"),
        "feature_catalog_sha256": feature_frame.attrs.get("feature_catalog_sha256"),
    }

    cell_key = (
        f"h{config.horizon}_tw{config.training_window_months}m_"
        f"hl{config.recency_half_life_days}d_dz{str(round(float(config.dead_zone_bps), 4)).replace('.', 'p')}"
    )
    label_audit = dict(candidate_label_report)
    label_audit.update(
        {
            "training_window_months": config.training_window_months,
            "recency_half_life_days": config.recency_half_life_days,
            "feature_set": config.feature_set,
            "trade_outcome_side": trade_outcome_side,
        }
    )

    candidate_frame = selected_frame.copy()
    candidate_frame.index = candidate_frame.index.set_names(["timestamp", "symbol"])
    plan = build_temporal_validation_plan(
        candidate_frame,
        training_windows_months=(config.training_window_months,),
        expanding_included=True,
        test_window_months=1,
        holdout_months=max(1, min(3, len(pd.PeriodIndex(candidate_frame.index.get_level_values("timestamp"), freq="M").unique()) // 6 or 1)),
        purge_bars=max(config.horizon, 24),
        min_train_rows=max(50, min_actionable_decisions),
    )

    oof_predictions: list[float] = []
    oof_truths: list[int] = []
    oof_forward_returns_bps: list[float] = []
    oof_symbols: list[str] = []
    oof_fold_ids: list[str] = []
    oof_rows: list[dict[str, Any]] = []
    oof_by_symbol: dict[str, list[tuple[float, int]]] = {}
    fold_rows: list[dict[str, Any]] = []
    fold_accuracies: list[float] = []
    selected_threshold_policy: dict[str, Any] = {"selected_threshold": 0.5, "selected_accuracy": 0.0, "source": "insufficient_oof_predictions"}
    selected_validation_folds = _select_validation_folds(
        plan.folds,
        preferred_variant=f"rolling_{config.training_window_months}m",
        max_folds=DEFAULT_MAX_VALIDATION_FOLDS,
    )
    forward_return_by_row = pd.Series(0.0, index=validation_frame.index, dtype=float)
    for _, sym_frame in validation_frame.groupby(level="symbol", sort=False):
        close = pd.to_numeric(sym_frame["close"], errors="coerce")
        forward_return_by_row.loc[sym_frame.index] = ((close.shift(-config.horizon) / close - 1.0) * 10_000.0).fillna(0.0)

    for fold in selected_validation_folds:
        X_train = candidate_frame.iloc[fold.train_indices]
        y_train = labels.iloc[fold.train_indices]
        X_test = candidate_frame.iloc[fold.test_indices]
        y_test = labels.iloc[fold.test_indices]
        weights = compute_recency_weights(
            pd.DatetimeIndex(X_train.index.get_level_values("timestamp")),
            half_life_days=float(config.recency_half_life_days),
        )
        if len(X_train) < 50 or len(X_test) < 20:
            continue
        model = train(
            X_train,
            y_train,
            horizon=config.horizon,
            sample_weight=weights,
            params_override=EXPERIMENT_TRAIN_PARAMS,
        )
        probs = predict_proba(model, X_test)
        preds = (probs >= 0.5).astype(int)
        acc = float((preds == y_test.to_numpy()).mean()) if len(y_test) else 0.0
        test_symbols = X_test.index.get_level_values("symbol").astype(str)
        test_forward_returns = forward_return_by_row.reindex(X_test.index).fillna(0.0).to_numpy(dtype=float)
        for idx, (proba, truth) in enumerate(zip(probs, y_test.to_numpy(), strict=False)):
            symbol = str(test_symbols[idx])
            forward_bps = float(test_forward_returns[idx])
            oof_by_symbol.setdefault(symbol, []).append((float(proba), int(truth)))
            oof_rows.append(
                {
                    "timestamp": str(X_test.index[idx][0]),
                    "symbol": symbol,
                    "probability": float(proba),
                    "truth": int(truth),
                    "forward_return_bps": forward_bps,
                    "fold_id": str(fold.fold_id),
                }
            )
            oof_forward_returns_bps.append(forward_bps)
            oof_symbols.append(symbol)
            oof_fold_ids.append(str(fold.fold_id))
        fold_accuracies.append(acc)
        fold_rows.append(
            {
                "fold_id": fold.fold_id,
                "variant": fold.variant,
                "train_rows": int(len(X_train)),
                "test_rows": int(len(X_test)),
                "train_start": str(fold.train_start),
                "train_end": str(fold.train_end),
                "test_start": str(fold.test_start),
                "test_end": str(fold.test_end),
                "fold_accuracy_at_0_5": acc,
                "effective_sample_size": float(effective_sample_size(weights)),
                "recency_half_life_days": config.recency_half_life_days,
                "train_params": dict(EXPERIMENT_TRAIN_PARAMS),
            }
        )
        oof_predictions.extend(float(value) for value in probs)
        oof_truths.extend(int(value) for value in y_test.to_numpy())

    holdout_indices = np.asarray(plan.holdout_indices, dtype=int)
    holdout_X = candidate_frame.iloc[holdout_indices]
    holdout_y = labels.iloc[holdout_indices]
    holdout_bars = validation_frame.loc[holdout_X.index]
    forward_return_bps = forward_return_by_row.reindex(holdout_X.index).fillna(0.0).astype(float)

    accuracy_threshold_policy = _select_threshold_from_oof_predictions(
        np.asarray(oof_predictions, dtype=float),
        np.asarray(oof_truths, dtype=int),
    )
    threshold_by_symbol: dict[str, float] = {}
    if variant_kind == "symbol_group_calibration":
        threshold_by_symbol = {
            symbol: float(
                _select_threshold_from_oof_predictions(
                    np.asarray([row[0] for row in rows], dtype=float),
                    np.asarray([row[1] for row in rows], dtype=int),
                )["selected_threshold"]
            )
            for symbol, rows in sorted(oof_by_symbol.items())
            if rows
        }
        if threshold_by_symbol:
            accuracy_threshold_policy = {
                **accuracy_threshold_policy,
                "source": "symbol_group_calibration",
                "threshold_scope": "symbol",
                "threshold_by_symbol": threshold_by_symbol,
                "selected_threshold": float(np.median(list(threshold_by_symbol.values()))),
            }

    if config.label_mode == "trade_outcome":
        utility_threshold_policy = select_threshold_by_utility(
            np.asarray(oof_predictions, dtype=float),
            np.asarray(oof_truths, dtype=float),
            np.asarray(oof_forward_returns_bps, dtype=float),
            symbols=oof_symbols,
            fold_ids=oof_fold_ids,
            config=EconomicThresholdConfig(
                threshold_min=0.50,
                threshold_max=0.80,
                threshold_step=0.05,
                min_actionable=max(1, int(min_actionable_decisions)),
                round_trip_cost_bps=float(config.trade_outcome_round_trip_cost_bps),
            ),
        )
        selected_threshold_policy = utility_threshold_policy
        if (
            selected_threshold_policy.get("selected_actionable", 0) < max(1, int(min_actionable_decisions))
            or selected_threshold_policy.get("source") != "economic_utility"
        ):
            selected_threshold_policy = {
                **accuracy_threshold_policy,
                "source": "accuracy_fallback",
                "fallback_reason": "economic_threshold_ineligible_or_unavailable",
            }
    else:
        selected_threshold_policy = accuracy_threshold_policy

    selected_threshold = float(selected_threshold_policy["selected_threshold"])

    dev_mask = _pre_holdout_training_mask(
        candidate_frame.index,
        holdout_start=plan.holdout_start,
        training_window_months=config.training_window_months,
        exclude_indices=holdout_indices,
    )
    dev_X = candidate_frame.iloc[np.flatnonzero(dev_mask)]
    dev_y = labels.iloc[np.flatnonzero(dev_mask)]
    dev_weights = compute_recency_weights(
        pd.DatetimeIndex(dev_X.index.get_level_values("timestamp")),
        half_life_days=float(config.recency_half_life_days),
    )
    holdout_model = train(
        dev_X,
        dev_y,
        horizon=config.horizon,
        sample_weight=dev_weights,
        params_override=EXPERIMENT_TRAIN_PARAMS,
    )
    holdout_probs = predict_proba(holdout_model, holdout_X)
    if variant_kind == "symbol_group_calibration" and threshold_by_symbol:
        holdout_thresholds = np.asarray(
            [
                float(threshold_by_symbol.get(str(symbol), selected_threshold))
                for symbol in holdout_X.index.get_level_values("symbol")
            ],
            dtype=float,
        )
        holdout_preds = (holdout_probs >= holdout_thresholds).astype(int)
    else:
        holdout_preds = (holdout_probs >= selected_threshold).astype(int)
    holdout_accuracy = float((holdout_preds == holdout_y.to_numpy()).mean()) if len(holdout_y) else 0.0
    cost_floor_bps = float(label_audit.get("estimated_cost_floor", {}).get("median_round_trip_cost_bps", 0.0))
    holdout_returns_bps = _candidate_decision_returns_bps(
        label_mode=config.label_mode,
        trade_outcome_side=trade_outcome_side,
        predictions=holdout_preds,
        forward_return_bps=forward_return_bps.to_numpy(dtype=float),
        cost_bps=0.0,
    )
    holdout_cost_adjusted_returns_bps = _candidate_decision_returns_bps(
        label_mode=config.label_mode,
        trade_outcome_side=trade_outcome_side,
        predictions=holdout_preds,
        forward_return_bps=forward_return_bps.to_numpy(dtype=float),
        cost_bps=cost_floor_bps,
    )
    cost_adjusted_expectancy_bps = float(np.mean(holdout_cost_adjusted_returns_bps)) if len(holdout_cost_adjusted_returns_bps) else 0.0
    gross_expectancy_bps = float(np.mean(holdout_returns_bps)) if len(holdout_returns_bps) else 0.0
    predicted_take_count = int((holdout_preds == 1).sum())
    predicted_skip_count = int((holdout_preds == 0).sum())
    if config.label_mode == "trade_outcome":
        predicted_buy_count = predicted_take_count if trade_outcome_side == "long" else 0
        predicted_sell_count = predicted_take_count if trade_outcome_side == "short" else 0
        predicted_hold_count = predicted_skip_count
        actionable_decisions = predicted_take_count
    else:
        predicted_buy_count = predicted_take_count
        predicted_sell_count = predicted_skip_count
        predicted_hold_count = 0
        actionable_decisions = int(len(holdout_preds))
    unique_symbols = int(holdout_X.index.get_level_values("symbol").nunique()) if not holdout_X.empty else 0
    prediction_audit = _summarize_prediction_audit(holdout_X.index, holdout_probs, holdout_preds)
    if config.label_mode == "trade_outcome":
        sample_count = int(len(holdout_preds))
        take_share = float(predicted_take_count / sample_count) if sample_count else 0.0
        skip_share = float(predicted_skip_count / sample_count) if sample_count else 0.0
        prediction_audit.update(
            {
                "label_mode": config.label_mode,
                "trade_outcome_side": trade_outcome_side,
                "predicted_take_count": predicted_take_count,
                "predicted_skip_count": predicted_skip_count,
                "predicted_buy_count": predicted_buy_count,
                "predicted_sell_count": predicted_sell_count,
                "predicted_hold_count": predicted_hold_count,
                "take_share": take_share,
                "skip_share": skip_share,
                "one_sided_take_collapse": bool(sample_count and (take_share < 0.01 or take_share > 0.99)),
            }
        )
    feature_importance = _feature_importance_report(holdout_model)

    replay_signal_resolver = None
    if variant_kind == "symbol_group_calibration" and threshold_by_symbol:
        def _symbol_threshold_resolver(
            actor: ReplayActorConfig,
            symbol: str,
            history: pd.DataFrame,
            timestamp: pd.Timestamp,
            market_risk,
        ) -> StrategySignal | None:
            signal = _candidate_signal_resolver(
                actor,
                symbol,
                history,
                timestamp,
                market_risk,
                feature_frame=feature_frame,
            )
            if signal is None:
                return None
            threshold = float(threshold_by_symbol.get(str(symbol), selected_threshold))
            if signal.signal == "HOLD":
                return signal
            if signal.signal == "BUY" and signal.confidence < threshold:
                return StrategySignal(
                    symbol=symbol,
                    timeframe=signal.timeframe,
                    horizon_bars=signal.horizon_bars,
                    signal="HOLD",
                    confidence=float(signal.confidence),
                    uncertainty=float(signal.uncertainty),
                    reason=f"symbol_threshold_hold_{threshold:.2f}",
                    regime=signal.regime,
                    session_hour_utc=signal.session_hour_utc,
                    market_risk=signal.market_risk,
                )
            if signal.signal == "SELL" and signal.confidence < threshold:
                return StrategySignal(
                    symbol=symbol,
                    timeframe=signal.timeframe,
                    horizon_bars=signal.horizon_bars,
                    signal="HOLD",
                    confidence=float(signal.confidence),
                    uncertainty=float(signal.uncertainty),
                    reason=f"symbol_threshold_hold_{threshold:.2f}",
                    regime=signal.regime,
                    session_hour_utc=signal.session_hour_utc,
                    market_risk=signal.market_risk,
                )
            return signal

        replay_signal_resolver = _symbol_threshold_resolver
    elif variant_kind == "regime_gated_abstain":
        timestamp_context = _build_timestamp_context(validation_frame)
        adverse_timestamps = set(
            pd.Timestamp(timestamp)
            for timestamp in timestamp_context.index[
                (timestamp_context["regime_high_vol_flag"].fillna(0.0) >= 0.5)
                | (timestamp_context["btc_drawdown_24h"].fillna(0.0) <= -0.05)
            ]
        )

        def _regime_gated_resolver(
            actor: ReplayActorConfig,
            symbol: str,
            history: pd.DataFrame,
            timestamp: pd.Timestamp,
            market_risk,
        ) -> StrategySignal | None:
            if pd.Timestamp(timestamp) in adverse_timestamps or bool(getattr(market_risk, "broad_selloff", False)):
                return StrategySignal(
                    symbol=symbol,
                    timeframe="1h",
                    horizon_bars=actor.horizon_bars,
                    signal="HOLD",
                    confidence=0.5,
                    uncertainty=0.5,
                    reason="regime_gated_abstain",
                    market_risk=market_risk,
                )
            return _candidate_signal_resolver(
                actor,
                symbol,
                history,
                timestamp,
                market_risk,
                feature_frame=feature_frame,
            )

        replay_signal_resolver = _regime_gated_resolver

    benchmark_like_behavior: str | None = None
    if config.label_mode == "trade_outcome":
        if trade_outcome_side == "long" and predicted_buy_count / max(1, len(holdout_preds)) > 0.95:
            benchmark_like_behavior = "long_only"
        elif trade_outcome_side == "short" and predicted_sell_count / max(1, len(holdout_preds)) > 0.95:
            benchmark_like_behavior = "short_only"
    candidate_side_reference = _candidate_side_label(
        config,
        {
            "benchmark_like_behavior": benchmark_like_behavior,
            "predicted_buy_count": predicted_buy_count,
            "predicted_sell_count": predicted_sell_count,
        },
    )
    same_side_actor = _same_side_actor_name(candidate_side_reference)
    symbol_pruning_report = _build_symbol_pruning_report(
        holdout_bars,
        model=holdout_model,
        feature_columns=feature_cols,
        threshold=selected_threshold,
        horizon=config.horizon,
        label_mode=config.label_mode,
        trade_outcome_side=trade_outcome_side,
        feature_frame=feature_frame,
        signal_resolver=replay_signal_resolver,
        same_side_actor=same_side_actor,
    )
    allowed_symbols = list(symbol_pruning_report.get("allowed_symbols") or [])
    pruned_holdout_bars = _slice_frame_by_symbols(holdout_bars, allowed_symbols) if allowed_symbols else holdout_bars.iloc[0:0].copy()
    if pruned_holdout_bars.empty:
        replay_report = _empty_candidate_replay_report(pruned_holdout_bars)
        holdout_benchmark_report = _empty_benchmark_replay_report(pruned_holdout_bars)
    else:
        replay_report = _candidate_replay_report(
            pruned_holdout_bars,
            model=holdout_model,
            feature_columns=feature_cols,
            threshold=selected_threshold,
            horizon=config.horizon,
            replay_feature_frame=feature_frame,
            signal_resolver=replay_signal_resolver,
            label_mode=config.label_mode,
            trade_outcome_side=trade_outcome_side,
        )
        holdout_benchmark_report = _benchmark_replay_report(pruned_holdout_bars)
    post_symbol_pruning_report = _build_symbol_pruning_report(
        pruned_holdout_bars,
        model=holdout_model,
        feature_columns=feature_cols,
        threshold=selected_threshold,
        horizon=config.horizon,
        label_mode=config.label_mode,
        trade_outcome_side=trade_outcome_side,
        feature_frame=feature_frame,
        signal_resolver=replay_signal_resolver,
        same_side_actor=same_side_actor,
    )
    regime_section_report = _build_regime_section_report(
        pruned_holdout_bars,
        model=holdout_model,
        feature_columns=feature_cols,
        threshold=selected_threshold,
        horizon=config.horizon,
        label_mode=config.label_mode,
        trade_outcome_side=trade_outcome_side,
        feature_frame=feature_frame,
        signal_resolver=replay_signal_resolver,
        same_side_actor=same_side_actor,
    )
    replay_report["holdout_benchmark_report"] = holdout_benchmark_report
    replay_candidate = replay_report["actor_summaries"]["candidate"]
    replay_flat = replay_report["actor_summaries"]["flat"]
    best_nonflat_benchmark = holdout_benchmark_report["comparisons"].get("best_nonflat_actor")
    benchmark_delta_warnings: list[str] = []
    if not best_nonflat_benchmark or best_nonflat_benchmark not in holdout_benchmark_report["actor_summaries"]:
        benchmark_delta_warnings.append("missing_best_nonflat_benchmark")
        best_nonflat_benchmark = "flat"
    best_nonflat_value = float(holdout_benchmark_report["actor_summaries"].get(best_nonflat_benchmark, {}).get("cost_adjusted_net_pnl_usd", 0.0))
    flat_value = float(replay_flat["cost_adjusted_net_pnl_usd"])
    candidate_value = float(replay_candidate["cost_adjusted_net_pnl_usd"])
    benchmark_best_value = max(
        float(summary["cost_adjusted_net_pnl_usd"])
        for name, summary in holdout_benchmark_report["actor_summaries"].items()
        if name != "flat"
    ) if any(name != "flat" for name in holdout_benchmark_report["actor_summaries"]) else 0.0
    benchmark_best_actor = max(
        (name for name in holdout_benchmark_report["actor_summaries"] if name != "flat"),
        key=lambda name: float(holdout_benchmark_report["actor_summaries"][name]["cost_adjusted_net_pnl_usd"]),
        default="flat",
    )
    same_side_summary = holdout_benchmark_report["actor_summaries"].get(same_side_actor)
    if same_side_summary is None:
        benchmark_delta_warnings.append(f"missing_same_side_benchmark:{same_side_actor}")
        same_side_actor = "flat"
        same_side_summary = holdout_benchmark_report["actor_summaries"].get("flat", {})
    same_side_value = float(same_side_summary.get("cost_adjusted_net_pnl_usd", 0.0))
    benchmark_delta_report = {
        "policy_version": "benchmark_delta_v1",
        "candidate_id": config.candidate_id(),
        "candidate_side": trade_outcome_side if config.label_mode == "trade_outcome" else "directional",
        "candidate_side_reference": candidate_side_reference,
        "threshold_source": str(selected_threshold_policy.get("source", "unknown")),
        "overall": {
            "candidate_pnl_usd": candidate_value,
            "candidate_return_bps": float(candidate_value / 1_000.0 * 10_000.0),
            "flat_pnl_usd": flat_value,
            "best_nonflat_actor": best_nonflat_benchmark,
            "best_nonflat_pnl_usd": best_nonflat_value,
            "same_side_actor": same_side_actor,
            "same_side_pnl_usd": same_side_value,
            "candidate_minus_flat_pnl_usd": float(candidate_value - flat_value),
            "candidate_minus_best_nonflat_pnl_usd": float(candidate_value - best_nonflat_value),
            "candidate_minus_same_side_pnl_usd": float(candidate_value - same_side_value),
            "candidate_minus_best_nonflat_bps": float((candidate_value - best_nonflat_value) / 1_000.0 * 10_000.0),
            "candidate_minus_same_side_bps": float((candidate_value - same_side_value) / 1_000.0 * 10_000.0),
            "take_count": int(replay_candidate.get("fill_count", 0) or 0),
            "fill_count": int(replay_candidate.get("fill_count", 0) or 0),
            "max_drawdown_bps": float(replay_candidate.get("max_drawdown_frac", 0.0)) * 10_000.0,
            "allowed_symbol_count": int(symbol_pruning_report.get("allowed_symbol_count", 0)),
            "rejected_symbol_count": int(symbol_pruning_report.get("rejected_symbol_count", 0)),
        },
        "by_symbol": dict(symbol_pruning_report.get("by_symbol", {})),
        "by_regime": dict(regime_section_report.get("by_dimension", {})),
        "symbol_pruning": symbol_pruning_report,
        "symbol_pruning_post_replay": post_symbol_pruning_report,
        "regime_coverage": dict(regime_section_report.get("coverage", {})),
        "regime_section": regime_section_report,
        "rejected_symbols": list(symbol_pruning_report.get("rejected_symbols", [])),
        "warnings": benchmark_delta_warnings,
        "pre_prune": {
            "symbol_pruning": symbol_pruning_report,
        },
        "post_prune": {
            "symbol_pruning": post_symbol_pruning_report,
            "regime_section": regime_section_report,
            "replay_report": replay_report,
            "benchmark_report": holdout_benchmark_report,
        },
    }
    replay_gap_diagnostics = _build_replay_gap_diagnostics(
        holdout_report={
            "cost_adjusted_expectancy_bps": cost_adjusted_expectancy_bps,
            "predicted_take_count": predicted_take_count,
            "predicted_skip_count": predicted_skip_count,
            "predicted_hold_count": predicted_hold_count,
        },
        replay_report=replay_report,
        benchmark_delta_report=benchmark_delta_report,
    )
    maintenance_report = {
        "candidate_is_active_model": False,
        "decayed": False,
        "no_trade_required": False,
        "proven_shadow_version_id": None,
        "recommended_action": "candidate_recovery_only",
        "blockers": [],
    }
    concentration = replay_candidate.get("exposure_by_symbol_usd", {}) or {}
    positive_pnl_by_symbol = {
        symbol: max(0.0, float(value))
        for symbol, value in concentration.items()
    }
    positive_pnl_total = sum(positive_pnl_by_symbol.values()) or 1.0
    concentration_share = max(positive_pnl_by_symbol.values()) / positive_pnl_total if positive_pnl_by_symbol else 0.0
    drawdown_frac = float(replay_candidate.get("max_drawdown_frac", 0.0))
    score = float(
        (holdout_accuracy * 1_000.0)
        + cost_adjusted_expectancy_bps
        + (candidate_value - flat_value) / 1_000.0 * 10_000.0
        - (abs(drawdown_frac) * 1_000.0)
        - (0.0 if concentration_share <= 0.50 else 100.0)
    )

    selection_risk_summary = {
        "policy_version": POLICY_VERSION,
        "candidate_id": config.candidate_id(),
        "passed": True,
        "score": score,
        "score_percentile_hint": None,
        "failure_reasons": [],
        "candidate_minus_flat_cost_adjusted_net_pnl_usd": float(candidate_value - flat_value),
        "candidate_minus_best_nonflat_cost_adjusted_net_pnl_usd": float(candidate_value - best_nonflat_value),
        "max_drawdown_frac": float(drawdown_frac),
        "symbol_concentration_share": float(concentration_share),
        "label_mode": config.label_mode,
        "trade_outcome_side": trade_outcome_side,
        "take_share": float(predicted_take_count / max(1, len(holdout_preds))),
        "skip_share": float(predicted_skip_count / max(1, len(holdout_preds))),
        "one_sided_take_collapse": bool(len(holdout_preds) and (predicted_take_count / len(holdout_preds) < 0.01 or predicted_take_count / len(holdout_preds) > 0.99)),
        "benchmark_like_behavior": benchmark_like_behavior,
        "fold_accuracy_std": float(np.std(fold_accuracies, ddof=0)) if len(fold_accuracies) > 1 else 0.0,
        "best_nonflat_benchmark": benchmark_best_actor,
        "variant_kind": variant_kind,
        "variant_id": variant_id,
    }
    failure_reasons: list[str] = []
    if holdout_accuracy < min_accuracy:
        failure_reasons.append(f"holdout_accuracy<{min_accuracy:.2f}")
    if cost_adjusted_expectancy_bps <= 0.0:
        failure_reasons.append("non_positive_cost_adjusted_expectancy")
    if candidate_value <= flat_value:
        failure_reasons.append("candidate_did_not_beat_flat")
    if candidate_value <= benchmark_best_value:
        failure_reasons.append("candidate_did_not_beat_nonflat_benchmark")
    if drawdown_frac < -abs(max_drawdown_frac):
        failure_reasons.append("max_drawdown_too_large")
    if actionable_decisions < min_actionable_decisions:
        failure_reasons.append("insufficient_actionable_decisions")
    if unique_symbols < 2:
        failure_reasons.append("insufficient_symbol_coverage")
    if concentration_share > 0.50:
        failure_reasons.append("symbol_concentration_gt_50pct")
    if len(fold_accuracies) >= 2 and float(np.std(fold_accuracies, ddof=0)) > 0.20:
        failure_reasons.append("fold_instability")
    if benchmark_like_behavior is not None:
        benchmark_like_value = float(holdout_benchmark_report["actor_summaries"].get(benchmark_like_behavior, {}).get("cost_adjusted_net_pnl_usd", float("nan")))
        if np.isfinite(benchmark_like_value) and candidate_value <= benchmark_like_value:
            failure_reasons.append("benchmark_like_behavior")

    candidate_quality_decision = evaluate_candidate_quality(
        {
            "candidate_id": config.candidate_id(),
            "config": asdict(config),
            "benchmark_delta_report": benchmark_delta_report,
            "selection_risk_summary": selection_risk_summary,
            "replay_gap_diagnostics": replay_gap_diagnostics,
            "maintenance_report": maintenance_report,
        },
        default_candidate_quality_rules(),
    )
    candidate_quality_report = _candidate_quality_report_from_decision(candidate_quality_decision)
    if not candidate_quality_decision.passed:
        failure_reasons.extend(
            f"quality_{rule_name}"
            for rule_name in candidate_quality_decision.summary.get("hard_failed_rules", []) or candidate_quality_decision.summary.get("failed_rules", [])
        )

    passed = not failure_reasons and candidate_quality_decision.passed
    candidate_model_path: str | None = None
    if passed:
        model_path = candidate_dir / "candidate_model.pkl"
        save_model_bundle(
            holdout_model,
            model_path,
            metadata={
                "policy_version": POLICY_VERSION,
                "candidate_id": config.candidate_id(),
                "feature_set": config.feature_set,
                "feature_columns": list(feature_cols),
                "threshold": float(selected_threshold_policy["selected_threshold"]),
                "threshold_policy": selected_threshold_policy,
                "dataset_digest": _frame_digest(candidate_frame),
                "label_policy_version": LABEL_POLICY_VERSION,
                "benchmark_policy_version": BENCHMARK_POLICY_VERSION,
                "train_params": dict(EXPERIMENT_TRAIN_PARAMS),
                "training_window_months": config.training_window_months,
                "recency_half_life_days": config.recency_half_life_days,
                "dead_zone_bps": config.dead_zone_bps,
                "horizon": config.horizon,
                "benchmark_delta_report": benchmark_delta_report,
                "candidate_quality_report": candidate_quality_report,
                "symbol_pruning_report": symbol_pruning_report,
                "replay_gap_diagnostics": replay_gap_diagnostics,
                "maintenance_report": maintenance_report,
            },
        )
        candidate_model_path = str(model_path)

    fold_ledger = {
        "policy_version": POLICY_VERSION,
        "candidate_id": config.candidate_id(),
        "folds": fold_rows,
        "mean_fold_accuracy": float(np.mean(fold_accuracies)) if fold_accuracies else 0.0,
        "std_fold_accuracy": float(np.std(fold_accuracies, ddof=0)) if len(fold_accuracies) > 1 else 0.0,
        "selected_threshold_policy": selected_threshold_policy,
        "plan": {
            "training_windows_months": list(plan.training_windows_months),
            "holdout_months": int(plan.holdout_months),
            "test_window_months": int(plan.test_window_months),
            "purge_bars": int(plan.purge_bars),
            "holdout_start": str(plan.holdout_start) if plan.holdout_start is not None else None,
            "holdout_end": str(plan.holdout_end) if plan.holdout_end is not None else None,
            "fold_count": int(len(plan.folds)),
            "selected_validation_fold_count": int(len(selected_validation_folds)),
            "selected_validation_fold_ids": [str(fold.fold_id) for fold in selected_validation_folds],
            "max_validation_folds": int(DEFAULT_MAX_VALIDATION_FOLDS),
            "train_params": dict(EXPERIMENT_TRAIN_PARAMS),
        },
    }
    holdout_report = {
        "policy_version": POLICY_VERSION,
        "candidate_id": config.candidate_id(),
        "label_mode": config.label_mode,
        "trade_outcome_side": trade_outcome_side,
        "threshold": float(selected_threshold_policy["selected_threshold"]),
        "holdout_accuracy": holdout_accuracy,
        "decision_count": actionable_decisions,
        "cost_adjusted_expectancy_bps": cost_adjusted_expectancy_bps,
        "gross_expectancy_bps": gross_expectancy_bps,
        "predicted_take_count": predicted_take_count,
        "predicted_skip_count": predicted_skip_count,
        "predicted_buy_count": predicted_buy_count,
        "predicted_sell_count": predicted_sell_count,
        "predicted_hold_count": predicted_hold_count,
        "prediction_audit": prediction_audit,
        "holdout_start": str(plan.holdout_start) if plan.holdout_start is not None else None,
        "holdout_end": str(plan.holdout_end) if plan.holdout_end is not None else None,
        "sample_count": int(len(holdout_preds)),
        "symbols": sorted(str(symbol) for symbol in holdout_X.index.get_level_values("symbol").unique()),
        "feature_importance": feature_importance,
    }
    selection_risk_summary = {
        "policy_version": POLICY_VERSION,
        "candidate_id": config.candidate_id(),
        "passed": passed,
        "score": score,
        "score_percentile_hint": None,
        "failure_reasons": failure_reasons,
        "candidate_minus_flat_cost_adjusted_net_pnl_usd": float(candidate_value - flat_value),
        "candidate_minus_best_nonflat_cost_adjusted_net_pnl_usd": float(candidate_value - best_nonflat_value),
        "max_drawdown_frac": float(drawdown_frac),
        "symbol_concentration_share": float(concentration_share),
        "label_mode": config.label_mode,
        "trade_outcome_side": trade_outcome_side,
        "take_share": float(predicted_take_count / max(1, len(holdout_preds))),
        "skip_share": float(predicted_skip_count / max(1, len(holdout_preds))),
        "one_sided_take_collapse": bool(len(holdout_preds) and (predicted_take_count / len(holdout_preds) < 0.01 or predicted_take_count / len(holdout_preds) > 0.99)),
        "benchmark_like_behavior": benchmark_like_behavior,
        "fold_accuracy_std": float(np.std(fold_accuracies, ddof=0)) if len(fold_accuracies) > 1 else 0.0,
        "best_nonflat_benchmark": benchmark_best_actor,
        "variant_kind": variant_kind,
        "variant_id": variant_id,
    }

    candidate_eval = CandidateEvaluation(
        config=config,
        candidate_id=config.candidate_id(),
        passed=passed,
        score=score,
        failure_reasons=tuple(failure_reasons),
        dataset_manifest=_to_jsonable(dataset_manifest),
        feature_manifest=_to_jsonable(feature_manifest),
        label_audit=_to_jsonable(label_audit),
        fold_ledger=_to_jsonable(fold_ledger),
        threshold_policy=_to_jsonable(selected_threshold_policy),
        accuracy_threshold_policy=_to_jsonable(accuracy_threshold_policy),
        holdout_report=_to_jsonable(holdout_report),
        replay_report=_to_jsonable(replay_report),
        selection_risk_summary=_to_jsonable(selection_risk_summary),
        benchmark_delta_report=_to_jsonable(benchmark_delta_report),
        candidate_quality_report=_to_jsonable(candidate_quality_report),
        replay_gap_diagnostics=_to_jsonable(replay_gap_diagnostics),
        symbol_pruning_report=_to_jsonable(symbol_pruning_report),
        maintenance_report=_to_jsonable(maintenance_report),
        model_artifact_path=candidate_model_path,
        variant_id=variant_id,
    )

    _write_json(
        candidate_dir / "candidate_config.json",
        {
            "variant_id": variant_id,
            "variant_kind": variant_kind,
            **asdict(config),
        },
    )
    _write_json(candidate_dir / "dataset_manifest.json", dataset_manifest)
    _write_json(candidate_dir / "feature_manifest.json", feature_manifest)
    _write_json(candidate_dir / "label_audit.json", label_audit)
    _write_json(candidate_dir / "fold_ledger.json", fold_ledger)
    _write_json(candidate_dir / "threshold_policy.json", selected_threshold_policy)
    _write_json(candidate_dir / "accuracy_threshold_policy.json", accuracy_threshold_policy)
    _write_json(candidate_dir / "holdout_report.json", holdout_report)
    _write_json(candidate_dir / "replay_report.json", replay_report)
    _write_json(candidate_dir / "selection_risk_summary.json", selection_risk_summary)
    _write_json(
        candidate_dir / "manifest.json",
        {
            "policy_version": POLICY_VERSION,
            "candidate_id": config.candidate_id(),
            "variant_id": variant_id,
            "variant_kind": variant_kind,
            "run_id": None,  # filled by caller
            "created_at": _utc_now(),
            "snapshot_digest": dataset_manifest.get("content_digests", {}).get("frame_sha256"),
            "horizon": config.horizon,
            "training_window_months": config.training_window_months,
            "recency_half_life_days": config.recency_half_life_days,
            "dead_zone_bps": config.dead_zone_bps,
            "feature_set": config.feature_set,
            "label_mode": config.label_mode,
            "trade_outcome_profit_target_bps": config.trade_outcome_profit_target_bps,
            "trade_outcome_stop_loss_bps": config.trade_outcome_stop_loss_bps,
            "trade_outcome_round_trip_cost_bps": config.trade_outcome_round_trip_cost_bps,
            "model_type": "lgbm_classifier",
            "train_params": dict(EXPERIMENT_TRAIN_PARAMS),
            "artifact_id": artifact_id,
            "quality_gates": {
                "min_accuracy": min_accuracy,
                "min_actionable_decisions": min_actionable_decisions,
                "max_drawdown_frac": max_drawdown_frac,
            },
            "pass_status": passed,
            "failure_reasons": failure_reasons,
            "model_artifact_path": candidate_model_path,
            "threshold_policy": selected_threshold_policy,
            "accuracy_threshold_policy": accuracy_threshold_policy,
            "selection_risk_summary": selection_risk_summary,
            "benchmark_delta_report": benchmark_delta_report,
            "candidate_quality_report": candidate_quality_report,
            "symbol_pruning_report": symbol_pruning_report,
            "replay_gap_diagnostics": replay_gap_diagnostics,
            "maintenance_report": maintenance_report,
        },
    )
    return candidate_eval


def _render_candidate_table(candidates: list[CandidateEvaluation]) -> str:
    rows = []
    for candidate in candidates[:5]:
        holdout_accuracy = float(candidate.holdout_report.get("holdout_accuracy", 0.0)) if isinstance(candidate.holdout_report, dict) else 0.0
        quality_report = candidate.candidate_quality_report if isinstance(candidate.candidate_quality_report, dict) else {}
        quality_summary = quality_report.get("summary", {}) if isinstance(quality_report, dict) else {}
        benchmark_delta = candidate.benchmark_delta_report if isinstance(candidate.benchmark_delta_report, dict) else {}
        overall_delta = benchmark_delta.get("overall", {}) if isinstance(benchmark_delta, dict) else {}
        symbol_pruning = benchmark_delta.get("symbol_pruning", {}) if isinstance(benchmark_delta, dict) else {}
        hard_failed_rules = list(quality_summary.get("hard_failed_rules", []) or quality_summary.get("failed_rules", []) or [])[:3]
        rows.append(
            [
                candidate.candidate_id,
                candidate.config.horizon,
                candidate.config.training_window_months,
                candidate.config.recency_half_life_days,
                f"{candidate.config.dead_zone_bps:.4f}",
                candidate.config.feature_set,
                f"{holdout_accuracy:.4f}",
                f"{candidate.score:.2f}",
                "pass" if candidate.passed else "fail",
                str(quality_summary.get("overall_decision", "")),
                str(quality_report.get("evidence_digest", "")),
                ",".join(hard_failed_rules),
                f"{float(overall_delta.get('candidate_minus_best_nonflat_bps', 0.0)):.2f}",
                f"{float(overall_delta.get('candidate_minus_same_side_bps', 0.0)):.2f}",
                int(symbol_pruning.get("allowed_symbol_count", 0)),
                int(symbol_pruning.get("rejected_symbol_count", 0)),
            ]
        )
    return _markdown_table(
        [
            "candidate_id",
            "horizon",
            "window_m",
            "half_life_d",
            "dead_zone",
            "feature_set",
            "holdout_acc",
            "score",
            "status",
            "quality_decision",
            "quality_evidence_digest",
            "quality_hard_fails",
            "best_nonflat_bps",
            "same_side_bps",
            "allowed_symbols",
            "rejected_symbols",
        ],
        rows,
    )


def _render_candidate_quality_summary_md(candidate_quality_summary: dict[str, Any]) -> str:
    rows = [
        ["evaluated_candidates", int(candidate_quality_summary.get("evaluated_candidates", 0))],
        ["passed_quality", int(candidate_quality_summary.get("passed_quality", 0))],
        ["watch_quality", int(candidate_quality_summary.get("watch_quality", 0))],
        ["failed_quality", int(candidate_quality_summary.get("failed_quality", 0))],
    ]
    lines = [
        "## Candidate Quality",
        "",
        _markdown_table(["metric", "value"], rows),
    ]
    top_failures = candidate_quality_summary.get("top_failure_reasons", []) or []
    if top_failures:
        lines.extend(["", "### Top Failure Reasons", ""])
        lines.extend(f"- `{item.get('reason', '')}`: {int(item.get('count', 0))}" for item in top_failures)
    return "\n".join(lines)


def _render_benchmark_table(benchmark_report: dict[str, Any]) -> str:
    rows = []
    for name, summary in benchmark_report["actor_summaries"].items():
        rows.append(
            [
                name,
                f"{float(summary['cost_adjusted_net_pnl_usd']):.2f}",
                f"{float(summary.get('max_drawdown_frac', 0.0)):.4f}",
                int(summary.get("fill_count", 0)),
            ]
        )
    return _markdown_table(["benchmark", "cost_adj_pnl_usd", "max_dd", "fills"], rows)


def _render_experiment_summary_md(summary: dict[str, Any], benchmark_report: dict[str, Any], candidates: list[CandidateEvaluation]) -> str:
    lines = [
        "# Model Recovery Experiment Summary",
        "",
        f"- run_id: `{summary['run_id']}`",
        f"- recommendation: `{summary['recommendation']}`",
        f"- selected_candidate_id: `{summary.get('selected_candidate_id') or ''}`",
        f"- evaluated_candidates: `{summary['evaluated_candidates']}`",
        f"- passed_candidates: `{summary['passed_candidates']}`",
        "",
        "## Benchmarks",
        "",
        _render_benchmark_table(benchmark_report),
        "",
        _render_candidate_quality_summary_md(summary.get("candidate_quality_summary", {})),
        "",
        "## Top Candidates",
        "",
        _render_candidate_table(candidates),
        "",
        "## Next Operator Action",
        "",
        f"`{summary['next_operator_action']}`",
    ]
    if summary.get("warnings"):
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in summary["warnings"])
    return "\n".join(lines)


def _render_phase4_diagnostics_md(diagnostics_report: dict[str, Any]) -> str:
    summary = diagnostics_report.get("summary", {}) or {}
    regime = diagnostics_report.get("regime_comparison", {}) or {}
    variant_reports = diagnostics_report.get("variant_reports", {}) or {}
    lines = [
        "# Research Input Diagnostics",
        "",
        f"- policy_version: `{diagnostics_report.get('policy_version')}`",
        f"- generated_at: `{diagnostics_report.get('generated_at')}`",
        f"- recommended_next_step: `{summary.get('recommended_next_step', '')}`",
        f"- variant_count: `{summary.get('variant_count', 0)}`",
        f"- passed_variant_count: `{summary.get('passed_variant_count', 0)}`",
        f"- best_failed_variant_id: `{summary.get('best_failed_variant_id') or ''}`",
        "",
        "## Failure Signals",
        "",
        _markdown_table(
            ["metric", "value"],
            [
                ["prediction_collapse_observed", bool(summary.get("prediction_collapse_observed", False))],
                ["failed_variant_count", int(summary.get("failed_variant_count", 0))],
                ["best_failed_score", summary.get("best_failed_score")],
            ],
        ),
        "",
        "## Variants",
        "",
        _markdown_table(
            ["variant_id", "passed", "holdout_acc", "cost_adj_expectancy_bps", "replay_pnl_usd"],
            [
                [
                    variant_id,
                    bool(report.get("passed", False)),
                    f"{float(report.get('holdout_accuracy', 0.0)):.4f}",
                    f"{float(report.get('cost_adjusted_expectancy_bps', 0.0)):.4f}",
                    f"{float(report.get('replay_cost_adjusted_net_pnl_usd', 0.0)):.2f}",
                ]
                for variant_id, report in sorted(variant_reports.items())
            ],
        ),
        "",
        "## Regime Comparison",
        "",
        _markdown_table(
            ["regime", "available", "rows", "best_actor", "best_minus_flat_pnl_usd"],
            [
                [
                    name,
                    bool(payload.get("available", False)),
                    int(payload.get("rows", 0)),
                    payload.get("best_actor", ""),
                    f"{float(payload.get('best_minus_flat_cost_adjusted_net_pnl_usd', 0.0)):.2f}",
                ]
                for name, payload in sorted(regime.items())
            ],
        ),
        "",
        "## JSON",
        "```json",
        json.dumps(diagnostics_report, indent=2, sort_keys=True, default=str),
        "```",
    ]
    return "\n".join(lines)


def _render_label_audit_md(label_audit: dict[str, Any]) -> str:
    best = label_audit.get("summary", {}).get("best_cell_stats") or {}
    lines = [
        "# Latest Label Audit",
        "",
        f"- policy_version: `{label_audit.get('policy_version')}`",
        f"- dataset_rows: `{label_audit.get('dataset', {}).get('rows', 0)}`",
        f"- cell_count: `{label_audit.get('summary', {}).get('cell_count', 0)}`",
        f"- best_cell: `{label_audit.get('summary', {}).get('best_cell') or ''}`",
        f"- recommended_dead_zone_floor_bps: `{label_audit.get('summary', {}).get('recommended_dead_zone_floor_bps', 0.0):.4f}`",
        "",
        "## Best Cell",
        "",
        _markdown_table(
            ["metric", "value"],
            [
                ["sample_count", best.get("sample_count", 0)],
                ["symbol_count", best.get("symbol_count", 0)],
                ["ambiguous_rate", f"{float(best.get('ambiguous_rate', 0.0)):.4f}"],
                ["label_balance_score", f"{float(best.get('label_balance_score', 0.0)):.4f}"],
            ],
        ),
    ]
    return "\n".join(lines)


def _render_benchmark_md(benchmark_report: dict[str, Any]) -> str:
    lines = [
        "# Latest Benchmark Replay",
        "",
        f"- policy_version: `{benchmark_report.get('policy_version')}`",
        f"- best_actor: `{benchmark_report.get('comparisons', {}).get('best_actor') or ''}`",
        f"- best_nonflat_actor: `{benchmark_report.get('comparisons', {}).get('best_nonflat_actor') or ''}`",
        f"- best_minus_flat_cost_adjusted_net_pnl_usd: `{float(benchmark_report.get('comparisons', {}).get('best_minus_flat_cost_adjusted_net_pnl_usd', 0.0)):.4f}`",
        "",
        _render_benchmark_table(benchmark_report),
    ]
    return "\n".join(lines)


def _render_manifest_summary(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_id": summary["run_id"],
        "snapshot_path": summary["snapshot_path"],
        "output_dir": summary["output_dir"],
        "docs_output_dir": summary["docs_output_dir"],
        "evaluated_candidates": summary["evaluated_candidates"],
        "passed_candidates": summary["passed_candidates"],
        "recommendation": summary["recommendation"],
        "selected_candidate_id": summary.get("selected_candidate_id"),
        "candidate_quality_summary": summary.get("candidate_quality_summary", {}),
        "next_operator_action": summary["next_operator_action"],
        "warnings": summary.get("warnings", []),
    }


def _selected_candidate(reports: list[CandidateEvaluation]) -> CandidateEvaluation | None:
    passed = [candidate for candidate in reports if candidate.passed]
    if not passed:
        return None
    return max(
        passed,
        key=lambda candidate: (
            float(candidate.score),
            float(candidate.holdout_report.get("holdout_accuracy", 0.0)),
            float(candidate.replay_report.get("actor_summaries", {}).get("candidate", {}).get("cost_adjusted_net_pnl_usd", 0.0)),
            candidate.candidate_id,
        ),
    )


def run_phase4_research_input_repair(
    dataset: pd.DataFrame,
    *,
    snapshot_path: Path | str,
    output_root: Path | str,
    docs_output_dir: Path | str,
    seed: int = DEFAULT_SEED,
    min_accuracy: float = DEFAULT_MIN_ACCURACY,
    min_actionable_decisions: int = DEFAULT_MIN_ACTIONABLE_DECISIONS,
    max_drawdown_frac: float = DEFAULT_MAX_DRAWDOWN_FRAC,
    label_mode: str = "directional_return",
    trade_outcome_profit_target_bps: float = 20.0,
    trade_outcome_stop_loss_bps: float = 30.0,
    trade_outcome_round_trip_cost_bps: float = 8.0,
    max_variants: int = 4,
) -> Phase4RepairRunResult:
    np.random.default_rng(int(seed))
    output_root = Path(output_root).expanduser()
    docs_output_dir = Path(docs_output_dir).expanduser()
    snapshot_path = Path(snapshot_path).expanduser()
    run_id = _run_id()
    run_dir = output_root / run_id / "phase4_repair"
    run_dir.mkdir(parents=True, exist_ok=True)
    docs_output_dir.mkdir(parents=True, exist_ok=True)

    frame = _ensure_multiindex_frame(dataset)
    if label_mode == "trade_outcome":
        label_audit_report = _build_trade_outcome_label_audit_report(
            frame,
            profit_target_bps=float(trade_outcome_profit_target_bps),
            stop_loss_bps=float(trade_outcome_stop_loss_bps),
            round_trip_cost_bps=float(trade_outcome_round_trip_cost_bps),
        )
    else:
        label_audit_report = _build_label_audit_report(frame)
    benchmark_report = _benchmark_replay_report(frame)
    variant_specs = build_phase4_variant_specs(
        label_audit_report,
        label_mode=label_mode,
        trade_outcome_profit_target_bps=float(trade_outcome_profit_target_bps),
        trade_outcome_stop_loss_bps=float(trade_outcome_stop_loss_bps),
        trade_outcome_round_trip_cost_bps=float(trade_outcome_round_trip_cost_bps),
    )[: max(1, min(int(max_variants), 4))]

    variant_evaluations: list[CandidateEvaluation] = []
    for spec in variant_specs:
        try:
            evaluation = _evaluate_candidate(
                frame,
                config=spec.config,
                benchmark_report={**benchmark_report, "label_audit": label_audit_report},
                run_dir=run_dir,
                min_accuracy=float(min_accuracy),
                min_actionable_decisions=int(min_actionable_decisions),
                max_drawdown_frac=float(max_drawdown_frac),
                variant_id=spec.variant_id,
                variant_kind=spec.variant_kind,
            )
        except Exception as exc:
            artifact_id = f"{spec.config.candidate_id()}__{spec.variant_id}" if spec.variant_id else spec.config.candidate_id()
            candidate_dir = run_dir / "candidates" / artifact_id
            candidate_dir.mkdir(parents=True, exist_ok=True)
            error_payload = {
                "policy_version": POLICY_VERSION,
                "candidate_id": spec.config.candidate_id(),
                "variant_id": spec.variant_id,
                "variant_kind": spec.variant_kind,
                "error": str(exc),
                "snapshot_path": str(snapshot_path),
            }
            for filename in (
                "candidate_config.json",
                "dataset_manifest.json",
                "feature_manifest.json",
                "label_audit.json",
                "fold_ledger.json",
                "threshold_policy.json",
                "holdout_report.json",
                "replay_report.json",
                "selection_risk_summary.json",
                "manifest.json",
            ):
                _write_json(candidate_dir / filename, error_payload)
            evaluation = CandidateEvaluation(
                config=spec.config,
                candidate_id=spec.config.candidate_id(),
                passed=False,
                score=float("-inf"),
                failure_reasons=(str(exc),),
                dataset_manifest={},
                feature_manifest={},
                label_audit={},
                fold_ledger={},
                threshold_policy={},
                holdout_report={},
                replay_report={},
                selection_risk_summary={"variant_kind": spec.variant_kind, "variant_id": spec.variant_id, "error": str(exc)},
                model_artifact_path=None,
                variant_id=spec.variant_id,
            )
        variant_evaluations.append(evaluation)

    diagnostics_report = build_research_input_diagnostics_report(
        frame,
        label_audit_report=label_audit_report,
        benchmark_report=benchmark_report,
        variant_evaluations=variant_evaluations,
    )
    passed = [variant for variant in variant_evaluations if variant.passed]
    candidate_quality_summary = _build_candidate_quality_summary(
        [
            evaluation.candidate_quality_report
            for evaluation in variant_evaluations
            if isinstance(evaluation.candidate_quality_report, dict) and evaluation.candidate_quality_report
        ]
    )
    selected = _selected_candidate(variant_evaluations)
    recommendation = "paper_quarantine_candidate" if selected is not None else "remain_no_trade"
    next_operator_action = (
        "paper_quarantine_selected_candidate" if selected is not None else "remain_no_trade_and_repair_research_inputs"
    )
    summary = {
        "policy_version": POLICY_VERSION,
        "run_id": run_id,
        "snapshot_path": str(snapshot_path),
        "output_dir": str(run_dir),
        "docs_output_dir": str(docs_output_dir),
        "label_mode": label_mode,
        "dataset": {
            "rows": int(len(frame)),
            "symbols": sorted(str(symbol) for symbol in frame.index.get_level_values("symbol").unique()),
            "start": str(pd.DatetimeIndex(frame.index.get_level_values("timestamp")).min()),
            "end": str(pd.DatetimeIndex(frame.index.get_level_values("timestamp")).max()),
            "frame_digest": _frame_digest(frame),
        },
        "evaluated_variants": len(variant_evaluations),
        "passed_variants": len(passed),
        "evaluated_candidates": len(variant_evaluations),
        "passed_candidates": len(passed),
        "selected_variant_id": selected.variant_id if selected is not None else None,
        "selected_candidate_id": selected.candidate_id if selected is not None else None,
        "candidate_quality_summary": candidate_quality_summary,
        "recommendation": recommendation,
        "next_operator_action": next_operator_action,
        "warnings": diagnostics_report.get("diagnostic_notes", []),
    }

    _write_json(run_dir / "phase4_research_input_repair.json", summary)
    (run_dir / "phase4_research_input_repair.md").write_text(
        _render_experiment_summary_md(summary, benchmark_report, variant_evaluations),
        encoding="utf-8",
    )
    _write_json(run_dir / "research_input_diagnostics.json", diagnostics_report)
    (run_dir / "research_input_diagnostics.md").write_text(_render_phase4_diagnostics_md(diagnostics_report), encoding="utf-8")

    _write_json(docs_output_dir / "research_input_diagnostics.json", diagnostics_report)
    (docs_output_dir / "research_input_diagnostics.md").write_text(_render_phase4_diagnostics_md(diagnostics_report), encoding="utf-8")
    phase4_docs_dir = docs_output_dir / "research_input_repair" / run_id
    phase4_docs_dir.mkdir(parents=True, exist_ok=True)
    _write_json(phase4_docs_dir / "summary.json", summary)
    (phase4_docs_dir / "summary.md").write_text(
        _render_experiment_summary_md(summary, benchmark_report, variant_evaluations),
        encoding="utf-8",
    )
    _write_json(phase4_docs_dir / "diagnostics.json", diagnostics_report)
    (phase4_docs_dir / "diagnostics.md").write_text(
        _render_phase4_diagnostics_md(diagnostics_report),
        encoding="utf-8",
    )
    _write_json(
        phase4_docs_dir / "variant_index.json",
        {
            "policy_version": POLICY_VERSION,
            "run_id": run_id,
            "variants": [
                {
                    "variant_id": evaluation.variant_id,
                    "candidate_id": evaluation.candidate_id,
                    "artifact_id": f"{evaluation.candidate_id}__{evaluation.variant_id}" if evaluation.variant_id else evaluation.candidate_id,
                    "passed": bool(evaluation.passed),
                    "score": float(evaluation.score),
                    "artifact_dir": str(run_dir / "candidates" / (f"{evaluation.candidate_id}__{evaluation.variant_id}" if evaluation.variant_id else evaluation.candidate_id)),
                }
                for evaluation in variant_evaluations
            ],
        },
    )

    return Phase4RepairRunResult(
        run_id=run_id,
        output_dir=run_dir,
        docs_output_dir=docs_output_dir,
        snapshot_path=snapshot_path,
        diagnostics_report=diagnostics_report,
        benchmark_report=benchmark_report,
        label_audit_report=label_audit_report,
        variant_evaluations=variant_evaluations,
        summary=summary,
    )


def run_model_recovery_experiments(
    dataset: pd.DataFrame,
    *,
    snapshot_path: Path | str,
    output_root: Path | str,
    docs_output_dir: Path | str,
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
    seed: int = DEFAULT_SEED,
    min_accuracy: float = DEFAULT_MIN_ACCURACY,
    min_actionable_decisions: int = DEFAULT_MIN_ACTIONABLE_DECISIONS,
    max_drawdown_frac: float = DEFAULT_MAX_DRAWDOWN_FRAC,
    label_mode: str = "directional_return",
    trade_outcome_profit_target_bps: float = 20.0,
    trade_outcome_stop_loss_bps: float = 30.0,
    trade_outcome_round_trip_cost_bps: float = 8.0,
    no_production_registry: bool = False,
) -> ExperimentRunResult:
    if not no_production_registry:
        raise ValueError("--no-production-registry is required for this workflow")
    if max_candidates < 1:
        raise ValueError("max_candidates must be positive")

    np.random.default_rng(int(seed))
    output_root = Path(output_root).expanduser()
    docs_output_dir = Path(docs_output_dir).expanduser()
    snapshot_path = Path(snapshot_path).expanduser()
    run_id = _run_id()
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    docs_output_dir.mkdir(parents=True, exist_ok=True)

    frame = _ensure_multiindex_frame(dataset)
    if label_mode == "trade_outcome":
        label_audit_report = _build_trade_outcome_label_audit_report(
            frame,
            profit_target_bps=float(trade_outcome_profit_target_bps),
            stop_loss_bps=float(trade_outcome_stop_loss_bps),
            round_trip_cost_bps=float(trade_outcome_round_trip_cost_bps),
        )
    else:
        label_audit_report = _build_label_audit_report(frame)
    benchmark_report = _benchmark_replay_report(frame)
    candidate_configs = _candidate_grid(
        label_audit_report,
        label_mode=label_mode,
        trade_outcome_profit_target_bps=float(trade_outcome_profit_target_bps),
        trade_outcome_stop_loss_bps=float(trade_outcome_stop_loss_bps),
        trade_outcome_round_trip_cost_bps=float(trade_outcome_round_trip_cost_bps),
    )
    candidate_configs = _limit_candidates(candidate_configs, max_candidates)

    evaluations: list[CandidateEvaluation] = []
    for config in candidate_configs:
        try:
            evaluation = _evaluate_candidate(
                frame,
                config=config,
                benchmark_report={**benchmark_report, "label_audit": label_audit_report},
                run_dir=run_dir,
                min_accuracy=float(min_accuracy),
                min_actionable_decisions=int(min_actionable_decisions),
                max_drawdown_frac=float(max_drawdown_frac),
            )
            candidate_dir = run_dir / "candidates" / config.candidate_id()
            candidate_manifest_path = candidate_dir / "manifest.json"
            if candidate_manifest_path.exists():
                manifest_payload = json.loads(candidate_manifest_path.read_text(encoding="utf-8"))
            else:
                manifest_payload = {
                    "policy_version": POLICY_VERSION,
                    "candidate_id": config.candidate_id(),
                    "created_at": _utc_now(),
                    "snapshot_digest": evaluation.dataset_manifest.get("content_digests", {}).get("frame_sha256")
                    if isinstance(evaluation.dataset_manifest, dict)
                    else None,
                    "horizon": config.horizon,
                    "training_window_months": config.training_window_months,
                    "recency_half_life_days": config.recency_half_life_days,
                    "dead_zone_bps": config.dead_zone_bps,
                    "feature_set": config.feature_set,
                    "model_type": "lgbm_classifier",
                    "quality_gates": {
                        "min_accuracy": min_accuracy,
                        "min_actionable_decisions": min_actionable_decisions,
                        "max_drawdown_frac": max_drawdown_frac,
                    },
                    "pass_status": evaluation.passed,
                    "failure_reasons": list(evaluation.failure_reasons),
                    "model_artifact_path": evaluation.model_artifact_path,
                    "threshold_policy": evaluation.threshold_policy,
                    "selection_risk_summary": evaluation.selection_risk_summary,
                    "benchmark_delta_report": evaluation.benchmark_delta_report,
                    "candidate_quality_report": evaluation.candidate_quality_report,
                    "symbol_pruning_report": evaluation.symbol_pruning_report,
                    "replay_gap_diagnostics": evaluation.replay_gap_diagnostics,
                    "maintenance_report": evaluation.maintenance_report,
                }
            manifest_payload["run_id"] = run_id
            _write_json(candidate_manifest_path, manifest_payload)
            evaluations.append(evaluation)
        except Exception as exc:
            logger.exception("Candidate evaluation failed for %s: %s", config.candidate_id(), exc)
            candidate_dir = run_dir / "candidates" / config.candidate_id()
            candidate_dir.mkdir(parents=True, exist_ok=True)
            error_payload = {
                "policy_version": POLICY_VERSION,
                "candidate_id": config.candidate_id(),
                "error": str(exc),
                "pass_status": False,
                "failure_reasons": [str(exc)],
                "run_id": run_id,
            }
            for filename in (
                "candidate_config.json",
                "dataset_manifest.json",
                "feature_manifest.json",
                "label_audit.json",
                "fold_ledger.json",
                "threshold_policy.json",
                "holdout_report.json",
                "replay_report.json",
                "selection_risk_summary.json",
                "manifest.json",
            ):
                _write_json(candidate_dir / filename, error_payload)
            failure = CandidateEvaluation(
                config=config,
                candidate_id=config.candidate_id(),
                passed=False,
                score=float("-inf"),
                failure_reasons=(str(exc),),
                dataset_manifest={},
                feature_manifest={},
                label_audit={},
                fold_ledger={},
                threshold_policy={},
                holdout_report={},
                replay_report={},
                selection_risk_summary={"error": str(exc)},
                model_artifact_path=None,
            )
            evaluations.append(failure)

    passed = [candidate for candidate in evaluations if candidate.passed]
    candidate_quality_summary = _build_candidate_quality_summary(
        [
            candidate.candidate_quality_report
            for candidate in evaluations
            if isinstance(candidate.candidate_quality_report, dict) and candidate.candidate_quality_report
        ]
    )
    selected = _selected_candidate(evaluations)
    recommendation = "paper_quarantine_candidate" if selected is not None else "remain_no_trade"
    next_operator_action = (
        "paper_quarantine_selected_candidate" if selected is not None else "remain_no_trade_and_collect_more_data"
    )
    warnings = []
    symbols = sorted(str(symbol) for symbol in frame.index.get_level_values("symbol").unique())
    if len(symbols) < 2:
        warnings.append("dataset contains fewer than two symbols")

    summary = {
        "policy_version": POLICY_VERSION,
        "run_id": run_id,
        "snapshot_path": str(snapshot_path),
        "output_dir": str(run_dir),
        "docs_output_dir": str(docs_output_dir),
        "dataset": {
            "rows": int(len(frame)),
            "symbols": symbols,
            "start": str(pd.DatetimeIndex(frame.index.get_level_values("timestamp")).min()),
            "end": str(pd.DatetimeIndex(frame.index.get_level_values("timestamp")).max()),
            "frame_digest": _frame_digest(frame),
        },
        "evaluated_candidates": len(evaluations),
        "passed_candidates": len(passed),
        "selected_candidate_id": selected.candidate_id if selected is not None else None,
        "candidate_quality_summary": candidate_quality_summary,
        "recommendation": recommendation,
        "next_operator_action": next_operator_action,
        "label_mode": label_mode,
        "warnings": warnings,
    }
    selection_risk_report = build_selection_risk_report(
        [
            {
                "candidate_id": candidate.candidate_id,
                "score": candidate.score,
                "holdout_accuracy": candidate.holdout_report.get("holdout_accuracy") if isinstance(candidate.holdout_report, dict) else None,
                "mean_accuracy": candidate.fold_ledger.get("mean_fold_accuracy") if isinstance(candidate.fold_ledger, dict) else None,
                "fold_ledger": candidate.fold_ledger,
            }
            for candidate in evaluations
        ],
        selected_candidate_id=selected.candidate_id if selected is not None else None,
        trial_count=len(evaluations),
    )
    summary["selection_risk_report"] = selection_risk_report
    if selection_risk_report.get("blockers"):
        selection_risk_report["selected_candidate_id"] = None
        summary["recommendation"] = "remain_no_trade"
        summary["next_operator_action"] = "remain_no_trade_and_collect_more_data"
        summary["selected_candidate_id"] = None

    _write_json(run_dir / "label_audit.json", label_audit_report)
    (run_dir / "label_audit.md").write_text(_render_label_audit_md(label_audit_report), encoding="utf-8")
    _write_json(run_dir / "benchmark_replay.json", benchmark_report)
    (run_dir / "benchmark_replay.md").write_text(_render_benchmark_md(benchmark_report), encoding="utf-8")
    _write_json(run_dir / "experiment_summary.json", summary)
    (run_dir / "experiment_summary.md").write_text(_render_experiment_summary_md(summary, benchmark_report, evaluations), encoding="utf-8")

    _write_json(docs_output_dir / "latest_label_audit.json", label_audit_report)
    (docs_output_dir / "latest_label_audit.md").write_text(_render_label_audit_md(label_audit_report), encoding="utf-8")
    _write_json(docs_output_dir / "latest_benchmark_replay.json", benchmark_report)
    (docs_output_dir / "latest_benchmark_replay.md").write_text(_render_benchmark_md(benchmark_report), encoding="utf-8")
    _write_json(docs_output_dir / "latest_experiment_summary.json", summary)
    experiment_summary_md = _render_experiment_summary_md(summary, benchmark_report, evaluations)
    (docs_output_dir / "latest_experiment_summary.md").write_text(experiment_summary_md, encoding="utf-8")
    (docs_output_dir / "latest_recovery_sweep_summary.json").write_text(
        json.dumps(_to_jsonable(summary), indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    (docs_output_dir / "latest_recovery_sweep_summary.md").write_text(experiment_summary_md, encoding="utf-8")

    result = ExperimentRunResult(
        run_id=run_id,
        output_dir=run_dir,
        docs_output_dir=docs_output_dir,
        snapshot_path=snapshot_path,
        evaluated_candidates=evaluations,
        passed_candidates=passed,
        benchmark_report=benchmark_report,
        label_audit_report=label_audit_report,
        summary=summary,
    )
    _write_json(run_dir / "summary_manifest.json", _render_manifest_summary(summary))
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run model recovery experiments")
    parser.add_argument("--snapshot-path", required=True, help="Path to a multi-symbol snapshot parquet")
    parser.add_argument("--output-root", default="models/experiments/model_recovery", help="Root directory for experiment runs")
    parser.add_argument("--docs-output-dir", default="docs/model_quality", help="Directory for latest markdown/json docs")
    parser.add_argument("--max-candidates", type=int, default=DEFAULT_MAX_CANDIDATES, help="Maximum candidate configs to evaluate")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed for deterministic tie breaks")
    parser.add_argument("--min-accuracy", type=float, default=DEFAULT_MIN_ACCURACY, help="Minimum holdout accuracy")
    parser.add_argument("--min-actionable-decisions", type=int, default=DEFAULT_MIN_ACTIONABLE_DECISIONS, help="Minimum actionable decisions on holdout")
    parser.add_argument("--max-drawdown-frac", type=float, default=DEFAULT_MAX_DRAWDOWN_FRAC, help="Maximum allowed drawdown fraction")
    parser.add_argument("--label-mode", choices=("directional_return", "trade_outcome"), default="directional_return", help="Label mode for recovery experiments")
    parser.add_argument("--trade-profit-target-bps", type=float, default=20.0, help="Trade-outcome profit target in bps")
    parser.add_argument("--trade-stop-loss-bps", type=float, default=30.0, help="Trade-outcome stop loss in bps")
    parser.add_argument("--trade-round-trip-cost-bps", type=float, default=8.0, help="Trade-outcome round-trip cost in bps")
    parser.add_argument("--phase4-research-input-repair", action="store_true", help="Run the bounded Phase 4 research-input repair suite")
    parser.add_argument("--no-production-registry", action="store_true", help="Safety flag preventing production registry writes")
    args = parser.parse_args(argv)

    if not args.no_production_registry:
        parser.error("--no-production-registry is required for this workflow")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
    dataset, _manifest = load_multi_symbol_snapshot(args.snapshot_path)
    if args.phase4_research_input_repair:
        result = run_phase4_research_input_repair(
            dataset,
            snapshot_path=Path(args.snapshot_path),
            output_root=Path(args.output_root),
            docs_output_dir=Path(args.docs_output_dir),
            seed=int(args.seed),
            min_accuracy=float(args.min_accuracy),
            min_actionable_decisions=int(args.min_actionable_decisions),
            max_drawdown_frac=float(args.max_drawdown_frac),
            label_mode=str(args.label_mode),
            trade_outcome_profit_target_bps=float(args.trade_profit_target_bps),
            trade_outcome_stop_loss_bps=float(args.trade_stop_loss_bps),
            trade_outcome_round_trip_cost_bps=float(args.trade_round_trip_cost_bps),
            max_variants=min(int(args.max_candidates), 4),
        )
        print(f"run_id={result.run_id}")
        print(f"output_dir={result.output_dir}")
        print(f"evaluated_variants={len(result.variant_evaluations)}")
        print(f"passed_variants={sum(1 for variant in result.variant_evaluations if variant.passed)}")
        print(f"recommendation={result.summary['recommendation']}")
        output = {
            "run_id": result.run_id,
            "output_dir": str(result.output_dir),
            "evaluated_variants": len(result.variant_evaluations),
            "passed_variants": sum(1 for variant in result.variant_evaluations if variant.passed),
            "recommendation": result.summary["recommendation"],
            "selected_variant_id": result.summary.get("selected_variant_id"),
            "selected_candidate_id": result.summary.get("selected_candidate_id"),
        }
        print(json.dumps(output, indent=2, sort_keys=True))
    else:
        result = run_model_recovery_experiments(
            dataset,
            snapshot_path=Path(args.snapshot_path),
            output_root=Path(args.output_root),
            docs_output_dir=Path(args.docs_output_dir),
            max_candidates=int(args.max_candidates),
            seed=int(args.seed),
            min_accuracy=float(args.min_accuracy),
            min_actionable_decisions=int(args.min_actionable_decisions),
            max_drawdown_frac=float(args.max_drawdown_frac),
            label_mode=str(args.label_mode),
            trade_outcome_profit_target_bps=float(args.trade_profit_target_bps),
            trade_outcome_stop_loss_bps=float(args.trade_stop_loss_bps),
            trade_outcome_round_trip_cost_bps=float(args.trade_round_trip_cost_bps),
            no_production_registry=bool(args.no_production_registry),
        )

        print(f"run_id={result.run_id}")
        print(f"output_dir={result.output_dir}")
        print(f"evaluated_candidates={len(result.evaluated_candidates)}")
        print(f"passed_candidates={len(result.passed_candidates)}")
        print(f"recommendation={result.summary['recommendation']}")
        output = {
            "run_id": result.run_id,
            "output_dir": str(result.output_dir),
            "evaluated_candidates": len(result.evaluated_candidates),
            "passed_candidates": len(result.passed_candidates),
            "recommendation": result.summary["recommendation"],
            "selected_candidate_id": result.summary.get("selected_candidate_id"),
        }
        print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
