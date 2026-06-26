"""Group-aware multi-symbol validation runner for v2 research."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd

from quant.config import get_research_config
from quant.data.session_filter import filter_sessions
from quant.features.pipeline import build_features, get_feature_columns
from quant.labels.labeler import add_labels
from quant.risk.cost_model import PercentageCostModel
from quant.validation.metrics import (
    FoldMetrics,
    aggregate_fold_metrics,
    compute_metrics,
    compute_trade_pnl,
    deflated_sharpe_ratio,
    probabilistic_sharpe_ratio,
)
from quant_v2.models import trainer as model_trainer
from quant_v2.models.predictor import predict_proba
from quant_v2.research.cross_sectional_features import add_cross_sectional_features
from quant_v2.research.event_labels import apply_event_aware_label_filters
from quant_v2.research.regime_context import add_regime_context_features
from quant_v2.validation.purged_group_cpcv import (
    PurgedGroupSplit,
    iter_purged_group_splits,
    summarize_split_coverage,
)


@dataclass(frozen=True)
class GroupValidationFoldResult:
    """One completed group-purged fold result."""

    split_id: str
    n_train_rows: int
    n_test_rows: int
    n_valid_rows: int
    test_symbols: tuple[str, ...]
    metrics: FoldMetrics


@dataclass(frozen=True)
class GroupValidationResult:
    """Aggregate output from group-purged validation."""

    horizon: int
    validation_mode: str
    split_summary: dict[str, int]
    folds: list[GroupValidationFoldResult]
    overall: dict[str, Any]
    robustness: dict[str, float]
    n_trials_assumed: int
    trial_count: int = 0
    fold_diagnostics: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    fold_dispersion: dict[str, Any] = field(default_factory=dict)
    selection_risk: dict[str, Any] = field(default_factory=dict)
    cost_sensitivity: dict[str, Any] = field(default_factory=dict)
    failure_reasons: tuple[str, ...] = field(default_factory=tuple)


def _metric_summary(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return {"count": 0, "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "median": 0.0}
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "median": float(np.median(arr)),
    }


def _summarize_fold_dispersion(folds: list[GroupValidationFoldResult]) -> dict[str, Any]:
    if not folds:
        return {
            "spread_adjusted_ev": _metric_summary([]),
            "win_rate": _metric_summary([]),
            "sharpe": _metric_summary([]),
            "max_drawdown": _metric_summary([]),
            "max_drawdown_duration": _metric_summary([]),
            "n_trades": _metric_summary([]),
        }

    return {
        "spread_adjusted_ev": _metric_summary([fold.metrics.spread_adjusted_ev for fold in folds]),
        "win_rate": _metric_summary([fold.metrics.win_rate for fold in folds]),
        "sharpe": _metric_summary([fold.metrics.sharpe for fold in folds]),
        "max_drawdown": _metric_summary([fold.metrics.max_drawdown for fold in folds]),
        "max_drawdown_duration": _metric_summary([float(fold.metrics.max_drawdown_duration) for fold in folds]),
        "n_trades": _metric_summary([float(fold.metrics.n_trades) for fold in folds]),
    }


def _summarize_selection_risk(
    folds: list[GroupValidationFoldResult],
    *,
    n_trials_assumed: int,
) -> dict[str, float]:
    scores = [float(fold.metrics.spread_adjusted_ev) for fold in folds if fold.metrics.n_trades > 0]
    if not scores:
        return {
            "pbo_equivalent": 0.0,
            "selected_fold_rank_percentile": 1.0,
            "n_folds": 0.0,
            "n_trials_assumed": float(max(1, n_trials_assumed)),
        }

    selected_score = max(scores)
    sorted_scores = sorted(scores)
    rank = sum(score <= selected_score for score in sorted_scores)
    percentile = rank / max(len(sorted_scores), 1)
    return {
        "pbo_equivalent": float(max(0.0, 1.0 - percentile)),
        "selected_fold_rank_percentile": float(min(max(percentile, 0.0), 1.0)),
        "n_folds": float(len(scores)),
        "n_trials_assumed": float(max(1, n_trials_assumed)),
    }


def _summarize_cost_sensitivity(fold_diagnostics: tuple[dict[str, Any], ...]) -> dict[str, Any]:
    scenarios: dict[str, list[dict[str, float]]] = {}
    for diag in fold_diagnostics:
        cost_sensitivity = diag.get("cost_sensitivity")
        if not isinstance(cost_sensitivity, dict):
            continue
        for scenario_name, summary in cost_sensitivity.items():
            if not isinstance(summary, dict):
                continue
            scenarios.setdefault(str(scenario_name), []).append(
                {
                    "spread_adjusted_ev": float(summary.get("spread_adjusted_ev", 0.0)),
                    "total_return": float(summary.get("total_return", 0.0)),
                    "win_rate": float(summary.get("win_rate", 0.0)),
                }
            )

    if not scenarios:
        return {}

    return {
        scenario: {
            "spread_adjusted_ev_mean": float(np.mean([entry["spread_adjusted_ev"] for entry in entries])),
            "total_return_mean": float(np.mean([entry["total_return"] for entry in entries])),
            "win_rate_mean": float(np.mean([entry["win_rate"] for entry in entries])),
        }
        for scenario, entries in sorted(scenarios.items())
    }


def _summarize_failure_reasons(fold_diagnostics: tuple[dict[str, Any], ...]) -> tuple[str, ...]:
    reasons: list[str] = []
    for diag in fold_diagnostics:
        for reason in diag.get("failure_reasons", []) or []:
            reasons.append(str(reason))
    return tuple(sorted(set(reasons)))


def prepare_multi_symbol_dataset(
    raw_df: pd.DataFrame,
    *,
    horizons: tuple[int, ...] | list[int],
) -> pd.DataFrame:
    """Build per-symbol features/labels and return canonical MultiIndex dataset."""

    if not isinstance(raw_df.index, pd.MultiIndex) or list(raw_df.index.names) != ["timestamp", "symbol"]:
        raise ValueError("raw_df must be MultiIndex with levels ['timestamp', 'symbol']")

    horizon_list = [int(h) for h in horizons]
    if not horizon_list:
        raise ValueError("horizons cannot be empty")

    pieces: list[pd.DataFrame] = []

    for symbol, sym_df in raw_df.groupby(level="symbol", sort=False):
        per_symbol = sym_df.droplevel("symbol")
        if per_symbol.empty:
            continue

        filtered = filter_sessions(per_symbol)
        if filtered.empty:
            continue

        featured = build_features(filtered)
        if featured.empty:
            continue

        labeled = add_labels(featured, horizons=horizon_list)
        if labeled.empty:
            continue

        labeled = apply_event_aware_label_filters(
            labeled,
            horizons=horizon_list,
        )

        labeled = labeled.copy()
        labeled["symbol"] = str(symbol)
        labeled = labeled.reset_index().set_index(["timestamp", "symbol"]).sort_index()
        pieces.append(labeled)

    if not pieces:
        raise RuntimeError("No symbol datasets produced after feature/label preparation")

    combined = pd.concat(pieces).sort_index()
    combined.index = combined.index.set_names(["timestamp", "symbol"])
    enriched = add_cross_sectional_features(combined)
    enriched = add_regime_context_features(enriched)
    context_cols = [c for c in enriched.columns if c.startswith("xs_") or c.startswith("regime_")]
    if context_cols:
        missing_context = enriched[context_cols].isna().any(axis=1)
        if missing_context.any():
            enriched = enriched.loc[~missing_context].copy()
    return enriched


def run_group_purged_validation(
    df: pd.DataFrame,
    *,
    horizon: int,
    threshold: float = 0.5,
    n_time_splits: int = 5,
    symbol_cluster_size: int = 2,
    embargo_bars: int = 24,
    min_train_rows: int = 200,
    params_override: dict[str, Any] | None = None,
    precomputed_splits: list[PurgedGroupSplit] | None = None,
    split_summary: dict[str, int] | None = None,
    train_fn: Callable[..., Any] | None = None,
    predict_fn: Callable[..., np.ndarray] | None = None,
) -> GroupValidationResult:
    """Execute group-aware purged validation on a prepared multi-symbol dataset."""

    if not isinstance(df.index, pd.MultiIndex) or list(df.index.names) != ["timestamp", "symbol"]:
        raise ValueError("df must be MultiIndex with levels ['timestamp', 'symbol']")

    label_col = f"label_{horizon}m"
    if label_col not in df.columns:
        raise ValueError(f"Missing label column: {label_col}")

    feature_cols = get_feature_columns(df)
    if not feature_cols:
        raise ValueError("No feature columns available for validation")

    if precomputed_splits is None:
        splits = iter_purged_group_splits(
            df,
            n_time_splits=n_time_splits,
            symbol_cluster_size=symbol_cluster_size,
            embargo_bars=embargo_bars,
            min_train_rows=min_train_rows,
        )
    else:
        splits = list(precomputed_splits)
        if not splits:
            raise ValueError("precomputed_splits cannot be empty")

        n_rows = len(df)
        for split in splits:
            if split.train_indices.size:
                if int(split.train_indices.min()) < 0 or int(split.train_indices.max()) >= n_rows:
                    raise ValueError(f"Invalid train indices in split {split.split_id} for dataset with {n_rows} rows")
            if split.test_indices.size:
                if int(split.test_indices.min()) < 0 or int(split.test_indices.max()) >= n_rows:
                    raise ValueError(f"Invalid test indices in split {split.split_id} for dataset with {n_rows} rows")

    split_summary_out = dict(split_summary) if split_summary is not None else summarize_split_coverage(splits)

    train_callable = train_fn or model_trainer.train
    predict_callable = predict_fn or predict_proba

    close = df["close"]
    price_moves_all = (close.groupby(level="symbol").shift(-horizon) - close).to_numpy()

    fold_results: list[GroupValidationFoldResult] = []
    pnl_parts: list[np.ndarray] = []
    fold_diagnostics: list[dict[str, Any]] = []

    cfg = get_research_config()

    for fold_idx, split in enumerate(splits):
        fold_result, pnl, diagnostics = _run_single_split(
            df,
            split=split,
            fold_idx=fold_idx,
            feature_cols=feature_cols,
            label_col=label_col,
            price_moves_all=price_moves_all,
            threshold=threshold,
            min_train_rows=min_train_rows,
            fee_rate=cfg.taker_fee_rate,
            params_override=params_override,
            train_fn=train_callable,
            predict_fn=predict_callable,
        )
        fold_diagnostics.append(diagnostics)
        if fold_result is None:
            continue

        fold_results.append(fold_result)
        if pnl is not None and len(pnl) > 0:
            pnl_parts.append(pnl)

    overall = aggregate_fold_metrics([f.metrics for f in fold_results])
    total_return = float(np.sum(np.concatenate(pnl_parts))) if pnl_parts else 0.0
    overall = {
        **overall,
        "total_return": total_return,
        "net_expectancy": float(overall.get("spread_adjusted_ev", 0.0)),
        "time_under_water": int(overall.get("max_drawdown_duration", 0)),
    }
    all_pnl = np.concatenate(pnl_parts) if pnl_parts else np.array([])

    n_trials_assumed = max(1, len(splits) * 7)
    if len(all_pnl) >= 2:
        psr = probabilistic_sharpe_ratio(all_pnl)
        dsr = deflated_sharpe_ratio(all_pnl, n_trials=n_trials_assumed)
    else:
        psr = 0.5
        dsr = 0.5

    robustness = {
        "probabilistic_sharpe_ratio": float(psr),
        "deflated_sharpe_ratio": float(dsr),
    }

    fold_diagnostics_tuple = tuple(fold_diagnostics)
    fold_dispersion = _summarize_fold_dispersion(fold_results)
    selection_risk = _summarize_selection_risk(fold_results, n_trials_assumed=n_trials_assumed)
    cost_sensitivity = _summarize_cost_sensitivity(fold_diagnostics_tuple)
    failure_reasons = list(_summarize_failure_reasons(fold_diagnostics_tuple))
    if not fold_results:
        failure_reasons.append("no_valid_folds")
    if len(all_pnl) == 0:
        failure_reasons.append("insufficient_trade_sample")

    return GroupValidationResult(
        horizon=horizon,
        validation_mode="group_purged_cpcv",
        split_summary=split_summary_out,
        folds=fold_results,
        overall=overall,
        robustness=robustness,
        n_trials_assumed=n_trials_assumed,
        trial_count=int(len(fold_results)),
        fold_diagnostics=fold_diagnostics_tuple,
        fold_dispersion=fold_dispersion,
        selection_risk=selection_risk,
        cost_sensitivity=cost_sensitivity,
        failure_reasons=tuple(sorted(set(failure_reasons))),
    )


def _run_single_split(
    df: pd.DataFrame,
    *,
    split: PurgedGroupSplit,
    fold_idx: int,
    feature_cols: list[str],
    label_col: str,
    price_moves_all: np.ndarray,
    threshold: float,
    min_train_rows: int,
    fee_rate: float,
    params_override: dict[str, Any] | None,
    train_fn: Callable[..., Any],
    predict_fn: Callable[..., np.ndarray],
) -> tuple[GroupValidationFoldResult | None, np.ndarray | None, dict[str, Any]]:
    train_df = df.iloc[split.train_indices]
    test_df = df.iloc[split.test_indices]
    diagnostics: dict[str, Any] = {
        "split_id": split.split_id,
        "test_symbols": list(split.test_symbols),
        "failure_reasons": [],
    }

    y_train = train_df[label_col]
    train_mask = y_train != -1
    if int(train_mask.sum()) < min_train_rows:
        diagnostics["failure_reasons"].append("insufficient_train_rows")
        return None, None, diagnostics

    X_train = train_df[feature_cols][train_mask]
    y_train_filtered = y_train[train_mask]
    if int(y_train_filtered.nunique(dropna=True)) < 2:
        diagnostics["failure_reasons"].append("class_collapse_train")

    try:
        trained = train_fn(
            X_train,
            y_train_filtered,
            horizon=int(label_col.removeprefix("label_").removesuffix("m")),
            params_override=params_override,
        )
    except Exception as exc:
        diagnostics["failure_reasons"].append(f"train_failed:{type(exc).__name__}")
        return None, None, diagnostics

    X_test = test_df[feature_cols]
    probas = np.asarray(predict_fn(trained, X_test), dtype=float)

    y_test = test_df[label_col].to_numpy()
    price_moves = price_moves_all[split.test_indices]

    cost_model = PercentageCostModel(fee_rate=fee_rate)
    cost_model.fit(train_df)
    costs = test_df.apply(cost_model.estimate_cost, axis=1).to_numpy()

    valid_mask = (y_test != -1) & np.isfinite(price_moves)
    n_valid = int(valid_mask.sum())
    if n_valid == 0:
        diagnostics["failure_reasons"].append("empty_validation_fold")
        return None, None, diagnostics

    valid_predictions = probas[valid_mask]
    valid_actuals = y_test[valid_mask]
    valid_moves = price_moves[valid_mask]
    valid_costs = costs[valid_mask]
    pnl = compute_trade_pnl(
        predictions=valid_predictions,
        actuals=valid_actuals,
        price_moves=valid_moves,
        threshold=threshold,
        spread=valid_costs,
        allow_short=True,
    )

    train_ts = train_df.index.get_level_values("timestamp")
    test_ts = test_df.index.get_level_values("timestamp")
    symbol_counts = test_df.index.get_level_values("symbol").value_counts(normalize=True)
    class_counts = pd.Series(valid_actuals).value_counts(normalize=True)
    cost_sensitivity = {
        "base": _summarize_trade_scenario(pnl),
        "adverse": _summarize_trade_scenario(
            compute_trade_pnl(
                predictions=valid_predictions,
                actuals=valid_actuals,
                price_moves=valid_moves,
                threshold=threshold,
                spread=valid_costs * 1.25,
                allow_short=True,
            )
        ),
        "severe": _summarize_trade_scenario(
            compute_trade_pnl(
                predictions=valid_predictions,
                actuals=valid_actuals,
                price_moves=valid_moves,
                threshold=threshold,
                spread=valid_costs * 1.5,
                allow_short=True,
            )
        ),
    }
    if class_counts.empty or len(class_counts.index) < 2:
        diagnostics["failure_reasons"].append("class_collapse_validation")
    if not symbol_counts.empty and float(symbol_counts.max()) > 0.9:
        diagnostics["failure_reasons"].append("symbol_concentration_too_high")

    metrics = compute_metrics(
        pnl=pnl,
        fold=fold_idx,
        train_start=str(train_ts.min()),
        test_start=str(test_ts.min()),
        test_end=str(test_ts.max()),
    )

    fold_result = GroupValidationFoldResult(
        split_id=split.split_id,
        n_train_rows=int(len(train_df)),
        n_test_rows=int(len(test_df)),
        n_valid_rows=n_valid,
        test_symbols=split.test_symbols,
        metrics=metrics,
    )
    diagnostics["cost_sensitivity"] = cost_sensitivity
    diagnostics["symbol_concentration"] = float(symbol_counts.max()) if not symbol_counts.empty else 0.0
    diagnostics["class_balance"] = {
        str(label): float(frac) for label, frac in class_counts.to_dict().items()
    }
    diagnostics["valid_trades"] = int(n_valid)
    diagnostics["failure_reasons"] = tuple(diagnostics["failure_reasons"])
    return fold_result, pnl, diagnostics


def _summarize_trade_scenario(pnl: np.ndarray) -> dict[str, float]:
    pnl = np.asarray(pnl, dtype=float)
    return {
        "spread_adjusted_ev": float(np.mean(pnl)) if pnl.size else 0.0,
        "total_return": float(np.sum(pnl)) if pnl.size else 0.0,
        "win_rate": float(np.mean(pnl > 0.0)) if pnl.size else 0.0,
        "n_trades": float(pnl.size),
    }
