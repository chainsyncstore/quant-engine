"""Group-aware multi-symbol validation runner for v2 research."""

from __future__ import annotations

from dataclasses import dataclass
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
    return add_regime_context_features(enriched)


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

    cfg = get_research_config()

    for fold_idx, split in enumerate(splits):
        fold_result, pnl = _run_single_split(
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
        if fold_result is None:
            continue

        fold_results.append(fold_result)
        if pnl is not None and len(pnl) > 0:
            pnl_parts.append(pnl)

    overall = aggregate_fold_metrics([f.metrics for f in fold_results])
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

    return GroupValidationResult(
        horizon=horizon,
        validation_mode="group_purged_cpcv",
        split_summary=split_summary_out,
        folds=fold_results,
        overall=overall,
        robustness=robustness,
        n_trials_assumed=n_trials_assumed,
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
) -> tuple[GroupValidationFoldResult | None, np.ndarray | None]:
    train_df = df.iloc[split.train_indices]
    test_df = df.iloc[split.test_indices]

    y_train = train_df[label_col]
    train_mask = y_train != -1
    if int(train_mask.sum()) < min_train_rows:
        return None, None

    X_train = train_df[feature_cols][train_mask]
    y_train_filtered = y_train[train_mask]

    trained = train_fn(
        X_train,
        y_train_filtered,
        horizon=int(label_col.removeprefix("label_").removesuffix("m")),
        params_override=params_override,
    )

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
        return None, None

    pnl = compute_trade_pnl(
        predictions=probas[valid_mask],
        actuals=y_test[valid_mask],
        price_moves=price_moves[valid_mask],
        threshold=threshold,
        spread=costs[valid_mask],
        allow_short=True,
    )

    train_ts = train_df.index.get_level_values("timestamp")
    test_ts = test_df.index.get_level_values("timestamp")

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
    return fold_result, pnl
