"""Trade-outcome label helpers for recovery and retrain research.

The helpers in this module turn a multi-symbol OHLC frame into explicit
trade-lifecycle labels. They are intentionally self-contained so they can be
reused by recovery experiments without pulling in execution/runtime code.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd

Side = Literal["long", "short"]


@dataclass(frozen=True)
class TradeOutcomeLabelConfig:
    horizon_bars: int = 4
    profit_target_bps: float = 20.0
    stop_loss_bps: float = 30.0
    dead_zone_bps: float = 5.0
    round_trip_cost_bps: float = 8.0
    funding_bps_per_8h: float = 0.0
    prefer_stop_on_same_bar: bool = True
    label_mode: str = "binary"


@dataclass(frozen=True)
class TradeOutcomeRecord:
    timestamp: str
    symbol: str
    side: str
    label: float
    gross_return_bps: float
    net_return_bps: float
    first_barrier: str
    holding_bars: int
    max_adverse_excursion_bps: float
    max_favorable_excursion_bps: float
    exit_price: float
    exit_timestamp: str | None


def _normalize_side(side: Side | str) -> Side:
    clean = str(side).strip().lower()
    if clean in {"long", "buy"}:
        return "long"
    if clean in {"short", "sell"}:
        return "short"
    raise ValueError(f"unsupported side={side!r}; expected 'long' or 'short'")


def _validate_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(frame.index, pd.MultiIndex) or list(frame.index.names) != ["timestamp", "symbol"]:
        raise ValueError("frame must be MultiIndex with levels ['timestamp', 'symbol']")
    timestamps = pd.DatetimeIndex(frame.index.get_level_values("timestamp"))
    if timestamps.tz is None:
        raise ValueError("timestamp level must be timezone-aware")
    if str(timestamps.tz) not in {"UTC", "UTC+00:00", "UTC-00:00"}:
        raise ValueError("timestamp level must be UTC")
    required = {"open", "high", "low", "close"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    ordered = frame.sort_index().copy()
    ordered.index = ordered.index.set_names(["timestamp", "symbol"])
    return ordered


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return float(default)
        if pd.isna(value):
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _row_cost_bps(row: pd.Series, *, holding_bars: int, config: TradeOutcomeLabelConfig) -> float:
    cost = abs(float(config.round_trip_cost_bps))
    cost += abs(float(config.funding_bps_per_8h)) * max(int(holding_bars), 0) / 8.0

    if "spread_bps" in row.index:
        cost += abs(_safe_float(row.get("spread_bps"), 0.0))

    for column in ("funding_rate_bps", "funding_bps"):
        if column in row.index:
            cost += _safe_float(row.get(column), 0.0) * max(int(holding_bars), 0) / 8.0
    return float(cost)


def _exit_price_from_bps(entry_price: float, *, side: Side, gross_return_bps: float) -> float:
    ratio = gross_return_bps / 10_000.0
    if side == "long":
        return float(entry_price * (1.0 + ratio))
    if ratio >= 0.0:
        return float(entry_price / (1.0 + ratio))
    return float(entry_price / (1.0 + ratio))


def _evaluate_trade_row(
    sym_frame: pd.DataFrame,
    idx: int,
    *,
    config: TradeOutcomeLabelConfig,
    side: Side,
) -> TradeOutcomeRecord:
    row = sym_frame.iloc[idx]
    entry_price = _safe_float(row["close"], 0.0)
    timestamp = pd.Timestamp(sym_frame.index.get_level_values("timestamp")[idx])
    symbol = str(sym_frame.index.get_level_values("symbol")[idx])
    future_limit = idx + int(config.horizon_bars)

    if future_limit >= len(sym_frame):
        return TradeOutcomeRecord(
            timestamp=str(timestamp),
            symbol=symbol,
            side=side,
            label=float("nan"),
            gross_return_bps=0.0,
            net_return_bps=0.0,
            first_barrier="insufficient_lookahead",
            holding_bars=0,
            max_adverse_excursion_bps=0.0,
            max_favorable_excursion_bps=0.0,
            exit_price=float("nan"),
            exit_timestamp=None,
        )

    future = sym_frame.iloc[idx + 1 : future_limit + 1]
    high = pd.to_numeric(future["high"], errors="coerce").to_numpy(dtype=float)
    low = pd.to_numeric(future["low"], errors="coerce").to_numpy(dtype=float)
    close = pd.to_numeric(future["close"], errors="coerce").to_numpy(dtype=float)

    if entry_price <= 0.0 or len(close) == 0:
        return TradeOutcomeRecord(
            timestamp=str(timestamp),
            symbol=symbol,
            side=side,
            label=float("nan"),
            gross_return_bps=0.0,
            net_return_bps=0.0,
            first_barrier="insufficient_lookahead",
            holding_bars=0,
            max_adverse_excursion_bps=0.0,
            max_favorable_excursion_bps=0.0,
            exit_price=float("nan"),
            exit_timestamp=None,
        )

    if side == "long":
        fav_bps = (high / entry_price - 1.0) * 10_000.0
        adv_bps = (low / entry_price - 1.0) * 10_000.0
        terminal_gross_bps = (close[-1] / entry_price - 1.0) * 10_000.0
        take_exit_price = _exit_price_from_bps(entry_price, side=side, gross_return_bps=float(config.profit_target_bps))
        stop_exit_price = _exit_price_from_bps(entry_price, side=side, gross_return_bps=-float(config.stop_loss_bps))
    else:
        fav_bps = (entry_price / low - 1.0) * 10_000.0
        adv_bps = (entry_price / high - 1.0) * 10_000.0
        terminal_gross_bps = (entry_price / close[-1] - 1.0) * 10_000.0
        take_exit_price = _exit_price_from_bps(entry_price, side=side, gross_return_bps=float(config.profit_target_bps))
        stop_exit_price = entry_price / (1.0 - float(config.stop_loss_bps) / 10_000.0)

    first_barrier = "time_exit"
    exit_price = float(close[-1])
    holding_bars = int(len(close))
    gross_return_bps = float(terminal_gross_bps)
    exit_timestamp = str(pd.Timestamp(future.index.get_level_values("timestamp")[-1]))

    for offset, (_, future_row) in enumerate(future.iterrows(), start=1):
        future_high = _safe_float(future_row["high"], np.nan)
        future_low = _safe_float(future_row["low"], np.nan)
        if side == "long":
            hit_tp = future_high >= entry_price * (1.0 + float(config.profit_target_bps) / 10_000.0)
            hit_sl = future_low <= entry_price * (1.0 - float(config.stop_loss_bps) / 10_000.0)
        else:
            hit_tp = future_low <= entry_price * (1.0 - float(config.profit_target_bps) / 10_000.0)
            hit_sl = future_high >= entry_price * (1.0 + float(config.stop_loss_bps) / 10_000.0)

        if hit_tp and hit_sl:
            if config.prefer_stop_on_same_bar:
                first_barrier = "same_bar_stop_loss"
                exit_price = float(stop_exit_price)
                gross_return_bps = -float(config.stop_loss_bps)
            else:
                first_barrier = "same_bar_take_profit"
                exit_price = float(take_exit_price)
                gross_return_bps = float(config.profit_target_bps)
            holding_bars = offset
            exit_timestamp = str(pd.Timestamp(future.index.get_level_values("timestamp")[offset - 1]))
            break

        if hit_sl:
            first_barrier = "stop_loss"
            exit_price = float(stop_exit_price)
            gross_return_bps = -float(config.stop_loss_bps)
            holding_bars = offset
            exit_timestamp = str(pd.Timestamp(future.index.get_level_values("timestamp")[offset - 1]))
            break

        if hit_tp:
            first_barrier = "take_profit"
            exit_price = float(take_exit_price)
            gross_return_bps = float(config.profit_target_bps)
            holding_bars = offset
            exit_timestamp = str(pd.Timestamp(future.index.get_level_values("timestamp")[offset - 1]))
            break

    cost_bps = _row_cost_bps(row, holding_bars=holding_bars, config=config)
    net_return_bps = float(gross_return_bps - cost_bps)

    if net_return_bps > float(config.dead_zone_bps):
        label = 1.0
    elif net_return_bps < -float(config.dead_zone_bps):
        label = 0.0
    else:
        label = float("nan")

    return TradeOutcomeRecord(
        timestamp=str(timestamp),
        symbol=symbol,
        side=side,
        label=label,
        gross_return_bps=float(gross_return_bps),
        net_return_bps=net_return_bps,
        first_barrier=first_barrier,
        holding_bars=int(holding_bars),
        max_adverse_excursion_bps=float(np.nanmin(adv_bps)) if len(adv_bps) else 0.0,
        max_favorable_excursion_bps=float(np.nanmax(fav_bps)) if len(fav_bps) else 0.0,
        exit_price=float(exit_price),
        exit_timestamp=exit_timestamp,
    )


def build_trade_outcome_labels(
    frame: pd.DataFrame,
    *,
    config: TradeOutcomeLabelConfig,
    side: Literal["long", "short"] | str = "long",
) -> pd.Series:
    ordered = _validate_frame(frame)
    normalized_side = _normalize_side(side)
    labels = pd.Series(np.nan, index=ordered.index, dtype=float)
    horizon = int(config.horizon_bars)
    if horizon <= 0:
        raise ValueError("horizon_bars must be positive")

    for _, sym_frame in ordered.groupby(level="symbol", sort=False):
        sym_frame = sym_frame.sort_index()
        n_rows = len(sym_frame)
        sym_values = np.full(n_rows, np.nan, dtype=float)
        if n_rows <= horizon:
            labels.loc[sym_frame.index] = pd.Series(sym_values, index=sym_frame.index, dtype=float)
            continue

        close = pd.to_numeric(sym_frame["close"], errors="coerce").to_numpy(dtype=float)
        high = pd.to_numeric(sym_frame["high"], errors="coerce").to_numpy(dtype=float)
        low = pd.to_numeric(sym_frame["low"], errors="coerce").to_numpy(dtype=float)
        row_numbers = np.arange(n_rows)
        valid = (row_numbers + horizon < n_rows) & (close > 0.0)
        valid_positions = np.flatnonzero(valid)
        if len(valid_positions) == 0:
            labels.loc[sym_frame.index] = pd.Series(sym_values, index=sym_frame.index, dtype=float)
            continue

        terminal_close = close[valid_positions + horizon]
        if normalized_side == "long":
            gross_return_bps = (terminal_close / close[valid_positions] - 1.0) * 10_000.0
        else:
            gross_return_bps = (close[valid_positions] / terminal_close - 1.0) * 10_000.0
        holding_bars = np.full(len(valid_positions), horizon, dtype=float)
        unresolved = np.ones(len(valid_positions), dtype=bool)
        position_lookup = np.full(n_rows, -1, dtype=int)
        position_lookup[valid_positions] = np.arange(len(valid_positions))

        take_bps = float(config.profit_target_bps)
        stop_bps = float(config.stop_loss_bps)
        for offset in range(1, horizon + 1):
            base_positions = np.flatnonzero(valid[: n_rows - offset])
            if len(base_positions) == 0:
                continue
            compact_positions = position_lookup[base_positions]
            active = compact_positions >= 0
            if not active.any():
                continue
            base_positions = base_positions[active]
            compact_positions = compact_positions[active]
            active_unresolved = unresolved[compact_positions]
            if not active_unresolved.any():
                continue
            base_positions = base_positions[active_unresolved]
            compact_positions = compact_positions[active_unresolved]

            entry = close[base_positions]
            future_high = high[base_positions + offset]
            future_low = low[base_positions + offset]
            if normalized_side == "long":
                hit_tp = future_high >= entry * (1.0 + take_bps / 10_000.0)
                hit_sl = future_low <= entry * (1.0 - stop_bps / 10_000.0)
            else:
                hit_tp = future_low <= entry * (1.0 - take_bps / 10_000.0)
                hit_sl = future_high >= entry * (1.0 + stop_bps / 10_000.0)

            if config.prefer_stop_on_same_bar:
                stop_mask = hit_sl
                take_mask = hit_tp & ~hit_sl
            else:
                take_mask = hit_tp
                stop_mask = hit_sl & ~hit_tp

            if stop_mask.any():
                target_positions = compact_positions[stop_mask]
                gross_return_bps[target_positions] = -stop_bps
                holding_bars[target_positions] = float(offset)
                unresolved[target_positions] = False
            if take_mask.any():
                target_positions = compact_positions[take_mask]
                gross_return_bps[target_positions] = take_bps
                holding_bars[target_positions] = float(offset)
                unresolved[target_positions] = False

        cost_bps = np.full(len(valid_positions), abs(float(config.round_trip_cost_bps)), dtype=float)
        cost_bps += abs(float(config.funding_bps_per_8h)) * holding_bars / 8.0
        if "spread_bps" in sym_frame.columns:
            spread = pd.to_numeric(sym_frame["spread_bps"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            cost_bps += np.abs(spread[valid_positions])
        for column in ("funding_rate_bps", "funding_bps"):
            if column in sym_frame.columns:
                funding = pd.to_numeric(sym_frame[column], errors="coerce").fillna(0.0).to_numpy(dtype=float)
                cost_bps += funding[valid_positions] * holding_bars / 8.0

        net_return_bps = gross_return_bps - cost_bps
        sym_values[valid_positions[net_return_bps > float(config.dead_zone_bps)]] = 1.0
        sym_values[valid_positions[net_return_bps < -float(config.dead_zone_bps)]] = 0.0
        sym_labels = pd.Series(sym_values, index=sym_frame.index, dtype=float)
        labels.loc[sym_frame.index] = sym_labels

    return labels.sort_index()


def _side_report(
    frame: pd.DataFrame,
    *,
    config: TradeOutcomeLabelConfig,
    side: Side,
) -> dict[str, Any]:
    labels = build_trade_outcome_labels(frame, config=config, side=side)
    ordered = _validate_frame(frame)
    records: list[TradeOutcomeRecord] = []
    for _, sym_frame in ordered.groupby(level="symbol", sort=False):
        sym_frame = sym_frame.sort_index()
        for idx in range(len(sym_frame)):
            records.append(_evaluate_trade_row(sym_frame, idx, config=config, side=side))

    valid_labels = labels.notna()
    take_count = int((labels == 1).sum())
    skip_count = int((labels == 0).sum())
    ambiguous_count = int((~valid_labels).sum())
    net_returns = [record.net_return_bps for record in records if not np.isnan(record.label)]
    holding_bars = [record.holding_bars for record in records if not np.isnan(record.label)]
    mae = [record.max_adverse_excursion_bps for record in records if not np.isnan(record.label)]
    mfe = [record.max_favorable_excursion_bps for record in records if not np.isnan(record.label)]
    barrier_counts: dict[str, int] = {}
    by_symbol: dict[str, Any] = {}
    for record in records:
        barrier_counts[record.first_barrier] = barrier_counts.get(record.first_barrier, 0) + 1
        bucket = by_symbol.setdefault(
            record.symbol,
            {
                "rows": 0,
                "take_count": 0,
                "skip_count": 0,
                "ambiguous_count": 0,
                "net_return_bps": [],
                "first_barriers": {},
            },
        )
        bucket["rows"] += 1
        bucket["first_barriers"][record.first_barrier] = bucket["first_barriers"].get(record.first_barrier, 0) + 1
        if np.isnan(record.label):
            bucket["ambiguous_count"] += 1
        elif record.label == 1.0:
            bucket["take_count"] += 1
            bucket["net_return_bps"].append(record.net_return_bps)
        else:
            bucket["skip_count"] += 1
            bucket["net_return_bps"].append(record.net_return_bps)

    for symbol, bucket in by_symbol.items():
        returns = bucket.pop("net_return_bps")
        bucket["net_return_bps"] = {
            "mean": float(np.mean(returns)) if returns else 0.0,
            "median": float(np.median(returns)) if returns else 0.0,
        }

    dataset_timestamps = pd.DatetimeIndex(ordered.index.get_level_values("timestamp"))
    return {
        "label_counts": {
            "take": take_count,
            "skip": skip_count,
            "ambiguous": ambiguous_count,
            "labelled": int(valid_labels.sum()),
        },
        "barrier_counts": barrier_counts,
        "net_return_bps": {
            "mean": float(np.mean(net_returns)) if net_returns else 0.0,
            "median": float(np.median(net_returns)) if net_returns else 0.0,
            "p25": float(np.quantile(net_returns, 0.25)) if net_returns else 0.0,
            "p75": float(np.quantile(net_returns, 0.75)) if net_returns else 0.0,
        },
        "holding_bars": {
            "mean": float(np.mean(holding_bars)) if holding_bars else 0.0,
            "median": float(np.median(holding_bars)) if holding_bars else 0.0,
        },
        "mae_bps": {
            "mean": float(np.mean(mae)) if mae else 0.0,
            "p95": float(np.quantile(mae, 0.95)) if mae else 0.0,
        },
        "mfe_bps": {
            "mean": float(np.mean(mfe)) if mfe else 0.0,
            "p95": float(np.quantile(mfe, 0.95)) if mfe else 0.0,
        },
        "by_symbol": by_symbol,
        "dataset": {
            "rows": int(len(ordered)),
            "symbols": sorted(str(symbol) for symbol in ordered.index.get_level_values("symbol").unique()),
            "start": str(dataset_timestamps.min()) if len(dataset_timestamps) else None,
            "end": str(dataset_timestamps.max()) if len(dataset_timestamps) else None,
        },
    }


def build_trade_outcome_report(
    frame: pd.DataFrame,
    *,
    config: TradeOutcomeLabelConfig,
) -> dict[str, Any]:
    ordered = _validate_frame(frame)
    long_report = _side_report(ordered, config=config, side="long")
    short_report = _side_report(ordered, config=config, side="short")
    return {
        "policy_version": "trade_outcome_labels_v1",
        "config": asdict(config),
        "dataset": long_report["dataset"],
        "long": long_report,
        "short": short_report,
    }
