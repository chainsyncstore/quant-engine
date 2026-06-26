"""Model-quality recovery diagnostics, benchmark replay, and report writing.

This module collects the evidence needed to recover from a failed retrain
without relaxing production gates. It intentionally reuses the existing
replay, accounting, and label-generation code paths so the reports reflect
the same runtime contracts the bot depends on.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from quant.data.binance_client import BinanceClient
from quant_v2.contracts import StrategySignal
from quant_v2.execution.cost_policy import ExecutionCostPolicy
from quant_v2.research.portfolio_replay import ReplayActorConfig, ReplayScenario, run_portfolio_replay
from quant_v2.portfolio.cost_model import BinanceCostModel

QUALITY_RECOVERY_POLICY_VERSION = "wp13-model-quality-recovery-v1"
VALIDATION_POLICY_VERSION = "model_quality_validation_policy_v1"

DEFAULT_HORIZONS: tuple[int, ...] = (2, 4, 8)
DEFAULT_DEAD_ZONES: tuple[float, ...] = (0.001, 0.0015, 0.002, 0.003, 0.005)
DEFAULT_TRAINING_WINDOWS_MONTHS: tuple[int, ...] = (3, 6, 9, 12)
DEFAULT_RECENCY_HALF_LIFES_DAYS: tuple[int, ...] = (30, 60, 90)
DEFAULT_BENCHMARK_NAMES: tuple[str, ...] = (
    "flat",
    "momentum",
    "mean_reversion",
    "volatility_filtered",
)


@dataclass(frozen=True)
class QualityRecoveryBundle:
    """Container for the generated recovery reports."""

    diagnostic: dict[str, Any]
    label_audit: dict[str, Any]
    benchmark_replay: dict[str, Any]
    candidate_selection: dict[str, Any]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def _sha256_payload(payload: Any) -> str:
    import hashlib

    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        if isinstance(value, bool):
            return float(value)
        if isinstance(value, (int, float, np.floating, np.integer)):
            value = float(value)
            if math.isnan(value):
                return default
            return value
        value = float(str(value))
        if math.isnan(value):
            return default
        return value
    except (TypeError, ValueError):
        return default


def _build_labels(frame: pd.DataFrame, horizon: int, dead_zone: float = 0.002) -> pd.Series:
    """Local label helper to avoid an import cycle with scheduled retraining."""

    labels = pd.Series(np.nan, index=frame.index)

    if isinstance(frame.index, pd.MultiIndex) and "symbol" in frame.index.names:
        grouped = frame.groupby(level="symbol", sort=False)
        for _, sym_frame in grouped:
            close = pd.to_numeric(sym_frame["close"], errors="coerce")
            future_return = close.shift(-horizon) / close - 1.0
            sym_labels = pd.Series(np.nan, index=sym_frame.index)
            sym_labels[future_return > dead_zone] = 1
            sym_labels[future_return < -dead_zone] = 0
            labels.loc[sym_frame.index] = sym_labels
        return labels

    close = pd.to_numeric(frame["close"], errors="coerce")
    future_return = close.shift(-horizon) / close - 1.0
    labels[future_return > dead_zone] = 1
    labels[future_return < -dead_zone] = 0
    return labels


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if isinstance(payload, dict):
        return payload
    return None


def _latest_failed_record(model_root: Path) -> tuple[Path | None, dict[str, Any] | None]:
    failed_dir = model_root / ".failed"
    if not failed_dir.exists():
        return None, None
    candidates = sorted(failed_dir.glob("model_*.json"))
    if not candidates:
        return None, None
    path = candidates[-1]
    return path, _load_json(path)


def build_failed_retrain_diagnostic(
    model_root: Path | str,
    registry_root: Path | str,
    *,
    failed_record_path: Path | str | None = None,
) -> dict[str, Any]:
    """Summarize the latest failed retrain plus registry/file-system state."""

    model_root = Path(model_root).expanduser()
    registry_root = Path(registry_root).expanduser()

    failure_path = Path(failed_record_path).expanduser() if failed_record_path else None
    failure_record: dict[str, Any] | None = _load_json(failure_path) if failure_path else None
    if failure_record is None:
        failure_path, failure_record = _latest_failed_record(model_root)

    active_path = registry_root / "active.json"
    registry_events_path = registry_root / "registry_events.jsonl"
    active_pointer = _load_json(active_path)
    registry_event_count = 0
    if registry_events_path.exists():
        registry_event_count = len(registry_events_path.read_text(encoding="utf-8").splitlines())

    visible_models = sorted(
        p.name for p in model_root.glob("model_*") if p.is_dir()
    )
    building_models = []
    building_root = model_root / ".building"
    if building_root.exists():
        building_models = sorted(p.name for p in building_root.glob("model_*") if p.is_dir())
    failed_models = []
    failed_root = model_root / ".failed"
    if failed_root.exists():
        failed_models = sorted(p.name for p in failed_root.glob("model_*.json"))

    failed_reason = None
    failed_details = {}
    if failure_record:
        failed_reason = str(failure_record.get("reason") or "")
        details = failure_record.get("details") or {}
        if isinstance(details, dict):
            failed_details = dict(details)

    return {
        "policy_version": QUALITY_RECOVERY_POLICY_VERSION,
        "generated_at": _utc_now(),
        "model_root": str(model_root),
        "registry_root": str(registry_root),
        "filesystem": {
            "visible_models": visible_models,
            "building_models": building_models,
            "failed_models": failed_models,
            "visible_model_count": len(visible_models),
            "building_model_count": len(building_models),
            "failed_record_count": len(failed_models),
        },
        "registry": {
            "active_pointer": active_pointer,
            "registry_event_count": registry_event_count,
            "registry_events_path": str(registry_events_path),
            "active_path": str(active_path),
        },
        "failure_record": {
            "path": str(failure_path) if failure_path else None,
            "reason": failed_reason,
            "details": failed_details,
        },
        "recommended_next_step": (
            "keep_no_trade_and_rebuild_candidate_evidence"
            if failed_reason
            else "no_failed_retrain_record_found"
        ),
    }


def _horizon_label_summary(frame: pd.DataFrame, horizon: int, dead_zone: float) -> dict[str, Any]:
    labels = _build_labels(frame, horizon=horizon, dead_zone=dead_zone)
    close = pd.to_numeric(frame["close"], errors="coerce")
    future_return = close.shift(-horizon) / close - 1.0
    valid = labels.notna()

    return {
        "horizon": int(horizon),
        "dead_zone": float(dead_zone),
        "rows": int(len(frame)),
        "labelled_rows": int(valid.sum()),
        "up_count": int((labels == 1).sum()),
        "down_count": int((labels == 0).sum()),
        "ambiguous_count": int(labels.isna().sum()),
        "up_ratio": float((labels == 1).mean()) if len(labels) else 0.0,
        "down_ratio": float((labels == 0).mean()) if len(labels) else 0.0,
        "ambiguous_ratio": float(labels.isna().mean()) if len(labels) else 0.0,
        "future_return_bps": {
            "mean": float(future_return.dropna().mean() * 10_000.0) if future_return.notna().any() else 0.0,
            "median": float(future_return.dropna().median() * 10_000.0) if future_return.notna().any() else 0.0,
            "p05": float(future_return.dropna().quantile(0.05) * 10_000.0) if future_return.notna().any() else 0.0,
            "p95": float(future_return.dropna().quantile(0.95) * 10_000.0) if future_return.notna().any() else 0.0,
        },
    }


def _monthly_label_summary(labels: pd.Series, raw: pd.DataFrame) -> dict[str, Any]:
    if not isinstance(raw.index, pd.MultiIndex) or "symbol" not in raw.index.names:
        return {}

    timestamps = pd.DatetimeIndex(raw.index.get_level_values("timestamp"))
    symbols = raw.index.get_level_values("symbol").astype(str)
    months = pd.PeriodIndex(timestamps, freq="M")
    df = pd.DataFrame({"label": labels.to_numpy(), "symbol": symbols.to_numpy(), "month": months.to_numpy()})

    out: dict[str, Any] = {}
    for (symbol, month), group in df.groupby(["symbol", "month"], sort=True):
        key = f"{symbol}:{month}"
        out[key] = {
            "rows": int(len(group)),
            "labelled_rows": int(group["label"].notna().sum()),
            "up_count": int((group["label"] == 1).sum()),
            "down_count": int((group["label"] == 0).sum()),
            "ambiguous_count": int(group["label"].isna().sum()),
        }
    return out


def _per_symbol_counts(frame: pd.DataFrame, horizon: int, dead_zone: float) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if not isinstance(frame.index, pd.MultiIndex) or "symbol" not in frame.index.names:
        return out
    grouped = frame.groupby(level="symbol", sort=True)
    for symbol, sym_frame in grouped:
        labels = _build_labels(sym_frame, horizon=horizon, dead_zone=dead_zone)
        out[str(symbol)] = {
            "rows": int(len(sym_frame)),
            "labelled_rows": int(labels.notna().sum()),
            "up_count": int((labels == 1).sum()),
            "down_count": int((labels == 0).sum()),
            "ambiguous_count": int(labels.isna().sum()),
            "ambiguous_ratio": float(labels.isna().mean()) if len(labels) else 0.0,
        }
    return out


def build_label_audit_report(
    dataset: pd.DataFrame,
    *,
    horizons: Iterable[int] = DEFAULT_HORIZONS,
    dead_zones: Iterable[float] = DEFAULT_DEAD_ZONES,
    training_windows_months: Iterable[int] = DEFAULT_TRAINING_WINDOWS_MONTHS,
    recency_half_lives_days: Iterable[int] = DEFAULT_RECENCY_HALF_LIFES_DAYS,
) -> dict[str, Any]:
    """Build a label-balance and dead-zone audit for a multi-symbol dataset."""

    if not isinstance(dataset.index, pd.MultiIndex) or list(dataset.index.names) != ["timestamp", "symbol"]:
        raise ValueError("dataset must be MultiIndex with levels ['timestamp', 'symbol']")
    if "close" not in dataset.columns:
        raise ValueError("dataset must include a close column")

    close = pd.to_numeric(dataset["close"], errors="coerce")
    if close.isna().all():
        raise ValueError("dataset close column is empty after coercion")

    horizons = tuple(int(h) for h in horizons)
    dead_zones = tuple(float(v) for v in dead_zones)
    training_windows_months = tuple(int(v) for v in training_windows_months)
    recency_half_lives_days = tuple(int(v) for v in recency_half_lives_days)

    per_horizon: dict[str, Any] = {}
    for horizon in horizons:
        dead_zone_summaries: dict[str, Any] = {}
        for dead_zone in dead_zones:
            labels = _build_labels(dataset, horizon=horizon, dead_zone=dead_zone)
            dead_zone_summaries[f"{dead_zone:.4f}"] = {
                **_horizon_label_summary(dataset, horizon, dead_zone),
                "by_symbol": _per_symbol_counts(dataset, horizon, dead_zone),
                "by_symbol_month": _monthly_label_summary(labels, dataset),
            }
        per_horizon[str(horizon)] = {
            "dead_zone_summaries": dead_zone_summaries,
            "selected_cost_floor_bps": _estimated_cost_floor_bps(dataset),
        }

    return {
        "policy_version": QUALITY_RECOVERY_POLICY_VERSION,
        "generated_at": _utc_now(),
        "grid": {
            "horizons": list(horizons),
            "dead_zones": list(dead_zones),
            "training_windows_months": list(training_windows_months),
            "recency_half_lives_days": list(recency_half_lives_days),
        },
        "dataset": {
            "rows": int(len(dataset)),
            "symbols": sorted(str(s) for s in dataset.index.get_level_values("symbol").unique()),
            "start": str(pd.DatetimeIndex(dataset.index.get_level_values("timestamp")).min()),
            "end": str(pd.DatetimeIndex(dataset.index.get_level_values("timestamp")).max()),
        },
        "horizons": per_horizon,
    }


def _estimated_cost_floor_bps(dataset: pd.DataFrame) -> dict[str, Any]:
    if not isinstance(dataset.index, pd.MultiIndex) or "symbol" not in dataset.index.names:
        return {"assumed_notional_usd": 10_000.0, "median_round_trip_cost_bps": 0.0}

    cost_model = BinanceCostModel()
    symbol_costs: dict[str, float] = {}
    for symbol, frame in dataset.groupby(level="symbol", sort=True):
        if frame.empty:
            continue
        close = pd.to_numeric(frame["close"], errors="coerce")
        volume = pd.to_numeric(frame.get("volume"), errors="coerce") if "volume" in frame.columns else None
        quote_volume = pd.to_numeric(frame.get("quote_volume"), errors="coerce") if "quote_volume" in frame.columns else None
        if quote_volume is not None and quote_volume.notna().any():
            adv_usd = float(quote_volume.dropna().tail(24).mean())
        elif volume is not None and close.notna().any():
            adv_usd = float((volume * close).dropna().tail(24).mean())
        else:
            adv_usd = 10_000_000.0
        estimate = cost_model.estimate(str(symbol), 10_000.0, adv_usd=adv_usd)
        symbol_costs[str(symbol)] = float(estimate.round_trip_cost_bps)
    median_cost = float(np.median(list(symbol_costs.values()))) if symbol_costs else 0.0
    return {
        "assumed_notional_usd": 10_000.0,
        "per_symbol_round_trip_cost_bps": symbol_costs,
        "median_round_trip_cost_bps": median_cost,
        "recommended_dead_zone_bps_floor": median_cost,
    }


def _signal_from_momentum(
    *,
    symbol: str,
    history: pd.DataFrame,
    horizon_bars: int,
    lookback: int,
    deadband: float,
    market_risk=None,
    invert: bool = False,
    vol_filter_bps: float | None = None,
) -> StrategySignal:
    close = pd.to_numeric(history.get("close"), errors="coerce").dropna()
    if len(close) <= lookback:
        return StrategySignal(
            symbol=symbol,
            timeframe="1h",
            horizon_bars=horizon_bars,
            signal="HOLD",
            confidence=0.5,
            uncertainty=0.5,
            reason="insufficient_history",
            market_risk=market_risk,
        )

    recent_return = float(close.iloc[-1] / close.iloc[-(lookback + 1)] - 1.0)
    if vol_filter_bps is not None and len(close) >= 20:
        vol = float(pd.Series(close.pct_change().dropna()).tail(20).std() or 0.0) * 10_000.0
        if vol > vol_filter_bps:
            return StrategySignal(
                symbol=symbol,
                timeframe="1h",
                horizon_bars=horizon_bars,
                signal="HOLD",
                confidence=0.5,
                uncertainty=0.5,
                reason="vol_filter_hold",
                market_risk=market_risk,
            )

    if abs(recent_return) < deadband:
        signal = "HOLD"
        confidence = 0.5
    elif recent_return > 0.0:
        signal = "SELL" if invert else "BUY"
        confidence = min(0.95, 0.55 + abs(recent_return) * 30.0)
    else:
        signal = "BUY" if invert else "SELL"
        confidence = min(0.95, 0.55 + abs(recent_return) * 30.0)

    return StrategySignal(
        symbol=symbol,
        timeframe="1h",
        horizon_bars=horizon_bars,
        signal=signal,
        confidence=float(confidence),
        uncertainty=max(0.0, 1.0 - float(confidence)),
        reason="benchmark_momentum" if not invert else "benchmark_mean_reversion",
        market_risk=market_risk,
    )


def _benchmark_signal_resolver(
    actor: ReplayActorConfig,
    symbol: str,
    history: pd.DataFrame,
    timestamp: pd.Timestamp,
    market_risk,
) -> StrategySignal:
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
        return _signal_from_momentum(
            symbol=symbol,
            history=history,
            horizon_bars=actor.horizon_bars,
            lookback=4,
            deadband=0.003,
            market_risk=market_risk,
        )
    if name == "mean_reversion":
        return _signal_from_momentum(
            symbol=symbol,
            history=history,
            horizon_bars=actor.horizon_bars,
            lookback=4,
            deadband=0.003,
            market_risk=market_risk,
            invert=True,
        )
    if name == "volatility_filtered":
        return _signal_from_momentum(
            symbol=symbol,
            history=history,
            horizon_bars=actor.horizon_bars,
            lookback=4,
            deadband=0.003,
            market_risk=market_risk,
            vol_filter_bps=120.0,
        )
    return StrategySignal(
        symbol=symbol,
        timeframe="1h",
        horizon_bars=actor.horizon_bars,
        signal="HOLD",
        confidence=0.5,
        uncertainty=0.5,
        reason=f"unknown_benchmark_{name}",
        market_risk=market_risk,
    )


def _estimate_actor_funding_pnl(result, dataset: pd.DataFrame) -> float:
    if not isinstance(dataset.index, pd.MultiIndex) or "symbol" not in dataset.index.names:
        return 0.0
    if not result.equity_curve:
        return 0.0

    frame = dataset.sort_index()
    funding_total = 0.0
    for point in result.equity_curve:
        timestamp = pd.Timestamp(point["timestamp"])
        positions = point.get("positions") or {}
        for symbol, qty in positions.items():
            if symbol not in frame.index.get_level_values("symbol"):
                continue
            try:
                row = frame.loc[(timestamp, symbol)]
            except Exception:
                continue
            if isinstance(row, pd.DataFrame):
                row = row.iloc[-1]
            price = _safe_float(row.get("close"))
            if price <= 0.0:
                continue
            funding_rate = row.get("funding_rate_raw", row.get("funding_rate", 0.0))
            funding_rate = _safe_float(funding_rate)
            if funding_rate == 0.0:
                continue
            notional = abs(_safe_float(qty)) * price
            funding_total += notional * funding_rate * (1.0 if _safe_float(qty) > 0.0 else -1.0)
    return funding_total


def _episode_trade_summary(fills: list[dict[str, Any]]) -> dict[str, Any]:
    per_symbol: dict[str, dict[str, float]] = {}
    realized: list[float] = []

    for fill in sorted(fills, key=lambda item: str(item.get("timestamp") or "")):
        symbol = str(fill.get("symbol") or "").strip().upper()
        if not symbol:
            continue
        state = per_symbol.setdefault(
            symbol,
            {"position": 0.0, "avg_entry": 0.0, "cost_usd": 0.0, "gross_pnl": 0.0},
        )
        signed_qty = _safe_float(fill.get("filled_qty", 0.0))
        if str(fill.get("side") or "").upper() == "SELL":
            signed_qty *= -1.0
        price = _safe_float(fill.get("price", 0.0))
        fill_cost = _safe_float(fill.get("fee_usd", 0.0)) + _safe_float(fill.get("slippage_usd", 0.0))

        pos = state["position"]
        avg_entry = state["avg_entry"]
        if pos == 0.0 or pos * signed_qty > 0.0:
            total_qty = abs(pos) + abs(signed_qty)
            if total_qty > 0.0:
                state["avg_entry"] = (abs(pos) * avg_entry + abs(signed_qty) * price) / total_qty
            state["position"] = pos + signed_qty
            state["cost_usd"] += fill_cost
            continue

        closing_qty = min(abs(pos), abs(signed_qty))
        if pos > 0.0:
            state["gross_pnl"] += closing_qty * (price - avg_entry)
        else:
            state["gross_pnl"] += closing_qty * (avg_entry - price)
        state["cost_usd"] += fill_cost
        remaining = pos + signed_qty
        if abs(remaining) < 1e-12:
            realized.append(state["gross_pnl"] - state["cost_usd"])
            per_symbol[symbol] = {"position": 0.0, "avg_entry": 0.0, "cost_usd": 0.0, "gross_pnl": 0.0}
        else:
            state["position"] = remaining
            state["avg_entry"] = price
            state["gross_pnl"] = 0.0
            state["cost_usd"] = 0.0

    win_rate = float(sum(1 for pnl in realized if pnl > 0.0) / len(realized)) if realized else 0.0
    average_trade = float(np.mean(realized)) if realized else 0.0
    return {
        "closed_trade_count": int(len(realized)),
        "win_rate": win_rate,
        "average_trade_usd": average_trade,
        "average_trade_pnl_usd": average_trade,
        "realized_trade_pnls_usd": [round(float(v), 6) for v in realized],
    }


def _equity_return_sharpe(equity_curve: list[dict[str, Any]]) -> float:
    if len(equity_curve) < 3:
        return 0.0
    equity = pd.Series(
        [float(point["equity_usd"]) for point in equity_curve],
        index=pd.DatetimeIndex([pd.Timestamp(point["timestamp"]) for point in equity_curve]),
        dtype=float,
    )
    daily = equity.resample("D").last().ffill().pct_change().dropna()
    if len(daily) < 2 or float(daily.std(ddof=0) or 0.0) == 0.0:
        return 0.0
    return float((daily.mean() / daily.std(ddof=0)) * math.sqrt(365.0))


def _summarize_actor_result(result, dataset: pd.DataFrame) -> dict[str, Any]:
    fills = [asdict(fill) for fill in result.fills]
    exposures_by_symbol: dict[str, float] = {}
    for point in result.equity_curve:
        positions = point.get("positions") or {}
        for symbol, qty in positions.items():
            try:
                symbol_frame = dataset.xs(symbol, level="symbol")
                timestamp = pd.Timestamp(point["timestamp"])
                price = float(symbol_frame.loc[:timestamp]["close"].iloc[-1])
            except Exception:
                price = 0.0
            exposure = abs(_safe_float(qty)) * price
            exposures_by_symbol[symbol] = max(exposures_by_symbol.get(symbol, 0.0), exposure)

    funding_pnl = _estimate_actor_funding_pnl(result, dataset)
    trade_summary = _episode_trade_summary(fills)
    net_pnl_usd = _safe_float(result.metrics.get("net_pnl_usd", 0.0))
    total_fees_usd = _safe_float(result.metrics.get("total_fees_usd", 0.0))
    total_slippage_usd = _safe_float(result.metrics.get("total_slippage_usd", 0.0))
    cost_adjusted_net_pnl_usd = net_pnl_usd + funding_pnl
    turnover = _safe_float(result.metrics.get("turnover", 0.0))
    max_drawdown_frac = _safe_float(result.metrics.get("max_drawdown_frac", 0.0))
    return {
        "actor": result.actor,
        "metrics": dict(result.metrics),
        "net_pnl_usd": net_pnl_usd,
        "total_fees_usd": total_fees_usd,
        "total_slippage_usd": total_slippage_usd,
        "estimated_funding_pnl_usd": funding_pnl,
        "cost_adjusted_net_pnl_usd": cost_adjusted_net_pnl_usd,
        "turnover": turnover,
        "max_drawdown_frac": max_drawdown_frac,
        "sharpe": _equity_return_sharpe(result.equity_curve),
        "win_rate": trade_summary["win_rate"],
        "average_trade_usd": trade_summary["average_trade_usd"],
        "closed_trade_count": trade_summary["closed_trade_count"],
        "open_order_count": int(result.metrics.get("open_order_count", 0) or 0),
        "fill_count": int(result.metrics.get("fill_count", 0) or 0),
        "blocked_intents": int(result.metrics.get("blocked_intents", 0) or 0),
        "exposure_by_symbol_usd": {symbol: round(float(value), 6) for symbol, value in sorted(exposures_by_symbol.items())},
        "reconciliation": dict(result.reconciliation),
        "state_digest": result.state_digest,
        "manifest": dict(result.manifest),
    }


def build_benchmark_replay_report(
    dataset: pd.DataFrame,
    *,
    initial_equity: float = 1_000.0,
) -> dict[str, Any]:
    """Replay transparent benchmarks through the exact portfolio replay engine."""

    if not isinstance(dataset.index, pd.MultiIndex) or list(dataset.index.names) != ["timestamp", "symbol"]:
        raise ValueError("dataset must be MultiIndex with levels ['timestamp', 'symbol']")
    if dataset.empty:
        raise ValueError("dataset cannot be empty")

    actors = {
        "flat": ReplayActorConfig(
            name="flat",
            kind="fixed",
            min_confidence=1.0,
            horizon_bars=4,
            metadata={"benchmark_name": "flat"},
        ),
        "momentum": ReplayActorConfig(
            name="momentum",
            kind="fixed",
            min_confidence=0.55,
            horizon_bars=4,
            baseline_lookback=4,
            baseline_deadband=0.003,
            metadata={"benchmark_name": "momentum"},
        ),
        "mean_reversion": ReplayActorConfig(
            name="mean_reversion",
            kind="fixed",
            min_confidence=0.55,
            horizon_bars=4,
            baseline_lookback=4,
            baseline_deadband=0.003,
            metadata={"benchmark_name": "mean_reversion"},
        ),
        "volatility_filtered": ReplayActorConfig(
            name="volatility_filtered",
            kind="fixed",
            min_confidence=0.55,
            horizon_bars=4,
            baseline_lookback=4,
            baseline_deadband=0.003,
            metadata={"benchmark_name": "volatility_filtered"},
        ),
    }
    replay = run_portfolio_replay(
        dataset,
        actors,
        initial_equity=initial_equity,
        scenario=ReplayScenario(name="quality_recovery_benchmarks"),
        dataset_manifest={
            "policy_version": QUALITY_RECOVERY_POLICY_VERSION,
            "row_count": len(dataset),
            "symbols": sorted(str(s) for s in dataset.index.get_level_values("symbol").unique()),
        },
        cost_policy=ExecutionCostPolicy(),
        signal_resolver=_benchmark_signal_resolver,
    )

    actor_summaries = {
        name: _summarize_actor_result(result, dataset)
        for name, result in replay.actors.items()
    }
    baseline = actor_summaries.get("flat", {})
    flat_cost_adjusted = _safe_float(baseline.get("cost_adjusted_net_pnl_usd", 0.0))
    best_name = max(actor_summaries, key=lambda name: _safe_float(actor_summaries[name].get("cost_adjusted_net_pnl_usd", 0.0)))
    best = actor_summaries[best_name]
    flat_vs_best = _safe_float(best.get("cost_adjusted_net_pnl_usd", 0.0)) - flat_cost_adjusted

    return {
        "policy_version": QUALITY_RECOVERY_POLICY_VERSION,
        "generated_at": _utc_now(),
        "dataset": {
            "rows": int(len(dataset)),
            "symbols": sorted(str(s) for s in dataset.index.get_level_values("symbol").unique()),
            "start": str(pd.DatetimeIndex(dataset.index.get_level_values("timestamp")).min()),
            "end": str(pd.DatetimeIndex(dataset.index.get_level_values("timestamp")).max()),
        },
        "replay_digest": replay.replay_digest,
        "actor_summaries": actor_summaries,
        "comparisons": {
            "best_actor": best_name,
            "flat_cost_adjusted_net_pnl_usd": flat_cost_adjusted,
            "best_cost_adjusted_net_pnl_usd": _safe_float(best.get("cost_adjusted_net_pnl_usd", 0.0)),
            "best_minus_flat_cost_adjusted_net_pnl_usd": flat_vs_best,
            "best_minus_flat_cost_adjusted_net_pnl_bps": flat_vs_best / max(float(initial_equity), 1e-9) * 10_000.0,
        },
        "replay": replay.to_dict(),
    }


def build_candidate_selection_report(
    *,
    diagnostic_report: dict[str, Any],
    label_audit_report: dict[str, Any],
    benchmark_replay_report: dict[str, Any],
) -> dict[str, Any]:
    """Summarize the next-step recommendation from diagnostics and replay evidence."""

    actor_summaries = benchmark_replay_report.get("actor_summaries", {})
    flat = actor_summaries.get("flat", {}) if isinstance(actor_summaries, dict) else {}
    best_name = benchmark_replay_report.get("comparisons", {}).get("best_actor")
    best = actor_summaries.get(best_name, {}) if isinstance(actor_summaries, dict) and best_name else {}
    best_vs_flat = _safe_float(
        benchmark_replay_report.get("comparisons", {}).get("best_minus_flat_cost_adjusted_net_pnl_usd", 0.0)
    )
    recommendation = "no_trade"
    if best_name and best_name != "flat" and best_vs_flat > 0.0 and _safe_float(best.get("win_rate", 0.0)) > 0.5:
        recommendation = "advance_to_candidate_grid"

    return {
        "policy_version": QUALITY_RECOVERY_POLICY_VERSION,
        "generated_at": _utc_now(),
        "diagnostic_status": diagnostic_report.get("recommended_next_step", ""),
        "label_audit_grid": label_audit_report.get("grid", {}),
        "benchmark_best_actor": best_name,
        "benchmark_best_vs_flat_cost_adjusted_net_pnl_usd": best_vs_flat,
        "benchmark_best_win_rate": _safe_float(best.get("win_rate", 0.0)),
        "flat_cost_adjusted_net_pnl_usd": _safe_float(flat.get("cost_adjusted_net_pnl_usd", 0.0)),
        "recommendation": recommendation,
        "paper_soak_admission": False,
        "notes": (
            "The system remains no-trade unless a benchmark or future candidate "
            "beats flat after costs and passes shadow/paper gates."
        ),
    }


def render_report_markdown(title: str, payload: dict[str, Any]) -> str:
    """Render a compact markdown report for docs/model_quality/ artifacts."""

    lines = [f"# {title}", "", f"- policy_version: `{payload.get('policy_version', QUALITY_RECOVERY_POLICY_VERSION)}`"]
    lines.append(f"- generated_at: `{payload.get('generated_at', _utc_now())}`")

    if "failure_record" in payload:
        failure = payload.get("failure_record", {}) or {}
        lines.extend(
            [
                "",
                "## Failed Retrain",
                f"- reason: `{failure.get('reason') or ''}`",
                f"- path: `{failure.get('path') or ''}`",
            ]
        )

    if "dataset" in payload:
        dataset = payload.get("dataset", {}) or {}
        lines.extend(
            [
                "",
                "## Dataset",
                f"- rows: `{dataset.get('rows', 0)}`",
                f"- symbols: `{', '.join(dataset.get('symbols', []))}`",
                f"- start: `{dataset.get('start', '')}`",
                f"- end: `{dataset.get('end', '')}`",
            ]
        )

    if "comparisons" in payload:
        comparisons = payload.get("comparisons", {}) or {}
        lines.extend(
            [
                "",
                "## Benchmark Comparison",
                f"- best_actor: `{comparisons.get('best_actor', '')}`",
                f"- best_minus_flat_cost_adjusted_net_pnl_usd: `{_safe_float(comparisons.get('best_minus_flat_cost_adjusted_net_pnl_usd', 0.0)):.4f}`",
            ]
        )

    if "recommendation" in payload:
        lines.extend(
            [
                "",
                "## Recommendation",
                f"- recommendation: `{payload.get('recommendation', '')}`",
                f"- paper_soak_admission: `{bool(payload.get('paper_soak_admission', False))}`",
            ]
        )

    lines.extend(["", "## JSON", "```json", json.dumps(payload, indent=2, sort_keys=True, default=str), "```"])
    return "\n".join(lines)


def write_quality_recovery_bundle(
    output_dir: Path | str,
    *,
    diagnostic_report: dict[str, Any],
    label_audit_report: dict[str, Any],
    benchmark_replay_report: dict[str, Any],
    candidate_selection_report: dict[str, Any],
) -> dict[str, Path]:
    """Persist the recovery reports as JSON and markdown artifacts."""

    output_dir = Path(output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    paths: dict[str, Path] = {}
    payloads = {
        "failed_retrain_diagnostic": diagnostic_report,
        "label_audit": label_audit_report,
        "benchmark_replay": benchmark_replay_report,
        "candidate_selection": candidate_selection_report,
    }
    for name, payload in payloads.items():
        json_path = output_dir / f"{name}_{timestamp}.json"
        md_path = output_dir / f"{name}_{timestamp}.md"
        json_path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
        md_path.write_text(render_report_markdown(name.replace("_", " ").title(), payload), encoding="utf-8")
        paths[f"{name}_json"] = json_path
        paths[f"{name}_md"] = md_path

    policy_path = output_dir / "validation_policy_v1.md"
    if not policy_path.exists():
        policy_path.write_text(
            "\n".join(
                [
                    "# Validation Policy v1",
                    "",
                    "- The retrain gate remains strict until benchmark replay proves positive cost-adjusted expectancy.",
                    "- A fresh candidate must beat flat and at least one transparent benchmark before shadow admission.",
                    "- No-trade is the correct state when every benchmark loses after costs.",
                    "- The validation policy version is `model_quality_validation_policy_v1`.",
                ]
            ),
            encoding="utf-8",
        )
    paths["validation_policy_md"] = policy_path
    return paths


def generate_quality_recovery_bundle(
    *,
    model_root: Path | str,
    registry_root: Path | str,
    dataset: pd.DataFrame,
    output_dir: Path | str,
    failed_record_path: Path | str | None = None,
) -> tuple[QualityRecoveryBundle, dict[str, Path]]:
    """Build all quality-recovery reports and write them to disk."""

    diagnostic = build_failed_retrain_diagnostic(
        model_root,
        registry_root,
        failed_record_path=failed_record_path,
    )
    label_audit = build_label_audit_report(dataset)
    benchmark_replay = build_benchmark_replay_report(dataset)
    candidate_selection = build_candidate_selection_report(
        diagnostic_report=diagnostic,
        label_audit_report=label_audit,
        benchmark_replay_report=benchmark_replay,
    )
    paths = write_quality_recovery_bundle(
        output_dir,
        diagnostic_report=diagnostic,
        label_audit_report=label_audit,
        benchmark_replay_report=benchmark_replay,
        candidate_selection_report=candidate_selection,
    )
    return (
        QualityRecoveryBundle(
            diagnostic=diagnostic,
            label_audit=label_audit,
            benchmark_replay=benchmark_replay,
            candidate_selection=candidate_selection,
        ),
        paths,
    )


def fetch_quality_recovery_dataset(
    *,
    months: int = 6,
    symbols: Iterable[str] | None = None,
    interval: str = "1h",
    client: BinanceClient | None = None,
) -> pd.DataFrame:
    """Fetch a multi-symbol dataset for quality recovery reporting."""

    from datetime import timedelta
    from quant_v2.data.multi_symbol_dataset import fetch_universe_dataset
    from quant_v2.config import default_universe_symbols

    now = datetime.now(timezone.utc)
    date_to = now
    date_from = now - timedelta(days=max(months, 1) * 30)
    selected_symbols = tuple(symbols or default_universe_symbols())
    return fetch_universe_dataset(
        selected_symbols,
        date_from=date_from,
        date_to=date_to,
        interval=interval,
        fail_fast=False,
        client=client,
    )
