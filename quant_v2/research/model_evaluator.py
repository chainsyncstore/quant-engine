"""Forward paper evaluation and promotion gate for quarantined models.

The weekly retrain service registers new artifacts as ``paper_quarantine``.
This module closes the lifecycle loop by:

1. running shadow signal cycles for quarantined candidates and the incumbent;
2. resolving those decisions against future marks;
3. writing paper-evaluation metrics into the model registry; and
4. optionally promoting only after a full rolling paper window beats incumbent.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import sqlite3
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from quant_v2.config import default_universe_symbols
from quant_v2.contracts import StrategySignal
from quant_v2.model_registry import ModelRegistry, ModelVersionRecord
from quant_v2.monitoring.shadow_drift import ShadowDriftStats, compute_shadow_live_drift
from quant_v2.research.portfolio_replay import (
    ReplayActorConfig,
    ReplayScenario,
    SignalResolver,
    run_portfolio_replay,
)

logger = logging.getLogger(__name__)

SHADOW_TABLE = "model_shadow_decisions"
CONTROL_FILE_NAME = "evaluator_control.json"

THRESHOLD_ENV_KEYS = (
    "BOT_V2_REGIME2_BUY_THRESHOLD",
    "BOT_V2_REGIME2_SELL_THRESHOLD",
    "BOT_V2_CHRONOS_DISAGREEMENT_MULT",
    "BOT_V2_REQUIRE_CHRONOS_AGREEMENT_FOR_ENTRY",
)
RISK_ENV_KEYS = (
    "BOT_V2_MARKET_SHORT_GUARD_LOOKBACK_HOURS",
    "BOT_V2_MARKET_SHORT_GUARD_DOWN_RATIO",
    "BOT_V2_MARKET_SHORT_GUARD_MEDIAN_RETURN",
    "BOT_V2_MARKET_SHORT_GUARD_BTC_RETURN",
    "BOT_V2_MARKET_SHORT_GUARD_NET_CAP_FRAC",
    "BOT_V2_MAX_HOLD_HOURS",
    "BOT_V2_MIN_NOTIONAL_USD",
    "BOT_V2_MIN_NOTIONAL_EQUITY_PCT",
    "BOT_V2_STRANDED_FLATTEN_CYCLES",
)


@dataclass(frozen=True)
class EvaluationPolicy:
    """Promotion and tuning gates for forward paper evaluation."""

    threshold_tuning_hours: float = 72.0
    promotion_window_hours: float = 168.0
    min_resolved_decisions: int = 500
    min_actionable_decisions: int = 30
    min_symbols: int = 3
    min_calendar_days: int = 0
    min_trading_days: int = 0
    min_symbol_coverage_fraction: float = 0.0
    min_edge_bps: float = 25.0
    max_drawdown_worse_bps: float = 50.0
    round_trip_cost_bps: float = 8.0
    max_symbol_concentration: float = 0.60

    @classmethod
    def from_env(cls) -> "EvaluationPolicy":
        return cls(
            threshold_tuning_hours=_env_float("MODEL_EVAL_THRESHOLD_TUNING_HOURS", 72.0),
            promotion_window_hours=_env_float("MODEL_EVAL_PROMOTION_WINDOW_HOURS", 168.0),
            min_resolved_decisions=_env_int("MODEL_EVAL_MIN_RESOLVED_DECISIONS", 500),
            min_actionable_decisions=_env_int("MODEL_EVAL_MIN_ACTIONABLE_DECISIONS", 30),
            min_symbols=_env_int("MODEL_EVAL_MIN_SYMBOLS", 3),
            min_calendar_days=_env_int("MODEL_EVAL_MIN_CALENDAR_DAYS", 30),
            min_trading_days=_env_int("MODEL_EVAL_MIN_TRADING_DAYS", 20),
            min_symbol_coverage_fraction=_env_float(
                "MODEL_EVAL_MIN_SYMBOL_COVERAGE_FRAC",
                1.0,
            ),
            min_edge_bps=_env_float("MODEL_EVAL_MIN_EDGE_BPS", 25.0),
            max_drawdown_worse_bps=_env_float("MODEL_EVAL_MAX_DRAWDOWN_WORSE_BPS", 50.0),
            round_trip_cost_bps=_env_float("MODEL_EVAL_ROUND_TRIP_COST_BPS", 8.0),
            max_symbol_concentration=_env_float("MODEL_EVAL_MAX_SYMBOL_CONCENTRATION", 0.60),
        )


@dataclass(frozen=True)
class RuntimeBlockers:
    """Operational conditions that can defer automatic promotion."""

    active_sessions: int = 0
    hard_risk_pauses: int = 0


@dataclass(frozen=True)
class PromotionDecision:
    promotion_eligible: bool
    threshold_tuning_ready: bool
    blockers: tuple[str, ...]
    notes: str


def ensure_shadow_schema(db_path: Path | str) -> None:
    """Create the model shadow-decision table for SQLite deployments."""

    path = Path(db_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {SHADOW_TABLE} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                quarantine_version_id TEXT NOT NULL,
                model_version_id TEXT NOT NULL,
                baseline_version_id TEXT NOT NULL,
                decision_role TEXT NOT NULL,
                evaluated_at TEXT NOT NULL,
                bar_timestamp TEXT NOT NULL,
                symbol TEXT NOT NULL,
                signal TEXT NOT NULL,
                probability REAL,
                buy_threshold REAL,
                sell_threshold REAL,
                close_price REAL NOT NULL,
                horizon_hours INTEGER NOT NULL,
                future_mark_price REAL,
                future_return_bps REAL,
                resolved_at TEXT,
                reason TEXT,
                threshold_config_hash TEXT,
                risk_config_hash TEXT,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            f"""
            CREATE UNIQUE INDEX IF NOT EXISTS idx_{SHADOW_TABLE}_unique
            ON {SHADOW_TABLE} (
                quarantine_version_id,
                model_version_id,
                decision_role,
                bar_timestamp,
                symbol,
                horizon_hours
            )
            """
        )
        conn.execute(
            f"""
            CREATE INDEX IF NOT EXISTS idx_{SHADOW_TABLE}_resolve
            ON {SHADOW_TABLE} (future_mark_price, evaluated_at, symbol)
            """
        )
        conn.commit()


def record_shadow_decision(
    db_path: Path | str,
    *,
    quarantine_version_id: str,
    model_version_id: str,
    baseline_version_id: str,
    decision_role: str,
    payload: dict[str, Any],
    horizon_hours: int,
    evaluated_at: datetime | None = None,
) -> bool:
    """Persist one candidate/incumbent shadow decision.

    Returns ``True`` when a row was inserted and ``False`` when the unique key
    already exists for that bar/model/symbol.
    """

    ensure_shadow_schema(db_path)
    now = datetime.now(timezone.utc)
    evaluated = evaluated_at or now
    symbol = str(payload.get("symbol") or "").strip().upper()
    if not symbol:
        return False
    close_price = _float(payload.get("close_price"), 0.0)
    if close_price <= 0.0:
        return False

    bar_timestamp = str(payload.get("timestamp") or evaluated.isoformat())
    signal = str(payload.get("signal") or "HOLD").strip().upper()
    threshold_hash = _config_hash(THRESHOLD_ENV_KEYS)
    risk_hash = _config_hash(RISK_ENV_KEYS)

    with sqlite3.connect(Path(db_path).expanduser()) as conn:
        cur = conn.execute(
            f"""
            INSERT OR IGNORE INTO {SHADOW_TABLE} (
                quarantine_version_id,
                model_version_id,
                baseline_version_id,
                decision_role,
                evaluated_at,
                bar_timestamp,
                symbol,
                signal,
                probability,
                buy_threshold,
                sell_threshold,
                close_price,
                horizon_hours,
                reason,
                threshold_config_hash,
                risk_config_hash,
                created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                quarantine_version_id,
                model_version_id,
                baseline_version_id,
                decision_role,
                evaluated.isoformat(),
                bar_timestamp,
                symbol,
                signal,
                _float_or_none(payload.get("probability")),
                _float_or_none(payload.get("_buy_th") or payload.get("buy_th")),
                _float_or_none(payload.get("_sell_th") or payload.get("sell_th")),
                close_price,
                int(horizon_hours),
                str(payload.get("reason") or "")[:1000],
                threshold_hash,
                risk_hash,
                now.isoformat(),
            ),
        )
        conn.commit()
        return cur.rowcount > 0


def resolve_due_shadow_decisions(
    db_path: Path | str,
    *,
    prices: dict[str, float],
    now: datetime | None = None,
) -> int:
    """Resolve due shadow decisions with latest available marks."""

    ensure_shadow_schema(db_path)
    resolved_at = now or datetime.now(timezone.utc)
    count = 0
    with sqlite3.connect(Path(db_path).expanduser()) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            f"""
            SELECT id, evaluated_at, symbol, signal, close_price, horizon_hours
            FROM {SHADOW_TABLE}
            WHERE future_mark_price IS NULL
            """
        ).fetchall()
        for row in rows:
            try:
                evaluated_at = datetime.fromisoformat(str(row["evaluated_at"]))
            except ValueError:
                continue
            if evaluated_at.tzinfo is None:
                evaluated_at = evaluated_at.replace(tzinfo=timezone.utc)
            horizon_hours = int(row["horizon_hours"] or 0)
            if resolved_at < evaluated_at + timedelta(hours=horizon_hours):
                continue
            symbol = str(row["symbol"]).upper()
            future_price = float(prices.get(symbol, 0.0) or 0.0)
            entry_price = float(row["close_price"] or 0.0)
            if future_price <= 0.0 or entry_price <= 0.0:
                continue
            future_return_bps = _directional_return_bps(
                signal=str(row["signal"]),
                entry_price=entry_price,
                future_price=future_price,
            )
            conn.execute(
                f"""
                UPDATE {SHADOW_TABLE}
                SET future_mark_price = ?,
                    future_return_bps = ?,
                    resolved_at = ?
                WHERE id = ?
                """,
                (future_price, future_return_bps, resolved_at.isoformat(), int(row["id"])),
            )
            count += 1
        conn.commit()
    return count


def count_due_shadow_decisions(
    db_path: Path | str,
    *,
    now: datetime | None = None,
) -> int:
    """Count unresolved shadow rows whose evaluation horizon has elapsed."""

    ensure_shadow_schema(db_path)
    resolved_at = now or datetime.now(timezone.utc)
    due = 0
    with sqlite3.connect(Path(db_path).expanduser()) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            f"""
            SELECT evaluated_at, horizon_hours
            FROM {SHADOW_TABLE}
            WHERE future_mark_price IS NULL
            """
        ).fetchall()
        for row in rows:
            try:
                evaluated_at = datetime.fromisoformat(str(row["evaluated_at"]))
            except ValueError:
                continue
            if evaluated_at.tzinfo is None:
                evaluated_at = evaluated_at.replace(tzinfo=timezone.utc)
            horizon_hours = int(row["horizon_hours"] or 0)
            if resolved_at >= evaluated_at + timedelta(hours=horizon_hours):
                due += 1
    return due


def _normalize_shadow_rows(rows: Iterable[sqlite3.Row]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for row in rows:
        bar_timestamp = _parse_datetime(str(row["bar_timestamp"] or ""))
        evaluated_at = _parse_datetime(str(row["evaluated_at"] or ""))
        resolved_at = _parse_datetime(str(row["resolved_at"] or ""))
        if bar_timestamp is None or evaluated_at is None:
            continue
        close_price = _float(row["close_price"], 0.0)
        if close_price <= 0.0:
            continue
        future_mark_price = _float_or_none(row["future_mark_price"])
        normalized.append(
            {
                "quarantine_version_id": str(row["quarantine_version_id"] or "").strip(),
                "model_version_id": str(row["model_version_id"] or "").strip(),
                "baseline_version_id": str(row["baseline_version_id"] or "").strip(),
                "decision_role": str(row["decision_role"] or "").strip().lower(),
                "bar_timestamp": bar_timestamp,
                "evaluated_at": evaluated_at,
                "resolved_at": resolved_at,
                "symbol": str(row["symbol"] or "").strip().upper(),
                "signal": str(row["signal"] or "HOLD").strip().upper(),
                "probability": _float_or_none(row["probability"]),
                "close_price": close_price,
                "future_mark_price": future_mark_price,
                "future_return_bps": _float_or_none(row["future_return_bps"]),
                "horizon_hours": int(row["horizon_hours"] or 0),
                "reason": str(row["reason"] or ""),
            }
        )
    return normalized


def _build_replay_market_dataset(rows: list[dict[str, Any]]) -> pd.DataFrame:
    bars: dict[tuple[pd.Timestamp, str], dict[str, float]] = {}
    for row in rows:
        symbol = str(row["symbol"])
        entry_ts = pd.Timestamp(row["bar_timestamp"])
        close_price = float(row["close_price"])
        bars[(entry_ts, symbol)] = {
            "open": close_price,
            "high": close_price,
            "low": close_price,
            "close": close_price,
            "volume": max(1.0, close_price),
            "quote_volume": max(1.0, close_price),
            "taker_buy_volume": max(0.5, close_price * 0.55),
            "taker_sell_volume": max(0.5, close_price * 0.45),
            "funding_rate": 0.0,
            "open_interest": 0.0,
            "open_interest_value": 0.0,
        }
        future_mark_price = row.get("future_mark_price")
        resolved_at = row.get("resolved_at")
        if future_mark_price is not None and resolved_at is not None:
            future_price = float(future_mark_price)
            bars[(pd.Timestamp(resolved_at), symbol)] = {
                "open": future_price,
                "high": future_price,
                "low": future_price,
                "close": future_price,
                "volume": max(1.0, future_price),
                "quote_volume": max(1.0, future_price),
                "taker_buy_volume": max(0.5, future_price * 0.55),
                "taker_sell_volume": max(0.5, future_price * 0.45),
                "funding_rate": 0.0,
                "open_interest": 0.0,
                "open_interest_value": 0.0,
            }

    if not bars:
        return pd.DataFrame()

    frame = pd.DataFrame(
        [
            {"timestamp": timestamp, "symbol": symbol, **payload}
            for (timestamp, symbol), payload in sorted(bars.items(), key=lambda item: (item[0][0], item[0][1]))
        ]
    )
    frame = frame.set_index(["timestamp", "symbol"]).sort_index()
    frame.index = frame.index.set_names(["timestamp", "symbol"])
    return frame


def _merge_shadow_episodes(rows: list[dict[str, Any]], role: str) -> list[dict[str, Any]]:
    episodes: list[dict[str, Any]] = []
    role_rows = [
        row
        for row in rows
        if row["decision_role"] == role and row["signal"] in {"BUY", "SELL"} and row["future_mark_price"] is not None
    ]
    role_rows.sort(key=lambda row: (row["symbol"], row["bar_timestamp"], row["horizon_hours"], row["signal"]))

    for symbol, symbol_rows in _group_rows(role_rows, key=lambda row: row["symbol"]):
        for direction, direction_rows in _group_rows(symbol_rows, key=lambda row: row["signal"]):
            current: dict[str, Any] | None = None
            for row in direction_rows:
                start = pd.Timestamp(row["bar_timestamp"])
                end = start + timedelta(hours=max(row["horizon_hours"], 1))
                if current is None:
                    current = {
                        "symbol": symbol,
                        "direction": direction,
                        "start": start,
                        "end": end,
                        "entry_price": float(row["close_price"]),
                        "exit_price": float(row["future_mark_price"] or row["close_price"]),
                        "confidence": _shadow_confidence(row),
                        "raw_count": 1,
                    }
                    continue
                if start <= current["end"]:
                    current["end"] = max(current["end"], end)
                    current["exit_price"] = float(row["future_mark_price"] or current["exit_price"])
                    current["confidence"] = max(current["confidence"], _shadow_confidence(row))
                    current["raw_count"] += 1
                else:
                    episodes.append(current)
                    current = {
                        "symbol": symbol,
                        "direction": direction,
                        "start": start,
                        "end": end,
                        "entry_price": float(row["close_price"]),
                        "exit_price": float(row["future_mark_price"] or row["close_price"]),
                        "confidence": _shadow_confidence(row),
                        "raw_count": 1,
                    }
            if current is not None:
                episodes.append(current)

    episodes.sort(key=lambda item: (item["start"], item["symbol"], item["direction"]))
    return episodes


def _shadow_confidence(row: dict[str, Any]) -> float:
    probability = row.get("probability")
    value = float(probability) if probability is not None else 0.5
    return float(max(value, 1.0 - value, 0.5))


def _group_rows(rows: list[dict[str, Any]], *, key) -> list[tuple[Any, list[dict[str, Any]]]]:
    grouped: list[tuple[Any, list[dict[str, Any]]]] = []
    current_key: Any | None = None
    current_rows: list[dict[str, Any]] = []
    for row in rows:
        row_key = key(row)
        if current_key is None or row_key != current_key:
            if current_rows:
                grouped.append((current_key, current_rows))
            current_key = row_key
            current_rows = [row]
        else:
            current_rows.append(row)
    if current_rows:
        grouped.append((current_key, current_rows))
    return grouped


def _mean_or_zero(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _build_performance_diagnostics(
    rows: list[dict[str, Any]],
    *,
    candidate_metrics: dict[str, Any],
    policy: EvaluationPolicy,
) -> dict[str, Any]:
    """Summarize calibration drift, turnover, and realized costs from replay rows."""

    candidate_probs: list[float] = []
    incumbent_probs: list[float] = []
    candidate_errors: list[float] = []
    incumbent_errors: list[float] = []
    paired_probs: dict[tuple[pd.Timestamp, str, int], dict[str, float]] = {}

    for row in rows:
        probability = _float_or_none(row.get("probability"))
        future_return = _float_or_none(row.get("future_return_bps"))
        if probability is None or future_return is None:
            continue
        target = 1.0 if future_return > 0.0 else 0.0
        error = abs(probability - target)
        role = str(row.get("decision_role") or "").strip().lower()
        if role == "candidate":
            candidate_probs.append(probability)
            candidate_errors.append(error)
        elif role == "incumbent":
            incumbent_probs.append(probability)
            incumbent_errors.append(error)

        key = (
            pd.Timestamp(row["bar_timestamp"]),
            str(row["symbol"] or "").strip().upper(),
            int(row["horizon_hours"] or 0),
        )
        role_bucket = paired_probs.setdefault(key, {})
        role_bucket[role] = probability

    paired_candidate: list[float] = []
    paired_incumbent: list[float] = []
    for pair in paired_probs.values():
        if "candidate" in pair and "incumbent" in pair:
            paired_candidate.append(float(pair["candidate"]))
            paired_incumbent.append(float(pair["incumbent"]))

    drift = (
        compute_shadow_live_drift(paired_candidate, paired_incumbent)
        if paired_candidate and paired_incumbent
        else ShadowDriftStats(mean_abs_error=0.0, directional_agreement=0.0, n_samples=0)
    )

    turnover = _float_or_none(candidate_metrics.get("turnover")) or 0.0
    total_cost_usd = (
        _float_or_none(candidate_metrics.get("total_fees_usd")) or 0.0
    ) + (
        _float_or_none(candidate_metrics.get("total_slippage_usd")) or 0.0
    )
    final_equity = _float_or_none(candidate_metrics.get("final_equity_usd")) or 0.0
    realized_cost_bps = (
        (total_cost_usd / max(final_equity, 1e-9)) * 10_000.0
        if final_equity > 0.0
        else 0.0
    )
    concentration = _float_or_none(candidate_metrics.get("max_symbol_concentration")) or 0.0
    symbol_headroom = max(float(policy.max_symbol_concentration) - concentration, 0.0)

    return {
        "candidate_probability_calibration_mae": _mean_or_zero(candidate_errors),
        "incumbent_probability_calibration_mae": _mean_or_zero(incumbent_errors),
        "candidate_probability_samples": int(len(candidate_probs)),
        "incumbent_probability_samples": int(len(incumbent_probs)),
        "paired_probability_samples": int(drift.n_samples),
        "candidate_incumbent_probability_drift_mae": float(drift.mean_abs_error),
        "candidate_incumbent_directional_agreement": float(drift.directional_agreement),
        "candidate_turnover": float(turnover),
        "candidate_realized_cost_bps": float(realized_cost_bps),
        "candidate_blocked_intents": int(candidate_metrics.get("blocked_intents", 0) or 0),
        "candidate_max_symbol_concentration": float(concentration),
        "candidate_symbol_concentration_headroom": float(symbol_headroom),
    }


def _model_threshold_floor_from_manifest(manifest: dict[str, Any]) -> float | None:
    training = manifest.get("training") or {}
    if not isinstance(training, dict):
        return None

    threshold_policy = training.get("threshold_policy") or {}
    if isinstance(threshold_policy, dict):
        selected_threshold = threshold_policy.get("selected_threshold")
        if selected_threshold is not None:
            try:
                return float(selected_threshold)
            except (TypeError, ValueError):
                return None

    threshold = training.get("threshold")
    if threshold is not None:
        try:
            return float(threshold)
        except (TypeError, ValueError):
            return None
    return None


def _model_threshold_floor_for_version(
    registry: ModelRegistry | None,
    version_id: str | None,
) -> float | None:
    if registry is None or not version_id:
        return None
    try:
        manifest = registry.get_artifact_manifest(str(version_id))
    except Exception:
        return None
    return _model_threshold_floor_from_manifest(manifest)


def _build_shadow_signal_resolver(episodes_by_role: dict[str, list[dict[str, Any]]]) -> SignalResolver:
    entry_map: dict[tuple[str, pd.Timestamp, str], tuple[str, float, float, str]] = {}
    exit_map: dict[tuple[str, pd.Timestamp, str], tuple[str, float, float, str]] = {}

    for role, episodes in episodes_by_role.items():
        for episode in episodes:
            start = pd.Timestamp(episode["start"])
            end = pd.Timestamp(episode["end"])
            direction = str(episode["direction"]).upper()
            opposite = "SELL" if direction == "BUY" else "BUY"
            confidence = float(episode.get("confidence", 0.5))
            entry_map[(role, start, str(episode["symbol"]))] = (
                direction,
                confidence,
                1.0 - confidence,
                "shadow_entry",
            )
            exit_map[(role, end, str(episode["symbol"]))] = (
                opposite,
                confidence,
                1.0 - confidence,
                "shadow_exit",
            )

    def _resolver(
        actor: ReplayActorConfig,
        symbol: str,
        history: pd.DataFrame,
        timestamp: pd.Timestamp,
        market_risk,
    ):
        key = (actor.name, pd.Timestamp(timestamp), symbol)
        payload = entry_map.get(key) or exit_map.get(key)
        if payload is None:
            return None
        signal, confidence, uncertainty, reason = payload
        return StrategySignal(
            symbol=symbol,
            timeframe="1h",
            horizon_bars=max(actor.horizon_bars, 1),
            signal=signal,
            confidence=float(confidence),
            uncertainty=float(uncertainty),
            reason=reason,
            market_risk=market_risk,
        )

    return _resolver


def _bootstrap_block_ci(
    block_values: list[float],
    *,
    seed_material: str,
    resamples: int = 1000,
    alpha: float = 0.05,
) -> dict[str, Any]:
    if not block_values:
        return {
            "block_count": 0,
            "resamples": 0,
            "net_return_bps": {"low": 0.0, "high": 0.0},
            "drawdown_bps": {"low": 0.0, "high": 0.0},
        }

    values = np.asarray(block_values, dtype=float)
    seed = int(hashlib.sha256(seed_material.encode("utf-8")).hexdigest()[:16], 16)
    rng = np.random.default_rng(seed)
    net_samples = np.empty(max(resamples, 1), dtype=float)
    dd_samples = np.empty(max(resamples, 1), dtype=float)

    for idx in range(net_samples.size):
        choice = rng.integers(0, len(values), size=len(values))
        sampled = values[choice]
        cumulative = np.cumsum(sampled, dtype=float)
        peak = np.maximum.accumulate(np.concatenate(([0.0], cumulative)))
        drawdown = np.minimum.accumulate(np.concatenate(([0.0], cumulative)) - peak).min()
        net_samples[idx] = float(sampled.sum())
        dd_samples[idx] = float(drawdown)

    low_q = alpha / 2.0
    high_q = 1.0 - low_q
    return {
        "block_count": int(len(values)),
        "resamples": int(net_samples.size),
        "net_return_bps": {
            "low": float(np.quantile(net_samples, low_q)),
            "high": float(np.quantile(net_samples, high_q)),
        },
        "drawdown_bps": {
            "low": float(np.quantile(dd_samples, low_q)),
            "high": float(np.quantile(dd_samples, high_q)),
        },
    }


def _replay_summary_for_quarantine(
    rows: list[dict[str, Any]],
    *,
    quarantine_version_id: str,
    policy: EvaluationPolicy,
    registry_root: Path | str | None = None,
) -> dict[str, Any]:
    if not rows:
        return {
            "replay": None,
            "coverage": {
                "calendar_days": 0,
                "trading_days": 0,
                "symbols": 0,
                "symbol_coverage_fraction": 0.0,
                "volatility_regime_coverage_ok": False,
                "calendar_blocks": 0,
                "position_episodes": 0,
                "outcome_units": 0,
            },
        }

    benchmark_rows = [row for row in rows if row["decision_role"] == "incumbent"]
    candidate_rows = [row for row in rows if row["decision_role"] == "candidate"]
    dataset = _build_replay_market_dataset(rows)
    candidate_episodes = _merge_shadow_episodes(rows, "candidate")
    incumbent_episodes = _merge_shadow_episodes(rows, "incumbent")
    episodes_by_role = {
        "candidate": candidate_episodes,
        "incumbent": incumbent_episodes,
    }
    signal_resolver = _build_shadow_signal_resolver(episodes_by_role)
    horizon_bars = max(max((row["horizon_hours"] for row in rows), default=1), 1)
    registry = ModelRegistry(registry_root) if registry_root is not None else None
    candidate_model_version = next(
        (
            str(row.get("model_version_id") or "").strip()
            for row in rows
            if str(row.get("decision_role") or "").strip().lower() == "candidate"
            and str(row.get("model_version_id") or "").strip()
        ),
        "",
    )
    incumbent_model_version = next(
        (
            str(row.get("model_version_id") or "").strip()
            for row in rows
            if str(row.get("decision_role") or "").strip().lower() == "incumbent"
            and str(row.get("model_version_id") or "").strip()
        ),
        "",
    )
    candidate_threshold = _model_threshold_floor_for_version(registry, candidate_model_version) or 0.5
    incumbent_threshold = _model_threshold_floor_for_version(registry, incumbent_model_version) or 0.5
    actors = {
        "candidate": ReplayActorConfig(
            name="candidate",
            kind="fixed",
            threshold=candidate_threshold,
            min_confidence=0.5,
            horizon_bars=horizon_bars,
            baseline_lookback=3,
            baseline_deadband=0.0,
            metadata={"role": "candidate", "quarantine_version_id": quarantine_version_id},
        ),
        "incumbent": ReplayActorConfig(
            name="incumbent",
            kind="fixed",
            threshold=incumbent_threshold,
            min_confidence=0.5,
            horizon_bars=horizon_bars,
            baseline_lookback=3,
            baseline_deadband=0.0,
            metadata={"role": "incumbent", "quarantine_version_id": quarantine_version_id},
        ),
        "benchmark": ReplayActorConfig(
            name="benchmark",
            kind="fixed",
            min_confidence=1.0,
            horizon_bars=horizon_bars,
            metadata={"role": "benchmark", "quarantine_version_id": quarantine_version_id, "baseline": "cash"},
        ),
    }
    replay = run_portfolio_replay(
        dataset,
        actors,
        initial_equity=1_000.0,
        scenario=ReplayScenario(name="shadow_eval"),
        dataset_manifest={
            "quarantine_version_id": quarantine_version_id,
            "row_count": len(rows),
            "candidate_raw_rows": len(candidate_rows),
            "incumbent_raw_rows": len(benchmark_rows),
        },
        signal_resolver=signal_resolver,
    )

    candidate_result = replay.actors["candidate"]
    incumbent_result = replay.actors["incumbent"]
    benchmark_result = replay.actors["benchmark"]

    unique_days = {pd.Timestamp(row["bar_timestamp"]).date() for row in rows}
    trading_days = {
        pd.Timestamp(point["timestamp"]).date()
        for result in replay.actors.values()
        for point in result.equity_curve
        if point.get("timestamp")
    }
    symbols = sorted({row["symbol"] for row in rows})
    universe = tuple(default_universe_symbols())
    symbol_coverage_fraction = (
        len(set(symbols).intersection(universe)) / max(len(universe), 1)
    )
    candidate_curve = pd.Series(
        [float(point["equity_usd"]) for point in candidate_result.equity_curve],
        index=pd.DatetimeIndex([pd.Timestamp(point["timestamp"]) for point in candidate_result.equity_curve]),
        dtype=float,
    )
    benchmark_curve = pd.Series(
        [float(point["equity_usd"]) for point in benchmark_result.equity_curve],
        index=pd.DatetimeIndex([pd.Timestamp(point["timestamp"]) for point in benchmark_result.equity_curve]),
        dtype=float,
    )
    candidate_daily = candidate_curve.resample("D").last().ffill()
    benchmark_daily = benchmark_curve.resample("D").last().ffill()
    aligned = pd.concat([candidate_daily, benchmark_daily], axis=1, keys=["candidate", "benchmark"]).dropna()
    candidate_diff = aligned["candidate"].diff().fillna(aligned["candidate"].iloc[0] - 1_000.0)
    benchmark_diff = aligned["benchmark"].diff().fillna(aligned["benchmark"].iloc[0] - 1_000.0)
    block_diff = (candidate_diff - benchmark_diff).tolist()
    bootstrap_ci = _bootstrap_block_ci(
        block_diff,
        seed_material=f"{quarantine_version_id}:{len(rows)}:{len(candidate_episodes)}:{len(incumbent_episodes)}",
    )

    def _net_return_bps(result) -> float:
        return float(result.metrics["net_pnl_usd"]) / 1_000.0 * 10_000.0

    def _drawdown_bps(result) -> float:
        return float(result.metrics["max_drawdown_frac"]) * 10_000.0

    candidate_symbol_counts = {}
    for fill in candidate_result.fills:
        candidate_symbol_counts[fill.symbol] = candidate_symbol_counts.get(fill.symbol, 0) + 1
    concentration = (
        max(candidate_symbol_counts.values()) / max(sum(candidate_symbol_counts.values()), 1)
        if candidate_symbol_counts
        else 0.0
    )
    coverage = {
        "calendar_days": int(len(unique_days)),
        "trading_days": int(len(trading_days)),
        "symbols": int(len(symbols)),
        "symbol_coverage_fraction": float(symbol_coverage_fraction),
        "volatility_regime_coverage_ok": bool(
            len(block_diff) >= 2 and any(value > 0.0 for value in block_diff) and any(value < 0.0 for value in block_diff)
        ),
        "calendar_blocks": int(len(unique_days)),
        "position_episodes": int(len(candidate_episodes)),
        "outcome_units": int(len(candidate_result.fills) // 2),
    }

    candidate_metrics = {
        **candidate_result.metrics,
        "resolved_decisions": int(len(rows)),
        "actionable_decisions": int(len(candidate_episodes)),
        "symbols": int(len(symbols)),
        "max_symbol_concentration": float(concentration),
        "net_return_bps": round(_net_return_bps(candidate_result), 4),
        "mean_pnl_usd": round(float(candidate_result.metrics["net_pnl_usd"]) / max(len(candidate_episodes), 1), 6),
        "mean_return_bps": round(_net_return_bps(candidate_result) / max(len(candidate_episodes), 1), 4),
        "calendar_blocks": coverage["calendar_blocks"],
        "position_episodes": coverage["position_episodes"],
        "outcome_units": coverage["outcome_units"],
    }
    incumbent_metrics = {
        **incumbent_result.metrics,
        "resolved_decisions": int(len(rows)),
        "actionable_decisions": int(len(incumbent_episodes)),
        "symbols": int(len(symbols)),
        "net_return_bps": round(_net_return_bps(incumbent_result), 4),
        "mean_pnl_usd": round(float(incumbent_result.metrics["net_pnl_usd"]) / max(len(incumbent_episodes), 1), 6),
        "mean_return_bps": round(_net_return_bps(incumbent_result) / max(len(incumbent_episodes), 1), 4),
        "calendar_blocks": coverage["calendar_blocks"],
        "position_episodes": int(len(incumbent_episodes)),
        "outcome_units": int(len(incumbent_result.fills) // 2),
    }
    benchmark_metrics = {
        **benchmark_result.metrics,
        "resolved_decisions": int(len(rows)),
        "actionable_decisions": int(len(benchmark_result.fills)),
        "symbols": int(len(symbols)),
        "net_return_bps": round(_net_return_bps(benchmark_result), 4),
        "mean_pnl_usd": round(float(benchmark_result.metrics["net_pnl_usd"]) / max(len(benchmark_result.fills) // 2, 1), 6),
        "mean_return_bps": round(_net_return_bps(benchmark_result) / max(len(benchmark_result.fills) // 2, 1), 4),
        "calendar_blocks": coverage["calendar_blocks"],
        "position_episodes": int(sum(1 for point in benchmark_result.equity_curve if point.get("open_orders"))),
        "outcome_units": int(len(benchmark_result.fills) // 2),
    }
    paired_metrics = {
        "candidate_net_return_bps": round(_net_return_bps(candidate_result), 4),
        "incumbent_net_return_bps": round(_net_return_bps(incumbent_result), 4),
        "benchmark_net_return_bps": round(_net_return_bps(benchmark_result), 4),
        "candidate_minus_incumbent_net_return_bps": round(
            _net_return_bps(candidate_result) - _net_return_bps(incumbent_result),
            4,
        ),
        "candidate_minus_benchmark_net_return_bps": round(
            _net_return_bps(candidate_result) - _net_return_bps(benchmark_result),
            4,
        ),
        "candidate_drawdown_bps": round(_drawdown_bps(candidate_result), 4),
        "incumbent_drawdown_bps": round(_drawdown_bps(incumbent_result), 4),
        "benchmark_drawdown_bps": round(_drawdown_bps(benchmark_result), 4),
        "candidate_minus_incumbent_drawdown_bps": round(
            _drawdown_bps(candidate_result) - _drawdown_bps(incumbent_result),
            4,
        ),
        "candidate_minus_benchmark_drawdown_bps": round(
            _drawdown_bps(candidate_result) - _drawdown_bps(benchmark_result),
            4,
        ),
        "bootstrap_ci": bootstrap_ci,
    }
    threshold_policy = {
        "candidate": {
            "model_version_id": candidate_model_version or None,
            "threshold_floor": float(candidate_threshold),
        },
        "incumbent": {
            "model_version_id": incumbent_model_version or None,
            "threshold_floor": float(incumbent_threshold),
        },
    }

    return {
        "replay": replay,
        "coverage": coverage,
        "candidate_metrics": candidate_metrics,
        "incumbent_metrics": incumbent_metrics,
        "benchmark_metrics": benchmark_metrics,
        "paired_metrics": paired_metrics,
        "threshold_policy": threshold_policy,
    }


def summarize_quarantine(
    db_path: Path | str,
    *,
    quarantine_version_id: str,
    policy: EvaluationPolicy,
    registry_root: Path | str | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Build candidate-vs-incumbent metrics for one quarantine version."""

    ensure_shadow_schema(db_path)
    current_time = now or datetime.now(timezone.utc)
    window_start = current_time - timedelta(hours=policy.promotion_window_hours)
    with sqlite3.connect(Path(db_path).expanduser()) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            f"""
            SELECT *
            FROM {SHADOW_TABLE}
            WHERE quarantine_version_id = ?
              AND future_return_bps IS NOT NULL
              AND evaluated_at >= ?
            ORDER BY evaluated_at ASC, id ASC
            """,
            (quarantine_version_id, window_start.isoformat()),
        ).fetchall()

    all_times = [
        _parse_datetime(str(row["evaluated_at"]))
        for row in rows
        if row["evaluated_at"] is not None
    ]
    valid_times = [ts for ts in all_times if ts is not None]
    if valid_times:
        evaluation_start = min(valid_times)
        evaluation_end = max(valid_times)
        window_hours = max((current_time - evaluation_start).total_seconds() / 3600.0, 0.0)
    else:
        evaluation_start = None
        evaluation_end = None
        window_hours = 0.0

    normalized_rows = _normalize_shadow_rows(rows)
    replay_summary = _replay_summary_for_quarantine(
        normalized_rows,
        quarantine_version_id=quarantine_version_id,
        policy=policy,
        registry_root=registry_root,
    )
    if replay_summary.get("replay") is not None:
        candidate = replay_summary["candidate_metrics"]
        incumbent = replay_summary["incumbent_metrics"]
        benchmark = replay_summary["benchmark_metrics"]
        paired = replay_summary["paired_metrics"]
        delta = {
            "net_return_bps": paired["candidate_minus_incumbent_net_return_bps"],
            "mean_return_bps": candidate["mean_pnl_usd"] - incumbent["mean_pnl_usd"],
            "max_drawdown_bps": paired["candidate_minus_incumbent_drawdown_bps"],
            "actionable_decisions": candidate["actionable_decisions"] - incumbent["actionable_decisions"],
        }
    else:
        by_role: dict[str, list[sqlite3.Row]] = {"candidate": [], "incumbent": []}
        for row in rows:
            role = str(row["decision_role"] or "").lower()
            if role in by_role:
                by_role[role].append(row)

        candidate = _metric_summary(by_role["candidate"], policy=policy)
        incumbent = _metric_summary(by_role["incumbent"], policy=policy)
        benchmark = {}
        paired = {}
        delta = {
            "net_return_bps": candidate["net_return_bps"] - incumbent["net_return_bps"],
            "mean_return_bps": candidate["mean_return_bps"] - incumbent["mean_return_bps"],
            "max_drawdown_bps": candidate["max_drawdown_bps"] - incumbent["max_drawdown_bps"],
            "actionable_decisions": candidate["actionable_decisions"] - incumbent["actionable_decisions"],
        }
    performance_diagnostics = _build_performance_diagnostics(
        normalized_rows,
        candidate_metrics=candidate,
        policy=policy,
    )

    return {
        "version_id": quarantine_version_id,
        "evaluated_at": current_time.isoformat(),
        "evaluation_window_start": evaluation_start.isoformat() if evaluation_start else None,
        "evaluation_window_end": evaluation_end.isoformat() if evaluation_end else None,
        "evaluation_window_hours": round(window_hours, 4),
        "policy": {
        "threshold_tuning_hours": policy.threshold_tuning_hours,
        "promotion_window_hours": policy.promotion_window_hours,
        "min_resolved_decisions": policy.min_resolved_decisions,
        "min_actionable_decisions": policy.min_actionable_decisions,
        "min_symbols": policy.min_symbols,
        "min_calendar_days": policy.min_calendar_days,
        "min_trading_days": policy.min_trading_days,
        "min_symbol_coverage_fraction": policy.min_symbol_coverage_fraction,
        "min_edge_bps": policy.min_edge_bps,
        "max_drawdown_worse_bps": policy.max_drawdown_worse_bps,
        "round_trip_cost_bps": policy.round_trip_cost_bps,
        "max_symbol_concentration": policy.max_symbol_concentration,
    },
        "candidate_metrics": candidate,
        "incumbent_metrics": incumbent,
        "benchmark_metrics": benchmark,
        "paired_metrics": paired,
        "delta_metrics": delta,
        "performance_diagnostics": performance_diagnostics,
        "resolved_decisions": len(rows),
        "coverage": replay_summary.get("coverage", {}),
        "threshold_policy": replay_summary.get("threshold_policy", {}),
    }


def decide_promotion(
    summary: dict[str, Any],
    *,
    policy: EvaluationPolicy,
    runtime: RuntimeBlockers | None = None,
) -> PromotionDecision:
    """Evaluate whether a quarantined candidate can be promoted."""

    runtime = runtime or RuntimeBlockers()
    candidate = summary.get("candidate_metrics") or {}
    incumbent = summary.get("incumbent_metrics") or {}
    paired = summary.get("paired_metrics") or {}
    coverage = summary.get("coverage") or {}
    delta = summary.get("delta_metrics") or {}
    blockers: list[str] = []

    window_hours = _float(summary.get("evaluation_window_hours"), 0.0)
    threshold_tuning_ready = window_hours >= policy.threshold_tuning_hours
    if window_hours < policy.promotion_window_hours:
        blockers.append(
            f"needs_{policy.promotion_window_hours:.0f}h_window_current_{window_hours:.1f}h"
        )
    if int(candidate.get("resolved_decisions", 0)) < policy.min_resolved_decisions:
        blockers.append(
            f"candidate_resolved_decisions<{policy.min_resolved_decisions}"
        )
    if int(incumbent.get("resolved_decisions", 0)) < policy.min_resolved_decisions:
        blockers.append(
            f"incumbent_resolved_decisions<{policy.min_resolved_decisions}"
        )
    if int(candidate.get("actionable_decisions", 0)) < policy.min_actionable_decisions:
        blockers.append(
            f"candidate_actionable_decisions<{policy.min_actionable_decisions}"
        )
    if int(candidate.get("symbols", 0)) < policy.min_symbols:
        blockers.append(f"candidate_symbols<{policy.min_symbols}")
    if _float(candidate.get("max_symbol_concentration"), 1.0) > policy.max_symbol_concentration:
        blockers.append("candidate_symbol_concentration_too_high")
    if _float(delta.get("net_return_bps"), 0.0) < policy.min_edge_bps:
        blockers.append(f"net_edge_bps<{policy.min_edge_bps:.1f}")
    if _float(delta.get("mean_return_bps"), 0.0) < 0.0:
        blockers.append("mean_return_worse_than_incumbent")
    if _float(delta.get("max_drawdown_bps"), 0.0) < -policy.max_drawdown_worse_bps:
        blockers.append("drawdown_worse_than_allowed")
    if paired:
        if _float(paired.get("candidate_net_return_bps"), 0.0) <= 0.0:
            blockers.append("candidate_negative_absolute_expectancy")
        if _float(paired.get("candidate_minus_benchmark_net_return_bps"), 0.0) < policy.min_edge_bps:
            blockers.append(f"benchmark_edge_bps<{policy.min_edge_bps:.1f}")
        if _float(paired.get("candidate_minus_incumbent_net_return_bps"), 0.0) < policy.min_edge_bps:
            blockers.append(f"incumbent_edge_bps<{policy.min_edge_bps:.1f}")
        if _float(paired.get("candidate_minus_benchmark_drawdown_bps"), 0.0) < -policy.max_drawdown_worse_bps:
            blockers.append("benchmark_drawdown_worse_than_allowed")
    strict_coverage = (
        policy.min_calendar_days > 0
        or policy.min_trading_days > 0
        or policy.min_symbol_coverage_fraction > 0.0
    )
    if policy.min_calendar_days > 0 and int(coverage.get("calendar_days", 0)) < policy.min_calendar_days:
        blockers.append(f"calendar_days<{policy.min_calendar_days}")
    if policy.min_trading_days > 0 and int(coverage.get("trading_days", 0)) < policy.min_trading_days:
        blockers.append(f"trading_days<{policy.min_trading_days}")
    if policy.min_symbol_coverage_fraction > 0.0 and _float(coverage.get("symbol_coverage_fraction"), 0.0) < policy.min_symbol_coverage_fraction:
        blockers.append("symbol_coverage_fraction_too_low")
    if strict_coverage and coverage and not bool(coverage.get("volatility_regime_coverage_ok", True)):
        blockers.append("volatility_regime_coverage_missing")
    if runtime.hard_risk_pauses > 0:
        blockers.append(f"hard_risk_pauses={runtime.hard_risk_pauses}")
    if runtime.active_sessions > 0:
        blockers.append(f"active_sessions={runtime.active_sessions}")

    eligible = not blockers
    if eligible:
        notes = "candidate beat incumbent over forward paper window"
    else:
        notes = "promotion blocked: " + ", ".join(blockers)
    return PromotionDecision(
        promotion_eligible=eligible,
        threshold_tuning_ready=threshold_tuning_ready,
        blockers=tuple(blockers),
        notes=notes,
    )


def record_registry_evaluation(
    registry: ModelRegistry,
    version_id: str,
    *,
    summary: dict[str, Any],
    decision: PromotionDecision,
) -> ModelVersionRecord:
    """Persist paper-evaluation metrics into the registry."""

    evaluation = {
        **summary,
        "threshold_tuning_ready": decision.threshold_tuning_ready,
        "threshold_policy": dict(summary.get("threshold_policy") or {}),
        "promotion_blockers": list(decision.blockers),
        "promotion_decision": "eligible" if decision.promotion_eligible else "blocked",
    }
    return registry.record_paper_evaluation(
        version_id,
        evaluation=evaluation,
        promotion_eligible=decision.promotion_eligible,
        notes=decision.notes,
    )


def load_evaluator_control(registry_root: Path | str) -> dict[str, Any]:
    """Load controls without allowing persistent state to exceed deployment policy."""

    path = Path(registry_root).expanduser() / CONTROL_FILE_NAME
    deployment_allows_auto_promote = _env_bool("MODEL_EVAL_AUTO_PROMOTE", False)
    if not path.exists():
        return {
            "auto_promote": deployment_allows_auto_promote,
            "deployment_auto_promote_allowed": deployment_allows_auto_promote,
        }
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {
            "auto_promote": False,
            "deployment_auto_promote_allowed": deployment_allows_auto_promote,
        }
    persistent_auto_promote = bool(payload.get("auto_promote", False))
    payload["auto_promote"] = (
        deployment_allows_auto_promote and persistent_auto_promote
    )
    payload["deployment_auto_promote_allowed"] = deployment_allows_auto_promote
    return payload


def write_evaluator_control(
    registry_root: Path | str,
    *,
    auto_promote: bool,
    updated_by: str = "",
) -> dict[str, Any]:
    """Persist evaluator runtime control flags."""

    path = Path(registry_root).expanduser() / CONTROL_FILE_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    deployment_allows_auto_promote = _env_bool("MODEL_EVAL_AUTO_PROMOTE", False)
    payload = {
        "auto_promote": bool(auto_promote) and deployment_allows_auto_promote,
        "deployment_auto_promote_allowed": deployment_allows_auto_promote,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "updated_by": updated_by,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def get_runtime_blockers(db_path: Path | str) -> RuntimeBlockers:
    """Read active-session and hard-risk blockers from the Telegram DB."""

    path = Path(db_path).expanduser()
    if not path.exists():
        return RuntimeBlockers()
    try:
        with sqlite3.connect(path) as conn:
            active_sessions = int(
                conn.execute(
                    "SELECT COUNT(*) FROM user_context WHERE COALESCE(is_active, 0) = 1"
                ).fetchone()[0]
            )
            hard_risk_pauses = int(
                conn.execute(
                    "SELECT COUNT(*) FROM user_context WHERE COALESCE(hard_risk_paused, 0) = 1"
                ).fetchone()[0]
            )
        return RuntimeBlockers(
            active_sessions=active_sessions,
            hard_risk_pauses=hard_risk_pauses,
        )
    except sqlite3.Error as exc:
        logger.warning("Could not read runtime blockers from %s: %s", path, exc)
        return RuntimeBlockers()


async def run_shadow_cycle_for_version(
    record: ModelVersionRecord,
    *,
    registry_root: Path | str,
    symbols: tuple[str, ...],
    horizon_bars: int = 4,
) -> list[dict[str, Any]]:
    """Run one no-order signal cycle for a registered model version."""

    from quant_v2.telebot.signal_manager import V2SignalManager, _SignalSession

    with tempfile.TemporaryDirectory(prefix="model_eval_registry_") as tmp:
        temp_registry = ModelRegistry(tmp)
        temp_registry.register_version(
            record.version_id,
            record.artifact_dir,
            metrics=dict(record.metrics),
            tags=dict(record.tags),
            description=record.description,
            status="active",
        )
        temp_registry.set_active_version(record.version_id)

        manager = V2SignalManager(
            Path(record.artifact_dir),
            registry_root=Path(tmp),
            symbols=symbols,
            horizon_bars=horizon_bars,
            loop_interval_seconds=24 * 3600,
            max_signal_log=max(len(symbols) * 2, 20),
        )
        collected: list[dict[str, Any]] = []

        def _collect(payload: dict[str, Any]) -> None:
            if str(payload.get("signal") or "").upper() == "CYCLE_DIGEST":
                return
            collected.append(dict(payload))

        client = manager._default_client_factory(
            {},
            False,
            symbols[0] if symbols else "BTCUSDT",
            manager.anchor_interval,
        )
        session = _SignalSession(
            user_id=0,
            live=False,
            client=client,
            on_signal=_collect,
            running=True,
        )
        await manager._run_cycle(session, cycle_cache={})
        return collected


async def evaluate_once(
    *,
    model_root: Path | str,
    registry_root: Path | str,
    db_path: Path | str,
    policy: EvaluationPolicy | None = None,
    collect_shadow: bool = True,
    auto_promote: bool | None = None,
) -> dict[str, Any]:
    """Run one evaluator pass and return a structured summary."""

    policy = policy or EvaluationPolicy.from_env()
    registry = ModelRegistry(registry_root)
    active = registry.get_active_version()
    ensure_shadow_schema(db_path)
    control = load_evaluator_control(registry_root)
    deployment_allows_auto_promote = bool(
        control.get("deployment_auto_promote_allowed", False)
    )
    requested_auto_promote = bool(control.get("auto_promote", False))
    if auto_promote is not None:
        requested_auto_promote = bool(auto_promote)
    effective_auto_promote = (
        deployment_allows_auto_promote and requested_auto_promote
    )

    if active is None:
        return {"status": "no_active_model", "evaluated": []}

    candidates = [
        record
        for record in registry.list_candidates()
        if record.version_id != active.version_id
        and (
            record.status == "paper_quarantine"
            or bool(record.metrics.get("paper_quarantine_required"))
        )
    ]
    symbols = tuple(default_universe_symbols())
    horizon_hours = _env_int("MODEL_EVAL_HORIZON_HOURS", 8)
    cycle_inserted = 0

    if collect_shadow and candidates:
        for candidate in candidates:
            try:
                incumbent_payloads = await run_shadow_cycle_for_version(
                    active,
                    registry_root=registry_root,
                    symbols=symbols,
                    horizon_bars=_env_int("MODEL_EVAL_HORIZON_BARS", 4),
                )
                candidate_payloads = await run_shadow_cycle_for_version(
                    candidate,
                    registry_root=registry_root,
                    symbols=symbols,
                    horizon_bars=_env_int("MODEL_EVAL_HORIZON_BARS", 4),
                )
            except Exception as exc:
                logger.exception("Shadow cycle failed for %s: %s", candidate.version_id, exc)
                continue
            for payload in incumbent_payloads:
                if record_shadow_decision(
                    db_path,
                    quarantine_version_id=candidate.version_id,
                    model_version_id=active.version_id,
                    baseline_version_id=active.version_id,
                    decision_role="incumbent",
                    payload=payload,
                    horizon_hours=horizon_hours,
                ):
                    cycle_inserted += 1
            for payload in candidate_payloads:
                if record_shadow_decision(
                    db_path,
                    quarantine_version_id=candidate.version_id,
                    model_version_id=candidate.version_id,
                    baseline_version_id=active.version_id,
                    decision_role="candidate",
                    payload=payload,
                    horizon_hours=horizon_hours,
                ):
                    cycle_inserted += 1

    due_shadow_decisions = count_due_shadow_decisions(db_path)
    resolved = 0
    if due_shadow_decisions > 0:
        prices = _fetch_latest_prices(symbols)
        resolved = resolve_due_shadow_decisions(db_path, prices=prices)
    runtime = get_runtime_blockers(db_path)

    evaluated: list[dict[str, Any]] = []
    for candidate in candidates:
        summary = summarize_quarantine(
            db_path,
            quarantine_version_id=candidate.version_id,
            policy=policy,
            registry_root=registry_root,
        )
        decision = decide_promotion(summary, policy=policy, runtime=runtime)
        updated = record_registry_evaluation(
            registry,
            candidate.version_id,
            summary=summary,
            decision=decision,
        )
        promoted = False
        if effective_auto_promote and decision.promotion_eligible:
            registry.promote_version(
                candidate.version_id,
                promoted_by="model_evaluator:auto",
                notes=decision.notes,
            )
            promoted = True
        evaluated.append(
            {
                "version_id": candidate.version_id,
                "status": updated.status,
                "promotion_eligible": decision.promotion_eligible,
                "threshold_tuning_ready": decision.threshold_tuning_ready,
                "blockers": list(decision.blockers),
                "promoted": promoted,
                "summary": summary,
            }
        )

    return {
        "status": "ok",
        "active_version_id": active.version_id,
        "candidates": len(candidates),
        "cycle_inserted": cycle_inserted,
        "resolved": resolved,
        "auto_promote": effective_auto_promote,
        "runtime_blockers": {
            "active_sessions": runtime.active_sessions,
            "hard_risk_pauses": runtime.hard_risk_pauses,
        },
        "evaluated": evaluated,
    }


def build_evaluation_report(
    *,
    registry_root: Path | str,
    db_path: Path | str,
    limit: int = 6,
) -> list[dict[str, Any]]:
    """Return latest paper-evaluation summaries for admin commands."""

    registry = ModelRegistry(registry_root)
    records = registry.list_candidates()
    out: list[dict[str, Any]] = []
    for record in records[-max(int(limit), 1):]:
        paper_eval = (record.metrics or {}).get("paper_evaluation") or {}
        out.append(
            {
                "version_id": record.version_id,
                "status": record.status,
                "paper_evaluation": paper_eval,
                "promotion_notes": record.promotion_notes,
            }
        )
    return out


def run_scheduler_loop() -> None:
    """Blocking evaluator loop for Docker service use."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    model_root = Path(os.getenv("BOT_MODEL_ROOT", "/app/models/production")).expanduser()
    registry_root = Path(
        os.getenv("BOT_MODEL_REGISTRY_ROOT", str(model_root / "registry"))
    ).expanduser()
    db_path = Path(os.getenv("BOT_DB_PATH", "/state/quant_bot.db")).expanduser()
    interval_seconds = _env_int("MODEL_EVAL_INTERVAL_SECONDS", 900)
    startup_delay = _env_int("MODEL_EVAL_STARTUP_DELAY_SECONDS", 180)
    collect_shadow = _env_bool("MODEL_EVAL_COLLECT_SHADOW", True)
    policy = EvaluationPolicy.from_env()

    logger.info(
        "Model evaluator started: interval=%ss promotion_window=%.0fh tuning_window=%.0fh db=%s",
        interval_seconds,
        policy.promotion_window_hours,
        policy.threshold_tuning_hours,
        db_path,
    )
    if startup_delay > 0:
        logger.info("Model evaluator waiting %ss before first pass...", startup_delay)
        time.sleep(startup_delay)

    while True:
        try:
            result = asyncio.run(
                evaluate_once(
                    model_root=model_root,
                    registry_root=registry_root,
                    db_path=db_path,
                    policy=policy,
                    collect_shadow=collect_shadow,
                )
            )
            logger.info(
                "Model evaluator pass: status=%s candidates=%s inserted=%s resolved=%s auto_promote=%s",
                result.get("status"),
                result.get("candidates"),
                result.get("cycle_inserted"),
                result.get("resolved"),
                result.get("auto_promote"),
            )
            for item in result.get("evaluated", []):
                logger.info(
                    "Evaluation %s eligible=%s tuning_ready=%s promoted=%s blockers=%s",
                    item.get("version_id"),
                    item.get("promotion_eligible"),
                    item.get("threshold_tuning_ready"),
                    item.get("promoted"),
                    ",".join(item.get("blockers") or []),
                )
        except Exception as exc:
            logger.exception("Model evaluator pass failed: %s", exc)
        time.sleep(max(interval_seconds, 1))


def _metric_summary(rows: Iterable[sqlite3.Row], *, policy: EvaluationPolicy) -> dict[str, Any]:
    rows_list = list(rows)
    returns: list[float] = []
    symbols: dict[str, int] = {}
    actionable = 0
    for row in rows_list:
        signal = str(row["signal"] or "").upper()
        symbol = str(row["symbol"] or "").upper()
        symbols[symbol] = symbols.get(symbol, 0) + 1
        if signal not in {"BUY", "SELL"}:
            continue
        actionable += 1
        returns.append(float(row["future_return_bps"] or 0.0) - policy.round_trip_cost_bps)

    net = float(sum(returns))
    mean = net / len(returns) if returns else 0.0
    wins = [value for value in returns if value > 0.0]
    losses = [value for value in returns if value < 0.0]
    hit_rate = len(wins) / len(returns) if returns else 0.0
    gross_win = sum(wins)
    gross_loss = abs(sum(losses))
    profit_factor = gross_win / gross_loss if gross_loss > 0.0 else (gross_win if gross_win > 0 else 0.0)
    max_drawdown = _max_drawdown_bps(returns)
    max_symbol_count = max(symbols.values()) if symbols else 0
    concentration = max_symbol_count / len(rows_list) if rows_list else 0.0
    return {
        "resolved_decisions": len(rows_list),
        "actionable_decisions": actionable,
        "symbols": len([symbol for symbol in symbols if symbol]),
        "net_return_bps": round(net, 4),
        "mean_return_bps": round(mean, 4),
        "hit_rate": round(hit_rate, 4),
        "profit_factor": round(profit_factor, 4),
        "max_drawdown_bps": round(max_drawdown, 4),
        "max_symbol_concentration": round(concentration, 4),
    }


def _fetch_latest_prices(symbols: tuple[str, ...]) -> dict[str, float]:
    from quant.config import BinanceAPIConfig
    from quant.data.binance_client import BinanceClient
    from quant_v2.telebot.signal_manager import V2SignalManager

    prices: dict[str, float] = {}
    for symbol in symbols:
        cfg = BinanceAPIConfig(
            base_url="https://fapi.binance.com",
            symbol=symbol,
            interval="1h",
        )
        client = BinanceClient(config=cfg)
        prices[symbol] = V2SignalManager._fetch_realtime_symbol_price(client, symbol, "1h")
    return prices


def _directional_return_bps(*, signal: str, entry_price: float, future_price: float) -> float:
    clean = str(signal).strip().upper()
    if clean == "BUY":
        return ((future_price / entry_price) - 1.0) * 10000.0
    if clean == "SELL":
        return ((entry_price / future_price) - 1.0) * 10000.0
    return 0.0


def _max_drawdown_bps(returns: list[float]) -> float:
    equity = 0.0
    peak = 0.0
    max_drawdown = 0.0
    for value in returns:
        equity += float(value)
        peak = max(peak, equity)
        max_drawdown = min(max_drawdown, equity - peak)
    return max_drawdown


def _config_hash(keys: tuple[str, ...]) -> str:
    payload = {key: os.getenv(key, "") for key in keys}
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def _parse_datetime(raw: str) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return bool(default)
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, str(default)).strip() or str(default)
    try:
        return int(raw)
    except ValueError:
        return int(default)


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name, str(default)).strip() or str(default)
    try:
        return float(raw)
    except ValueError:
        return float(default)


def _float(value: object, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _float_or_none(value: object) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


if __name__ == "__main__":
    run_scheduler_loop()
