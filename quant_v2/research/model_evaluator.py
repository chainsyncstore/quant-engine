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

from quant_v2.config import default_universe_symbols
from quant_v2.model_registry import ModelRegistry, ModelVersionRecord

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


def summarize_quarantine(
    db_path: Path | str,
    *,
    quarantine_version_id: str,
    policy: EvaluationPolicy,
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

    by_role: dict[str, list[sqlite3.Row]] = {"candidate": [], "incumbent": []}
    for row in rows:
        role = str(row["decision_role"] or "").lower()
        if role in by_role:
            by_role[role].append(row)

    candidate = _metric_summary(by_role["candidate"], policy=policy)
    incumbent = _metric_summary(by_role["incumbent"], policy=policy)
    delta = {
        "net_return_bps": candidate["net_return_bps"] - incumbent["net_return_bps"],
        "mean_return_bps": candidate["mean_return_bps"] - incumbent["mean_return_bps"],
        "max_drawdown_bps": candidate["max_drawdown_bps"] - incumbent["max_drawdown_bps"],
        "actionable_decisions": candidate["actionable_decisions"] - incumbent["actionable_decisions"],
    }

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
            "min_edge_bps": policy.min_edge_bps,
            "max_drawdown_worse_bps": policy.max_drawdown_worse_bps,
            "round_trip_cost_bps": policy.round_trip_cost_bps,
            "max_symbol_concentration": policy.max_symbol_concentration,
        },
        "candidate_metrics": candidate,
        "incumbent_metrics": incumbent,
        "delta_metrics": delta,
        "resolved_decisions": len(rows),
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
    """Load evaluator runtime control flags from registry storage."""

    path = Path(registry_root).expanduser() / CONTROL_FILE_NAME
    if not path.exists():
        return {"auto_promote": _env_bool("MODEL_EVAL_AUTO_PROMOTE", False)}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"auto_promote": _env_bool("MODEL_EVAL_AUTO_PROMOTE", False)}
    payload.setdefault("auto_promote", _env_bool("MODEL_EVAL_AUTO_PROMOTE", False))
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
    payload = {
        "auto_promote": bool(auto_promote),
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
    effective_auto_promote = bool(control.get("auto_promote", False))
    if auto_promote is not None:
        effective_auto_promote = bool(auto_promote)

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

    prices = _fetch_latest_prices(symbols)
    resolved = resolve_due_shadow_decisions(db_path, prices=prices)
    runtime = get_runtime_blockers(db_path)

    evaluated: list[dict[str, Any]] = []
    for candidate in candidates:
        summary = summarize_quarantine(
            db_path,
            quarantine_version_id=candidate.version_id,
            policy=policy,
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
