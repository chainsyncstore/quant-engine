"""Deterministic executable portfolio replay for v2 research and validation."""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

import pandas as pd

from quant.features.pipeline import build_features, get_feature_columns
from quant_v2.accounting import AccountingStore
from quant_v2.accounting import LedgerDifference, LedgerReconciliationReport
from quant_v2.contracts import MarketRiskSnapshot, StrategySignal
from quant_v2.execution.cost_policy import ExecutionCostPolicy
from quant_v2.execution.planner import PlannerConfig, build_execution_intents, intents_to_order_plans
from quant_v2.execution.reconciler import reconcile_ledger_state
from quant_v2.models.predictor import predict_proba
from quant_v2.models.trainer import TrainedModel
from quant_v2.portfolio.risk_policy import HardRiskLimits, OperatingRiskLimits

_REPLAY_LOGGER_NAMES = (
    "quant_v2.portfolio.allocation",
    "quant.features.pipeline",
    "quant_v2.models.trainer",
)


class _ResearchAccountingRecorder:
    """Fast no-op recorder for research replay runs.

    The full SQLite-backed store remains the default path so existing tests and
    production-style replay semantics stay unchanged.  Recovery experiments can
    opt into this recorder to avoid per-event database overhead while still
    driving the same signal, planner, order, fill, and equity logic.
    """

    def append_lifecycle_event(self, **_: Any) -> None:
        return None

    def append_order(self, **_: Any) -> None:
        return None

    def append_fill(self, **_: Any) -> None:
        return None

    def append_fee(self, **_: Any) -> None:
        return None

    def append_mark(self, **_: Any) -> None:
        return None


def _research_reconciliation_report(
    *,
    account_id: int,
    positions: dict[str, float],
    open_orders: dict[str, float],
    cash_usd: float,
    mark_prices: dict[str, float],
    projection_sequence_no: int,
) -> LedgerReconciliationReport:
    ledger_positions = {symbol: float(qty) for symbol, qty in positions.items()}
    equity_usd = float(cash_usd + sum(float(qty) * float(mark_prices.get(symbol, 0.0)) for symbol, qty in positions.items()))
    return LedgerReconciliationReport(
        account_id=account_id,
        status="research_mode_skipped",
        blocked_new_exposure=bool(open_orders),
        ledger_positions=ledger_positions,
        external_positions=dict(ledger_positions),
        checkpoint_positions=dict(ledger_positions),
        cash_delta_usd=0.0,
        equity_delta_usd=0.0,
        differences=(),
        legacy_unverifiable_count=0,
        projection_sequence_no=projection_sequence_no,
    )


def _model_threshold_floor(model: TrainedModel | None) -> float | None:
    if model is None:
        return None

    manifest = dict(getattr(model, "artifact_manifest", {}) or {})
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

    selected_threshold = training.get("threshold")
    if selected_threshold is not None:
        try:
            return float(selected_threshold)
        except (TypeError, ValueError):
            return None

    return None


@dataclass(frozen=True)
class ReplayScenario:
    """Deterministic replay perturbations."""

    name: str = "base"
    spread_multiplier: float = 1.0
    latency_bars: float = 0.0
    fill_ratio: float = 1.0
    reject_symbols: tuple[str, ...] = ()
    mark_jump_bps: float = 0.0
    data_gaps: dict[str, tuple[str, ...]] = field(default_factory=dict)
    restart_after_bars: int | None = None

    def __post_init__(self) -> None:
        if self.spread_multiplier <= 0.0:
            raise ValueError("spread_multiplier must be positive")
        if not 0.0 <= self.fill_ratio <= 1.0:
            raise ValueError("fill_ratio must be within [0, 1]")
        if self.latency_bars < 0.0:
            raise ValueError("latency_bars cannot be negative")


@dataclass(frozen=True)
class ReplayActorConfig:
    """Configuration for one isolated replay actor."""

    name: str
    kind: str = "model"
    model: TrainedModel | None = None
    threshold: float = 0.55
    min_confidence: float = 0.55
    horizon_bars: int = 4
    baseline_lookback: int = 4
    baseline_deadband: float = 0.001
    target_headroom_ratio: float = 0.85
    max_symbol_exposure_frac: float = 0.15
    max_gross_exposure_frac: float = 1.0
    max_net_exposure_frac: float = 0.50
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("name cannot be empty")
        threshold = float(self.threshold)
        if self.kind == "model" and self.model is not None and abs(threshold - 0.55) < 1e-12:
            model_floor = _model_threshold_floor(self.model)
            if model_floor is not None:
                threshold = max(threshold, model_floor)
        object.__setattr__(self, "threshold", threshold)
        if not 0.0 < self.threshold <= 1.0:
            raise ValueError("threshold must be within (0, 1]")
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError("min_confidence must be within [0, 1]")
        if self.horizon_bars <= 0:
            raise ValueError("horizon_bars must be positive")
        if self.baseline_lookback <= 0:
            raise ValueError("baseline_lookback must be positive")
        if not 0.0 < self.target_headroom_ratio <= 1.0:
            raise ValueError("target_headroom_ratio must be within (0, 1]")

    def manifest(self) -> dict[str, Any]:
        model_manifest = {}
        if self.model is not None:
            model_manifest = dict(getattr(self.model, "artifact_manifest", {}) or {})
        return {
            "name": self.name,
            "kind": self.kind,
            "threshold": float(self.threshold),
            "min_confidence": float(self.min_confidence),
            "horizon_bars": int(self.horizon_bars),
            "baseline_lookback": int(self.baseline_lookback),
            "baseline_deadband": float(self.baseline_deadband),
            "target_headroom_ratio": float(self.target_headroom_ratio),
            "max_symbol_exposure_frac": float(self.max_symbol_exposure_frac),
            "max_gross_exposure_frac": float(self.max_gross_exposure_frac),
            "max_net_exposure_frac": float(self.max_net_exposure_frac),
            "metadata": dict(self.metadata),
            "model_manifest": model_manifest,
        }


SignalResolver = Callable[
    [ReplayActorConfig, str, pd.DataFrame, pd.Timestamp, MarketRiskSnapshot | None],
    StrategySignal | None,
]


@dataclass(frozen=True)
class ReplayFill:
    actor: str
    timestamp: str
    symbol: str
    side: str
    requested_qty: float
    filled_qty: float
    price: float
    fee_usd: float
    slippage_usd: float
    outcome: str
    note: str = ""


@dataclass(frozen=True)
class ReplayActorResult:
    actor: str
    metrics: dict[str, Any]
    equity_curve: list[dict[str, Any]]
    fills: list[ReplayFill]
    blocked_intents: list[dict[str, Any]]
    risk_transitions: list[dict[str, Any]]
    reconciliation: dict[str, Any]
    manifest: dict[str, Any]
    state_digest: str


@dataclass(frozen=True)
class ReplayResult:
    replay_digest: str
    manifest: dict[str, Any]
    actors: dict[str, ReplayActorResult]
    timestamp_count: int
    event_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "replay_digest": self.replay_digest,
            "manifest": self.manifest,
            "timestamp_count": self.timestamp_count,
            "event_count": self.event_count,
            "actors": {
                name: {
                    **asdict(result),
                    "fills": [asdict(fill) for fill in result.fills],
                }
                for name, result in self.actors.items()
            },
        }


def _canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def _sha256_payload(payload: Any) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _frame_digest(frame: pd.DataFrame) -> str:
    ordered = frame.sort_index().sort_index(axis=1)
    hashed = pd.util.hash_pandas_object(ordered, index=True).values.tobytes()
    payload = {
        "columns": list(ordered.columns),
        "dtypes": {str(k): str(v) for k, v in ordered.dtypes.items()},
        "hash": hashlib.sha256(hashed).hexdigest(),
    }
    return _sha256_payload(payload)


def _extract_model_manifest(actor: ReplayActorConfig) -> dict[str, Any]:
    if actor.model is None:
        return {}
    return dict(getattr(actor.model, "artifact_manifest", {}) or {})


def _build_market_risk_snapshot(
    histories: dict[str, pd.DataFrame],
    *,
    lookback_hours: int = 24,
) -> MarketRiskSnapshot | None:
    returns: dict[str, float] = {}
    for symbol, history in histories.items():
        if isinstance(history, pd.DataFrame) and "close" in history.columns:
            close_raw = history["close"]
        else:
            close_raw = getattr(history, "get", lambda *_: None)("close")
        close_series = pd.Series(close_raw, dtype="float64") if not isinstance(close_raw, pd.Series) else close_raw
        close = pd.to_numeric(close_series, errors="coerce").dropna()
        if len(close) < lookback_hours + 1:
            continue
        start = float(close.iloc[-(lookback_hours + 1)])
        end = float(close.iloc[-1])
        if start <= 0.0 or end <= 0.0:
            continue
        returns[symbol] = (end / start) - 1.0

    if not returns:
        return None

    values = list(returns.values())
    down_ratio = sum(1 for value in values if value < 0.0) / len(values)
    median_return = float(pd.Series(values).median())
    btc_return = returns.get("BTCUSDT")
    broad_selloff = (
        down_ratio >= 0.7
        and median_return <= -0.015
        and (btc_return is None or btc_return <= -0.02)
    )
    return MarketRiskSnapshot(
        lookback_hours=lookback_hours,
        symbols_evaluated=len(values),
        down_ratio=float(down_ratio),
        median_return=median_return,
        btc_return=float(btc_return) if btc_return is not None else None,
        broad_selloff=broad_selloff,
    )


def _baseline_signal(
    actor: ReplayActorConfig,
    symbol: str,
    history: pd.DataFrame,
    *,
    market_risk: MarketRiskSnapshot | None,
) -> StrategySignal:
    close = pd.to_numeric(history.get("close"), errors="coerce").dropna()
    if len(close) <= actor.baseline_lookback:
        return StrategySignal(
            symbol=symbol,
            timeframe="1h",
            horizon_bars=actor.horizon_bars,
            signal="HOLD",
            confidence=0.5,
            uncertainty=0.5,
            reason="insufficient_history",
            market_risk=market_risk,
        )

    prior = float(close.iloc[-(actor.baseline_lookback + 1)])
    latest = float(close.iloc[-1])
    momentum = (latest / prior) - 1.0 if prior > 0.0 else 0.0
    if abs(momentum) < actor.baseline_deadband:
        signal = "HOLD"
        confidence = 0.5
    elif momentum > 0.0:
        signal = "BUY"
        confidence = min(0.95, 0.5 + abs(momentum) * 20.0)
    else:
        signal = "SELL"
        confidence = min(0.95, 0.5 + abs(momentum) * 20.0)
    return StrategySignal(
        symbol=symbol,
        timeframe="1h",
        horizon_bars=actor.horizon_bars,
        signal=signal,
        confidence=float(confidence),
        uncertainty=max(0.0, 1.0 - float(confidence)),
        reason=f"momentum_{actor.baseline_lookback}",
        market_risk=market_risk,
    )


def _model_signal(
    actor: ReplayActorConfig,
    symbol: str,
    history: pd.DataFrame,
    *,
    market_risk: MarketRiskSnapshot | None,
    featured: pd.DataFrame | None = None,
) -> StrategySignal:
    if actor.model is None:
        return _baseline_signal(actor, symbol, history, market_risk=market_risk)

    featured = featured if featured is not None else build_features(history.copy())
    if featured.empty:
        return StrategySignal(
            symbol=symbol,
            timeframe="1h",
            horizon_bars=actor.horizon_bars,
            signal="HOLD",
            confidence=0.5,
            uncertainty=0.5,
            reason="no_feature_rows",
            market_risk=market_risk,
        )

    feature_cols = get_feature_columns(featured)
    if not feature_cols:
        return StrategySignal(
            symbol=symbol,
            timeframe="1h",
            horizon_bars=actor.horizon_bars,
            signal="HOLD",
            confidence=0.5,
            uncertainty=0.5,
            reason="no_feature_columns",
            market_risk=market_risk,
        )

    feature_row = featured.iloc[[-1]][feature_cols]
    proba = float(predict_proba(actor.model, feature_row)[0])
    if proba >= actor.threshold:
        signal = "BUY"
        confidence = proba
        reason = f"proba_up>={actor.threshold:.2f}"
    elif proba <= (1.0 - actor.threshold):
        signal = "SELL"
        confidence = 1.0 - proba
        reason = f"proba_down>={actor.threshold:.2f}"
    else:
        signal = "HOLD"
        confidence = max(proba, 1.0 - proba)
        reason = "inside_deadband"

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


def _load_event_sequence(dataset: pd.DataFrame) -> tuple[list[pd.Timestamp], list[str]]:
    timestamps = [pd.Timestamp(ts) for ts in dataset.index.get_level_values("timestamp").unique()]
    timestamps = sorted(pd.DatetimeIndex(timestamps).to_list())
    symbols = sorted(str(sym) for sym in dataset.index.get_level_values("symbol").unique())
    return timestamps, symbols


def _build_replay_frame_cache(
    dataset: pd.DataFrame,
) -> tuple[dict[str, pd.DataFrame], dict[pd.Timestamp, pd.DataFrame]]:
    symbol_frames: dict[str, pd.DataFrame] = {}
    for symbol in dataset.index.get_level_values("symbol").unique():
        symbol_frame = dataset.xs(symbol, level="symbol").sort_index()
        symbol_frame.index = pd.DatetimeIndex(symbol_frame.index)
        symbol_frames[str(symbol)] = symbol_frame

    timestamp_rows: dict[pd.Timestamp, pd.DataFrame] = {}
    for timestamp in dataset.index.get_level_values("timestamp").unique():
        timestamp_frame = dataset.xs(timestamp, level="timestamp").sort_index()
        timestamp_frame.index = pd.Index(timestamp_frame.index)
        timestamp_rows[pd.Timestamp(timestamp)] = timestamp_frame

    return symbol_frames, timestamp_rows


def _asof_symbol_history(history: pd.DataFrame, *, timestamp: pd.Timestamp) -> pd.DataFrame:
    if history.empty:
        return history
    if not isinstance(history.index, pd.DatetimeIndex):
        hist = history.copy()
        hist.index = pd.to_datetime(hist.index, utc=True)
        return hist.loc[:timestamp]
    return history.loc[:timestamp]


def _safe_price(row: pd.Series, *, mark_jump_bps: float) -> float:
    price = float(row.get("close", 0.0) or 0.0)
    if price <= 0.0:
        return 0.0
    return float(price * (1.0 + (mark_jump_bps / 10_000.0)))


def _build_replay_manifest(
    *,
    dataset: pd.DataFrame,
    actors: Mapping[str, ReplayActorConfig],
    scenario: ReplayScenario,
    dataset_manifest: dict[str, Any] | None,
    policy_manifest: dict[str, Any] | None,
    code_sha: str | None,
) -> dict[str, Any]:
    actor_manifests = {name: actor.manifest() for name, actor in sorted(actors.items())}
    return {
        "schema_version": "wp11-portfolio-replay-v1",
        "dataset_digest": _frame_digest(dataset),
        "dataset_manifest": dict(dataset_manifest or {}),
        "actors": actor_manifests,
        "scenario": asdict(scenario),
        "policy_manifest": dict(policy_manifest or {}),
        "code_sha": code_sha,
    }


def _git_sha() -> str | None:
    try:
        import subprocess

        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            cwd=Path(__file__).resolve().parents[2],
        )
    except Exception:
        return None
    return result.stdout.strip() or None


def _replay_actor(
    *,
    actor: ReplayActorConfig,
    dataset: pd.DataFrame,
    symbol_frames: dict[str, pd.DataFrame] | None,
    timestamp_rows: dict[pd.Timestamp, pd.DataFrame] | None,
    scenario: ReplayScenario,
    timestamps: list[pd.Timestamp],
    symbols: list[str],
    cost_policy: ExecutionCostPolicy,
    policy: OperatingRiskLimits,
    initial_equity: float,
    research_mode: bool = False,
    signal_resolver: SignalResolver | None = None,
) -> ReplayActorResult:
    store: AccountingStore | _ResearchAccountingRecorder
    if research_mode:
        store = _ResearchAccountingRecorder()
    else:
        store = AccountingStore("sqlite:///:memory:")
    account_id = abs(hash(actor.name)) % 2_000_000_000 + 1
    reject_symbols = set(scenario.reject_symbols)
    data_gaps = {symbol: set(gaps) for symbol, gaps in scenario.data_gaps.items()}

    cash_usd = float(initial_equity)
    positions: dict[str, float] = {symbol: 0.0 for symbol in symbols}
    open_orders: dict[str, float] = {}
    equity_curve: list[dict[str, Any]] = []
    fills: list[ReplayFill] = []
    blocked_intents: list[dict[str, Any]] = []
    risk_transitions: list[dict[str, Any]] = []
    symbol_frames = symbol_frames or {
        str(symbol): dataset.xs(symbol, level="symbol").sort_index()
        for symbol in dataset.index.get_level_values("symbol").unique()
    }
    timestamp_rows = timestamp_rows or {
        pd.Timestamp(ts): dataset.xs(ts, level="timestamp").sort_index()
        for ts in dataset.index.get_level_values("timestamp").unique()
    }
    history_lengths: dict[str, int] = {symbol: 0 for symbol in symbols}
    mark_prices: dict[str, float] = {symbol: 0.0 for symbol in symbols}
    last_risk_state = "ACTIVE"
    event_count = 0

    store.append_lifecycle_event(
        account_id=account_id,
        state="ACTIVE",
        owner=actor.name,
        reason="replay_start",
        policy_version=policy.policy_version,
    )
    event_count += 1

    for ts_idx, timestamp in enumerate(timestamps):
        if scenario.restart_after_bars is not None and ts_idx == int(scenario.restart_after_bars):
            open_orders = {}
            store.append_lifecycle_event(
                account_id=account_id,
                state="RESTART",
                owner=actor.name,
                reason="scenario_restart",
                policy_version=policy.policy_version,
                created_at=timestamp.to_pydatetime(),
            )
            event_count += 1

        rows = timestamp_rows.get(timestamp)
        if rows is None:
            rows = dataset.loc[pd.IndexSlice[timestamp, :]].sort_index()
        if isinstance(rows, pd.Series):
            rows = rows.to_frame().T

        for symbol in symbols:
            if symbol not in rows.index:
                continue
            if symbol in data_gaps and timestamp.isoformat() in data_gaps[symbol]:
                continue
            row = rows.loc[symbol]
            history_lengths[symbol] = history_lengths.get(symbol, 0) + 1

            if open_orders.get(symbol):
                pending_qty = open_orders[symbol]
                side = "BUY" if pending_qty > 0 else "SELL"
                price = _safe_price(row, mark_jump_bps=scenario.mark_jump_bps)
                if price > 0.0 and symbol not in reject_symbols:
                    requested_qty = abs(pending_qty)
                    filled_qty = requested_qty * scenario.fill_ratio
                    if filled_qty > 0.0:
                        notional = filled_qty * price
                        estimates = cost_policy.estimate_fill_cost(
                            symbol,
                            side,
                            notional,
                            adv_usd=float(row.get("quote_volume", 0.0) or 0.0) or None,
                            latency_bars=scenario.latency_bars,
                            scenario="base",
                        )
                        fee_usd = estimates.fee_usd * scenario.spread_multiplier
                        slip_usd = (estimates.spread_usd + estimates.slippage_usd + estimates.impact_usd) * scenario.spread_multiplier
                        signed_fill = filled_qty if side == "BUY" else -filled_qty
                        positions[symbol] = positions.get(symbol, 0.0) + signed_fill
                        cash_usd -= fee_usd + slip_usd
                        cash_usd -= signed_fill * price
                        open_orders[symbol] = pending_qty - signed_fill
                        if abs(open_orders[symbol]) <= 1e-12:
                            open_orders.pop(symbol, None)
                        store.append_order(
                            account_id=account_id,
                            symbol=symbol,
                            request_id=f"{actor.name}:{timestamp.isoformat()}:{symbol}:open",
                            side=side,
                            requested_qty=requested_qty,
                            order_status="FILLED" if filled_qty >= requested_qty else "PARTIAL",
                            outcome="PARTIAL_FILL" if filled_qty < requested_qty else "NEW_FILL",
                            occurred_at=timestamp.to_pydatetime(),
                            source_event_id=f"{actor.name}:{timestamp.isoformat()}:{symbol}:open:order",
                        )
                        store.append_fill(
                            account_id=account_id,
                            symbol=symbol,
                            side=side,
                            requested_qty=requested_qty,
                            newly_filled_qty=filled_qty,
                            cumulative_qty=filled_qty,
                            avg_price=price,
                            fees_usd=fee_usd,
                            fill_id=f"{actor.name}:{timestamp.isoformat()}:{symbol}:open:fill",
                            request_id=f"{actor.name}:{timestamp.isoformat()}:{symbol}:open",
                            outcome="PARTIAL_FILL" if filled_qty < requested_qty else "NEW_FILL",
                            source_event_id=f"{actor.name}:{timestamp.isoformat()}:{symbol}:open:fill",
                        )
                        if fee_usd > 0.0:
                            store.append_fee(
                                account_id=account_id,
                                amount_usd=fee_usd,
                                fee_type="execution",
                                symbol=symbol,
                                occurred_at=timestamp.to_pydatetime(),
                                source_event_id=f"{actor.name}:{timestamp.isoformat()}:{symbol}:open:fee",
                            )
                        fills.append(
                            ReplayFill(
                                actor=actor.name,
                                timestamp=timestamp.isoformat(),
                                symbol=symbol,
                                side=side,
                                requested_qty=requested_qty,
                                filled_qty=filled_qty,
                                price=price,
                                fee_usd=fee_usd,
                                slippage_usd=slip_usd,
                                outcome="PARTIAL_FILL" if filled_qty < requested_qty else "NEW_FILL",
                                note="open_order_fill",
                            )
                        )
                        event_count += 3

            if symbol not in rows.index:
                continue

        current_histories = {
            symbol: symbol_frames[symbol].iloc[:history_lengths[symbol]]
            for symbol in symbols
            if history_lengths.get(symbol, 0) > 0 and symbol in symbol_frames
        }

        market_risk = _build_market_risk_snapshot(current_histories)
        featured_by_symbol: dict[str, pd.DataFrame] = {}
        for symbol in symbols:
            hist = current_histories.get(symbol)
            if hist is None or hist.empty:
                continue
            hist = _asof_symbol_history(hist, timestamp=timestamp)
            if hist.empty:
                continue
            if actor.kind == "model" and signal_resolver is None:
                featured_by_symbol[symbol] = build_features(hist.copy())

        signals: list[StrategySignal] = []
        for symbol in symbols:
            hist = current_histories.get(symbol)
            if hist is None or hist.empty:
                continue
            hist = _asof_symbol_history(hist, timestamp=timestamp)
            if hist.empty:
                continue
            signal = None
            if signal_resolver is not None:
                signal = signal_resolver(actor, symbol, hist, timestamp, market_risk)
            if signal is None:
                if actor.kind == "fixed":
                    signal = StrategySignal(
                        symbol=symbol,
                        timeframe="1h",
                        horizon_bars=actor.horizon_bars,
                        signal="HOLD",
                        confidence=0.5,
                        uncertainty=0.5,
                        reason="fixed_signal_unavailable",
                        market_risk=market_risk,
                    )
                elif actor.kind != "model":
                    signal = _baseline_signal(actor, symbol, hist, market_risk=market_risk)
                else:
                    signal = _model_signal(
                        actor,
                        symbol,
                        hist,
                        market_risk=market_risk,
                        featured=featured_by_symbol.get(symbol),
                    )
            signals.append(
                StrategySignal(
                    symbol=signal.symbol,
                    timeframe=signal.timeframe,
                    horizon_bars=signal.horizon_bars,
                    signal=signal.signal,
                    confidence=signal.confidence,
                    uncertainty=signal.uncertainty,
                    reason=signal.reason,
                    regime=signal.regime,
                    session_hour_utc=int(timestamp.hour),
                    market_risk=signal.market_risk,
                )
            )

        projected_positions = dict(positions)
        for symbol, qty in open_orders.items():
            projected_positions[symbol] = projected_positions.get(symbol, 0.0) + qty

        planner_config = PlannerConfig(
            equity_usd=max(cash_usd + sum(
                float(position_qty) * _safe_price(current_histories[symbol].iloc[-1], mark_jump_bps=scenario.mark_jump_bps)
                if symbol in current_histories and not current_histories[symbol].empty else 0.0
                for symbol, position_qty in projected_positions.items()
            ), 1.0),
            min_confidence=actor.min_confidence,
            target_headroom_ratio=actor.target_headroom_ratio,
            max_symbol_exposure_frac=actor.max_symbol_exposure_frac,
        )
        intent_plan = build_execution_intents(
            signals,
            policy=policy,
            config=planner_config,
            current_positions=projected_positions,
        )
        prices = {
            symbol: _safe_price(current_histories[symbol].iloc[-1], mark_jump_bps=scenario.mark_jump_bps)
            for symbol in symbols
            if symbol in current_histories and not current_histories[symbol].empty
        }
        order_plans = intents_to_order_plans(
            intent_plan.intents,
            prices=prices,
            equity_usd=planner_config.equity_usd,
            min_qty=0.0,
        )

        state_name = "REDUCE_ONLY" if intent_plan.policy_result.constraints_applied else "ACTIVE"
        if state_name != last_risk_state:
            risk_transitions.append(
                {
                    "timestamp": timestamp.isoformat(),
                    "from": last_risk_state,
                    "to": state_name,
                    "reason": ",".join(intent_plan.policy_result.constraints_applied) or "no_constraints",
                }
            )
            last_risk_state = state_name

        for intent in intent_plan.intents:
            if intent.symbol in open_orders:
                blocked_intents.append(
                    {
                        "timestamp": timestamp.isoformat(),
                        "symbol": intent.symbol,
                        "reason": "open_order_pending",
                    }
                )
                continue

        for order in order_plans:
            if order.symbol in data_gaps and timestamp.isoformat() in data_gaps[order.symbol]:
                blocked_intents.append(
                    {
                        "timestamp": timestamp.isoformat(),
                        "symbol": order.symbol,
                        "reason": "data_gap",
                    }
                )
                continue

            price = prices.get(order.symbol, 0.0)
            if price <= 0.0:
                blocked_intents.append(
                    {
                        "timestamp": timestamp.isoformat(),
                        "symbol": order.symbol,
                        "reason": "missing_price",
                    }
                )
                continue
            if order.symbol in reject_symbols:
                store.append_order(
                    account_id=account_id,
                    symbol=order.symbol,
                    request_id=f"{actor.name}:{timestamp.isoformat()}:{order.symbol}:{order.side}",
                    side=order.side,
                    requested_qty=order.quantity,
                    order_status="REJECTED",
                    outcome="VENUE_REJECTED",
                    occurred_at=timestamp.to_pydatetime(),
                    source_event_id=f"{actor.name}:{timestamp.isoformat()}:{order.symbol}:{order.side}:reject",
                )
                blocked_intents.append(
                    {
                        "timestamp": timestamp.isoformat(),
                        "symbol": order.symbol,
                        "reason": "venue_rejected",
                    }
                )
                event_count += 1
                continue

            filled_qty = order.quantity * scenario.fill_ratio
            if filled_qty <= 0.0:
                blocked_intents.append(
                    {
                        "timestamp": timestamp.isoformat(),
                        "symbol": order.symbol,
                        "reason": "zero_fill",
                    }
                )
                continue

            notional = filled_qty * price
            estimates = cost_policy.estimate_fill_cost(
                order.symbol,
                order.side,
                notional,
                adv_usd=float(current_histories[order.symbol].iloc[-1].get("quote_volume", 0.0) or 0.0) or None,
                latency_bars=scenario.latency_bars,
                scenario="base",
            )
            fee_usd = estimates.fee_usd * scenario.spread_multiplier
            slip_usd = (estimates.spread_usd + estimates.slippage_usd + estimates.impact_usd) * scenario.spread_multiplier
            signed_fill = filled_qty if order.side == "BUY" else -filled_qty
            positions[order.symbol] = positions.get(order.symbol, 0.0) + signed_fill
            cash_usd -= fee_usd + slip_usd
            cash_usd -= signed_fill * price

            remainder = order.quantity - filled_qty
            if remainder > 1e-12:
                open_orders[order.symbol] = remainder if order.side == "BUY" else -remainder
            else:
                open_orders.pop(order.symbol, None)

            store.append_order(
                account_id=account_id,
                symbol=order.symbol,
                request_id=f"{actor.name}:{timestamp.isoformat()}:{order.symbol}:{order.side}",
                side=order.side,
                requested_qty=order.quantity,
                order_status="FILLED" if remainder <= 1e-12 else "PARTIAL",
                outcome="PARTIAL_FILL" if remainder > 1e-12 else "NEW_FILL",
                occurred_at=timestamp.to_pydatetime(),
                source_event_id=f"{actor.name}:{timestamp.isoformat()}:{order.symbol}:{order.side}:order",
                extra={"reduce_only": bool(order.reduce_only)},
            )
            store.append_fill(
                account_id=account_id,
                symbol=order.symbol,
                side=order.side,
                requested_qty=order.quantity,
                newly_filled_qty=filled_qty,
                cumulative_qty=filled_qty,
                avg_price=price,
                fees_usd=fee_usd,
                fill_id=f"{actor.name}:{timestamp.isoformat()}:{order.symbol}:{order.side}:fill",
                request_id=f"{actor.name}:{timestamp.isoformat()}:{order.symbol}:{order.side}",
                outcome="PARTIAL_FILL" if remainder > 1e-12 else "NEW_FILL",
                source_event_id=f"{actor.name}:{timestamp.isoformat()}:{order.symbol}:{order.side}:fill",
            )
            if fee_usd > 0.0:
                store.append_fee(
                    account_id=account_id,
                    amount_usd=fee_usd,
                    fee_type="execution",
                    symbol=order.symbol,
                    occurred_at=timestamp.to_pydatetime(),
                    source_event_id=f"{actor.name}:{timestamp.isoformat()}:{order.symbol}:{order.side}:fee",
                )
            fills.append(
                ReplayFill(
                    actor=actor.name,
                    timestamp=timestamp.isoformat(),
                    symbol=order.symbol,
                    side=order.side,
                    requested_qty=order.quantity,
                    filled_qty=filled_qty,
                    price=price,
                    fee_usd=fee_usd,
                    slippage_usd=slip_usd,
                    outcome="PARTIAL_FILL" if remainder > 1e-12 else "NEW_FILL",
                )
            )
            event_count += 3

        for symbol in symbols:
            if symbol not in current_histories or current_histories[symbol].empty:
                continue
            price = _safe_price(current_histories[symbol].iloc[-1], mark_jump_bps=scenario.mark_jump_bps)
            mark_prices[symbol] = price
            store.append_mark(
                account_id=account_id,
                symbol=symbol,
                mark_price=price,
                marked_at=timestamp.to_pydatetime(),
                source_event_id=f"{actor.name}:{timestamp.isoformat()}:{symbol}:mark",
            )
            event_count += 1

        equity_value = cash_usd + sum(
            float(position_qty) * float(mark_prices.get(symbol, 0.0))
            for symbol, position_qty in positions.items()
        )
        equity_curve.append(
            {
                "timestamp": timestamp.isoformat(),
                "equity_usd": float(equity_value),
                "cash_usd": float(cash_usd),
                "positions": {sym: float(qty) for sym, qty in positions.items()},
                "open_orders": dict(open_orders),
            }
        )

    if research_mode:
        reconciliation = _research_reconciliation_report(
            account_id=account_id,
            positions=positions,
            open_orders=open_orders,
            cash_usd=cash_usd,
            mark_prices=mark_prices,
            projection_sequence_no=event_count,
        )
    else:
        reconciliation = reconcile_ledger_state(
            account_id,
            store=store,
            adapter_positions={symbol: float(qty) for symbol, qty in positions.items()},
            open_orders=dict(open_orders),
        )

    equity_series = pd.Series(
        [float(point["equity_usd"]) for point in equity_curve],
        index=pd.DatetimeIndex([pd.Timestamp(point["timestamp"]) for point in equity_curve]),
        dtype=float,
    )
    fill_pnls = [fill.filled_qty * fill.price - fill.fee_usd - fill.slippage_usd for fill in fills]
    gross_pnl = float(sum(fill_pnls))
    total_fees = float(sum(fill.fee_usd for fill in fills))
    total_slippage = float(sum(fill.slippage_usd for fill in fills))
    net_pnl = float(equity_series.iloc[-1] - initial_equity) if not equity_series.empty else 0.0
    turnover = float(sum(abs(fill.filled_qty * fill.price) for fill in fills) / max(initial_equity, 1e-9))
    max_equity = float(equity_series.cummax().max()) if not equity_series.empty else initial_equity
    min_equity = float(equity_series.min()) if not equity_series.empty else initial_equity

    final_equity = float(cash_usd + sum(float(position_qty) * float(mark_prices.get(symbol, 0.0)) for symbol, position_qty in positions.items()))
    metrics = {
        "final_equity_usd": final_equity,
        "gross_pnl_usd": gross_pnl,
        "net_pnl_usd": net_pnl,
        "total_fees_usd": total_fees,
        "total_slippage_usd": total_slippage,
        "turnover": turnover,
        "max_equity_usd": max_equity,
        "min_equity_usd": min_equity,
        "max_drawdown_frac": float(((equity_series - equity_series.cummax()) / equity_series.cummax()).min()) if not equity_series.empty else 0.0,
        "blocked_intents": len(blocked_intents),
        "risk_transition_count": len(risk_transitions),
        "fill_count": len(fills),
        "open_order_count": len(open_orders),
    }
    actor_manifest = actor.manifest()
    actor_payload = {
        "actor": actor.name,
        "metrics": metrics,
        "equity_curve": equity_curve,
        "fills": [asdict(fill) for fill in fills],
        "blocked_intents": blocked_intents,
        "risk_transitions": risk_transitions,
        "reconciliation": asdict(reconciliation),
        "manifest": actor_manifest,
    }
    state_digest = _sha256_payload(actor_payload)

    return ReplayActorResult(
        actor=actor.name,
        metrics=metrics,
        equity_curve=equity_curve,
        fills=fills,
        blocked_intents=blocked_intents,
        risk_transitions=risk_transitions,
        reconciliation=asdict(reconciliation),
        manifest=actor_manifest,
        state_digest=state_digest,
    )


def run_portfolio_replay(
    dataset: pd.DataFrame,
    actors: Mapping[str, ReplayActorConfig],
    *,
    initial_equity: float = 300.0,
    scenario: ReplayScenario | None = None,
    dataset_manifest: dict[str, Any] | None = None,
    policy_manifest: dict[str, Any] | None = None,
    cost_policy: ExecutionCostPolicy | None = None,
    hard_limits: HardRiskLimits | None = None,
    signal_resolver: SignalResolver | None = None,
    throttle_allocation_logs: bool = False,
    research_mode: bool = False,
) -> ReplayResult:
    """Replay a multi-symbol event stream with isolated actor state."""

    if not isinstance(dataset.index, pd.MultiIndex) or list(dataset.index.names) != ["timestamp", "symbol"]:
        raise ValueError("dataset must be MultiIndex with levels ['timestamp', 'symbol']")
    if not actors:
        raise ValueError("actors cannot be empty")
    if initial_equity <= 0.0:
        raise ValueError("initial_equity must be positive")

    scenario = scenario or ReplayScenario()
    cost_policy = cost_policy or ExecutionCostPolicy()
    hard_limits = hard_limits or HardRiskLimits()
    operating_limits = OperatingRiskLimits.from_hard_limits(
        hard_limits,
        target_headroom_ratio=0.85,
    )
    sorted_dataset = dataset.sort_index()
    timestamps, symbols = _load_event_sequence(sorted_dataset)
    symbol_frames, timestamp_rows = _build_replay_frame_cache(sorted_dataset)
    code_sha = _git_sha()
    manifest = _build_replay_manifest(
        dataset=dataset,
        actors=actors,
        scenario=scenario,
        dataset_manifest=dataset_manifest,
        policy_manifest={
            "risk_policy_version": operating_limits.policy_version,
            "cost_policy_version": cost_policy.policy_version,
            **dict(policy_manifest or {}),
        },
        code_sha=code_sha,
    )
    stable_manifest = dict(manifest)
    stable_manifest.pop("generated_at", None)

    replay_loggers = {name: logging.getLogger(name) for name in _REPLAY_LOGGER_NAMES}
    replay_previous_levels = {name: logger.level for name, logger in replay_loggers.items()}
    if throttle_allocation_logs:
        for logger in replay_loggers.values():
            logger.setLevel(logging.WARNING)
    try:
        actor_results = {
            name: _replay_actor(
                actor=actor,
                dataset=sorted_dataset,
                scenario=scenario,
                timestamps=timestamps,
                symbols=symbols,
                symbol_frames=symbol_frames,
                timestamp_rows=timestamp_rows,
                cost_policy=cost_policy,
                policy=operating_limits,
                initial_equity=initial_equity,
                research_mode=research_mode,
                signal_resolver=signal_resolver,
            )
            for name, actor in sorted(actors.items())
        }
    finally:
        if throttle_allocation_logs:
            for name, logger in replay_loggers.items():
                logger.setLevel(replay_previous_levels[name])

    payload = {
        "manifest": stable_manifest,
        "actors": {
            name: {
                "metrics": result.metrics,
                "state_digest": result.state_digest,
                "fill_count": len(result.fills),
                "equity_curve_digest": _sha256_payload(result.equity_curve),
            }
            for name, result in actor_results.items()
        },
        "timestamp_count": len(timestamps),
        "event_count": sum(len(result.fills) + len(result.blocked_intents) + len(result.risk_transitions) for result in actor_results.values()),
    }
    replay_digest = _sha256_payload(payload)
    return ReplayResult(
        replay_digest=replay_digest,
        manifest=manifest,
        actors=actor_results,
        timestamp_count=len(timestamps),
        event_count=int(payload["event_count"]),
    )


def write_replay_artifact(
    result: ReplayResult,
    output_dir: Path | str,
    *,
    prefix: str = "portfolio_replay",
) -> Path:
    """Write a content-addressed replay artifact."""

    output_dir = Path(output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{prefix}_{result.replay_digest}.json"
    payload = result.to_dict()
    payload["replay_digest"] = result.replay_digest
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return path
