from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from quant.telebot.models import Base, User, UserContext
from quant.telebot.auth import CryptoManager
from quant.telebot.model_selection import resolve_model_dir
from quant_v2.config import default_universe_symbols
from quant_v2.contracts import StrategySignal
from quant_v2.execution.service import RoutedExecutionService
from quant_v2.model_registry import ModelRegistry
from quant_v2.monitoring.health_dashboard import build_session_health_summary
from quant_v2.monitoring.kill_switch import MonitoringSnapshot
from quant_v2.telebot.bridge import V2ExecutionBridge, convert_legacy_signal_payload
from quant_v2.telebot.signal_manager import V2SignalManager

if TYPE_CHECKING:
    from quant.telebot.manager import BotManager

# Logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


def _resolve_execution_backend(raw_backend: str, *, allow_legacy_runtime: bool) -> str:
    """Resolve runtime backend with explicit legacy fallback semantics."""

    clean = (raw_backend or "").strip().lower()
    if not clean:
        clean = "v2_memory"

    if clean in {"v1", "v1_legacy"}:
        return "v1_legacy" if allow_legacy_runtime else "v2_memory"

    return clean


FOOTER = "\n\nℹ️ Run /help to see command list"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_ROOT = Path(
    os.getenv("BOT_MODEL_ROOT", str(PROJECT_ROOT / "models" / "production"))
).expanduser()
MODEL_REGISTRY_ROOT = Path(
    os.getenv("BOT_MODEL_REGISTRY_ROOT", str(MODEL_ROOT / "registry"))
).expanduser()
MASTER_KEY_PATH = Path(
    os.getenv("BOT_MASTER_KEY_FILE", str(PROJECT_ROOT / "quant_bot.master.key"))
).expanduser()
DB_PATH = Path(
    os.getenv("BOT_DB_PATH", str(PROJECT_ROOT / "quant_bot.db"))
).expanduser()
ALLOW_LEGACY_RUNTIME = os.getenv("BOT_ALLOW_LEGACY_RUNTIME", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
_RAW_EXECUTION_BACKEND = os.getenv("BOT_EXECUTION_BACKEND", "v2_memory").strip().lower()
EXECUTION_BACKEND = _resolve_execution_backend(
    _RAW_EXECUTION_BACKEND,
    allow_legacy_runtime=ALLOW_LEGACY_RUNTIME,
)
if _RAW_EXECUTION_BACKEND in {"v1", "v1_legacy"} and EXECUTION_BACKEND != "v1_legacy":
    logger.warning(
        "Legacy backend `%s` ignored because BOT_ALLOW_LEGACY_RUNTIME is disabled. "
        "Using `%s`.",
        _RAW_EXECUTION_BACKEND,
        EXECUTION_BACKEND,
    )
V2_ALLOW_LIVE_EXECUTION = os.getenv("BOT_V2_ALLOW_LIVE_EXECUTION", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
try:
    V2_SIGNAL_LOOP_SECONDS = max(
        int((os.getenv("BOT_V2_SIGNAL_LOOP_SECONDS", "3600").strip() or "3600")),
        1,
    )
except ValueError:
    V2_SIGNAL_LOOP_SECONDS = 3600
DEFAULT_V2_SYMBOL = default_universe_symbols()[0] if default_universe_symbols() else "BTCUSDT"


def _load_or_create_master_key() -> str:
    """Load BOT_MASTER_KEY from env or a persisted key file."""
    env_key = os.getenv("BOT_MASTER_KEY", "").strip()
    if env_key:
        return env_key

    if MASTER_KEY_PATH.exists():
        key = MASTER_KEY_PATH.read_text(encoding="utf-8").strip()
        if not key:
            raise RuntimeError(
                f"Persisted BOT_MASTER_KEY file is empty: {MASTER_KEY_PATH}. "
                "Set BOT_MASTER_KEY explicitly and restart."
            )
        os.environ["BOT_MASTER_KEY"] = key
        logger.warning(
            "BOT_MASTER_KEY not set; loaded persisted key from %s.",
            MASTER_KEY_PATH,
        )
        return key

    key = CryptoManager.generate_key()
    MASTER_KEY_PATH.parent.mkdir(parents=True, exist_ok=True)
    MASTER_KEY_PATH.write_text(key, encoding="utf-8")
    if os.name != "nt":
        os.chmod(MASTER_KEY_PATH, 0o600)

    os.environ["BOT_MASTER_KEY"] = key
    logger.warning(
        "BOT_MASTER_KEY not set; generated persistent key at %s. "
        "Set BOT_MASTER_KEY in deployment env for stronger ops hygiene.",
        MASTER_KEY_PATH,
    )
    return key

# ---------------------------------------------------------------------------
# Global State
# ---------------------------------------------------------------------------
MODEL_REGISTRY = ModelRegistry(MODEL_REGISTRY_ROOT)
MODEL_RESOLUTION = resolve_model_dir(MODEL_ROOT, MODEL_REGISTRY_ROOT)
MODEL_DIR = MODEL_RESOLUTION.model_dir

if MODEL_RESOLUTION.warning:
    logger.warning("%s", MODEL_RESOLUTION.warning)

if not MODEL_DIR:
    logger.warning(
        "No production model found under %s. Bot will fail to start trading.",
        MODEL_ROOT,
    )
else:
    source_label = MODEL_RESOLUTION.source
    if MODEL_RESOLUTION.active_version_id:
        source_label = f"{source_label}:{MODEL_RESOLUTION.active_version_id}"
    logger.info("Using model: %s (source=%s)", MODEL_DIR, source_label)

# Db - StaticPool + check_same_thread=False for SQLite in async context
DB_PATH.parent.mkdir(parents=True, exist_ok=True)
ENGINE = create_engine(
    f"sqlite:///{DB_PATH.resolve()}",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
SessionLocal = sessionmaker(bind=ENGINE)
Base.metadata.create_all(ENGINE)


def _ensure_user_context_schema() -> None:
    """Backfill required UserContext columns for existing SQLite deployments."""

    required: tuple[tuple[str, str], ...] = (
        ("strategy_profile", "VARCHAR DEFAULT 'core_v2'"),
        ("active_model_version", "VARCHAR"),
        ("active_model_source", "VARCHAR"),
    )

    with ENGINE.connect() as conn:
        existing = {
            str(row[1])
            for row in conn.execute(text("PRAGMA table_info(user_context)"))
        }
        for column_name, column_sql in required:
            if column_name in existing:
                continue
            conn.execute(text(f"ALTER TABLE user_context ADD COLUMN {column_name} {column_sql}"))
            logger.info("Added missing user_context column `%s`", column_name)
        conn.commit()


_ensure_user_context_schema()

# Enable WAL mode for better concurrent read/write
with ENGINE.connect() as conn:
    conn.execute(text("PRAGMA journal_mode=WAL"))
    conn.commit()

# Crypto
_load_or_create_master_key()

CRYPTO = CryptoManager()

# Manager
MANAGER: BotManager | None = None
V2_SIGNAL_MANAGER: V2SignalManager | None = None
V2_BRIDGE: V2ExecutionBridge | None = None
V2_DEGRADED_ALERTED_USERS: set[int] = set()


def _using_v2_backend() -> bool:
    return EXECUTION_BACKEND in {"v2", "v2_memory", "v2_shadow_memory"}


def _using_shadow_backend() -> bool:
    return EXECUTION_BACKEND == "v2_shadow_memory"


def _using_v2_primary_backend() -> bool:
    return EXECUTION_BACKEND in {"v2", "v2_memory"}


def _using_v1_primary_backend() -> bool:
    return EXECUTION_BACKEND in {"v1_legacy", "v2_shadow_memory"}


def _using_manager_signal_source() -> bool:
    return EXECUTION_BACKEND in {"v1_legacy", "v2_shadow_memory"}


def _is_admin_user(user_id: int) -> bool:
    """Return whether the Telegram user is configured as admin."""

    admin_id = os.getenv("ADMIN_ID", "").strip()
    return bool(admin_id) and str(user_id) == admin_id


def _resolve_runtime_model_dir() -> Path | None:
    """Resolve runtime model directory from active registry pointer or fallback discovery."""

    global MODEL_RESOLUTION
    global MODEL_DIR

    MODEL_RESOLUTION = resolve_model_dir(MODEL_ROOT, MODEL_REGISTRY_ROOT)
    MODEL_DIR = MODEL_RESOLUTION.model_dir
    if MODEL_RESOLUTION.warning:
        logger.warning("%s", MODEL_RESOLUTION.warning)
    return MODEL_DIR


def _get_v2_bridge() -> V2ExecutionBridge | None:
    """Get or lazily initialize v2 execution bridge when enabled."""
    global V2_BRIDGE

    if not _using_v2_backend():
        return None

    if V2_BRIDGE:
        return V2_BRIDGE

    if EXECUTION_BACKEND in {"v2", "v2_memory", "v2_shadow_memory"}:
        if EXECUTION_BACKEND == "v2":
            service = RoutedExecutionService(allow_live_execution=V2_ALLOW_LIVE_EXECUTION)
        elif EXECUTION_BACKEND in {"v2_memory", "v2_shadow_memory"}:
            service = RoutedExecutionService(allow_live_execution=False)

        V2_BRIDGE = V2ExecutionBridge(
            service,
            default_strategy_profile="core_v2",
            default_universe=default_universe_symbols(),
        )
        logger.info(
            "Initialized v2 execution bridge backend=%s universe=%s",
            EXECUTION_BACKEND,
            ",".join(default_universe_symbols()),
        )
        return V2_BRIDGE

    logger.error(
        "Unsupported BOT_EXECUTION_BACKEND=%s. Supported: v1_legacy, v2, v2_memory, v2_shadow_memory",
        EXECUTION_BACKEND,
    )
    return None


def _get_manager(*, allow_reload_with_active_sessions: bool = False) -> BotManager | None:
    """Get manager bound to the currently resolved active model pointer."""

    global MANAGER

    resolved_model_dir = _resolve_runtime_model_dir()
    if MANAGER is not None:
        if resolved_model_dir is None:
            # Keep current manager when resolution is temporarily unavailable.
            return MANAGER

        current_dir = Path(MANAGER.model_dir).expanduser().resolve()
        target_dir = Path(resolved_model_dir).expanduser().resolve()
        if current_dir == target_dir:
            return MANAGER

        if MANAGER.get_active_count() > 0 and not allow_reload_with_active_sessions:
            logger.warning(
                "Deferred manager model switch from %s to %s because active sessions are running.",
                current_dir,
                target_dir,
            )
            return MANAGER

    if resolved_model_dir is None:
        return None

    from quant.telebot.manager import BotManager as LegacyBotManager

    MANAGER = LegacyBotManager(resolved_model_dir)
    source_label = MODEL_RESOLUTION.source
    if MODEL_RESOLUTION.active_version_id:
        source_label = f"{source_label}:{MODEL_RESOLUTION.active_version_id}"
    logger.info("Initialized BotManager with model: %s (source=%s)", resolved_model_dir, source_label)
    return MANAGER


def _get_v2_signal_manager(*, allow_reload_with_active_sessions: bool = False) -> V2SignalManager | None:
    """Get native v2 signal manager bound to active model pointer."""

    global V2_SIGNAL_MANAGER

    if not _using_v2_primary_backend():
        return None

    resolved_model_dir = _resolve_runtime_model_dir()
    if resolved_model_dir is None:
        return None

    if V2_SIGNAL_MANAGER is not None:
        current_dir = Path(V2_SIGNAL_MANAGER.model_dir).expanduser().resolve()
        target_dir = Path(resolved_model_dir).expanduser().resolve()
        if current_dir == target_dir:
            return V2_SIGNAL_MANAGER

        if V2_SIGNAL_MANAGER.get_active_count() > 0 and not allow_reload_with_active_sessions:
            logger.warning(
                "Deferred native v2 signal manager model switch from %s to %s because active sessions are running.",
                current_dir,
                target_dir,
            )
            return V2_SIGNAL_MANAGER

    V2_SIGNAL_MANAGER = V2SignalManager(
        model_dir=resolved_model_dir,
        symbols=default_universe_symbols(),
        loop_interval_seconds=V2_SIGNAL_LOOP_SECONDS,
    )
    source_label = MODEL_RESOLUTION.source
    if MODEL_RESOLUTION.active_version_id:
        source_label = f"{source_label}:{MODEL_RESOLUTION.active_version_id}"
    logger.info(
        "Initialized V2SignalManager with model: %s (source=%s, loop=%ss)",
        resolved_model_dir,
        source_label,
        V2_SIGNAL_LOOP_SECONDS,
    )
    return V2_SIGNAL_MANAGER


def _get_signal_source_manager(*, allow_reload_with_active_sessions: bool = False):
    """Return active signal-source manager for current backend selection."""

    if _using_v2_primary_backend():
        return _get_v2_signal_manager(
            allow_reload_with_active_sessions=allow_reload_with_active_sessions
        )
    if _using_manager_signal_source():
        return _get_manager(
            allow_reload_with_active_sessions=allow_reload_with_active_sessions
        )
    return None


def _persist_user_session_flags(
    user_id: int,
    *,
    is_active: bool | None = None,
    live_mode: bool | None = None,
    strategy_profile: str | None = None,
    active_model_version: str | None = None,
    active_model_source: str | None = None,
) -> None:
    """Persist runtime session continuity and routing metadata for a user context."""
    session = SessionLocal()
    try:
        db_user = session.query(User).filter_by(telegram_id=user_id).first()
        if not db_user:
            return

        db_ctx = db_user.context
        changed = False
        if not db_ctx:
            db_ctx = UserContext(telegram_id=user_id)
            db_user.context = db_ctx
            changed = True

        if live_mode is not None and db_ctx.live_mode != live_mode:
            db_ctx.live_mode = live_mode
            changed = True
        if is_active is not None and db_ctx.is_active != is_active:
            db_ctx.is_active = is_active
            changed = True
        if strategy_profile is not None and db_ctx.strategy_profile != strategy_profile:
            db_ctx.strategy_profile = strategy_profile
            changed = True
        if active_model_version is not None and db_ctx.active_model_version != active_model_version:
            db_ctx.active_model_version = active_model_version
            changed = True
        if active_model_source is not None and db_ctx.active_model_source != active_model_source:
            db_ctx.active_model_source = active_model_source
            changed = True

        if changed:
            session.commit()
    except Exception as e:
        session.rollback()
        logger.warning(f"Failed to persist session flags for user {user_id}: {e}")
    finally:
        session.close()


def _resolve_runtime_metadata(*, bridge: V2ExecutionBridge | None) -> tuple[str, str | None, str]:
    """Return strategy/model metadata for session persistence."""

    strategy_profile = "legacy_v1"
    if bridge is not None and _using_v2_backend():
        strategy_profile = bridge.default_strategy_profile
    elif _using_shadow_backend():
        strategy_profile = "legacy_v1_shadow"

    _resolve_runtime_model_dir()
    active_model_version = MODEL_RESOLUTION.active_version_id
    active_model_source = MODEL_RESOLUTION.source
    return strategy_profile, active_model_version, active_model_source


def _build_creds_from_context(ctx: UserContext | None, *, live: bool) -> dict:
    """Build credential payload expected by BotManager.start_session()."""
    binance_key = CRYPTO.decrypt(ctx.binance_api_key) if ctx and ctx.binance_api_key else ""
    binance_secret = CRYPTO.decrypt(ctx.binance_api_secret) if ctx and ctx.binance_api_secret else ""

    if live and (not binance_key or not binance_secret):
        raise RuntimeError("Binance API credentials required for live trading")

    return {
        "live": live,
        "binance_api_key": binance_key,
        "binance_api_secret": binance_secret,
    }


def _bounded_rate(value: object) -> float:
    """Normalize optional rate fields into [0, 1] for MonitoringSnapshot."""

    try:
        rate = float(value)
    except (TypeError, ValueError):
        return 0.0
    if rate < 0.0:
        return 0.0
    if rate > 1.0:
        return 1.0
    return rate


def _build_monitoring_snapshot(result: dict) -> MonitoringSnapshot:
    """Convert legacy signal payload metadata into v2 monitoring snapshot."""

    risk_status_raw = result.get("risk_status")
    risk_status = risk_status_raw if isinstance(risk_status_raw, dict) else {}

    signal_type = str(result.get("signal", "")).strip().upper()
    reason = str(result.get("reason", "")).lower()

    execution_anomaly_rate = _bounded_rate(
        result.get("execution_anomaly_rate", risk_status.get("execution_anomaly_rate", 0.0))
    )
    connectivity_error_rate = _bounded_rate(
        result.get("connectivity_error_rate", risk_status.get("connectivity_error_rate", 0.0))
    )

    hard_risk_breach = bool(risk_status.get("hard_risk_breach", False))
    if risk_status.get("can_trade") is False:
        hard_risk_breach = True

    return MonitoringSnapshot(
        feature_drift_alert=bool(result.get("drift_alert", False)) or signal_type == "DRIFT_ALERT",
        confidence_collapse_alert=("confidence drift" in reason or "confidence collapse" in reason),
        execution_anomaly_rate=execution_anomaly_rate,
        connectivity_error_rate=connectivity_error_rate,
        hard_risk_breach=hard_risk_breach,
    )


def _build_signal_notifier(bot, user_id: int):
    """Create the async callback used by the engine to notify signals."""

    async def notify_signal(result):
        global V2_DEGRADED_ALERTED_USERS

        try:
            signal_type = result.get("signal", "HOLD")
            bridge = _get_v2_bridge() if _using_v2_backend() else None

            # Handle engine crash notification
            if signal_type == "ENGINE_CRASH":
                reason = result.get("reason", "Unknown error")
                msg = (
                    f"⚠️ **ENGINE CRASHED**\n\n"
                    f"Reason: {reason}\n\n"
                    f"The engine has stopped. Use /start_demo or /start_live to restart."
                )
                await bot.send_message(chat_id=user_id, text=msg)
                _persist_user_session_flags(user_id, is_active=False)

                # Clean up dead session
                source_manager = _get_signal_source_manager()
                if source_manager and source_manager.is_running(user_id):
                    try:
                        await source_manager.stop_session(user_id)
                    except Exception as source_stop_err:
                        logger.warning(
                            "Failed stopping dead signal-source session after crash for user %s: %s",
                            user_id,
                            source_stop_err,
                        )

                if bridge and bridge.is_running(user_id):
                    try:
                        await bridge.stop_session(user_id)
                    except Exception as bridge_stop_err:
                        logger.warning(
                            "Failed stopping orphaned v2 bridge session after source crash for user %s: %s",
                            user_id,
                            bridge_stop_err,
                        )
                V2_DEGRADED_ALERTED_USERS.discard(user_id)
                return

            if _using_v2_primary_backend():
                bridge_running = bool(bridge and bridge.is_running(user_id))
                if not bridge_running:
                    if user_id not in V2_DEGRADED_ALERTED_USERS:
                        await bot.send_message(
                            chat_id=user_id,
                            text=(
                                "⚠️ **Execution Session Degraded**\n\n"
                                "Signal source is running but v2 execution bridge is offline.\n"
                                "No orders will be routed until session pair is restored.\n"
                                "Run `/stop` then `/start_demo` or `/start_live`."
                            ),
                        )
                        V2_DEGRADED_ALERTED_USERS.add(user_id)
                    return

                V2_DEGRADED_ALERTED_USERS.discard(user_id)

            if bridge and bridge.is_running(user_id):
                monitoring_snapshot = result.get("v2_monitoring_snapshot")
                if not isinstance(monitoring_snapshot, MonitoringSnapshot):
                    monitoring_snapshot = _build_monitoring_snapshot(result)
                try:
                    bridge.set_monitoring_snapshot(user_id, monitoring_snapshot)
                except Exception as monitoring_err:
                    logger.warning(
                        "Failed updating monitoring snapshot for user %s: %s",
                        user_id,
                        monitoring_err,
                    )

                if signal_type == "DRIFT_ALERT":
                    evaluation = bridge.get_kill_switch_evaluation(user_id)
                    reasons = ", ".join(evaluation.reasons) if evaluation and evaluation.reasons else "feature_drift"
                    msg = (
                        "⚠️ **Model Drift Alert**\n\n"
                        f"{result.get('reason', 'Live feature/probability drift detected.')}\n\n"
                        f"Kill-switch active: `{reasons}`\n"
                        "Trading remains paused for this cycle."
                    )
                    await bot.send_message(chat_id=user_id, text=msg)
                    return

                if signal_type == "HOLD":
                    return  # Don't spam user with HOLD signals

                native_signal = result.get("v2_signal")
                native_prices = result.get("v2_prices")

                if isinstance(native_signal, StrategySignal) and isinstance(native_prices, dict):
                    shadow_signal = native_signal
                    shadow_prices = {
                        str(symbol): float(price)
                        for symbol, price in native_prices.items()
                        if float(price) > 0.0
                    }
                else:
                    mapped = convert_legacy_signal_payload(
                        result,
                        default_symbol=DEFAULT_V2_SYMBOL,
                        timeframe="1h",
                    )
                    if mapped is None:
                        shadow_signal = None
                        shadow_prices = {}
                    else:
                        shadow_signal, shadow_prices = mapped

                if shadow_signal is not None and shadow_prices:
                    try:
                        shadow_results = await bridge.route_signals(
                            user_id,
                            signals=(shadow_signal,),
                            prices=shadow_prices,
                            monitoring_snapshot=monitoring_snapshot,
                        )
                        if shadow_results:
                            logger.info(
                                "Shadow v2 routed %d order(s) for user %s from %s signal",
                                len(shadow_results),
                                user_id,
                                shadow_signal.signal,
                            )
                    except Exception as shadow_route_err:
                        logger.warning(
                            "Shadow v2 route failure for user %s: %s",
                            user_id,
                            shadow_route_err,
                        )

            if signal_type == "DRIFT_ALERT":
                msg = (
                    "⚠️ **Model Drift Alert**\n\n"
                    f"{result.get('reason', 'Live feature/probability drift detected.')}\n\n"
                    "Trading remains paused for this cycle."
                )
                await bot.send_message(chat_id=user_id, text=msg)
                return

            if signal_type == "HOLD":
                return  # Don't spam user with HOLD signals

            emoji = {"BUY": "🟢", "SELL": "🔴"}.get(signal_type, "❓")
            symbol = str(result.get("symbol") or DEFAULT_V2_SYMBOL)
            msg = (
                f"🔔 **Trading Signal**\n\n"
                f"{emoji} Action: **{signal_type}**\n"
                f"Symbol: `{symbol}`\n"
                f"Price: `{result.get('close_price', 0.0):.5f}`\n"
                f"Regime: {result.get('regime', '?')} | P(up): {result.get('probability', 0.0):.3f}\n"
                f"Reason: {result.get('reason', 'N/A')}"
            )
            if result.get("position"):
                pos = result["position"]
                msg += f"\nSize: {pos.get('lot_size', 0):.2f} lots | Risk: {pos.get('risk_fraction', 0)*100:.1f}%"
            await bot.send_message(chat_id=user_id, text=msg)
        except Exception as e:
            logger.error(f"Failed to send signal notification to user {user_id}: {e}")

    return notify_signal


def _build_execution_diagnostics_text(bridge: V2ExecutionBridge, user_id: int) -> str:
    """Format compact execution diagnostics text for Telegram stats."""

    diagnostics = bridge.get_execution_diagnostics(user_id)
    if diagnostics is None or diagnostics.total_orders <= 0:
        return ""

    lines = [
        "Execution Telemetry:",
        f"- Orders: {diagnostics.total_orders} "
        f"(accepted={diagnostics.accepted_orders}, rejected={diagnostics.rejected_orders})",
        f"- Reject rate: {diagnostics.reject_rate*100:.2f}%",
        f"- Avg adverse slippage: {diagnostics.avg_adverse_slippage_bps:.2f} bps "
        f"across {diagnostics.slippage_sample_count} fills",
    ]

    rollout_reasons = tuple(getattr(diagnostics, "rollout_gate_reasons", ()) or ())
    rollout_enabled = (
        bool(rollout_reasons)
        or bool(getattr(diagnostics, "rollback_required", False))
        or not bool(getattr(diagnostics, "live_go_no_go_passed", True))
        or int(getattr(diagnostics, "rollout_failure_streak", 0) or 0) > 0
    )
    if rollout_enabled:
        go_status = "PASS" if bool(getattr(diagnostics, "live_go_no_go_passed", True)) else "FAIL"
        rollback_status = "REQUIRED" if bool(getattr(diagnostics, "rollback_required", False)) else "clear"
        failure_streak = int(getattr(diagnostics, "rollout_failure_streak", 0) or 0)
        reasons = ", ".join(rollout_reasons) if rollout_reasons else "none"
        lines.extend(
            [
                "Rollout Gates:",
                f"- Go/No-Go: {go_status}",
                f"- Rollback: {rollback_status}",
                f"- Failure streak: {failure_streak}",
                f"- Reasons: {reasons}",
            ]
        )

    session_health = build_session_health_summary(
        user_id=user_id,
        diagnostics=diagnostics,
        kill_switch=bridge.get_kill_switch_evaluation(user_id),
    )
    if session_health:
        lines.extend(["", session_health])

    return "\n".join(lines)


def _build_kill_switch_text(bridge: V2ExecutionBridge, user_id: int) -> str:
    """Format latest kill-switch state for Telegram stats diagnostics."""

    evaluation = bridge.get_kill_switch_evaluation(user_id)
    if evaluation is None:
        return ""

    status = "PAUSED" if evaluation.pause_trading else "CLEAR"
    reasons = ", ".join(evaluation.reasons) if evaluation.reasons else "none"
    return (
        "Kill Switch:\n"
        f"- State: {status}\n"
        f"- Reasons: {reasons}"
    )


def _build_source_signal_diagnostics_text(source_manager, user_id: int) -> str:
    """Format native signal-source diagnostics when available."""

    if source_manager is None:
        return ""

    stats_getter = getattr(source_manager, "get_signal_stats", None)
    if not callable(stats_getter):
        return ""

    stats = stats_getter(user_id)
    if not isinstance(stats, dict):
        return ""

    total_signals = int(stats.get("total_signals", 0) or 0)
    buys = int(stats.get("buys", 0) or 0)
    sells = int(stats.get("sells", 0) or 0)
    holds = int(stats.get("holds", 0) or 0)
    drift_alerts = int(stats.get("drift_alerts", 0) or 0)
    symbols = int(stats.get("symbols", 0) or 0)

    lines = [
        "Signal Source:",
        f"- Signals: {total_signals} (BUY={buys}, SELL={sells}, HOLD={holds}, DRIFT_ALERT={drift_alerts})",
        f"- Symbols observed: {symbols}",
    ]

    recent_getter = getattr(source_manager, "get_recent_signals", None)
    if callable(recent_getter):
        recent = recent_getter(user_id, limit=5)
        if recent:
            lines.append("- Recent:")
            for entry in recent:
                signal_type = str(entry.get("signal", "?")).upper()
                symbol = str(entry.get("symbol", "?")).upper()
                try:
                    price = float(entry.get("close_price", 0.0) or 0.0)
                except (TypeError, ValueError):
                    price = 0.0
                try:
                    probability = float(entry.get("probability", 0.0) or 0.0)
                except (TypeError, ValueError):
                    probability = 0.0
                lines.append(
                    f"  - {symbol} {signal_type} @ {price:.2f} (P={probability:.3f})"
                )

    return "\n".join(lines)


async def _start_v2_primary_sessions(
    *,
    user_id: int,
    live: bool,
    creds: dict,
    source_manager,
    bridge: V2ExecutionBridge,
    notify_signal,
) -> bool:
    """Start v2 primary mode session pair (signal source + execution bridge) atomically."""

    global V2_DEGRADED_ALERTED_USERS

    source_was_running = source_manager.is_running(user_id)
    bridge_was_running = bridge.is_running(user_id)

    source_started = False
    bridge_started = False

    if not source_was_running:
        source_started = await source_manager.start_session(
            user_id,
            creds,
            on_signal=notify_signal,
            execute_orders=False,
        )
    if not bridge_was_running:
        bridge_started = await bridge.start_session(
            user_id,
            live=live,
            credentials=creds,
        )

    source_running = source_manager.is_running(user_id)
    bridge_running = bridge.is_running(user_id)
    if source_running and bridge_running:
        V2_DEGRADED_ALERTED_USERS.discard(user_id)
        return source_started or bridge_started

    if source_started and source_manager.is_running(user_id):
        await source_manager.stop_session(user_id)
    if bridge_started and bridge.is_running(user_id):
        await bridge.stop_session(user_id)

    raise RuntimeError(
        "Failed to start v2 primary session pair "
        f"(source_running={source_running}, exec_running={bridge_running})."
    )


async def _restore_active_sessions(application):
    """Restore active sessions from DB on bot startup."""
    bridge = _get_v2_bridge()
    source_manager = _get_signal_source_manager()

    if _using_v2_primary_backend() and (source_manager is None or bridge is None):
        logger.warning(
            "Auto-restore skipped: v2 primary requires both signal source and execution bridge."
        )
        return

    if _using_v2_backend() and source_manager is None:
        logger.warning("Auto-restore skipped: no production model found under %s", MODEL_ROOT)
        return

    if source_manager is None and bridge is None:
        logger.warning("Auto-restore skipped: no production model found under %s", MODEL_ROOT)
        return

    session = SessionLocal()
    restore_targets: list[tuple[int, bool, dict]] = []
    try:
        active_users = (
            session.query(User)
            .join(UserContext, UserContext.telegram_id == User.telegram_id)
            .filter(User.status.in_(("active", "approved")), UserContext.is_active.is_(True))
            .all()
        )

        for db_user in active_users:
            ctx = db_user.context
            if not ctx:
                continue

            live = bool(ctx.live_mode)
            try:
                creds = _build_creds_from_context(ctx, live=live)
            except Exception as cred_err:
                logger.warning(f"Skipping auto-restore for user {db_user.telegram_id}: {cred_err}")
                ctx.is_active = False
                continue

            restore_targets.append((db_user.telegram_id, live, creds))

        session.commit()
    except Exception as e:
        session.rollback()
        logger.error(f"Failed loading sessions for auto-restore: {e}", exc_info=True)
        return
    finally:
        session.close()

    if not restore_targets:
        logger.info("No persisted active sessions to restore.")
        return

    logger.info("Restoring %d persisted session(s)...", len(restore_targets))
    for user_id, live, creds in restore_targets:
        try:
            started_primary = False
            if source_manager and _using_v1_primary_backend():
                started_primary = await source_manager.start_session(
                    user_id,
                    creds,
                    on_signal=_build_signal_notifier(application.bot, user_id),
                    execute_orders=True,
                )
                if _using_shadow_backend() and bridge:
                    try:
                        shadow_started = await bridge.start_session(
                            user_id,
                            live=live,
                            credentials=creds,
                        )
                        logger.info(
                            "Shadow v2 restore for user %s: %s",
                            user_id,
                            "started" if shadow_started else "already running",
                        )
                    except Exception as shadow_err:
                        logger.warning(
                            "Shadow v2 restore failed for user %s: %s",
                            user_id,
                            shadow_err,
                        )
            elif source_manager and bridge and _using_v2_primary_backend():
                started_primary = await _start_v2_primary_sessions(
                    user_id=user_id,
                    live=live,
                    creds=creds,
                    source_manager=source_manager,
                    bridge=bridge,
                    notify_signal=_build_signal_notifier(application.bot, user_id),
                )
                if started_primary:
                    logger.info("Restored paired v2 primary sessions for user %s", user_id)
                else:
                    logger.info(
                        "Paired v2 primary sessions already running for user %s",
                        user_id,
                    )
            elif source_manager:
                started_primary = await source_manager.start_session(
                    user_id,
                    creds,
                    on_signal=_build_signal_notifier(application.bot, user_id),
                    execute_orders=True,
                )
            elif bridge:
                started_primary = await bridge.start_session(
                    user_id,
                    live=live,
                    credentials=creds,
                )

            if started_primary:
                strategy_profile, active_model_version, active_model_source = _resolve_runtime_metadata(
                    bridge=bridge
                )
                _persist_user_session_flags(
                    user_id,
                    is_active=True,
                    live_mode=live,
                    strategy_profile=strategy_profile,
                    active_model_version=active_model_version,
                    active_model_source=active_model_source,
                )
                logger.info("Restored session for user %s in %s mode.", user_id, "LIVE" if live else "DEMO")
                try:
                    await application.bot.send_message(
                        chat_id=user_id,
                        text=(
                            "♻️ **Session Restored**\n\n"
                            "System restarted, and your trading session resumed automatically."
                            + FOOTER
                        ),
                    )
                except Exception as notify_err:
                    logger.warning(f"Failed to notify restored session for user {user_id}: {notify_err}")
        except Exception as restore_err:
            logger.error(f"Failed to restore session for user {user_id}: {restore_err}")
            _persist_user_session_flags(user_id, is_active=False)
            try:
                await application.bot.send_message(
                    chat_id=user_id,
                    text=(
                        "⚠️ Could not auto-resume your trading session.\n"
                        "Please run `/start_demo` or `/start_live`."
                        + FOOTER
                    ),
                )
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    
    session = SessionLocal()
    try:
        db_user = session.query(User).filter_by(telegram_id=user.id).first()
        
        if not db_user:
            # Register new user
            new_user = User(
                telegram_id=user.id,
                username=user.username,
                role='user',
                status='pending'
            )
            context_rec = UserContext(telegram_id=user.id)
            new_user.context = context_rec
            session.add(new_user)
            session.commit()
            
            await update.message.reply_text(
                f"👋 Welcome {user.first_name}!\n\n"
                "⏳ **Account Pending**\n"
                "Your request has been sent to the administrator.\n\n"
                "👉 **Next Step:** Wait for approval notification.\n"
                "_(You will be notified here automatically)_" + FOOTER
            )
            
            # Notify Admin
            admin_id = os.getenv("ADMIN_ID")
            if admin_id:
                try:
                    await context.bot.send_message(
                        chat_id=admin_id,
                        text=f"🔔 **New User:** {user.first_name} (@{user.username}) [ID: {user.id}]\n"
                             f"👉 Run `/approve {user.id}` to grant access."
                    )
                except Exception as e:
                    logger.error(f"Failed to notify admin: {e}")
        elif db_user.status == 'pending':
            await update.message.reply_text("⏳ Request still pending. Please wait for admin approval." + FOOTER)
        elif db_user.status == 'banned':
            await update.message.reply_text("🚫 Access denied. Contact admin." + FOOTER)
        else:
            await update.message.reply_text(
                f"✅ **Welcome back, {user.first_name}!**\n\n"
                "System is ready.\n"
                "👉 `/start_demo` — paper trading (no API keys needed)\n"
                "👉 `/start_live` — real trading (requires `/setup` first)\n"
                "👉 `/status` — check engine state" + FOOTER
            )
    except Exception as e:
        logger.error(f"Start error: {e}")
        await update.message.reply_text("⚠️ System error. Try again later.")
    finally:
        session.close()


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.info(f"HELP COMMAND CALLED by {update.effective_user.id}")
    try:
        v2_enabled = _get_v2_bridge() is not None
        shadow_enabled = _using_shadow_backend()
        trading_caption = (
            "v1 execution enabled with v2 shadow portfolio diagnostics (1H anchor + 4H context)."
            if shadow_enabled
            else (
                "Multi-symbol v2 portfolio mode enabled (1H anchor + 4H context)."
                if v2_enabled
                else "System trades BTCUSDT perpetual futures at 1H timeframe."
            )
        )

        msg = (
            "COMMAND LIST\n\n"
            "Basics\n"
            "/start - Check account status\n"
            "/help - Show this menu\n\n"
            "Setup\n"
            "/setup key secret - Connect Binance (optional for paper)\n"
            "  Paper trading works without credentials!\n\n"
            "Trading\n"
            "/start_demo - Start PAPER trading (BTCUSDT 1H)\n"
            "/start_live - Start REAL trading\n"
            "/stop - Stop execution\n"
            "/reset_demo - Reset paper state\n"
            "/status - Check if running\n"
            "/stats - View live performance\n\n"
            + trading_caption + "\n"
            "Paper trading starts at $10,000 with 2% stop loss.\n"
            "Trades auto-close at 4H horizon or stop loss."
        )
        
        user_id = update.effective_user.id
        if _is_admin_user(user_id):
            msg += (
                "\n\nAdmin\n"
                "/approve <id> - Approve user\n"
                "/revoke <id> - Freeze user\n"
                "/model_active - Show runtime model routing\n"
                "/model_versions - List registered model versions\n"
                "/model_rollback [version_id] - Switch active pointer (default: previous)"
            )
        
        await update.message.reply_text(msg)
    except Exception as e:
        logger.error(f"CRITICAL HELP ERROR: {e}", exc_info=True)
        await update.message.reply_text("⚠️ Error displaying help menu.")

async def _start_engine(update: Update, context: ContextTypes.DEFAULT_TYPE, live: bool):
    bridge = _get_v2_bridge()
    source_manager = _get_signal_source_manager()

    if _using_v2_primary_backend() and (source_manager is None or bridge is None):
        await update.message.reply_text(
            "❌ v2 primary mode requires both signal source and execution bridge. Contact admin."
            + FOOTER
        )
        return

    if _using_v2_backend() and source_manager is None:
        await update.message.reply_text("❌ No production model found. Contact admin." + FOOTER)
        return

    if source_manager is None and bridge is None:
        await update.message.reply_text("❌ No production model found. Contact admin." + FOOTER)
        return
    user_id = update.effective_user.id

    session = SessionLocal()
    user = session.query(User).filter_by(telegram_id=user_id).first()

    if not user or user.status in {'pending', 'banned'}:
        await update.message.reply_text("⛔ Account not approved." + FOOTER)
        session.close()
        return

    # Build credentials dict based on mode
    try:
        if not user.context:
            user.context = UserContext(telegram_id=user_id)
            session.commit()
        ctx = user.context
        creds = _build_creds_from_context(ctx, live=live)
    except Exception as e:
        if live and "Binance API credentials required for live trading" in str(e):
            logger.warning("User %s tried /start_live without Binance API credentials", user_id)
            await update.message.reply_text(
                "❌ Binance API credentials required for live trading.\n\n"
                "Run: `/setup BINANCE_API_KEY BINANCE_API_SECRET`\n\n"
                "For paper trading without credentials, use `/start_demo`" + FOOTER
            )
            session.close()
            return

        logger.error(f"Credential loading failed for user {user_id}: {e}")
        await update.message.reply_text("❌ Failed to load credentials. Try `/setup` again." + FOOTER)
        session.close()
        return

    session.close()

    mode_str = "LIVE 🔴" if live else "DEMO 🟢"
    notify_signal = _build_signal_notifier(context.bot, user_id)

    try:
        started_primary = False
        if source_manager and _using_v1_primary_backend():
            started_primary = await source_manager.start_session(
                user_id,
                creds,
                on_signal=notify_signal,
                execute_orders=True,
            )
            if _using_shadow_backend() and bridge:
                try:
                    shadow_started = await bridge.start_session(
                        user_id,
                        live=live,
                        credentials=creds,
                    )
                    logger.info(
                        "Shadow v2 session for user %s: %s",
                        user_id,
                        "started" if shadow_started else "already running",
                    )
                except Exception as shadow_err:
                    logger.warning(
                        "Shadow v2 start failed for user %s: %s",
                        user_id,
                        shadow_err,
                    )
        elif source_manager and bridge and _using_v2_primary_backend():
            started_primary = await _start_v2_primary_sessions(
                user_id=user_id,
                live=live,
                creds=creds,
                source_manager=source_manager,
                bridge=bridge,
                notify_signal=notify_signal,
            )
        elif source_manager:
            started_primary = await source_manager.start_session(
                user_id,
                creds,
                on_signal=notify_signal,
                execute_orders=True,
            )
        elif bridge:
            started_primary = await bridge.start_session(
                user_id,
                live=live,
                credentials=creds,
            )

        if started_primary:
            # Persist requested mode and active state only after successful start.
            strategy_profile, active_model_version, active_model_source = _resolve_runtime_metadata(
                bridge=bridge
            )
            _persist_user_session_flags(
                user_id,
                live_mode=live,
                is_active=True,
                strategy_profile=strategy_profile,
                active_model_version=active_model_version,
                active_model_source=active_model_source,
            )

            extra_parts: list[str] = []
            if _using_shadow_backend():
                extra_parts.append("📡 v2 shadow diagnostics enabled.")

            if bridge:
                session_mode = bridge.get_session_mode(user_id)
                if session_mode == "paper_shadow":
                    extra_parts.append("🧪 v2 live execution disabled; running paper-shadow mode.")

            extra = "\n" + "\n".join(extra_parts) if extra_parts else ""
            await update.message.reply_text(
                f"🚀 **{mode_str} Trading STARTED**\n\n"
                "✅ Analysis running...\n"
                "👉 **Next Step:** Monitor performance.\n"
                f"Run: `/stats`{extra}" + FOOTER
            )
        else:
            strategy_profile, active_model_version, active_model_source = _resolve_runtime_metadata(
                bridge=bridge
            )
            _persist_user_session_flags(
                user_id,
                live_mode=live,
                is_active=True,
                strategy_profile=strategy_profile,
                active_model_version=active_model_version,
                active_model_source=active_model_source,
            )
            await update.message.reply_text("⚠️ Engine already running." + FOOTER)
    except Exception as e:
        _persist_user_session_flags(user_id, is_active=False)
        logger.error(f"Failed to start engine for user {user_id}: {e}")
        await update.message.reply_text(f"❌ Failed to start engine: {e}" + FOOTER)

async def start_demo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await _start_engine(update, context, live=False)

async def start_live(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await _start_engine(update, context, live=True)

async def reset_demo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id

    if _using_v2_primary_backend():
        source_manager = _get_signal_source_manager()
        bridge = _get_v2_bridge()
        if source_manager is None or bridge is None:
            await update.message.reply_text("❌ No production model found. Contact admin." + FOOTER)
            return

        source_running = source_manager.is_running(user_id)
        bridge_running = bridge.is_running(user_id)
        if not source_running and not bridge_running:
            await update.message.reply_text("⚠️ Engine not running. Start with `/start_demo` first." + FOOTER)
            return

        source_mode_getter = getattr(source_manager, "get_session_mode", None)
        source_mode = source_mode_getter(user_id) if callable(source_mode_getter) else None
        bridge_mode = bridge.get_session_mode(user_id)
        if source_mode == "live" or bridge_mode == "live":
            await update.message.reply_text("❌ Cannot reset a live account. Use `/stop` first." + FOOTER)
            return

        source_reset = False
        source_resetter = getattr(source_manager, "reset_session_state", None)
        if source_running and callable(source_resetter):
            source_reset = bool(source_resetter(user_id))

        bridge_reset = bridge.reset_session_state(user_id) if bridge_running else False
        reset_parts: list[str] = []
        if source_reset:
            reset_parts.append("signal source")
        if bridge_reset:
            reset_parts.append("execution")

        if reset_parts:
            reset_scope = " and ".join(reset_parts)
            degraded_after_reset = source_manager.is_running(user_id) != bridge.is_running(user_id)
            details = f"Reset: {reset_scope} paper state.\n"
            if degraded_after_reset:
                V2_DEGRADED_ALERTED_USERS.add(user_id)
                details += (
                    "⚠️ Session remains degraded; run `/stop` then `/start_demo` "
                    "or `/start_live` to re-sync both sides."
                )
            else:
                V2_DEGRADED_ALERTED_USERS.discard(user_id)
                details += "v2 session continues running with fresh state."
            await update.message.reply_text(
                "🔄 **Demo Reset!**\n\n"
                + details
                + FOOTER
            )
        else:
            await update.message.reply_text(
                "⚠️ Unable to reset demo state for this session." + FOOTER
            )
        return

    manager = _get_manager()
    if not manager:
        await update.message.reply_text("❌ No production model found. Contact admin." + FOOTER)
        return

    if not manager.is_running(user_id):
        await update.message.reply_text("⚠️ Engine not running. Start with `/start_demo` first." + FOOTER)
        return

    engine = manager.sessions[user_id]
    gen = engine.gen

    if gen.live:
        await update.message.reply_text("❌ Cannot reset a live account. Use `/stop` first." + FOOTER)
        return

    gen.reset_paper_balance()
    await update.message.reply_text(
        f"🔄 **Demo Reset!**\n\n"
        f"💰 Balance: ${gen.paper_balance:,.2f}\n"
        f"All signals and stats cleared.\n"
        f"Engine continues running with fresh state." + FOOTER
    )

async def stop_trading(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global V2_DEGRADED_ALERTED_USERS

    bridge = _get_v2_bridge()
    source_manager = _get_signal_source_manager()
    if source_manager is None and bridge is None:
        await update.message.reply_text("❌ No production model found. Contact admin." + FOOTER)
        return
    user_id = update.effective_user.id

    if source_manager and bridge:
        stopped_source = await source_manager.stop_session(user_id)
        stopped_exec = await bridge.stop_session(user_id)
        stopped = stopped_source or stopped_exec
    elif source_manager:
        stopped = await source_manager.stop_session(user_id)
    else:
        stopped = await bridge.stop_session(user_id)
    # Explicit /stop means user opted out of auto-resume.
    _persist_user_session_flags(user_id, is_active=False)
    V2_DEGRADED_ALERTED_USERS.discard(user_id)

    if stopped:
        await update.message.reply_text(
            "Bzzt. **Engine STOPPED** 🛑\n\n"
            "To resume:\n"
            "`/start_demo` or `/start_live`" + FOOTER
        )
    else:
        await update.message.reply_text("⚠️ Engine not running. Auto-resume disabled." + FOOTER)


async def model_active(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show active model registry pointer and runtime model binding (admin only)."""

    user_id = update.effective_user.id
    if not _is_admin_user(user_id):
        return

    active = MODEL_REGISTRY.get_active_version()
    active_pointer = MODEL_REGISTRY.get_active_pointer()
    resolved = _resolve_runtime_model_dir()
    bridge = _get_v2_bridge()
    source_label = MODEL_RESOLUTION.source
    if MODEL_RESOLUTION.active_version_id:
        source_label = f"{source_label}:{MODEL_RESOLUTION.active_version_id}"

    source_manager = _get_signal_source_manager()
    source_model = (
        str(Path(source_manager.model_dir).expanduser())
        if source_manager and hasattr(source_manager, "model_dir")
        else "(not initialized)"
    )
    source_active_sessions = source_manager.get_active_count() if source_manager else 0
    source_kind = "v2_native" if _using_v2_primary_backend() else "legacy"
    bridge_active_sessions = bridge.get_active_count() if bridge is not None else 0

    lines = [
        "🧭 **Model Routing Status**",
        "",
        f"Registry root: `{MODEL_REGISTRY_ROOT}`",
        f"Model root: `{MODEL_ROOT}`",
        f"Runtime source: `{source_label}`",
        f"Runtime model: `{resolved}`" if resolved else "Runtime model: `(unresolved)`",
    ]

    if active:
        lines.extend(
            [
                f"Active version: `{active.version_id}`",
                f"Active artifact: `{active.artifact_dir}`",
            ]
        )
        previous_active = active_pointer.previous_version_id if active_pointer else None
        lines.append(
            f"Previous active: `{previous_active}`"
            if previous_active
            else "Previous active: `(none)`"
        )
    else:
        lines.append("Active version: `(none)`")

    lines.extend(
        [
            f"Signal source kind: `{source_kind}`",
            f"Signal source model: `{source_model}`",
            f"Signal source active sessions: `{source_active_sessions}`",
            f"Bridge active sessions: `{bridge_active_sessions}`",
        ]
    )
    await update.message.reply_text("\n".join(lines) + FOOTER)


async def model_versions(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """List registered model versions with active marker (admin only)."""

    user_id = update.effective_user.id
    if not _is_admin_user(user_id):
        return

    records = MODEL_REGISTRY.list_versions()
    if not records:
        await update.message.reply_text("⚠️ No registered versions found in registry." + FOOTER)
        return

    active = MODEL_REGISTRY.get_active_version()
    active_id = active.version_id if active else ""

    lines = ["🗂️ **Registered Model Versions**", ""]
    for record in records[-12:]:
        marker = "🟢" if record.version_id == active_id else "⚪"
        lines.append(f"{marker} `{record.version_id}` -> `{record.artifact_dir}`")

    lines.append("")
    lines.append("Use: `/model_rollback <version_id>` or `/model_rollback` (previous)")
    await update.message.reply_text("\n".join(lines) + FOOTER)


async def model_rollback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Roll active model pointer back to a registered version (admin only)."""

    global MANAGER
    global V2_SIGNAL_MANAGER

    user_id = update.effective_user.id
    if not _is_admin_user(user_id):
        return

    target_record = None
    target_source = "explicit"
    if context.args:
        target_version = str(context.args[0]).strip()
        target_record = MODEL_REGISTRY.get_version(target_version)
        if target_record is None:
            await update.message.reply_text(f"❌ Unknown model version: `{target_version}`" + FOOTER)
            return
    else:
        target_source = "previous"
        target_record = MODEL_REGISTRY.get_previous_active_version()
        if target_record is None:
            await update.message.reply_text(
                "⚠️ No previous active model version available for automatic rollback." + FOOTER
            )
            return

    target_artifact = Path(target_record.artifact_dir).expanduser().resolve()
    if not target_artifact.is_dir() or not (target_artifact / "config.json").exists():
        await update.message.reply_text(
            "❌ Registered version artifact missing required `config.json`: "
            f"`{target_artifact}`"
            + FOOTER
        )
        return

    bridge = _get_v2_bridge()
    source_manager = _get_signal_source_manager()
    source_active_sessions = source_manager.get_active_count() if source_manager is not None else 0
    bridge_active_sessions = bridge.get_active_count() if bridge is not None else 0

    if source_active_sessions > 0 or bridge_active_sessions > 0:
        await update.message.reply_text(
            "⚠️ Cannot switch model while sessions are running. Stop active sessions first." + FOOTER
        )
        return

    previous = MODEL_REGISTRY.get_active_version()
    previous_id = previous.version_id if previous else "(none)"
    if previous and previous.version_id == target_record.version_id:
        await update.message.reply_text(
            f"ℹ️ Model `{target_record.version_id}` is already active." + FOOTER
        )
        return

    pre_switch_pointer = MODEL_REGISTRY.get_active_pointer()
    if target_source == "previous":
        switched = MODEL_REGISTRY.rollback_to_previous_version()
        if switched is None:
            await update.message.reply_text(
                "⚠️ No previous active model version available for automatic rollback." + FOOTER
            )
            return
        target_record = switched
    else:
        MODEL_REGISTRY.set_active_version(target_record.version_id)

    MANAGER = None
    V2_SIGNAL_MANAGER = None
    reloaded_source = _get_signal_source_manager(allow_reload_with_active_sessions=True)
    if reloaded_source is None:
        try:
            if pre_switch_pointer is not None:
                MODEL_REGISTRY.set_active_version(
                    pre_switch_pointer.version_id,
                    previous_version_id=pre_switch_pointer.previous_version_id,
                )
            elif previous is not None:
                MODEL_REGISTRY.set_active_version(previous.version_id)
            else:
                MODEL_REGISTRY.clear_active_version()
        except Exception as restore_err:
            logger.error(
                "Failed restoring previous model pointer after rollback load failure: %s",
                restore_err,
            )

        MANAGER = None
        V2_SIGNAL_MANAGER = None
        restored_source = _get_signal_source_manager(allow_reload_with_active_sessions=True)
        restored_model = (
            str(Path(restored_source.model_dir).expanduser())
            if restored_source is not None and hasattr(restored_source, "model_dir")
            else "(unresolved)"
        )
        await update.message.reply_text(
            "❌ Rollback pointer update failed runtime load; reverted to previous pointer.\n"
            f"Restored active version: `{previous_id}`\n"
            f"Runtime model: `{restored_model}`"
            + FOOTER
        )
        return

    await update.message.reply_text(
        "✅ **Model Rollback Applied**\n\n"
        f"From: `{previous_id}`\n"
        f"To: `{target_record.version_id}`\n"
        f"Runtime model: `{Path(reloaded_source.model_dir).expanduser()}`"
        + FOOTER
    )


async def revoke(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if not _is_admin_user(user_id):
        return

    try:
        target_id = int(context.args[0])
        session = SessionLocal()
        user = session.query(User).filter_by(telegram_id=target_id).first()
        if user:
            user.status = 'banned'
            if user.context:
                user.context.is_active = False
            session.commit()
            
            # Stop engine if running
            source_manager = _get_signal_source_manager()
            bridge = _get_v2_bridge()
            stopped_any = False
            if source_manager and source_manager.is_running(target_id):
                await source_manager.stop_session(target_id)
                stopped_any = True
            if bridge and bridge.is_running(target_id):
                await bridge.stop_session(target_id)
                stopped_any = True
            if stopped_any:
                await update.message.reply_text(f"🛑 active session for {target_id} stopped.")
                
            await update.message.reply_text(f"🚫 User {target_id} has been **FROZEN**." + FOOTER)

            try:
                await context.bot.send_message(target_id, "⛔ Your access has been revoked by the administrator.")
            except:
                pass 
        else:
            await update.message.reply_text("❌ User not found." + FOOTER)
        session.close()
    except (IndexError, ValueError):
        await update.message.reply_text("Usage: /revoke <user_id>" + FOOTER)

async def approve(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if not _is_admin_user(user_id):
        return

    try:
        target_id = int(context.args[0])
        session = SessionLocal()
        user = session.query(User).filter_by(telegram_id=target_id).first()
        if user:
            user.status = 'active'
            session.commit()
            await update.message.reply_text(f"✅ User {target_id} has been **APPROVED**." + FOOTER)
            try:
                approve_msg = (
                    "🎉 **Account Approved!**\n\n"
                    "You're all set for paper trading!\n"
                    "Run `/start_demo` to begin (no API keys needed).\n\n"
                    "For live trading later:\n"
                    "`/setup BINANCE_API_KEY BINANCE_API_SECRET`\n"
                    "then `/start_live`"
                )
                await context.bot.send_message(target_id, approve_msg)
            except:
                pass
        else:
            await update.message.reply_text("❌ User not found." + FOOTER)
        session.close()
    except (IndexError, ValueError):
        await update.message.reply_text("Usage: /approve <user_id>" + FOOTER)

async def setup(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    args = context.args

    # Crypto mode: optional Binance API key + secret (for future live trading)
    # Paper trading works without credentials
    if len(args) < 2:
        await update.message.reply_text(
            "**Crypto Mode Setup**\n\n"
            "For paper trading, no setup needed!\n"
            "Just run `/start_demo`\n\n"
            "For future live trading:\n"
            "`/setup BINANCE_API_KEY BINANCE_API_SECRET`\n\n"
            "**Security Note:** Your keys are encrypted." + FOOTER
        )
        return

    api_key = args[0]
    api_secret = args[1]

    session = SessionLocal()
    user = session.query(User).filter_by(telegram_id=user_id).first()
    if user:
        if not user.context:
            user.context = UserContext(telegram_id=user_id)
        user.context.binance_api_key = CRYPTO.encrypt(api_key)
        user.context.binance_api_secret = CRYPTO.encrypt(api_secret)
        session.commit()
        await update.message.reply_text(
            "✅ **Binance Credentials Saved!**\n\n"
            "Run: `/start_demo` or `/start_live`" + FOOTER
        )
    else:
        await update.message.reply_text("⛔ User not found." + FOOTER)
    session.close()

async def stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    bridge = _get_v2_bridge()
    source_manager = _get_signal_source_manager()
    manager = _get_manager() if _using_v1_primary_backend() else None
    if bridge is None and source_manager is None:
        await update.message.reply_text("❌ No production model found. Contact admin." + FOOTER)
        return
    user_id = update.effective_user.id

    if manager and manager.is_running(user_id) and _using_v1_primary_backend():
        engine = manager.sessions[user_id]
        gen = engine.gen
        mode_label = "LIVE" if gen.live else "PAPER CRYPTO"

        wr = gen.get_win_rate_stats()

        recent = gen.signal_log[-5:] if gen.signal_log else []
        if recent:
            trade_lines = []
            for s in recent:
                outcome = s.get('outcome', '')
                tag = ""
                if outcome == "win":
                    pnl_usd = s.get('pnl_usd', 0)
                    tag = f" W ${pnl_usd:+.0f}" if pnl_usd else " W"
                elif outcome == "loss":
                    pnl_usd = s.get('pnl_usd', 0)
                    exit_r = " SL" if s.get('exit_reason') == 'stop_loss' else ""
                    tag = f" L{exit_r} ${pnl_usd:+.0f}" if pnl_usd else f" L{exit_r}"
                price_fmt = f"{s['close_price']:.2f}"
                trade_lines.append(
                    f"- {s['signal']} @ {price_fmt} (P={s['probability']:.3f}){tag}"
                )
            trade_str = "\n".join(trade_lines)
        else:
            trade_str = "(No signals yet)"

        # Balance & PnL
        balance_str = f"${wr.get('paper_balance', 10000):,.2f}"
        pnl_str = wr.get('total_pnl', '$0.00')
        stop_losses = wr.get('stop_losses', 0)

        if wr["evaluated"] > 0:
            wr_section = (
                f"\n**Win Rate**\n"
                f"Win Rate: {wr['win_rate']}% ({wr['wins']}W / {wr['losses']}L)\n"
                f"PnL: {pnl_str}\n"
                f"Evaluated: {wr['evaluated']} | Pending: {wr['pending']}\n"
            )
            if stop_losses > 0:
                wr_section += f"Stop losses: {stop_losses}\n"
        else:
            pending = wr['pending']
            if pending > 0:
                wr_section = f"\n**Win Rate**\n(Evaluating {pending} signal(s)...)\n"
            else:
                wr_section = ""

        instrument = "BTCUSDT 1H"
        msg = (
            f"📊 **{mode_label} Statistics** ({instrument})\n\n"
            f"💰 **Balance:** {balance_str}\n"
            f"📈 **PnL:** {pnl_str}\n"
            f"{wr_section}\n"
            f"**Signals:** {wr['total_signals']} total "
            f"({wr['buys']} BUY / {wr['sells']} SELL / {wr['holds']} HOLD)\n\n"
            f"**Recent:**\n{trade_str}"
        )

        if _using_shadow_backend() and bridge and bridge.is_running(user_id):
            shadow_mode = bridge.get_session_mode(user_id)
            shadow_label = "SHADOW"
            if shadow_mode == "live":
                shadow_label = "SHADOW LIVE"
            elif shadow_mode == "paper_shadow":
                shadow_label = "SHADOW PAPER"

            shadow_text = bridge.build_stats_text(user_id, mode_label=shadow_label)
            if shadow_text:
                msg += "\n\n" + shadow_text

            shadow_diag = _build_execution_diagnostics_text(bridge, user_id)
            if shadow_diag:
                msg += "\n\n" + shadow_diag

            shadow_kill_switch = _build_kill_switch_text(bridge, user_id)
            if shadow_kill_switch:
                msg += "\n\n" + shadow_kill_switch

        await update.message.reply_text(msg + FOOTER)
        return

    if bridge:
        if not bridge.is_running(user_id):
            if _using_v2_primary_backend() and source_manager and source_manager.is_running(user_id):
                source_diag = _build_source_signal_diagnostics_text(source_manager, user_id)
                msg = (
                    "⚠️ **Session Degraded**\n\n"
                    "Signal source is running but execution bridge is offline.\n"
                    "No orders are currently being routed.\n"
                    "Run `/stop` then `/start_demo` or `/start_live` to re-sync."
                )
                if source_diag:
                    msg += "\n\n" + source_diag
                await update.message.reply_text(msg + FOOTER)
                return

            await update.message.reply_text("⚠️ Engine not running." + FOOTER)
            return

        session = SessionLocal()
        try:
            db_user = session.query(User).filter_by(telegram_id=user_id).first()
            live_mode = bool(db_user.context.live_mode) if db_user and db_user.context else False
        finally:
            session.close()

        mode_label = "LIVE" if live_mode else "PAPER"
        bridge_mode = bridge.get_session_mode(user_id)
        if bridge_mode == "paper_shadow":
            mode_label = "PAPER SHADOW"
        elif bridge_mode == "live":
            mode_label = "LIVE"
        stats_text = bridge.build_stats_text(user_id, mode_label=mode_label)
        if not stats_text:
            await update.message.reply_text("⚠️ Stats unavailable for this session." + FOOTER)
            return

        diag_text = _build_execution_diagnostics_text(bridge, user_id)
        if diag_text:
            stats_text += "\n\n" + diag_text

        kill_switch_text = _build_kill_switch_text(bridge, user_id)
        if kill_switch_text:
            stats_text += "\n\n" + kill_switch_text

        if _using_v2_primary_backend():
            source_diag_text = _build_source_signal_diagnostics_text(source_manager, user_id)
            if source_diag_text:
                stats_text += "\n\n" + source_diag_text

        await update.message.reply_text(stats_text + FOOTER)
        return

    await update.message.reply_text("⚠️ Engine not running." + FOOTER)

async def status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global V2_DEGRADED_ALERTED_USERS

    bridge = _get_v2_bridge()
    source_manager = _get_signal_source_manager()
    if bridge is None and source_manager is None:
        await update.message.reply_text("❌ Bot Manager not initialized." + FOOTER)
        return
        
    user_id = update.effective_user.id

    if _using_v2_primary_backend() and (source_manager is None or bridge is None):
        await update.message.reply_text(
            "⚠️ **Session Degraded**\n\n"
            "v2 primary requires both signal source and execution bridge, but one side is unavailable."
            + FOOTER
        )
        return

    if _using_v2_primary_backend() and source_manager is not None and bridge is not None:
        source_running = source_manager.is_running(user_id)
        exec_running = bridge.is_running(user_id)
        if source_running != exec_running:
            V2_DEGRADED_ALERTED_USERS.add(user_id)
            await update.message.reply_text(
                "⚠️ **Session Degraded**\n\n"
                f"Signal source running: `{source_running}`\n"
                f"Execution bridge running: `{exec_running}`\n\n"
                "Run `/stop` then `/start_demo` or `/start_live` to re-sync." + FOOTER
            )
            return
        if source_running and exec_running:
            V2_DEGRADED_ALERTED_USERS.discard(user_id)

    is_running = False
    if source_manager and source_manager.is_running(user_id):
        is_running = True
    elif bridge and bridge.is_running(user_id):
        is_running = True
    if is_running:
        await update.message.reply_text("✅ **Engine Running** 🏃\n\nMonitoring market..." + FOOTER)
    else:
        await update.message.reply_text("🛑 **Engine Stopped**" + FOOTER)


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.error("Exception while handling an update:", exc_info=context.error)

def main():
    token = os.getenv("TELEGRAM_TOKEN")
    if not token:
        print("Error: TELEGRAM_TOKEN environment variable is missing.")
        return
        
    async def post_init(app):
        await _restore_active_sessions(app)

    application = ApplicationBuilder().token(token).post_init(post_init).build()
    
    application.add_handler(CommandHandler('start', start))
    application.add_handler(CommandHandler('help', help_command))
    application.add_handler(CommandHandler('menu', help_command))
    application.add_handler(CommandHandler('commands', help_command))
    
    application.add_handler(CommandHandler('approve', approve))
    application.add_handler(CommandHandler('revoke', revoke))
    application.add_handler(CommandHandler('model_active', model_active))
    application.add_handler(CommandHandler('model_versions', model_versions))
    application.add_handler(CommandHandler('model_rollback', model_rollback))
    application.add_handler(CommandHandler('setup', setup))
    application.add_handler(CommandHandler('start_demo', start_demo))
    application.add_handler(CommandHandler('star_demo', start_demo))
    application.add_handler(CommandHandler('start_live', start_live))
    application.add_handler(CommandHandler('star_live', start_live))
    application.add_handler(CommandHandler('stop', stop_trading))
    application.add_handler(CommandHandler('reset_demo', reset_demo))
    application.add_handler(CommandHandler('status', status))
    application.add_handler(CommandHandler('stats', stats))
    
    # Debug: Log all updates
    async def debug_log(update: Update, context: ContextTypes.DEFAULT_TYPE):
        logger.info(f"DEBUG: Received update: {update}")
    
    from telegram.ext import MessageHandler, filters
    application.add_handler(MessageHandler(filters.ALL, debug_log), group=1)
    
    application.add_error_handler(error_handler)
    
    print("Bot is polling...")
    application.run_polling(drop_pending_updates=True)

if __name__ == '__main__':
    main()
