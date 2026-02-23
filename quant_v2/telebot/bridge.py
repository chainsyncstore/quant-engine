"""Bridge helpers to integrate Telegram commands with v2 execution services."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from quant_v2.contracts import PortfolioSnapshot, StrategySignal
from quant_v2.execution.adapters import ExecutionResult
from quant_v2.execution.service import ExecutionDiagnostics, ExecutionService, SessionRequest
from quant_v2.monitoring.kill_switch import KillSwitchEvaluation, MonitoringSnapshot


def format_portfolio_snapshot(snapshot: PortfolioSnapshot, *, mode_label: str) -> str:
    """Render a compact Telegram-friendly portfolio status string."""

    risk = snapshot.risk
    gross = f"{risk.gross_exposure_frac*100:.2f}%" if risk else "n/a"
    net = f"{risk.net_exposure_frac*100:.2f}%" if risk else "n/a"
    dd = f"{risk.max_drawdown_frac*100:.2f}%" if risk else "n/a"
    budget = f"{risk.risk_budget_used_frac*100:.2f}%" if risk else "n/a"

    symbol_notionals = snapshot.symbol_notional_usd or {}
    total_notional = float(sum(symbol_notionals.values()))
    cash_available = max(snapshot.equity_usd - total_notional, 0.0)
    avg_notional = (
        total_notional / snapshot.symbol_count if snapshot.symbol_count else 0.0
    )

    lines = [
        f"📊 **{mode_label} Portfolio Stats (v2)**",
        "",
        f"🕒 Timestamp: `{snapshot.timestamp}`",
        f"💰 Equity: `${snapshot.equity_usd:,.2f}`",
        f"📦 Open symbols: `{snapshot.symbol_count}`",
        f"📈 Gross Exposure: `{gross}`",
        f"↔️ Net Exposure: `{net}`",
        f"📉 Max Drawdown: `{dd}`",
        f"🛡️ Risk Budget Used: `{budget}`",
        f"💵 Total Notional: `${total_notional:,.2f}`",
        f"💼 Cash Available: `${cash_available:,.2f}`",
    ]

    if snapshot.symbol_count:
        lines.append(f"⚖️ Avg per symbol: `${avg_notional:,.2f}`")

    if symbol_notionals:
        lines.append("")
        lines.append("Per-symbol stake:")
        ordered = sorted(symbol_notionals.items(), key=lambda kv: abs(kv[1]), reverse=True)
        for symbol, notional in ordered[:10]:
            lines.append(f"- {symbol}: `${notional:,.2f}`")

    if snapshot.symbol_pnl_usd:
        top = sorted(snapshot.symbol_pnl_usd.items(), key=lambda kv: abs(kv[1]), reverse=True)[:5]
        lines.append("")
        lines.append("Top Symbol PnL:")
        for symbol, pnl in top:
            lines.append(f"- {symbol}: `${pnl:+.2f}`")

    return "\n".join(lines)


def convert_legacy_signal_payload(
    payload: dict[str, Any],
    *,
    default_symbol: str,
    timeframe: str = "1h",
) -> tuple[StrategySignal, dict[str, float]] | None:
    """Map legacy signal-generator payload into v2 signal + prices bundle."""

    signal_type = str(payload.get("signal", "HOLD")).strip().upper()
    if signal_type not in {"BUY", "SELL"}:
        return None

    try:
        close_price = float(payload.get("close_price", 0.0))
    except Exception:
        return None
    if close_price <= 0.0:
        return None

    symbol = str(payload.get("symbol") or default_symbol).strip().upper()
    if not symbol:
        return None

    raw_proba = payload.get("probability", 0.5)
    try:
        proba = float(raw_proba)
    except Exception:
        proba = 0.5
    proba = min(max(proba, 0.0), 1.0)

    confidence = proba if signal_type == "BUY" else (1.0 - proba)
    confidence = min(max(confidence, 0.0), 1.0)
    uncertainty = min(max(1.0 - confidence, 0.0), 1.0)

    try:
        horizon = int(payload.get("horizon", 4) or 4)
    except Exception:
        horizon = 4
    if horizon <= 0:
        horizon = 4

    reason = str(payload.get("reason", ""))
    signal = StrategySignal(
        symbol=symbol,
        timeframe=timeframe,
        horizon_bars=horizon,
        signal=signal_type,
        confidence=confidence,
        uncertainty=uncertainty,
        reason=reason,
    )
    return signal, {symbol: close_price}


class V2ExecutionBridge:
    """Thin bridge used by Telegram handlers to control a v2 execution service."""

    def __init__(
        self,
        service: ExecutionService,
        *,
        default_strategy_profile: str = "core_v2",
        default_universe: tuple[str, ...] = (),
    ) -> None:
        self.service = service
        self.default_strategy_profile = default_strategy_profile
        self.default_universe = default_universe

    async def start_session(
        self,
        user_id: int,
        *,
        live: bool,
        strategy_profile: str | None = None,
        universe: tuple[str, ...] | None = None,
        credentials: dict[str, str] | None = None,
    ) -> bool:
        """Start execution session for a Telegram user."""

        request = SessionRequest(
            user_id=user_id,
            live=live,
            strategy_profile=strategy_profile or self.default_strategy_profile,
            universe=universe if universe is not None else self.default_universe,
            credentials=credentials or {},
        )
        return await self.service.start_session(request)

    async def stop_session(self, user_id: int) -> bool:
        """Stop execution session for a Telegram user."""

        return await self.service.stop_session(user_id)

    def reset_session_state(self, user_id: int) -> bool:
        """Reset in-session paper state for a Telegram user when supported."""

        resetter = getattr(self.service, "reset_session_state", None)
        if not callable(resetter):
            return False
        return bool(resetter(user_id))

    def is_running(self, user_id: int) -> bool:
        """Check whether a Telegram user has an active session."""

        return self.service.is_running(user_id)

    def get_active_count(self) -> int:
        """Return total active session count for diagnostics."""

        return self.service.get_active_count()

    def get_session_mode(self, user_id: int) -> str | None:
        """Return backend session mode label when the service exposes it."""

        return self.service.get_session_mode(user_id)

    async def route_signals(
        self,
        user_id: int,
        *,
        signals: Iterable[StrategySignal],
        prices: dict[str, float],
        monitoring_snapshot: MonitoringSnapshot | None = None,
    ) -> tuple[ExecutionResult, ...]:
        """Route strategy signals through the bound execution service."""

        return await self.service.route_signals(
            user_id,
            signals=signals,
            prices=prices,
            monitoring_snapshot=monitoring_snapshot,
        )

    def set_monitoring_snapshot(
        self,
        user_id: int,
        snapshot: MonitoringSnapshot,
    ) -> KillSwitchEvaluation | None:
        """Update runtime monitoring snapshot when supported by the bound service."""

        return self.service.set_monitoring_snapshot(user_id, snapshot)

    def get_kill_switch_evaluation(self, user_id: int) -> KillSwitchEvaluation | None:
        """Return latest kill-switch state when supported by the bound service."""

        return self.service.get_kill_switch_evaluation(user_id)

    def get_execution_diagnostics(self, user_id: int) -> ExecutionDiagnostics | None:
        """Return execution diagnostics when supported by the bound service."""

        return self.service.get_execution_diagnostics(user_id)

    def build_stats_text(self, user_id: int, *, mode_label: str) -> str | None:
        """Build formatted stats string for the user, if snapshot exists."""

        snapshot = self.service.get_portfolio_snapshot(user_id)
        if snapshot is None:
            return None
        return format_portfolio_snapshot(snapshot, mode_label=mode_label)
