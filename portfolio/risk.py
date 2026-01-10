from datetime import date, datetime, timedelta
from typing import Callable, Dict, Optional, Protocol, Tuple

from config.execution_policies import ExecutionPolicy
from config.execution_policy_guard import ExecutionPolicyGuard, PolicyOrderContext
from hypotheses.base import IntentType, TradeIntent
from portfolio.models import PortfolioAllocation, PortfolioState
from execution_live.events import (
    COMPETITION_DAILY_HALT,
    COMPETITION_TRADE_BLOCKED,
)


class RiskRule(Protocol):
    """
    Protocol for risk management rules.
    """

    def can_execute(
        self,
        intent: TradeIntent,
        allocation: PortfolioAllocation,
        portfolio_state: PortfolioState,
    ) -> Tuple[bool, str]:
        """
        Check if a trade intent can be executed given the current state.

        Returns:
            Tuple of (allowed, reason)
        """

    def on_trade_allowed(
        self,
        intent: TradeIntent,
        allocation: PortfolioAllocation,
        portfolio_state: PortfolioState,
    ) -> None:
        """
        Optional hook invoked after a trade intent passes all risk checks.
        """


class MaxDrawdownRule:
    """
    Rejects NEW risk (Entries) if portfolio drawdown exceeds limit.
    """

    def __init__(self, max_drawdown_pct: float):
        self.max_drawdown_pct = max_drawdown_pct

    def can_execute(
        self,
        intent: TradeIntent,
        allocation: PortfolioAllocation,
        portfolio_state: PortfolioState,
    ) -> Tuple[bool, str]:
        # Always allow exits (risk reduction)
        if intent.type in [IntentType.CLOSE, IntentType.SELL]:  # Assuming Sell is exit or short
            # Note: Shorting adds risk, but CLOSE reduces it.
            # If intention is purely CLOSE existing, allow.
            # If intention is SELL (Short entry), check risk.
            if intent.type == IntentType.CLOSE:
                return True, "Risk reduction allowed"

            # If SELL means entering SHORT, we check drawdown.

        if portfolio_state.drawdown_pct > self.max_drawdown_pct:
            return False, f"Portfolio Drawdown {portfolio_state.drawdown_pct:.2f}% > Limit {self.max_drawdown_pct:.2f}%"

        return True, "Drawdown within limits"


class ExecutionPolicyRule:
    """
    Composite risk rule enforcing ExecutionPolicy constraints at the portfolio/risk layer.
    """

    def __init__(self, policy: ExecutionPolicy):
        self.policy = policy
        self._guard = ExecutionPolicyGuard(policy)

    def can_execute(
        self,
        intent: TradeIntent,
        allocation: PortfolioAllocation,
        portfolio_state: PortfolioState,
    ) -> Tuple[bool, str]:
        if intent.type == IntentType.CLOSE:
            return True, "Risk reduction allowed"

        ctx = self._build_context(intent, allocation, portfolio_state)
        self._guard.observe_equity(portfolio_state.timestamp, portfolio_state.total_capital)
        return self._guard.evaluate_order(ctx)

    def on_trade_allowed(
        self,
        intent: TradeIntent,
        allocation: PortfolioAllocation,
        portfolio_state: PortfolioState,
    ) -> None:
        if intent.type == IntentType.CLOSE:
            return
        ctx = self._build_context(intent, allocation, portfolio_state)
        self._guard.record_entry(ctx)

    def _build_context(
        self, intent: TradeIntent, allocation: PortfolioAllocation, portfolio_state: PortfolioState
    ) -> PolicyOrderContext:
        reference_price = allocation.reference_price
        if reference_price is None:
            raise ValueError("PortfolioAllocation.reference_price is required for execution policy evaluation.")
        notional = intent.size * reference_price
        return PolicyOrderContext(
            timestamp=portfolio_state.timestamp,
            symbol=allocation.symbol,
            notional=abs(notional),
            is_entry=intent.type in {IntentType.BUY, IntentType.SELL},
        )


TelemetryEmitter = Callable[[str, dict], None]


class TradeThrottle:
    """
    Enforces a per-symbol cooldown between entry attempts.
    """

    def __init__(
        self,
        cooldown: timedelta = timedelta(minutes=30),
        telemetry_hook: Optional[TelemetryEmitter] = None,
    ):
        self.cooldown = cooldown
        self._last_entry_time_by_symbol: Dict[str, Optional[datetime]] = {}
        self._telemetry_hook = telemetry_hook

    def can_execute(
        self,
        intent: TradeIntent,
        allocation: PortfolioAllocation,
        portfolio_state: PortfolioState,
    ) -> Tuple[bool, str]:
        if intent.type == IntentType.CLOSE:
            return True, "Risk reduction allowed"

        symbol = allocation.symbol
        if not symbol:
            return False, "Symbol information required for throttle evaluation"

        last_entry_time = self._last_entry_time_by_symbol.get(symbol)
        if last_entry_time:
            elapsed = portfolio_state.timestamp - last_entry_time
            if elapsed < self.cooldown:
                remaining = self.cooldown - elapsed
                remaining_minutes = max(0, int(remaining.total_seconds() // 60))
                if self._telemetry_hook:
                    self._telemetry_hook(
                        COMPETITION_TRADE_BLOCKED,
                        {
                            "symbol": symbol,
                            "cooldown_minutes": int(self.cooldown.total_seconds() // 60),
                            "remaining_minutes": remaining_minutes,
                            "timestamp": portfolio_state.timestamp.isoformat(),
                            "intent_type": intent.type.value,
                        },
                    )
                return False, f"Trade throttled for {symbol}: cooldown active ({remaining_minutes}m remaining)"

        return True, "Throttle clear"

    def on_trade_allowed(
        self,
        intent: TradeIntent,
        allocation: PortfolioAllocation,
        portfolio_state: PortfolioState,
    ) -> None:
        if intent.type == IntentType.CLOSE:
            return
        symbol = allocation.symbol
        if not symbol:
            return
        self._last_entry_time_by_symbol[symbol] = portfolio_state.timestamp


class LossStreakGuard:
    """
    Blocks new entries after a configurable number of realized losses per UTC day.
    """

    def __init__(
        self,
        max_losses: int = 2,
        telemetry_hook: Optional[TelemetryEmitter] = None,
    ):
        self.max_losses = max_losses
        self._current_day: Optional[date] = None
        self._losses_today = 0
        self._last_realized_pnl: Optional[float] = None
        self._telemetry_hook = telemetry_hook

    def can_execute(
        self,
        intent: TradeIntent,
        allocation: PortfolioAllocation,
        portfolio_state: PortfolioState,
    ) -> Tuple[bool, str]:
        self._refresh_state(portfolio_state)

        if intent.type == IntentType.CLOSE:
            return True, "Risk reduction allowed"

        if self._losses_today >= self.max_losses:
            if self._telemetry_hook:
                self._telemetry_hook(
                    COMPETITION_DAILY_HALT,
                    {
                        "losses_today": self._losses_today,
                        "max_losses": self.max_losses,
                        "timestamp": portfolio_state.timestamp.isoformat(),
                        "total_realized_pnl": portfolio_state.total_realized_pnl,
                    },
                )
            return False, f"Loss streak guard active ({self._losses_today} losses recorded today)"

        return True, "Loss streak guard clear"

    def _refresh_state(self, portfolio_state: PortfolioState) -> None:
        timestamp_day = portfolio_state.timestamp.date()
        if self._current_day != timestamp_day:
            self._current_day = timestamp_day
            self._losses_today = 0
            self._last_realized_pnl = portfolio_state.total_realized_pnl
            return

        current_realized = portfolio_state.total_realized_pnl
        if self._last_realized_pnl is not None:
            delta = current_realized - self._last_realized_pnl
            if delta < 0:
                self._losses_today += 1
        self._last_realized_pnl = current_realized
