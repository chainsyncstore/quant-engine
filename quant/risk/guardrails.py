"""
Risk guardrails for live trading.

Implements circuit breakers to prevent catastrophic losses:
- Max daily loss limit
- Consecutive loss circuit breaker
- Max daily trade count
- Session-level kill switch
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List

logger = logging.getLogger(__name__)


@dataclass
class TradeRecord:
    """Record of a single trade for risk tracking."""

    timestamp: datetime
    pnl: float
    signal: str  # BUY or SELL
    regime: int
    probability: float


@dataclass
class RiskGuardrails:
    """
    Live trading risk guardrails with circuit breakers.

    Tracks trades within the current session and enforces limits.
    """

    max_daily_loss: float = 0.02        # Max daily loss as fraction of capital (2%)
    max_consecutive_losses: int = 3     # Circuit breaker after N consecutive losses
    max_daily_trades: int = 10          # Max trades per day
    cooldown_minutes: int = 30          # Cooldown after circuit breaker (minutes)

    # Internal state
    _trades_today: List[TradeRecord] = field(default_factory=list)
    _consecutive_losses: int = 0
    _circuit_breaker_active: bool = False
    _circuit_breaker_until: datetime | None = None
    _session_date: str = ""
    _initial_capital: float = 0.0

    def initialize(self, capital: float) -> None:
        """Set initial capital for the session."""
        self._initial_capital = capital
        self._reset_session()

    def _reset_session(self) -> None:
        """Reset daily counters."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if self._session_date != today:
            self._trades_today = []
            self._consecutive_losses = 0
            self._circuit_breaker_active = False
            self._circuit_breaker_until = None
            self._session_date = today
            logger.info("Risk guardrails: new session %s", today)

    def can_trade(self) -> tuple[bool, str]:
        """
        Check if trading is allowed under current risk constraints.

        Returns:
            (allowed, reason) tuple.
        """
        self._reset_session()
        now = datetime.now(timezone.utc)

        # Check circuit breaker cooldown
        if self._circuit_breaker_active:
            if self._circuit_breaker_until and now < self._circuit_breaker_until:
                remaining = (self._circuit_breaker_until - now).seconds // 60
                return False, f"Circuit breaker active ({remaining}min remaining)"
            else:
                self._circuit_breaker_active = False
                self._consecutive_losses = 0
                logger.info("Circuit breaker cooldown expired — trading resumed")

        # Check daily trade limit
        if len(self._trades_today) >= self.max_daily_trades:
            return False, f"Daily trade limit reached ({self.max_daily_trades})"

        # Check daily loss limit
        daily_pnl = sum(t.pnl for t in self._trades_today)
        if self._initial_capital > 0:
            daily_loss_pct = daily_pnl / self._initial_capital
            if daily_loss_pct < -self.max_daily_loss:
                return False, (
                    f"Daily loss limit hit: {daily_loss_pct:.1%} "
                    f"(limit: {-self.max_daily_loss:.1%})"
                )

        # Check consecutive losses
        if self._consecutive_losses >= self.max_consecutive_losses:
            return False, (
                f"Consecutive loss limit: {self._consecutive_losses} losses in a row "
                f"(limit: {self.max_consecutive_losses})"
            )

        return True, "OK"

    def record_trade(self, trade: TradeRecord) -> None:
        """Record a completed trade and update risk counters."""
        self._reset_session()
        self._trades_today.append(trade)

        if trade.pnl < 0:
            self._consecutive_losses += 1
            logger.warning(
                "Loss #%d (PnL=%.6f). %d/%d consecutive losses.",
                len(self._trades_today),
                trade.pnl,
                self._consecutive_losses,
                self.max_consecutive_losses,
            )

            # Trigger circuit breaker
            if self._consecutive_losses >= self.max_consecutive_losses:
                self._circuit_breaker_active = True
                from datetime import timedelta
                self._circuit_breaker_until = (
                    datetime.now(timezone.utc) + timedelta(minutes=self.cooldown_minutes)
                )
                logger.warning(
                    "🚨 CIRCUIT BREAKER: %d consecutive losses. "
                    "Trading paused for %d minutes.",
                    self._consecutive_losses,
                    self.cooldown_minutes,
                )
        else:
            self._consecutive_losses = 0

        # Log daily summary
        daily_pnl = sum(t.pnl for t in self._trades_today)
        daily_trades = len(self._trades_today)
        daily_wins = sum(1 for t in self._trades_today if t.pnl > 0)

        logger.info(
            "Risk status: %d trades today, PnL=%.6f, WR=%.0f%%, consec_losses=%d",
            daily_trades,
            daily_pnl,
            (daily_wins / daily_trades * 100) if daily_trades > 0 else 0,
            self._consecutive_losses,
        )

    def get_status(self) -> dict:
        """Get current risk status summary."""
        self._reset_session()
        daily_pnl = sum(t.pnl for t in self._trades_today)
        return {
            "trades_today": len(self._trades_today),
            "daily_pnl": daily_pnl,
            "consecutive_losses": self._consecutive_losses,
            "circuit_breaker_active": self._circuit_breaker_active,
            "can_trade": self.can_trade()[0],
            "can_trade_reason": self.can_trade()[1],
        }
