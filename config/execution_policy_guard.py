"""
Shared enforcement utilities for execution policies.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, time
from typing import Dict, Optional, Tuple

from config.execution_policies import ExecutionPolicy


@dataclass(frozen=True)
class PolicyOrderContext:
    """
    Minimal, transport-agnostic order representation for policy checks.

    Attributes:
        timestamp: Event timestamp (timezone-aware preferred).
        symbol: Optional instrument identifier.
        notional: Absolute notional amount for the order.
        is_entry: True if the order increases risk, False if it reduces risk.
    """

    timestamp: datetime
    symbol: Optional[str]
    notional: float
    is_entry: bool


class ExecutionPolicyGuard:
    """
    Stateful evaluator that tracks daily metrics and enforces ExecutionPolicy limits.
    """

    def __init__(self, policy: ExecutionPolicy):
        self.policy = policy
        self._daily_trade_counts: Dict[date, int] = defaultdict(int)
        self._daily_peak_equity: Dict[date, float] = {}
        self._daily_trough_equity: Dict[date, float] = {}

    def observe_equity(self, timestamp: datetime, equity: float) -> None:
        """Track equity to compute daily drawdown."""
        day = timestamp.date()
        peak = self._daily_peak_equity.get(day)
        trough = self._daily_trough_equity.get(day)

        if peak is None or equity > peak:
            self._daily_peak_equity[day] = equity
            peak = equity

        if trough is None or equity < trough:
            self._daily_trough_equity[day] = equity

        # Ensure trough is initialized even if equity never drops below peak
        if trough is None:
            self._daily_trough_equity[day] = equity

    def current_drawdown_pct(self, timestamp: datetime) -> float:
        """Return current intra-day drawdown percentage."""
        day = timestamp.date()
        peak = self._daily_peak_equity.get(day)
        trough = self._daily_trough_equity.get(day)

        if not peak or peak <= 0:
            return 0.0

        current = trough if trough is not None else peak
        return max(0.0, (peak - current) / peak * 100.0)

    def evaluate_order(self, ctx: PolicyOrderContext) -> Tuple[bool, str]:
        """
        Evaluate an order against policy constraints.

        The guard assumes `observe_equity` has been called with up-to-date equity
        prior to evaluation so drawdown metrics are accurate.
        """
        if not ctx.is_entry:
            return True, "Risk reduction allowed"

        drawdown = self.current_drawdown_pct(ctx.timestamp)
        if self.policy.max_daily_drawdown_pct > 0 and drawdown > self.policy.max_daily_drawdown_pct:
            return (
                False,
                f"Daily drawdown {drawdown:.2f}% exceeds limit {self.policy.max_daily_drawdown_pct:.2f}%",
            )

        if self._is_forced_flat(ctx.timestamp.time()):
            return False, "Forced-flat window active"

        if self.policy.allowed_instruments:
            if ctx.symbol is None or ctx.symbol not in self.policy.allowed_instruments:
                return False, f"Instrument {ctx.symbol or 'UNKNOWN'} not permitted by policy"

        if self.policy.max_position_notional > 0 and ctx.notional > self.policy.max_position_notional:
            return (
                False,
                f"Order notional {ctx.notional:,.2f} exceeds limit {self.policy.max_position_notional:,.2f}",
            )

        if self.policy.max_trades_per_day > 0:
            day = ctx.timestamp.date()
            if self._daily_trade_counts[day] >= self.policy.max_trades_per_day:
                return False, f"Max trades per day ({self.policy.max_trades_per_day}) reached"

        return True, "Within execution policy"

    def record_entry(self, ctx: PolicyOrderContext) -> None:
        """Record a filled entry order for trade-count tracking."""
        if not ctx.is_entry:
            return
        day = ctx.timestamp.date()
        self._daily_trade_counts[day] += 1

    def serialize_policy(self) -> Dict[str, object]:
        """Helper for logging."""
        return self.policy.serialize()

    def label(self) -> str:
        return self.policy.label or self.policy.policy_id

    def _is_forced_flat(self, current_time: time) -> bool:
        window = self.policy.forced_flat_window_utc
        if not window:
            return False
        start, end = window
        if start <= end:
            return start <= current_time <= end
        # Window wraps midnight
        return current_time >= start or current_time <= end


__all__ = ["ExecutionPolicyGuard", "PolicyOrderContext"]
