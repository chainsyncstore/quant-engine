"""
Execution policy registry and definitions.

Execution policies capture capital-protection guardrails that must be enforced
consistently across paper trading, research simulations, and live adapters.
"""

from __future__ import annotations

from datetime import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ExecutionPolicy(BaseModel):
    """
    Immutable execution guardrail configuration.

    Fields are expressed in native units to keep serialization human-friendly.
    """

    model_config = ConfigDict(frozen=True, populate_by_name=True)

    policy_id: str
    description: str
    version: str = Field(default="1.0")
    label: Optional[str] = Field(
        default=None,
        description="Optional human-readable label override; defaults to policy_id.",
    )
    max_total_drawdown_pct: float = Field(
        default=0.0,
        description="Maximum allowable peak-to-trough drawdown over the lifespan of the policy.",
        alias="max_drawdown",
    )
    max_daily_drawdown_pct: float = Field(
        default=20.0,
        description="Maximum allowable peak-to-trough drawdown per UTC day.",
        alias="daily_drawdown",
    )
    max_position_notional: float = Field(
        default=0.0,
        description="Absolute notional cap per entry (0 disables the check).",
    )
    max_order_notional_pct_of_equity: float = Field(
        default=0.0,
        alias="max_order_notional",
        description="Relative notional cap per entry expressed as a fraction of observed equity (0 disables the check).",
    )
    min_order_notional_pct_of_equity: float = Field(
        default=0.0,
        alias="min_order_notional",
        description="Minimum relative notional per entry expressed as a fraction of observed equity (0 disables the check).",
    )
    max_trades_per_day: int = Field(
        default=0, description="Maximum number of opening trades per UTC day (0 disables the check)."
    )
    max_concurrent_positions: int = Field(
        default=0,
        description="Maximum number of simultaneously open positions (0 disables the check).",
    )
    trading_window_start_utc: Optional[time] = Field(
        default=None,
        alias="trading_window_start",
        description="UTC time when entries are first permitted.",
    )
    trading_window_end_utc: Optional[time] = Field(
        default=None,
        alias="trading_window_end",
        description="UTC time after which entries are disallowed.",
    )
    forced_flat_window_utc: Optional[Tuple[time, time]] = Field(
        default=None,
        description="Optional UTC window (start, end) where no new risk may be taken.",
    )
    forced_flat_time_utc: Optional[time] = Field(
        default=None,
        alias="forced_flat_time",
        description="Optional UTC time by which all positions must be flat (converted into a forced flat window).",
    )
    allowed_instruments: Sequence[str] = Field(
        default_factory=tuple,
        description="Whitelist of tradable symbols. Empty => allow all.",
        alias="instrument_whitelist",
    )
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Arbitrary annotations.")

    @model_validator(mode="before")
    @classmethod
    def _normalize_flat_window(cls, values: Dict[str, object]) -> Dict[str, object]:
        raw_window = values.get("forced_flat_window_utc")
        if raw_window and isinstance(raw_window, (tuple, list)):
            start, end = raw_window
            if isinstance(start, str):
                start = time.fromisoformat(start)
            if isinstance(end, str):
                end = time.fromisoformat(end)
            values["forced_flat_window_utc"] = (start, end)

        flat_time = values.get("forced_flat_time_utc")
        if isinstance(flat_time, str):
            flat_time = time.fromisoformat(flat_time)
            values["forced_flat_time_utc"] = flat_time
        if flat_time and not values.get("forced_flat_window_utc"):
            values["forced_flat_window_utc"] = (flat_time, flat_time)

        for key in ("trading_window_start_utc", "trading_window_end_utc"):
            raw_val = values.get(key)
            if isinstance(raw_val, str):
                values[key] = time.fromisoformat(raw_val)

        return values

    def serialize(self) -> Dict[str, object]:
        """JSON-serializable representation for audit logs."""
        data = self.model_dump()
        if self.forced_flat_window_utc:
            start, end = self.forced_flat_window_utc
            data["forced_flat_window_utc"] = (start.isoformat(), end.isoformat())
        if self.forced_flat_time_utc:
            data["forced_flat_time_utc"] = self.forced_flat_time_utc.isoformat()
        if self.trading_window_start_utc:
            data["trading_window_start_utc"] = self.trading_window_start_utc.isoformat()
        if self.trading_window_end_utc:
            data["trading_window_end_utc"] = self.trading_window_end_utc.isoformat()
        data["allowed_instruments"] = list(self.allowed_instruments)
        return data

    @property
    def label_display(self) -> str:
        """Human-readable identifier."""
        base = self.label or self.policy_id
        return f"{base} v{self.version}"


_EXECUTION_POLICIES: Dict[str, ExecutionPolicy] = {}


def register_execution_policy(policy: ExecutionPolicy) -> None:
    """Register an execution policy."""
    _EXECUTION_POLICIES[policy.policy_id] = policy


def get_execution_policy(policy_id: str) -> ExecutionPolicy:
    """Fetch a registered execution policy."""
    if policy_id not in _EXECUTION_POLICIES:
        raise ValueError(
            f"Unknown execution policy '{policy_id}'. Available: {list(_EXECUTION_POLICIES.keys())}"
        )
    return _EXECUTION_POLICIES[policy_id]


def list_execution_policies() -> List[ExecutionPolicy]:
    """Return all registered execution policies."""
    return list(_EXECUTION_POLICIES.values())


from config.execution_profiles.competition_5percenters import (  # noqa: E402
    COMPETITION_5PERCENTERS,
)


# --------------------------------------------------------------------------- #
# Standard Policies
# --------------------------------------------------------------------------- #

register_execution_policy(
    ExecutionPolicy(
        policy_id="RESEARCH",
        description="Lenient defaults appropriate for offline research iterations.",
        version="1.0",
        daily_drawdown=40.0,
        max_position_notional=1_000_000.0,
        max_trades_per_day=1000,
        instrument_whitelist=("SYNTHETIC",),
    )
)

register_execution_policy(
    ExecutionPolicy(
        policy_id="PAPER_SAFE",
        description="Paper-trading default with moderate risk caps.",
        version="1.1",
        daily_drawdown=15.0,
        max_position_notional=250_000.0,
        max_trades_per_day=50,
        forced_flat_window_utc=(time(20, 45), time(21, 15)),
    )
)

register_execution_policy(
    ExecutionPolicy(
        policy_id="PROP_FIRM",
        description="Prop-firm style constraints emulating evaluation combines.",
        version="1.2",
        daily_drawdown=5.0,
        max_position_notional=100_000.0,
        max_trades_per_day=10,
        forced_flat_window_utc=(time(20, 45), time(21, 15)),
        instrument_whitelist=("ES", "NQ", "CL", "GC", "SYNTHETIC"),
    )
)

register_execution_policy(
    ExecutionPolicy(
        policy_id="LIVE_CONSERVATIVE",
        description="Conservative live-trading posture with lower throughput.",
        version="1.0",
        daily_drawdown=8.0,
        max_position_notional=150_000.0,
        max_trades_per_day=25,
        forced_flat_window_utc=(time(21, 0), time(21, 30)),
    )
)


# --------------------------------------------------------------------------- #
# Competition / Temporary Policies
# --------------------------------------------------------------------------- #

register_execution_policy(COMPETITION_5PERCENTERS)
