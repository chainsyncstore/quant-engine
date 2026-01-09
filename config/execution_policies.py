"""
Execution policy registry and definitions.

Execution policies capture capital-protection guardrails that must be enforced
consistently across paper trading, research simulations, and live adapters.
"""

from __future__ import annotations

from datetime import time
from typing import Dict, List, Optional, Sequence, Tuple

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ExecutionPolicy(BaseModel):
    """
    Immutable execution guardrail configuration.

    Fields are expressed in native units to keep serialization human-friendly.
    """

    model_config = ConfigDict(frozen=True)

    policy_id: str
    description: str
    version: str = Field(default="1.0")

    max_daily_drawdown_pct: float = Field(
        default=20.0, description="Maximum allowable peak-to-trough drawdown per UTC day."
    )
    max_position_notional: float = Field(
        default=0.0,
        description="Absolute notional cap per entry (0 disables the check).",
    )
    max_trades_per_day: int = Field(
        default=0, description="Maximum number of opening trades per UTC day (0 disables the check)."
    )
    forced_flat_window_utc: Optional[Tuple[time, time]] = Field(
        default=None,
        description="Optional UTC window (start, end) where no new risk may be taken.",
    )
    allowed_instruments: Sequence[str] = Field(
        default_factory=tuple,
        description="Whitelist of tradable symbols. Empty => allow all.",
    )

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
        return values

    def serialize(self) -> Dict[str, object]:
        """JSON-serializable representation for audit logs."""
        data = self.model_dump()
        if self.forced_flat_window_utc:
            start, end = self.forced_flat_window_utc
            data["forced_flat_window_utc"] = (start.isoformat(), end.isoformat())
        data["allowed_instruments"] = list(self.allowed_instruments)
        return data

    @property
    def label(self) -> str:
        """Human-readable identifier."""
        return f"{self.policy_id} v{self.version}"


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


# --------------------------------------------------------------------------- #
# Standard Policies
# --------------------------------------------------------------------------- #

register_execution_policy(
    ExecutionPolicy(
        policy_id="RESEARCH",
        description="Lenient defaults appropriate for offline research iterations.",
        version="1.0",
        max_daily_drawdown_pct=40.0,
        max_position_notional=1_000_000.0,
        max_trades_per_day=1000,
        allowed_instruments=("SYNTHETIC",),
    )
)

register_execution_policy(
    ExecutionPolicy(
        policy_id="PAPER_SAFE",
        description="Paper-trading default with moderate risk caps.",
        version="1.1",
        max_daily_drawdown_pct=15.0,
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
        max_daily_drawdown_pct=5.0,
        max_position_notional=100_000.0,
        max_trades_per_day=10,
        forced_flat_window_utc=(time(20, 45), time(21, 15)),
        allowed_instruments=("ES", "NQ", "CL", "GC", "SYNTHETIC"),
    )
)

register_execution_policy(
    ExecutionPolicy(
        policy_id="LIVE_CONSERVATIVE",
        description="Conservative live-trading posture with lower throughput.",
        version="1.0",
        max_daily_drawdown_pct=8.0,
        max_position_notional=150_000.0,
        max_trades_per_day=25,
        forced_flat_window_utc=(time(21, 0), time(21, 30)),
    )
)
