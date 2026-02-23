"""Idempotency helpers for deterministic execution routing."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from quant_v2.contracts import OrderPlan


def build_idempotency_key(
    *,
    user_id: int,
    plan: OrderPlan,
    epoch_minute: int | None = None,
) -> str:
    """Build a stable idempotency key for order routing."""

    if user_id <= 0:
        raise ValueError("user_id must be positive")

    minute = epoch_minute
    if minute is None:
        minute = int(datetime.now(timezone.utc).timestamp() // 60)

    payload = {
        "u": user_id,
        "m": minute,
        "s": plan.symbol,
        "side": plan.side,
        "qty": round(plan.quantity, 8),
        "reduce_only": bool(plan.reduce_only),
        "client_order_id": plan.client_order_id or "",
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:24]


@dataclass
class InMemoryIdempotencyJournal:
    """Simple in-memory journal for idempotent order outcomes."""

    _items: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self._items is None:
            self._items = {}

    def seen(self, key: str) -> bool:
        return key in self._items

    def record(self, key: str, value: Any) -> None:
        self._items[key] = value

    def get(self, key: str) -> Any:
        return self._items.get(key)

    def size(self) -> int:
        return len(self._items)
