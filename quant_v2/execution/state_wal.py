"""Write-Ahead Log (WAL) for crash-resilient execution state persistence."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any

import re

logger = logging.getLogger(__name__)

_SENSITIVE_KEYS = re.compile(
    r"(api_key|api_secret|secret|password|token|credentials)",
    re.IGNORECASE,
)


def _coerce_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _coerce_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _stream_id_age_seconds(stream_id: str, *, now_ms: int | None = None) -> float | None:
    raw = str(stream_id or "").strip()
    if not raw or "-" not in raw:
        return None

    head = raw.split("-", 1)[0]
    try:
        entry_ms = int(head)
    except ValueError:
        return None

    current_ms = now_ms if now_ms is not None else int(datetime.now(timezone.utc).timestamp() * 1000.0)
    return max((current_ms - entry_ms) / 1000.0, 0.0)


def _scrub_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Recursively strip sensitive keys from a payload dict.

    Defense-in-depth: ensures no API keys, secrets, or credentials
    ever reach the Redis WAL stream, even if upstream forgets to scrub.
    """
    scrubbed: dict[str, Any] = {}
    for key, value in payload.items():
        if _SENSITIVE_KEYS.search(key):
            scrubbed[key] = "***REDACTED***"
        elif isinstance(value, dict):
            scrubbed[key] = _scrub_payload(value)
        else:
            scrubbed[key] = value
    return scrubbed


LIFECYCLE_STATE_ORDER: dict[str, int] = {
    "ACTIVE": 0,
    "SOFT_BREACH": 1,
    "REDUCE_ONLY": 2,
    "INCIDENT": 3,
    "FLATTENING": 4,
    "FLAT_CONFIRMED": 5,
    "PAUSED": 6,
}


def _normalize_lifecycle_state(state: str) -> str:
    normalized = str(state or "").strip().upper()
    if normalized not in LIFECYCLE_STATE_ORDER:
        raise ValueError(f"Unknown lifecycle state: {state!r}")
    return normalized


def validate_lifecycle_transition(previous_state: str | None, next_state: str) -> None:
    """Raise if a lifecycle transition would move backwards."""

    next_rank = LIFECYCLE_STATE_ORDER[_normalize_lifecycle_state(next_state)]
    if previous_state is None:
        return

    previous_rank = LIFECYCLE_STATE_ORDER[_normalize_lifecycle_state(previous_state)]
    if next_rank < previous_rank:
        raise ValueError(
            f"Lifecycle transition must be monotonic: {previous_state!r} -> {next_state!r}"
        )


@dataclass(frozen=True)
class LifecycleStateRecord:
    """Durable lifecycle state snapshot for an execution session."""

    state: str
    owner: str
    retry_count: int
    reason: str
    policy_version: str
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    transitioned_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def __post_init__(self) -> None:
        object.__setattr__(self, "state", _normalize_lifecycle_state(self.state))
        object.__setattr__(self, "owner", str(self.owner or "").strip())
        object.__setattr__(self, "reason", str(self.reason or "").strip())
        object.__setattr__(self, "policy_version", str(self.policy_version or "").strip())
        object.__setattr__(self, "retry_count", int(self.retry_count))

        if not self.owner:
            raise ValueError("Lifecycle owner cannot be empty")
        if not self.reason:
            raise ValueError("Lifecycle reason cannot be empty")
        if not self.policy_version:
            raise ValueError("Lifecycle policy_version cannot be empty")
        if self.retry_count < 0:
            raise ValueError("retry_count must be >= 0")

    def to_payload(self) -> dict[str, Any]:
        """Return a JSON-serialisable payload for WAL persistence."""

        return asdict(self)

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "LifecycleStateRecord":
        """Restore a record from a WAL payload."""

        timestamps = payload.get("timestamps")
        if isinstance(timestamps, dict):
            created_at = str(timestamps.get("created_at", "") or "")
            transitioned_at = str(timestamps.get("transitioned_at", "") or "")
            updated_at = str(timestamps.get("updated_at", "") or "")
        else:
            created_at = str(payload.get("created_at", "") or "")
            transitioned_at = str(payload.get("transitioned_at", "") or "")
            updated_at = str(payload.get("updated_at", "") or "")

        return cls(
            state=str(payload.get("state", "")),
            owner=str(payload.get("owner", "")),
            retry_count=int(payload.get("retry_count", 0) or 0),
            reason=str(payload.get("reason", "")),
            policy_version=str(payload.get("policy_version", "")),
            created_at=created_at or datetime.now(timezone.utc).isoformat(),
            transitioned_at=transitioned_at or datetime.now(timezone.utc).isoformat(),
            updated_at=updated_at or datetime.now(timezone.utc).isoformat(),
        )


@dataclass(frozen=True)
class WALEntry:
    """A single state mutation event."""

    event_type: str
    user_id: int
    payload: dict[str, Any]
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_json(self) -> str:
        data = asdict(self)
        data["payload"] = _scrub_payload(data.get("payload", {}))
        return json.dumps(data, separators=(",", ":"), default=str)

    @classmethod
    def from_json(cls, raw: str | bytes) -> "WALEntry":
        data = json.loads(raw)
        return cls(
            event_type=data["event_type"],
            user_id=int(data["user_id"]),
            payload=data.get("payload", {}),
            timestamp=data.get("timestamp", ""),
        )


# Event type constants
EVT_SESSION_STARTED = "session_started"
EVT_SESSION_STOPPED = "session_stopped"
EVT_POSITION_UPDATED = "position_updated"
EVT_EQUITY_UPDATED = "equity_updated"
EVT_LIFECYCLE_CHANGED = "lifecycle_changed"
EVT_KILL_SWITCH_TRIGGERED = "kill_switch_triggered"
EVT_KILL_SWITCH_CLEARED = "kill_switch_cleared"
EVT_ORDER_EXECUTED = "order_executed"
EVT_STATE_CHECKPOINT = "state_checkpoint"

WAL_STREAM_KEY = "wal:execution"


class RedisWAL:
    """Redis Streams-backed Write-Ahead Log for execution state.

    Every state mutation is appended to a Redis Stream. On restart,
    the engine replays the stream to reconstruct all active sessions.
    """

    # Default cap: ~100K entries ≈ 50MB. Approximate trim is O(1).
    DEFAULT_MAX_STREAM_LEN = 100_000

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        *,
        max_stream_len: int = DEFAULT_MAX_STREAM_LEN,
    ) -> None:
        self._redis_url = redis_url
        self._redis: Any = None
        self._max_stream_len = max(int(max_stream_len), 1000)

    async def connect(self) -> None:
        try:
            import redis.asyncio as aioredis
        except ImportError:
            raise ImportError(
                "redis[async] package is required. Install with: pip install redis[async]"
            )

        self._redis = aioredis.from_url(
            self._redis_url,
            decode_responses=True,
            retry_on_timeout=True,
        )
        await self._redis.ping()
        logger.info("RedisWAL connected to %s", self._redis_url)

    async def disconnect(self) -> None:
        if self._redis:
            await self._redis.close()
            self._redis = None

    async def append(self, entry: WALEntry) -> str:
        """Append a WAL entry to the Redis Stream. Returns the stream entry ID."""
        if not self._redis:
            raise RuntimeError("RedisWAL is not connected")

        entry_id = await self._redis.xadd(
            WAL_STREAM_KEY,
            {"data": entry.to_json()},
            maxlen=self._max_stream_len,
            approximate=True,
        )
        logger.debug(
            "WAL append: %s user=%d (id=%s)",
            entry.event_type,
            entry.user_id,
            entry_id,
        )
        return entry_id

    async def replay(self, since_id: str = "0-0") -> list[WALEntry]:
        """Read all WAL entries from the stream since a given ID."""
        if not self._redis:
            raise RuntimeError("RedisWAL is not connected")

        raw_entries = await self._redis.xrange(WAL_STREAM_KEY, min=since_id)
        entries: list[WALEntry] = []

        for entry_id, fields in raw_entries:
            try:
                entry = WALEntry.from_json(fields["data"])
                entries.append(entry)
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning("Skipping malformed WAL entry %s: %s", entry_id, e)

        logger.info("WAL replay: %d entries since %s", len(entries), since_id)
        return entries

    async def trim(self, max_entries: int = 50_000) -> int:
        """Trim the WAL stream to keep at most max_entries."""
        if not self._redis:
            raise RuntimeError("RedisWAL is not connected")

        trimmed = await self._redis.xtrim(WAL_STREAM_KEY, maxlen=max_entries, approximate=True)
        if trimmed > 0:
            logger.info("WAL trimmed %d old entries (max=%d)", trimmed, max_entries)
        return trimmed

    async def get_stream_health(self, *, stale_after_seconds: float = 120.0) -> dict[str, Any]:
        """Summarize WAL freshness and append-only depth."""

        if not self._redis:
            raise RuntimeError("RedisWAL is not connected")

        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000.0)

        try:
            raw_info = await self._redis.xinfo_stream(WAL_STREAM_KEY)
        except Exception:
            raw_info = {}
        stream_info = raw_info if isinstance(raw_info, dict) else {}

        entry_count = _coerce_int(stream_info.get("length", 0))
        first_entry = stream_info.get("first-entry")
        last_entry = stream_info.get("last-entry")

        def _entry_id(entry: Any) -> str:
            if isinstance(entry, (list, tuple)) and entry:
                return str(entry[0] or "")
            if isinstance(entry, dict):
                return str(entry.get("id", "") or "")
            return ""

        oldest_entry_id = _entry_id(first_entry)
        latest_entry_id = _entry_id(last_entry)
        oldest_entry_age_seconds = _stream_id_age_seconds(oldest_entry_id, now_ms=now_ms)
        latest_entry_age_seconds = _stream_id_age_seconds(latest_entry_id, now_ms=now_ms)

        status = "unknown"
        if entry_count > 0:
            status = "healthy"
            if latest_entry_age_seconds is not None and latest_entry_age_seconds >= stale_after_seconds:
                status = "warning"
            if latest_entry_age_seconds is not None and latest_entry_age_seconds >= stale_after_seconds * 2.0:
                status = "degraded"

        return {
            "status": status,
            "stream_key": WAL_STREAM_KEY,
            "entry_count": entry_count,
            "oldest_entry_id": oldest_entry_id,
            "latest_entry_id": latest_entry_id,
            "oldest_entry_age_seconds": oldest_entry_age_seconds,
            "latest_entry_age_seconds": latest_entry_age_seconds,
            "stale_after_seconds": float(stale_after_seconds),
        }

    # -- Convenience helpers for common events --

    async def log_session_started(
        self,
        user_id: int,
        *,
        live: bool,
        strategy_profile: str = "",
        universe: tuple[str, ...] = (),
        lifecycle_state: LifecycleStateRecord | None = None,
    ) -> str:
        payload: dict[str, Any] = {
            "live": live,
            "strategy_profile": strategy_profile,
            "universe": list(universe),
        }
        if lifecycle_state is not None:
            payload["lifecycle_state"] = lifecycle_state.to_payload()
        return await self.append(WALEntry(
            event_type=EVT_SESSION_STARTED,
            user_id=user_id,
            payload=payload,
        ))

    async def log_session_stopped(self, user_id: int) -> str:
        return await self.append(WALEntry(
            event_type=EVT_SESSION_STOPPED,
            user_id=user_id,
            payload={},
        ))

    async def log_position_updated(
        self,
        user_id: int,
        *,
        symbol: str,
        quantity: float,
        avg_price: float,
    ) -> str:
        return await self.append(WALEntry(
            event_type=EVT_POSITION_UPDATED,
            user_id=user_id,
            payload={
                "symbol": symbol,
                "quantity": quantity,
                "avg_price": avg_price,
            },
        ))

    async def log_equity_updated(
        self,
        user_id: int,
        *,
        equity_usd: float,
    ) -> str:
        return await self.append(WALEntry(
            event_type=EVT_EQUITY_UPDATED,
            user_id=user_id,
            payload={"equity_usd": equity_usd},
        ))

    async def log_lifecycle_transition(
        self,
        user_id: int,
        *,
        record: LifecycleStateRecord,
    ) -> str:
        return await self.append(WALEntry(
            event_type=EVT_LIFECYCLE_CHANGED,
            user_id=user_id,
            payload=record.to_payload(),
        ))

    async def log_order_executed(
        self,
        user_id: int,
        *,
        symbol: str,
        side: str,
        quantity: float,
        avg_price: float,
        status: str,
        risk_policy_version: str = "",
        outcome: str = "",
        newly_filled_qty: float = 0.0,
        request_id: str = "",
        venue_order_id: str = "",
        fill_id: str = "",
        accounting_transaction_id: str = "",
        original_order_id: str = "",
        original_fill_id: str = "",
        replayed_at: str = "",
        correlation_id: str = "",
    ) -> str:
        return await self.append(WALEntry(
            event_type=EVT_ORDER_EXECUTED,
            user_id=user_id,
            payload={
                "symbol": symbol,
                "side": side,
                "quantity": quantity,
                "avg_price": avg_price,
                "status": status,
                "risk_policy_version": risk_policy_version,
                "outcome": outcome,
                "newly_filled_qty": newly_filled_qty,
                "request_id": request_id,
                "venue_order_id": venue_order_id,
                "fill_id": fill_id,
                "accounting_transaction_id": accounting_transaction_id,
                "original_order_id": original_order_id,
                "original_fill_id": original_fill_id,
                "replayed_at": replayed_at,
                "correlation_id": correlation_id,
            },
        ))

    async def log_state_checkpoint(
        self,
        user_id: int,
        *,
        equity_baseline_usd: float,
        open_positions: dict[str, float],
        paper_entry_prices: dict[str, float],
    ) -> str:
        return await self.append(WALEntry(
            event_type=EVT_STATE_CHECKPOINT,
            user_id=user_id,
            payload={
                "equity_baseline_usd": equity_baseline_usd,
                "open_positions": {k: v for k, v in open_positions.items() if abs(v) > 1e-12},
                "paper_entry_prices": dict(paper_entry_prices),
            },
        ))

    async def log_kill_switch(self, user_id: int, *, triggered: bool, reasons: tuple[str, ...] = ()) -> str:
        evt = EVT_KILL_SWITCH_TRIGGERED if triggered else EVT_KILL_SWITCH_CLEARED
        return await self.append(WALEntry(
            event_type=evt,
            user_id=user_id,
            payload={"reasons": list(reasons)},
        ))


class InMemoryWAL:
    """In-memory WAL for testing (no Redis dependency)."""

    def __init__(self) -> None:
        self._entries: list[WALEntry] = []
        self._seq = 0

    async def connect(self) -> None:
        pass

    async def disconnect(self) -> None:
        pass

    async def append(self, entry: WALEntry) -> str:
        self._seq += 1
        self._entries.append(entry)
        return f"{self._seq}-0"

    async def replay(self, since_id: str = "0-0") -> list[WALEntry]:
        return list(self._entries)

    async def trim(self, max_entries: int = 50_000) -> int:
        overflow = max(0, len(self._entries) - max_entries)
        if overflow > 0:
            self._entries = self._entries[overflow:]
        return overflow

    # Convenience helpers matching RedisWAL interface
    async def log_session_started(self, user_id: int, *, live: bool, strategy_profile: str = "", universe: tuple[str, ...] = (), lifecycle_state: LifecycleStateRecord | None = None) -> str:
        payload: dict[str, Any] = {
            "live": live,
            "strategy_profile": strategy_profile,
            "universe": list(universe),
        }
        if lifecycle_state is not None:
            payload["lifecycle_state"] = lifecycle_state.to_payload()
        return await self.append(WALEntry(event_type=EVT_SESSION_STARTED, user_id=user_id, payload=payload))

    async def log_session_stopped(self, user_id: int) -> str:
        return await self.append(WALEntry(event_type=EVT_SESSION_STOPPED, user_id=user_id, payload={}))

    async def log_position_updated(self, user_id: int, *, symbol: str, quantity: float, avg_price: float) -> str:
        return await self.append(WALEntry(event_type=EVT_POSITION_UPDATED, user_id=user_id, payload={"symbol": symbol, "quantity": quantity, "avg_price": avg_price}))

    async def log_equity_updated(self, user_id: int, *, equity_usd: float) -> str:
        return await self.append(WALEntry(event_type=EVT_EQUITY_UPDATED, user_id=user_id, payload={"equity_usd": equity_usd}))

    async def log_lifecycle_transition(self, user_id: int, *, record: LifecycleStateRecord) -> str:
        return await self.append(WALEntry(event_type=EVT_LIFECYCLE_CHANGED, user_id=user_id, payload=record.to_payload()))

    async def log_order_executed(
        self,
        user_id: int,
        *,
        symbol: str,
        side: str,
        quantity: float,
        avg_price: float,
        status: str,
        risk_policy_version: str = "",
        outcome: str = "",
        newly_filled_qty: float = 0.0,
        request_id: str = "",
        venue_order_id: str = "",
        fill_id: str = "",
        accounting_transaction_id: str = "",
        original_order_id: str = "",
        original_fill_id: str = "",
        replayed_at: str = "",
        correlation_id: str = "",
    ) -> str:
        return await self.append(
            WALEntry(
                event_type=EVT_ORDER_EXECUTED,
                user_id=user_id,
                payload={
                    "symbol": symbol,
                    "side": side,
                    "quantity": quantity,
                    "avg_price": avg_price,
                    "status": status,
                    "risk_policy_version": risk_policy_version,
                    "outcome": outcome,
                    "newly_filled_qty": newly_filled_qty,
                    "request_id": request_id,
                    "venue_order_id": venue_order_id,
                    "fill_id": fill_id,
                    "accounting_transaction_id": accounting_transaction_id,
                    "original_order_id": original_order_id,
                    "original_fill_id": original_fill_id,
                    "replayed_at": replayed_at,
                    "correlation_id": correlation_id,
                },
            )
        )

    async def log_state_checkpoint(self, user_id: int, *, equity_baseline_usd: float, open_positions: dict[str, float], paper_entry_prices: dict[str, float]) -> str:
        return await self.append(WALEntry(event_type=EVT_STATE_CHECKPOINT, user_id=user_id, payload={"equity_baseline_usd": equity_baseline_usd, "open_positions": {k: v for k, v in open_positions.items() if abs(v) > 1e-12}, "paper_entry_prices": dict(paper_entry_prices)}))

    async def log_kill_switch(self, user_id: int, *, triggered: bool, reasons: tuple[str, ...] = ()) -> str:
        evt = EVT_KILL_SWITCH_TRIGGERED if triggered else EVT_KILL_SWITCH_CLEARED
        return await self.append(WALEntry(event_type=evt, user_id=user_id, payload={"reasons": list(reasons)}))
