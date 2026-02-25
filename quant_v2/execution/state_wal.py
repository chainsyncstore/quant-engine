"""Write-Ahead Log (WAL) for crash-resilient execution state persistence."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any

import re

logger = logging.getLogger(__name__)

_SENSITIVE_KEYS = re.compile(
    r"(api_key|api_secret|secret|password|token|credentials)",
    re.IGNORECASE,
)


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

    # -- Convenience helpers for common events --

    async def log_session_started(
        self,
        user_id: int,
        *,
        live: bool,
        strategy_profile: str = "",
        universe: tuple[str, ...] = (),
    ) -> str:
        return await self.append(WALEntry(
            event_type=EVT_SESSION_STARTED,
            user_id=user_id,
            payload={
                "live": live,
                "strategy_profile": strategy_profile,
                "universe": list(universe),
            },
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

    async def log_order_executed(
        self,
        user_id: int,
        *,
        symbol: str,
        side: str,
        quantity: float,
        avg_price: float,
        status: str,
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
    async def log_session_started(self, user_id: int, *, live: bool, strategy_profile: str = "", universe: tuple[str, ...] = ()) -> str:
        return await self.append(WALEntry(event_type=EVT_SESSION_STARTED, user_id=user_id, payload={"live": live, "strategy_profile": strategy_profile, "universe": list(universe)}))

    async def log_session_stopped(self, user_id: int) -> str:
        return await self.append(WALEntry(event_type=EVT_SESSION_STOPPED, user_id=user_id, payload={}))

    async def log_position_updated(self, user_id: int, *, symbol: str, quantity: float, avg_price: float) -> str:
        return await self.append(WALEntry(event_type=EVT_POSITION_UPDATED, user_id=user_id, payload={"symbol": symbol, "quantity": quantity, "avg_price": avg_price}))

    async def log_equity_updated(self, user_id: int, *, equity_usd: float) -> str:
        return await self.append(WALEntry(event_type=EVT_EQUITY_UPDATED, user_id=user_id, payload={"equity_usd": equity_usd}))

    async def log_order_executed(self, user_id: int, *, symbol: str, side: str, quantity: float, avg_price: float, status: str) -> str:
        return await self.append(WALEntry(event_type=EVT_ORDER_EXECUTED, user_id=user_id, payload={"symbol": symbol, "side": side, "quantity": quantity, "avg_price": avg_price, "status": status}))

    async def log_kill_switch(self, user_id: int, *, triggered: bool, reasons: tuple[str, ...] = ()) -> str:
        evt = EVT_KILL_SWITCH_TRIGGERED if triggered else EVT_KILL_SWITCH_CLEARED
        return await self.append(WALEntry(event_type=evt, user_id=user_id, payload={"reasons": list(reasons)}))
