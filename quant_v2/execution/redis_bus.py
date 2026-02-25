"""Redis Pub/Sub command bus for decoupling Telegram I/O from execution engine."""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Awaitable

logger = logging.getLogger(__name__)

# Channel names
CMD_EXEC_CHANNEL = "cmd:exec"
EVT_TG_CHANNEL = "evt:tg"


@dataclass(frozen=True)
class BusMessage:
    """Envelope for inter-service messages."""

    action: str
    payload: dict[str, Any]
    timestamp: str
    correlation_id: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self), separators=(",", ":"), default=str)

    @classmethod
    def from_json(cls, raw: str | bytes) -> "BusMessage":
        data = json.loads(raw)
        return cls(
            action=data["action"],
            payload=data.get("payload", {}),
            timestamp=data.get("timestamp", ""),
            correlation_id=data.get("correlation_id", ""),
        )


class RedisCommandBus:
    """Async Redis Pub/Sub bridge between Telegram and Execution containers.

    Usage (publisher side):
        bus = RedisCommandBus(redis_url)
        await bus.connect()
        await bus.publish("cmd:exec", BusMessage(action="start_session", ...))

    Usage (subscriber side):
        bus = RedisCommandBus(redis_url)
        await bus.connect()
        await bus.subscribe("cmd:exec", handler_coroutine)
    """

    def __init__(self, redis_url: str = "redis://localhost:6379") -> None:
        self._redis_url = redis_url
        self._redis: Any = None
        self._pubsub: Any = None
        self._listener_task: asyncio.Task | None = None
        self._handlers: dict[str, list[Callable[[BusMessage], Awaitable[None]]]] = {}

    async def connect(self) -> None:
        """Connect to Redis."""
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
        logger.info("RedisCommandBus connected to %s", self._redis_url)

    async def disconnect(self) -> None:
        """Cleanly shut down subscriptions and connection."""
        if self._listener_task and not self._listener_task.done():
            self._listener_task.cancel()
            try:
                await self._listener_task
            except asyncio.CancelledError:
                pass

        if self._pubsub:
            await self._pubsub.unsubscribe()
            await self._pubsub.close()
            self._pubsub = None

        if self._redis:
            await self._redis.close()
            self._redis = None

        logger.info("RedisCommandBus disconnected")

    async def publish(self, channel: str, message: BusMessage) -> int:
        """Publish a message to a Redis Pub/Sub channel."""
        if not self._redis:
            raise RuntimeError("RedisCommandBus is not connected")

        receivers = await self._redis.publish(channel, message.to_json())
        logger.debug(
            "Published %s to %s (%d receivers)", message.action, channel, receivers
        )
        return receivers

    async def subscribe(
        self,
        channel: str,
        handler: Callable[[BusMessage], Awaitable[None]],
    ) -> None:
        """Subscribe to a channel and dispatch messages to handler."""
        if not self._redis:
            raise RuntimeError("RedisCommandBus is not connected")

        if channel not in self._handlers:
            self._handlers[channel] = []
        self._handlers[channel].append(handler)

        if self._pubsub is None:
            self._pubsub = self._redis.pubsub()

        await self._pubsub.subscribe(channel)
        logger.info("Subscribed to channel: %s", channel)

        if self._listener_task is None or self._listener_task.done():
            self._listener_task = asyncio.create_task(self._listen_loop())

    async def _listen_loop(self) -> None:
        """Background listener that dispatches incoming messages."""
        try:
            async for raw_message in self._pubsub.listen():
                if raw_message["type"] != "message":
                    continue

                channel = raw_message["channel"]
                handlers = self._handlers.get(channel, [])
                if not handlers:
                    continue

                try:
                    msg = BusMessage.from_json(raw_message["data"])
                except (json.JSONDecodeError, KeyError) as e:
                    logger.warning("Malformed message on %s: %s", channel, e)
                    continue

                for handler in handlers:
                    try:
                        await handler(msg)
                    except Exception:
                        logger.exception(
                            "Handler error for action=%s on channel=%s",
                            msg.action,
                            channel,
                        )
        except asyncio.CancelledError:
            logger.debug("Listener loop cancelled")
        except Exception:
            logger.exception("Listener loop crashed")

    # -- Convenience helpers for typed publishing --

    async def send_command(
        self,
        action: str,
        payload: dict[str, Any],
        correlation_id: str = "",
    ) -> int:
        """Publish a command to the execution engine."""
        msg = BusMessage(
            action=action,
            payload=payload,
            timestamp=datetime.now(timezone.utc).isoformat(),
            correlation_id=correlation_id,
        )
        return await self.publish(CMD_EXEC_CHANNEL, msg)

    async def send_event(
        self,
        action: str,
        payload: dict[str, Any],
        correlation_id: str = "",
    ) -> int:
        """Publish an event back to Telegram."""
        msg = BusMessage(
            action=action,
            payload=payload,
            timestamp=datetime.now(timezone.utc).isoformat(),
            correlation_id=correlation_id,
        )
        return await self.publish(EVT_TG_CHANNEL, msg)
