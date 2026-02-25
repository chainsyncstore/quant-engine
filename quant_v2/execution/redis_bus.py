"""Redis Pub/Sub command bus for decoupling Telegram I/O from execution engine."""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Awaitable

logger = logging.getLogger(__name__)

# Channel names (Pub/Sub - used for engine → Telegram event replies only)
CMD_EXEC_CHANNEL = "cmd:exec"
EVT_TG_CHANNEL = "evt:tg"

# Stream names (Redis Streams - used for Telegram → engine guaranteed commands)
STREAM_CMD_KEY = "stream:cmd:exec"
STREAM_CONSUMER_GROUP = "execution_consumer_group"
STREAM_CONSUMER_NAME = "execution_engine"


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


class RedisStreamCommandBus:
    """At-least-once guaranteed command bus using Redis Streams + Consumer Groups.

    Replaces the fire-and-forget Pub/Sub channel for the safety-critical
    Telegram → Execution command path (XADD enqueue, XREADGROUP consume, XACK commit).

    If the execution container is down when a /stop or /set_stoploss command
    is sent, the command queues in the Redis Stream and is processed immediately
    on container restart — no commands are ever silently lost.

    Usage (Telegram/publisher side):
        bus = RedisStreamCommandBus(redis_url)
        await bus.connect()
        msg_id = await bus.enqueue(BusMessage(action="stop_session", ...))

    Usage (ExecutionEngine/consumer side):
        bus = RedisStreamCommandBus(redis_url)
        await bus.connect()
        await bus.start_consuming(handler_coroutine)
    """

    DEFAULT_MAX_STREAM_LEN = 100_000
    GRACEFUL_DRAIN_TIMEOUT_S = 10.0

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        *,
        stream_key: str = STREAM_CMD_KEY,
        consumer_group: str = STREAM_CONSUMER_GROUP,
        consumer_name: str = STREAM_CONSUMER_NAME,
        batch_size: int = 10,
        block_ms: int = 2000,
        max_stream_len: int = DEFAULT_MAX_STREAM_LEN,
    ) -> None:
        self._redis_url = redis_url
        self._stream_key = stream_key
        self._consumer_group = consumer_group
        self._consumer_name = consumer_name
        self._batch_size = batch_size
        self._block_ms = block_ms
        self._max_stream_len = max(int(max_stream_len), 1000)
        self._redis: Any = None
        self._consume_task: asyncio.Task | None = None
        self._running = False

    async def connect(self) -> None:
        """Connect to Redis and ensure the consumer group exists."""
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

        # Create the consumer group if it doesn't already exist.
        # '>' means: start reading new messages from now (existing commands in stream
        # before the first consumer group creation are NOT replayed).
        # MKSTREAM creates the stream if absent.
        try:
            await self._redis.xgroup_create(
                self._stream_key,
                self._consumer_group,
                id="0",  # '0' = consume from beginning on first boot; pending messages survive restart
                mkstream=True,
            )
            logger.info(
                "Created consumer group '%s' on stream '%s'",
                self._consumer_group,
                self._stream_key,
            )
        except Exception as exc:
            # BusyGroup error means the group already exists — safe to continue.
            if "BUSYGROUP" in str(exc):
                logger.debug(
                    "Consumer group '%s' already exists on '%s'",
                    self._consumer_group,
                    self._stream_key,
                )
            else:
                raise

        logger.info(
            "RedisStreamCommandBus connected (stream=%s, group=%s, consumer=%s)",
            self._stream_key,
            self._consumer_group,
            self._consumer_name,
        )

    async def disconnect(self) -> None:
        """Stop consuming and close the Redis connection.

        Uses a graceful drain: sets _running=False so the consumer
        loop exits after finishing the current in-flight handler,
        then waits up to GRACEFUL_DRAIN_TIMEOUT_S before force-cancelling.
        This prevents orphaned exchange orders on SIGTERM.
        """
        self._running = False
        if self._consume_task and not self._consume_task.done():
            logger.info("Waiting for active stream consumer to drain...")
            try:
                await asyncio.wait_for(
                    self._consume_task,
                    timeout=self.GRACEFUL_DRAIN_TIMEOUT_S,
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "Consumer drain timeout (%.0fs). Force cancelling.",
                    self.GRACEFUL_DRAIN_TIMEOUT_S,
                )
                self._consume_task.cancel()
                try:
                    await self._consume_task
                except asyncio.CancelledError:
                    pass
            except asyncio.CancelledError:
                pass
        if self._redis:
            await self._redis.close()
            self._redis = None
        logger.info("RedisStreamCommandBus disconnected")

    async def enqueue(self, msg: BusMessage) -> str:
        """Publish a command to the stream. Returns the Redis stream entry ID.

        Used by the Telegram container to enqueue commands durably.
        """
        if not self._redis:
            raise RuntimeError("RedisStreamCommandBus is not connected")

        entry_id = await self._redis.xadd(
            self._stream_key,
            {"data": msg.to_json()},
            maxlen=self._max_stream_len,
            approximate=True,
        )
        logger.debug(
            "Stream enqueue: action=%s id=%s", msg.action, entry_id
        )
        return entry_id

    async def start_consuming(
        self,
        handler: Callable[[BusMessage], Awaitable[None]],
    ) -> None:
        """Start the background consumer loop.

        Used by the execution engine container. Messages are ACKed only after
        the handler coroutine returns without raising, ensuring at-least-once
        processing even across container restarts.
        """
        if not self._redis:
            raise RuntimeError("RedisStreamCommandBus is not connected")
        self._running = True
        self._consume_task = asyncio.create_task(
            self._consume_loop(handler),
            name="stream_cmd_consumer",
        )
        logger.info("Stream consumer started (stream=%s)", self._stream_key)

    async def stop_consuming(self) -> None:
        """Stop the background consumer loop gracefully.

        Waits for the in-flight handler to finish before cancelling
        to prevent orphaned exchange orders.
        """
        self._running = False
        if self._consume_task and not self._consume_task.done():
            logger.info("Waiting for active stream consumer to drain...")
            try:
                await asyncio.wait_for(
                    self._consume_task,
                    timeout=self.GRACEFUL_DRAIN_TIMEOUT_S,
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "Consumer drain timeout (%.0fs). Force cancelling.",
                    self.GRACEFUL_DRAIN_TIMEOUT_S,
                )
                self._consume_task.cancel()
                try:
                    await self._consume_task
                except asyncio.CancelledError:
                    pass
            except asyncio.CancelledError:
                pass
        logger.info("Stream consumer stopped gracefully")

    async def _consume_loop(
        self,
        handler: Callable[[BusMessage], Awaitable[None]],
    ) -> None:
        """Main consumer loop: XREADGROUP → handle → XACK."""
        # First pass: re-deliver any pending (unacknowledged) messages from
        # a previous container run so nothing is ever silently dropped.
        await self._drain_pending(handler)

        try:
            while self._running:
                # '>' means: fetch only new messages not yet delivered to any consumer.
                raw = await self._redis.xreadgroup(
                    self._consumer_group,
                    self._consumer_name,
                    {self._stream_key: ">"},
                    count=self._batch_size,
                    block=self._block_ms,
                )
                if not raw:
                    continue  # Timeout — no new messages, loop again.

                for _stream, entries in raw:
                    for entry_id, fields in entries:
                        await self._process_entry(entry_id, fields, handler)

        except asyncio.CancelledError:
            logger.debug("Stream consumer loop cancelled")
        except Exception:
            logger.exception("Stream consumer loop crashed")

    async def _drain_pending(
        self,
        handler: Callable[[BusMessage], Awaitable[None]],
    ) -> None:
        """On startup, re-process any messages that were delivered but not ACKed."""
        try:
            pending = await self._redis.xpending_range(
                self._stream_key,
                self._consumer_group,
                min="-",
                max="+",
                count=500,
            )
        except Exception:
            logger.warning("Could not fetch pending stream entries — skipping drain")
            return

        if not pending:
            return

        logger.warning(
            "Draining %d pending (unacknowledged) stream entries on boot",
            len(pending),
        )
        for item in pending:
            entry_id = item["message_id"]
            raw_entries = await self._redis.xrange(
                self._stream_key, min=entry_id, max=entry_id
            )
            for _, fields in raw_entries:
                await self._process_entry(entry_id, fields, handler)

    async def _process_entry(
        self,
        entry_id: str,
        fields: dict[str, str],
        handler: Callable[[BusMessage], Awaitable[None]],
    ) -> None:
        """Parse a stream entry, invoke the handler, and ACK on success."""
        try:
            msg = BusMessage.from_json(fields["data"])
        except (json.JSONDecodeError, KeyError) as exc:
            logger.warning("Malformed stream entry %s: %s — ACKing to skip", entry_id, exc)
            await self._redis.xack(self._stream_key, self._consumer_group, entry_id)
            return

        try:
            await handler(msg)
            # ACK only after successful processing — crash before here means
            # the message will be re-delivered on next boot.
            await self._redis.xack(self._stream_key, self._consumer_group, entry_id)
            logger.debug("Stream ACK: action=%s id=%s", msg.action, entry_id)
        except Exception:
            logger.exception(
                "Stream handler error for action=%s id=%s — NOT ACKing (will retry on restart)",
                msg.action,
                entry_id,
            )
