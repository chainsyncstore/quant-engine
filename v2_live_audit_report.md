# Quant Engine V2 - Live Capital Readiness Audit

## Final Score: 45 / 100 (NOT SAFE FOR LIVE CAPITAL)
While the architectural transition from a monolithic engine to an event-driven, multi-container architecture is a massive step forward, the current implementation contains critical structural flaws. These flaws guarantee dropped orders during volatility spikes, complete state amnesia upon container restarts, and frozen event loops that will prevent stop-losses from triggering.

This codebase requires structural patching before it can be trusted with live capital.

---

## 1. The Vulnerability Report (Critical Path Failures)

### A. Concurrency & Event Loop Bottlenecks (GIL/I/O Freezes)
- **Location:** `quant_v2/execution/binance_adapter.py` and `quant_v2/execution/main.py`
- **Vulnerability:** The execution engine runs a single `asyncio` event loop in `main.py` that listens to the Redis command bus and Watchdog. However, the `BinanceExecutionAdapter` uses a **synchronous** `BinanceClient`. When `_cmd_route_signals` routes a signal, it calls `adapter.place_order()`. Because this network request is synchronous, **it blocks the entire asyncio event loop** until Binance responds.
- **Impact:** During high volatility (when Binance API latency spikes to 500ms-2s), the execution container will freeze. It will not process new Telegram commands (like `/stop`), it will drop incoming market data ticks, and the Watchdog will pause.

### B. Message Delivery Guarantees (Lost Life-or-Death Commands)
- **Location:** `quant_v2/execution/redis_bus.py`
- **Vulnerability:** The cross-container communication uses standard Redis Pub/Sub (`cmd:exec`, `evt:tg`). Redis Pub/Sub is strictly "fire-and-forget."
- **Impact:** If the execution container restarts or is momentarily disconnected from Redis while the user sends a critical `/set_stoploss` or `/stop` command from Telegram, the message is permanently lost. The system assumes it was received, leaving the user completely exposed.

### C. State Persistence (Complete Memory Amnesia)
- **Location:** `quant_v2/execution/main.py` and `quant_v2/execution/state_wal.py`
- **Vulnerability:** A robust Write-Ahead Log (`RedisWAL`) was implemented to persist execution state to Redis Streams. However, in `main.py`'s `start()` method, the engine connects to the WAL but **never calls `await self._wal.replay()`**.
- **Impact:** If the execution container suffers an OOM kill or hardware fault, it boots up completely blank. It loses track of all active user sessions, dynamically adjusted stop-losses, user intents, and diagnostics. While it can sync physical positions from Binance, the *quant logic constraints* (risk limits, lifecycle rules) for those positions are wiped out.

### D. Tick Starvation on Stop-Losses
- **Location:** `quant_v2/execution/watchdog.py` and `quant_v2/execution/main.py`
- **Vulnerability:** Stop-loss checks depend on `session.latest_mtm_equity_usd`. This variable is *only* updated when new signals/prices arrive in `_cmd_route_signals`. If the market data feed drops (tick starvation), MTM equity freezes. While the Watchdog effectively detects this and fires a `heartbeat_stale` alert, `main.py` **does not auto-flatten positions on heartbeat stale**; it only flattens on `horizon_expired` or `stop_loss_triggered`.
- **Impact:** If the websocket data connection drops while the market is crashing, the stop-loss will never trigger. The watchdog will just helplessly emit warning logs while your capital evaporates.

---

## 2. The Strengths (Institutional-Grade Decisions)

1. **Mark-to-Market (MTM) Reality Checks:** `adapter.compute_mtm_equity()` correctly utilizes orderbook Top (`get_orderbook_top`) instead of generic "Last Traded Price". It conservatively values longs at the Bid and shorts at the Ask. This prevents paper-profit illusions and protects against wide-spread illiquid spikes.
2. **Post-Only Institutional Routing:** Limit orders dynamically adjust and strictly enforce `post_only=True` to capture the spread, gracefully falling back to a 15bps bounded limit order if immediately matched (Error `-2010`). This is a highly mature execution algorithm.
3. **Tick-Immune Horizon Rules:** The Watchdog enforces time-based lifecycle rules (`horizon_deadline_utc`) using strict UTC wall-clock time (`now >= session.horizon_deadline_utc`) rather than counting ticks. This makes time-based exits fully resilient to data outages.

---

## 3. Actionable Fixes (The Patch Plan)

### Fix 1: Unblock the Event Loop (Concurrency)
Wrap the synchronous adapter calls in `asyncio.to_thread()` or `run_in_executor()` inside `service.py` or `binance_adapter.py`.
```python
# In RoutedExecutionService (service.py):
result = await asyncio.to_thread(
    state.adapter.place_order,
    plan,
    idempotency_key=idempotency_key,
    mark_price=mark_price,
    limit_price=limit_price,
    post_only=True,
)
```
*(Alternatively, migrate to an asynchronous client like `aiohttp` or `binance.AsyncClient`.)*

### Fix 2: Upgrade to Reliable Messaging (Delivery Guarantees)
Deprecate Redis Pub/Sub in `redis_bus.py`. Switch to **Redis Streams** with Consumer Groups using `XADD`, `XREADGROUP`, and `XACK`. This ensures that if the execution container is down, incoming Telegram commands queue in the stream and are processed immediately upon reboot.

### Fix 3: Implement WAL Replay on Boot (Fault Tolerance)
In `main.py`, rebuild the `_sessions` map from the WAL before starting the command listener:
```python
async def start(self) -> None:
    await self._bus.connect()
    await self._wal.connect()
    
    # REPLAY WAL to rebuild active sessions
    entries = await self._wal.replay("0-0")
    self._rebuild_state_from_wal(entries)
    
    await self._watchdog.start()
    await self._bus.subscribe(CMD_EXEC_CHANNEL, self._handle_command)
```

### Fix 4: Auto-Flatten on Tick Starvation
In `main.py`, treat `heartbeat_stale` (loss of market data) as a critical risk failure and trigger the kill switch.
```python
async def _handle_watchdog_alert(self, alert: WatchdogAlert) -> None:
    # Auto-flatten positions if we lose data feed, hit stop-loss, or horizon expires
    if alert.alert_type in ("stop_loss_triggered", "horizon_expired", "heartbeat_stale"):
        logger.info("Auto-flattening positions for user=%d due to %s", alert.user_id, alert.alert_type)
        await self._wal.log_kill_switch(alert.user_id, triggered=True, reasons=(alert.alert_type,))
```
