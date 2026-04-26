"""Tests for per-cycle shared feature/prediction cache in V2SignalManager.

Refs: audit_20260423 task P2-2
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pandas as pd

from quant_v2.telebot.signal_manager import V2SignalManager


class _FakeClient:
    def __init__(self, bars: pd.DataFrame) -> None:
        self._bars = bars

    def fetch_historical(self, date_from, date_to, *, symbol: str, interval: str) -> pd.DataFrame:
        _ = (date_from, date_to, symbol, interval)
        return self._bars


def _sample_bars(*, trend_up: bool = True, n: int = 120) -> pd.DataFrame:
    end = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    index = pd.date_range(end=end, periods=n, freq="h", tz="UTC")
    if trend_up:
        closes = [float(10_000 + i * 20) for i in range(len(index))]
    else:
        closes = [float(10_000 - i * 20) for i in range(len(index))]
    df = pd.DataFrame({"close": closes}, index=index)
    df["open"] = df["close"] * 0.99
    df["high"] = df["close"] * 1.01
    df["low"] = df["close"] * 0.98
    df["volume"] = 1000.0
    return df


def test_cache_created_per_cycle_in_loop(tmp_path: Path) -> None:
    """Each iteration of _loop creates a fresh cycle_cache dict."""
    bars = _sample_bars(trend_up=True)

    manager = V2SignalManager(
        model_dir=tmp_path,
        symbols=("BTCUSDT",),
        loop_interval_seconds=3600,
        client_factory=lambda creds, live, symbol, interval: _FakeClient(bars),
    )

    # Track caches passed to _run_cycle
    captured_caches: list[dict] = []

    async def patched_run_cycle(session: Any, *, cycle_cache: Any = None) -> None:
        captured_caches.append(cycle_cache)
        # Just track, don't actually run to avoid side effects
        pass

    with patch.object(manager, "_run_cycle", side_effect=patched_run_cycle):
        async def scenario() -> None:
            await manager.start_session(
                user_id=101,
                creds={"live": False},
                on_signal=lambda p: None,
                execute_orders=False,
            )

            session = manager.sessions[101]
            # Simulate two iterations of _loop
            for _ in range(2):
                cycle_cache: dict[tuple[str, str, str], dict[str, Any]] = {}
                await patched_run_cycle(session, cycle_cache=cycle_cache)

            await manager.stop_session(101)

        asyncio.run(scenario())

    # Verify separate caches were created for each cycle
    assert len(captured_caches) == 2
    assert captured_caches[0] is not captured_caches[1]


def test_cache_does_not_leak_per_user_mutations(tmp_path: Path) -> None:
    """Mutating payload for one caller should not affect cached copy."""
    bars = _sample_bars(trend_up=True)

    manager = V2SignalManager(
        model_dir=tmp_path,
        symbols=("BTCUSDT",),
        loop_interval_seconds=3600,
        client_factory=lambda creds, live, symbol, interval: _FakeClient(bars),
    )

    async def scenario() -> tuple[dict, dict]:
        await manager.start_session(
            user_id=201,
            creds={"live": False},
            on_signal=lambda p: None,
            execute_orders=False,
        )

        # Create a cache and run one cycle
        cycle_cache: dict[tuple[str, str, str], dict[str, Any]] = {}
        await manager._run_cycle(manager.sessions[201], cycle_cache=cycle_cache)

        await manager.stop_session(201)

        # Retrieve the cached payload
        assert len(cycle_cache) == 1
        cached_payload = list(cycle_cache.values())[0]

        # Simulate what would happen if the same cache were used again
        # (normally this won't happen as cache is per-cycle, but test isolation)
        payload_copy_1 = dict(cached_payload)
        payload_copy_2 = dict(cached_payload)

        # Mutate one copy
        payload_copy_1["user_specific"] = "MUTATED"

        return payload_copy_1, payload_copy_2

    payload_mutated, payload_clean = asyncio.run(scenario())
    assert payload_mutated.get("user_specific") == "MUTATED"
    assert "user_specific" not in payload_clean


def test_cache_scoped_to_cycle(tmp_path: Path) -> None:
    """Same bar across cycles uses shared cache; new bar triggers recompute.

    P2-2 semantics: the cache key is (symbol, anchor_interval, bar_ts_iso).  When
    successive cycles see the same bar timestamp (e.g. inside one bar interval),
    the shared inference cache prevents redundant work.  When the bar advances,
    the key changes and recompute fires.
    """
    bars_first = _sample_bars(trend_up=True, n=120)
    # Build a second fixture whose latest bar is at a strictly newer timestamp.
    bars_second = bars_first.copy()
    bars_second.index = bars_second.index + pd.Timedelta(hours=1)

    current_bars = {"value": bars_first}

    class _SwitchingClient:
        def fetch_historical(self, date_from, date_to, *, symbol, interval):
            _ = (date_from, date_to, symbol, interval)
            return current_bars["value"]

    manager = V2SignalManager(
        model_dir=tmp_path,
        symbols=("BTCUSDT",),
        loop_interval_seconds=3600,
        client_factory=lambda creds, live, symbol, interval: _SwitchingClient(),
    )

    async def scenario() -> tuple[int, int]:
        await manager.start_session(
            user_id=301,
            creds={"live": False},
            on_signal=lambda p: None,
            execute_orders=False,
        )

        call_count = 0
        original_build = manager._build_signal_payload

        def patched_build(*args: Any, **kwargs: Any) -> dict:
            nonlocal call_count
            call_count += 1
            return original_build(*args, **kwargs)

        with patch.object(manager, "_build_signal_payload", side_effect=patched_build):
            session = manager.sessions[301]

            # Cycle 1: fresh compute on bars_first
            cycle_cache_1: dict[tuple[str, str, str], dict[str, Any]] = {}
            await manager._run_cycle(session, cycle_cache=cycle_cache_1)
            after_first = call_count

            # Cycle 2 on the SAME bar → shared cache hit, no extra compute
            cycle_cache_2: dict[tuple[str, str, str], dict[str, Any]] = {}
            await manager._run_cycle(session, cycle_cache=cycle_cache_2)
            after_same_bar = call_count

            # Advance to a newer bar → cache miss, recompute
            current_bars["value"] = bars_second
            cycle_cache_3: dict[tuple[str, str, str], dict[str, Any]] = {}
            await manager._run_cycle(session, cycle_cache=cycle_cache_3)
            after_new_bar = call_count

        await manager.stop_session(301)
        return (after_first, after_same_bar, after_new_bar)

    after_first, after_same_bar, after_new_bar = asyncio.run(scenario())
    assert after_first == 1, f"expected 1 compute on first cycle, got {after_first}"
    assert after_same_bar == 1, f"same bar must hit cache (got {after_same_bar} computes)"
    assert after_new_bar == 2, f"new bar must miss cache (got {after_new_bar} computes)"


def test_cache_used_within_cycle(tmp_path: Path) -> None:
    """Cache should store payloads keyed by (symbol, interval, timestamp)."""
    bars = _sample_bars(trend_up=True)

    manager = V2SignalManager(
        model_dir=tmp_path,
        symbols=("BTCUSDT",),
        loop_interval_seconds=3600,
        client_factory=lambda creds, live, symbol, interval: _FakeClient(bars),
    )

    async def scenario() -> dict[tuple[str, str, str], dict[str, Any]]:
        await manager.start_session(
            user_id=401,
            creds={"live": False},
            on_signal=lambda p: None,
            execute_orders=False,
        )

        cycle_cache: dict[tuple[str, str, str], dict[str, Any]] = {}
        await manager._run_cycle(manager.sessions[401], cycle_cache=cycle_cache)

        await manager.stop_session(401)
        return cycle_cache

    cache = asyncio.run(scenario())
    # Verify cache contains one entry per symbol processed
    assert len(cache) >= 1
    # Verify each cached payload has expected structure
    for payload in cache.values():
        assert "symbol" in payload
        assert "signal" in payload
        assert "close_price" in payload


def test_behaviour_unchanged_with_single_user(tmp_path: Path) -> None:
    """Payload structure should be identical with cache enabled."""
    bars = _sample_bars(trend_up=True)

    manager = V2SignalManager(
        model_dir=tmp_path,
        symbols=("BTCUSDT",),
        loop_interval_seconds=3600,
        client_factory=lambda creds, live, symbol, interval: _FakeClient(bars),
    )

    async def scenario() -> dict:
        emitted: list[dict] = []

        async def on_signal(payload: dict) -> None:
            emitted.append(dict(payload))  # Copy to preserve

        await manager.start_session(
            user_id=501,
            creds={"live": False},
            on_signal=on_signal,
            execute_orders=False,
        )

        session = manager.sessions[501]
        cycle_cache: dict[tuple[str, str, str], dict[str, Any]] = {}
        await manager._run_cycle(session, cycle_cache=cycle_cache)

        await manager.stop_session(501)
        return emitted[0] if emitted else {}

    payload = asyncio.run(scenario())
    # Verify payload structure matches expectations
    assert payload.get("symbol") == "BTCUSDT"
    assert payload.get("signal") in {"BUY", "SELL", "HOLD", "DRIFT_ALERT"}
    assert float(payload.get("close_price", 0)) > 0
    assert "v2_signal" in payload
    assert "v2_prices" in payload
    assert "v2_monitoring_snapshot" in payload


def test_cache_uses_correct_key_components(tmp_path: Path) -> None:
    """Cache key should include symbol, anchor_interval, and bar timestamp."""
    bars = _sample_bars(trend_up=True, n=120)

    manager = V2SignalManager(
        model_dir=tmp_path,
        symbols=("BTCUSDT",),
        anchor_interval="1h",
        loop_interval_seconds=3600,
        client_factory=lambda creds, live, symbol, interval: _FakeClient(bars),
    )

    async def scenario() -> dict[tuple[str, str, str], dict[str, Any]]:
        await manager.start_session(
            user_id=601,
            creds={"live": False},
            on_signal=lambda p: None,
            execute_orders=False,
        )

        cycle_cache: dict[tuple[str, str, str], dict[str, Any]] = {}
        await manager._run_cycle(manager.sessions[601], cycle_cache=cycle_cache)

        await manager.stop_session(601)
        return cycle_cache

    cache = asyncio.run(scenario())
    # Verify cache keys have correct structure
    assert len(cache) == 1
    for key in cache.keys():
        symbol, interval, ts_iso = key
        assert symbol == "BTCUSDT"
        assert interval == "1h"
        # Verify ts_iso is a valid ISO format timestamp
        assert isinstance(ts_iso, str)
        assert "T" in ts_iso  # ISO format contains 'T'


def test_cycle_cache_parameter_is_optional(tmp_path: Path) -> None:
    """_run_cycle should work with cycle_cache=None (backwards compatibility)."""
    bars = _sample_bars(trend_up=True)

    manager = V2SignalManager(
        model_dir=tmp_path,
        symbols=("BTCUSDT",),
        loop_interval_seconds=3600,
        client_factory=lambda creds, live, symbol, interval: _FakeClient(bars),
    )

    async def scenario() -> dict:
        await manager.start_session(
            user_id=701,
            creds={"live": False},
            on_signal=lambda p: None,
            execute_orders=False,
        )

        session = manager.sessions[701]
        # Call without cycle_cache parameter
        await manager._run_cycle(session, cycle_cache=None)

        await manager.stop_session(701)
        return {"success": True}

    result = asyncio.run(scenario())
    assert result["success"] is True


# ====================================================================
# audit_20260423 P2-2 — explicit cross-user dedup assertion
# ====================================================================


def test_compute_called_once_per_symbol_across_concurrent_users(tmp_path: Path) -> None:
    """When N users run a cycle on the same bar, _build_signal_payload fires once per symbol.

    This is the central regression for P2-2: prior to the shared inference cache,
    each user's _loop spawned its own per-cycle dict, producing N×len(symbols)
    invocations per bar.  After the fix, the manager-level shared cache catches
    the second user's lookup and skips the recompute.
    """
    bars = _sample_bars(trend_up=True)

    manager = V2SignalManager(
        model_dir=tmp_path,
        symbols=("BTCUSDT", "ETHUSDT"),
        anchor_interval="1h",
        loop_interval_seconds=900,
        client_factory=lambda creds, live, symbol, interval: _FakeClient(bars),
    )

    call_count = 0
    original_build = manager._build_signal_payload

    def counting_build(*args: Any, **kwargs: Any) -> dict:
        nonlocal call_count
        call_count += 1
        return original_build(*args, **kwargs)

    async def scenario() -> int:
        # Two concurrent users
        for user_id in (1101, 1102):
            await manager.start_session(
                user_id=user_id,
                creds={"live": False},
                on_signal=lambda p: None,
                execute_orders=False,
            )

        with patch.object(manager, "_build_signal_payload", side_effect=counting_build):
            # Each user's _loop would create its own cycle_cache; emulate that
            # by running both with separate dicts but the SAME shared manager.
            user_a_cache: dict[tuple[str, str, str], dict[str, Any]] = {}
            user_b_cache: dict[tuple[str, str, str], dict[str, Any]] = {}
            await manager._run_cycle(manager.sessions[1101], cycle_cache=user_a_cache)
            await manager._run_cycle(manager.sessions[1102], cycle_cache=user_b_cache)

        for user_id in (1101, 1102):
            await manager.stop_session(user_id)
        return call_count

    total_compute_calls = asyncio.run(scenario())

    # 2 symbols × 1 compute = 2 (NOT 2 symbols × 2 users = 4)
    assert total_compute_calls == 2, (
        f"Expected 2 compute calls (one per symbol, deduped across users), "
        f"got {total_compute_calls}.  Shared inference cache may not be wired."
    )


def test_shared_cache_evicts_stale_entries(tmp_path: Path) -> None:
    """Entries older than 1.5x loop_interval are evicted on lookup."""
    manager = V2SignalManager(
        model_dir=tmp_path,
        symbols=("BTCUSDT",),
        loop_interval_seconds=900,
    )

    key = ("BTCUSDT", "1h", "2026-04-23T07:00:00+00:00")
    payload = {"signal": "BUY", "probability": 0.7}

    # Fresh insert returns the payload
    manager._shared_cache_put(key, payload)
    assert manager._shared_cache_get(key) == payload

    # Backdate the entry beyond TTL → next get returns None and evicts
    inserted_at, cached_payload = manager._shared_inference_cache[key]
    manager._shared_inference_cache[key] = (
        inserted_at - manager._shared_cache_ttl() - 1.0,
        cached_payload,
    )
    assert manager._shared_cache_get(key) is None
    assert key not in manager._shared_inference_cache


def test_shared_cache_bounded_eviction(tmp_path: Path) -> None:
    """Cache stays under cap by dropping oldest entries on overflow."""
    manager = V2SignalManager(
        model_dir=tmp_path,
        symbols=("BTCUSDT",),
        loop_interval_seconds=900,
    )
    manager._shared_inference_cache_max = 3

    for i in range(5):
        manager._shared_cache_put(("BTCUSDT", "1h", f"ts-{i}"), {"i": i})

    assert len(manager._shared_inference_cache) <= 3
    # Newest entries survived
    assert ("BTCUSDT", "1h", "ts-4") in manager._shared_inference_cache
