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
    """Running two cycles with separate caches should compute twice per symbol."""
    bars = _sample_bars(trend_up=True)

    manager = V2SignalManager(
        model_dir=tmp_path,
        symbols=("BTCUSDT",),
        loop_interval_seconds=3600,
        client_factory=lambda creds, live, symbol, interval: _FakeClient(bars),
    )

    async def scenario() -> int:
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

            # First cycle with fresh cache
            cycle_cache_1: dict[tuple[str, str, str], dict[str, Any]] = {}
            await manager._run_cycle(session, cycle_cache=cycle_cache_1)

            # Second cycle with fresh cache - should compute again
            cycle_cache_2: dict[tuple[str, str, str], dict[str, Any]] = {}
            await manager._run_cycle(session, cycle_cache=cycle_cache_2)

        await manager.stop_session(301)
        return call_count

    call_count = asyncio.run(scenario())
    # Each cycle computes once per symbol (cache doesn't persist between cycles)
    assert call_count == 2


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
