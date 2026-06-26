"""Synthetic provider probes for market-data observability."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from time import perf_counter
from typing import Any, Iterable

import pandas as pd

from quant_v2.data.multi_symbol_dataset import MarketDataClient, fetch_symbol_dataset


def _normalize_symbols(symbols: Iterable[str], *, max_symbols: int | None = None) -> tuple[str, ...]:
    cleaned = [str(symbol).strip().upper() for symbol in symbols if str(symbol).strip()]
    if max_symbols is not None and max_symbols > 0:
        cleaned = cleaned[:max_symbols]
    return tuple(dict.fromkeys(cleaned))


def _frame_freshness_minutes(frame: pd.DataFrame, *, now: datetime) -> float | None:
    if frame.empty or not isinstance(frame.index, pd.Index):
        return None

    if isinstance(frame.index, pd.MultiIndex):
        timestamps = pd.DatetimeIndex(frame.index.get_level_values("timestamp"))
    else:
        timestamps = pd.DatetimeIndex(frame.index)

    if timestamps.empty:
        return None

    latest = timestamps.max()
    if latest.tzinfo is None:
        latest = latest.replace(tzinfo=timezone.utc)
    return max((now - latest).total_seconds() / 60.0, 0.0)


def probe_market_data_provider(
    symbols: Iterable[str],
    *,
    client: MarketDataClient | None,
    interval: str = "1h",
    lookback_hours: int = 6,
    stale_after_minutes: float = 90.0,
    max_symbols: int | None = 3,
) -> dict[str, Any]:
    """Probe the configured market-data client for freshness and availability."""

    probe_symbols = _normalize_symbols(symbols, max_symbols=max_symbols)
    now = datetime.now(timezone.utc)
    date_to = now
    date_from = date_to - timedelta(hours=max(int(lookback_hours), 1))

    latency_ms_by_symbol: dict[str, float] = {}
    freshness_minutes_by_symbol: dict[str, float] = {}
    row_count_by_symbol: dict[str, int] = {}
    stale_symbols: list[str] = []
    failed_symbols: dict[str, str] = {}
    rate_limit_snapshot: dict[str, Any] = {}

    for symbol in probe_symbols:
        started = perf_counter()
        try:
            frame = fetch_symbol_dataset(
                symbol,
                date_from=date_from,
                date_to=date_to,
                interval=interval,
                client=client,
            )
            latency_ms_by_symbol[symbol] = max((perf_counter() - started) * 1000.0, 0.0)
            row_count_by_symbol[symbol] = int(len(frame))
            freshness = _frame_freshness_minutes(frame, now=now)
            if freshness is None:
                failed_symbols[symbol] = "empty_dataset"
                continue
            freshness_minutes_by_symbol[symbol] = freshness
            if freshness > float(stale_after_minutes):
                stale_symbols.append(symbol)
        except Exception as exc:
            latency_ms_by_symbol[symbol] = max((perf_counter() - started) * 1000.0, 0.0)
            failed_symbols[symbol] = exc.__class__.__name__

    if client is not None:
        snapshot_getter = getattr(client, "get_rate_limit_snapshot", None)
        if callable(snapshot_getter):
            try:
                snapshot = snapshot_getter()
            except Exception as exc:
                rate_limit_snapshot = {
                    "provider_name": "binance_futures_rest",
                    "status": "unknown",
                    "error": exc.__class__.__name__,
                }
            else:
                if isinstance(snapshot, dict):
                    rate_limit_snapshot = dict(snapshot)

    probe_count = len(probe_symbols)
    failure_count = len(failed_symbols)
    stale_count = len(stale_symbols)
    latency_values = list(latency_ms_by_symbol.values())
    freshness_values = list(freshness_minutes_by_symbol.values())

    status = "healthy"
    circuit_breaker_triggered = False
    if failure_count > 0:
        status = "degraded"
        circuit_breaker_triggered = True
    elif stale_count > 0:
        status = "degraded"
        circuit_breaker_triggered = True

    rate_limit_status = str(rate_limit_snapshot.get("status", "unknown") or "unknown")
    if rate_limit_status == "degraded":
        status = "degraded"
    elif rate_limit_status == "warning" and status == "healthy":
        status = "warning"

    used_weight_1m = int(rate_limit_snapshot.get("used_weight_1m", 0) or 0)
    weight_limit_1m = int(rate_limit_snapshot.get("weight_limit_1m", 0) or 0)
    headroom_1m = int(rate_limit_snapshot.get("headroom_1m", 0) or 0)
    pressure_fraction = rate_limit_snapshot.get("pressure_fraction")

    return {
        "provider_name": "market_data",
        "status": status,
        "probe_count": probe_count,
        "failure_count": failure_count,
        "stale_count": stale_count,
        "circuit_breaker_triggered": circuit_breaker_triggered,
        "stale_after_minutes": float(stale_after_minutes),
        "latency_ms_max": max(latency_values) if latency_values else 0.0,
        "latency_ms_avg": (sum(latency_values) / len(latency_values)) if latency_values else 0.0,
        "freshness_minutes_max": max(freshness_values) if freshness_values else None,
        "freshness_minutes_avg": (sum(freshness_values) / len(freshness_values)) if freshness_values else None,
        "latency_ms_by_symbol": latency_ms_by_symbol,
        "freshness_minutes_by_symbol": freshness_minutes_by_symbol,
        "row_count_by_symbol": row_count_by_symbol,
        "stale_symbols": stale_symbols,
        "failed_symbols": failed_symbols,
        "rate_limit_snapshot": rate_limit_snapshot,
        "rate_limit_status": rate_limit_status,
        "rate_limit_used_weight_1m": used_weight_1m,
        "rate_limit_weight_limit_1m": weight_limit_1m,
        "rate_limit_headroom_1m": headroom_1m,
        "rate_limit_pressure_fraction": pressure_fraction,
        "probe_window": {
            "date_from": date_from.isoformat(),
            "date_to": date_to.isoformat(),
            "interval": interval,
        },
    }
