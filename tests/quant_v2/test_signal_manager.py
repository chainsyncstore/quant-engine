from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from quant_v2.contracts import StrategySignal
from quant_v2.monitoring.kill_switch import MonitoringSnapshot
from quant_v2.telebot.signal_manager import V2SignalManager


class _FakeClient:
    def __init__(self, bars: pd.DataFrame) -> None:
        self._bars = bars

    def fetch_historical(self, date_from, date_to, *, symbol: str, interval: str) -> pd.DataFrame:
        _ = (date_from, date_to, symbol, interval)
        return self._bars


def _sample_bars(*, trend_up: bool = True) -> pd.DataFrame:
    end = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    index = pd.date_range(end=end, periods=120, freq="h", tz="UTC")
    if trend_up:
        closes = [float(10_000 + i * 20) for i in range(len(index))]
    else:
        closes = [float(10_000 - i * 20) for i in range(len(index))]
    return pd.DataFrame({"close": closes}, index=index)


def test_v2_signal_manager_emits_signal_and_tracks_lifecycle(tmp_path: Path) -> None:
    bars = _sample_bars(trend_up=True)

    manager = V2SignalManager(
        model_dir=tmp_path,
        symbols=("BTCUSDT",),
        loop_interval_seconds=1,
        client_factory=lambda creds, live, symbol, interval: _FakeClient(bars),
    )

    async def scenario() -> dict:
        emitted: list[dict] = []
        fired = asyncio.Event()

        async def on_signal(payload: dict) -> None:
            emitted.append(payload)
            fired.set()

        started = await manager.start_session(
            user_id=11,
            creds={"live": False},
            on_signal=on_signal,
            execute_orders=False,
        )
        assert started is True
        assert manager.is_running(11) is True
        assert manager.get_active_count() == 1
        assert manager.get_session_mode(11) == "paper"

        await asyncio.wait_for(fired.wait(), timeout=2.0)
        assert emitted

        stats = manager.get_signal_stats(11)
        assert stats["total_signals"] >= 1
        assert stats["symbols"] >= 1

        recent = manager.get_recent_signals(11, limit=5)
        assert recent
        assert recent[-1]["symbol"] == "BTCUSDT"

        stopped = await manager.stop_session(11)
        assert stopped is True
        assert manager.is_running(11) is False
        assert manager.get_active_count() == 0
        return emitted[0]

    first_signal = asyncio.run(scenario())
    assert first_signal["symbol"] == "BTCUSDT"
    assert first_signal["signal"] in {"BUY", "SELL", "HOLD", "DRIFT_ALERT"}
    assert first_signal["signal"] != "ENGINE_CRASH"
    assert float(first_signal["close_price"]) > 0.0
    assert isinstance(first_signal["v2_signal"], StrategySignal)
    assert first_signal["v2_signal"].symbol == "BTCUSDT"
    assert isinstance(first_signal["v2_monitoring_snapshot"], MonitoringSnapshot)
    assert first_signal["v2_prices"] == {"BTCUSDT": float(first_signal["close_price"])}


def test_v2_signal_manager_live_requires_credentials(tmp_path: Path) -> None:
    bars = _sample_bars(trend_up=True)
    manager = V2SignalManager(
        model_dir=tmp_path,
        symbols=("BTCUSDT",),
        client_factory=lambda creds, live, symbol, interval: _FakeClient(bars),
    )

    async def scenario() -> None:
        with pytest.raises(RuntimeError):
            await manager.start_session(
                user_id=12,
                creds={"live": True, "binance_api_key": "", "binance_api_secret": ""},
                on_signal=lambda payload: None,
                execute_orders=False,
            )

    asyncio.run(scenario())


def test_v2_signal_manager_resolves_joblib_active_model_path(tmp_path: Path) -> None:
    bars = _sample_bars(trend_up=True)
    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir(parents=True)
    legacy_joblib = artifact_dir / "model_4m.joblib"
    legacy_joblib.write_bytes(b"placeholder")

    manager = V2SignalManager(
        model_dir=tmp_path,
        symbols=("BTCUSDT",),
        horizon_bars=4,
        client_factory=lambda creds, live, symbol, interval: _FakeClient(bars),
    )

    resolved = manager._resolve_active_model_path(artifact_dir)
    assert resolved == legacy_joblib


def test_v2_signal_manager_dedupes_stale_bar_timestamps(tmp_path: Path) -> None:
    bars = _sample_bars(trend_up=False)

    manager = V2SignalManager(
        model_dir=tmp_path,
        symbols=("BTCUSDT",),
        client_factory=lambda creds, live, symbol, interval: _FakeClient(bars),
    )

    async def scenario() -> int:
        emitted: list[dict] = []

        async def on_signal(payload: dict) -> None:
            emitted.append(payload)

        assert await manager.start_session(
            user_id=13,
            creds={"live": False},
            on_signal=on_signal,
            execute_orders=False,
        )

        session = manager.sessions[13]
        await manager._run_cycle(session)
        await manager._run_cycle(session)

        await manager.stop_session(13)
        return len(emitted)

    count = asyncio.run(scenario())
    assert count == 1


def test_v2_signal_manager_reset_session_state_clears_paper_runtime_state(tmp_path: Path) -> None:
    bars = _sample_bars(trend_up=True)
    manager = V2SignalManager(
        model_dir=tmp_path,
        symbols=("BTCUSDT",),
        loop_interval_seconds=3600,
        client_factory=lambda creds, live, symbol, interval: _FakeClient(bars),
    )

    async def scenario() -> None:
        assert await manager.start_session(
            user_id=14,
            creds={"live": False},
            on_signal=lambda payload: None,
            execute_orders=False,
        )

        session = manager.sessions[14]
        session.last_bar_timestamp["BTCUSDT"] = pd.Timestamp("2026-01-01T00:00:00Z")
        session.signal_log.append({"symbol": "BTCUSDT", "signal": "BUY", "close_price": 12345.0})

        assert manager.reset_session_state(14) is True
        assert manager.is_running(14) is True
        assert manager.get_signal_stats(14)["total_signals"] == 0
        assert manager.get_recent_signals(14) == ()
        assert session.last_bar_timestamp == {}

        await manager.stop_session(14)

    asyncio.run(scenario())


def test_v2_signal_manager_reset_session_state_rejects_live_and_missing(tmp_path: Path) -> None:
    bars = _sample_bars(trend_up=True)
    manager = V2SignalManager(
        model_dir=tmp_path,
        symbols=("BTCUSDT",),
        loop_interval_seconds=3600,
        client_factory=lambda creds, live, symbol, interval: _FakeClient(bars),
    )

    async def scenario() -> None:
        assert manager.reset_session_state(999) is False

        assert await manager.start_session(
            user_id=15,
            creds={"live": True, "binance_api_key": "k", "binance_api_secret": "s"},
            on_signal=lambda payload: None,
            execute_orders=False,
        )
        assert manager.get_session_mode(15) == "live"
        assert manager.reset_session_state(15) is False
        await manager.stop_session(15)

    asyncio.run(scenario())
