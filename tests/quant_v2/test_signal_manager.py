from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from quant_v2.contracts import ModelSourceDetails, StrategySignal
from quant_v2.monitoring.kill_switch import MonitoringSnapshot
from quant_v2.telebot.signal_manager import V2SignalManager


class _FakeClient:
    def __init__(self, bars: pd.DataFrame) -> None:
        self._bars = bars

    def fetch_historical(self, date_from, date_to, *, symbol: str, interval: str) -> pd.DataFrame:
        _ = (date_from, date_to, symbol, interval)
        return self._bars


class _OrderbookClient(_FakeClient):
    def __init__(self, bars: pd.DataFrame, books: dict[str, dict]) -> None:
        super().__init__(bars)
        self._books = books

    def get_orderbook(self, symbol: str, limit: int = 5) -> dict:
        _ = limit
        return self._books.get(symbol, {"bids": [], "asks": []})


def _sample_bars(*, trend_up: bool = True) -> pd.DataFrame:
    end = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    index = pd.date_range(end=end, periods=120, freq="h", tz="UTC")
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


def _close_history_with_return(total_return: float, *, periods: int = 40) -> pd.Series:
    end = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    index = pd.date_range(end=end, periods=periods, freq="h", tz="UTC")
    start = 100.0
    finish = start * (1.0 + total_return)
    return pd.Series([start + (finish - start) * i / (periods - 1) for i in range(periods)], index=index)


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

        await asyncio.wait_for(fired.wait(), timeout=60.0)
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


def test_v2_signal_manager_get_realtime_prices_prefers_orderbook_midpoint(tmp_path: Path) -> None:
    bars = _sample_bars(trend_up=True)
    books = {
        "BTCUSDT": {
            "bids": [["100.0", "3"]],
            "asks": [["102.0", "2"]],
        },
    }

    manager = V2SignalManager(
        model_dir=tmp_path,
        symbols=("BTCUSDT", "ETHUSDT"),
        loop_interval_seconds=3600,
        client_factory=lambda creds, live, symbol, interval: _OrderbookClient(bars, books),
    )

    async def scenario() -> dict[str, float]:
        assert await manager.start_session(
            user_id=16,
            creds={"live": False},
            on_signal=lambda payload: None,
            execute_orders=False,
        )
        prices = await manager.get_realtime_prices(16)
        await manager.stop_session(16)
        return prices

    refreshed = asyncio.run(scenario())
    assert refreshed.get("BTCUSDT") == pytest.approx(101.0)
    assert refreshed.get("ETHUSDT", 0.0) > 0.0


def test_market_short_guard_snapshot_inactive_for_mixed_market(tmp_path: Path) -> None:
    manager = V2SignalManager(model_dir=tmp_path, symbols=("BTCUSDT", "ETHUSDT", "SOLUSDT"))

    snapshot = manager._compute_market_risk_snapshot(
        {
            "BTCUSDT": _close_history_with_return(0.01),
            "ETHUSDT": _close_history_with_return(-0.01),
            "SOLUSDT": _close_history_with_return(0.02),
        }
    )

    assert snapshot is not None
    assert snapshot.broad_selloff is False
    assert snapshot.down_ratio == pytest.approx(1 / 3)


def test_market_short_guard_snapshot_active_when_universe_sells_off(tmp_path: Path) -> None:
    manager = V2SignalManager(model_dir=tmp_path, symbols=("BTCUSDT", "ETHUSDT", "SOLUSDT"))

    snapshot = manager._compute_market_risk_snapshot(
        {
            "BTCUSDT": _close_history_with_return(-0.03),
            "ETHUSDT": _close_history_with_return(-0.02),
            "SOLUSDT": _close_history_with_return(-0.01),
        }
    )

    assert snapshot is not None
    assert snapshot.broad_selloff is True
    assert snapshot.down_ratio == pytest.approx(1.0)
    assert snapshot.median_return <= -0.015
    assert snapshot.btc_return is not None
    assert snapshot.btc_return <= -0.02


def test_market_short_guard_env_thresholds_override_defaults(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("BOT_V2_MARKET_SHORT_GUARD_LOOKBACK_HOURS", "24")
    monkeypatch.setenv("BOT_V2_MARKET_SHORT_GUARD_DOWN_RATIO", "0.50")
    monkeypatch.setenv("BOT_V2_MARKET_SHORT_GUARD_MEDIAN_RETURN", "-0.005")
    monkeypatch.setenv("BOT_V2_MARKET_SHORT_GUARD_BTC_RETURN", "-0.010")
    monkeypatch.setenv("BOT_V2_MARKET_SHORT_GUARD_STRONG_CONFIDENCE", "0.80")
    manager = V2SignalManager(model_dir=tmp_path, symbols=("BTCUSDT", "ETHUSDT"))

    snapshot = manager._compute_market_risk_snapshot(
        {
            "BTCUSDT": _close_history_with_return(-0.011, periods=30),
            "ETHUSDT": _close_history_with_return(-0.02, periods=30),
        }
    )

    assert manager.market_short_guard_lookback_hours == 24
    assert manager.market_short_guard_strong_confidence == pytest.approx(0.80)
    assert snapshot is not None
    assert snapshot.broad_selloff is True
    assert snapshot.down_ratio_threshold == pytest.approx(0.50)


def test_regime_thresholds_keep_regime1_normal_and_regime2_stricter(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.delenv("BOT_V2_REGIME2_BUY_THRESHOLD", raising=False)
    monkeypatch.delenv("BOT_V2_REGIME2_SELL_THRESHOLD", raising=False)
    manager = V2SignalManager(model_dir=tmp_path, symbols=("BTCUSDT",))

    buy1, sell1, note1 = manager._resolve_regime_thresholds(regime=1, regime_risk=0.0)
    buy2, sell2, note2 = manager._resolve_regime_thresholds(regime=2, regime_risk=0.35)

    assert buy1 == pytest.approx(0.55)
    assert sell1 == pytest.approx(0.45)
    assert note1 == ""
    assert buy2 > buy1
    assert sell2 == pytest.approx(0.35)
    assert "regime2_conservative" in note2


def test_regime_thresholds_keep_regime3_and_4_existing_shape(tmp_path: Path) -> None:
    manager = V2SignalManager(model_dir=tmp_path, symbols=("BTCUSDT",))

    buy3, sell3, note3 = manager._resolve_regime_thresholds(regime=3, regime_risk=0.5)
    buy4, sell4, note4 = manager._resolve_regime_thresholds(regime=4, regime_risk=1.0)

    assert buy3 == pytest.approx(0.59)
    assert sell3 == pytest.approx(0.41)
    assert note3 == ""
    assert buy4 == pytest.approx(0.63)
    assert sell4 == pytest.approx(0.37)
    assert note4 == ""


def test_regime2_threshold_env_overrides_defaults(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("BOT_V2_REGIME2_BUY_THRESHOLD", "0.65")
    monkeypatch.setenv("BOT_V2_REGIME2_SELL_THRESHOLD", "0.30")
    manager = V2SignalManager(model_dir=tmp_path, symbols=("BTCUSDT",))

    buy2, sell2, note2 = manager._resolve_regime_thresholds(regime=2, regime_risk=0.35)

    assert buy2 == pytest.approx(0.65)
    assert sell2 == pytest.approx(0.30)
    assert "buy=0.65" in note2
    assert "sell=0.30" in note2


def test_native_v2_signal_preserves_regime_metadata(tmp_path: Path) -> None:
    manager = V2SignalManager(model_dir=tmp_path, symbols=("BTCUSDT",))
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "symbol": "ETHUSDT",
        "close_price": 100.0,
        "signal": "SELL",
        "probability": 0.30,
        "regime": 2,
        "reason": "regime2_conservative",
    }

    enriched = manager._attach_native_v2_fields(payload)

    assert enriched["v2_signal"].regime == 2
    assert enriched["v2_signal"].signal == "SELL"


def test_native_v2_signal_preserves_model_source_metadata(tmp_path: Path) -> None:
    manager = V2SignalManager(model_dir=tmp_path, symbols=("BTCUSDT",))
    details = ModelSourceDetails(
        lgbm_probability=0.28,
        chronos_probability=0.62,
        final_probability=0.399,
        lgbm_direction="SELL",
        chronos_direction="BUY",
        agreement=False,
        chronos_enabled=True,
    )
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "symbol": "ETHUSDT",
        "close_price": 100.0,
        "signal": "SELL",
        "probability": 0.399,
        "reason": "source disagreement",
        "_model_sources": details,
    }

    enriched = manager._attach_native_v2_fields(payload)

    assert enriched["v2_signal"].model_sources is details
    assert enriched["v2_signal"].model_sources.agreement is False


def test_predict_with_uncertainty_caches_full_ensemble_source_details(tmp_path: Path) -> None:
    manager = V2SignalManager(model_dir=tmp_path, symbols=("BTCUSDT",))
    details = ModelSourceDetails(
        lgbm_probability=0.70,
        chronos_probability=0.30,
        final_probability=0.56,
        lgbm_direction="BUY",
        chronos_direction="SELL",
        agreement=False,
        chronos_enabled=True,
    )

    class FakeFullEnsemble:
        def predict_with_details(self, feature_row, close_series, prediction_length):  # noqa: ANN001
            _ = (feature_row, close_series, prediction_length)
            return 0.56, 0.20, 0.0, details

    manager.full_ensemble = FakeFullEnsemble()  # type: ignore[assignment]
    feature_row = pd.DataFrame({"feature": [1.0]})
    close = _close_history_with_return(0.01)

    proba, uncertainty = manager._predict_with_uncertainty(feature_row, close)

    assert proba == pytest.approx(0.56)
    assert uncertainty == pytest.approx(0.20)
    assert manager._last_model_agreement == pytest.approx(0.0)
    assert manager._last_model_sources is details


def test_format_model_source_details_contains_probabilities(tmp_path: Path) -> None:
    manager = V2SignalManager(model_dir=tmp_path, symbols=("BTCUSDT",))
    details = ModelSourceDetails(
        lgbm_probability=0.28,
        chronos_probability=0.62,
        final_probability=0.399,
        lgbm_direction="SELL",
        chronos_direction="BUY",
        agreement=False,
        chronos_enabled=True,
    )

    text = manager._format_model_source_details(details)

    assert "lgbm=0.280" in text
    assert "chronos=0.620" in text
    assert "final=0.399" in text
    assert "agreement=disagree" in text


def test_market_short_guard_blocks_weak_sell_payload(tmp_path: Path) -> None:
    manager = V2SignalManager(model_dir=tmp_path, symbols=("BTCUSDT",))
    snapshot = manager._compute_market_risk_snapshot(
        {
            "BTCUSDT": _close_history_with_return(-0.03),
            "ETHUSDT": _close_history_with_return(-0.02),
            "SOLUSDT": _close_history_with_return(-0.02),
        }
    )
    assert snapshot is not None and snapshot.broad_selloff
    session = SimpleNamespace(last_known_positions={})
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "symbol": "ETHUSDT",
        "close_price": 100.0,
        "signal": "SELL",
        "probability": 0.32,
        "reason": "weak sell",
        "risk_status": {"can_trade": True},
    }

    manager._apply_market_short_guard(session, payload, snapshot)

    assert payload["signal"] == "HOLD"
    assert payload["market_short_guard"] == "blocked_weak_sell"
    assert payload["market_risk"]["broad_selloff"] is True
    assert isinstance(payload["v2_signal"], StrategySignal)
    assert payload["v2_signal"].signal == "HOLD"
    assert payload["v2_signal"].market_risk is snapshot


def test_market_short_guard_allows_strong_sell_payload(tmp_path: Path) -> None:
    manager = V2SignalManager(model_dir=tmp_path, symbols=("BTCUSDT",))
    snapshot = manager._compute_market_risk_snapshot(
        {
            "BTCUSDT": _close_history_with_return(-0.03),
            "ETHUSDT": _close_history_with_return(-0.02),
            "SOLUSDT": _close_history_with_return(-0.02),
        }
    )
    assert snapshot is not None and snapshot.broad_selloff
    session = SimpleNamespace(last_known_positions={})
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "symbol": "ETHUSDT",
        "close_price": 100.0,
        "signal": "SELL",
        "probability": 0.20,
        "reason": "strong sell",
        "risk_status": {"can_trade": True},
    }

    manager._apply_market_short_guard(session, payload, snapshot)

    assert payload["signal"] == "SELL"
    assert payload["market_short_guard"] == "strong_sell_allowed"
    assert payload["v2_signal"].signal == "SELL"
    assert payload["v2_signal"].confidence == pytest.approx(0.80)


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
        if session.task:
            session.task.cancel()
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


def test_default_loop_interval_is_900(tmp_path: Path, monkeypatch) -> None:
    """Default loop interval should be 900s (15 min) when env is unset."""
    monkeypatch.delenv("BOT_V2_SIGNAL_LOOP_SECONDS", raising=False)
    manager = V2SignalManager(
        model_dir=tmp_path,
        symbols=("BTCUSDT",),
        loop_interval_seconds=None,
    )
    assert manager.loop_interval_seconds == 900


def test_env_override_still_works(tmp_path: Path, monkeypatch) -> None:
    """BOT_V2_SIGNAL_LOOP_SECONDS env var should override default."""
    monkeypatch.setenv("BOT_V2_SIGNAL_LOOP_SECONDS", "60")
    manager = V2SignalManager(
        model_dir=tmp_path,
        symbols=("BTCUSDT",),
        loop_interval_seconds=None,
    )
    assert manager.loop_interval_seconds == 60


def test_explicit_kwarg_beats_env(tmp_path: Path, monkeypatch) -> None:
    """Explicit loop_interval_seconds kwarg should beat env var."""
    monkeypatch.setenv("BOT_V2_SIGNAL_LOOP_SECONDS", "60")
    manager = V2SignalManager(
        model_dir=tmp_path,
        symbols=("BTCUSDT",),
        loop_interval_seconds=120,
    )
    assert manager.loop_interval_seconds == 120
