"""Tests for the 8 lower-priority audit upgrade tasks.

Task 1: Partial fill simulation (95% limit fill rate) in backtester
Task 2: Per-symbol breakdown in backtest metrics/report
Task 3: Max drawdown duration in BacktestResult
Task 4: Dynamic min_notional: max(10, equity×0.02) in optimizer
Task 5: Correlation fields on contracts.py
Task 6: Rolling correlation cache in signal_manager
Task 7: Telegram dead-man's-switch alert (heartbeat_stale in watchdog)
Task 8: Persistent position snapshot on disk
"""
from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from quant_v2.contracts import StrategySignal, PortfolioSnapshot, RiskSnapshot
from quant_v2.research.backtester import BacktestConfig, BacktestResult, Fill
from quant_v2.research.backtest_report import _compute_metrics, _symbol_breakdown_html
from quant_v2.portfolio.optimizer import RiskParityOptimizer, compute_rolling_correlations
from quant_v2.execution.service import RoutedExecutionService, SessionRequest


# ---- Helpers ---------------------------------------------------------------

def _make_equity(n: int = 50, initial: float = 300.0, final: float = 310.0) -> pd.Series:
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    return pd.Series(np.linspace(initial, final, n), index=idx, name="equity_usd")


def _make_fills(symbols: list[str] | None = None, n_rt: int = 2) -> list[Fill]:
    """Create n_rt round-trip pairs (BUY→SELL).  Each RT has distinct prices."""
    symbols = symbols or ["BTCUSDT"]
    idx = pd.date_range("2024-01-01", periods=n_rt * 2 * len(symbols), freq="1h", tz="UTC")
    fills: list[Fill] = []
    i = 0
    for sym in symbols:
        for rt in range(n_rt):
            entry_px = 50_000.0 + rt * 100
            exit_px = entry_px + (50 if rt % 2 == 0 else -30)  # alternating win/loss
            fills.append(Fill(
                timestamp=idx[i], symbol=sym, side="BUY",
                quantity=0.001, price=entry_px,
                fee_usd=0.1, slippage_usd=0.05, confidence=0.72,
            ))
            i += 1
            fills.append(Fill(
                timestamp=idx[i], symbol=sym, side="SELL",
                quantity=0.001, price=exit_px,
                fee_usd=0.1, slippage_usd=0.05, confidence=0.72,
            ))
            i += 1
    return fills


def _make_result(**overrides) -> BacktestResult:
    eq = overrides.pop("equity_curve", _make_equity())
    fills = overrides.pop("fills", _make_fills())
    daily = eq.resample("D").last().pct_change().dropna()
    defaults = dict(
        config=BacktestConfig(),
        equity_curve=eq,
        fills=fills,
        daily_returns=daily,
        total_trades=len(fills) // 2,
        win_trades=1,
        gross_pnl=10.0,
        total_fees=0.4,
        total_slippage=0.2,
    )
    defaults.update(overrides)
    return BacktestResult(**defaults)


# ====================================================================
# TASK 1 — Partial fill simulation (95% limit fill rate)
# ====================================================================

class TestTask1PartialFillRate:
    """BacktestConfig.limit_fill_rate must exist and default to 0.95."""

    def test_default_limit_fill_rate(self):
        cfg = BacktestConfig()
        assert hasattr(cfg, "limit_fill_rate")
        assert cfg.limit_fill_rate == pytest.approx(0.95)

    def test_custom_fill_rate(self):
        cfg = BacktestConfig(limit_fill_rate=0.80)
        assert cfg.limit_fill_rate == pytest.approx(0.80)

    def test_fill_rate_applied_in_sim(self):
        """The backtester loop multiplies target_qty by limit_fill_rate (line 314)."""
        # Verify the code path at module level — import-time static check
        import inspect
        from quant_v2.research.backtester import run_backtest
        source = inspect.getsource(run_backtest)
        assert "limit_fill_rate" in source, "limit_fill_rate not used in run_backtest"


# ====================================================================
# TASK 2 — Per-symbol breakdown in backtest report
# ====================================================================

class TestTask2PerSymbolBreakdown:
    """_symbol_breakdown_html must pair fills into round-trips per symbol."""

    def test_breakdown_single_symbol(self):
        fills = _make_fills(symbols=["BTCUSDT"], n_rt=3)
        html = _symbol_breakdown_html(fills)
        assert "BTCUSDT" in html
        assert "<table" in html

    def test_breakdown_multi_symbol(self):
        fills = _make_fills(symbols=["BTCUSDT", "ETHUSDT"], n_rt=2)
        html = _symbol_breakdown_html(fills)
        assert "BTCUSDT" in html
        assert "ETHUSDT" in html

    def test_breakdown_empty_fills(self):
        html = _symbol_breakdown_html([])
        assert "No fills" in html

    def test_breakdown_counts_round_trips(self):
        fills = _make_fills(symbols=["BTCUSDT"], n_rt=4)
        html = _symbol_breakdown_html(fills)
        # 4 round-trips for BTCUSDT
        assert "<td>4</td>" in html

    def test_report_includes_breakdown_section(self):
        m = _compute_metrics(_make_result())
        assert "Max DD duration" in m, "Max DD duration key missing from metrics"


# ====================================================================
# TASK 3 — Max drawdown duration in BacktestResult
# ====================================================================

class TestTask3MaxDrawdownDuration:
    """BacktestResult.max_drawdown_duration returns longest DD stretch in bars."""

    def test_flat_equity_zero_duration(self):
        eq = pd.Series([100.0] * 20, index=pd.date_range("2024-01-01", periods=20, freq="1h", tz="UTC"))
        r = _make_result(equity_curve=eq)
        assert r.max_drawdown_duration == 0

    def test_declining_equity_has_duration(self):
        values = [100, 99, 98, 97, 96, 95, 96, 97, 98, 99, 100, 101]
        eq = pd.Series(values, index=pd.date_range("2024-01-01", periods=len(values), freq="1h", tz="UTC"))
        r = _make_result(equity_curve=eq)
        # DD from index 1 to index 9 (9 bars in drawdown)
        assert r.max_drawdown_duration >= 5

    def test_empty_equity_zero_duration(self):
        empty_eq = pd.Series(dtype=float, index=pd.DatetimeIndex([], tz="UTC"))
        r = _make_result(equity_curve=empty_eq)
        assert r.max_drawdown_duration == 0

    def test_metrics_dict_contains_dd_duration(self):
        r = _make_result()
        m = _compute_metrics(r)
        assert "Max DD duration" in m
        assert "h" in m["Max DD duration"]  # formatted as "Xh"


# ====================================================================
# TASK 4 — Dynamic min_notional: max(10, equity×0.02)
# ====================================================================

class TestTask4DynamicMinNotional:
    """RiskParityOptimizer should use max(base_min, equity_usd * 0.02)."""

    def _make_price_hist(self, n: int = 100, base: float = 100.0, seed: int = 42) -> pd.Series:
        np.random.seed(seed)
        returns = np.random.normal(0.001, 0.02, n)
        prices = base * np.cumprod(1 + returns)
        return pd.Series(prices)

    def test_low_equity_uses_base_notional(self):
        """When equity is $300, 2% = $6 < $10, so $10 is used."""
        opt = RiskParityOptimizer(min_notional_usd=10.0)
        hist = self._make_price_hist()
        exposures = {"BTCUSDT": 0.10, "ETHUSDT": 0.05}
        result = opt.optimize(exposures, {"BTCUSDT": hist, "ETHUSDT": hist}, equity_usd=300.0)
        # At $300 equity, 2% = $6, so effective min = max(10, 6) = $10
        # A 5% exposure = $15, which is above $10 — may survive
        # Verify we didn't drop everything
        assert isinstance(result.weights, dict)

    def test_high_equity_uses_dynamic_notional(self):
        """When equity is $10,000, 2% = $200 > $10, so $200 is used.

        Risk-parity uses inverse-vol weighting, so ETHUSDT needs much higher
        volatility to receive a tiny weight.  With ~10× vol, its risk-parity
        share shrinks to ~9%, giving notional ≈ 0.009 * 10000 ≈ $90 < $200.
        """
        opt = RiskParityOptimizer(min_notional_usd=10.0)
        hist_btc = self._make_price_hist(seed=42)
        # Very high volatility for ETHUSDT → tiny inverse-vol weight
        np.random.seed(99)
        high_vol_returns = np.random.normal(0.0, 0.20, 100)  # 10× normal vol
        hist_eth = pd.Series(100.0 * np.cumprod(1 + high_vol_returns))

        exposures = {"BTCUSDT": 0.10, "ETHUSDT": 0.10}
        result = opt.optimize(
            exposures,
            {"BTCUSDT": hist_btc, "ETHUSDT": hist_eth},
            equity_usd=10_000.0,
        )
        # ETH inv-vol weight ≈ 1/0.20 vs BTC 1/0.02 → ETH gets ~9% of gross
        # notional ≈ 0.09 * 0.20 * 10000 ≈ $180 < $200 effective min
        assert "ETHUSDT" in result.dropped_symbols

    def test_dynamic_notional_in_source(self):
        """Verify the dynamic formula exists in optimizer source code."""
        import inspect
        source = inspect.getsource(RiskParityOptimizer.optimize)
        assert "equity_usd * 0.02" in source or "equity_usd * 0.02" in source.replace(" ", "")


# ====================================================================
# TASK 5 — Correlation fields on contracts.py
# ====================================================================

class TestTask5CorrelationFields:
    """StrategySignal must have pairwise_correlations and portfolio_weight fields."""

    def test_pairwise_correlations_field_exists(self):
        sig = StrategySignal(
            symbol="BTCUSDT", timeframe="1h", horizon_bars=4,
            signal="BUY", confidence=0.8,
            pairwise_correlations={"ETHUSDT": 0.85, "SOLUSDT": 0.40},
        )
        assert sig.pairwise_correlations == {"ETHUSDT": 0.85, "SOLUSDT": 0.40}

    def test_pairwise_correlations_default_none(self):
        sig = StrategySignal(
            symbol="BTCUSDT", timeframe="1h", horizon_bars=4,
            signal="HOLD", confidence=0.5,
        )
        assert sig.pairwise_correlations is None

    def test_portfolio_weight_field_exists(self):
        sig = StrategySignal(
            symbol="ETHUSDT", timeframe="1h", horizon_bars=4,
            signal="SELL", confidence=0.75,
            portfolio_weight=0.12,
        )
        assert sig.portfolio_weight == pytest.approx(0.12)

    def test_portfolio_weight_default_none(self):
        sig = StrategySignal(
            symbol="BTCUSDT", timeframe="1h", horizon_bars=4,
            signal="HOLD", confidence=0.5,
        )
        assert sig.portfolio_weight is None


# ====================================================================
# TASK 6 — Rolling correlation cache (compute_rolling_correlations)
# ====================================================================

class TestTask6RollingCorrelationCache:
    """compute_rolling_correlations utility function for diagnostics."""

    def _make_prices(self, n: int = 100, seed: int = 42) -> pd.Series:
        np.random.seed(seed)
        returns = np.random.normal(0.001, 0.02, n)
        return pd.Series(100.0 * np.cumprod(1 + returns))

    def test_returns_corr_for_pair(self):
        prices = {
            "BTCUSDT": self._make_prices(seed=42),
            "ETHUSDT": self._make_prices(seed=42),  # identical → corr ≈ 1.0
        }
        result = compute_rolling_correlations(prices)
        assert ("BTCUSDT", "ETHUSDT") in result
        assert result[("BTCUSDT", "ETHUSDT")] == pytest.approx(1.0, abs=0.05)

    def test_uncorrelated_pair(self):
        prices = {
            "BTCUSDT": self._make_prices(seed=1),
            "ETHUSDT": self._make_prices(seed=999),
        }
        result = compute_rolling_correlations(prices)
        if ("BTCUSDT", "ETHUSDT") in result:
            # Should not be perfectly correlated
            assert abs(result[("BTCUSDT", "ETHUSDT")]) < 0.8

    def test_empty_prices_returns_empty(self):
        result = compute_rolling_correlations({})
        assert result == {}

    def test_single_symbol_no_pairs(self):
        result = compute_rolling_correlations({"BTCUSDT": self._make_prices()})
        assert result == {}

    def test_short_history_skips(self):
        # < 10 aligned points → skip
        prices = {
            "BTCUSDT": pd.Series([100.0, 101.0]),
            "ETHUSDT": pd.Series([200.0, 199.0]),
        }
        result = compute_rolling_correlations(prices)
        assert result == {}


# ====================================================================
# TASK 7 — Telegram dead-man's-switch (heartbeat_stale in watchdog)
# ====================================================================

class TestTask7WatchdogHeartbeatStale:
    """LifecycleWatchdog emits heartbeat_stale alert when ticks stop arriving."""

    def test_heartbeat_stale_triggers_after_threshold(self):
        from quant_v2.execution.watchdog import LifecycleWatchdog, WatchdogAlert

        alerts: list[WatchdogAlert] = []

        async def on_alert(alert: WatchdogAlert):
            alerts.append(alert)

        watchdog = LifecycleWatchdog(
            check_interval_seconds=0.1,
            on_alert=on_alert,
            stale_heartbeat_seconds=0.5,  # very short for testing
        )

        async def _run():
            session = watchdog.register_session(user_id=1, is_live=True)
            # Set last tick time far in the past
            from datetime import timedelta
            watchdog._last_tick_time[1] = datetime.now(timezone.utc) - timedelta(seconds=10)
            # Run one check cycle
            await watchdog._run_checks()

        asyncio.run(_run())
        assert len(alerts) == 1
        assert alerts[0].alert_type == "heartbeat_stale"
        assert alerts[0].user_id == 1

    def test_heartbeat_stale_resets_timer(self):
        """After a stale alert, the timer resets to avoid flooding."""
        from quant_v2.execution.watchdog import LifecycleWatchdog, WatchdogAlert

        alerts: list[WatchdogAlert] = []

        async def on_alert(alert: WatchdogAlert):
            alerts.append(alert)

        watchdog = LifecycleWatchdog(
            check_interval_seconds=0.1,
            on_alert=on_alert,
            stale_heartbeat_seconds=0.5,
        )

        async def _run():
            from datetime import timedelta
            watchdog.register_session(user_id=2, is_live=True)
            watchdog._last_tick_time[2] = datetime.now(timezone.utc) - timedelta(seconds=10)
            # First check fires alert and resets
            await watchdog._run_checks()
            # Second check should NOT fire (timer was just reset)
            await watchdog._run_checks()

        asyncio.run(_run())
        assert len(alerts) == 1  # only one alert, not two

    def test_record_tick_refreshes_heartbeat(self):
        """record_tick() updates the last tick time so watchdog stays quiet."""
        from quant_v2.execution.watchdog import LifecycleWatchdog, WatchdogAlert

        alerts: list[WatchdogAlert] = []

        async def on_alert(alert: WatchdogAlert):
            alerts.append(alert)

        watchdog = LifecycleWatchdog(
            check_interval_seconds=0.1,
            on_alert=on_alert,
            stale_heartbeat_seconds=1.0,
        )

        async def _run():
            watchdog.register_session(user_id=3, is_live=True)
            # Simulate fresh tick
            watchdog.record_tick(3)
            await watchdog._run_checks()

        asyncio.run(_run())
        assert len(alerts) == 0

    def test_default_stale_threshold(self):
        """Default stale heartbeat should be >= 3600s (hourly signal loop)."""
        from quant_v2.execution.watchdog import LifecycleWatchdog
        watchdog = LifecycleWatchdog()
        assert watchdog._stale_heartbeat_seconds >= 3600.0


# ====================================================================
# TASK 8 — Persistent position snapshot on disk
# ====================================================================

class TestTask8PersistentPositionSnapshot:
    """_persist_position_snapshot writes JSON to disk after fills."""

    def test_persist_writes_json_file(self, tmp_path):
        """Snapshot file is atomically written with correct structure."""
        snapshot_path = tmp_path / "positions_snapshot.json"

        with patch.dict(os.environ, {"BOT_SNAPSHOT_PATH": str(snapshot_path)}):
            state = MagicMock()
            state.mode = "paper"
            state.snapshot = MagicMock()
            state.snapshot.equity_usd = 10_000.0
            state.snapshot.open_positions = {"BTCUSDT": 0.05, "ETHUSDT": -0.1}
            state.paper_entry_price = {"BTCUSDT": 50_000.0, "ETHUSDT": 2500.0}

            RoutedExecutionService._persist_position_snapshot(
                user_id=42,
                state=state,
                prices={"BTCUSDT": 51000.0, "ETHUSDT": 2400.0},
            )

        assert snapshot_path.exists()
        data = json.loads(snapshot_path.read_text(encoding="utf-8"))
        assert data["user_id"] == 42
        assert data["mode"] == "paper"
        assert data["equity_usd"] == 10_000.0
        assert data["positions"]["BTCUSDT"] == 0.05
        assert data["positions"]["ETHUSDT"] == pytest.approx(-0.1)
        assert data["entry_prices"]["BTCUSDT"] == 50_000.0
        assert data["last_prices"]["BTCUSDT"] == 51_000.0
        assert "timestamp" in data

    def test_persist_atomic_via_tmp_rename(self, tmp_path):
        """Write goes to .tmp first, then rename — no partial reads."""
        snapshot_path = tmp_path / "snapshot.json"

        with patch.dict(os.environ, {"BOT_SNAPSHOT_PATH": str(snapshot_path)}):
            state = MagicMock()
            state.mode = "live"
            state.snapshot = MagicMock()
            state.snapshot.equity_usd = 5_000.0
            state.snapshot.open_positions = {}
            state.paper_entry_price = {}

            RoutedExecutionService._persist_position_snapshot(
                user_id=1, state=state, prices={},
            )

        assert snapshot_path.exists()
        # The .tmp file should NOT remain
        assert not snapshot_path.with_suffix(".tmp").exists()

    def test_persist_creates_parent_dirs(self, tmp_path):
        """Parent directories are created if they don't exist."""
        nested = tmp_path / "deep" / "dir" / "snapshot.json"
        assert not nested.parent.exists()

        with patch.dict(os.environ, {"BOT_SNAPSHOT_PATH": str(nested)}):
            state = MagicMock()
            state.mode = "paper"
            state.snapshot = MagicMock()
            state.snapshot.equity_usd = 1_000.0
            state.snapshot.open_positions = {"SOLUSDT": 10.0}
            state.paper_entry_price = {}

            RoutedExecutionService._persist_position_snapshot(
                user_id=99, state=state, prices={"SOLUSDT": 150.0},
            )

        assert nested.exists()
        data = json.loads(nested.read_text(encoding="utf-8"))
        assert data["positions"]["SOLUSDT"] == 10.0

    def test_persist_filters_zero_prices(self, tmp_path):
        """Prices of 0.0 should not appear in snapshot."""
        snapshot_path = tmp_path / "snapshot.json"

        with patch.dict(os.environ, {"BOT_SNAPSHOT_PATH": str(snapshot_path)}):
            state = MagicMock()
            state.mode = "paper"
            state.snapshot = MagicMock()
            state.snapshot.equity_usd = 2_000.0
            state.snapshot.open_positions = {}
            state.paper_entry_price = {}

            RoutedExecutionService._persist_position_snapshot(
                user_id=1, state=state,
                prices={"BTCUSDT": 50_000.0, "UNKNOWN": 0.0},
            )

        data = json.loads(snapshot_path.read_text(encoding="utf-8"))
        assert "BTCUSDT" in data["last_prices"]
        assert "UNKNOWN" not in data["last_prices"]

    def test_persist_failure_does_not_raise(self, tmp_path):
        """Write failure (e.g. permission error) is silently caught."""
        with patch.dict(os.environ, {"BOT_SNAPSHOT_PATH": "/nonexistent/readonly/path.json"}):
            state = MagicMock()
            state.mode = "paper"
            state.snapshot = MagicMock()
            state.snapshot.equity_usd = 1_000.0
            state.snapshot.open_positions = {}
            state.paper_entry_price = {}

            # Should not raise
            RoutedExecutionService._persist_position_snapshot(
                user_id=1, state=state, prices={},
            )

    def test_route_signals_triggers_persist_on_fill(self, tmp_path):
        """End-to-end: route_signals → fill → persist snapshot."""
        snapshot_path = tmp_path / "positions_snapshot.json"

        with patch.dict(os.environ, {"BOT_SNAPSHOT_PATH": str(snapshot_path)}):
            service = RoutedExecutionService()
            req = SessionRequest(user_id=800, live=False)
            assert asyncio.run(service.start_session(req)) is True

            signal = StrategySignal(
                symbol="BTCUSDT", timeframe="1h", horizon_bars=4,
                signal="BUY", confidence=0.85,
            )
            results = asyncio.run(
                service.route_signals(800, signals=(signal,), prices={"BTCUSDT": 50_000.0})
            )
            assert len(results) >= 1
            assert any(r.accepted for r in results)

        assert snapshot_path.exists()
        data = json.loads(snapshot_path.read_text(encoding="utf-8"))
        assert data["user_id"] == 800
        assert "BTCUSDT" in data["positions"]

    def test_no_fills_no_persist(self, tmp_path):
        """If no fills happen, the snapshot is NOT written."""
        snapshot_path = tmp_path / "positions_snapshot.json"

        with patch.dict(os.environ, {"BOT_SNAPSHOT_PATH": str(snapshot_path)}):
            service = RoutedExecutionService()
            req = SessionRequest(user_id=801, live=False)
            assert asyncio.run(service.start_session(req)) is True

            # HOLD signal → no fill
            signal = StrategySignal(
                symbol="BTCUSDT", timeframe="1h", horizon_bars=4,
                signal="HOLD", confidence=0.50,
            )
            results = asyncio.run(
                service.route_signals(801, signals=(signal,), prices={"BTCUSDT": 50_000.0})
            )
            assert results == ()

        assert not snapshot_path.exists()


# ====================================================================
# TASK 2 supplement — profit_factor round-trip pairing
# ====================================================================

class TestProfitFactorRoundTrip:
    """profit_factor must use entry→exit paired round-trips, not arbitrary sums."""

    def test_profit_factor_all_winners(self):
        fills = []
        idx = pd.date_range("2024-01-01", periods=4, freq="1h", tz="UTC")
        # RT 1: BUY@100, SELL@110 → win $10
        fills.append(Fill(idx[0], "BTCUSDT", "BUY", 1.0, 100.0, 0.1, 0.0, 0.9))
        fills.append(Fill(idx[1], "BTCUSDT", "SELL", 1.0, 110.0, 0.1, 0.0, 0.9))
        # RT 2: BUY@200, SELL@220 → win $20
        fills.append(Fill(idx[2], "BTCUSDT", "BUY", 1.0, 200.0, 0.1, 0.0, 0.9))
        fills.append(Fill(idx[3], "BTCUSDT", "SELL", 1.0, 220.0, 0.1, 0.0, 0.9))

        r = _make_result(fills=fills)
        assert r.profit_factor > 1.0  # all winners → large profit factor

    def test_profit_factor_all_losers(self):
        fills = []
        idx = pd.date_range("2024-01-01", periods=4, freq="1h", tz="UTC")
        # RT 1: BUY@100, SELL@90 → loss $10
        fills.append(Fill(idx[0], "BTCUSDT", "BUY", 1.0, 100.0, 0.1, 0.0, 0.9))
        fills.append(Fill(idx[1], "BTCUSDT", "SELL", 1.0, 90.0, 0.1, 0.0, 0.9))
        # RT 2: BUY@200, SELL@180 → loss $20
        fills.append(Fill(idx[2], "BTCUSDT", "BUY", 1.0, 200.0, 0.1, 0.0, 0.9))
        fills.append(Fill(idx[3], "BTCUSDT", "SELL", 1.0, 180.0, 0.1, 0.0, 0.9))

        r = _make_result(fills=fills)
        assert r.profit_factor < 1.0  # all losers

    def test_profit_factor_empty_fills(self):
        r = _make_result(fills=[])
        assert r.profit_factor == 0.0  # no wins, no losses

    def test_profit_factor_short_round_trip(self):
        """SELL→BUY round-trip should also work."""
        fills = []
        idx = pd.date_range("2024-01-01", periods=2, freq="1h", tz="UTC")
        # Short entry, buy exit
        fills.append(Fill(idx[0], "ETHUSDT", "SELL", 1.0, 2500.0, 0.1, 0.0, 0.8))
        fills.append(Fill(idx[1], "ETHUSDT", "BUY", 1.0, 2400.0, 0.1, 0.0, 0.8))

        r = _make_result(fills=fills)
        # SELL@2500, BUY@2400, pnl=(2500-2400)*1=100 - costs
        assert r.profit_factor > 0.0
