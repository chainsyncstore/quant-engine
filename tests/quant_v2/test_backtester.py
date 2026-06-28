"""Tests for the backtester engine and report generator using synthetic data."""
from __future__ import annotations

import json
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from quant_v2.research.backtester import (
    BacktestConfig,
    BacktestResult,
    Fill,
    _bar_windows,
    _predict,
    _sim_fill,
)
from quant_v2.execution.cost_policy import ExecutionCostPolicy
from quant_v2.research.backtest_report import (
    _compute_metrics,
    _monthly_returns_table,
    generate_report,
)
from quant_v2.model_registry import ModelRegistry
from quant_v2.models.trainer import save_model_bundle, train


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_result(
    n_bars: int = 50,
    initial_equity: float = 300.0,
    n_fills: int = 4,
    gross_pnl: float = 12.0,
    total_fees: float = 1.0,
    total_slippage: float = 0.5,
) -> BacktestResult:
    idx = pd.date_range("2024-01-01", periods=n_bars, freq="1h", tz="UTC")
    equity = pd.Series(
        initial_equity + np.linspace(0, gross_pnl - total_fees - total_slippage, n_bars),
        index=idx,
        name="equity_usd",
    )
    daily = equity.resample("D").last().pct_change().dropna()
    fills = [
        Fill(
            timestamp=idx[i * 10],
            symbol="BTCUSDT",
            side="BUY" if i % 2 == 0 else "SELL",
            quantity=0.001,
            price=50_000.0 + i * 100,
            fee_usd=0.1,
            slippage_usd=0.05,
            confidence=0.72,
        )
        for i in range(n_fills)
    ]
    return BacktestResult(
        config=BacktestConfig(
            symbol="BTCUSDT",
            start_date="2024-01-01",
            end_date="2024-03-01",
            initial_equity=initial_equity,
        ),
        equity_curve=equity,
        fills=fills,
        daily_returns=daily,
        total_trades=n_fills,
        win_trades=n_fills // 2,
        gross_pnl=gross_pnl,
        total_fees=total_fees,
        total_slippage=total_slippage,
    )


# ---------------------------------------------------------------------------
# _bar_windows
# ---------------------------------------------------------------------------

class TestBarWindows:

    def test_yields_expanding_windows(self):
        df = pd.DataFrame({"close": range(10)})
        windows = list(_bar_windows(df, warmup=5))
        assert len(windows) == 5  # bars 5..9 inclusive
        for i, w in enumerate(windows):
            assert len(w) == 6 + i

    def test_warmup_equals_length_yields_one(self):
        df = pd.DataFrame({"close": range(5)})
        windows = list(_bar_windows(df, warmup=4))
        assert len(windows) == 1

    def test_warmup_exceeds_length_yields_none(self):
        df = pd.DataFrame({"close": range(3)})
        windows = list(_bar_windows(df, warmup=5))
        assert len(windows) == 0


# ---------------------------------------------------------------------------
# _predict
# ---------------------------------------------------------------------------

class TestPredict:

    def _make_model(self, prob: float):
        m = MagicMock()
        m.predict_proba.return_value = np.array([[1 - prob, prob]])
        return m

    def test_single_model_returns_prob(self):
        models = {4: self._make_model(0.72)}
        p, unc = _predict(models, pd.DataFrame({"a": [1.0]}))
        assert p == pytest.approx(0.72)
        assert unc == 0.0  # only one model, no std

    def test_ensemble_average(self):
        models = {2: self._make_model(0.60), 4: self._make_model(0.80)}
        p, unc = _predict(models, pd.DataFrame({"a": [1.0]}))
        assert p == pytest.approx(0.70)
        assert unc > 0.0

    def test_empty_models_returns_neutral(self):
        p, unc = _predict({}, pd.DataFrame())
        assert p == 0.5
        assert unc == 1.0

    def test_broken_model_skipped(self):
        bad = MagicMock()
        bad.predict_proba.side_effect = ValueError("broken")
        good = self._make_model(0.65)
        p, unc = _predict({2: bad, 4: good}, pd.DataFrame({"a": [1.0]}))
        assert p == pytest.approx(0.65)


# ---------------------------------------------------------------------------
# _sim_fill
# ---------------------------------------------------------------------------

class TestSimFill:

    def test_fee_scales_with_notional(self):
        fee1, _ = _sim_fill(50_000.0, 0.01, "BUY", 2.0, 500.0)
        fee2, _ = _sim_fill(50_000.0, 0.02, "BUY", 2.0, 1000.0)
        assert fee2 == pytest.approx(fee1 * 2)

    def test_slippage_square_root_of_participation(self):
        # Doubling ADV halves participation → sqrt(0.5) reduction in impact
        _, slip_small = _sim_fill(50_000.0, 0.01, "BUY", 2.0, 100.0, adv_usd=1_000_000.0)
        _, slip_large = _sim_fill(50_000.0, 0.01, "BUY", 2.0, 100.0, adv_usd=4_000_000.0)
        ratio = slip_small / max(slip_large, 1e-12)
        assert ratio == pytest.approx(2.0, rel=0.01)

    def test_zero_notional_zero_costs(self):
        fee, slip = _sim_fill(50_000.0, 0.0, "BUY", 2.0, 0.0)
        assert fee == 0.0
        assert slip == 0.0


# ---------------------------------------------------------------------------
# Execution cost policy
# ---------------------------------------------------------------------------

class TestExecutionCostPolicy:

    def test_scenarios_are_named_and_monotonic(self):
        policy = ExecutionCostPolicy(policy_version="wp07-execution-cost-v1")
        estimates = policy.scenario_estimates(
            "BTCUSDT",
            "BUY",
            10_000.0,
            adv_usd=1_000_000.0,
            funding_rate_bps=0.25,
            latency_bars=1.0,
        )
        assert set(estimates) == {"base", "adverse", "severe"}
        assert estimates["base"].policy_version == "wp07-execution-cost-v1"
        assert estimates["base"].total_cost_usd <= estimates["adverse"].total_cost_usd
        assert estimates["adverse"].total_cost_usd <= estimates["severe"].total_cost_usd
        assert estimates["base"].total_cost_bps <= estimates["adverse"].total_cost_bps


def test_load_model_prefers_registry_active_pointer(tmp_path, monkeypatch):
    monkeypatch.setenv("BOT_MODEL_ROOT", str(tmp_path / "models"))
    registry_root = tmp_path / "models" / "registry"
    monkeypatch.setenv("BOT_MODEL_REGISTRY_ROOT", str(registry_root))
    artifact = tmp_path / "models" / "candidate_a"
    artifact.mkdir(parents=True)
    idx = pd.date_range("2026-01-01", periods=180, freq="h", tz="UTC")
    X = pd.DataFrame(
        {
            "f1": np.linspace(-2.0, 2.0, len(idx)),
            "f2": np.sin(np.linspace(0.0, 8.0, len(idx))),
            "f3": np.cos(np.linspace(0.0, 6.0, len(idx))),
        },
        index=idx,
    )
    y = pd.Series((np.sin(np.linspace(0.0, 12.0, len(idx))) > 0.0).astype(int), index=idx)
    monkeypatch.setenv("QUANT_IMAGE", "registry.example/quant-bot@sha256:" + "a" * 64)
    trained = train(X, y, horizon=4, calibration_frac=0.2)
    save_model_bundle(trained, artifact / "model_4m.pkl")

    registry = ModelRegistry(registry_root)
    registry.register_version("candidate_a", artifact, metrics={"promotion_eligible": True})
    registry.set_active_version("candidate_a")

    active_file = registry_root / "active.json"
    poisoned_dir = tmp_path / "models" / "poisoned"
    poisoned_dir.mkdir(parents=True)
    (poisoned_dir / "model_4m.pkl").write_text("placeholder", encoding="utf-8")
    active_file.write_text(
        json.dumps({"version_id": "poisoned", "updated_at": "2026-01-01T00:00:00Z"}),
        encoding="utf-8",
    )

    from quant_v2.research.backtester import _load_model, BacktestConfig

    models = _load_model(BacktestConfig())
    assert set(models) == {4}
    assert active_file.read_text(encoding="utf-8").find("poisoned") != -1


# ---------------------------------------------------------------------------
# BacktestResult metrics
# ---------------------------------------------------------------------------

class TestBacktestResultMetrics:

    def test_net_pnl_is_gross_minus_costs(self):
        r = _make_result(gross_pnl=10.0, total_fees=1.0, total_slippage=0.5)
        assert r.net_pnl == pytest.approx(8.5)

    def test_win_rate(self):
        r = _make_result(n_fills=4)
        r2 = BacktestResult(
            config=r.config, equity_curve=r.equity_curve,
            fills=r.fills, daily_returns=r.daily_returns,
            total_trades=10, win_trades=7,
            gross_pnl=10.0, total_fees=1.0, total_slippage=0.5,
        )
        assert r2.win_rate == pytest.approx(0.7)

    def test_sharpe_positive_for_upward_equity(self):
        r = _make_result(gross_pnl=50.0)
        assert r.sharpe > 0.0

    def test_max_drawdown_zero_for_flat_equity(self):
        r = _make_result()
        eq = pd.Series([300.0] * 50, index=r.equity_curve.index)
        daily = eq.resample("D").last().pct_change().fillna(0)
        r2 = BacktestResult(
            config=r.config, equity_curve=eq, fills=[],
            daily_returns=daily, total_trades=0, win_trades=0,
            gross_pnl=0.0, total_fees=0.0, total_slippage=0.0,
        )
        # Flat equity: (eq - rolling_max) / rolling_max = 0 everywhere
        assert r2.max_drawdown <= 0.0

    def test_max_drawdown_negative_for_declining_equity(self):
        r = _make_result(gross_pnl=-20.0)
        assert r.max_drawdown < 0.0


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

class TestBacktestReport:

    def test_compute_metrics_keys(self):
        r = _make_result()
        m = _compute_metrics(r)
        assert "Sharpe ratio" in m
        assert "Net PnL" in m
        assert "Win rate" in m
        assert "Cost drag" in m
        assert "Max drawdown" in m

    def test_monthly_returns_table_non_empty(self):
        r = _make_result(n_bars=500)
        html = _monthly_returns_table(r.daily_returns)
        assert "<table" in html
        assert "%" in html

    def test_monthly_returns_table_empty(self):
        html = _monthly_returns_table(pd.Series(dtype=float))
        assert "<p>" in html

    def test_generate_report_creates_html(self, tmp_path):
        r = _make_result(n_bars=200)
        out = tmp_path / "report.html"
        path = generate_report(r, output_path=out)
        assert path.exists()
        content = path.read_text()
        assert "chart.js" in content.lower()
        assert "BTCUSDT" in content
        assert "Sharpe" in content

    def test_generate_report_comparison(self, tmp_path):
        r_a = _make_result(n_bars=200, gross_pnl=20.0)
        r_b = _make_result(n_bars=200, gross_pnl=5.0)
        out = tmp_path / "compare.html"
        path = generate_report(r_a, output_path=out, comparison=r_b)
        content = path.read_text()
        assert "Model A" in content
        assert "Model B" in content
        assert "compChart" in content

    def test_generate_report_default_path(self, tmp_path):
        r = _make_result(n_bars=100)
        real_path = tmp_path / "reports" / "backtest_test.html"
        real_path.parent.mkdir(parents=True, exist_ok=True)
        path = generate_report(r, output_path=real_path)
        assert path.exists()
        assert path.suffix == ".html"

    def test_report_contains_trade_log(self, tmp_path):
        r = _make_result(n_fills=5)
        out = tmp_path / "trades.html"
        path = generate_report(r, output_path=out)
        content = path.read_text()
        assert "Trade Log" in content
        assert "BUY" in content or "SELL" in content

    def test_generate_report_includes_cost_sensitivity(self, tmp_path):
        r = _make_result(n_bars=120)
        r.cost_policy_version = "wp07-execution-cost-v1"
        r.cost_components_usd = {
            "fees": 1.0,
            "slippage": 0.5,
            "spread": 0.25,
            "funding": 0.1,
            "latency": 0.05,
            "impact": 0.2,
            "total_cost_usd": 2.1,
        }
        r.cost_scenarios = {
            "base": {
                "notional_usd": 10_000.0,
                "total_cost_usd": 2.1,
                "total_cost_bps": 2.1,
            },
            "adverse": {
                "notional_usd": 10_000.0,
                "total_cost_usd": 3.2,
                "total_cost_bps": 3.2,
            },
            "severe": {
                "notional_usd": 10_000.0,
                "total_cost_usd": 4.4,
                "total_cost_bps": 4.4,
            },
        }
        out = tmp_path / "sensitivity.html"
        path = generate_report(r, output_path=out)
        content = path.read_text()
        assert "Cost Sensitivity" in content
        assert "wp07-execution-cost-v1" in content
        assert "Adverse" in content
        assert "Severe" in content
