from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone

import pandas as pd

import quant_v2.research.run_group_validation as runner
from quant.validation.metrics import FoldMetrics
from quant_v2.data.storage import MultiSymbolSnapshot
from quant_v2.research.experiment_score import ExperimentScoreReport
from quant_v2.research.group_validation import GroupValidationFoldResult, GroupValidationResult
from quant_v2.research.scorecard import GateInputs, GateResult, ScoreInputs
from quant_v2.research.stage1_pipeline import Stage1Result


def _prepared_df() -> pd.DataFrame:
    idx = pd.MultiIndex.from_product(
        [
            pd.date_range("2025-01-01", periods=2, freq="1h", tz="UTC"),
            ["BTCUSDT", "ETHUSDT"],
        ],
        names=["timestamp", "symbol"],
    )
    return pd.DataFrame(
        {
            "open": [100.0, 80.0, 101.0, 81.0],
            "high": [101.0, 81.0, 102.0, 82.0],
            "low": [99.0, 79.0, 100.0, 80.0],
            "close": [100.5, 80.5, 101.5, 81.5],
            "volume": [1000.0, 900.0, 1005.0, 910.0],
            "f1": [0.6, 0.4, 0.7, 0.3],
            "label_1m": [1, 0, -1, -1],
            "label_4m": [-1, -1, -1, -1],
        },
        index=idx,
    )


def _group_validation_result(horizon: int) -> GroupValidationResult:
    metrics = FoldMetrics(
        fold=0,
        train_start="2025-01-01T00:00:00+00:00",
        test_start="2025-01-01T01:00:00+00:00",
        test_end="2025-01-01T02:00:00+00:00",
        spread_adjusted_ev=1.2,
        win_rate=0.6,
        n_trades=10,
        sharpe=0.8,
        max_drawdown=-3.0,
        worst_losing_streak=2,
    )
    fold = GroupValidationFoldResult(
        split_id="t00_g00",
        n_train_rows=100,
        n_test_rows=20,
        n_valid_rows=18,
        test_symbols=("BTCUSDT",),
        metrics=metrics,
    )
    return GroupValidationResult(
        horizon=horizon,
        validation_mode="group_purged_cpcv",
        split_summary={
            "n_splits": 1,
            "n_test_symbols": 1,
            "n_unique_test_symbols": 1,
            "n_total_test_rows": 20,
            "n_total_train_rows": 100,
        },
        folds=[fold],
        overall={
            "spread_adjusted_ev": 1.2,
            "win_rate": 0.6,
            "sharpe": 0.8,
            "max_drawdown": -3.0,
            "n_trades": 10,
            "worst_losing_streak": 2,
        },
        robustness={
            "probabilistic_sharpe_ratio": 0.71,
            "deflated_sharpe_ratio": 0.58,
        },
        n_trials_assumed=7,
        trial_count=3,
        fold_dispersion={
            "spread_adjusted_ev": {
                "count": 1,
                "mean": 1.2,
                "std": 0.0,
                "min": 1.2,
                "max": 1.2,
                "median": 1.2,
            }
        },
        selection_risk={
            "pbo_equivalent": 0.25,
            "selected_fold_rank_percentile": 0.75,
            "n_folds": 1.0,
            "n_trials_assumed": 7.0,
        },
        cost_sensitivity={
            "base": {"spread_adjusted_ev_mean": 1.2, "total_return_mean": 12.0, "win_rate_mean": 0.6}
        },
        failure_reasons=(),
    )


class _ProbeClient:
    def fetch_historical(self, date_from, date_to, symbol=None, interval=None):
        idx = pd.date_range(datetime.now(timezone.utc) - pd.Timedelta(hours=2), periods=3, freq="1h", tz="UTC")
        return pd.DataFrame(
            {
                "open": [100.0, 101.0, 102.0],
                "high": [101.0, 102.0, 103.0],
                "low": [99.0, 100.0, 101.0],
                "close": [100.5, 101.5, 102.5],
                "volume": [1000.0, 1001.0, 1002.0],
            },
            index=idx,
        )

    def fetch_funding_rates(self, date_from, date_to, symbol=None):
        idx = pd.date_range(datetime.now(timezone.utc) - pd.Timedelta(hours=2), periods=2, freq="2h", tz="UTC")
        return pd.DataFrame({"funding_rate_raw": [0.0001, 0.0002]}, index=idx)

    def fetch_open_interest(self, date_from, date_to, symbol=None, period="1h"):
        idx = pd.date_range(datetime.now(timezone.utc) - pd.Timedelta(hours=2), periods=3, freq="1h", tz="UTC")
        return pd.DataFrame(
            {
                "open_interest": [1000.0, 1001.0, 1002.0],
                "open_interest_value": [10_000.0, 10_010.0, 10_020.0],
            },
            index=idx,
        )

    def get_rate_limit_snapshot(self):
        return {
            "provider_name": "binance_futures_rest",
            "status": "healthy",
            "used_weight_1m": 900,
            "weight_limit_1m": 2400,
            "throttle_threshold_1m": 1800,
            "headroom_1m": 1500,
            "pressure_fraction": 900 / 2400,
            "throttle_interval_seconds": 0.1,
            "last_request_age_seconds": 1.25,
        }


def test_parse_helpers() -> None:
    assert runner.parse_csv_symbols("btcusdt, ethusdt") == ("BTCUSDT", "ETHUSDT")
    assert runner.parse_csv_ints("1,4,12", default=(3,)) == (1, 4, 12)


def test_run_validation_pipeline_writes_report(tmp_path, monkeypatch) -> None:
    prepared = _prepared_df()

    snap_path = tmp_path / "snap.parquet"
    snap_manifest = tmp_path / "snap.manifest.json"
    snap_path.write_text("placeholder", encoding="utf-8")
    snap_manifest.write_text("{}", encoding="utf-8")

    snapshot = MultiSymbolSnapshot(
        parquet_path=snap_path,
        manifest_path=snap_manifest,
        manifest={"dataset_name": "unit"},
    )

    def fake_load_or_build_dataset(**kwargs):
        return prepared, snapshot

    def fake_prepare_multi_symbol_dataset(raw_df, horizons):
        return prepared

    stage1_summary = {
        "n_splits": 1,
        "n_test_symbols": 1,
        "n_unique_test_symbols": 1,
        "n_total_test_rows": 20,
        "n_total_train_rows": 100,
    }
    stage1_splits = [object()]

    def fake_build_stage1_result(dataset, snapshot, **kwargs):
        assert dataset is prepared
        return Stage1Result(
            dataset=prepared,
            snapshot=snapshot,
            splits=stage1_splits,
            split_summary=stage1_summary,
        )

    def fake_run_group_purged_validation(df, *, horizon, precomputed_splits=None, split_summary=None, **kwargs):
        assert precomputed_splits is stage1_splits
        assert split_summary == stage1_summary
        return _group_validation_result(horizon=horizon)

    score_report = ExperimentScoreReport(
        score_inputs=ScoreInputs(
            robustness=72.0,
            tradability=70.0,
            risk=68.0,
            generalization=71.0,
            live_readiness=66.0,
        ),
        gate_inputs=GateInputs(
            dsr_majority=0.60,
            positive_ev_fold_ratio=0.70,
            ruin_probability=0.20,
            single_symbol_ev_dependency=0.30,
            shadow_live_drift_ok=True,
        ),
        score=70.1,
        gates=GateResult(
            passed=True,
            checks={
                "dsr_majority": True,
                "positive_ev_fold_ratio": True,
                "ruin_probability": True,
                "single_symbol_ev_dependency": True,
                "shadow_live_drift_ok": True,
            },
        ),
        trial_count=6,
        selection_risk={
            "1": {"pbo_equivalent": 0.25, "selected_holdout_rank_percentile": 0.75}
        },
    )

    def fake_build_report_from_experiment(experiment):
        return score_report

    monkeypatch.setattr(runner, "load_or_build_dataset", fake_load_or_build_dataset)
    monkeypatch.setattr(runner, "prepare_multi_symbol_dataset", fake_prepare_multi_symbol_dataset)
    monkeypatch.setattr(runner, "build_stage1_result", fake_build_stage1_result)
    monkeypatch.setattr(runner, "run_group_purged_validation", fake_run_group_purged_validation)
    monkeypatch.setattr(runner, "build_report_from_experiment", fake_build_report_from_experiment)

    out_path = tmp_path / "report.json"
    report = runner.run_validation_pipeline(
        snapshot_path="dummy.parquet",
        months=1,
        symbols=("BTCUSDT", "ETHUSDT"),
        horizons=(1, 4),
        output_path=out_path,
        client=_ProbeClient(),
    )

    assert out_path.exists()
    assert report["scorecard"]["score"] == 70.1
    assert sorted(report["horizons"].keys()) == ["1", "4"]
    assert report["dataset"]["stage1_split_summary"]["n_splits"] == 1
    assert report["scorecard"]["gates"]["passed"] is True
    assert report["scorecard"]["trial_count"] == 6
    assert report["scorecard"]["selection_risk"]["1"]["pbo_equivalent"] == 0.25
    assert report["forward_live_simulation"]["mode"] == "shadow_forward_live_simulation"
    assert "aggregate" in report["forward_live_simulation"]
    assert "replay_regression" in report
    assert report["replay_regression"]["aggregate"]["n_compared_horizons"] >= 1
    assert report["horizons"]["1"]["validation_evidence"]["trial_count"] == 3
    assert report["horizons"]["1"]["fold_dispersion"]["spread_adjusted_ev"]["mean"] == 1.2
    assert report["horizons"]["1"]["cost_sensitivity"]["base"]["spread_adjusted_ev_mean"] == 1.2
    assert "health_dashboard" in report
    health = report["health_dashboard"]
    assert health["payload"]["run_id"]
    assert report["provider_probes"]["market_data"]["probe_count"] == 2
    assert report["provider_probes"]["market_data"]["rate_limit_status"] == "healthy"
    assert report["provider_probes"]["market_data"]["rate_limit_used_weight_1m"] == 900
    assert health["payload"]["data_health"]["provider_status"] == "healthy"
    assert health["payload"]["data_health"]["provider_rate_limit_status"] == "healthy"
    assert Path(health["json_path"]).exists()
    assert Path(health["summary_path"]).exists()
    assert Path(health["audit_path"]).exists()
