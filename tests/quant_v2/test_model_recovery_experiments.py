from __future__ import annotations

import json
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quant_v2.contracts import StrategySignal
from quant_v2.research.portfolio_replay import ReplayActorConfig, ReplayActorResult, ReplayFill
import quant_v2.research.model_recovery_experiments as experiments
from quant_v2.research.model_recovery_experiments import CandidateEvaluation, RecoveryCandidateConfig


def _raw_dataset(rows: int = 240) -> pd.DataFrame:
    timestamps = pd.date_range("2024-01-01", periods=rows, freq="D", tz="UTC")
    pieces = []
    for idx, symbol in enumerate(("BTCUSDT", "ETHUSDT")):
        phase = np.linspace(0.0, 8.0, rows) + idx
        close = 100.0 + (idx * 15.0) + np.linspace(0.0, 80.0, rows) + np.sin(phase) * 0.5
        volume = 1_000.0 + np.cos(phase) * 50.0
        frame = pd.DataFrame(
            {
                "open": close - 0.25,
                "high": close + 0.50,
                "low": close - 0.75,
                "close": close,
                "volume": volume,
                "quote_volume": volume * close,
                "funding_rate_raw": np.where(np.arange(rows) % 8 == 0, 0.0001, 0.0),
                "open_interest": 10_000.0 + np.linspace(0.0, 500.0, rows),
                "open_interest_value": 1_000_000.0 + np.linspace(0.0, 15_000.0, rows),
            },
            index=timestamps,
        )
        frame.index.name = "timestamp"
        frame["symbol"] = symbol
        pieces.append(frame.reset_index().set_index(["timestamp", "symbol"]).sort_index())
    return pd.concat(pieces).sort_index()


def _fake_feature_frame(raw: pd.DataFrame) -> pd.DataFrame:
    frame = raw.copy()
    close = pd.to_numeric(frame["close"], errors="coerce")
    open_ = pd.to_numeric(frame["open"], errors="coerce")
    high = pd.to_numeric(frame["high"], errors="coerce")
    low = pd.to_numeric(frame["low"], errors="coerce")
    volume = pd.to_numeric(frame["volume"], errors="coerce")
    oi = pd.to_numeric(frame["open_interest"], errors="coerce")

    def _group_apply(series: pd.Series, func):
        return series.groupby(level="symbol", sort=False).transform(func)

    frame["roc_1"] = _group_apply(close, lambda s: s.pct_change().fillna(0.0))
    frame["atr_14"] = _group_apply(high - low, lambda s: s.rolling(14, min_periods=1).mean().fillna(0.0))
    frame["body_range_ratio"] = ((close - open_).abs() / (high - low).replace(0.0, np.nan)).fillna(0.0)
    frame["ema_slope_5"] = _group_apply(close, lambda s: s.ewm(span=5, adjust=False).mean().pct_change().fillna(0.0))
    frame["vol_zscore"] = _group_apply(volume, lambda s: ((s - s.rolling(20, min_periods=1).mean()) / s.rolling(20, min_periods=1).std().replace(0.0, np.nan)).fillna(0.0))
    frame["hour_sin"] = np.sin(np.arange(len(frame)) / 10.0)
    frame["return_autocorr_5"] = _group_apply(close.pct_change().fillna(0.0), lambda s: s.rolling(5, min_periods=1).mean().fillna(0.0))
    frame["roc_60"] = _group_apply(close, lambda s: s.pct_change(60).fillna(0.0))
    frame["funding_rate"] = pd.to_numeric(frame["funding_rate_raw"], errors="coerce").fillna(0.0)
    frame["funding_rate_ma8"] = _group_apply(frame["funding_rate"], lambda s: s.rolling(8, min_periods=1).mean().fillna(0.0))
    frame["oi_roc_1"] = _group_apply(oi, lambda s: s.pct_change().fillna(0.0))
    frame["oi_funding_pressure"] = frame["funding_rate"] * frame["oi_roc_1"]
    frame["btc_return_4h"] = _group_apply(close, lambda s: s.pct_change(4).fillna(0.0))
    frame["bid_ask_spread_bps"] = 5.0
    frame["asia_session"] = (np.arange(len(frame)) % 2).astype(float)
    frame["feature_catalog_version"] = "test"
    frame.attrs.update({"feature_catalog_version": "test", "feature_catalog_sha256": "test"})
    return frame


def test_cli_requires_no_production_registry(monkeypatch, tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshot.parquet"
    snapshot.write_text("placeholder", encoding="utf-8")

    with pytest.raises(SystemExit):
        experiments.main(
            [
                "--snapshot-path",
                str(snapshot),
                "--output-root",
                str(tmp_path / "out"),
                "--docs-output-dir",
                str(tmp_path / "docs"),
            ]
        )


def test_label_audit_reports_grid_and_cost_floor() -> None:
    raw = _raw_dataset()
    report = experiments._build_label_audit_report(raw)

    assert report["policy_version"] == experiments.LABEL_POLICY_VERSION
    assert report["summary"]["cell_count"] == len(experiments.DEFAULT_HORIZONS) * len(experiments.DEFAULT_TRAINING_WINDOWS_MONTHS) * len(experiments.DEFAULT_RECENCY_HALF_LIFES_DAYS) * len(experiments.DEFAULT_DEAD_ZONES)
    assert report["summary"]["best_cell"] is not None
    assert report["summary"]["recommended_dead_zone_floor_bps"] > 0.0
    best = report["cells"][report["summary"]["best_cell"]]
    assert best["sample_count"] > 0
    assert "BTCUSDT" in best["by_symbol_label_counts"]
    assert any(key.startswith("BTCUSDT:") for key in best["by_month_label_counts"])


def test_candidate_grid_is_deterministic_and_horizon_diverse() -> None:
    raw = _raw_dataset()
    report = experiments._build_label_audit_report(raw)

    configs_1 = experiments._candidate_grid(report)
    configs_2 = experiments._candidate_grid(report)
    assert [cfg.candidate_id() for cfg in configs_1] == [cfg.candidate_id() for cfg in configs_2]

    limited = experiments._limit_candidates(configs_1, 5)
    assert len(limited) == 5
    assert {cfg.horizon for cfg in limited} == {2, 4, 8}


def test_candidate_grid_expands_trade_outcome_sides() -> None:
    raw = _raw_dataset()
    report = experiments._build_label_audit_report(raw)

    configs = experiments._candidate_grid(report, label_mode="trade_outcome")

    assert configs
    assert {cfg.trade_outcome_side for cfg in configs} == {"long", "short"}
    assert len({cfg.candidate_id() for cfg in configs}) == len(configs)
    assert [cfg.trade_outcome_side for cfg in configs[:2]] == ["long", "short"]


def test_trade_outcome_candidate_ids_and_side_validation() -> None:
    directional = RecoveryCandidateConfig(4, 3, 30, 0.001, "full")
    trade_long = RecoveryCandidateConfig(4, 3, 30, 0.001, "full", label_mode="trade_outcome", trade_outcome_side="long")
    trade_short = RecoveryCandidateConfig(4, 3, 30, 0.001, "full", label_mode="trade_outcome", trade_outcome_side="short")

    assert directional.candidate_id() == "h4_tw3m_hl30d_dz0p0010_fsfull"
    assert trade_long.candidate_id().endswith("_sidelong")
    assert trade_short.candidate_id().endswith("_sideshort")
    assert trade_long.candidate_id() != trade_short.candidate_id()
    with pytest.raises(ValueError, match="unsupported trade_outcome_side"):
        RecoveryCandidateConfig(4, 3, 30, 0.001, "full", label_mode="trade_outcome", trade_outcome_side="up")


def test_phase4_variant_specs_are_bounded_and_distinct() -> None:
    report = experiments._build_label_audit_report(_raw_dataset())

    specs = experiments.build_phase4_variant_specs(report)

    assert len(specs) == 4
    assert {spec.variant_id for spec in specs} == {
        "cost_aware_ternary",
        "horizon_specific_features",
        "symbol_group_calibration",
        "regime_gated_abstain",
    }
    floor_fraction = report["summary"]["recommended_dead_zone_floor_bps"] / 10_000.0
    assert all(spec.config.dead_zone_bps >= floor_fraction for spec in specs)


def test_phase4_variant_specs_expand_trade_outcome_sides() -> None:
    report = experiments._build_label_audit_report(_raw_dataset())

    specs = experiments.build_phase4_variant_specs(report, label_mode="trade_outcome")

    assert len(specs) == 8
    assert {spec.variant_id for spec in specs} == {
        "cost_aware_ternary_long",
        "cost_aware_ternary_short",
        "horizon_specific_features_long",
        "horizon_specific_features_short",
        "symbol_group_calibration_long",
        "symbol_group_calibration_short",
        "regime_gated_abstain_long",
        "regime_gated_abstain_short",
    }
    assert {spec.config.trade_outcome_side for spec in specs} == {"long", "short"}
    assert [spec.config.trade_outcome_side for spec in specs[:2]] == ["long", "short"]


def test_build_candidate_labels_supports_trade_outcome_mode() -> None:
    raw = _raw_dataset(rows=40)
    config = RecoveryCandidateConfig(
        horizon=2,
        training_window_months=3,
        recency_half_life_days=30,
        dead_zone_bps=0.001,
        feature_set="full",
        label_mode="trade_outcome",
        trade_outcome_profit_target_bps=20.0,
        trade_outcome_stop_loss_bps=30.0,
        trade_outcome_round_trip_cost_bps=0.0,
    )

    labels, report = experiments._build_candidate_labels(raw, config)

    assert report["label_mode"] == "trade_outcome"
    assert report["trade_outcome_side"] == "long"
    assert report["policy_version"] == "trade_outcome_labels_v1"
    assert len(labels) == len(raw)
    assert set(report["label_counts"]) >= {"take", "skip", "ambiguous", "labelled"}
    short_labels, short_report = experiments._build_candidate_labels(
        raw,
        RecoveryCandidateConfig(
            horizon=2,
            training_window_months=3,
            recency_half_life_days=30,
            dead_zone_bps=0.001,
            feature_set="full",
            label_mode="trade_outcome",
            trade_outcome_side="short",
            trade_outcome_profit_target_bps=20.0,
            trade_outcome_stop_loss_bps=30.0,
            trade_outcome_round_trip_cost_bps=0.0,
        ),
    )
    assert short_report["trade_outcome_side"] == "short"
    assert not labels.equals(short_labels)


def test_phase4_diagnostics_report_summarizes_variant_failures() -> None:
    frame = _raw_dataset(rows=120)
    label_audit = experiments._build_label_audit_report(frame)
    benchmark_report = {
        "policy_version": experiments.BENCHMARK_POLICY_VERSION,
        "actor_summaries": {
            "flat": {"cost_adjusted_net_pnl_usd": 0.0},
            "momentum": {"cost_adjusted_net_pnl_usd": 2.0},
            "mean_reversion": {"cost_adjusted_net_pnl_usd": 1.0},
            "volatility_filtered": {"cost_adjusted_net_pnl_usd": 1.5},
        },
        "comparisons": {"best_actor": "momentum", "best_nonflat_actor": "momentum"},
    }
    variants = [
        CandidateEvaluation(
            config=RecoveryCandidateConfig(4, 3, 30, 0.001, "full"),
            candidate_id="h4_tw3m_hl30d_dz0p0010_fsfull",
            passed=False,
            score=1.0,
            failure_reasons=("candidate_did_not_beat_flat",),
            dataset_manifest={},
            feature_manifest={"selected_feature_columns": ["roc_1", "atr_14"]},
            label_audit={},
            fold_ledger={},
            threshold_policy={},
            holdout_report={
                "holdout_accuracy": 0.5,
                "cost_adjusted_expectancy_bps": -1.0,
                "prediction_audit": {"one_sided_collapse": True},
                "feature_importance": {"roc_1": 0.7},
            },
            replay_report={"actor_summaries": {"candidate": {"cost_adjusted_net_pnl_usd": -2.0}}},
            selection_risk_summary={"variant_kind": "cost_aware_ternary"},
            model_artifact_path=None,
            variant_id="cost_aware_ternary",
        ),
        CandidateEvaluation(
            config=RecoveryCandidateConfig(4, 3, 30, 0.001, "no_open_interest"),
            candidate_id="h4_tw3m_hl30d_dz0p0010_fsno_open_interest",
            passed=False,
            score=2.0,
            failure_reasons=("candidate_did_not_beat_nonflat_benchmark",),
            dataset_manifest={},
            feature_manifest={"selected_feature_columns": ["roc_1", "atr_14"]},
            label_audit={},
            fold_ledger={},
            threshold_policy={},
            holdout_report={
                "holdout_accuracy": 0.55,
                "cost_adjusted_expectancy_bps": -0.5,
                "prediction_audit": {"one_sided_collapse": False},
                "feature_importance": {"atr_14": 0.9},
            },
            replay_report={"actor_summaries": {"candidate": {"cost_adjusted_net_pnl_usd": -1.0}}},
            selection_risk_summary={"variant_kind": "horizon_specific_features"},
            model_artifact_path=None,
            variant_id="horizon_specific_features",
        ),
    ]

    diagnostics = experiments.build_research_input_diagnostics_report(
        frame,
        label_audit_report=label_audit,
        benchmark_report=benchmark_report,
        variant_evaluations=variants,
    )

    assert diagnostics["summary"]["variant_count"] == 2
    assert diagnostics["summary"]["failed_variant_count"] == 2
    assert diagnostics["summary"]["prediction_collapse_observed"] is True
    assert diagnostics["summary"]["best_failed_variant_id"] == "horizon_specific_features"
    assert diagnostics["variant_reports"]["cost_aware_ternary"]["feature_health"]["selected_feature_count"] == 2
    assert diagnostics["regime_comparison"]


def test_phase4_runner_writes_diagnostics_artifacts(tmp_path: Path, monkeypatch) -> None:
    raw = _raw_dataset(rows=120)
    benchmark_report = {
        "policy_version": experiments.BENCHMARK_POLICY_VERSION,
        "actor_summaries": {
            "flat": {"cost_adjusted_net_pnl_usd": 0.0},
            "momentum": {"cost_adjusted_net_pnl_usd": 2.0},
            "mean_reversion": {"cost_adjusted_net_pnl_usd": 1.0},
            "volatility_filtered": {"cost_adjusted_net_pnl_usd": 1.5},
        },
        "comparisons": {"best_actor": "momentum", "best_nonflat_actor": "momentum"},
    }

    monkeypatch.setattr(experiments, "_benchmark_replay_report", lambda frame: benchmark_report)
    monkeypatch.setattr(experiments, "_build_label_audit_report", lambda frame: {"summary": {"best_cell_stats": {"horizon": 4, "training_window_months": 3, "recency_half_life_days": 30, "dead_zone_bps": 0.001}, "recommended_dead_zone_floor_bps": 0.001}})
    monkeypatch.setattr(
        experiments,
        "_evaluate_candidate",
        lambda *args, **kwargs: CandidateEvaluation(
            config=kwargs["config"],
            candidate_id=kwargs["config"].candidate_id(),
            passed=False,
            score=1.0,
            failure_reasons=("forced_fail",),
            dataset_manifest={},
            feature_manifest={"selected_feature_columns": ["roc_1", "atr_14"]},
            label_audit={},
            fold_ledger={},
            threshold_policy={},
            holdout_report={
                "holdout_accuracy": 0.5,
                "cost_adjusted_expectancy_bps": -1.0,
                "prediction_audit": {"one_sided_collapse": False},
                "feature_importance": {"roc_1": 0.5},
            },
            replay_report={"actor_summaries": {"candidate": {"cost_adjusted_net_pnl_usd": -1.0}}},
            selection_risk_summary={"variant_kind": kwargs.get("variant_kind", "")},
            model_artifact_path=None,
            variant_id=kwargs.get("variant_id", ""),
        ),
    )

    result = experiments.run_phase4_research_input_repair(
        raw,
        snapshot_path=tmp_path / "snapshot.parquet",
        output_root=tmp_path / "out",
        docs_output_dir=tmp_path / "docs",
        max_variants=2,
    )

    assert result.summary["recommendation"] == "remain_no_trade"
    assert result.summary["evaluated_variants"] == 2
    assert (tmp_path / "docs" / "research_input_diagnostics.md").exists()
    assert (tmp_path / "docs" / "research_input_repair" / result.run_id / "variant_index.json").exists()


def test_phase4_runner_fails_closed_when_validation_cannot_build(tmp_path: Path, monkeypatch) -> None:
    raw = _raw_dataset(rows=120)
    benchmark_report = {
        "policy_version": experiments.BENCHMARK_POLICY_VERSION,
        "actor_summaries": {
            "flat": {"cost_adjusted_net_pnl_usd": 0.0},
            "momentum": {"cost_adjusted_net_pnl_usd": 2.0},
            "mean_reversion": {"cost_adjusted_net_pnl_usd": 1.0},
            "volatility_filtered": {"cost_adjusted_net_pnl_usd": 1.5},
        },
        "comparisons": {"best_actor": "momentum", "best_nonflat_actor": "momentum"},
    }

    monkeypatch.setattr(experiments, "_benchmark_replay_report", lambda frame: benchmark_report)
    monkeypatch.setattr(experiments, "_build_label_audit_report", lambda frame: {"summary": {"best_cell_stats": {"horizon": 4, "training_window_months": 3, "recency_half_life_days": 30, "dead_zone_bps": 0.001}, "recommended_dead_zone_floor_bps": 0.001}})

    def _raise(*_args, **_kwargs):
        raise ValueError("Not enough timestamps for temporal validation")

    monkeypatch.setattr(experiments, "_evaluate_candidate", _raise)

    result = experiments.run_phase4_research_input_repair(
        raw,
        snapshot_path=tmp_path / "snapshot.parquet",
        output_root=tmp_path / "out",
        docs_output_dir=tmp_path / "docs",
        max_variants=1,
    )

    assert result.summary["recommendation"] == "remain_no_trade"
    assert result.summary["evaluated_variants"] == 1
    assert result.summary["passed_variants"] == 0
    assert result.variant_evaluations[0].passed is False
    assert "Not enough timestamps" in result.variant_evaluations[0].failure_reasons[0]
    assert (tmp_path / "docs" / "research_input_diagnostics.md").exists()
    assert (tmp_path / "docs" / "research_input_repair" / result.run_id / "variant_index.json").exists()


def test_threshold_selection_prefers_best_dev_threshold() -> None:
    result = experiments._select_threshold_from_oof_predictions(
        np.array([0.10, 0.20, 0.80, 0.90]),
        np.array([0, 0, 1, 1]),
    )

    assert result["source"] == "oof_dev_predictions"
    assert result["selected_threshold"] == pytest.approx(0.50)
    assert result["selected_accuracy"] == pytest.approx(1.0)


def test_benchmark_signal_resolver_returns_strategy_signal() -> None:
    history = pd.DataFrame(
        {
            "close": [100.0, 101.0, 102.0, 103.0, 104.0],
            "open": [99.0, 100.0, 101.0, 102.0, 103.0],
            "high": [101.0, 102.0, 103.0, 104.0, 105.0],
            "low": [98.0, 99.0, 100.0, 101.0, 102.0],
            "volume": [1000.0] * 5,
        },
        index=pd.date_range("2024-01-01", periods=5, freq="h", tz="UTC"),
    )
    actor = ReplayActorConfig(
        name="momentum",
        kind="fixed",
        metadata={"benchmark_name": "momentum"},
    )

    signal = experiments._benchmark_signal_resolver(
        actor,
        "BTCUSDT",
        history,
        pd.Timestamp("2024-01-01T04:00:00Z"),
        None,
    )

    assert isinstance(signal, StrategySignal)
    assert signal.symbol == "BTCUSDT"


def test_candidate_signal_resolver_uses_precomputed_features(monkeypatch) -> None:
    timestamp = pd.Timestamp("2024-01-01T00:00:00Z")
    feature_frame = pd.DataFrame(
        {"roc_1": [0.02], "atr_14": [1.5]},
        index=pd.MultiIndex.from_tuples([(timestamp, "BTCUSDT")], names=["timestamp", "symbol"]),
    )
    actor = ReplayActorConfig(
        name="candidate",
        kind="model",
        model=SimpleNamespace(),
        threshold=0.60,
        metadata={"feature_columns": ["roc_1", "atr_14"]},
    )

    monkeypatch.setattr(experiments, "build_features", lambda *_args, **_kwargs: pytest.fail("feature pipeline should be precomputed"))
    monkeypatch.setattr(experiments, "predict_proba", lambda model, X: np.asarray([0.75]))

    signal = experiments._candidate_signal_resolver(
        actor,
        "BTCUSDT",
        pd.DataFrame({"close": [100.0]}, index=[timestamp]),
        timestamp,
        None,
        feature_frame=feature_frame,
    )

    assert isinstance(signal, StrategySignal)
    assert signal.signal == "BUY"
    assert signal.confidence == pytest.approx(0.75)


def test_candidate_signal_resolver_trade_outcome_long_and_short_semantics(monkeypatch) -> None:
    timestamp = pd.Timestamp("2024-01-01T00:00:00Z")
    feature_frame = pd.DataFrame(
        {"roc_1": [0.02], "atr_14": [1.5]},
        index=pd.MultiIndex.from_tuples([(timestamp, "BTCUSDT")], names=["timestamp", "symbol"]),
    )
    long_actor = ReplayActorConfig(
        name="candidate",
        kind="model",
        model=SimpleNamespace(),
        threshold=0.60,
        metadata={"feature_columns": ["roc_1", "atr_14"], "label_mode": "trade_outcome", "trade_outcome_side": "long"},
    )
    short_actor = ReplayActorConfig(
        name="candidate",
        kind="model",
        model=SimpleNamespace(),
        threshold=0.60,
        metadata={"feature_columns": ["roc_1", "atr_14"], "label_mode": "trade_outcome", "trade_outcome_side": "short"},
    )

    monkeypatch.setattr(experiments, "build_features", lambda *_args, **_kwargs: pytest.fail("feature pipeline should be precomputed"))
    monkeypatch.setattr(experiments, "predict_proba", lambda model, X: np.asarray([0.75]))

    long_signal = experiments._candidate_signal_resolver(
        long_actor,
        "BTCUSDT",
        pd.DataFrame({"close": [100.0]}, index=[timestamp]),
        timestamp,
        None,
        feature_frame=feature_frame,
    )
    short_signal = experiments._candidate_signal_resolver(
        short_actor,
        "BTCUSDT",
        pd.DataFrame({"close": [100.0]}, index=[timestamp]),
        timestamp,
        None,
        feature_frame=feature_frame,
    )

    assert long_signal.signal == "BUY"
    assert long_signal.reason == "trade_outcome_long_take>=0.60"
    assert short_signal.signal == "SELL"
    assert short_signal.reason == "trade_outcome_short_take>=0.60"

    monkeypatch.setattr(experiments, "predict_proba", lambda model, X: np.asarray([0.25]))
    low_long_signal = experiments._candidate_signal_resolver(
        long_actor,
        "BTCUSDT",
        pd.DataFrame({"close": [100.0]}, index=[timestamp]),
        timestamp,
        None,
        feature_frame=feature_frame,
    )
    low_short_signal = experiments._candidate_signal_resolver(
        short_actor,
        "BTCUSDT",
        pd.DataFrame({"close": [100.0]}, index=[timestamp]),
        timestamp,
        None,
        feature_frame=feature_frame,
    )

    assert low_long_signal.signal == "HOLD"
    assert low_short_signal.signal == "HOLD"


def test_candidate_signal_resolver_directional_return_preserves_inverse_side(monkeypatch) -> None:
    timestamp = pd.Timestamp("2024-01-01T00:00:00Z")
    feature_frame = pd.DataFrame(
        {"roc_1": [0.02], "atr_14": [1.5]},
        index=pd.MultiIndex.from_tuples([(timestamp, "BTCUSDT")], names=["timestamp", "symbol"]),
    )
    actor = ReplayActorConfig(
        name="candidate",
        kind="model",
        model=SimpleNamespace(),
        threshold=0.60,
        metadata={"feature_columns": ["roc_1", "atr_14"], "label_mode": "directional_return"},
    )

    monkeypatch.setattr(experiments, "build_features", lambda *_args, **_kwargs: pytest.fail("feature pipeline should be precomputed"))
    monkeypatch.setattr(experiments, "predict_proba", lambda model, X: np.asarray([0.25]))

    signal = experiments._candidate_signal_resolver(
        actor,
        "BTCUSDT",
        pd.DataFrame({"close": [100.0]}, index=[timestamp]),
        timestamp,
        None,
        feature_frame=feature_frame,
    )

    assert signal.signal == "SELL"
    assert signal.reason == "candidate_proba_down>=0.60"


def test_candidate_decision_returns_bps_respects_trade_outcome_side() -> None:
    predictions = np.asarray([1, 0], dtype=int)
    forward_return_bps = np.asarray([10.0, -7.0], dtype=float)

    long_returns = experiments._candidate_decision_returns_bps(
        label_mode="trade_outcome",
        trade_outcome_side="long",
        predictions=predictions,
        forward_return_bps=forward_return_bps,
        cost_bps=2.0,
    )
    short_returns = experiments._candidate_decision_returns_bps(
        label_mode="trade_outcome",
        trade_outcome_side="short",
        predictions=predictions,
        forward_return_bps=forward_return_bps,
        cost_bps=2.0,
    )
    directional_returns = experiments._candidate_decision_returns_bps(
        label_mode="directional_return",
        trade_outcome_side="long",
        predictions=predictions,
        forward_return_bps=forward_return_bps,
        cost_bps=2.0,
    )

    assert long_returns.tolist() == [8.0, 0.0]
    assert short_returns.tolist() == [-12.0, 0.0]
    assert directional_returns.tolist() == [8.0, 5.0]


def test_evaluate_trade_outcome_candidate_reports_take_and_skip_counts(tmp_path: Path, monkeypatch) -> None:
    raw = _raw_dataset()
    benchmark_report = {
        "actor_summaries": {
            "flat": {"cost_adjusted_net_pnl_usd": 0.0},
            "momentum": {"cost_adjusted_net_pnl_usd": 3.0},
            "mean_reversion": {"cost_adjusted_net_pnl_usd": 2.0},
            "volatility_filtered": {"cost_adjusted_net_pnl_usd": 1.5},
            "long_only": {"cost_adjusted_net_pnl_usd": 0.5},
            "short_only": {"cost_adjusted_net_pnl_usd": 0.25},
        },
        "comparisons": {"best_nonflat_actor": "momentum"},
    }

    monkeypatch.setattr(experiments, "_build_feature_frame", lambda frame: _fake_feature_frame(frame))
    monkeypatch.setattr(
        experiments,
        "_candidate_replay_report",
        lambda *args, **kwargs: {
            "actor_summaries": {
                "candidate": {
                    "cost_adjusted_net_pnl_usd": 5.0,
                    "max_drawdown_frac": 0.05,
                    "exposure_by_symbol_usd": {"BTCUSDT": 100.0, "ETHUSDT": 100.0},
                },
                "flat": {"cost_adjusted_net_pnl_usd": 0.0},
            }
        },
    )
    monkeypatch.setattr(experiments, "_benchmark_replay_report", lambda frame: benchmark_report)
    monkeypatch.setattr(
        experiments,
        "train",
        lambda X, y, horizon, sample_weight=None, params_override=None: SimpleNamespace(
            feature_names=list(X.columns),
            feature_dtypes={str(column): str(dtype) for column, dtype in X.dtypes.items()},
            artifact_manifest={},
        ),
    )
    monkeypatch.setattr(experiments, "predict_proba", lambda model, X: np.full(len(X), 0.25))
    monkeypatch.setattr(
        experiments,
        "save_model_bundle",
        lambda model, path, metadata=None: (
            Path(path).write_text("model", encoding="utf-8"),
            Path(path).with_suffix(".manifest.json").write_text("{}", encoding="utf-8"),
        ),
    )
    monkeypatch.setattr(
        experiments,
        "select_threshold_by_utility",
        lambda *args, **kwargs: {
            "source": "economic_utility",
            "selected_threshold": 0.90,
            "selected_score": 1.0,
            "selected_expectancy_bps": 1.0,
            "selected_actionable": 1,
            "accuracy_at_selected": 1.0,
            "accuracy_optimal_threshold": 0.50,
            "accuracy_optimal_accuracy": 1.0,
            "thresholds": [],
            "config": {},
        },
    )

    config = RecoveryCandidateConfig(
        horizon=4,
        training_window_months=3,
        recency_half_life_days=30,
        dead_zone_bps=0.001,
        feature_set="full",
        label_mode="trade_outcome",
        trade_outcome_side="long",
        trade_outcome_round_trip_cost_bps=0.0,
    )
    evaluation = experiments._evaluate_candidate(
        raw,
        config=config,
        benchmark_report=benchmark_report,
        run_dir=tmp_path / "run",
        min_accuracy=0.60,
        min_actionable_decisions=1,
        max_drawdown_frac=0.25,
    )

    holdout_report = evaluation.holdout_report
    selection = evaluation.selection_risk_summary

    assert holdout_report["label_mode"] == "trade_outcome"
    assert holdout_report["trade_outcome_side"] == "long"
    assert holdout_report["predicted_take_count"] == 0
    assert holdout_report["predicted_skip_count"] == holdout_report["sample_count"]
    assert holdout_report["predicted_hold_count"] == holdout_report["sample_count"]
    assert holdout_report["predicted_buy_count"] == 0
    assert holdout_report["predicted_sell_count"] == 0
    assert holdout_report["cost_adjusted_expectancy_bps"] == pytest.approx(0.0)
    assert holdout_report["gross_expectancy_bps"] == pytest.approx(0.0)
    assert selection["take_share"] == pytest.approx(0.0)
    assert selection["skip_share"] == pytest.approx(1.0)
    assert selection["one_sided_take_collapse"] is True


def test_replay_manifest_is_compact() -> None:
    fill = ReplayFill(
        actor="candidate",
        timestamp="2024-01-01T00:00:00+00:00",
        symbol="BTCUSDT",
        side="BUY",
        requested_qty=0.1,
        filled_qty=0.1,
        price=100.0,
        fee_usd=0.01,
        slippage_usd=0.02,
        outcome="NEW_FILL",
    )
    actor_result = ReplayActorResult(
        actor="candidate",
        metrics={"net_pnl_usd": 1.0, "fill_count": 1},
        equity_curve=[{"timestamp": fill.timestamp, "equity_usd": 1001.0, "positions": {"BTCUSDT": 0.1}}],
        fills=[fill],
        blocked_intents=[{"timestamp": fill.timestamp, "reason": "test"}],
        risk_transitions=[],
        reconciliation={},
        manifest={"actor": "candidate"},
        state_digest="state",
    )
    replay = SimpleNamespace(
        replay_digest="digest",
        manifest={"scenario": "test"},
        timestamp_count=1,
        event_count=2,
        actors={"candidate": actor_result},
    )

    manifest = experiments._replay_manifest(replay)

    assert manifest["actors"]["candidate"]["fill_count"] == 1
    assert manifest["actors"]["candidate"]["equity_curve_points"] == 1
    assert "equity_curve" not in manifest["actors"]["candidate"]
    assert "fills" not in manifest["actors"]["candidate"]


def test_summarize_replay_actor_caps_reported_fills() -> None:
    timestamp = "2024-01-01T00:00:00+00:00"
    fills = [
        ReplayFill(
            actor="candidate",
            timestamp=timestamp,
            symbol="BTCUSDT",
            side="BUY",
            requested_qty=0.1,
            filled_qty=0.1,
            price=100.0 + idx,
            fee_usd=0.01,
            slippage_usd=0.02,
            outcome="NEW_FILL",
        )
        for idx in range(3)
    ]
    actor_result = ReplayActorResult(
        actor="candidate",
        metrics={"net_pnl_usd": 1.0, "max_drawdown_frac": -0.01, "fill_count": 3},
        equity_curve=[{"timestamp": timestamp, "equity_usd": 1001.0, "positions": {"BTCUSDT": 0.5}}],
        fills=fills,
        blocked_intents=[],
        risk_transitions=[],
        reconciliation={},
        manifest={"actor": "candidate"},
        state_digest="state",
    )
    dataset = _raw_dataset(rows=4)

    summary = experiments._summarize_replay_actor(actor_result, dataset, max_report_fills=2)

    assert summary["fill_count"] == 3
    assert summary["fills_payload_count"] == 2
    assert summary["fills_truncated"] is True
    assert len(summary["fills"]) == 2


def test_tail_bars_per_symbol_keeps_recent_rows_per_symbol() -> None:
    dataset = _raw_dataset(rows=8)

    trimmed = experiments._tail_bars_per_symbol(dataset, max_bars_per_symbol=3)

    assert len(trimmed) == 6
    for symbol, symbol_frame in trimmed.groupby(level="symbol", sort=False):
        source_symbol = dataset.xs(symbol, level="symbol")
        assert len(symbol_frame) == 3
        assert list(symbol_frame.index.get_level_values("timestamp")) == list(source_symbol.tail(3).index)


def test_pre_holdout_training_mask_respects_candidate_window() -> None:
    raw = _raw_dataset(rows=210)
    candidate_frame = _fake_feature_frame(raw).loc[:, ["roc_1", "atr_14", "funding_rate"]]
    holdout_start = pd.Timestamp("2024-06-01T00:00:00Z")
    timestamps = pd.DatetimeIndex(candidate_frame.index.get_level_values("timestamp"))
    holdout_indices = np.flatnonzero(np.asarray(timestamps >= holdout_start))

    mask = experiments._pre_holdout_training_mask(
        candidate_frame.index,
        holdout_start=holdout_start,
        training_window_months=3,
        exclude_indices=holdout_indices,
    )

    selected_timestamps = pd.DatetimeIndex(candidate_frame.index[mask].get_level_values("timestamp"))
    assert selected_timestamps.min() >= holdout_start - pd.DateOffset(months=3)
    assert selected_timestamps.max() < holdout_start
    assert not np.intersect1d(np.flatnonzero(mask), holdout_indices).size


def test_candidate_plan_keeps_holdout_after_validation() -> None:
    raw = _raw_dataset()
    monkey = _fake_feature_frame(raw)
    plan = experiments.build_temporal_validation_plan(
        monkey.loc[:, ["roc_1", "atr_14", "funding_rate"]],
        training_windows_months=(3,),
        expanding_included=True,
        test_window_months=1,
        holdout_months=2,
        purge_bars=24,
        min_train_rows=10,
    )

    assert plan.holdout_start is not None
    assert plan.holdout_end is not None
    for fold in plan.folds:
        assert fold.train_end < fold.test_start
        assert fold.test_end < plan.holdout_start


def test_validation_fold_selection_prefers_latest_candidate_window() -> None:
    raw = _raw_dataset(rows=360)
    monkey = _fake_feature_frame(raw)
    plan = experiments.build_temporal_validation_plan(
        monkey.loc[:, ["roc_1", "atr_14", "funding_rate"]],
        training_windows_months=(3,),
        expanding_included=True,
        test_window_months=1,
        holdout_months=2,
        purge_bars=24,
        min_train_rows=10,
    )

    selected = experiments._select_validation_folds(
        plan.folds,
        preferred_variant="rolling_3m",
        max_folds=3,
    )

    assert len(selected) == 3
    assert all(fold.variant == "rolling_3m" for fold in selected)
    assert [fold.test_start for fold in selected] == sorted(fold.test_start for fold in selected)
    assert selected[-1].test_end < plan.holdout_start


def test_evaluate_candidate_writes_paper_quarantine_artifacts(tmp_path: Path, monkeypatch) -> None:
    raw = _raw_dataset()
    benchmark_report = {
        "actor_summaries": {
            "flat": {"cost_adjusted_net_pnl_usd": 0.0},
            "momentum": {"cost_adjusted_net_pnl_usd": 3.0},
            "mean_reversion": {"cost_adjusted_net_pnl_usd": 2.0},
            "volatility_filtered": {"cost_adjusted_net_pnl_usd": 1.5},
        },
        "comparisons": {"best_nonflat_actor": "momentum"},
    }
    replay_frames = []

    monkeypatch.setattr(experiments, "_build_feature_frame", lambda frame: _fake_feature_frame(frame))
    def _fake_candidate_replay(frame, *args, **kwargs):
        replay_frames.append(frame)
        return {
            "actor_summaries": {
                "candidate": {
                    "cost_adjusted_net_pnl_usd": 5.0,
                    "max_drawdown_frac": 0.005,
                    "fill_count": 12,
                    "exposure_by_symbol_usd": {"BTCUSDT": 100.0, "ETHUSDT": 100.0},
                },
                "flat": {"cost_adjusted_net_pnl_usd": 0.0, "fill_count": 0},
            }
        }

    monkeypatch.setattr(experiments, "_candidate_replay_report", _fake_candidate_replay)
    monkeypatch.setattr(experiments, "_benchmark_replay_report", lambda frame: benchmark_report)
    monkeypatch.setattr(
        experiments,
        "train",
        lambda X, y, horizon, sample_weight=None, params_override=None: SimpleNamespace(
            feature_names=list(X.columns),
            feature_dtypes={str(column): str(dtype) for column, dtype in X.dtypes.items()},
            artifact_manifest={},
        ),
    )
    monkeypatch.setattr(experiments, "predict_proba", lambda model, X: np.full(len(X), 0.90))
    monkeypatch.setattr(
        experiments,
        "save_model_bundle",
        lambda model, path, metadata=None: (
            Path(path).write_text("model", encoding="utf-8"),
            Path(path).with_suffix(".manifest.json").write_text("{}", encoding="utf-8"),
        ),
    )

    config = RecoveryCandidateConfig(
        horizon=4,
        training_window_months=3,
        recency_half_life_days=30,
        dead_zone_bps=0.001,
        feature_set="full",
    )
    evaluation = experiments._evaluate_candidate(
        raw,
        config=config,
        benchmark_report=benchmark_report,
        run_dir=tmp_path / "run",
        min_accuracy=0.60,
        min_actionable_decisions=10,
        max_drawdown_frac=0.25,
    )

    candidate_dir = tmp_path / "run" / "candidates" / config.candidate_id()
    assert evaluation.passed is True
    assert evaluation.model_artifact_path is not None
    assert Path(evaluation.model_artifact_path).exists()
    assert (candidate_dir / "manifest.json").exists()
    assert json.loads((candidate_dir / "manifest.json").read_text(encoding="utf-8"))["pass_status"] is True
    assert evaluation.threshold_policy["source"] in {"oof_dev_predictions", "class_collapse"}
    assert evaluation.threshold_policy["selected_threshold"] == pytest.approx(0.50)
    assert replay_frames
    assert len(replay_frames[0]) < len(raw)


def test_evaluate_trade_outcome_candidate_uses_oof_aligned_economic_threshold_inputs(tmp_path: Path, monkeypatch) -> None:
    raw = _raw_dataset()
    benchmark_report = {
        "actor_summaries": {
            "flat": {"cost_adjusted_net_pnl_usd": 0.0},
            "momentum": {"cost_adjusted_net_pnl_usd": 3.0},
            "mean_reversion": {"cost_adjusted_net_pnl_usd": 2.0},
            "volatility_filtered": {"cost_adjusted_net_pnl_usd": 1.5},
        },
        "comparisons": {"best_nonflat_actor": "momentum"},
    }
    selector_calls: list[int] = []

    monkeypatch.setattr(experiments, "_build_feature_frame", lambda frame: _fake_feature_frame(frame))
    monkeypatch.setattr(
        experiments,
        "_candidate_replay_report",
        lambda *args, **kwargs: {
            "actor_summaries": {
                "candidate": {
                    "cost_adjusted_net_pnl_usd": 5.0,
                    "max_drawdown_frac": 0.05,
                    "exposure_by_symbol_usd": {"BTCUSDT": 100.0, "ETHUSDT": 100.0},
                },
                "flat": {"cost_adjusted_net_pnl_usd": 0.0},
            }
        },
    )
    monkeypatch.setattr(experiments, "_benchmark_replay_report", lambda frame: benchmark_report)
    monkeypatch.setattr(
        experiments,
        "train",
        lambda X, y, horizon, sample_weight=None, params_override=None: SimpleNamespace(
            feature_names=list(X.columns),
            feature_dtypes={str(column): str(dtype) for column, dtype in X.dtypes.items()},
            artifact_manifest={},
        ),
    )
    monkeypatch.setattr(experiments, "predict_proba", lambda model, X: np.full(len(X), 0.90))
    monkeypatch.setattr(
        experiments,
        "save_model_bundle",
        lambda model, path, metadata=None: (
            Path(path).write_text("model", encoding="utf-8"),
            Path(path).with_suffix(".manifest.json").write_text("{}", encoding="utf-8"),
        ),
    )

    def _assert_oof_aligned(probabilities, labels, forward_return_bps, *, symbols=None, fold_ids=None, config=None):
        assert len(probabilities) == len(labels) == len(forward_return_bps)
        assert symbols is not None and len(symbols) == len(probabilities)
        assert fold_ids is not None and len(fold_ids) == len(probabilities)
        selector_calls.append(len(probabilities))
        return {
            "source": "economic_utility",
            "selected_threshold": 0.50,
            "selected_score": 1.0,
            "selected_expectancy_bps": 1.0,
            "selected_actionable": len(probabilities),
            "accuracy_at_selected": 1.0,
            "accuracy_optimal_threshold": 0.50,
            "accuracy_optimal_accuracy": 1.0,
            "thresholds": [],
            "config": {},
        }

    monkeypatch.setattr(experiments, "select_threshold_by_utility", _assert_oof_aligned)

    config = RecoveryCandidateConfig(
        horizon=4,
        training_window_months=3,
        recency_half_life_days=30,
        dead_zone_bps=0.001,
        feature_set="full",
        label_mode="trade_outcome",
        trade_outcome_round_trip_cost_bps=0.0,
    )
    evaluation = experiments._evaluate_candidate(
        raw,
        config=config,
        benchmark_report=benchmark_report,
        run_dir=tmp_path / "run",
        min_accuracy=0.60,
        min_actionable_decisions=10,
        max_drawdown_frac=0.25,
    )

    assert selector_calls
    assert evaluation.threshold_policy["source"] == "economic_utility"


def test_run_model_recovery_experiments_remain_no_trade(tmp_path: Path, monkeypatch) -> None:
    raw = _raw_dataset()
    config = RecoveryCandidateConfig(
        horizon=4,
        training_window_months=3,
        recency_half_life_days=30,
        dead_zone_bps=0.001,
        feature_set="full",
    )

    monkeypatch.setattr(experiments, "_build_label_audit_report", lambda frame: {"cells": {}, "summary": {"cell_count": 0}})
    monkeypatch.setattr(experiments, "_benchmark_replay_report", lambda frame: {"actor_summaries": {"flat": {"cost_adjusted_net_pnl_usd": 0.0}, "momentum": {"cost_adjusted_net_pnl_usd": 1.0}}, "comparisons": {"best_nonflat_actor": "momentum"}})
    monkeypatch.setattr(experiments, "_candidate_grid", lambda *_args, **_kwargs: [config])
    monkeypatch.setattr(
        experiments,
        "_evaluate_candidate",
        lambda *args, **kwargs: CandidateEvaluation(
            config=config,
            candidate_id=config.candidate_id(),
            passed=False,
            score=-1.0,
            failure_reasons=("forced_fail",),
            dataset_manifest={},
            feature_manifest={},
            label_audit={},
            fold_ledger={},
            threshold_policy={},
            holdout_report={},
            replay_report={},
            selection_risk_summary={},
            model_artifact_path=None,
        ),
    )

    result = experiments.run_model_recovery_experiments(
        raw,
        snapshot_path=tmp_path / "snapshot.parquet",
        output_root=tmp_path / "out",
        docs_output_dir=tmp_path / "docs",
        max_candidates=1,
        no_production_registry=True,
    )

    assert result.summary["recommendation"] == "remain_no_trade"
    assert result.summary["selected_candidate_id"] is None
    assert (tmp_path / "docs" / "latest_experiment_summary.md").exists()


def test_run_model_recovery_experiments_trade_outcome_label_mode_passthrough(tmp_path: Path, monkeypatch) -> None:
    raw = _raw_dataset()
    config = RecoveryCandidateConfig(
        horizon=4,
        training_window_months=3,
        recency_half_life_days=30,
        dead_zone_bps=0.001,
        feature_set="full",
        label_mode="trade_outcome",
    )

    monkeypatch.setattr(experiments, "_build_trade_outcome_label_audit_report", lambda *args, **kwargs: {"cells": {}, "summary": {"cell_count": 0, "best_cell_stats": {"horizon": 4, "training_window_months": 3, "recency_half_life_days": 30, "dead_zone_bps": 0.001}, "recommended_dead_zone_floor_bps": 0.001}})
    monkeypatch.setattr(experiments, "_candidate_grid", lambda *_args, **_kwargs: [config])
    monkeypatch.setattr(
        experiments,
        "_evaluate_candidate",
        lambda *args, **kwargs: CandidateEvaluation(
            config=config,
            candidate_id=config.candidate_id(),
            passed=False,
            score=-1.0,
            failure_reasons=("forced_fail",),
            dataset_manifest={},
            feature_manifest={},
            label_audit={},
            fold_ledger={"mean_fold_accuracy": 0.0, "std_fold_accuracy": 0.0},
            threshold_policy={},
            accuracy_threshold_policy={},
            holdout_report={},
            replay_report={},
            selection_risk_summary={},
            model_artifact_path=None,
        ),
    )

    result = experiments.run_model_recovery_experiments(
        raw,
        snapshot_path=tmp_path / "snapshot.parquet",
        output_root=tmp_path / "out",
        docs_output_dir=tmp_path / "docs",
        max_candidates=1,
        label_mode="trade_outcome",
        no_production_registry=True,
    )

    assert result.summary["label_mode"] == "trade_outcome"
    assert result.summary["recommendation"] == "remain_no_trade"


def test_run_model_recovery_experiments_paper_quarantine_candidate(tmp_path: Path, monkeypatch) -> None:
    raw = _raw_dataset()
    pass_config = RecoveryCandidateConfig(
        horizon=4,
        training_window_months=3,
        recency_half_life_days=30,
        dead_zone_bps=0.001,
        feature_set="full",
    )
    fail_config = RecoveryCandidateConfig(
        horizon=2,
        training_window_months=6,
        recency_half_life_days=60,
        dead_zone_bps=0.002,
        feature_set="no_orderbook_placeholders",
    )

    monkeypatch.setattr(experiments, "_build_label_audit_report", lambda frame: {"cells": {}, "summary": {"cell_count": 0}})
    monkeypatch.setattr(experiments, "_benchmark_replay_report", lambda frame: {"actor_summaries": {"flat": {"cost_adjusted_net_pnl_usd": 0.0}, "momentum": {"cost_adjusted_net_pnl_usd": 1.0}}, "comparisons": {"best_nonflat_actor": "momentum"}})
    monkeypatch.setattr(experiments, "_candidate_grid", lambda *_args, **_kwargs: [pass_config, fail_config])

    def _evaluate(_frame, *, config, **_kwargs):
        if config == pass_config:
            return CandidateEvaluation(
                config=config,
                candidate_id=config.candidate_id(),
                passed=True,
                score=10.0,
                failure_reasons=(),
                dataset_manifest={},
                feature_manifest={},
                label_audit={},
                fold_ledger={"mean_fold_accuracy": 0.70, "std_fold_accuracy": 0.05},
                threshold_policy={},
                holdout_report={"holdout_accuracy": 0.75},
                replay_report={"actor_summaries": {"candidate": {"cost_adjusted_net_pnl_usd": 5.0}}},
                selection_risk_summary={},
                benchmark_delta_report={
                    "overall": {
                        "candidate_minus_best_nonflat_bps": 15.0,
                        "candidate_minus_same_side_bps": 12.0,
                    },
                    "symbol_pruning": {"allowed_symbol_count": 2, "rejected_symbol_count": 0},
                },
                candidate_quality_report={
                    "overall_decision": "pass",
                    "passed": True,
                    "evidence_digest": "digest-pass",
                    "summary": {"overall_decision": "pass", "hard_failed_rules": [], "failed_rules": []},
                    "rule_results": [],
                },
                symbol_pruning_report={"allowed_symbol_count": 2, "rejected_symbol_count": 0},
                replay_gap_diagnostics={"gap_bps": 0.0},
                maintenance_report={"decayed": False, "no_trade_required": False, "proven_shadow_version_id": None, "blockers": []},
                model_artifact_path=str(tmp_path / "candidate.pkl"),
            )
        return CandidateEvaluation(
            config=config,
            candidate_id=config.candidate_id(),
            passed=False,
            score=1.0,
            failure_reasons=("forced_fail",),
            dataset_manifest={},
            feature_manifest={},
            label_audit={},
            fold_ledger={},
            threshold_policy={},
            holdout_report={"holdout_accuracy": 0.50},
            replay_report={"actor_summaries": {"candidate": {"cost_adjusted_net_pnl_usd": -1.0}}},
            selection_risk_summary={},
            benchmark_delta_report={
                "overall": {
                    "candidate_minus_best_nonflat_bps": -5.0,
                    "candidate_minus_same_side_bps": -8.0,
                },
                "symbol_pruning": {"allowed_symbol_count": 0, "rejected_symbol_count": 2},
            },
            candidate_quality_report={
                "overall_decision": "fail",
                "passed": False,
                "evidence_digest": "digest-fail",
                "summary": {"overall_decision": "fail", "hard_failed_rules": ["best_nonflat_benchmark_delta"], "failed_rules": ["best_nonflat_benchmark_delta"]},
                "rule_results": [],
            },
            symbol_pruning_report={"allowed_symbol_count": 0, "rejected_symbol_count": 2},
            replay_gap_diagnostics={"gap_bps": 100.0},
            maintenance_report={"decayed": False, "no_trade_required": False, "proven_shadow_version_id": None, "blockers": []},
            model_artifact_path=None,
        )

    monkeypatch.setattr(experiments, "_evaluate_candidate", _evaluate)

    result = experiments.run_model_recovery_experiments(
        raw,
        snapshot_path=tmp_path / "snapshot.parquet",
        output_root=tmp_path / "out",
        docs_output_dir=tmp_path / "docs",
        max_candidates=2,
        no_production_registry=True,
    )

    assert result.summary["recommendation"] == "paper_quarantine_candidate"
    assert result.summary["selected_candidate_id"] == pass_config.candidate_id()
    assert result.summary["candidate_quality_summary"]["evaluated_candidates"] == 2
    assert result.summary["candidate_quality_summary"]["passed_quality"] == 1
    assert (tmp_path / "docs" / "latest_experiment_summary.md").exists()
