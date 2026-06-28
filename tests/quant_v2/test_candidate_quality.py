from __future__ import annotations

from quant_v2.research.candidate_quality import (
    BestNonFlatBenchmarkDeltaRule,
    CandidateQualityRule,
    DecayMaintenanceRule,
    MinimumBenchmarkMarginRule,
    PositiveAbsoluteExpectancyRule,
    ReplayAgreementRule,
    SameSideBenchmarkDeltaRule,
    SelectionRiskRule,
    SymbolCoverageRule,
    RegimeCoverageRule,
    default_candidate_quality_rules,
    evaluate_candidate_quality,
)


def _base_report() -> dict[str, object]:
    return {
        "candidate_id": "cand_1",
        "benchmark_delta_report": {
            "overall": {
                "candidate_pnl_usd": 18.0,
                "candidate_minus_best_nonflat_pnl_usd": 6.0,
                "candidate_minus_same_side_pnl_usd": 4.0,
                "candidate_minus_best_nonflat_bps": 12.0,
                "candidate_minus_same_side_bps": 9.0,
            },
            "symbol_pruning": {
                "allowed_symbols": ["BTCUSDT", "ETHUSDT"],
                "rejected_symbols": [],
                "rejected_symbol_count": 0,
                "by_symbol": {
                    "BTCUSDT": {"take_count": 20},
                    "ETHUSDT": {"take_count": 13},
                },
            },
            "regime_coverage": {
                "available_bucket_count": 4,
                "unknown_bucket_count": 0,
                "total_bucket_count": 4,
            },
        },
        "selection_risk_summary": {
            "passed": True,
            "failure_reasons": [],
            "pbo_proxy": 0.1,
            "fold_accuracy_std": 0.05,
            "symbol_concentration_share": 0.2,
        },
        "replay_gap_diagnostics": {
            "gap_bps": 4.5,
            "row_level_expectancy_bps": 11.0,
            "replay_net_return_bps": 6.5,
        },
        "maintenance_report": {
            "decayed": False,
            "no_trade_required": False,
            "proven_shadow_version_id": None,
            "blockers": [],
        },
    }


def test_candidate_quality_passes_when_all_hard_rules_pass() -> None:
    decision = evaluate_candidate_quality(_base_report())

    assert decision.passed is True
    assert decision.overall_decision == "pass"
    assert decision.evidence_digest
    assert len(decision.rule_results) >= 8


def test_candidate_quality_fails_closed_on_missing_benchmark_metrics() -> None:
    report = _base_report()
    del report["benchmark_delta_report"]["overall"]["candidate_minus_best_nonflat_pnl_usd"]

    decision = evaluate_candidate_quality(report)

    assert decision.passed is False
    assert decision.overall_decision == "fail"
    assert any(result.rule_name == "best_nonflat_benchmark_delta" and not result.passed for result in decision.rule_results)


def test_candidate_quality_watch_when_only_regime_coverage_is_soft_fail() -> None:
    report = _base_report()
    report["benchmark_delta_report"]["regime_coverage"] = {
        "available_bucket_count": 1,
        "unknown_bucket_count": 3,
        "total_bucket_count": 4,
    }

    decision = evaluate_candidate_quality(report)

    assert decision.passed is False
    assert decision.overall_decision == "watch"
    assert any(result.rule_name == "regime_coverage" and not result.passed for result in decision.rule_results)


def test_candidate_quality_fails_closed_on_missing_maintenance_context() -> None:
    report = _base_report()
    del report["maintenance_report"]

    decision = evaluate_candidate_quality(report)

    assert decision.passed is False
    assert decision.overall_decision == "fail"
    assert any(result.rule_name == "decay_maintenance" and not result.passed for result in decision.rule_results)


def test_candidate_quality_is_deterministic() -> None:
    report = _base_report()
    first = evaluate_candidate_quality(report)
    second = evaluate_candidate_quality(report)

    assert first.evidence_digest == second.evidence_digest
    assert first.summary == second.summary


def test_explicit_rule_thresholds_and_failures() -> None:
    report = _base_report()
    report["selection_risk_summary"]["passed"] = False
    report["selection_risk_summary"]["failure_reasons"] = ["fold_instability"]
    report["replay_gap_diagnostics"]["gap_bps"] = 80.0
    report["maintenance_report"]["decayed"] = True
    report["maintenance_report"]["proven_shadow_version_id"] = None

    rules: tuple[CandidateQualityRule, ...] = (
        PositiveAbsoluteExpectancyRule(),
        BestNonFlatBenchmarkDeltaRule(min_delta_pnl_usd=5.0),
        SameSideBenchmarkDeltaRule(min_delta_pnl_usd=3.0),
        MinimumBenchmarkMarginRule(min_margin_bps=5.0),
        SymbolCoverageRule(min_allowed_symbols=2, min_take_count=10),
        RegimeCoverageRule(min_available_buckets=3, max_unknown_bucket_share=0.25),
        SelectionRiskRule(),
        ReplayAgreementRule(max_abs_gap_bps=25.0),
        DecayMaintenanceRule(),
    )

    decision = evaluate_candidate_quality(report, rules)

    assert decision.passed is False
    assert decision.overall_decision == "fail"
    assert any(not result.passed for result in decision.rule_results if result.rule_name in {"selection_risk", "replay_agreement", "decay_maintenance"})
