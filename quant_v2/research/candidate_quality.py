"""Candidate quality ledger for benchmark-relative recovery decisions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
import hashlib
import json
from typing import Any, Mapping, Sequence

_MISSING = object()


def _canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def _sha256(payload: Any) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        if value is None:
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _safe_int(value: object, default: int = 0) -> int:
    try:
        if value is None:
            return int(default)
        return int(value)
    except Exception:
        return int(default)


def _nested_get(payload: Mapping[str, Any] | None, *path: str, default: Any = None) -> Any:
    current: Any = payload
    for key in path:
        if not isinstance(current, Mapping) or key not in current:
            return default
        current = current[key]
    return current


def _mapping_section(payload: Mapping[str, Any] | None, *path: str) -> Mapping[str, Any] | None:
    current: Any = payload
    for key in path:
        if not isinstance(current, Mapping) or key not in current:
            return None
        current = current[key]
    return current if isinstance(current, Mapping) else None


def _mapping_value(section: Mapping[str, Any] | None, key: str) -> Any:
    if not isinstance(section, Mapping) or key not in section:
        return _MISSING
    return section[key]


@dataclass(frozen=True)
class CandidateRuleResult:
    rule_name: str
    passed: bool
    severity: str
    reason: str
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CandidateQualityDecision:
    candidate_id: str
    overall_decision: str
    passed: bool
    rule_results: tuple[CandidateRuleResult, ...]
    evidence_digest: str
    summary: dict[str, Any] = field(default_factory=dict)


class CandidateQualityRule(ABC):
    name: str
    severity: str

    @abstractmethod
    def evaluate(self, candidate_report: Mapping[str, Any]) -> CandidateRuleResult:
        raise NotImplementedError


@dataclass(frozen=True)
class PositiveAbsoluteExpectancyRule(CandidateQualityRule):
    name: str = "positive_absolute_expectancy"
    severity: str = "hard"

    def evaluate(self, candidate_report: Mapping[str, Any]) -> CandidateRuleResult:
        overall = _mapping_section(candidate_report, "benchmark_delta_report", "overall")
        candidate_pnl_raw = _mapping_value(overall, "candidate_pnl_usd")
        candidate_pnl = _safe_float(candidate_pnl_raw, 0.0)
        missing_inputs = candidate_pnl_raw is _MISSING or overall is None
        passed = (not missing_inputs) and candidate_pnl > 0.0
        return CandidateRuleResult(
            rule_name=self.name,
            passed=passed,
            severity=self.severity,
            reason="candidate_pnl_usd>0" if passed else ("missing_candidate_pnl_usd" if missing_inputs else "candidate_pnl_usd<=0"),
            metrics={"candidate_pnl_usd": candidate_pnl},
        )


@dataclass(frozen=True)
class BestNonFlatBenchmarkDeltaRule(CandidateQualityRule):
    min_delta_pnl_usd: float = 0.0
    name: str = "best_nonflat_benchmark_delta"
    severity: str = "hard"

    def evaluate(self, candidate_report: Mapping[str, Any]) -> CandidateRuleResult:
        overall = _mapping_section(candidate_report, "benchmark_delta_report", "overall")
        delta_raw = _mapping_value(overall, "candidate_minus_best_nonflat_pnl_usd")
        delta = _safe_float(delta_raw, float("-inf"))
        missing_inputs = delta_raw is _MISSING or overall is None
        passed = (not missing_inputs) and delta > float(self.min_delta_pnl_usd)
        return CandidateRuleResult(
            rule_name=self.name,
            passed=passed,
            severity=self.severity,
            reason=(
                f"candidate_minus_best_nonflat_pnl_usd>{self.min_delta_pnl_usd}"
                if passed
                else ("missing_candidate_minus_best_nonflat_pnl_usd" if missing_inputs else "candidate_minus_best_nonflat_pnl_usd<=threshold")
            ),
            metrics={
                "candidate_minus_best_nonflat_pnl_usd": delta,
                "min_delta_pnl_usd": float(self.min_delta_pnl_usd),
            },
        )


@dataclass(frozen=True)
class SameSideBenchmarkDeltaRule(CandidateQualityRule):
    min_delta_pnl_usd: float = 0.0
    name: str = "same_side_benchmark_delta"
    severity: str = "hard"

    def evaluate(self, candidate_report: Mapping[str, Any]) -> CandidateRuleResult:
        overall = _mapping_section(candidate_report, "benchmark_delta_report", "overall")
        delta_raw = _mapping_value(overall, "candidate_minus_same_side_pnl_usd")
        delta = _safe_float(delta_raw, float("-inf"))
        missing_inputs = delta_raw is _MISSING or overall is None
        passed = (not missing_inputs) and delta > float(self.min_delta_pnl_usd)
        return CandidateRuleResult(
            rule_name=self.name,
            passed=passed,
            severity=self.severity,
            reason=(
                f"candidate_minus_same_side_pnl_usd>{self.min_delta_pnl_usd}"
                if passed
                else ("missing_candidate_minus_same_side_pnl_usd" if missing_inputs else "candidate_minus_same_side_pnl_usd<=threshold")
            ),
            metrics={
                "candidate_minus_same_side_pnl_usd": delta,
                "min_delta_pnl_usd": float(self.min_delta_pnl_usd),
            },
        )


@dataclass(frozen=True)
class MinimumBenchmarkMarginRule(CandidateQualityRule):
    min_margin_bps: float = 5.0
    name: str = "minimum_benchmark_margin"
    severity: str = "hard"

    def evaluate(self, candidate_report: Mapping[str, Any]) -> CandidateRuleResult:
        overall = _mapping_section(candidate_report, "benchmark_delta_report", "overall")
        best_nonflat_bps_raw = _mapping_value(overall, "candidate_minus_best_nonflat_bps")
        same_side_bps_raw = _mapping_value(overall, "candidate_minus_same_side_bps")
        best_nonflat_bps = _safe_float(best_nonflat_bps_raw, float("-inf"))
        same_side_bps = _safe_float(same_side_bps_raw, float("-inf"))
        missing_inputs = overall is None or best_nonflat_bps_raw is _MISSING or same_side_bps_raw is _MISSING
        passed = (not missing_inputs) and best_nonflat_bps >= float(self.min_margin_bps) and same_side_bps >= float(self.min_margin_bps)
        return CandidateRuleResult(
            rule_name=self.name,
            passed=passed,
            severity=self.severity,
            reason=(
                f"candidate_vs_benchmarks_margin>={self.min_margin_bps}"
                if passed
                else ("missing_benchmark_margin_inputs" if missing_inputs else "candidate_vs_benchmarks_margin_below_threshold")
            ),
            metrics={
                "candidate_minus_best_nonflat_bps": best_nonflat_bps,
                "candidate_minus_same_side_bps": same_side_bps,
                "min_margin_bps": float(self.min_margin_bps),
            },
        )


@dataclass(frozen=True)
class SymbolCoverageRule(CandidateQualityRule):
    min_allowed_symbols: int = 1
    min_take_count: int = 10
    name: str = "symbol_coverage"
    severity: str = "hard"

    def evaluate(self, candidate_report: Mapping[str, Any]) -> CandidateRuleResult:
        pruning = _mapping_section(candidate_report, "benchmark_delta_report", "symbol_pruning")
        if not isinstance(pruning, Mapping):
            return CandidateRuleResult(
                rule_name=self.name,
                passed=False,
                severity=self.severity,
                reason="missing_symbol_pruning",
                metrics={
                    "allowed_symbol_count": 0,
                    "rejected_symbol_count": 0,
                    "min_allowed_symbols": int(self.min_allowed_symbols),
                    "min_take_count": int(self.min_take_count),
                    "min_allowed_take_count": 0,
                },
            )
        allowed = list(pruning.get("allowed_symbols") or [])
        by_symbol = pruning.get("by_symbol") or {}
        take_counts = {
            str(symbol): _safe_int((payload or {}).get("take_count"), 0)
            for symbol, payload in (by_symbol.items() if isinstance(by_symbol, Mapping) else [])
        }
        allowed_take_min = min((take_counts.get(str(symbol), 0) for symbol in allowed), default=0)
        passed = len(allowed) >= int(self.min_allowed_symbols) and allowed_take_min >= int(self.min_take_count)
        return CandidateRuleResult(
            rule_name=self.name,
            passed=passed,
            severity=self.severity,
            reason="allowed_symbols_and_take_count_ok" if passed else "insufficient_allowed_symbols_or_take_count",
            metrics={
                "allowed_symbol_count": int(len(allowed)),
                "rejected_symbol_count": int(_safe_int(pruning.get("rejected_symbol_count"), 0)),
                "min_allowed_symbols": int(self.min_allowed_symbols),
                "min_take_count": int(self.min_take_count),
                "min_allowed_take_count": int(allowed_take_min),
            },
        )


@dataclass(frozen=True)
class RegimeCoverageRule(CandidateQualityRule):
    min_available_buckets: int = 3
    max_unknown_bucket_share: float = 0.35
    name: str = "regime_coverage"
    severity: str = "soft"

    def evaluate(self, candidate_report: Mapping[str, Any]) -> CandidateRuleResult:
        coverage = _mapping_section(candidate_report, "benchmark_delta_report", "regime_coverage")
        if not isinstance(coverage, Mapping):
            return CandidateRuleResult(
                rule_name=self.name,
                passed=False,
                severity=self.severity,
                reason="missing_regime_coverage",
                metrics={
                    "available_bucket_count": 0,
                    "unknown_bucket_count": 0,
                    "total_bucket_count": 0,
                    "unknown_bucket_share": 1.0,
                    "min_available_buckets": int(self.min_available_buckets),
                    "max_unknown_bucket_share": float(self.max_unknown_bucket_share),
                },
            )
        available = _safe_int(coverage.get("available_bucket_count"), 0)
        unknown = _safe_int(coverage.get("unknown_bucket_count"), 0)
        total = max(_safe_int(coverage.get("total_bucket_count"), 0), available + unknown)
        unknown_share = (unknown / total) if total else 1.0
        passed = available >= int(self.min_available_buckets) and unknown_share <= float(self.max_unknown_bucket_share)
        return CandidateRuleResult(
            rule_name=self.name,
            passed=passed,
            severity=self.severity,
            reason="regime_coverage_ok" if passed else "regime_coverage_incomplete",
            metrics={
                "available_bucket_count": int(available),
                "unknown_bucket_count": int(unknown),
                "total_bucket_count": int(total),
                "unknown_bucket_share": float(unknown_share),
                "min_available_buckets": int(self.min_available_buckets),
                "max_unknown_bucket_share": float(self.max_unknown_bucket_share),
            },
        )


@dataclass(frozen=True)
class SelectionRiskRule(CandidateQualityRule):
    name: str = "selection_risk"
    severity: str = "hard"

    def evaluate(self, candidate_report: Mapping[str, Any]) -> CandidateRuleResult:
        selection = _nested_get(candidate_report, "selection_risk_summary", default=_MISSING)
        if selection is _MISSING or not isinstance(selection, Mapping):
            return CandidateRuleResult(
                rule_name=self.name,
                passed=False,
                severity=self.severity,
                reason="missing_selection_risk_summary",
                metrics={
                    "passed": False,
                    "blockers": [],
                    "pbo_proxy": 0.0,
                    "fold_accuracy_std": 0.0,
                    "symbol_concentration_share": 0.0,
                },
            )
        passed = bool(selection.get("passed", False)) and not list(selection.get("failure_reasons") or [])
        blockers = list(selection.get("failure_reasons") or selection.get("blockers") or [])
        return CandidateRuleResult(
            rule_name=self.name,
            passed=passed,
            severity=self.severity,
            reason="selection_risk_passed" if passed else "selection_risk_blocked",
            metrics={
                "passed": bool(selection.get("passed", False)),
                "blockers": blockers,
                "pbo_proxy": _safe_float(selection.get("pbo_proxy"), 0.0),
                "fold_accuracy_std": _safe_float(selection.get("fold_accuracy_std"), 0.0),
                "symbol_concentration_share": _safe_float(selection.get("symbol_concentration_share"), 0.0),
            },
        )


@dataclass(frozen=True)
class ReplayAgreementRule(CandidateQualityRule):
    max_abs_gap_bps: float = 25.0
    name: str = "replay_agreement"
    severity: str = "hard"

    def evaluate(self, candidate_report: Mapping[str, Any]) -> CandidateRuleResult:
        replay_gap = _mapping_section(candidate_report, "replay_gap_diagnostics")
        if not isinstance(replay_gap, Mapping):
            return CandidateRuleResult(
                rule_name=self.name,
                passed=False,
                severity=self.severity,
                reason="missing_replay_gap_diagnostics",
                metrics={
                    "gap_bps": float("inf"),
                    "max_abs_gap_bps": float(self.max_abs_gap_bps),
                    "row_level_expectancy_bps": 0.0,
                    "replay_net_return_bps": 0.0,
                },
            )
        gap_bps = _safe_float(replay_gap.get("gap_bps"), float("inf"))
        passed = abs(gap_bps) <= float(self.max_abs_gap_bps)
        return CandidateRuleResult(
            rule_name=self.name,
            passed=passed,
            severity=self.severity,
            reason="replay_gap_within_tolerance" if passed else "replay_gap_exceeds_tolerance",
            metrics={
                "gap_bps": gap_bps,
                "max_abs_gap_bps": float(self.max_abs_gap_bps),
                "row_level_expectancy_bps": _safe_float(replay_gap.get("row_level_expectancy_bps"), 0.0),
                "replay_net_return_bps": _safe_float(replay_gap.get("replay_net_return_bps"), 0.0),
            },
        )


@dataclass(frozen=True)
class DecayMaintenanceRule(CandidateQualityRule):
    name: str = "decay_maintenance"
    severity: str = "hard"

    def evaluate(self, candidate_report: Mapping[str, Any]) -> CandidateRuleResult:
        maintenance = _mapping_section(candidate_report, "maintenance_report")
        if not isinstance(maintenance, Mapping) or not maintenance:
            return CandidateRuleResult(
                rule_name=self.name,
                passed=False,
                severity=self.severity,
                reason="missing_maintenance_context",
                metrics={"maintenance_context": False, "decayed": True, "no_trade_required": True},
            )
        decayed = bool(maintenance.get("decayed", False))
        proven_shadow = maintenance.get("proven_shadow_version_id")
        blockers = list(maintenance.get("blockers") or [])
        no_trade_required = bool(maintenance.get("no_trade_required", decayed))
        passed = (not decayed) or bool(proven_shadow)
        return CandidateRuleResult(
            rule_name=self.name,
            passed=passed,
            severity=self.severity,
            reason="maintenance_ok" if passed else "decayed_without_proven_shadow",
            metrics={
                "decayed": decayed,
                "no_trade_required": no_trade_required,
                "proven_shadow_version_id": proven_shadow,
                "blockers": blockers,
            },
        )


def default_candidate_quality_rules(
    *,
    min_margin_bps: float = 5.0,
    min_allowed_symbols: int = 1,
    min_take_count: int = 10,
    max_abs_gap_bps: float = 25.0,
) -> tuple[CandidateQualityRule, ...]:
    return (
        PositiveAbsoluteExpectancyRule(),
        BestNonFlatBenchmarkDeltaRule(),
        SameSideBenchmarkDeltaRule(),
        MinimumBenchmarkMarginRule(min_margin_bps=min_margin_bps),
        SymbolCoverageRule(min_allowed_symbols=min_allowed_symbols, min_take_count=min_take_count),
        RegimeCoverageRule(),
        SelectionRiskRule(),
        ReplayAgreementRule(max_abs_gap_bps=max_abs_gap_bps),
        DecayMaintenanceRule(),
    )


def evaluate_candidate_quality(
    candidate_report: Mapping[str, Any],
    rules: Sequence[CandidateQualityRule] | None = None,
) -> CandidateQualityDecision:
    candidate_id = str(candidate_report.get("candidate_id") or candidate_report.get("config", {}).get("candidate_id") or "")
    active_rules = tuple(rules or default_candidate_quality_rules())
    rule_results = tuple(rule.evaluate(candidate_report) for rule in active_rules)

    hard_failures = [result for result in rule_results if not result.passed and result.severity == "hard"]
    soft_failures = [result for result in rule_results if not result.passed and result.severity != "hard"]
    if hard_failures:
        overall_decision = "fail"
    elif soft_failures:
        overall_decision = "watch"
    else:
        overall_decision = "pass"
    passed = overall_decision == "pass"

    summary = {
        "candidate_id": candidate_id,
        "overall_decision": overall_decision,
        "passed": passed,
        "hard_failure_count": len(hard_failures),
        "soft_failure_count": len(soft_failures),
        "failed_rules": [result.rule_name for result in rule_results if not result.passed],
        "hard_failed_rules": [result.rule_name for result in hard_failures],
        "soft_failed_rules": [result.rule_name for result in soft_failures],
        "top_failure_reasons": [result.reason for result in rule_results if not result.passed][:5],
        "rule_names": [result.rule_name for result in rule_results],
    }
    evidence_digest = _sha256(
        {
            "candidate_id": candidate_id,
            "summary": summary,
            "rule_results": [asdict(result) for result in rule_results],
        }
    )
    return CandidateQualityDecision(
        candidate_id=candidate_id,
        overall_decision=overall_decision,
        passed=passed,
        rule_results=rule_results,
        evidence_digest=evidence_digest,
        summary=summary,
    )
