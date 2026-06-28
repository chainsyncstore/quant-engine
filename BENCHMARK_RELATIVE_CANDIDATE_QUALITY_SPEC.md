# Benchmark-Relative Candidate Quality Spec

**Spec date:** 2026-06-28
**Implementer profile:** GPT-5.4 mini orchestrating spawned implementation agents
**Audit owner:** GPT-5.5 lead/audit agent
**Design reference:** legacy `origin/master` hypothesis promotion pipeline (`promotion/*`, `hypotheses/*`)
**Current runtime disposition:** **NO PRODUCTION TRADING**

## 1. Objective

Add a legacy-inspired, crypto-specific candidate quality layer to the model recovery pipeline.

The implementation must stop treating candidate quality as only:

```text
Is the model profitable after costs?
```

and must instead answer:

```text
Where, and under what regimes, does this model beat the simple transparent baseline it would replace?
```

The output must be an auditable rule ledger for every candidate. The ledger must give explicit pass/fail reasons for absolute expectancy, benchmark-relative edge, regime coverage, symbol coverage, selection risk, replay agreement, and active-model decay.

This spec uses the old FX-era hypothesis promotion design only as a pattern:

- explicit rule objects;
- explicit decisions;
- explicit reasons;
- explicit candidate states;
- explicit decay and maintenance checks.

Do **not** reintroduce FX execution logic, MetaTrader/MQL5 logic, competition "hail mary" logic, or old unsafe auto-promotion behavior.

## 2. Non-Negotiable Constraints

- Do not resume live trading.
- Do not resume demo/paper trading as part of this spec unless a later lead/audit instruction explicitly says to run a paper soak.
- Do not write to production registry from recovery experiments.
- Keep recovery runs under no-production-registry mode.
- Do not weaken existing gates to force a model pass.
- Do not delete historical models, registry records, or audit artifacts.
- Do not copy legacy FX strategies into production or recovery replay.
- Do not use raw accuracy as a promotion reason by itself.
- Do not allow a candidate to pass because it beats a weak incumbent while losing to a transparent benchmark.
- Do not allow active-model decay to promote an unproven shadow model.
- Do not bypass existing approval, artifact manifest, hard-risk, or two-person promotion controls.
- Do not commit secrets, `.env`, exchange keys, chat IDs, hostnames, IPs, database dumps, or private operational credentials.
- Preserve deterministic tests where deterministic output already exists.

## 3. Required Sub-Agent Workflow

5.4 mini must orchestrate one phase at a time. A sub-agent may not start the next phase until the lead/audit agent accepts the previous phase.

Every implementation sub-agent must return this exact end note:

```text
PHASE:
STATUS: completed | blocked | failed
FILES CHANGED:
COMMANDS RUN:
RESULTS:
ARTIFACTS:
RISKS / OPEN QUESTIONS:
ROLLBACK NOTES:
RECOMMENDED NEXT ACTION:
```

The lead/audit agent must then respond with this exact audit block:

```text
LEAD AUDIT DECISION: accept | reject
ACCEPTANCE EVIDENCE:
REJECTION REASON:
CORRECTION INSTRUCTIONS:
AUDIT_REPORT_UPDATE:
```

No phase is accepted until:

- targeted tests pass;
- generated artifacts are inspectable and deterministic enough for audit;
- `audit_report.md` has a dated checkpoint for the accepted phase.

## 4. Preflight

Run locally before Phase 1:

```powershell
git status --short
git rev-parse HEAD
rg -n "class RecoveryCandidateConfig|class CandidateEvaluation|def run_model_recovery_experiments|benchmark|replay_report|holdout_report|selection_risk" quant_v2\research\model_recovery_experiments.py
rg -n "class EvaluationPolicy|def decide_promotion|record_shadow_decision|paper_evaluation|promotion_eligible" quant_v2\research\model_evaluator.py quant_v2\model_registry.py
rg -n "class ScoreInputs|GateInputs|compute_system_score|evaluate_claim_70_plus_gates" quant_v2\research\scorecard.py
rg -n "regime_|funding|drawdown|btc|symbol" quant_v2 quant tests -S
```

Record the findings in the Phase 1 end note. Do not modify files during preflight except `audit_report.md` if the lead explicitly asks for a checkpoint.

## 5. Definitions

### Candidate

A candidate is one evaluated recovery model configuration, including:

- horizon;
- training window;
- recency half-life;
- feature set;
- label mode;
- trade outcome side;
- threshold policy;
- symbol whitelist or all-symbol mode;
- regime gate or global mode.

### Transparent Benchmarks

At minimum, the quality layer must support these benchmark actors:

- `flat`: no trades;
- `moving_average_trend`: transparent trend-following baseline;
- `adverse_excursion_exit`: simple entry with bounded adverse excursion exit;
- `long_only`: same-side long baseline where applicable;
- `short_only`: same-side short baseline where applicable;
- `best_nonflat`: best benchmark excluding `flat` for the relevant slice;
- `regime_best`: best transparent benchmark active in the evaluated regime slice.

If some benchmark actors already exist under different names, adapt to the existing names but preserve the semantic roles above in the report.

### Relevant Benchmark

For each candidate decision:

- long-side model must beat `flat`, `best_nonflat`, and the same-side long baseline if present;
- short-side model must beat `flat`, `best_nonflat`, and `short_only`;
- regime-gated model must beat the best transparent benchmark inside the regime where it is allowed to trade;
- symbol-gated model must beat the relevant benchmark per included symbol.

### Proven Shadow Candidate

A candidate is "proven shadow" only when all of these are true:

- registry status allows promotion (`candidate`, not terminal);
- artifact directory exists and manifests validate;
- paper/shadow evaluation exists;
- paper/shadow evaluation has `promotion_eligible=true`;
- candidate quality ledger has `overall_decision=pass`;
- benchmark delta gates pass;
- same-side benchmark gates pass;
- regime/symbol gates pass for the exact allowed trading domain;
- selection-risk gates pass;
- no active hard-risk blockers exist;
- if two-person approval is enabled, matching unexpired approvals exist for the evidence digest.

Anything less than this is **not** a proven shadow candidate.

## 6. Phase 1 - Legacy-Inspired Candidate Rule Ledger

### Goal

Create a reusable, testable rule ledger that mirrors the old `PromotionRule` pattern without importing legacy FX code.

### Files To Add

- `quant_v2/research/candidate_quality.py`
- `tests/quant_v2/test_candidate_quality.py`

### Required API

Implement immutable or effectively immutable dataclasses:

```python
CandidateRuleResult
  rule_name: str
  passed: bool
  severity: str                 # "hard" | "soft" | "info"
  reason: str
  metrics: dict[str, Any]

CandidateQualityDecision
  candidate_id: str
  overall_decision: str          # "pass" | "fail" | "watch"
  passed: bool
  rule_results: tuple[CandidateRuleResult, ...]
  evidence_digest: str
  summary: dict[str, Any]

CandidateQualityRule
  name: str
  severity: str
  evaluate(candidate_report: Mapping[str, Any]) -> CandidateRuleResult
```

Implement:

```python
evaluate_candidate_quality(
    candidate_report: Mapping[str, Any],
    rules: Sequence[CandidateQualityRule] | None = None,
) -> CandidateQualityDecision
```

Required behavior:

- hard-rule failure makes `overall_decision="fail"`;
- no hard failures and at least one soft failure makes `overall_decision="watch"`;
- all hard and soft rules passing makes `overall_decision="pass"`;
- `evidence_digest` must be deterministic from the normalized decision payload;
- rule results must preserve order;
- missing metrics must fail closed for hard rules.

### Initial Rules

Implement these as concrete classes or factory-created rule objects:

- `PositiveAbsoluteExpectancyRule`
- `BestNonFlatBenchmarkDeltaRule`
- `SameSideBenchmarkDeltaRule`
- `MinimumBenchmarkMarginRule`
- `SymbolCoverageRule`
- `RegimeCoverageRule`
- `SelectionRiskRule`
- `ReplayAgreementRule`
- `DecayMaintenanceRule`

Rules may be implemented with placeholder-friendly metric lookups in Phase 1, but each rule must have tests proving pass/fail behavior.

### Acceptance Tests

Run:

```powershell
python -m pytest tests\quant_v2\test_candidate_quality.py -q
```

Expected:

- deterministic digest test passes;
- hard failure makes decision fail;
- soft failure makes decision watch;
- all-pass report makes decision pass;
- missing benchmark metrics fail closed.

## 7. Phase 2 - Benchmark Delta Report

### Goal

Add a `benchmark_delta_report` object to every `CandidateEvaluation`.

### Primary Files

- `quant_v2/research/model_recovery_experiments.py`
- `tests/quant_v2/test_model_recovery_experiments.py`
- optional: `tests/quant_v2/test_candidate_quality.py`

### Required Report Shape

Each candidate report must include:

```json
{
  "benchmark_delta_report": {
    "policy_version": "benchmark_delta_v1",
    "candidate_id": "...",
    "candidate_side": "long|short|directional",
    "threshold_source": "...",
    "overall": {
      "candidate_pnl_usd": 0.0,
      "candidate_return_bps": 0.0,
      "flat_pnl_usd": 0.0,
      "best_nonflat_actor": "...",
      "best_nonflat_pnl_usd": 0.0,
      "same_side_actor": "...",
      "same_side_pnl_usd": 0.0,
      "candidate_minus_flat_pnl_usd": 0.0,
      "candidate_minus_best_nonflat_pnl_usd": 0.0,
      "candidate_minus_same_side_pnl_usd": 0.0,
      "candidate_minus_best_nonflat_bps": 0.0,
      "candidate_minus_same_side_bps": 0.0,
      "take_count": 0,
      "fill_count": 0,
      "max_drawdown_bps": 0.0
    },
    "by_symbol": {},
    "by_regime": {},
    "rejected_symbols": [],
    "warnings": []
  }
}
```

Required calculations:

- compare candidate against `flat`;
- compare candidate against best non-flat benchmark;
- compare candidate against same-side benchmark;
- calculate candidate-minus-baseline in USD and bps;
- preserve benchmark actor names in the artifact;
- fail closed if required benchmark actor output is missing.

### Acceptance Tests

Run:

```powershell
python -m pytest tests\quant_v2\test_model_recovery_experiments.py tests\quant_v2\test_candidate_quality.py -q
```

Expected:

- every candidate evaluation exposes `benchmark_delta_report`;
- short-side candidate uses `short_only` as same-side baseline when present;
- long-side candidate does not get credit from inverse-side baseline behavior;
- missing non-flat benchmark adds warning and fails the corresponding quality rule.

## 8. Phase 3 - Regime And Symbol Slices

### Goal

Add regime-aware and symbol-aware candidate-vs-benchmark tables.

### Required Regime Slices

For every candidate, report performance by:

- broad market trend: `bullish`, `bearish`, `sideways`, `unknown`;
- volatility: `low`, `normal`, `high`, `unknown`;
- BTC 24h drawdown bucket: `none`, `mild`, `moderate`, `severe`, `unknown`;
- BTC 7d drawdown bucket: `none`, `mild`, `moderate`, `severe`, `unknown`;
- funding regime: `positive`, `neutral`, `negative`, `unknown`;
- symbol.

If an input column does not exist yet, derive the best available equivalent from existing features. If no safe equivalent exists, emit `unknown` and include a warning. Do not invent future-looking regime labels.

### Required Symbol Table

For each symbol:

```json
{
  "symbol": "SOLUSDT",
  "candidate_pnl_usd": 0.0,
  "benchmark_actor": "short_only",
  "benchmark_pnl_usd": 0.0,
  "candidate_minus_benchmark_pnl_usd": 0.0,
  "candidate_minus_benchmark_bps": 0.0,
  "take_count": 0,
  "fill_count": 0,
  "max_drawdown_bps": 0.0,
  "decision": "allow|reject",
  "reason": "..."
}
```

### Acceptance Tests

Run:

```powershell
python -m pytest tests\quant_v2\test_model_recovery_experiments.py tests\quant_v2\test_candidate_quality.py -q
```

Expected:

- per-symbol deltas exist;
- per-regime deltas exist;
- unknown regime labels are explicit, not omitted;
- candidate report answers: "where does this candidate beat flat, best_nonflat, and same-side benchmark?"

## 9. Phase 4 - Symbol-Level Pruning Before Replay

### Goal

Prevent candidates from trading symbols where validation says they do not beat the relevant benchmark.

### Required Behavior

Before final replay for a candidate:

1. Build provisional per-symbol benchmark deltas.
2. Build `allowed_symbols`.
3. Reject symbols that fail any hard symbol gate.
4. Re-run final candidate replay only on allowed symbols.
5. Preserve both pre-prune and post-prune reports.

Required rejection reasons:

- `negative_absolute_expectancy`;
- `does_not_beat_same_side_baseline`;
- `does_not_beat_best_nonflat`;
- `insufficient_take_count`;
- `drawdown_too_high`;
- `missing_benchmark`;
- `unknown_error`.

If no symbols survive pruning, the candidate must fail closed and must not produce a model artifact.

### Config

Add environment/CLI-configurable defaults without weakening current gates:

- `MODEL_RECOVERY_MIN_SYMBOL_TAKE_COUNT`, default `10`;
- `MODEL_RECOVERY_MIN_SYMBOL_BENCHMARK_MARGIN_BPS`, default `5.0`;
- `MODEL_RECOVERY_MAX_SYMBOL_DRAWDOWN_BPS`, default should use the stricter existing drawdown policy where available.

### Acceptance Tests

Run:

```powershell
python -m pytest tests\quant_v2\test_model_recovery_experiments.py tests\quant_v2\test_candidate_quality.py -q
```

Expected:

- symbol with negative benchmark delta is rejected;
- final replay excludes rejected symbols;
- no surviving symbols means candidate fails;
- report lists rejected symbols and reasons.

## 10. Phase 5 - Benchmark Delta Gates

### Goal

Make benchmark-relative edge a hard promotion/recovery gate.

### Required Hard Gates

A candidate must fail if any of these are false:

```text
candidate_minus_flat_pnl_usd > 0
candidate_minus_best_nonflat_pnl_usd > 0
candidate_minus_same_side_pnl_usd > 0
candidate_minus_best_nonflat_bps >= configured_margin
candidate_minus_same_side_bps >= configured_margin
```

Default margin:

```text
MODEL_RECOVERY_MIN_BENCHMARK_MARGIN_BPS=5.0
```

The gate must be reported by rule name in the candidate quality ledger.

### Required Integration

The recovery runner summary must include:

```json
{
  "candidate_quality_summary": {
    "evaluated_candidates": 0,
    "passed_quality": 0,
    "watch_quality": 0,
    "failed_quality": 0,
    "top_failure_reasons": []
  }
}
```

Every candidate row in the top-candidates report must include:

- `quality_decision`;
- `quality_evidence_digest`;
- first three failing hard rules;
- `candidate_minus_best_nonflat_bps`;
- `candidate_minus_same_side_bps`;
- `allowed_symbol_count`;
- `rejected_symbol_count`.

### Acceptance Tests

Run:

```powershell
python -m pytest tests\quant_v2\test_candidate_quality.py tests\quant_v2\test_model_recovery_experiments.py -q
```

Expected:

- candidate that beats flat but loses to `short_only` fails;
- candidate that beats benchmark by less than margin fails;
- quality summary counts pass/watch/fail correctly.

## 11. Phase 6 - Regime-Gated Candidate Families

### Goal

Generate and evaluate candidate variants that only trade in measured-edge regimes.

### Required Variant Families

Add regime-gated variants for trade-outcome candidates:

- `global`: existing behavior;
- `long_bullish_recovery_only`;
- `short_bearish_drawdown_only`;
- `high_vol_only`;
- `low_vol_only`;
- `funding_positive_only`;
- `funding_negative_only`;
- `symbol_whitelist_only`;
- `symbol_and_regime_whitelist`.

If a regime label cannot be derived, the gated variant must abstain or fail closed; it must not treat unknown as favorable.

### Required Report Fields

Each variant must include:

- `variant_id`;
- `variant_kind`;
- `allowed_regimes`;
- `blocked_regimes`;
- `allowed_symbols`;
- `rejected_symbols`;
- benchmark delta report;
- quality ledger.

### Acceptance Tests

Run:

```powershell
python -m pytest tests\quant_v2\test_model_recovery_experiments.py tests\quant_v2\test_candidate_quality.py -q
```

Expected:

- long bullish/recovery variant emits no short trades;
- short bearish/drawdown variant emits no long trades;
- unknown regimes are not treated as pass;
- gated variant can pass only inside its measured edge domain.

## 12. Phase 7 - Active-Model Decay And Maintenance Lifecycle

### Goal

Implement active-model decay detection without unsafe auto-promotion.

### Files

Prefer adding:

- `quant_v2/research/model_maintenance.py`
- `tests/quant_v2/test_model_maintenance.py`

Integrate only where necessary with:

- `quant_v2/research/model_evaluator.py`
- `quant_v2/model_registry.py`
- Telegram/admin surface only if an existing diagnostics command can safely show the result.

### Required Lifecycle

If active model decay is detected:

1. Mark active model as `maintenance_required` in diagnostics.
2. Block new trade entries.
3. Continue collecting signal/shadow evidence where safe.
4. Search registry for a proven shadow candidate.
5. If a proven shadow candidate exists, emit `recommended_action="promote_proven_shadow_via_approval_path"`.
6. If no proven shadow candidate exists, emit `recommended_action="remain_no_trade_and_trigger_recovery_retrain"`.
7. Trigger or recommend immediate recovery retrain, depending on available existing orchestration.
8. Never promote an unproven model.
9. Never bypass hard-risk pauses.
10. Never bypass approval-gated promotion.

### Required API

Implement:

```python
ModelMaintenanceDecision
  active_version_id: str
  decayed: bool
  no_trade_required: bool
  proven_shadow_version_id: str | None
  recommended_action: str
  blockers: tuple[str, ...]
  evidence_digest: str
  metrics: dict[str, Any]

evaluate_active_model_maintenance(
    active_record: Mapping[str, Any] | ModelVersionRecord,
    recent_evidence: Mapping[str, Any],
    candidate_records: Sequence[Mapping[str, Any] | ModelVersionRecord],
    *,
    hard_risk_pauses: int = 0,
    approval_required: bool = True,
) -> ModelMaintenanceDecision
```

### Decay Criteria

A model is decayed if any hard decay condition is true:

- recent net return after costs is below maintenance floor;
- recent drawdown exceeds maintenance ceiling;
- recent candidate-vs-benchmark delta is negative for active model;
- shadow/live drift is outside tolerance;
- active model fails same-side benchmark maintenance check;
- active model has insufficient recent sample and production risk state is elevated.

Default thresholds:

- `MODEL_MAINT_MIN_RECENT_NET_BPS=0.0`;
- `MODEL_MAINT_MAX_DRAWDOWN_BPS=100.0`;
- `MODEL_MAINT_MIN_BENCHMARK_DELTA_BPS=0.0`;
- `MODEL_MAINT_MAX_SHADOW_DRIFT_MAE=0.10`;
- `MODEL_MAINT_MIN_RECENT_ACTIONABLE=30`.

### Proven Shadow Promotion Rule

The maintenance evaluator may recommend a proven shadow candidate. It must not directly call `promote_version()` unless a separate existing operator-approved promotion command invokes it.

The recommended output must distinguish:

```text
decayed + proven shadow exists -> no-trade active, recommend governed promotion
decayed + no proven shadow     -> no-trade active, trigger/recommend recovery retrain
not decayed                    -> continue, no maintenance action
```

### Acceptance Tests

Run:

```powershell
python -m pytest tests\quant_v2\test_model_maintenance.py tests\quant_v2\test_model_registry.py tests\quant_v2\test_model_evaluator.py -q
```

Expected:

- decayed active model enters no-trade-required state;
- unproven shadow is not recommended;
- proven shadow is recommended but not auto-promoted;
- hard-risk pause blocks promotion recommendation;
- missing evidence fails closed into no-trade/watch, not live continuation.

## 13. Phase 8 - Long-Side Replay Gap Diagnostics

### Goal

Explain cases where row-level long-side expectancy is positive but executable replay PnL is negative.

### Required Diagnostics

For each candidate, emit:

- row-level expected return;
- replay realized PnL;
- candidate row-to-fill conversion rate;
- repeated-fill count by timestamp/symbol;
- average hold duration;
- losing hold duration;
- adverse excursion before exit;
- favorable excursion before exit;
- cost paid per completed trade;
- cost as percentage of gross edge;
- signal lag estimate;
- symbol concentration;
- sizing impact;
- top three reasons row-level and replay disagree.

### Required Report Shape

```json
{
  "replay_gap_diagnostics": {
    "policy_version": "replay_gap_v1",
    "row_level_expectancy_bps": 0.0,
    "replay_net_return_bps": 0.0,
    "gap_bps": 0.0,
    "cost_drag_bps": 0.0,
    "hold_duration_summary": {},
    "adverse_excursion_summary": {},
    "symbol_concentration": {},
    "top_gap_reasons": []
  }
}
```

### Acceptance Tests

Run:

```powershell
python -m pytest tests\quant_v2\test_model_recovery_experiments.py -q
```

Expected:

- diagnostics exist for long-side and short-side candidates;
- positive row expectancy with negative replay returns includes at least one gap reason;
- diagnostics are included in generated markdown/JSON artifacts.

## 14. Phase 9 - Reporting, Audit, And No-Production Rerun

### Goal

Make the new quality evidence visible and rerun recovery without touching production registry state.

### Required Report Updates

Update generated recovery markdown to include:

- candidate quality summary;
- top quality failure reasons;
- benchmark delta table;
- per-symbol table;
- per-regime table;
- rejected symbols;
- regime-gated variant table;
- replay gap diagnostics;
- maintenance/decay recommendation if active-model evidence is provided.

### Required Commands

Run locally:

```powershell
python -m pytest tests\quant_v2\test_candidate_quality.py tests\quant_v2\test_model_recovery_experiments.py tests\quant_v2\test_model_maintenance.py -q
python -m pytest tests\quant_v2\test_model_evaluator.py tests\quant_v2\test_model_registry.py -q
```

If local tests pass, the lead may ask for a 4arm no-production validation run. If so, run only no-production-registry commands and capture:

- command;
- git revision;
- container/runtime identity if available;
- run id;
- evaluated candidates;
- passed quality count;
- selected candidate id;
- recommendation;
- artifact paths.

### `audit_report.md`

Append a dated checkpoint after each accepted phase. The final checkpoint must state:

- whether any candidate passed;
- whether production remains blocked;
- whether a proven shadow candidate exists;
- whether active model maintenance recommends no-trade;
- next recommended action.

## 15. Final Acceptance Criteria

This spec is complete only when all are true:

- every candidate emits a candidate quality ledger;
- every candidate emits benchmark delta report;
- every candidate emits per-symbol and per-regime benchmark-relative tables;
- benchmark delta gates are hard gates;
- symbol pruning occurs before final replay;
- regime-gated variants are evaluated;
- active-model decay evaluator exists and fails safely;
- proven shadow candidate definition is enforced;
- long-side replay gap diagnostics are emitted;
- targeted tests pass;
- `audit_report.md` is updated with accepted progress;
- no production registry write occurs;
- no live/demo trading resumes from this work alone.

## 16. Explicit Rejection Conditions

The lead/audit agent must reject the implementation if any of these occur:

- old FX/MQL5/competition strategy code is imported into current runtime;
- a model is marked pass while losing to `best_nonflat`;
- a short-side model is marked pass while losing to `short_only`;
- a long-side model is marked pass using inverse-side skipped-row returns;
- missing benchmark data is treated as pass;
- unknown regime is treated as favorable;
- symbol pruning is reported but final replay still trades rejected symbols;
- decay directly promotes an unproven shadow candidate;
- decay bypasses hard-risk pause or approval controls;
- any production registry write happens during recovery experiments;
- tests are skipped without a documented blocker.
