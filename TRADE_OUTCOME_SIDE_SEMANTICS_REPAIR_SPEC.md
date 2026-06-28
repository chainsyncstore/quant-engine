# Trade-Outcome Side Semantics Repair Spec

**Spec date:** 2026-06-28
**Implementer profile:** GPT-5.4 mini orchestrating spawned implementation agents
**Audit owner:** GPT-5.5 lead/audit agent
**Source evidence:** 4arm recovery run `20260627T191451Z_b1593641`
**Current runtime disposition:** **NO PRODUCTION TRADING**

## 1. Objective

Repair the remaining semantic gap in the model recovery pipeline.

The current trade-outcome label implementation correctly defines:

```text
label 1.0 = this side's trade is worth taking after costs
label 0.0 = this side's trade is not worth taking
```

But the recovery replay path still behaves like a directional classifier in some places:

```text
high probability -> BUY
low probability  -> SELL
middle           -> HOLD
```

That is wrong for trade-outcome labels. For trade-outcome models, class `0` means `HOLD/SKIP`, not the opposite side.

The repaired pipeline must evaluate side-specific trade decisions:

```text
long-side model:
  probability >= threshold -> BUY
  probability < threshold  -> HOLD

short-side model:
  probability >= threshold -> SELL
  probability < threshold  -> HOLD

directional_return legacy model:
  probability >= threshold       -> BUY
  probability <= 1 - threshold   -> SELL
  otherwise                      -> HOLD
```

Production trading remains disabled until a later audit explicitly approves paper quarantine.

## 2. Non-Negotiable Constraints

- Do not resume live or demo trading.
- Do not promote or register a model.
- Do not write to production registry or production model roots.
- Do not weaken accuracy, expectancy, drawdown, benchmark, or selection-risk gates.
- Do not delete old models or registry records.
- Preserve `directional_return` behavior as a legacy diagnostic path.
- Keep all recovery runs under `--no-production-registry`.
- Keep secrets, `.env`, exchange keys, chat IDs, IPs, database dumps, and Tailscale hostnames out of commits and new docs.
- Any candidate that is profitable only because class `0` became `SELL` must fail after this repair.

## 3. Required Sub-Agent End Note

Every implementation sub-agent must return this exact block:

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

The lead/audit agent must respond with:

```text
LEAD AUDIT DECISION: accept | reject
ACCEPTANCE EVIDENCE:
REJECTION REASON:
CORRECTION INSTRUCTIONS:
AUDIT_REPORT_UPDATE:
```

No phase is accepted until tests pass and `audit_report.md` receives a dated checkpoint.

## 4. Preflight

Run locally before edits:

```powershell
git status --short
rg -n "def _candidate_signal_resolver|def _candidate_replay_report|def _evaluate_candidate|def _build_candidate_labels|class RecoveryCandidateConfig" quant_v2\research\model_recovery_experiments.py
rg -n "proba >= actor.threshold|proba <= \\(1.0 - actor.threshold\\)|signal = \"SELL\"|signal = \"BUY\"" quant_v2\research\portfolio_replay.py quant_v2\research\model_recovery_experiments.py
```

Run on 4arm only after local validation passes:

```bash
cd /home/admin-4arm/hypothesis-research-engine
git status --short
ps -eo pid,etime,cmd | grep -E 'run_model_recovery_experiments|model_recovery_experiments' | grep -v grep || true
```

Stop if any recovery runner is already active.

## 5. Phase 1 - Add Explicit Trade Side To Candidate Config

### Goal

Make trade-outcome candidate identity side-aware so long and short trade-taking models cannot collide.

### Files May Edit

- `quant_v2/research/model_recovery_experiments.py`
- `tests/quant_v2/test_model_recovery_experiments.py`

### Required Changes

Extend `RecoveryCandidateConfig` with:

```python
trade_outcome_side: str = "long"
```

Allowed values:

```text
long
short
```

Validation rule:

- If `label_mode != "trade_outcome"`, `trade_outcome_side` is ignored but still serialized.
- If `label_mode == "trade_outcome"` and side is not `long` or `short`, raise:

```python
ValueError("unsupported trade_outcome_side=...")
```

Update `candidate_id()`:

- Existing directional-return candidate IDs must remain unchanged.
- Trade-outcome candidate IDs must include side:

```text
h2_tw3m_hl30d_dz0p0010_fsfull_sidelong
h2_tw3m_hl30d_dz0p0010_fsfull_sideshort
```

### Required Tests

Add tests proving:

- default `trade_outcome_side == "long"`;
- directional candidate IDs are unchanged;
- trade-outcome long/short candidate IDs are distinct;
- unsupported trade-outcome side raises `ValueError`.

### Acceptance Command

```powershell
python -m pytest tests/quant_v2/test_model_recovery_experiments.py -q
```

## 6. Phase 2 - Build Side-Specific Trade-Outcome Labels

### Goal

Train side-specific take/skip models instead of treating trade-outcome `0` as opposite direction.

### Files May Edit

- `quant_v2/research/model_recovery_experiments.py`
- `tests/quant_v2/test_model_recovery_experiments.py`
- `tests/quant_v2/test_trade_outcome_labels.py` only if adding side coverage

### Required Changes

Update `_build_candidate_labels(frame, config)`:

- For `directional_return`, preserve existing behavior exactly.
- For `trade_outcome`, call:

```python
build_trade_outcome_labels(frame, config=trade_cfg, side=config.trade_outcome_side)
```

Return a report containing:

```python
{
    "label_mode": "trade_outcome",
    "trade_outcome_side": "long" | "short",
    "policy_version": "trade_outcome_labels_v1",
    ...
}
```

The underlying `trade_outcome_report` must still include both long and short diagnostics, but the selected training labels must come only from `config.trade_outcome_side`.

### Required Tests

Add tests proving:

- long-side trade-outcome labels can differ from short-side labels on the same frame;
- `_build_candidate_labels(... side="long")` returns the long-side selected report;
- `_build_candidate_labels(... side="short")` returns the short-side selected report;
- report includes `trade_outcome_side`.

### Acceptance Command

```powershell
python -m pytest tests/quant_v2/test_trade_outcome_labels.py tests/quant_v2/test_model_recovery_experiments.py -q
```

## 7. Phase 3 - Correct Candidate Replay Signal Semantics

### Goal

Ensure trade-outcome class `0` maps to `HOLD`, never inverse-side trade.

### Files May Edit

- `quant_v2/research/model_recovery_experiments.py`
- `tests/quant_v2/test_model_recovery_experiments.py`
- `quant_v2/research/portfolio_replay.py` only if the cleanest implementation requires metadata-aware replay behavior
- `tests/quant_v2/test_portfolio_replay.py` only if `portfolio_replay.py` is edited

### Required Changes

Update `_candidate_signal_resolver(...)` in `model_recovery_experiments.py`.

It must read:

```python
label_mode = actor.metadata.get("label_mode", "directional_return")
trade_outcome_side = actor.metadata.get("trade_outcome_side", "long")
```

Required behavior:

```python
if label_mode == "trade_outcome":
    if proba >= threshold and trade_outcome_side == "long":
        signal = "BUY"
        confidence = proba
        reason = f"trade_outcome_long_take>={threshold:.2f}"
    elif proba >= threshold and trade_outcome_side == "short":
        signal = "SELL"
        confidence = proba
        reason = f"trade_outcome_short_take>={threshold:.2f}"
    else:
        signal = "HOLD"
        confidence = max(proba, 1.0 - proba)
        reason = "trade_outcome_skip"
elif label_mode == "directional_return":
    preserve current BUY / SELL / HOLD behavior
else:
    return HOLD with reason "unsupported_label_mode"
```

Never use this rule for trade-outcome models:

```python
proba <= 1.0 - threshold -> SELL
```

Update `_candidate_replay_report(...)` so the `ReplayActorConfig` metadata includes:

```python
{
    "label_mode": config.label_mode,
    "trade_outcome_side": config.trade_outcome_side,
}
```

If `_candidate_replay_report` does not currently receive `config`, add keyword arguments:

```python
label_mode: str = "directional_return"
trade_outcome_side: str = "long"
```

Pass them from `_evaluate_candidate`.

### Required Tests

Add tests proving:

- trade-outcome long model with `proba >= threshold` emits `BUY`;
- trade-outcome long model with `proba < threshold` emits `HOLD`;
- trade-outcome long model with `proba <= 1 - threshold` still emits `HOLD`, not `SELL`;
- trade-outcome short model with `proba >= threshold` emits `SELL`;
- trade-outcome short model with `proba < threshold` emits `HOLD`;
- directional-return behavior still emits `SELL` when `proba <= 1 - threshold`;
- replay actor metadata includes `label_mode` and `trade_outcome_side`.

### Acceptance Command

```powershell
python -m pytest tests/quant_v2/test_model_recovery_experiments.py tests/quant_v2/test_portfolio_replay.py -q
```

## 8. Phase 4 - Correct Holdout Economics For Trade-Outcome Models

### Goal

Stop scoring skipped trade-outcome rows as if they earned the opposite-side return.

### Files May Edit

- `quant_v2/research/model_recovery_experiments.py`
- `tests/quant_v2/test_model_recovery_experiments.py`

### Current Bug Pattern To Remove

Any code equivalent to this is invalid for trade-outcome labels:

```python
signed_returns = np.where(holdout_preds == 1, forward_return_bps, -forward_return_bps)
```

For trade-outcome models:

- `holdout_preds == 1` means take the configured side;
- `holdout_preds == 0` means skip;
- skipped rows have realized model return `0.0`, not inverse forward return.

### Required Return Semantics

Implement a helper:

```python
def _candidate_decision_returns_bps(
    *,
    label_mode: str,
    trade_outcome_side: str,
    predictions: np.ndarray,
    forward_return_bps: np.ndarray,
    cost_bps: float,
) -> np.ndarray:
    ...
```

Behavior:

For `directional_return`:

```text
prediction 1 -> +forward_return_bps - cost_bps
prediction 0 -> -forward_return_bps - cost_bps
```

For `trade_outcome` and side `long`:

```text
prediction 1 -> +forward_return_bps - cost_bps
prediction 0 -> 0.0
```

For `trade_outcome` and side `short`:

```text
prediction 1 -> -forward_return_bps - cost_bps
prediction 0 -> 0.0
```

Use this helper for:

- `cost_adjusted_expectancy_bps`;
- `gross_expectancy_bps`;
- candidate scoring where it currently uses `signed_returns`;
- any holdout report field that represents model decision economics.

Do not change the `select_threshold_by_utility` input contract in this phase.

### Required Holdout Report Additions

Add to `holdout_report`:

```python
"label_mode": config.label_mode,
"trade_outcome_side": config.trade_outcome_side,
"predicted_take_count": int(...),
"predicted_skip_count": int(...),
"predicted_buy_count": int(...),
"predicted_sell_count": int(...),
"predicted_hold_count": int(...),
```

For trade-outcome long:

```text
predicted_take_count == predicted_buy_count
predicted_skip_count == predicted_hold_count
predicted_sell_count == 0
```

For trade-outcome short:

```text
predicted_take_count == predicted_sell_count
predicted_skip_count == predicted_hold_count
predicted_buy_count == 0
```

For directional-return:

```text
preserve current buy/sell semantics; hold count may remain 0 for hard labels
```

### Required Tests

Add tests proving:

- trade-outcome skipped rows contribute `0.0` return;
- trade-outcome long taken rows use positive forward returns minus costs;
- trade-outcome short taken rows use negative forward returns minus costs;
- directional-return rows preserve inverse-side behavior;
- holdout report counts are side-consistent.

### Acceptance Command

```powershell
python -m pytest tests/quant_v2/test_model_recovery_experiments.py -q
```

## 9. Phase 5 - Expand Trade-Outcome Candidate Grid To Long And Short

### Goal

Evaluate long-taking and short-taking trade-outcome candidates separately.

### Files May Edit

- `quant_v2/research/model_recovery_experiments.py`
- `tests/quant_v2/test_model_recovery_experiments.py`

### Required Changes

Update `_candidate_grid(...)`.

If `label_mode == "trade_outcome"`:

- generate both `trade_outcome_side="long"` and `trade_outcome_side="short"` configs for each selected grid cell;
- ensure candidate IDs are unique;
- ensure `max_candidates` still caps final candidates after side expansion;
- preserve deterministic ordering:

```text
for each base cell:
  long candidate first
  short candidate second
```

If `label_mode == "directional_return"`:

- preserve existing candidate generation exactly.

Update `build_phase4_variant_specs(...)` similarly if it accepts `label_mode="trade_outcome"`:

- produce side-aware configs;
- either duplicate variants by side or add a clearly named variant suffix:

```text
cost_aware_ternary_long
cost_aware_ternary_short
```

### Required Tests

Add tests proving:

- trade-outcome candidate grid contains both long and short sides;
- candidate IDs are unique;
- `max_candidates` is respected after side expansion;
- directional-return candidate grid output is unchanged.

### Acceptance Command

```powershell
python -m pytest tests/quant_v2/test_model_recovery_experiments.py -q
```

## 10. Phase 6 - One-Sided Collapse And Benchmark Gate Audit

### Goal

Make diagnostics explicit when a candidate only behaves like a weaker benchmark.

### Files May Edit

- `quant_v2/research/model_recovery_experiments.py`
- `tests/quant_v2/test_model_recovery_experiments.py`

### Required Changes

Update `prediction_audit` and `selection_risk_summary` for trade-outcome models:

```python
"trade_outcome_side": config.trade_outcome_side,
"take_share": predicted_take_count / sample_count,
"skip_share": predicted_skip_count / sample_count,
"one_sided_take_collapse": take_share < 0.01 or take_share > 0.99,
"benchmark_like_behavior": "short_only" | "long_only" | None,
```

Rules:

- If trade-outcome short candidate has `predicted_sell_count / sample_count > 0.95`, mark `benchmark_like_behavior="short_only"`.
- If trade-outcome long candidate has `predicted_buy_count / sample_count > 0.95`, mark `benchmark_like_behavior="long_only"`.
- Add failure reason `benchmark_like_behavior` if candidate does not beat that same benchmark.

Do not fail a candidate merely for being side-specific. Fail only if it behaves like a simple benchmark and does not beat it.

### Required Tests

Add tests proving:

- one-sided take collapse is reported;
- benchmark-like behavior is reported;
- candidate that behaves like short-only and fails to beat short-only gets `benchmark_like_behavior`;
- candidate that beats the benchmark is not failed solely for being side-specific.

### Acceptance Command

```powershell
python -m pytest tests/quant_v2/test_model_recovery_experiments.py -q
```

## 11. Phase 7 - End-To-End Local Validation

Run locally:

```powershell
python -m pytest `
  tests/quant_v2/test_trade_outcome_labels.py `
  tests/quant_v2/test_economic_thresholds.py `
  tests/quant_v2/test_selection_risk.py `
  tests/quant_v2/test_model_recovery_experiments.py `
  tests/quant_v2/test_model_quality_recovery.py `
  tests/quant_v2/test_scheduled_retrain_candidates.py `
  tests/quant_v2/test_portfolio_replay.py `
  -q
```

If the combined command times out, split it and report every split command exactly.

Required local pass criteria:

- all listed tests pass;
- no production registry write occurs;
- no recovery runner is left running;
- `audit_report.md` contains a dated checkpoint for the accepted local repair.

## 12. Phase 8 - 4arm Sync And Host Validation

Only after local validation passes, sync changed files to 4arm.

Required 4arm validation:

```bash
cd /home/admin-4arm/hypothesis-research-engine
PYTHONWARNINGS=ignore .venv/bin/python -m pytest \
  tests/quant_v2/test_trade_outcome_labels.py \
  tests/quant_v2/test_economic_thresholds.py \
  tests/quant_v2/test_selection_risk.py \
  tests/quant_v2/test_model_recovery_experiments.py \
  tests/quant_v2/test_model_quality_recovery.py \
  tests/quant_v2/test_scheduled_retrain_candidates.py \
  tests/quant_v2/test_portfolio_replay.py \
  -q
```

If this combined command times out, split it into:

```bash
PYTHONWARNINGS=ignore .venv/bin/python -m pytest tests/quant_v2/test_trade_outcome_labels.py tests/quant_v2/test_economic_thresholds.py tests/quant_v2/test_selection_risk.py -q
PYTHONWARNINGS=ignore .venv/bin/python -m pytest tests/quant_v2/test_model_quality_recovery.py tests/quant_v2/test_scheduled_retrain_candidates.py -q
PYTHONWARNINGS=ignore .venv/bin/python -m pytest tests/quant_v2/test_model_recovery_experiments.py -q
PYTHONWARNINGS=ignore .venv/bin/python -m pytest tests/quant_v2/test_portfolio_replay.py -q
```

Required host pass criteria:

- all listed tests pass;
- no production registry write occurs;
- no recovery runner is left running.

## 13. Phase 9 - 4arm No-Production-Registry Recovery Rerun

Run only after Phase 8 passes.

Required command:

```bash
cd /home/admin-4arm/hypothesis-research-engine
PYTHONWARNINGS=ignore .venv/bin/python scripts/run_model_recovery_experiments.py \
  --snapshot-path datasets/v2/snapshots/model_recovery_real_1h_20260626_20260626_141210.parquet \
  --docs-output-dir docs/model_quality \
  --max-candidates 12 \
  --label-mode trade_outcome \
  --trade-profit-target-bps 20 \
  --trade-stop-loss-bps 30 \
  --trade-round-trip-cost-bps 8 \
  --no-production-registry
```

If runtime exceeds 90 minutes but artifacts are progressing, continue monitoring. Do not start a second run.

Required run pass criteria:

- command completes;
- summary contains `"label_mode": "trade_outcome"`;
- evaluated candidate IDs include both `_sidelong` and `_sideshort`;
- no trade-outcome long candidate emits `SELL` signals;
- no trade-outcome short candidate emits `BUY` signals;
- holdout reports contain `predicted_hold_count`;
- threshold policy source is `economic_utility` or documented fallback;
- benchmark report includes expanded baseline availability;
- selection-risk report exists;
- production registry is not mutated.

## 14. Paper Quarantine Gate

This repair does **not** open production trading.

Paper quarantine may be proposed only if the 4arm rerun produces a selected candidate satisfying all of:

- `passed_candidates >= 1`;
- selected candidate beats flat after costs;
- selected candidate beats `short_only`, `moving_average_trend`, `adverse_excursion_exit`, and the best available non-flat benchmark after costs;
- selected candidate max drawdown is within configured limit;
- selection-risk report has no blockers;
- candidate does not rely on invalid inverse-side semantics;
- model artifact is exported only to a research/staging location;
- GPT-5.5 lead/audit agent explicitly approves paper quarantine in `audit_report.md`.

Production live trading remains blocked until after a separate paper-soak audit.

## 15. Expected Outcome

The expected near-term result may still be:

```text
recommendation = remain_no_trade
```

That is acceptable.

The objective is not to force a model to pass. The objective is to make the recovery evidence trustworthy by ensuring:

- long take/skip models do not accidentally short;
- short take/skip models do not accidentally long;
- skipped trades earn zero model return;
- candidate economics are measured against actual executable side behavior;
- any future passing candidate beats simple transparent baselines for the right reason.
