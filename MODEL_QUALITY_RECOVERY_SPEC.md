# Model Quality Recovery And Paper-Soak Readiness Spec

## Objective

Recover model quality after the lineage reset without weakening production safety gates. The system must remain in no-trade mode until a fresh candidate proves positive executable paper expectancy under realistic costs, passes horizon-specific validation, and is eligible for shadow/paper-soak evaluation.

This spec starts from the 2026-06-25 clean retrain result on 4arm:

- Runtime correctly entered no-active-model/no-trade mode.
- Fresh full-universe retrain produced healthy feature coverage: `42,600` featured rows across ten symbols.
- All required horizons failed the configured `RETRAIN_MIN_ACCURACY=0.60` development gate:
  - `2h = 0.5229`
  - `4h = 0.5199`
  - `8h = 0.5408`
- No candidate was registered or published.

## Non-Goals

- Do not lower `RETRAIN_MIN_ACCURACY` merely to force a model into production.
- Do not mark an old Ubuntu-era or AWS-era artifact active unless it passes current manifest, runtime, and replay gates.
- Do not resume live trading.
- Do not bypass no-active-model behavior.
- Do not promote directly from retrain output.
- Do not optimize only classification accuracy while ignoring fees, spread, slippage, funding, turnover, and drawdown.

## Required Direction

The failed clean retrain is a useful signal: the corrected pipeline is now capable of refusing weak candidates. The next step is to diagnose whether the alpha, labels, thresholds, costs, universe, horizons, or validation policy are mis-specified.

The system should move through:

1. Reproducible diagnostics.
2. Transparent benchmark construction.
3. Label and threshold calibration.
4. Cost-adjusted portfolio replay.
5. Fresh candidate generation.
6. Shadow/paper-soak admission.

## Phase 1 - Freeze Current No-Trade Baseline

### Required Actions

1. Keep Telegram in no-active-model/no-trade mode.
2. Preserve the failed retrain record under `models/production/.failed/`.
3. Export a compact evidence bundle for the failed run:
   - requested universe
   - date range
   - per-symbol raw rows
   - per-symbol feature rows
   - horizon label counts
   - fold metrics
   - selected thresholds
   - failure reason
4. Add a command or script that can reproduce this diagnostic report locally and on 4arm.

### Acceptance

- Running the diagnostic command on 4arm reports the same model root, registry root, universe, row counts, and horizon scores.
- No fresh `models/production/model_*` directory exists unless a complete candidate is published.
- `registry_events.jsonl` remains empty or contains only valid fresh candidate events.

## Phase 2 - Benchmark Incumbent For No-Active-Model Mode

### Required Actions

Create transparent benchmark strategies that can serve as comparison incumbents when no trusted model exists:

1. Flat/no-trade benchmark.
2. Low-turnover momentum benchmark.
3. Low-turnover mean-reversion benchmark.
4. Optional volatility-filtered benchmark using only price/volume features.

Each benchmark must use the same:

- universe
- bars
- cost model
- risk sizing rules
- portfolio replay engine
- accounting outputs

### Acceptance

- Benchmark results include net PnL, Sharpe or equivalent risk-adjusted return, max drawdown, turnover, fees, funding, win rate, average trade, and exposure by symbol.
- ML candidates must beat the flat benchmark and at least one simple transparent trading benchmark before paper-soak admission.
- If every benchmark loses after costs, the system remains no-trade until market/regime assumptions are revisited.

## Phase 3 - Label And Dead-Zone Audit

### Required Actions

Audit `_build_labels()` and label distributions for each symbol and horizon.

Measure:

- up/down/ambiguous counts
- class balance
- ambiguous rate under current `dead_zone`
- forward return distribution
- expected fee/spread/funding cost per horizon
- label stability across symbols and monthly blocks

Evaluate a small grid:

```text
horizons: 2h, 4h, 8h
dead_zone: cost_floor, 0.10%, 0.15%, 0.20%, 0.30%, 0.50%
training_window: 3m, 6m, 9m, 12m
recency_half_life: 30d, 60d, 90d
```

Do not choose a configuration from the final holdout.

### Acceptance

- A label audit report is written under `docs/model_quality/`.
- Any selected `dead_zone` is justified by executable cost floor, not arbitrary classification accuracy.
- Each horizon has enough non-ambiguous labels per symbol and month to support temporal validation.

## Phase 4 - Horizon-Specific Validation Policy

### Required Actions

Replace the single global `RETRAIN_MIN_ACCURACY=0.60` decision with a documented validation policy that may include horizon-specific gates, but only after Phase 3 evidence exists.

Permitted gates:

- minimum development accuracy by horizon
- minimum positive edge versus benchmark after costs
- minimum precision at selected threshold
- minimum actionable decision count
- maximum turnover
- maximum drawdown
- minimum symbol coverage
- minimum fold count

Disallowed gates:

- pass because accuracy is above random by a tiny margin
- pass because there is no incumbent
- pass because the bot needs to trade
- pass because a threshold was tuned on final holdout or paper-soak data

### Acceptance

- Validation policy is versioned and included in model manifests.
- The policy explains why any threshold below `0.60` is economically acceptable.
- Failing horizons cannot be silently omitted while `RETRAIN_REQUIRE_ALL_HORIZONS=1`.

## Phase 5 - Cost-Adjusted Portfolio Replay

### Required Actions

Run candidate signals through executable portfolio replay before registration or paper-soak admission.

Replay must include:

- fees
- spread
- slippage
- funding
- order minimums and rounding
- position caps
- gross/net exposure limits
- cooldowns
- reduce-only behavior
- route/accounting ledger

Replay must compare candidate, flat benchmark, transparent benchmarks, and any trusted incumbent over identical calendar blocks.

### Acceptance

- A candidate cannot enter paper soak with negative absolute cost-adjusted expectancy.
- Replay outputs are stored as immutable evidence under `models/evidence/<version_id>/` or equivalent.
- Replay uses the same risk and accounting code paths as production where practical.

## Phase 6 - Candidate Generation Loop

### Required Actions

Implement a controlled experiment runner that trains candidate variants without touching production registry active state.

Candidate axes:

- training window: 3m, 6m, 9m, 12m
- recency weighting: 30d, 60d, 90d
- dead-zone policy from Phase 3
- threshold policy from development folds
- feature set variants:
  - full feature set
  - price/volume/funding only
  - no open-interest-derived features
  - no order-book placeholder features
- per-horizon calibration method

Every attempted candidate must write:

- dataset manifest
- feature manifest
- label audit
- fold ledger
- holdout report
- replay report
- selection-risk summary

### Acceptance

- Experiment attempts are append-only and auditable.
- The selected candidate has a better holdout rank than alternatives, not merely a lucky single run.
- Selection risk is summarized before paper-soak admission.

## Phase 7 - Shadow And Paper-Soak Admission

### Required Actions

Only after a candidate clears Phases 3-6:

1. Register candidate as `paper_quarantine`.
2. Keep `BOT_RETRAIN_AUTO_PROMOTE=0`.
3. Keep `MODEL_EVAL_AUTO_PROMOTE=1` only if evaluator gates require positive absolute expectancy and benchmark comparison.
4. Start paper soak with flat starting books.
5. Require at least `168` hours of forward shadow/paper evidence unless manually extended.

### Acceptance

- Candidate has no direct active promotion from retrain.
- Paper soak starts from flat books.
- Candidate records enough resolved/actionable decisions across at least three symbols.
- Candidate remains blocked if hard-risk, data-stale, registry, or execution health checks fail.

## Phase 8 - Production Resume Gate

Production trading can resume only when all are true:

- Fresh candidate artifact is complete, manifest-valid, and registry-approved.
- Candidate passes validation policy version current at time of training.
- Candidate beats flat and transparent benchmark after costs.
- Candidate completes forward paper soak with positive absolute expectancy.
- Paper soak drawdown remains inside approved bounds.
- No active hard-risk pauses exist.
- Paper books reconcile to flat before any live canary.
- Deployment image, source SHA, env policy, and model manifest match.

## Deliverables

1. `docs/model_quality/failed_retrain_diagnostic_<timestamp>.md`
2. `docs/model_quality/label_audit_<timestamp>.md`
3. `docs/model_quality/validation_policy_v1.md`
4. `docs/model_quality/benchmark_replay_<timestamp>.md`
5. `docs/model_quality/candidate_selection_<timestamp>.md`
6. Tests for label audit, horizon gates, replay gates, and no-active-model blocking.

## Lead Agent Notes

- Treat the 2026-06-25 failed retrain as a healthy refusal, not an operational failure.
- The immediate objective is not to make the bot trade; it is to make the next trade defensible.
- A model with `~0.52-0.54` directional accuracy may still be profitable only if thresholded decisions have positive expectancy after costs. That must be proven by replay and paper evidence, not assumed.
- If no ML candidate beats simple benchmarks, keep the system no-trade and revisit the strategy hypothesis.
