# Model Lineage Reset And Auto-Promotion Refactor Spec

## Objective

Reset model operations so Ubuntu-era and failed retrain artifacts cannot be selected accidentally, then create a clean candidate lineage where new models are trained, shadowed, evaluated, and optionally auto-promoted only after passing forward paper gates.

This spec is intentionally prescriptive. Do not broaden it into unrelated strategy work, exchange integration work, UI redesign, or live-capital activation.

## Current Failure Summary

- Runtime selected the lexicographically latest directory, `model_20260624_221949`, even though it was an empty failed retrain artifact.
- Registry active pointer still referenced `model_20260602_082230`; runtime fallback ignored this because it accepted newest folder discovery.
- Scheduled retrain creates the final `models/production/model_*` directory before data validation and model artifact completion.
- Recent retrain failed on stale ADAUSDT funding data and left visible incomplete production directories.
- Existing Ubuntu-era model lineage is not trusted because profitability degraded after migration from AWS and prior audit evidence found partial-data promotions.
- Forward shadow/evaluator logic exists, but deployment currently disables evaluator auto-promotion through `MODEL_EVAL_AUTO_PROMOTE=0`.

## Non-Goals

- Do not delete model artifacts permanently.
- Do not activate live trading.
- Do not bypass hard-risk pause rules.
- Do not turn on `BOT_RETRAIN_AUTO_PROMOTE`.
- Do not promote a model directly from retrain output.
- Do not lower validation thresholds just to force a trade.

## Target Behavior

1. Runtime loads only a registry-approved, structurally valid model artifact.
2. Failed retrains leave no visible `models/production/model_*` candidate directory.
3. Old model lineage is archived outside runtime selection but preserved for forensic comparison.
4. Fresh retrains register as `paper_quarantine` candidates.
5. Candidates shadow the sitting model for the configured forward window.
6. Evaluator may auto-promote only after the candidate beats incumbent under executable, cost-adjusted paper evaluation and all safety gates pass.
7. Direct retrain auto-promotion remains disabled.

## Environment Policy

Create or update the deployment `.env` on 4arm with these non-secret values:

```dotenv
BOT_RETRAIN_AUTO_PROMOTE=0
MODEL_EVAL_AUTO_PROMOTE=1
MODEL_EVAL_PROMOTION_WINDOW_HOURS=168
MODEL_EVAL_THRESHOLD_TUNING_HOURS=72
MODEL_EVAL_MIN_RESOLVED_DECISIONS=500
MODEL_EVAL_MIN_ACTIONABLE_DECISIONS=30
MODEL_EVAL_MIN_EDGE_BPS=25
MODEL_EVAL_MIN_SYMBOLS=3
RETRAIN_INTERVAL_HOURS=168
RETRAIN_TRAIN_MONTHS=6
RETRAIN_REQUIRE_ALL_SYMBOLS=1
RETRAIN_REQUIRE_ALL_HORIZONS=1
```

Important distinction:

- `BOT_RETRAIN_AUTO_PROMOTE=0` means fresh retrain output is never activated immediately.
- `MODEL_EVAL_AUTO_PROMOTE=1` means the evaluator may promote a quarantined model after the shadow window and promotion gates pass.

If active-session runtime blockers remain in place, auto-promotion may still be blocked while users are active. Either keep this as an explicit safety blocker or implement the demo-only safe reload path in Phase 5.

## Phase 1 - Artifact Quarantine

### Required Actions

On 4arm, move invalid or untrusted production model directories out of runtime discovery:

- Empty failed retrain directories:
  - `models/production/model_20260624_220737`
  - `models/production/model_20260624_221344`
  - `models/production/model_20260624_221949`
- Any other `model_*` directory that has no `model_*m.pkl`, no `config.json`, and no valid manifest.

Use an archive path outside runtime selection:

```text
models/archive/ubuntu_lineage_<timestamp>/
models/archive/failed_retrains_<timestamp>/
```

Do not delete files.

### Acceptance

- `find models/production -maxdepth 1 -type d -name 'model_*'` shows no empty model directories.
- `telegram_bot` restart no longer resolves to an empty directory.
- A before/after manifest of moved paths is written under `docs/model_lineage_reset/`.

## Phase 2 - Registry-Only Runtime Model Resolution

### Required Code Changes

Update `quant/telebot/model_selection.py`.

Rules:

1. Prefer `ModelRegistry.get_active_version()`.
2. Validate active artifact with a strict artifact check.
3. If active registry artifact is invalid, fail closed with a clear warning/error.
4. Do not fall back to lexicographic latest folder in production mode.
5. `find_latest_model()` may remain only for explicit local/dev bootstrap, and it must filter with `_looks_like_model_artifact()`.

Strict artifact check:

- Directory exists.
- Contains `config.json`, or contains all required horizon files `model_2m.pkl`, `model_4m.pkl`, `model_8m.pkl`.
- Reject empty directories.
- Reject directories without registry metadata unless an explicit dev flag is set.

### Acceptance

- Unit test: empty newest `model_*` directory is ignored or fails closed.
- Unit test: invalid active registry artifact does not silently fallback to latest.
- Unit test: valid active registry artifact resolves correctly.
- Runtime log says `source=registry_active:<version_id>` for production deploy.

## Phase 3 - Atomic Retrain Artifact Publishing

### Required Code Changes

Update `quant_v2/research/scheduled_retrain.py`.

Current issue: `artifact_dir = model_root / version_id` is created before data quality and validation succeed.

Required behavior:

1. Build under a staging directory:

```text
models/production/.building/<version_id>/
```

2. Run all data-quality checks before final artifact publish.
3. Train all required horizons.
4. Write model files and required manifests.
5. Validate artifact loadability in the same runtime path used by Telegram.
6. Atomically rename staging directory to:

```text
models/production/<version_id>/
```

7. Register candidate only after atomic publish succeeds.
8. On failure, remove staging directory and write a failure record under:

```text
models/production/.failed/<version_id>.json
```

### Acceptance

- Failed data-quality validation leaves no `models/production/model_*` directory.
- Failed model training leaves no `models/production/model_*` directory.
- Successful retrain creates a complete artifact and registry record.
- Tests cover stale ADAUSDT funding failure and prove no visible production artifact remains.

## Phase 4 - Clean Fresh Lineage

### Required Actions

After Phases 1-3 are deployed:

1. Archive current registry projections and event log.
2. Create a new registry namespace or explicit lineage marker:

```text
models/production/registry/lineage.json
```

3. Mark old Ubuntu-era versions as archived/rejected, not deleted.
4. Keep one known-loadable baseline as incumbent only if it passes artifact validation.
5. If no trusted incumbent exists, create a transparent benchmark incumbent record for comparison and keep the bot in shadow/paper-only mode until a fresh candidate passes.

### Acceptance

- `/model_versions` or registry inspection clearly separates archived lineage from fresh lineage.
- New retrain candidates do not inherit promotion eligibility from archived records.
- Fresh candidate starts in `paper_quarantine`.

## Phase 5 - Auto-Promotion After Shadow Window

### Required Code Review

Review `quant_v2/research/model_evaluator.py`.

Promotion must require:

- `MODEL_EVAL_AUTO_PROMOTE=1`.
- Persistent evaluator control has `auto_promote=true`.
- Candidate has at least `MODEL_EVAL_PROMOTION_WINDOW_HOURS=168`.
- Threshold tuning window is complete.
- Candidate beats incumbent by configured edge.
- Candidate positive absolute expectancy.
- Candidate beats benchmark if benchmark replay is available.
- Minimum resolved/actionable decisions pass.
- Required symbol coverage passes.
- No hard-risk pauses.
- No live sessions active.

Demo-session handling options:

Option A, conservative:

- Keep active demo sessions as promotion blockers.
- Auto-promotion occurs only during a flat/idle maintenance window.

Option B, controlled demo reload:

- Allow auto-promotion while demo sessions are active only if all are `live_mode=0`.
- Persist paper state.
- Promote registry pointer.
- Restart/reload Telegram signal manager.
- Restore demo sessions.
- Emit audit event and Telegram admin notification.

Do not allow auto-promotion while any `live_mode=1` session is active.

### Acceptance

- Test: `MODEL_EVAL_AUTO_PROMOTE=0` blocks promotion even if control JSON says true.
- Test: `MODEL_EVAL_AUTO_PROMOTE=1` plus passing gates promotes.
- Test: hard-risk pause blocks promotion.
- Test: active live session blocks promotion.
- If Option B is implemented, test active demo sessions are saved, reloaded, and restored.

## Phase 6 - Data Quality Policy For Fresh Retrains

### Required Decision

Stale funding/open-interest data must not silently poison training.

Choose one policy:

Policy 1, strict:

- Any required symbol with stale funding/OI fails the whole retrain.
- No visible production artifact remains.

Policy 2, explicit exclusion:

- A stale symbol is excluded only if `RETRAIN_REQUIRE_ALL_SYMBOLS=0`.
- Registry records `symbols_failed`.
- Candidate is not promotion eligible unless minimum symbol coverage is met.

Recommended policy for this system: strict while rebuilding trust.

### Acceptance

- ADAUSDT stale funding failure produces a failure record and no visible `model_*`.
- Registry metadata for successful candidates includes requested/fetched/failed symbols and data quality summary.

## Phase 7 - Observability

### Required Code Changes

Add clear runtime visibility when no positions are opened:

- Log every cycle summary:
  - model version/source
  - number of symbols evaluated
  - BUY count
  - SELL count
  - HOLD count
  - blocked by market short guard
  - skipped by deadband
  - accepted order count
- Telegram admin command should report:
  - active model version
  - artifact validity
  - last signal cycle time
  - last route event time
  - why no position is held

### Acceptance

- After one quiet cycle, logs explain whether no positions are due to `no_active_model`, market guard, thresholds, deadband, or risk block.
- No more silent multi-hour `HOLD/no_active_model` behavior.

## Validation Commands

Run locally before deploy:

```powershell
python -m pytest tests/quant/test_model_selection.py tests/quant_v2/test_scheduled_retrain_candidates.py tests/quant_v2/test_model_evaluator.py -q
```

Run on 4arm after deploy:

```bash
docker compose ps
docker logs --since 20m quant_telegram | egrep -i 'Using model|registry_active|no_active_model|Signal decision|cycle summary|V2 route audit|ERROR|Traceback'
docker logs --since 20m quant_retrain | egrep -i 'Retrain|failed|registered|ERROR|Traceback'
docker logs --since 20m quant_model_eval | egrep -i 'promotion|auto_promote|eligible|blocked|ERROR|Traceback'
```

## Operator Rollout Sequence

1. Stop relying on current runtime latest-folder selection.
2. Archive empty failed artifacts.
3. Deploy registry-only model selection and atomic retrain publishing.
4. Restart `telegram_bot`.
5. Confirm runtime uses a valid registry-active model or intentionally runs no-trade shadow mode.
6. Enable `.env` evaluator auto-promotion:

```dotenv
MODEL_EVAL_AUTO_PROMOTE=1
BOT_RETRAIN_AUTO_PROMOTE=0
```

7. Restart `model_evaluator`.
8. Trigger fresh retrain.
9. Confirm fresh candidate registered as `paper_quarantine`.
10. Observe full shadow period.
11. Allow evaluator promotion only if all gates pass.

## Shadow Model Lifecycle Explanation

The system does not spawn an unlimited number of shadow models at once by design.

- The retrain scheduler runs on `RETRAIN_INTERVAL_HOURS`, currently `168` hours by default.
- Each successful retrain creates one new candidate model version.
- The evaluator compares quarantined candidates against the sitting incumbent over the shadow window.
- If a candidate does not outperform the incumbent or does not pass gates, it should remain blocked/rejected/expired.
- A later scheduled retrain creates a new candidate; the old underperforming candidate should not keep shadowing forever.

Implementation must add or verify expiration/rejection for stale candidates so the registry does not accumulate indefinitely active shadow candidates.
