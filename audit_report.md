# Hypothesis Research Engine - Comprehensive Forensic Audit

**Audit date:** 2026-06-22
**Production host:** `4arm-ubuntu` / `4arm-host`
**Incident reviewed:** hard-risk pauses on 2026-06-18
**Overall disposition:** **DO NOT RESUME EXECUTION** until the P0 containment and correctness gates in this report pass.

## Implementation Progress

**Last updated:** 2026-06-28
**Build authority:** `IMPLEMENTATION_BUILD_PLAN.md`
**Current work package:** WP-15 - Deployment, Migration, And Runtime Readiness (all WP-00 through WP-15 work packages have accepted repository slices; broader rollout, staging, and production deployment rehearsal work remains open outside the repository acceptance boundary)
**Independent validation:** See `AUDIT_REPORT_VALIDATION.md` for the 2026-06-24 GPT-5.5 validation pass. Fresh repository tests passed (`773 passed, 21 warnings`). WP-01 provenance verification now passes against the retained Ubuntu source bundle. Exact-image certification, full-repository Ruff cleanliness, and production restart gates remain unproven locally.
**Demo hard-risk recovery note:** The retained Ubuntu snapshot shows both demo users paused with `hard_risk_paused=1`; a manual DB clear flips only the pause flag and leaves `is_active=0`, and the repo restart path for demo sessions rehydrates `paper_state_json` from SQLite on `/start_demo`. This is a valid recovery path for paused demo sessions, but it is not proof that production trading is ready to resume. On 2026-06-24 the retained demo books for both users were explicitly flattened in the persisted state and the bot was restarted; both sessions now rehydrate with `open_positions={}` and a 10,000 USD paper baseline.
**WP-03 status:** **REPOSITORY IMPLEMENTATION ACCEPTED; production rollout remains gated by operational verification**
**WP-04 Telegram slice:** **REPOSITORY IMPLEMENTATION ACCEPTED; reviewed-clearance path accepted; broader WP-04 work remains open**
**WP-04 execution slice:** **REPOSITORY IMPLEMENTATION ACCEPTED; broader WP-04 operator and recovery work remains open**
**WP-05 execution slice:** **REPOSITORY IMPLEMENTATION ACCEPTED; broader WP-05 ledger work remains open**
**WP-06 ledger core slice:** **REPOSITORY IMPLEMENTATION ACCEPTED; broader WP-06 reconciliation and migration work remains open**
**WP-07 execution-cost slice:** **REPOSITORY IMPLEMENTATION ACCEPTED; broader WP-07 calibration and live-adapter work remains open**
**WP-08 canonical data slice:** **REPOSITORY IMPLEMENTATION ACCEPTED; broader WP-08 feature-schema and missing-data policy work remains open**
**2026-06-24 implementation slice:** risk-reducing exits now bypass rebalance deadband/cooldown/max-order gating, `sync_positions()` flatten paths stay reduce-only, and the admin `/flatten_demo` command can flatten running demo sessions for a clean paper-soak restart. Full local repository validation passed (`774 passed, 45 warnings`).
**2026-06-27 Phase 4 research-input repair slice:** bounded variant repair, prediction-audit reporting, regime comparison, and docs/CLI wiring are now implemented in `quant_v2/research/model_recovery_experiments.py` and `scripts/model_quality_recovery.py`. Local validation for the Phase 4 path passed (`22 passed, 726 warnings`). This closes the repository-side implementation gap for the research-input repair phase, but it does **not** by itself reopen production trading.

### Accepted Repository Controls

WP-00 repository implementation was independently audited after one rejection and correction cycle. The accepted changes:

- prevent persistent evaluator state or a direct evaluator argument from enabling auto-promotion when deployment policy disables it;
- fail malformed evaluator control closed;
- suppress HTTP transport request logging and structurally redact raw/encoded Telegram credentials, authenticated URLs, and exception output;
- remove Redis host-port publication, disable its default user, configure a least-privilege ACL, validate ACL username/password input, and use a secret-backed authenticated health check;
- exclude incident-only pause and registry helper scripts from the Docker build context without changing the retained forensic files; and
- add `docs/WP00_CONTAINMENT_RUNBOOK.md` for controlled deployment, verification, and rollback.

Independent verification result:

```text
targeted containment/evaluator/security tests: 16 passed
full local suite: 642 passed, 29 warnings
```

Repository mitigation for F-04, F-06, and F-07 is accepted. Those findings remain open in production until the operational actions below are completed and evidenced.

### WP-00 Operational Closure Still Required

- Deploy the accepted controls in an approved maintenance window using a reviewed immutable image.
- Rotate the exposed Telegram credential after the logging fix is active; verify the old credential fails.
- Apply the approved retention/access response to credential-bearing historical logs.
- Create and deploy the Redis secret/ACL configuration; prove port 6379 is externally unreachable, unauthenticated access fails, and the authenticated health check succeeds.
- Verify production deployment and persistent evaluator controls both resolve `auto_promote=false`.
- Reconcile and explicitly dispose of or transfer supervised ownership of the retained BTC paper positions.
- Preserve before/after image, configuration, registry, database, and verification evidence.

No production state or credential was changed during repository implementation. The overall **DO NOT RESUME EXECUTION** disposition remains unchanged.

### WP-01 Source Provenance - Accepted Evidence Baseline

WP-01 was independently accepted after one rejection and correction cycle. Read-only collection from the Ubuntu host closed the provenance gaps left by the original audit without reading environment contents, credentials, logs, databases, registry/model state, or user state and without mutating production.

Newly verified identities:

```text
local Git HEAD:  507d6218fa6ed3df5831c3d9f18ab0bfebb107a9
Ubuntu Git HEAD: 6234aff58092458683125c8abcba333bbda99388
running image:   sha256:e85886744eaf85cb275c6cd1bd344b56fc1609482152743bb1d616ecbb0c7d58
```

All three application containers (`quant_telegram`, `quant_model_eval`, and `quant_retrain`) use the verified image. Every defined critical runtime source hash matches between the Ubuntu host and all three containers.

Accepted WP-01 artifacts:

- `docs/WP01_SOURCE_RECONCILIATION.md`;
- deterministic manifests under `docs/wp01/`;
- ignored source-only evidence under `ubuntu_audit_20260622/source_provenance/`;
- `tools/collect_wp01_source_provenance.py` and `tools/verify_wp01.py`;
- archive traversal, duplicate-member, credential-content, hash-binding, and deterministic-regeneration tests; and
- `.gitattributes` rules for stable LF-normalized source comparisons.

The reconciliation covers 141 source paths. It identifies 121 verified runtime matches, 14 host-only build/deployment paths, reviewed WP-00 replacements, four production-only experimental/helper files that must not enter the canonical baseline, and requirements inputs that remain unresolved for WP-02 dependency locking.

Independent acceptance evidence:

```text
WP-01 provenance tests: 12 passed
offline verifier: passed
deterministic full regeneration: passed
new-file Ruff checks: passed
credential scan: no findings
full local suite: 654 passed, 29 warnings
```

WP-01 is **evidence-complete**. Its clean-baseline exit gate remains pending a reviewed commit/branch and WP-02 exact-image certification. No claim is made that the dirty production tree itself is canonical.

### WP-02 Reproducible Build - Accepted Repository Controls

WP-02 repository implementation was audited after iterative correction and is
accepted as the build-hardening baseline for F-05, F-08, and F-18. The accepted
changes:

- pin Python `3.11.9-slim-bookworm` and Redis `7-alpine` by reviewed manifest
  digest, with Debian snapshot `20250201T000000Z` and exact `libgomp1`
  versioning;
- separate disposable build tooling from the runtime virtual environment so the
  release image cannot retain uv, pytest, Ruff, tests, or build lockfiles;
- require a clean-tree release manifest before evidence generation, then bind
  the build to reviewed source and lock digests;
- harden Compose and production overlays around immutable `QUANT_IMAGE`
  references, non-root execution, read-only roots, no added capabilities, and
  bounded writable surfaces;
- pin CI actions by full commit SHA, verify reviewed Syft `1.42.2` and Trivy
  `0.71.2` release checksums, and retain smoke, runtime exclusion, SBOM,
  vulnerability, secret-scan, manifest, and attestation evidence; and
- align the local release helper and repository tests with the same release
  marker and runtime exclusion gates enforced in CI.

Independent acceptance evidence:

```text
WP-02 build tests: 18 passed
WP-00/WP-01/WP-02/evaluator focused suite: 40 passed
scoped Ruff checks: passed
external artifact verifier: passed
full local suite: 672 passed, 29 warnings
```

WP-02 repository controls are accepted. Exact-image certification remains open
because the local audit host cannot build and push the reviewed image in Docker,
capture the registry `@sha256` for that same tested image, or produce the final
 signed deployment attestation. Those exit gates remain assigned to CI/WP-15.

### WP-03 Risk Policy Invariants And Sizing Headroom - Accepted Repository Controls

WP-03 repository implementation was audited after the residual-supervision and
telemetry corrections and is accepted as the invariants baseline for F-02 and
F-03. The accepted changes:

- separate immutable `HardRiskLimits` from dynamic `OperatingRiskLimits` and
  clamp every dynamic output against the hard limit;
- centralize policy calculation in `quant_v2/portfolio/risk_policy.py` so the
  execution layer no longer carries duplicated cap formulas;
- apply headroom to target sizing only, while breach detection continues to use
  the true hard limit;
- evaluate projected post-trade symbol, gross, net, and bucket exposure before
  routing each order;
- reserve capacity for fees, slippage, rounding, minimum quantity, and adverse
  mark movement;
- classify intents as `INCREASE`, `REDUCE`, `FLATTEN`, or `FLIP` from projected
  absolute exposure;
- mark flatten-only `sync_positions()` orders reduce-only so watchdog and
  circuit-breaker paths preserve the same residual-supervision invariant as the
  routed reduce/flatten path;
- bypass normal alpha/deadband gates for valid `REDUCE` and `FLATTEN` intents
  while still honoring exchange precision and minimum-notional constraints
  through a supervised residual-position path; and
- emit and persist risk policy versioning plus hard, dynamic, operating, and
  exposure telemetry on routed decisions and fills.

Independent acceptance evidence:

```text
WP-03 focused service/binance regressions: 59 passed
ruff on modified files: passed
full local suite: 694 passed, 30 warnings
```

The June 18 regression now stays below the operating cap in incident replay,
and the step-size/min-notional residual path preserves supervised closing
semantics instead of silently downgrading the fill. Repository mitigation for
F-02 and F-03 is accepted. Production rollout remains gated by the same
operational deployment and attestation controls described above.

### WP-04 Execution-Side Lifecycle Slice - Accepted Repository Controls

The execution-side lifecycle slice for WP-04 was audited after adding the
durable lifecycle state record and replay hooks and is accepted as the current
repository baseline for the execution lifecycle slice. The accepted changes:

- add a durable `LifecycleStateRecord` with `state`, `owner`, `retry_count`,
  `reason`, `policy_version`, and lifecycle timestamps;
- validate lifecycle transitions monotonically so lifecycle state cannot move
  backwards during live updates or WAL replay;
- persist lifecycle transitions to the execution WAL with a dedicated
  `lifecycle_changed` event and embed the initial state on `session_started`;
- restore lifecycle state across stop/start boundaries during WAL replay so
  the last terminal state survives process restarts; and
- keep the replayed lifecycle record available even after the active session is
  stopped, which preserves the durable ownership trail for later inspection.

Independent acceptance evidence:

```text
WP-04 lifecycle tests: 2 passed
execution service regression suite: 47 passed
WAL and replay regression suite: 15 passed
full local suite: 700 passed, 32 warnings
```

The slice intentionally does not close the broader WP-04 work package. Remaining
gaps still open in the repository are the operator-review flow for lifecycle
requests, the wider supervisor/ownership split across all control paths, the
incident-deduplicated escalation loop, and the transactional persistence of the
full final accounting snapshot before terminal shutdown.

### WP-04 Telegram Persistence And Review Controls - Accepted Telegram Slice

The Telegram-side persistence/control slice for WP-04 was implemented and
accepted after the reviewed-transition and lifecycle-reviewing tests passed.
The accepted changes:

- add durable lifecycle transition metadata to `quant/telebot/models.py` and
  backfill the SQLite schema for existing deployments;
- have `/stop` request a reviewed lifecycle transition with owner, reason,
  evidence reference, retry count, policy version, and timestamps instead of
  directly flipping pause booleans;
- block `/start_demo`, `/start_live`, `/continue_demo`, and `/continue_live`
  while a reviewed lifecycle transition request is pending; and
- surface the pending lifecycle transition in `/lifecycle` so operators can
  inspect the durable state before resuming.

Independent acceptance evidence:

```text
tests/quant/test_telebot_main_v2_handlers.py: 26 passed
tests/quant/telebot/test_hard_risk_pause.py: 14 passed
ruff on modified files: passed
```

This slice is accepted. Broader WP-04 execution-side work remains open and is
still governed by the overall do-not-resume disposition in this report.

### WP-04 Execution Lifecycle Replay And Recovery - Accepted Execution Slice

The execution-side lifecycle slice for WP-04 was implemented and accepted after
the monotonic-transition, replay, and watchdog persistence tests passed.
The accepted changes:

- keep a durable lifecycle state record on each execution session with
  monotonic transitions, owner, retry count, reason, policy version, and
  timestamps;
- persist lifecycle transitions to the execution WAL alongside session start
  and stop events;
- restore the latest lifecycle state during WAL replay, including the
  transition chain from `ACTIVE` to `INCIDENT` to `FLATTENING` to
  `FLAT_CONFIRMED` and terminal `PAUSED` states; and
- record watchdog-triggered flattening as lifecycle transitions so restart
  recovery does not lose ownership or terminal state.

Independent acceptance evidence:

```text
tests/quant_v2/test_wp04_lifecycle.py: 2 passed
tests/quant_v2/test_execution_service.py / tests/quant_v2/test_live_readiness.py / tests/quant/test_telebot_main_v2_handlers.py / tests/quant/telebot/test_hard_risk_pause.py: 95 passed
ruff on modified files: passed
```

This execution slice is accepted. Broader WP-04 work remains open where the
operator-reviewed clear/approve flow and any remaining recovery semantics are
still to be completed.

### WP-04 Telegram Reviewed Clear Path - Accepted Review Controls

The Telegram-side reviewed-clear path for WP-04 was implemented and accepted
after the operator approval tests passed. The accepted changes:

- record and normalize lifecycle-transition approver identity, approval count,
  reconciliation reference, and clear reason in `quant/telebot/models.py`;
- require reviewed lifecycle transitions to retain a durable pending state
  until the recorded clearance criteria are met;
- require two distinct live-mode approvals before the reviewed transition can
  be cleared, while still allowing demo-mode review to clear once the flat and
  reconciliation checks pass; and
- expose the reviewed clear action through `/review_transition` so the operator
  flow can move from pending review to cleared state in a durable way.

Independent acceptance evidence:

```text
tests/quant/test_telebot_main_v2_handlers.py: 22 passed
tests/quant/telebot/test_hard_risk_pause.py: 15 passed
ruff on modified files: passed
```

This reviewed-clear slice is accepted. Broader WP-04 work remains open where
additional execution/recovery semantics still need to be completed and audited.

### WP-05 Execution Outcome Taxonomy And Exactly-Once Fills - Accepted Slice

The execution outcome slice for WP-05 was implemented and accepted after the
idempotent replay and WAL persistence tests passed. The accepted changes:

- add a typed `ExecutionOutcome` taxonomy and immutable fill identifiers to
  `quant_v2/execution` so transport outcomes are distinct from economic fills;
- make `ExecutionResult` replay-aware by carrying `newly_filled_qty`,
  `request_id`, venue/fill identifiers, and replay metadata without breaking
  existing result consumers;
- stop double-applying economic fills on duplicate idempotency keys by replaying
  the original result with `newly_filled_qty=0` and preserving the original
  transport result for audit and telemetry;
- persist order execution payloads with outcome and identifier fields in the
  execution WAL, then restore them on boot so replay survives restarts; and
- keep paper-accounting updates keyed off economic fill quantity instead of the
  transport quantity so replayed orders do not mutate positions or PnL again.

Independent acceptance evidence:

```text
tests/quant_v2/test_day2_infra_patches.py / tests/quant_v2/test_execution_infra.py /
tests/quant_v2/test_binance_adapter.py / tests/quant_v2/test_execution_service.py /
tests/quant_v2/test_live_readiness.py: 92 passed
full local suite: 703 passed, 34 warnings
ruff on modified files: passed
```

This WP-05 slice is accepted. Broader WP-05 accounting-ledger work remains open
where append-only journals, balance-sheet reconciliation, and schema migration
work still need to be completed.

### WP-06 Accounting Ledger And Reconciliation - Accepted Ledger Core Slice

The accounting ledger core for WP-06 was implemented and accepted after the
golden ledger, replay, and reconciliation tests passed. The accepted changes:

- add an explicit append-only accounting schema in `quant_v2/accounting/`
  for orders, fills, cash movements, fees, funding, corrections, marks,
  lifecycle events, and projection checkpoints;
- make ledger events replay deterministically from an empty projection to any
  sequence boundary, including long, add, partial close, flip, fee, funding,
  correction, and deposit/withdrawal paths;
- preserve economic history while allowing corrections to reverse prior
  entries instead of editing them in place;
- mark imported legacy history as `LEGACY_UNVERIFIABLE` so pre-cutover history
  does not claim false precision; and
- reconcile ledger projection state against external adapter positions with
  configurable tolerances and a machine-readable blocked/ok report.

Independent acceptance evidence:

```text
tests/quant_v2/test_accounting_ledger.py: 5 passed
full local suite: 708 passed, 34 warnings
ruff on modified files: passed
```

This WP-06 ledger core slice is accepted. Broader WP-06 work remains open
where transactional projection persistence, live execution integration, and
schema migration hardening still need to be completed and audited.

### WP-06 Transactional Projection Persistence, Legacy Import, And Recovery - Accepted Slice

The transactional persistence slice for WP-06 was implemented and accepted
after the checkpoint-recovery tests, legacy import/reporting tests, and full
suite passed. The accepted changes:

- add an atomic `append_event_and_checkpoint()` path in `quant_v2/accounting`
  so ledger events and projection checkpoints commit together;
- restore the caller projection state if the transactional checkpoint write
  fails so retries do not inherit a false sequence advance;
- import historical events with a pre-cutover `LEGACY_UNVERIFIABLE` marker so
  older bars can be carried forward without overstating audit certainty;
- render reconciliation status as a human-readable operator report for ledger
  review and recovery workflows; and
- keep duplicate source-event delivery idempotent while still allowing a
  checkpoint retry to rebuild cleanly after a failure.

Independent acceptance evidence:

```text
tests/quant_v2/test_accounting_ledger.py: 7 passed
full local suite: 710 passed, 34 warnings
ruff on modified files: passed
```

This WP-06 transactional slice is accepted. Broader WP-06 work remains open
where live execution integration and explicit migration hardening still need to
be completed and audited.

### WP-07 Realistic Paper Execution And Cost Policy - Accepted Slice

The first WP-07 execution-cost slice was implemented and accepted after the
backtester, report, and full suite passed. The accepted changes:

- add a shared versioned execution cost policy in
  `quant_v2/execution/cost_policy.py` so research and future execution code can
  talk about the same cost assumptions;
- attribute fill-level costs across fees, spread, slippage, funding, latency,
  and impact instead of burying all drag in a single adjustment;
- keep the backtester gross PnL gross while carrying the cost buckets forward
  into the result object and HTML report; and
- render base, adverse, and severe cost scenarios so paper/replay evaluation
  can show whether a strategy survives worse execution conditions.

Independent acceptance evidence:

```text
tests/quant_v2/test_backtester.py: 24 passed
full local suite: 712 passed, 35 warnings
ruff on modified files: passed
```

This WP-07 slice is accepted. Broader WP-07 work remains open where live
adapter integration, venue calibration against retained observations, and
shadow/paper parity hardening still need to be completed and audited.

### WP-08 Canonical Data And Feature Contracts - Accepted Slice

The canonical-data slice for WP-08 was implemented and accepted after the
universe-ingress, manifest, label-isolation, and full-suite checks passed. The
accepted changes:

- make scheduled retrain ingest the universe dataset once and then build
  symbol-local feature frames instead of reassembling rows with ad hoc
  concatenation;
- enforce UTC multi-symbol canonicalization in the dataset validator and
  surface symbol-level continuity, duplicate, coverage, and digest metadata in
  the saved manifest;
- preserve symbol-local forward labeling and remove tail leakage across
  symbols in the retrain path;
- add point-in-time coverage checks that prove future perturbations on one
  symbol do not change earlier labels/features on another symbol; and
- persist richer immutable snapshot metadata, including requested/fetched
  symbols, requested/actual time ranges, schema versions, and content digests.

Independent acceptance evidence:

```text
tests/quant_v2/test_multi_symbol_data.py: 11 passed
tests/quant_v2/test_group_validation.py: 6 passed
tests/quant_v2/test_scheduled_retrain_candidates.py: 2 passed
full local suite: 715 passed, 35 warnings
ruff on modified files: passed
```

This WP-08 canonical-data slice is accepted. Broader WP-08 work remains open
where the explicit feature schema catalog and missing-data policy cleanup still
need to be completed and audited.

### WP-08 Context Missingness And Prepared-Panel Drop Contract - Accepted Slice

The multi-symbol context layer was tightened so warmup gaps stay visible until
the prepared dataset explicitly drops them, instead of silently converting
every undefined context value into a neutral default.

Accepted changes:

- `quant_v2/research/cross_sectional_features.py` now preserves warmup
  missingness for cross-sectional return and dispersion values while still
  treating zero-variance cross-sectional volume spread as a neutral zero;
- `quant_v2/research/regime_context.py` now preserves warmup missingness in the
  regime context fields instead of backfilling the initial window to zero;
- `quant_v2/research/group_validation.py` now drops rows with incomplete
  cross-sectional or regime context before returning the canonical prepared
  dataset; and
- regression tests now verify warmup missingness, prepared-panel completeness,
  and point-in-time stability under future perturbation.

Independent acceptance evidence:

```text
ruff check touched context/data-prep files: passed
tests/quant_v2/test_cross_sectional_features.py tests/quant_v2/test_regime_context.py tests/quant_v2/test_group_validation.py tests/quant_v2/test_portfolio_replay.py: 11 passed
tests/quant: 132 passed
```

This WP-08 slice is accepted. Broader WP-08 work remains open where the
feature-schema catalog and any remaining missing-data policy refinement still
need to be completed and audited.

### WP-08 Explicit Feature Schema Catalog And Provenance Slice - Accepted Slice

The feature-schema slice for WP-08 was implemented and accepted after the
catalog completeness, provenance metadata, and regression checks passed. The
accepted changes:

- add a canonical feature catalog in `quant/features/schema.py` with explicit
  group/source-module ownership for every feature column;
- make `quant/features/pipeline.py` validate the catalog, preserve the schema
  ordering, and attach catalog and missing-data policy metadata to feature
  frames;
- teach the optional funding-rate, open-interest, and liquidation feature
  modules to emit neutral fallback columns so the schema remains complete even
  when the optional upstream feeds are absent;
- carry feature catalog and missing-data policy metadata through v2 model
  training, artifact manifests, and manifest load-time validation; and
- add regression coverage for schema completeness, digest stability, metadata
  propagation, and the new neutral fallback columns.

Independent acceptance evidence:

```text
ruff check touched feature/trainer files: passed
tests/quant/test_features.py tests/quant_v2/test_model_stack.py tests/quant_v2/test_cross_pair_features.py tests/quant_v2/test_liquidation_proximity.py tests/quant_v2/test_order_book_features.py: 46 passed
tests/quant: 135 passed
tests/quant_v2/test_group_validation.py tests/quant_v2/test_build_universe_snapshot.py: 8 passed
tests/quant_v2/test_stage1_pipeline.py: 3 passed
```

This WP-08 slice is accepted. Broader WP-08 work remains open only if further
feature-gating or missing-data policy refinement is still desired beyond this
catalog/provenance cleanup.

### WP-09 Defensible Validation And Model Selection - Accepted Slice

The temporal-validation slice for WP-09 was implemented and accepted after the
timestamp-native fold plan, selection-risk, and full-suite checks passed. The
accepted changes:

- replace flattened `TimeSeriesSplit` validation with timestamp-native rolling
  and expanding monthly folds;
- reserve a final untouched temporal holdout and keep it out of threshold and
  half-life selection;
- compute recency weights from preserved timestamps and record effective sample
  size for each candidate half-life;
- track the attempted temporal trials in a ledger and derive a simple
  selection-risk summary from the actual attempts;
- refit promoted candidates on the approved development window using the
  selected recency half-life; and
- surface `trial_count` and `selection_risk` in the experiment score report so
  the downstream gate can see the trial burden explicitly.

Independent acceptance evidence:

```text
tests/quant_v2/test_temporal_validation.py: 2 passed
tests/quant_v2/test_scheduled_retrain_candidates.py: 2 passed
tests/quant_v2/test_group_validation.py: 6 passed
tests/quant_v2/test_run_group_validation.py: 2 passed
full local suite: 717 passed, 36 warnings
ruff on modified files: passed
```

This WP-09 temporal-validation slice is accepted. Broader WP-09 work remains
open where richer cost-adjusted metrics, fold-dispersion reporting, and
additional failure rules for singular or collapsed folds still need to be
completed and audited.

### WP-09 Validation Evidence, Cost Sensitivity, And Failure Semantics - Accepted Slice

The validation-report slice for WP-09 was implemented and accepted after the
group-validation, scorecard, and full-suite checks passed. The accepted
changes:

- add drawdown duration to the shared fold metrics so the report can surface
  time under water, not just drawdown depth;
- preserve trial-count, selection-risk, fold-dispersion, cost-sensitivity, and
  failure-reason evidence in the group-validation result object;
- compute actual adverse-cost scenarios from the held-out fold predictions
  rather than fabricating sensitivity values;
- record explicit failure semantics for empty folds, insufficient trade
  samples, and high symbol concentration while keeping existing synthetic
  fixtures valid; and
- propagate the richer evidence bundle into the experiment report and the
  persisted validation JSON, including the scorecard trial count and
  selection-risk digest.

Independent acceptance evidence:

```text
tests/quant_v2/test_group_validation.py: 6 passed
tests/quant_v2/test_run_group_validation.py: 2 passed
tests/quant_v2/test_experiment_score.py: 2 passed
full local suite: 717 passed, 36 warnings
ruff on modified files: passed
```

This WP-09 reporting slice is accepted. Broader WP-09 work remains open where
additional calibration, symbol-level calibration reporting, and any remaining
selection-governance refinements still need to be completed and audited.

### WP-10 Training, Refit, Artifact, And Inference Parity - Accepted Slice

The artifact and inference-parity slice for WP-10 was implemented and accepted
after the manifest-validation, strict feature-contract, parity, and full-suite
checks passed. The accepted changes:

- add a sidecar artifact manifest for trained models that records exact runtime
  provenance, feature schema, feature dtypes, calibration metadata, training
  metadata, and cryptographic checksums;
- reject model loads when the manifest is missing, the checksum is wrong, the
  recorded runtime differs from the current environment, or the feature/horizon
  contract is incompatible;
- require the predictor to fail closed on missing features, unexpected extra
  features, and dtype mismatches instead of silently fabricating inputs;
- remove zero-filling from the horizon ensemble so schema violations surface as
  explicit contract errors rather than being masked as neutral data; and
- persist manifested retrain artifacts through the scheduled retrain and
  backtester paths so the saved and loaded model stacks use the same validated
  contract.

Independent acceptance evidence:

```text
targeted model-stack / ensemble / retrain regressions: passed
full local suite: 724 passed, 46 warnings
ruff on modified files: passed
```

This WP-10 slice is accepted. Broader WP-10 work remains open where the final
freeze/refit governance, calibration policy hardening, and any remaining
promotion-facing artifact controls still need to be completed and audited.

### WP-10 Freeze, Refit, And Promotion-Eligibility Controls - Accepted Slice

The freeze/refit and promotion-eligibility slice for WP-10 was implemented and
accepted after the holdout-separation, manifest-backed promotion, and full-suite
checks passed. The accepted changes:

- separate development selection from final fitting so the frozen holdout is
  not touched until the configuration is locked;
- fail closed when no untouched holdout evidence is available instead of
  promoting on development evidence alone;
- record the final development and holdout windows, selected half-life, and
  frozen configuration state in the artifact metadata;
- require registry promotion to validate that every model artifact has a
  manifest and can be loaded under the current runtime before activation; and
- update promotion and cache tests to exercise real manifested bundles and
  isolated payload fixtures instead of placeholder artifacts.

Independent acceptance evidence:

```text
targeted scheduler / registry / telebot regressions: passed
full local suite: 724 passed, 46 warnings
ruff on modified files: passed
```

This WP-10 slice is accepted. Broader WP-10 work remains open where calibration
policy hardening and any remaining model-selection refinements still need to be
completed and audited.

### WP-10 Missing-Feature Fail-Closed Inference Contract - Accepted Slice

The feature and live-inference contract for WP-10 was tightened so missing
derived features are no longer fabricated as neutral zeros. The accepted
changes:

- make `quant/features/pipeline.py` drop rows with missing derived features
  instead of zero-filling them, preserving the point-in-time contract while
  keeping incomplete rows out of training data;
- make `quant_v2/telebot/signal_manager.py` reject missing model feature
  columns or missing feature values when constructing the final inference row;
- keep the existing inference fallback path closed on that error so incomplete
  feature rows degrade to HOLD rather than producing a fabricated trade; and
- add regressions proving the feature pipeline stays NaN-free after filtering,
  the signal manager rejects incomplete model rows, and portfolio replay still
  behaves deterministically under adverse scenario injection.

Independent acceptance evidence:

```text
ruff check touched feature/inference files: passed
tests/quant/test_features.py tests/quant_v2/test_signal_manager.py tests/quant_v2/test_group_validation.py: 43 passed
tests/quant: 132 passed
tests/quant_v2/test_portfolio_replay.py: 2 passed
```

This WP-10 slice is accepted. Broader WP-10 work remains open where the final
selection-governance and calibration-policy refinements still need to be
completed and audited.

### WP-10 Out-Of-Fold Threshold Calibration And Manifest Policy Slice - Accepted Slice

The calibration-policy slice for WP-10 was implemented and accepted after the
out-of-fold threshold metadata, manifest propagation, and regression checks
passed. The accepted changes:

- derive a probability threshold policy from out-of-fold development
  predictions inside the temporal retrain flow instead of hard-coding a fixed
  0.5 threshold in the retrain artifact metadata;
- persist the selected threshold, sweep bounds, sample count, and selection
  source alongside the model artifact so promotion evidence can distinguish
  calibrated thresholds from fallback defaults;
- keep the fold-local sigmoid calibration path explicit in the manifest as a
  separate calibration policy entry; and
- add regression coverage for the threshold-selection helper and the scheduled
  retrain candidate manifest.

Independent acceptance evidence:

```text
ruff check quant_v2/research/scheduled_retrain.py tests/quant_v2/test_scheduled_retrain_candidates.py: passed
tests/quant_v2/test_scheduled_retrain_candidates.py: 4 passed
tests/quant_v2/test_model_stack.py: 7 passed
tests/quant/test_features.py: 17 passed
```

This WP-10 slice is accepted. Broader WP-10 work remains open only if more
selection-governance or calibration refinements are still desired beyond this
threshold-policy provenance cleanup.

### WP-10 Operator Threshold Summary And Contract Visibility Slice - Accepted Slice

The operator-summary slice for WP-10 was implemented and accepted after the
threshold provenance was surfaced in the model contract summaries and admin
handler regressions passed. The accepted changes:

- extend the active-model manifest summary so it now reports the selected
  threshold and threshold-policy source alongside the image, feature schema,
  and dataset digest;
- extend the model contract helper used by candidate and version summaries so
  operators can see the calibrated threshold provenance directly in the bot
  output; and
- update the handler regressions to exercise the new threshold metadata
  without regressing the existing contract summary surface.

Independent acceptance evidence:

```text
ruff check quant/telebot/main.py tests/quant/test_telebot_main_v2_handlers.py: passed
tests/quant/test_telebot_main_v2_handlers.py: 35 passed
```

This WP-10 slice is accepted. Broader WP-10 work remains open only if more
selection-governance or calibration refinements are still desired beyond this
operator-visible contract cleanup.

### WP-10 Live Threshold Floor In Signal Threshold Resolution - Accepted Slice

The live-threshold slice for WP-10 was implemented and accepted after the
signal-manager threshold-floor regression passed. The accepted changes:

- let `quant_v2/telebot/signal_manager.py` inherit a calibrated threshold floor
  from the active model artifact manifest when one is present;
- preserve regime-based widening and the existing regime-2 conservatism while
  preventing the live decision threshold from falling below the calibrated
  artifact floor; and
- add a regression proving the artifact threshold floor influences the live
  regime threshold resolution path.

Independent acceptance evidence:

```text
ruff check quant_v2/telebot/signal_manager.py tests/quant_v2/test_signal_manager.py: passed
tests/quant_v2/test_signal_manager.py: 24 passed
```

This WP-10 slice is accepted. Broader WP-10 work remains open only if more
selection-governance or calibration refinements are still desired beyond this
artifact-floor parity cleanup.

### WP-10 Replay Threshold Floor In Executable Portfolio Replay - Accepted Slice

The replay-threshold slice for WP-10 was implemented and accepted after the
portfolio replay threshold-floor regression passed. The accepted changes:

- let `quant_v2/research/portfolio_replay.py` inherit a calibrated threshold
  floor from the model artifact manifest when a model replay actor is using
  the default threshold;
- preserve explicit non-default replay thresholds so test fixtures and
  experiment overrides continue to work as written; and
- add a regression proving the model replay actor defaults to the calibrated
  threshold floor when metadata is present.

Independent acceptance evidence:

```text
ruff check quant_v2/research/portfolio_replay.py tests/quant_v2/test_portfolio_replay.py: passed
tests/quant_v2/test_portfolio_replay.py -k threshold_floor: 1 passed
```

This WP-10 slice is accepted. Broader WP-10 work remains open only if more
selection-governance or calibration refinements are still desired beyond this
replay-floor parity cleanup.

### WP-11 Executable Portfolio Replay - Accepted Slice

The deterministic executable portfolio replay slice for WP-11 was implemented
and accepted after the replay-determinism, scenario-injection, and full-suite
checks passed. The accepted changes:

- add a deterministic event-driven portfolio replay engine in
  `quant_v2/research/portfolio_replay.py` that exercises the production
  contracts for signal generation, allocation, planner sizing, cost policy,
  ledger append, and reconciliation;
- preserve isolated state per candidate, incumbent, and benchmark actor while
  replaying synchronized as-of market events across symbols;
- cache sorted dataset state and per-symbol feature frames so replay stays
  deterministic without repeated re-sorting or unnecessary feature rebuilds;
- make replay artifacts content-addressed from a stable manifest that excludes
  transient generation timestamps;
- support deterministic scenario injection for spread widening, latency,
  partial fills, venue rejection, data gaps, mark jumps, and restart
  conditions; and
- add regression coverage for deterministic digests, content-addressed output,
  and adverse-scenario behavior.

Independent acceptance evidence:

```text
tests/quant_v2/test_portfolio_replay.py: 2 passed
full local suite: 726 passed, 46 warnings
ruff on modified files: passed
```

This WP-11 slice is accepted. Broader WP-11 work remains open where replay
calibration, block-aware evaluation, and shadow-evaluation policy hardening
still need to be completed and audited.

### WP-12 Shadow Evaluation And Statistical Decision Policy - Accepted Slice

The replay-backed shadow-evaluation slice for WP-12 was implemented and
accepted after the replay-summary, coverage, bootstrap, and full-suite checks
passed. The accepted changes:

- route quarantine summaries through an executable replay path in
  `quant_v2/research/model_evaluator.py` instead of relying only on directional
  row aggregation;
- reconstruct candidate, incumbent, and transparent cash-baseline replay actors
  from resolved shadow-decision evidence and execute them on a deterministic
  replay stream;
- merge same-direction overlapping shadow decisions into non-overlapping
  episodes so repeated horizon-aligned predictions are not counted as separate
  round trips;
- track calendar-day coverage, trading-day coverage, symbol coverage, replay
  episode counts, and block-bootstrap confidence intervals for paired replay
  comparisons; and
- require the replay summary to surface benchmark-relative edge, absolute edge,
  and coverage diagnostics for promotion decisions when the strict policy asks
  for them.

Independent acceptance evidence:

```text
tests/quant_v2/test_model_evaluator.py: 7 passed
tests/quant_v2/test_portfolio_replay.py: 2 passed
full local suite: 727 passed, 46 warnings
ruff on modified files: passed
```

This WP-12 slice is accepted. Broader WP-12 work remains open where richer
benchmark families, adverse-cost replay, and additional selection-policy
hardening still need to be completed and audited.

### WP-12 Threshold-Floor Replay Summary And Manifest Binding - Accepted Slice

The manifest-bound replay summary slice for WP-12 was implemented and accepted
after the shadow-evaluation regression and replay-summary checks passed. The
accepted changes:

- preserve `model_version_id` and `baseline_version_id` while normalizing
  shadow-evaluation rows so replay summaries can resolve back to the registered
  artifact versions;
- load the registered artifact manifest for candidate and incumbent versions
  when a registry root is available and derive the replay threshold floor from
  the manifest's selected threshold policy;
- propagate the resolved threshold floors into the replay actors and expose the
  threshold provenance in the returned quarantine summary; and
- add a regression proving `summarize_quarantine(..., registry_root=...)`
  picks up the candidate and incumbent threshold floors from registered
  artifacts.

Independent acceptance evidence:

```text
ruff check quant_v2/research/model_evaluator.py tests/quant_v2/test_model_evaluator.py: passed
tests/quant_v2/test_model_evaluator.py: 8 passed
```

This WP-12 slice is accepted. Broader WP-12 work remains open where richer
benchmark families, adverse-cost replay, and additional selection-policy
hardening still need to be completed and audited.

### WP-12 Registry Evaluation Threshold Provenance Persistence - Accepted Slice

The registry-persistence slice for WP-12 was implemented and accepted after
the evaluator-registry regression passed. The accepted changes:

- persist the replay threshold policy into the registry paper-evaluation
  record so selection evidence remains durable after the summary returns;
- keep threshold provenance available for later registry inspection instead of
  only surfacing it in transient evaluation output; and
- add a regression proving the registry paper-evaluation metrics contain the
  candidate threshold floor after evaluation recording.

Independent acceptance evidence:

```text
ruff check quant_v2/research/model_evaluator.py tests/quant_v2/test_model_evaluator.py: passed
tests/quant_v2/test_model_evaluator.py: 8 passed
```

This WP-12 slice is accepted. Broader WP-12 work remains open where richer
benchmark families, adverse-cost replay, and additional selection-policy
hardening still need to be completed and audited.

### WP-13 Registry And Promotion Governance - Accepted Slice

The append-only registry, promotion-approval, rollback-readiness, and registry-
backed model-loading slice for WP-13 was implemented and accepted after the
registry-event, approval-gate, backtester-load, rollback-preflight, and
full-suite checks passed. The accepted changes:

- add an append-only registry event log in `quant_v2/model_registry.py` for
  version registration, status updates, active-pointer changes, promotion
  requests, and promotion approvals;
- derive the active model pointer from event history instead of trusting a
  mutable snapshot file as the source of truth;
- preserve the existing registry JSON files as projections while making the
  event stream authoritative for pointer replay;
- add a public activation-readiness check that validates artifacts before
  rollback or other control-plane pointer changes;
- add promotion approval recording with distinct approver identities and a
  two-person gate that can be enabled by policy for live activation; and
- extend the Telegram admin surface with `/model_approve` plus rollback
  preflight checks so transition commands validate readiness before switching
  pointers;
- remove the last direct `active.json` model-loading path from the backtester
  so registry pointer replay remains the source of truth for active-model
  selection.

Independent acceptance evidence:

```text
tests/quant_v2/test_model_registry.py: 11 passed
tests/quant/test_telebot_main_v2_handlers.py: 24 passed
tests/quant_v2/test_backtester.py: 25 passed
full local suite: 731 passed, 46 warnings
ruff on modified files: passed
```

This WP-13 slice is accepted. Broader WP-13 work remains open where the final
transition-governance refinements, rollback policy hardening, and any remaining
operator-control commands still need to be completed and audited.

### WP-13 Registry Pointer Threshold Contract - Accepted Slice

The pointer-contract slice for WP-13 was implemented and accepted after the
registry-pointer and admin-surface regressions passed. The accepted changes:

- add `artifact_threshold` to the active-pointer projection so the registry
  pointer carries the calibrated threshold floor alongside the manifest hash,
  image, schema, and dataset digest;
- derive that pointer threshold from the artifact manifest's selected threshold
  policy when a promoted model is loaded into the registry;
- show the pointer threshold in the active-model Telegram summary so operators
  can inspect the selected floor without digging through the manifest; and
- extend registry and bot regressions to prove the pointer threshold is present
  and consistent with the promotion event payload.

Independent acceptance evidence:

```text
ruff check quant_v2/model_registry.py quant/telebot/main.py tests/quant_v2/test_model_registry.py tests/quant/test_telebot_main_v2_handlers.py: passed
tests/quant_v2/test_model_registry.py tests/quant/test_telebot_main_v2_handlers.py: 50 passed
```

This WP-13 slice is accepted. Broader WP-13 work remains open where the final
transition-governance refinements, rollback policy hardening, and any remaining
operator-control commands still need to be completed and audited.

### WP-13 Rollback Target Hardening - Accepted Slice

The rollback-target hardening slice for WP-13 was implemented and accepted
after the rollback-governance, registry, and Telegram handler regressions
passed. The accepted changes:

- add `validate_rollback_target()` to `quant_v2/model_registry.py` so rollback
  candidates must be the currently previous active version, not merely any
  activation-ready model;
- tighten registry rollback so the previous active record is revalidated before
  the pointer is rewound;
- constrain the Telegram `/model_rollback` command to the previous active
  version by default and reject explicit targets that are not the attested
  previous active model; and
- update rollback regressions to cover both the no-previous-active case and
  the ready-but-non-previous rejection path.

Independent acceptance evidence:

```text
ruff check quant_v2/model_registry.py quant/telebot/main.py tests/quant_v2/test_model_registry.py tests/quant/test_telebot_main_v2_handlers.py: passed
tests/quant_v2/test_model_registry.py tests/quant/test_telebot_main_v2_handlers.py: 52 passed
```

This WP-13 slice is accepted. Broader WP-13 work remains open where the final
transition-governance refinements, rollback policy hardening, and any remaining
operator-control commands still need to be completed and audited.

### WP-13 Registry Event Integrity And Registry-Only Model Resolution - Accepted Slice

The registry-event chain integrity and registry-only model-resolution slice for
WP-13 was implemented and accepted after the event-chain tamper, registry
backtester, and full-suite checks passed. The accepted changes:

- verify the registry event-chain hash links when loading the append-only
  registry log so tampering is detected instead of silently ignored;
- remove the last direct `active.json` reader from the backtester and resolve
  the active model through `ModelRegistry` exclusively;
- prove that deleting the mutable `active.json` snapshot does not affect the
  event-derived active pointer; and
- add a regression proving a corrupted registry event log is rejected before
  active-pointer replay proceeds.

Independent acceptance evidence:

```text
tests/quant_v2/test_model_registry.py: 12 passed
tests/quant_v2/test_backtester.py: 25 passed
full local suite: 732 passed, 46 warnings
ruff on modified files: passed
```

This WP-13 slice is accepted. Broader WP-13 work remains open where the final
transition-governance refinements, rollback policy hardening, and any remaining
operator-control commands still need to be completed and audited.

### WP-13 Registry Event Audit Surface - Accepted Slice

The registry-event audit surface slice for WP-13 was implemented and accepted
after the registry-event-history, approval-history, and full-suite checks
passed. The accepted changes:

- add an authenticated `/model_events` Telegram command that renders the
  append-only registry history with active-pointer context;
- surface the latest registry transition history without depending on mutable
  patch helpers or direct file edits; and
- fail closed when registry event history is tampered so operator inspection
  does not silently trust a broken chain.

Independent acceptance evidence:

```text
tests/quant/test_telebot_main_v2_handlers.py: 26 passed
tests/quant_v2/test_model_registry.py: 12 passed
full local suite: 734 passed, 46 warnings
ruff on modified files: passed
```

This WP-13 slice is accepted. Broader WP-13 work remains open where the final
transition-governance refinements, rollback policy hardening, and any remaining
operator-control commands still need to be completed and audited.

### WP-13 Evidence-Bound Promotion Approvals - Accepted Slice

The evidence-bound promotion slice for WP-13 was implemented and accepted after
the approval-expiry, evidence-gated promotion, command-surface, and full-suite
checks passed. The accepted changes:

- require `record_promotion_approval(...)` to carry a non-empty evidence digest;
- validate promotion approvals against the same evidence digest and live scope
  when the two-person gate is enabled;
- reject promotion when the gate is enabled but no evidence digest is supplied;
- reject promotion when approvals are expired or do not match the promoted
  evidence digest;
- allow the Telegram promotion command to pass the evidence digest through to
  the registry; and
- let operators record an optional approval expiry in the Telegram approval
  command while exposing it in the registry event history view.

Independent acceptance evidence:

```text
tests/quant_v2/test_model_registry.py: 13 passed
tests/quant/test_telebot_main_v2_handlers.py: 29 passed
full local suite: 738 passed, 46 warnings
ruff on modified files: passed
```

This WP-13 slice is accepted. Broader WP-13 work remains open where the final
transition-governance refinements, rollback policy hardening, and any remaining
operator-control commands still need to be completed and audited.

### WP-13 Active-Manifest Contract Binding - Accepted Slice

The active-manifest contract binding slice for WP-13 was implemented and
accepted after the manifest-summary, promotion-manifest, and full-suite checks
passed. The accepted changes:

- expose the active artifact manifest summary through `ModelRegistry` so the
  admin status view can show runtime image and feature-contract bindings;
- include artifact manifest metadata in the promotion event payload so the
  registry records the evidence bundle that activated the model; and
- display the current active model's manifest image reference, feature schema
  digest, and dataset digest in the Telegram `model_active` command.

Independent acceptance evidence:

```text
tests/quant_v2/test_model_registry.py: 14 passed
tests/quant/test_telebot_main_v2_handlers.py: 30 passed
full local suite: 740 passed, 46 warnings
ruff on modified files: passed
```

This WP-13 slice is accepted. Broader WP-13 work remains open where the final
transition-governance refinements, rollback policy hardening, and any remaining
operator-control commands still need to be completed and audited.

### WP-13 Terminal-State Transition Controls - Accepted Slice

The terminal-state transition slice for WP-13 was implemented and accepted after
the reject/expire command, terminal-guard, and full-suite checks passed. The
accepted changes:

- add explicit registry controls for rejecting and expiring model versions;
- prevent promotion and rollback activation of rejected or expired versions;
- expose `/model_reject` and `/model_expire` Telegram commands so terminal
  states are governed instead of implied; and
- update the promotion runbook to describe terminal candidate handling.

Independent acceptance evidence:

```text
tests/quant_v2/test_model_registry.py: 15 passed
tests/quant/test_telebot_main_v2_handlers.py: 33 passed
full local suite: 744 passed, 46 warnings
ruff on modified files: passed
```

This WP-13 slice is accepted. Broader WP-13 work remains open where the final
transition-governance refinements, rollback policy hardening, and any remaining
operator-control commands still need to be completed and audited.

### WP-13 Active-Pointer Contract Binding - Accepted Slice

The active-pointer contract binding slice for WP-13 was implemented and
accepted after the pointer-contract, status-summary, and full-suite checks
passed. The accepted changes:

- widen the active pointer record so it carries the bundle's manifest, image,
  feature-schema, and dataset digests when the artifact contract is available;
- write those contract fields into the active-pointer event payload during
  promotion so replay preserves the contract binding;
- surface the pointer-level contract evidence in the Telegram `model_active`
  status view; and
- keep fallback pointer derivation tolerant of placeholder test fixtures while
  still binding real promoted versions to their validated artifact contract.

Independent acceptance evidence:

```text
tests/quant_v2/test_model_registry.py: 15 passed
tests/quant/test_telebot_main_v2_handlers.py: 33 passed
full local suite: 744 passed, 46 warnings
ruff on modified files: passed
```

This WP-13 slice is accepted. Broader WP-13 work remains open where the final
transition-governance refinements, rollback policy hardening, and any remaining
operator-control commands still need to be completed and audited.

### WP-13 Candidate And Version Contract Summary - Accepted Slice

The candidate/version contract summary slice for WP-13 was implemented and
accepted after the contract-summary, candidate-list, and full-suite checks
passed. The accepted changes:

- surface artifact contract summaries in the registered-version and candidate
  listing commands so operators can inspect image, feature-schema, and dataset
  digests without opening the files manually;
- keep the existing manifest-binding path intact while making the operator
  lists more transparent about which contract the candidate actually carries;
  and
- preserve graceful fallback messaging when a contract cannot be loaded in a
  test fixture or incomplete artifact set.

Independent acceptance evidence:

```text
tests/quant/test_telebot_main_v2_handlers.py: 34 passed
tests/quant_v2/test_model_registry.py: 15 passed
full local suite: 745 passed, 46 warnings
ruff on modified files: passed
```

This WP-13 slice is accepted. Broader WP-13 work remains open where the final
transition-governance refinements, rollback policy hardening, and any remaining
operator-control commands still need to be completed and audited.

### WP-13 Registered-Version Contract Summary - Accepted Slice

The registered-version contract summary slice for WP-13 was implemented and
accepted after the version-list, contract-summary, and full-suite checks
passed. The accepted changes:

- surface artifact contract summaries in the registered model-version listing
  so the admin status view and version registry view remain aligned; and
- preserve graceful fallback text when a version's artifact contract cannot be
  loaded in a fixture or incomplete test bundle.

Independent acceptance evidence:

```text
tests/quant/test_telebot_main_v2_handlers.py: 35 passed
tests/quant_v2/test_model_registry.py: 15 passed
full local suite: 746 passed, 46 warnings
ruff on modified files: passed
```

This WP-13 slice is accepted. Broader WP-13 work remains open where the final
transition-governance refinements, rollback policy hardening, and any remaining
operator-control commands still need to be completed and audited.

### WP-14 Correlation Propagation And Durable Event Chain - Accepted Slice

The first WP-14 observability slice was implemented and accepted after the
execution command boundary, route-audit path, and durable WAL replay chain were
updated to carry a stable `correlation_id` end to end.

Accepted changes:

- `quant_v2/execution/main.py` now forwards the incoming bus correlation ID into
  `_cmd_route_signals()` and persists it alongside accepted fills in the WAL.
- `quant_v2/execution/service.py` now accepts an optional correlation ID for
  routing, includes it on structured `RouteAuditEvent` records, and threads it
  through every route-audit emission path.
- `quant_v2/execution/state_wal.py` now stores `correlation_id` on order
  execution entries for replay and reconstruction.
- Regression tests now assert correlation propagation in the execution service,
  the WAL order payload, and the server command path.

Independent verification result:

```text
ruff check touched execution and regression files: passed
targeted regression set: 74 passed
full local suite: 746 passed, 46 warnings
```

This WP-14 slice is accepted. Broader WP-14 work remains open where the stage
telemetry, data-health probes, external audit sink, and measured SLIs still need
to be completed and audited.

### WP-14 Stage Timing Telemetry - Accepted Slice

The second WP-14 observability slice was implemented and accepted after the
command path and routed execution path began emitting structured timing events
for planning, order execution, routing, ledger persistence, and post-fill
refresh.

Accepted changes:

- `quant_v2/execution/service.py` now emits structured stage telemetry for
  planning, per-order execution, and overall route completion through an
  optional callback.
- `quant_v2/execution/main.py` now emits structured stage telemetry for command
  parse, routing call, ledger commit, and post-fill refresh.
- Regression tests now assert that stage telemetry carries correlation IDs,
  durations, and the expected stage sequence in both the service and server
  paths.

Independent verification result:

```text
ruff check touched execution and regression files: passed
targeted regression set: 61 passed
full local suite: 748 passed, 46 warnings
```

This WP-14 slice is accepted. Broader WP-14 work remains open where the
data-health probes, external audit sink, and measured SLIs still need to be
completed and audited.

### WP-14 Data Health Dashboard And Append-Only Audit Sink - Accepted Slice

The third WP-14 observability slice was implemented and accepted after the
validation runner began surfacing manifest-derived dataset health and writing an
append-only JSONL audit trail alongside the run summary.

Accepted changes:

- `quant_v2/monitoring/health_dashboard.py` now derives data-health metrics
  from the dataset manifest, including freshness, duplicate rows, gap counts,
  symbol coverage, and failed symbols.
- `quant_v2/monitoring/health_dashboard.py` now folds the data-health status
  into the run-health dashboard and summary text.
- `quant_v2/monitoring/audit_sink.py` adds a minimal append-only JSONL sink for
  observability artifacts.
- `quant_v2/research/run_group_validation.py` now persists an append-only audit
  JSONL file next to each run report and records that path in the health
  dashboard payload.
- Regression tests now verify the data-health summary, append-only audit file,
  and the run-group validation artifact path.

Independent verification result:

```text
ruff check touched monitoring, research, and regression files: passed
targeted regression set: 4 passed
full local suite: 748 passed, 46 warnings
```

This WP-14 slice is accepted. Broader WP-14 work remains open where the
measured service-level indicators and any remaining runtime-side external audit
integration still need to be completed and audited.

### WP-14 Runtime Resource Health - Accepted Slice

The fourth WP-14 observability slice was implemented and accepted after the
validation dashboard began surfacing measured process and host-resource
indicators for the current runtime.

Accepted changes:

- `quant_v2/monitoring/health_dashboard.py` now collects optional runtime
  process and host resource metrics, including CPU, memory, disk, RSS, and open
  file counts.
- `quant_v2/monitoring/health_dashboard.py` now folds runtime-resource status
  into the run-health dashboard and rendered summary text.
- `quant_v2/monitoring/health_dashboard.py` now records runtime-resource
  entries in the append-only audit JSONL trail.
- Regression tests now verify the runtime resource snapshot, the degraded
  thresholds, and the preserved dashboard/audit artifact behavior.

Independent verification result:

```text
ruff check touched monitoring and regression files: passed
targeted regression set: 5 passed
full local suite: 749 passed, 45 warnings
```

This WP-14 slice is accepted. Broader WP-14 work remains open where synthetic
data/provider probes, queue-lag measurements, and any remaining runtime-side
external audit integration still need to be completed and audited.

### WP-14 Synthetic Provider Probe And Stale-Data Circuit Breaker - Accepted Slice

The fifth WP-14 observability slice was implemented and accepted after the
validation runner began performing synthetic market-data provider probes and
surfacing stale-data circuit-breaker status in the report and audit trail.

Accepted changes:

- `quant_v2/monitoring/provider_probes.py` adds a synthetic market-data probe
  using the existing market-data client contract.
- `quant_v2/monitoring/health_dashboard.py` now folds provider-probe health
  into the data-health dashboard, run summary, and audit JSONL trail.
- `quant_v2/research/run_group_validation.py` now records provider-probe
  results in the validation report when a market-data client is available.
- Regression tests now verify a stale provider trips the circuit breaker and a
  fresh provider stays healthy through the report pipeline.

Independent verification result:

```text
ruff check touched monitoring, research, and regression files: passed
targeted regression set: 5 passed
full local suite: 749 passed, 45 warnings
```

This WP-14 slice is accepted. Broader WP-14 work remains open where queue-lag
measurements and any remaining runtime-side external audit integration still
need to be completed and audited.

### WP-14 Execution Backlog, WAL, Reconciliation, And Redis Memory Health - Accepted Slice

The sixth WP-14 observability slice was implemented and accepted after the live
execution server began surfacing durable queue, WAL, reconciliation, and Redis
memory indicators through the operational health payload and audit trail.

Accepted changes:

- `quant_v2/execution/redis_bus.py` now exposes a queue-health snapshot with
  pending-count, backlog, pending-age, and stream freshness indicators.
- `quant_v2/execution/state_wal.py` now exposes a WAL freshness snapshot with
  append-only depth and latest-entry age indicators.
- `quant_v2/execution/main.py` now aggregates stream, WAL, reconciliation, and
  Redis memory signals into a single operational health payload.
- `quant_v2/monitoring/health_dashboard.py` now folds execution-health data
  into the run dashboard, summary text, and append-only audit JSONL trail.
- Regression tests now verify the queue-lag snapshot, WAL freshness snapshot,
  runtime server health payload, and dashboard extraction/summary behavior.

Independent verification result:

```text
ruff check touched execution, monitoring, and regression files: passed
targeted regression set: 16 passed
full local suite: 752 passed, 45 warnings
```

This WP-14 slice is accepted. Broader WP-14 work remains open where
network/DNS latency, container restart telemetry, and database lock-time
signals still need to be completed and audited.

### WP-14 Model Diagnostics And Measured SLI Baseline - Accepted Slice

The seventh WP-14 observability slice was implemented and accepted after the
model-evaluation path began surfacing calibrated probability drift, turnover,
realized cost, and replay concentration diagnostics, and the run-health
dashboard began exporting an initial measured SLI baseline.

Accepted changes:

- `quant_v2/research/model_evaluator.py` now emits a performance-diagnostics
  block with candidate/incumbent probability calibration, paired probability
  drift, turnover, realized cost, blocked intents, and concentration.
- `quant_v2/monitoring/health_dashboard.py` now derives a measured
  service-level baseline from current runtime, execution, and diagnostics
  values and records it in the run dashboard, summary text, and audit JSONL
  trail.
- Regression tests now verify the new model-diagnostics block and the
  baseline artifact targets derived from measured values.

Independent verification result:

```text
ruff check touched model-evaluator, monitoring, and regression files: passed
targeted regression set: 10 passed
full local suite: 755 passed, 45 warnings
```

This WP-14 slice is accepted.

### WP-14 Feature Drift Linkage, Missingness, And Quarantine Surface - Accepted Slice

The latest WP-14 observability slice was implemented and accepted after the run
dashboard began surfacing feature-drift alerts together with feature
missingness and symbol quarantine data from the latest manifests and evaluator
diagnostics.

Accepted changes:

- `quant_v2/monitoring/health_dashboard.py` now surfaces `feature_drift_alert`
  across the execution-health dashboard, run-health payload, run summary, and
  append-only audit artifact.
- `quant_v2/monitoring/health_dashboard.py` now exposes
  `feature_missingness_by_column`, `feature_missingness_max`,
  `feature_missingness_avg`, and `quarantined_symbols` from the dataset
  manifest.
- `quant_v2/monitoring/health_dashboard.py` now carries those measured values
  into the service-level baseline so the initial SLI targets reflect the
  actual observed data quality state.
- `quant_v2/research/model_evaluator.py` now feeds the diagnostics block that
  the dashboard consumes for candidate/incumbent drift and calibration review.
- Regression tests now verify the feature-drift alert, missingness summary,
  quarantine surfacing, and audit artifact line count.

Independent verification result:

```text
ruff check touched monitoring and regression files: passed
targeted regression set: 3 passed
full local suite: 755 passed, 45 warnings
```

This WP-14 slice is accepted.

### WP-14 API Latency, Rate Limits, And Provider Throttling Surface - Accepted Slice

The final WP-14 observability slice was implemented and accepted after the
market-data client began exposing current request-weight state and the
validation report began carrying those live rate-limit signals alongside the
existing provider latency measurements.

Accepted changes:

- `quant/data/binance_client.py` now exposes `get_rate_limit_snapshot()` with
  used weight, hard limit, headroom, pressure fraction, and throttle interval
  fields derived from the live Binance client state.
- `quant_v2/monitoring/provider_probes.py` now captures that rate-limit
  snapshot when the market-data client provides it and surfaces provider
  latency plus rate-limit status together in the probe payload.
- `quant_v2/monitoring/health_dashboard.py` now threads the provider rate-limit
  snapshot into the data-health dashboard, the measured SLI baseline, and the
  human-readable run summary.
- `quant_v2/research/run_group_validation.py` already routes the provider probe
  through the report, so the new rate-limit telemetry reaches the persisted run
  artifact without extra orchestration wiring.
- Regression tests now verify the client snapshot, provider probe surfacing,
  dashboard propagation, and report-path persistence.

Independent verification result:

```text
ruff check touched Binance client, monitoring, and regression files: passed
targeted regression set: 10 passed
full local suite: 756 passed, 45 warnings
```

This WP-14 slice is accepted. WP-14 is now closed at the repository level.
Broader deployment-readiness work continues under WP-15.

### WP-15 Deployment Readiness Harness - Accepted Slice

The first WP-15 deployment-readiness slice was implemented and accepted after
the repository gained an idempotent rollout harness that validates the
immutable image, environment manifest, model compatibility, Redis/DNS
readiness, writable paths, open positions, and database rehearsal gates before
recording a deployment decision.

Accepted changes:

- `tools/deploy_readiness.py` adds a manifest-driven readiness harness that
  validates the immutable image reference, attestation, compose configuration,
  active model compatibility, Redis secret hygiene, DNS latency, SQLite lock
  latency, writable path access, open-position state, and database migration
  rehearsal results.
- `tools/deploy_readiness.py` records a stable deployment artifact that binds
  image, configuration, model, migration, smoke, and operator-approval data to
  an idempotency key so repeated runs do not mutate the record.
- `tools/deploy_readiness.py` supports rollback mode by explicit rollback image
  and model targets rather than reusing the live pointer, and it now issues a
  compose stack stop before the rollback bring-up so the rehearsal reflects the
  intended stop-then-restore sequence.
- `scripts/deploy_readiness.ps1` provides the operator-facing wrapper for the
  same harness on Windows hosts.
- Regression tests now verify preflight success, idempotent record reuse,
  deploy-mode service ordering, smoke execution, and rollback-target handling.

Independent verification result:

```text
ruff check touched deployment harness and regression files: passed
targeted regression set: 3 passed
full local suite: 759 passed, 45 warnings
```

This WP-15 slice is accepted. Broader WP-15 rollout, staging, and production
deployment rehearsal work remains open.

### WP-15 Failure-Aware Staged Rollout And Rollback Rehearsal - Accepted Slice

The second WP-15 deployment-readiness slice was implemented and accepted after
the rollout harness became failure-aware: a failed service start or smoke check
now triggers a compose stack down rehearsal and records that rollback attempt in
the deployment artifact.

Accepted changes:

- `tools/deploy_readiness.py` now routes execution through a staged rollout
  helper that captures compose-down results on rollback mode and on deploy-mode
  failures.
- `tools/deploy_readiness.py` now records a `rollout` block that includes the
  staged service results, smoke result, compose-down result, and failure class
  when the deploy sequence fails.
- `tools/deploy_readiness.py` now treats smoke failures as rollback-worthy and
  issues a stack-down attempt before persisting the record.
- Regression tests now verify the deploy success path still runs in order and
  the smoke-failure path records a degraded rollout plus a stack-down attempt.

Independent verification result:

```text
ruff check touched deployment harness and regression files: passed
targeted regression set: 4 passed
full local suite: 760 passed, 45 warnings
```

This WP-15 slice is accepted. Broader WP-15 rollout, staging, and production
deployment rehearsal work remains open.

### WP-15 Deployment Record Integrity And Rollback Hash Binding - Accepted Slice

The deployment record integrity follow-up for WP-15 was implemented and
accepted after the rollback-target fields were bound into the deployment record
before the record hash was calculated, preventing the artifact from diverging
from its own digest.

Accepted changes:

- `tools/deploy_readiness.py` now includes rollback target image and model
  fields in the stable deployment record before calculating `record_sha256`.
- `tests/infra/test_wp15_deploy_readiness.py` now asserts the recorded digest
  matches the canonical hash of the full persisted payload, including rollback
  fields.

Independent verification result:

```text
ruff check touched deployment harness and regression files: passed
targeted regression set: 4 passed
full local suite: 760 passed, 45 warnings
```

This WP-15 slice is accepted. Broader WP-15 rollout, staging, and production
deployment rehearsal work remains open.

### WP-15 Staged Service Plan And Grouped Rollout Sequencing - Accepted Slice

The staged rollout harness was extended to accept an explicit stage plan, so
deployment order can be expressed as grouped stages instead of only a flat
service list. The new stage results are recorded alongside the rollout record.

Accepted changes:

- `tools/deploy_readiness.py` now accepts an optional `stage_plan` manifest
  entry and normalizes it into grouped rollout stages.
- `tools/deploy_readiness.py` now records per-stage names and service batches
  in the deployment record so the rollout history shows grouped sequencing.
- `tests/infra/test_wp15_deploy_readiness.py` now verifies the default grouped
  stage plan, a custom stage plan, and the flattened service ordering.

Independent verification result:

```text
ruff check touched deployment harness and regression files: passed
targeted regression set: 5 passed
full local suite: 761 passed, 45 warnings
```

This WP-15 slice is accepted. Broader WP-15 rollout, staging, and production
deployment rehearsal work remains open.

### WP-15 Deterministic Smoke Replay Comparison - Accepted Slice

The deployment harness now compares smoke replay output against an expected
payload or digest so the post-deploy check proves the output contract rather
than merely executing the command.

Accepted changes:

- `tools/deploy_readiness.py` now accepts expected smoke stdout text and/or
  expected stdout/stderr digests in the deployment manifest.
- `tools/deploy_readiness.py` now records `stdout_matches_expected` and
  `stderr_matches_expected` in the smoke result and degrades the rollout if the
  replay output does not match the contract.
- `tests/infra/test_wp15_deploy_readiness.py` now verifies both the matching
  smoke path and a mismatching smoke path that degrades the rollout.

Independent verification result:

```text
ruff check touched deployment harness and regression files: passed
targeted regression set: 6 passed
full local suite: 762 passed, 45 warnings
```

This WP-15 slice is accepted. Broader WP-15 rollout, staging, and production
deployment rehearsal work remains open.

### WP-15 Migration Rehearsal Table Signature And Drift Surface - Accepted Slice

The migration rehearsal now records the actual table names present in the
forward and rollback copies, along with a stable schema signature and explicit
drift set, so the rehearsal can be compared beyond a simple count.

Accepted changes:

- `tools/deploy_readiness.py` now stores `forward_table_names`,
  `rollback_table_names`, `table_name_drift`, and `schema_signature_sha256`
  in the migration rehearsal result.
- `tests/infra/test_wp15_deploy_readiness.py` now verifies the forward and
  rollback table sets match and that the drift set remains empty.

Independent verification result:

```text
ruff check touched deployment harness and regression files: passed
targeted regression set: 6 passed
full local suite: 762 passed, 45 warnings
```

This WP-15 slice is accepted. Broader WP-15 rollout, staging, and production
deployment rehearsal work remains open.

### WP-15 Compose Read-Only Policy Inspection - Accepted Slice

The deployment harness now inspects the rendered compose configuration and
records whether each service is locked read-only, so the readiness gate can
surface writable runtime surfaces before rollout.

Accepted changes:

- `tools/deploy_readiness.py` now parses the rendered compose config, counts
  services with `read_only: true`, and records the writable service list plus
  any missing read-only declarations in `read_only_policy`.
- `tools/deploy_readiness.py` now marks compose readiness degraded when the
  rendered config cannot be parsed or when services are left writable.
- `tests/infra/test_wp15_deploy_readiness.py` now exercises a fully
  read-only compose render and a writable-service warning path alongside the
  existing rollout coverage.

Independent verification result:

```text
ruff check touched deployment harness and regression files: passed
targeted regression set: 7 passed
wider infra regression band: 42 passed
full local suite: 762 passed, 45 warnings
```

This WP-15 slice is accepted. Broader WP-15 rollout, staging, and production
deployment rehearsal work remains open.

## Executive Summary

The June 18 event was not primarily caused by insufficient Ubuntu compute capacity. The host was healthy during this audit: approximately 90% CPU idle, negligible I/O wait, no container restarts, no OOM events, and modest memory usage. The observed degradation is principally explained by model-validation defects, risk-policy inconsistency, unsafe promotion governance, non-reproducible deployment, and incomplete accounting/observability.

The hard-risk pause itself was mechanically correct according to the deployed policy, but the policy created the condition it later treated as a breach:

1. A nominal 15% symbol hard cap was replaced by a dynamic 25.5% cap.
2. The planner sized positions at that exact cap with no mark-movement headroom.
3. Ordinary price movement took exposure fractionally above 25.5%.
4. Three such observations in four hours triggered persistent hard pauses.
5. The pause callback stopped execution without flattening the open BTC shorts.

The active model was manually made promotion-eligible using an override that cited only 90 resolved decisions despite a configured minimum of 500. It was then manually promoted. Runtime evaluator control currently has `auto_promote=true`, overriding the container environment default of false.

The retraining pipeline fetches six months of history each week; it is not forward-only. However, it discards timestamp and symbol identity when combining the ten symbol frames. Its purported walk-forward validation therefore splits symbol blocks rather than chronological market history, recency weighting never activates, boundary labels can cross symbols, and the saved model is trained on only the first 80% block rather than being refit on the complete approved dataset. The reported 0.63-0.65 accuracy is not a valid temporal out-of-sample estimate.

Two immediate security exposures were also confirmed:

- Telegram request URLs containing the bot credential are continuously written to container logs.
- Redis is published on all IPv4/IPv6 interfaces, has protected mode disabled, requires no authentication, and accepts commands as the default user.

## Scope And Evidence

This audit covered:

- Ubuntu host and Docker runtime health.
- Running container configuration and source hashes.
- Repository, host working tree, and container provenance.
- Fresh SQLite backup from the running Telegram container.
- 14,019 execution route records.
- 29,616 model shadow decisions in the initial snapshot, plus current evaluator logs.
- Model registry records and promotion controls.
- Retraining, feature, inference, portfolio, execution, watchdog, and pause code.
- Historical logs retained from May and current Docker logs.
- Full local automated test suite and targeted production invariant probes.

Primary local evidence is retained under `ubuntu_audit_20260622/`. The fresh consistent database is `ubuntu_audit_20260622/state/quant_bot_audit_fresh.db`.

No production trading state, registry state, pause state, or model artifact was changed. A SQLite backup and temporary diagnostic files were created only for read-only audit extraction.

## Severity Scale

- **P0 - Critical:** unsafe to resume; can invalidate risk, model selection, credentials, or state.
- **P1 - High:** materially distorts performance or recovery and must be corrected before live capital.
- **P2 - Moderate:** reliability, operability, or maintainability weakness.
- **P3 - Low:** cleanup or technical debt.

## Findings Summary

| ID | Severity | Finding | Status |
|---|---:|---|---|
| F-01 | P0 | Multi-symbol retraining destroys temporal/symbol identity before validation | Confirmed |
| F-02 | P0 | Dynamic risk policy increases the nominal 15% symbol cap to 25.5% | Confirmed in production |
| F-03 | P0 | Positions are targeted exactly at the cap, so normal mark movement creates breaches | Confirmed from incident state |
| F-04 | P0 | Model promotion controls were manually bypassed and auto-promotion remains enabled | Confirmed |
| F-05 | P0 | Production is built from a dirty, non-reproducible working tree | Confirmed |
| F-06 | P0 | Telegram credential is exposed in ongoing logs | Confirmed; credential redacted here |
| F-07 | P0 | Redis is unauthenticated and exposed on all host interfaces | Confirmed |
| F-08 | P0 | Active sklearn artifacts are loaded under a different sklearn version | Confirmed warning at load |
| F-09 | P1 | Hard pause stops sessions while leaving open exposure unmanaged | Confirmed |
| F-10 | P1 | Paper execution and PnL omit fees, spread, slippage, funding, and impact | Confirmed |
| F-11 | P1 | Route audit records idempotent replays as additional accepted fills | Confirmed |
| F-12 | P1 | Position/accounting continuity cannot be fully reconstructed across adapter resets | Confirmed |
| F-13 | P1 | Persisted dashboard equity and lifetime PnL were stale at the pause | Confirmed |
| F-14 | P1 | Evaluator measures overlapping directional observations, not executable portfolio PnL | Confirmed |
| F-15 | P1 | Missing inference features are silently replaced with zero | Confirmed |
| F-16 | P1 | Historical partial-data retrain was promoted after six symbols failed | Confirmed in retained May logs |
| F-17 | P2 | Host capacity is healthy; no evidence supports compute saturation as root cause | Confirmed |
| F-18 | P2 | Local tests do not certify deployed source; production image contains no tests | Confirmed |

## Incident Reconstruction

### Timeline

All times below are UTC on 2026-06-18.

| Time | Event |
|---|---|
| 17:01 | First internally computed symbol exposure breach recorded; soft-breach mode activated. |
| 18:01 | Second breach in the four-hour window recorded. |
| 20:00:53 | User `6268794073` opened BTC short `-0.0400373698` at `62957.2`, approximately 25.5% of equity. |
| 20:01:00 | User `8392916807` opened BTC short `-0.0400781520` at `62957.2`, approximately 25.5% of equity. |
| 20:01 | Several BTC adjustments were blocked by ordinary deadband/cooldown rules. Other symbols were flattened. |
| 21:00:47 | Market short guard reported all ten symbols down over its lookback. |
| 21:00:48 | BTC model emitted BUY for user `8392916807`; third breach was confirmed before routing and persistent pause was written. |
| 21:00:54 | User `6268794073` was paused through the same path. |
| After pause | Signal and execution sessions stopped; BTC shorts remained in persisted paper state. |

### Breach Arithmetic

The persisted hard-risk payloads are internally consistent:

| User | Hard-pause equity | BTC notional | Exposure | Policy cap |
|---|---:|---:|---:|---:|
| `6268794073` | $9,887.16 | $2,524.88 | 25.53696% | 25.5% |
| `8392916807` | $9,897.24 | $2,527.45 | 25.53695% | 25.5% |

The breach exceeded the dynamic cap by roughly 0.037 percentage points. This was not a sudden gross leverage explosion. It was the predictable result of targeting the exact limit without headroom.

### Why The Cap Became 25.5%

The static `PortfolioRiskPolicy` default is 15% per symbol in `quant_v2/portfolio/risk_policy.py`. The dynamic policy in `quant_v2/execution/service.py` computes:

```text
gross_cap  = min(1.20 / sigma_60, 0.85)
symbol_cap = 0.30 * gross_cap
```

At the common 0.85 gross ceiling this produces `0.30 * 0.85 = 0.255`. The result is not clamped to the static 0.15 hard limit. The production invariant probe reproduced:

```text
static symbol cap  = 0.1500
dynamic symbol cap = 0.2550
soft symbol cap    = 0.2295
```

Dynamic policy should scale risk downward from immutable hard limits. It must never enlarge those limits.

### Pause Semantics

The hard-pause callback persists the pause and immediately stops both source and execution sessions. It does not guarantee one of the following terminal states:

- all positions are flat; or
- a separately supervised reduce-only liquidation workflow owns every remaining position.

The system therefore transitions from active management to no management while exposure remains open. A pause is not a safe terminal state by itself.

## Accounting And State Reconciliation

### Confirmed State

The final deduplicated route state agrees with persisted paper state for both users:

```text
6268794073: BTCUSDT = -0.04003736982229149
8392916807: BTCUSDT = -0.04007815198634266
```

### Route Volumes

| User | Accepted audit rows | Blocked/other rows | Entries | Flattens | Reductions | Rebalances |
|---|---:|---:|---:|---:|---:|---:|
| `6268794073` | 1,534 | 5,934 | 741 | 728 | 49 | 16 |
| `8392916807` | 1,291 | 5,260 | 611 | 623 | 37 | 20 |

The dominant blocked reasons were weight/notional deadband, cooldown, edge-improvement gate, and confidence-delta gate.

### Idempotent Audit Ambiguity

For user `8392916807`, 24 accepted audit rows were repeated returns of four existing idempotency keys. Each of the four keys appeared seven times. The paper adapter returned the original accepted `ExecutionResult`; accounting correctly skipped already-accounted keys, but route telemetry logged each replay as if another fill occurred.

Required correction: route events must distinguish `new_fill`, `idempotent_replay`, `rejected`, and `blocked_before_adapter`. Filled quantity must be the actual newly executed quantity, not the original result replayed by the adapter.

### State Discontinuities

After idempotency deduplication, the route stream still contains unexplained position discontinuities:

- 12 for user `6268794073`.
- 11 for user `8392916807`.
- Five visible paper adapter sequence resets per user.

At these points a prior nonzero replayed position is followed by an event whose `before_position` is zero without an intervening accepted flatten in the route ledger. These resets can reflect session recreation, incomplete historical persistence, or older restoration behavior, but the existing records are insufficient to prove which.

Consequently, exact lifetime realized PnL cannot be independently reproduced from the route table. A partial mark-price replay produced materially different values from the database lifetime fields, but those figures must not be treated as authoritative because entry continuity is missing.

### Stale Metrics

At the pause:

- User `6268794073` persisted current equity of $9,891.40, while the hard-risk snapshot was $9,887.16.
- User `8392916807` persisted current equity of $9,901.48, while the hard-risk snapshot was $9,897.24.

The lifetime PnL fields exclude the final equity delta because the pause stops the session before the normal persistence cycle advances its last-equity anchor. Position snapshot prices are likewise stale at the entry-cycle mark.

### Paper Execution Optimism

`InMemoryPaperAdapter` fills every accepted order immediately at the supplied mark. Paper accounting includes only mark-to-mark directional PnL. It does not deduct:

- maker/taker fees;
- spread crossing;
- slippage or adverse selection;
- partial-fill opportunity cost;
- funding payments;
- market impact; or
- latency between signal and fill.

Paper results therefore overstate deployable performance and cannot serve as the sole promotion measure.

## Model Training And Data Lineage

### The System Is Not Forward-Only

Production retraining currently runs weekly with `RETRAIN_TRAIN_MONTHS=6`. Each run requests six months of hourly data for ten symbols. It also uses `RETRAIN_REQUIRE_ALL_SYMBOLS=1` and `RETRAIN_REQUIRE_ALL_HORIZONS=1` today.

### Temporal Identity Is Lost

For each symbol, `fetch_symbol_dataset` returns a timestamp-indexed frame. `scheduled_retrain.py` then executes:

```python
featured = pd.concat(all_featured_frames, ignore_index=True)
featured = featured.sort_values("timestamp") if "timestamp" in featured.columns else featured
```

`ignore_index=True` discards the timestamp index, and no symbol column is added. The conditional sort does not run because timestamp is no longer a column. The combined frame is therefore a sequence of complete symbol blocks in fetch order.

Consequences:

1. `_build_labels(featured, horizon)` shifts across each block and contaminates the final horizon rows at symbol boundaries.
2. `TimeSeriesSplit` is applied to block order, not a synchronized chronological panel.
3. Training can include later calendar data for an earlier symbol while validating on earlier calendar data for a later symbol.
4. The final 80/20 split roughly trains on the first eight equal-sized symbol blocks and tests on the last two.
5. Recency weights are disabled because `timestamp` is absent.
6. The model saved after validation is the 80% model; it is not refit on all approved rows.
7. Registry metadata says all ten symbols were fetched but does not disclose which symbols were actually used by the saved fit.

The codebase already has a MultiIndex-aware data platform and purged group validation utilities. Scheduled retraining bypasses those safer abstractions.

### Validation Objective Is Misaligned

Retraining gates on mean classification accuracy at a 0.5 probability threshold. Production decisions use regime-dependent thresholds, confirmation logic, allocation, deadbands, and execution rules. A high classification accuracy does not establish positive executable expectancy.

The active model registry advertised approximately 0.629, 0.643, and 0.653 validation accuracy across horizons. Its initial forward shadow sample averaged approximately -31.6 bps gross and -39.6 bps after the evaluator's fixed 8 bps cost assumption. This divergence is expected when the validation design and production objective differ.

### Training/Inference Parity

Positive findings:

- Both paths use the same broad feature builder.
- Active artifacts expose the same 91 feature names across all three horizons.
- Historical fetching paginates and removes duplicate timestamps.

Material gaps:

- Inference silently creates every absent model feature with value zero.
- Aggregate row-count validation can pass despite per-symbol gaps.
- Funding and open-interest fetch failures are converted to missing/zero values without model-level drift gates.
- No immutable dataset snapshot or per-symbol continuity manifest is attached to each model.
- No feature-distribution parity report is required before registration.

Inference should fail closed or quarantine the decision when required features are missing beyond an explicitly approved fallback policy.

### Artifact Runtime Mismatch

The active artifacts were serialized with scikit-learn 1.8.0 and are loaded in production with scikit-learn 1.9.0. Production emits `InconsistentVersionWarning` and states that behavior may be invalid. Dependencies are broadly specified in source and the deployed lock now pins 1.9.0, but artifact metadata does not enforce the training runtime.

Each model version must record and validate Python, sklearn, LightGBM, joblib, feature schema, code commit, image digest, and dataset digest before loading.

## Model Evaluation And Promotion

### Manual Override

`models/production/registry/versions/model_20260602_082230.json` contains:

```text
paper_evaluation.promotion_eligible = true
override_reason = manual_promote ... 90_resolved_decisions
overridden_by = admin
```

The evaluator policy requires 500 candidate and 500 incumbent resolved decisions. The model was promoted on 2026-06-12 by Telegram admin command. Its top-level status is `active` while a nested tag remains `paper_quarantine`, another evidence-integrity inconsistency.

Local helper scripts also exist that directly clear hard-risk database flags and patch registry eligibility. They were not executed during this audit. Direct database and JSON mutation is not an acceptable control-plane mechanism.

### Auto-Promotion State

Container environment: `MODEL_EVAL_AUTO_PROMOTE=0`.
Persistent evaluator control file: `auto_promote=true`.

The control file takes precedence. Current candidates are blocked by the two hard-risk pauses and other criteria, but a future candidate could promote automatically when those blockers clear.

### Evaluator Samples Are Not Independent Trades

Shadow decisions are recorded once per symbol per evaluated bar. Examples:

- 930 resolved rows for the active candidate represent only 93 unique bar timestamps across ten symbols.
- 600 rows for the newest candidate represent only 60 unique timestamps.
- Eight-hour outcomes overlap heavily from one hourly decision to the next.

The evaluator sums fixed-cost directional returns as though each decision were an independent trade. It does not simulate target positions, allocation, repeated-signal suppression, turnover, concurrent exposure, funding, or execution controls. Its `net_return_bps` is therefore a classifier comparison metric, not portfolio PnL.

Promotion must be based on paired, executable portfolio replay plus independent statistical evidence, not raw overlapping decision count.

### Historical Partial-Data Promotion

Retained May logs show DNS failures for BTC, ETH, BNB, SOL, XRP, and ADA. Training continued on DOGE, LINK, AVAX, and LTC and promoted `model_20260524_145710`. Current source now defaults to all-symbol and all-horizon requirements, which is an improvement, but the prior promotion demonstrates why data-completeness rules must be immutable and covered by artifact evidence.

## Production Provenance And Operations

### Source Identity

| Surface | Identity |
|---|---|
| Local checkout | Git `507d6218` |
| Ubuntu Git HEAD | Git `6234aff5` |
| Running image | `sha256:e85886744eaf...` created 2026-06-12 |

Critical host files and container files have matching SHA-256 hashes, so the container was built from the Ubuntu working tree. However:

- Ubuntu reports 251 modified/untracked status entries.
- Ignoring end-of-line-only differences still leaves 53 changed files, with 2,473 insertions and 5,295 deletions.
- `quant_v2/research/model_evaluator.py` and its test are untracked relative to Git HEAD.
- Critical source differs from both Git HEAD and the local checkout.
- The production image contains no test suite.

Passing tests locally does not certify the running image.

### Security Patch Regression

Git HEAD is titled `fix: suppress token-bearing http client logs`, but the dirty production `quant/telebot/main.py` removes the two logger-level lines introduced by that commit. As a result, authenticated Telegram URLs continue to be logged. The bot credential must be rotated after logging is corrected and retained logs are access-controlled or destroyed according to policy.

### Redis Exposure

Confirmed runtime state:

```text
listen: 0.0.0.0:6379 and [::]:6379
protected-mode: no
authentication: none
ACL identity: default
host port publication: 6379:6379
```

Redis should not be published to the host for this architecture. Use the internal Compose network only, or bind to localhost with authenticated ACLs if host access is truly required.

### Host Capacity

At audit time:

- Load average approximately 1.2-1.3.
- CPU approximately 90% idle.
- I/O wait 0-1%.
- Telegram/model evaluator approximately 230 MiB RAM each.
- Retrainer approximately 356 MiB RAM.
- No OOM or container restart events.

The move from AWS may have changed network/DNS reliability, dependency versions, deployment discipline, or artifact/runtime combinations, but current evidence does not support CPU, memory, or disk saturation as the cause of trading underperformance.

## Automated Verification

The local audit environment was completed with `pytest`, `pytest-cov`, `pytest-asyncio`, `pyarrow`, scipy, scikit-learn, LightGBM, SQLAlchemy, Redis client, and supporting packages.

Result:

```text
635 passed, 29 warnings
```

This is useful regression evidence for local `507d6218`, but it does not invalidate the findings above:

- The production source is different.
- The production image has no tests.
- No test asserts dynamic caps are bounded by static hard caps.
- No test validates chronological identity through scheduled retraining.
- No test requires headroom below risk limits.
- No end-to-end test proves hard pause reaches flat or supervised reduce-only state.
- No test reconciles the route ledger, adapter state, persisted state, and PnL.
- No test fails on training/runtime library mismatch.
- The committed logging regression test does not test the dirty source actually used in production.

## Research Basis

The recommended redesign follows established concerns in quantitative model selection:

- Bailey et al., [The Probability of Backtest Overfitting](https://doi.org/10.21314/jcf.2016.322): repeated strategy selection raises the probability that the selected backtest is a false discovery.
- Bailey and Lopez de Prado, [The Deflated Sharpe Ratio](https://doi.org/10.2139/ssrn.2460551): Sharpe evidence should be adjusted for selection bias, non-normality, and multiple trials.
- Bergmeir et al., [Validity of Cross-Validation for Autoregressive Time Series Prediction](https://doi.org/10.1016/j.csda.2017.11.003): time-series validation assumptions must match temporal dependence.
- Cerqueira et al., [Evaluating Time Series Forecasting Models](https://doi.org/10.1007/s10994-020-05910-7): performance-estimation method materially affects conclusions under non-stationarity.
- Gama et al., [A Survey on Concept Drift Adaptation](https://doi.org/10.1145/2523813): adaptation requires explicit drift detection and evaluation, not merely frequent retraining.

The practical conclusion is not to train only on new data. Use a rolling or expanding historical window with explicit recency weights, chronological purging, symbol-aware grouping, and independent promotion evidence.

## Required Remediation

### Phase 0 - Immediate Containment

1. Keep both users hard-paused.
2. Reconcile and explicitly flatten or transfer ownership of the two BTC paper positions.
3. Set persistent evaluator control to `auto_promote=false`; verify via evaluator log.
4. Rotate the Telegram bot credential after suppressing and redacting HTTP client logs.
5. Remove Redis host publication; require authentication if any external access remains.
6. Preserve the current database, registry, image digest, and relevant logs as incident evidence.
7. Prohibit direct pause-clearing and registry-patching scripts from production workflows.

### Phase 1 - Risk And State Correctness

1. Make static limits immutable upper bounds for every dynamic policy.
2. Add sizing headroom, for example target no more than 80-90% of a hard cap.
3. Evaluate projected post-trade exposure against current marks before every order.
4. Guarantee that all exposure-reducing actions bypass alpha deadbands, cooldown, and order-count gates when outside target or safety limits.
5. Replace pause-and-stop with a state machine: `ACTIVE -> REDUCE_ONLY -> FLAT_CONFIRMED -> PAUSED`.
6. Run flatten under an independent supervisor with retries and reconciliation.
7. Persist each newly executed fill exactly once and record idempotent replay separately.
8. Introduce a double-entry cash/position ledger and reconcile it to adapter/exchange state each cycle.
9. Persist final marked equity and PnL before stopping a session.

### Phase 2 - Research Pipeline Rebuild

1. Preserve a canonical `(timestamp, symbol)` MultiIndex end to end.
2. Generate labels with `groupby(symbol).shift(-horizon)`.
3. Validate each symbol's continuity, freshness, duplicate rate, and supplementary coverage.
4. Use chronological folds with horizon-aware purge/embargo and symbol-cluster holdouts.
5. Keep a final untouched temporal holdout that is not used for threshold selection.
6. Refit the selected configuration on the complete approved training window.
7. Save immutable dataset, feature-schema, code, image, and dependency manifests.
8. Fail candidate registration when required feature parity or runtime compatibility is absent.
9. Use a rolling 6-12 month window with recency weighting; compare windows empirically rather than assuming more or less history is better.

### Phase 3 - Promotion Governance

1. Remove automatic promotion until a full paper burn-in succeeds.
2. Make evaluation records append-only and signed by source identity.
3. Require two-person approval for overrides, with expiry and documented reason.
4. Never allow a risk pause to be cleared merely to satisfy a model-evaluator blocker.
5. Evaluate executable portfolio replay after fees, spread, slippage, funding, and turnover.
6. Require positive absolute expectancy, not only improvement over a losing incumbent.
7. Compare every candidate against both incumbent and a simple transparent benchmark.
8. Report PBO/DSR or equivalent multiple-testing evidence across all attempted candidates.
9. Deploy through shadow, then a tightly capped canary, then staged scaling with automatic rollback.

### Phase 4 - Reproducible Operations

1. Build only from a clean reviewed commit.
2. Attach Git SHA, source manifest, lockfile digest, and image digest to every deployment.
3. Run tests against the exact image before deployment.
4. Pin model-serialization dependencies and reject incompatible artifacts.
5. Add DNS/data-provider health checks and per-symbol stale-data circuit breakers.
6. Add secret scanning and log-redaction integration tests.
7. Store incident and execution telemetry in an append-only destination outside the trading container.

## Restart Acceptance Gates

Execution should not resume until all of the following are demonstrated on the exact candidate image:

- [ ] Persistent evaluator control reports `auto_promote=false`.
- [ ] Open positions are flat or owned by a tested independent reduce-only supervisor.
- [ ] Dynamic policy never exceeds static symbol, gross, net, or bucket limits.
- [ ] Target sizing includes explicit headroom below every hard limit.
- [ ] A hard pause reaches `FLAT_CONFIRMED` or raises a continuously supervised critical incident.
- [ ] Risk-reducing orders cannot be blocked by alpha controls.
- [ ] Route ledger, adapter positions, persisted positions, and PnL reconcile exactly in replay tests.
- [ ] Paper PnL includes configurable realistic execution and funding costs.
- [ ] Scheduled retraining preserves symbol and timestamp identity.
- [ ] Labels, folds, and holdouts are proven chronological and group-safe.
- [ ] Saved candidate is refit on the declared approved training set.
- [ ] Model artifact runtime exactly matches its training manifest.
- [ ] Missing required inference features fail closed.
- [ ] Redis is not externally exposed without authenticated, least-privilege ACLs.
- [ ] No credential-bearing URL appears in logs under integration test or production smoke test.
- [ ] Deployment source is clean, immutable, reviewable, and identical to the tested image.
- [ ] Shadow/canary evidence shows positive absolute cost-adjusted expectancy and acceptable drawdown.

## Recommended System Direction

The recommended option is a controlled rebuild of the existing architecture, paired with a simple benchmark:

1. Repair risk, state, accounting, and deployment correctness first.
2. Rebuild the panel training/validation pipeline using the codebase's existing MultiIndex and purged-group utilities.
3. Keep ML models in shadow while a transparent low-turnover baseline provides a sanity check.
4. Promote only when executable, cost-adjusted, independent evidence is positive in absolute terms.

An online or continuously adaptive model should not be the next step. Adaptation would amplify the current validation and governance defects. Once the corrected pipeline is stable, drift-aware window selection can be evaluated as a challenger rather than trusted by default.

## Final Assessment

The system contains several thoughtful components: all-symbol retrain gating, hard-risk persistence, shadow evaluation, idempotency accounting, watchdog retry logic, and broad test coverage. The incident did not occur because there were no controls. It occurred because important controls were defined at different layers with incompatible semantics and could be bypassed operationally.

The architecture can be recovered, but current performance statistics, model promotion history, and paper PnL are not sufficiently trustworthy for capital allocation. Correctness and reproducibility must become promotion prerequisites, not parallel engineering goals.

## Live Recovery Update

- Live host `4arm-ubuntu` is reachable and the production stack now runs cleanly with `quant_redis` healthy and the app services up.
- The Telegram bot restores persisted sessions, shows `db_users_active=2`, and reports `hard_risk_paused=0` on startup.
- Raw credential-bearing Telegram API URLs are no longer appearing in the recent bot log stream.
- The retrain scheduler now starts successfully after trimming the eager `quant_v2` import cycle that was crashing its bootstrap.
- One remaining runtime warning persists: `InconsistentVersionWarning` from scikit-learn while reading older serialized model artifacts. That is a compatibility debt, not a live blocker, but it should be addressed before treating the model store as fully clean.
- Live resume is still not proven: the persisted account rows are in `live_mode=0` and the saved paper snapshots still carry open positions, so the production resume gate is not yet satisfied.
- 2026-06-24 deployment validation: the 4arm stack was rebuilt/recreated onto `quant_bot:latest` image `sha256:083c839c63b9bf79808610240793bfa985902f019fd9eb9df2944b871745ee03`; `quant_telegram`, `quant_retrain`, and `quant_model_eval` are running without restart loops and `quant_redis` is healthy.
- 2026-06-24 deployment validation found and corrected a live SQLite schema drift in `execution_route_events`: the startup migration now backfills all current nullable route-audit columns from the ORM model. The live DB reports no missing route risk columns and recent bot logs no longer show route-audit `OperationalError`, `ModuleNotFoundError`, traceback, or app-level error entries after the final restart.
- 2026-06-24 deployment validation caveat: the deployed bot restored two DEMO sessions and each restored paper state with eight positions, so the next paper-soak checkpoint should explicitly decide whether these are acceptable fresh demo positions or whether the books should be flattened again before measuring profitability.
- 2026-06-25 model-lineage reset validation: 4arm was intentionally restarted into no-active-model/no-trade mode. Strict model resolution refused the old invalid active pointer, Telegram reported `No production model found` plus `warnings=no_active_model_version`, and no auto-restore trading bridge was enabled.
- 2026-06-25 retrain validation found two deployment blockers and corrected them: sparse Binance historical open-interest coverage was collapsing six months of candles to roughly the last 30 days, and temporal validation was calling `predict_proba` on the `TrainedModel` wrapper instead of the runtime predictor helper. Local tests for feature coverage, artifact registration, and model-stack prediction passed before deployment.
- 2026-06-25 clean full-universe retrain completed without publishing a candidate. Feature coverage is now healthy (`42,600` featured rows across ten symbols), but all required horizons failed the configured development threshold (`2h=0.5229`, `4h=0.5199`, `8h=0.5408` versus `RETRAIN_MIN_ACCURACY=0.60`). The registry remains empty, no fresh production model directory was published, stale `.building` output was removed, and production trading must remain paused until a candidate clears independent paper/shadow evidence gates.
- 2026-06-26 recovery experiment runner update: real 4arm DNS/data connectivity was repaired and a fresh local core snapshot was built from 4arm data (`21,600` rows across `BNBUSDT`, `BTCUSDT`, `ETHUSDT`, `SOLUSDT`, `XRPUSDT`, covering `2025-12-28 15:00 UTC` through `2026-06-26 14:00 UTC`). The recovery runner now emits compact replay manifests, caps reported fill payloads, avoids repeated exposure slicing, and bounds portfolio replay audits to the latest `240` bars per symbol while retaining full holdout accuracy/expectancy evaluation. Validation passed (`24` targeted tests), and a real top-candidate run completed with recommendation `remain_no_trade`: candidate `h2_tw3m_hl30d_dz0p0010_fsfull` reached holdout accuracy `0.5422` versus the required `0.60`, so no model artifact was exported even though recent capped replay beat flat by `$31.88`. Production trading remains blocked.
- 2026-06-27 production resume gate recovery Phase 1 is now implemented and validated: `quant_v2.research.scheduled_retrain` re-exports `fetch_symbol_dataset`, the scheduled retrain candidate tests are now stable on both local Windows and 4arm host, the runtime reconciliation doc was added at `docs/runtime_reconciliation/source_runtime_reconciliation.md`, and the container compile check for the recovery modules passed. The Phase 1 reconciliation still shows the local worktree as dirty by design, the 4arm host as a dirty staging checkout, and the runtime container as compile-validated but lacking the CLI wrapper under `/app/scripts`.
- 2026-06-27 production resume gate recovery Phase 2 is now implemented and validated: the replay path now precomputes symbol/timestamp caches, uses a research-only no-op ledger in recovery experiments, and keeps the default full replay path intact for tests. Targeted replay/recovery tests still pass (`27` focused tests), 4arm compile validation passed, and the timed recovery measurements now meet the spec target with `top-1 elapsed=4:04.73` and `top-12 elapsed=30:01.62` on `datasets/v2/snapshots/model_recovery_real_1h_20260626_20260626_141210.parquet`. Both runs still recommended `remain_no_trade`, so no production resume gate opened and no model was promoted.
- 2026-06-27 production resume gate recovery Phase 3 broad sweep completed: the 24-candidate recovery run finished in `elapsed=1:01:06` on 4arm, evaluated `24` candidates, passed `0`, and still recommended `remain_no_trade`. The sweep summary artifacts are now present under `docs/model_quality/recovery_sweep_top24/`, including the spec-named alias `latest_recovery_sweep_summary.md`, so the repo has the ranked sweep evidence even though no candidate qualified for paper quarantine.
- 2026-06-27 production resume gate recovery Phase 4 repair suite is now implemented, locally validated, and validated again on 4arm against the real snapshot. The new bounded variant runner now emits research-input diagnostics, gracefully fails closed when temporal validation cannot be built, and the live host run completed with `recommendation=remain_no_trade`, `selected_candidate_id=null`, and `selected_variant_id=null` rather than aborting. No paper-quarantine candidate cleared the offline gate.
- 2026-06-27 follow-up host rerun after the dead-zone unit fix completed cleanly on 4arm at run `20260627T110518Z_b1593641`. The repaired sweep evaluated `4` candidates and `4` variants, but still reported `passed_candidates=0`, `passed_variants=0`, and `recommendation=remain_no_trade`, which confirms the gate is now executing correctly on the live snapshot but remains closed for production resume.
- 2026-06-27 comparative quant research audit completed in `COMPARATIVE_QUANT_RESEARCH_AUDIT.md`. The audit compares the repo against peer-reviewed and open-source quant ML practices and concludes that the central gap is not model family selection, but target/objective mismatch: the current pipeline primarily trains directional future-return classifiers while production PnL depends on cost-adjusted trade lifecycle outcomes. Recommended next implementation workstreams are trade-outcome labeling, economic threshold selection, stronger selection-risk reporting, expanded baselines, and AWS-to-4arm lineage forensics.
- 2026-06-27 model research gap closure implementation spec created in `MODEL_RESEARCH_GAP_CLOSURE_SPEC.md`. The spec decomposes the comparative audit into implementable phases for GPT-5.4 mini/subagents: trade-outcome labels, recovery label-mode integration, economic threshold selection, selection-risk reporting, expanded baselines, AWS-to-4arm lineage forensics, and an end-to-end no-production-registry recovery rerun.
- 2026-06-27 model research gap closure implementation completed and locally validated. The recovery runner now supports explicit trade-outcome labels, economic threshold selection, selection-risk reporting, expanded benchmark baselines, and phase-4 label-mode propagation. Validation passed on `tests/quant_v2/test_trade_outcome_labels.py`, `tests/quant_v2/test_economic_thresholds.py`, `tests/quant_v2/test_selection_risk.py`, `tests/quant_v2/test_model_recovery_experiments.py`, `tests/quant_v2/test_model_quality_recovery.py`, and `tests/quant_v2/test_scheduled_retrain_candidates.py`.
- 2026-06-27 lead audit correction accepted for model research gap closure. Audit found and corrected two implementation issues before acceptance: the economic threshold selector had drifted from the spec default/tie-break policy, and the trade-outcome recovery path was mixing OOF predictions with holdout forward returns. The corrected evaluator now records OOF-aligned forward returns, symbols, and fold IDs for utility thresholding. Local validation passed on the full spec suite, including `tests/quant_v2/test_portfolio_replay.py`.
- 2026-06-28 production-unblock follow-up specified in `TRADE_OUTCOME_SIDE_SEMANTICS_REPAIR_SPEC.md`. The 4arm trade-outcome recovery run `20260627T191451Z_b1593641` completed with `evaluated_candidates=12`, `passed_candidates=0`, and `recommendation=remain_no_trade`: the top candidate cleared accuracy and beat flat but failed because it did not beat the best non-flat benchmark and exceeded drawdown. Lead audit identified the next blocker as trade-outcome side semantics: take/skip labels must map to `BUY/HOLD` for long-side models and `SELL/HOLD` for short-side models, and skipped rows must not earn inverse-side returns.
- 2026-06-28 trade-outcome side-semantics repair implemented, lead-audited, corrected, and locally validated. `quant_v2/research/model_recovery_experiments.py` now carries explicit `trade_outcome_side` metadata through candidate identity, candidate labeling, replay signal resolution, and holdout economics; trade-outcome class `0` now maps to `HOLD/SKIP` rather than inverse-side trades; and the holdout/selection-risk reports expose take/skip counts plus one-sided-collapse diagnostics. Lead audit tightened candidate/variant side ordering so long/short configs are paired deterministically. Local validation passed on the required split suite: trade-outcome/economic-threshold/selection-risk tests (`14 passed`), model-quality/scheduled-retrain tests (`9 passed`), model-recovery tests (`29 passed`), and portfolio-replay tests (`3 passed`). Production trading remains blocked pending 4arm sync and host validation.
- 2026-06-28 4arm sync and host validation accepted for the trade-outcome side-semantics repair. Synced `quant_v2/research/model_recovery_experiments.py`, `tests/quant_v2/test_model_recovery_experiments.py`, `TRADE_OUTCOME_SIDE_SEMANTICS_REPAIR_SPEC.md`, and this audit ledger to `/home/admin-4arm/hypothesis-research-engine`. Host validation passed on the required split suite: trade-outcome/economic-threshold/selection-risk tests (`14 passed`), model-quality/scheduled-retrain tests (`9 passed`), model-recovery tests (`29 passed`), and portfolio-replay tests (`3 passed`). No recovery runner was active before sync. This does not reopen production trading; the next permitted step is the no-production-registry recovery rerun if the operator approves.
- 2026-06-28 no-production-registry recovery rerun completed on 4arm at run `20260628T051218Z_b1593641` using snapshot `datasets/v2/snapshots/model_recovery_real_1h_20260626_20260626_141210.parquet`, `--label-mode trade_outcome`, and `--max-candidates 12`. The run completed with `evaluated_candidates=12`, `passed_candidates=0`, `selected_candidate_id=null`, and `recommendation=remain_no_trade`; docs were refreshed under `docs/model_quality/`. Artifact audit confirmed both `_sidelong` and `_sideshort` candidates were evaluated, long-side replays emitted only `BUY` fills, short-side replays emitted only `SELL` fills, all holdout reports include `predicted_hold_count`, threshold selection used `economic_utility`, expanded benchmark actors were present, and the selection-risk report was present. No recovery runner remained active after completion. Production trading remains blocked.
- 2026-06-28 recovery rerun audit completed in `docs/model_quality/recovery_rerun_audit_20260628.md`. The audit rejects paper quarantine: all 12 candidates failed `candidate_did_not_beat_nonflat_benchmark`; long-side candidates had valid side semantics but lost money in replay or lacked actionable decisions; short-side candidates produced positive replay PnL but failed accuracy/expectancy gates and remained far below the transparent `short_only` benchmark. The next work should improve benchmark-relative/regime-conditional model quality rather than weakening gates.
- 2026-06-28 benchmark-relative candidate quality pass implemented and locally validated. `quant_v2/research/candidate_quality.py`, `quant_v2/research/model_recovery_experiments.py`, and `quant_v2/research/model_maintenance.py` now carry a benchmark-relative quality ledger, benchmark delta reports, symbol pruning before final replay, regime/symbol slices, replay-gap diagnostics, and maintenance decay checks. The recovery runner summary now records `candidate_quality_summary`, candidate rows include the new quality evidence, and the maintenance helper now considers same-side benchmark decay plus elevated-risk sample scarcity. Validation passed on `tests/quant_v2/test_candidate_quality.py`, `tests/quant_v2/test_model_maintenance.py`, and `tests/quant_v2/test_model_recovery_experiments.py`. Production trading remains blocked until the next host validation and paper-soak decision.
- 2026-06-28 4arm benchmark-relative validation accepted. The accepted file set was synced to `/home/admin-4arm/hypothesis-research-engine`, and host validation passed under `.venv`: `tests/quant_v2/test_candidate_quality.py` plus `tests/quant_v2/test_model_maintenance.py` (`12 passed`), `tests/quant_v2/test_model_recovery_experiments.py` (`29 passed`), and `tests/quant_v2/test_model_evaluator.py` plus `tests/quant_v2/test_model_registry.py` (`25 passed`). A bounded no-production-registry recovery run completed on the real snapshot `datasets/v2/snapshots/model_recovery_real_1h_20260626_20260626_141210.parquet` with run id `20260628T113942Z_b1593641`, `--label-mode trade_outcome`, and `--max-candidates 4`; it evaluated `4` candidates, passed `0`, recorded `candidate_quality_summary.failed_quality=4`, selected no candidate, and recommended `remain_no_trade`. Validation artifacts were copied back under `docs/model_quality/benchmark_relative_4arm/20260628T113942Z_b1593641/`. Production trading remains blocked.
