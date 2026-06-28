# Hypothesis Research Engine - Implementation And Build Plan

**Plan date:** 2026-06-22
**Source audit:** `audit_report.md`
**Current production disposition:** **DO NOT RESUME EXECUTION**
**Plan status:** Proposed for implementation approval

## 1. Purpose

This document converts every confirmed finding in the June 22 forensic audit into an ordered implementation program. It is the build authority for restoring the existing system; it is not permission to clear a hard-risk pause, activate a model, trade live capital, or mutate retained incident evidence.

The recommended direction is a controlled rebuild of the existing architecture. The system already has useful risk, execution, research, registry, and test components. The work should preserve those components where their contracts are sound, while replacing the paths that invalidate state, temporal validation, promotion evidence, security, or deployment provenance.

The governing order is:

1. Establish a canonical source and secure the runtime.
2. Make risk reduction, state transitions, and accounting mechanically correct.
3. Rebuild the research dataset and validation contract.
4. Evaluate candidates as executable portfolios rather than independent classifications.
5. Restore promotion only through immutable evidence and staged deployment.

No alpha or model-complexity work should bypass this order.

## 2. Outcomes And Non-Goals

### Required outcomes

- One clean, reviewable Git commit must correspond exactly to every image and model artifact.
- Dynamic risk may reduce static limits but may never increase them.
- A hard-risk event must end in either `FLAT_CONFIRMED` or continuously supervised `REDUCE_ONLY`; it must never silently abandon exposure.
- Every newly executed fill must be recorded once and be sufficient to reconstruct cash, positions, realized PnL, unrealized PnL, fees, and funding.
- Training data must retain `(timestamp, symbol)` identity through labeling, validation, fitting, artifact creation, and inference.
- Validation and promotion must use cost-adjusted executable portfolio outcomes on genuinely out-of-sample time.
- Promotion must require positive absolute evidence, not merely relative improvement over a losing incumbent.
- Secrets, runtime dependencies, and infrastructure must pass automated pre-deployment gates.
- Restart must be governed by evidence produced from the exact image proposed for deployment.

### Non-goals for the recovery release

- Online learning, continuous parameter adaptation, or reinforcement learning.
- New model families added solely to improve headline backtest metrics.
- Automatic model promotion.
- Direct Telegram, JSON, or database edits that bypass the control plane.
- Reconstructing exact historical lifetime PnL where the audited ledger lacks enough evidence. That period will be explicitly marked non-reconcilable.
- Live-capital deployment before a completed shadow, paper, and canary sequence.

## 3. Architectural Decisions

### AD-01: Canonical source before feature work

The Ubuntu host, local checkout, Git HEAD, and running image currently disagree. The first deliverable is a clean recovery branch assembled from reviewed source, not a wholesale copy of either dirty tree. Each production-only change must be classified as keep, replace, or discard, with a reviewer and test reference.

### AD-02: Immutable hard limits with operational headroom

Static symbol, gross, net, bucket, and drawdown limits are immutable ceilings. Dynamic volatility or regime policy can only multiply them downward. Normal target sizing will use a lower operating limit so mark movement does not immediately create a breach.

Initial policy shape:

```text
hard_limit      = configured immutable ceiling
dynamic_limit   = min(hard_limit, volatility_limit, regime_limit)
operating_limit = target_headroom_ratio * dynamic_limit
```

`target_headroom_ratio` should begin conservatively at 0.85 and be configurable only within an approved range. The exact value must be stress-tested; it is not an alpha tuning parameter.

### AD-03: Risk lifecycle is a state machine

Replace pause-as-a-boolean with explicit states:

```text
ACTIVE
  -> SOFT_BREACH
  -> REDUCE_ONLY
  -> FLATTENING
  -> FLAT_CONFIRMED
  -> PAUSED

Any state -> INCIDENT when ownership, prices, venue, or reconciliation is unavailable.
```

`INCIDENT` must retain an independent supervisor, retries, alerts, and operator-visible ownership. Session shutdown is allowed only after final accounting persistence and flat confirmation, or after a separate supervisor has durably accepted ownership of every open position.

### AD-04: Append-only fills and derived state

Accepted fills, fees, funding, deposits/withdrawals, corrections, and mark snapshots are immutable events. Position and PnL tables are projections that can be rebuilt from those events. Idempotent replay is a transport outcome, not another fill.

### AD-05: Canonical panel data contract

All multi-symbol research data uses a sorted, unique `pandas.MultiIndex` named `['timestamp', 'symbol']`, with UTC timestamps. Labels are produced within symbol. A dataset cannot be registered unless every symbol passes continuity, freshness, duplicate, schema, and supplementary-data checks.

### AD-06: Two distinct validation questions

The system needs both of these and must not conflate them:

- **Temporal generalization:** expanding or rolling forward folds where all training timestamps precede validation timestamps, with horizon-aware purging.
- **Cross-symbol robustness:** secondary symbol-cluster holdouts that test concentration and transfer. These are stress tests, not the primary production estimate.

Threshold selection, calibration, and model selection occur inside development folds. A final temporal holdout remains untouched until one configuration is frozen.

### AD-07: Promotion evidence comes from portfolio replay

Candidate and incumbent signals must pass through the same allocation, risk, planner, order, and accounting contracts used in production. Evaluation uses paired calendar blocks and realized portfolio state, not sums of overlapping symbol/horizon directional returns.

### AD-08: Models are immutable deployment units

A model version includes artifacts plus a signed manifest containing dataset digest, feature schema, code SHA, image digest, Python and dependency versions, training parameters, validation policy version, cost policy version, and checksums. Runtime rejects incompatible artifacts before loading.

### AD-09: Manual, evidence-bound promotion

Automatic promotion remains disabled. A promotion is an append-only approval event requiring two distinct authorized identities, an unexpired evidence bundle, and passing runtime blockers. An override may tighten or defer a decision; it may not manufacture missing quantitative evidence.

## 4. Audit Finding Traceability

| Finding | Remediation work packages | Closure evidence |
|---|---|---|
| F-01 temporal/symbol identity loss | WP-08, WP-09, WP-10 | Panel invariants, boundary-label tests, chronological fold proof, untouched holdout report |
| F-02 dynamic cap exceeds static cap | WP-03 | Property tests over volatility/regime space and runtime policy telemetry |
| F-03 sizing at hard cap | WP-03 | Headroom and projected-exposure tests; stress replay cannot breach from one normal mark move |
| F-04 promotion bypass/auto-promote | WP-00, WP-13 | Auto-promote absent/false, append-only approvals, two-person gate tests |
| F-05 dirty non-reproducible production | WP-01, WP-02, WP-15 | Clean SHA, source manifest, immutable image digest, deployment attestation |
| F-06 Telegram credential in logs | WP-00, WP-02 | Rotated token and integration test proving no credential-bearing URLs |
| F-07 unauthenticated exposed Redis | WP-00, WP-02 | No public bind; authenticated least-privilege health probe |
| F-08 sklearn artifact mismatch | WP-02, WP-10 | Locked dependencies and load-time manifest rejection test |
| F-09 pause strands exposure | WP-04 | End-to-end lifecycle reaches flat or supervised incident under injected failures |
| F-10 optimistic paper execution | WP-07, WP-11 | Cost attribution and conservative replay calibration report |
| F-11 replay logged as accepted fill | WP-05 | Outcome taxonomy and exactly-once fill integration tests |
| F-12 accounting discontinuities | WP-05, WP-06 | Rebuildable ledger and cycle reconciliation proof |
| F-13 stale equity/PnL on pause | WP-04, WP-06 | Final mark/checkpoint committed before session stop |
| F-14 evaluator is not portfolio PnL | WP-11, WP-12 | Paired executable replay and block-aware statistical report |
| F-15 missing features silently zeroed | WP-08, WP-10 | Feature contract and fail-closed inference tests |
| F-16 partial-data promotion | WP-08, WP-10, WP-13 | Immutable completeness gate and intentionally failed partial-data build |
| F-17 host capacity not root cause | WP-14 | Stage-level latency/resource telemetry; no speculative infrastructure migration |
| F-18 local tests do not certify image | WP-02, WP-15 | Tests executed against exact image and recorded in attestation |

## 5. Delivery Work Packages

## WP-00 - Immediate Containment And Evidence Preservation

**Priority:** P0
**Dependencies:** None
**Production mutation:** Requires a separately approved maintenance window

### Build tasks

1. Keep all affected users hard-paused and disable all model activation paths.
2. Set both environment and persistent evaluator control to `auto_promote=false`; change control-file semantics so a file cannot enable a feature disabled by environment policy.
3. Reconcile the two retained BTC paper positions and explicitly choose one recorded terminal action: flatten, archive as incident state, or transfer to the future supervisor. Do not silently clear them.
4. Preserve read-only copies and hashes of the database, registry, active artifacts, Compose files, image digest, and relevant logs.
5. Suppress `httpx` and `httpcore` authenticated request logging before token rotation. Add a structural redaction filter as defense in depth.
6. Rotate the Telegram bot token, restart only the bot service, verify the old token fails, and apply retention policy to exposed logs.
7. Remove Redis host port publication. If host access is operationally required, bind only to localhost and require ACL authentication; containers receive separate least-privilege credentials through secrets.
8. Move pause-clearing and registry-patching helper scripts out of operational paths. Preserve forensic copies, but prevent packaging and production execution.

### Primary files

- `docker-compose.yml`
- `docker-compose.override.yml`
- `quant/telebot/main.py`
- `quant_v2/research/model_evaluator.py`
- `.dockerignore`
- deployment and incident runbooks

### Required tests and evidence

- Log-capture integration test sends Telegram requests containing a canary token and asserts the raw and URL-encoded token never appear.
- Compose policy test asserts Redis has no public `ports` entry and has authenticated health checks where authentication is enabled.
- Evaluator startup test proves persistent state cannot turn auto-promotion on when the deployment policy disables it.
- Secret scan of tracked files, image layers, Compose output, and retained non-incident logs.
- Signed containment checklist recording token rotation time, Redis reachability, and evidence hashes.

### Exit gate

No credential appears in new logs, Redis is unreachable from non-approved host interfaces, model activation is impossible, and incident evidence is immutable.

## WP-01 - Canonical Recovery Baseline

**Priority:** P0
**Dependencies:** WP-00 evidence capture

### Build tasks

1. Create a recovery branch from an agreed reviewed Git commit.
2. Generate a three-way inventory: local checkout, Ubuntu working tree, and running-container source.
3. For each divergent production file, record:
   - source hash;
   - purpose of the change;
   - audit finding affected;
   - keep/replace/discard decision;
   - tests that establish intended behavior.
4. Port approved behavior as small reviewed commits. Do not import transient state, generated files, credentials, helper scripts, databases, or model artifacts into Git.
5. Remove line-ending-only noise and enforce repository attributes so future provenance comparisons are content-stable.
6. Tag the retained incident image and artifacts as quarantined and non-promotable.
7. Produce a baseline manifest of every application file and its SHA-256 digest.

### Exit gate

The recovery branch is clean, full tests pass, all production-only behavior has an explicit disposition, and two reviewers can reproduce the source manifest.

## WP-02 - Reproducible And Secure Build

**Priority:** P0
**Dependencies:** WP-01

### Build tasks

1. Replace broad runtime dependency ranges with a generated, hashed lock for Python 3.11 and the target architecture. Training and inference must share the exact sklearn, LightGBM, joblib, numpy, pandas, and Python versions.
2. Make the Docker build install from the lock rather than an independent package list in `Dockerfile`.
3. Add a test stage that includes the repository tests and executes them inside the exact dependency environment used to construct the runtime image.
4. Build once and promote by immutable digest. Do not rebuild separately per environment and do not deploy `:latest` as the authority.
5. Attach OCI labels for Git SHA, source-manifest digest, lock digest, build time, and CI run identity.
6. Generate an SBOM and vulnerability/secret scan result. Define severity exceptions as expiring reviewed records.
7. Run the container as non-root with read-only root filesystem where practical, dropped capabilities, `no-new-privileges`, bounded memory/CPU, and explicit writable volumes.
8. Introduce startup compatibility checks for every active model manifest. Fail readiness rather than merely warning on a version mismatch.
9. Add image-level smoke tests for imports, model load, database migration status, Redis ACL, signal dry-run, and health endpoints.

### Primary files

- `pyproject.toml`
- new dependency lock files
- `Dockerfile`
- `docker-compose.yml`
- `.dockerignore`
- CI workflow and deployment scripts

### Exit gate

Given a clean SHA, CI produces one digest, executes the complete suite against that image, and emits an attestation tying source, dependencies, tests, and image together.

## WP-03 - Risk Policy Invariants And Sizing Headroom

**Priority:** P0
**Dependencies:** WP-01

### Build tasks

1. Separate immutable `HardRiskLimits` from dynamic `OperatingRiskLimits`. Dynamic policy constructors must require the hard-limit object and clamp every output.
2. Eliminate duplicated cap formulas in `quant_v2/execution/service.py`; centralize policy calculation in `quant_v2/portfolio/risk_policy.py` or a dedicated policy module.
3. Apply headroom to target sizing, not to breach detection. Breach detection always uses the true hard limit.
4. Evaluate projected post-trade symbol, gross, net, and bucket exposures at current conservative marks before routing every order.
5. Reserve capacity for fees, slippage, rounding, minimum quantity, and adverse mark movement.
6. Classify every intent as `INCREASE`, `REDUCE`, `FLATTEN`, or `FLIP` based on projected absolute exposure.
7. Ensure `REDUCE` and `FLATTEN` bypass alpha deadbands, cooldowns, confidence deltas, normal order-count gates, and minimum-allocation filters. Exchange precision/min-notional constraints still apply and must trigger a supervised residual-position path.
8. Emit hard limit, dynamic limit, operating limit, current exposure, projected exposure, and headroom in route telemetry.
9. Version the risk policy and persist the version on every decision and fill.

### Primary files

- `quant_v2/portfolio/risk_policy.py`
- `quant_v2/portfolio/allocation.py`
- `quant_v2/execution/planner.py`
- `quant_v2/execution/reconciler.py`
- `quant_v2/execution/service.py`
- corresponding execution and portfolio tests

### Required tests

- Property test: for broad volatility/regime inputs, every dynamic cap is less than or equal to its hard cap.
- Property test: a risk-reducing order never increases absolute symbol or aggregate exposure.
- Boundary tests around fees, rounding, minimum notional, zero/negative equity, stale prices, and mark jumps.
- Regression replay of the June 18 state: target remains below operating cap and cannot create the audited 25.5369% breach through ordinary sizing.
- Metamorphic test: worsening volatility cannot enlarge any limit.

### Exit gate

All cap invariants hold in unit, property, and incident-replay tests, and no ordinary alpha gate can block a valid reduction.

## WP-04 - Supervised Risk Lifecycle And Flattening

**Priority:** P0
**Dependencies:** WP-03, Redis/security baseline from WP-00

### Build tasks

1. Implement the lifecycle states in a durable state table with monotonic transition rules, timestamps, reason, owner, retry count, and policy version.
2. Split alpha-session ownership from liquidation-supervisor ownership. Stopping alpha generation must not stop risk supervision.
3. On a confirmed hard breach:
   - persist the breach and final marked account snapshot;
   - reject all exposure-increasing intents;
   - transfer open positions to the supervisor;
   - generate reduce-only plans;
   - retry with bounded backoff and refreshed marks;
   - reconcile after each outcome;
   - stop only after `FLAT_CONFIRMED`.
4. When prices, venue, credentials, Redis, or database are unavailable, enter `INCIDENT`, retain the retry loop, and continuously alert with deduplicated escalation.
5. Add an operator command that requests a reviewed transition; it must not directly update pause fields. Clearing requires flat confirmation, reconciliation, evidence reference, and two authorized approvals for live mode.
6. Persist marked equity, realized/unrealized PnL, positions, outstanding orders, and reconciliation status transactionally before session termination.
7. Make watchdog health independent of the Telegram process and expose heartbeat/ownership telemetry.

### Primary files

- `quant_v2/execution/service.py`
- `quant_v2/execution/watchdog.py`
- `quant_v2/execution/state_wal.py`
- `quant_v2/telebot/signal_manager.py`
- `quant/telebot/main.py`
- database models and migrations

### Required tests

- End-to-end hard breach with open long, open short, multiple symbols, partial fills, stale marks, rejected orders, Redis restart, process restart, and exchange outage.
- Crash tests at every state transition prove restart resumes from durable ownership.
- Test that `PAUSED` cannot coexist with unmanaged nonzero exposure.
- Test that final equity and PnL are committed before the source/execution session stops.
- Alert tests prove a continuing `INCIDENT` remains visible without producing an unbounded notification storm.

### Exit gate

Chaos tests always end in flat confirmation or a live supervised incident with a named owner and retry heartbeat.

## WP-05 - Execution Outcome Taxonomy And Exactly-Once Fills

**Priority:** P1
**Dependencies:** WP-01; coordinate with WP-04

### Build tasks

1. Replace overloaded accepted/status fields with a typed outcome:

```text
NEW_FILL
PARTIAL_FILL
IDEMPOTENT_REPLAY
ADAPTER_REJECTED
BLOCKED_PRE_ROUTE
CANCELLED
UNKNOWN_REQUIRES_RECONCILIATION
```

2. Give every order request, venue order, fill, and accounting transaction separate immutable identifiers.
3. For an idempotent replay, return the original result for transport semantics but set `newly_filled_qty=0` and record a replay event linked to the original fill.
4. Make fill insertion unique on venue/account/fill identity and idempotency key plus fill sequence where a venue lacks a fill ID.
5. Record requested quantity, newly filled quantity, cumulative quantity, average price, fees, timestamps, source event, policy version, and model version.
6. Stop using route audit rows as the authoritative accounting ledger. They remain operational telemetry linked to immutable fill records.
7. Add schema versioning and backward-compatible readers for retained audit records.

### Primary files

- `quant_v2/execution/adapters.py`
- `quant_v2/execution/idempotency.py`
- `quant_v2/execution/service.py`
- `quant_v2/execution/binance_adapter.py`
- database models and migrations

### Exit gate

Replaying any request any number of times creates exactly one economic fill and an explicit replay trail.

## WP-06 - Accounting Ledger And Reconciliation

**Priority:** P1
**Dependencies:** WP-05

### Build tasks

1. Add append-only tables for orders, fills, cash movements, fees, funding, corrections, marks, and lifecycle events.
2. Add balanced journal entries in the reporting currency for each economic event, while preserving native asset quantities and prices. Corrections reverse prior entries; they never edit them.
3. Derive materialized positions, average cost, realized PnL, unrealized PnL, cash, equity, and lifetime metrics from the ledger.
4. Implement deterministic replay from an empty projection to any event sequence number.
5. Reconcile each cycle among:
   - ledger-derived positions;
   - adapter/exchange positions and open orders;
   - persisted projections;
   - marked equity and cash.
6. Define tolerances by symbol precision and reporting-currency rounding. Any unexplained difference blocks new exposure and enters supervised reconciliation.
7. Commit ledger events and projection checkpoints transactionally. Redis WAL is transport/recovery support, not the sole accounting authority.
8. Import auditable historical events. Mark the pre-cutover period `LEGACY_UNVERIFIABLE` where the June audit proved discontinuities; do not synthesize false precision.
9. Provide an operator reconciliation report and machine-readable health state.

### Primary files

- new `quant_v2/accounting/` package
- `quant_v2/execution/reconciler.py`
- `quant_v2/execution/state_wal.py`
- `quant_v2/execution/service.py`
- `quant/telebot/models.py`
- Alembic or an equivalent explicit migration framework

### Required tests

- Golden ledgers for long/short open, add, partial close, flip, fees, funding, correction, liquidation, and deposit/withdrawal.
- Randomized event replay yields the same projections after restart.
- Duplicate and out-of-order delivery does not alter economic state.
- June 18 final deduplicated positions reproduce exactly, while unsupported lifetime history is marked unverifiable.
- Database failure between fill receipt and projection update is recovered without loss or duplication.

### Exit gate

Ledger replay, persisted projections, and adapter state reconcile to defined tolerances across normal, duplicate, restart, and failure scenarios.

## WP-07 - Realistic Paper Execution And Cost Policy

**Priority:** P1
**Dependencies:** WP-05, WP-06

### Build tasks

1. Define a versioned `ExecutionCostPolicy` shared by research replay, shadow evaluation, and paper execution.
2. Model, at minimum:
   - maker/taker fees by venue tier;
   - bid/ask spread crossing;
   - slippage as a function of volatility, notional, and available liquidity proxy;
   - funding by symbol and timestamp;
   - signal-to-order latency;
   - partial fills and unfilled opportunity cost;
   - conservative impact tiers where order-book evidence is unavailable.
3. Use as-of information only. No future spread, funding, or book data may enter an order decision.
4. Calibrate cost assumptions against retained live/paper venue observations and retain conservative fallback values when coverage is missing.
5. Attribute gross return, each cost component, and net return per fill, position, symbol, model, and period.
6. Add sensitivity reports at base, adverse, and severe cost scenarios. Promotion must pass the approved adverse scenario.

### Primary files

- `quant_v2/portfolio/cost_model.py`
- `quant/risk/cost_model.py`
- `quant_v2/execution/adapters.py`
- new shared execution simulation module

### Exit gate

Paper and replay PnL expose complete cost attribution, and the same event sequence produces the same ledger result under a named cost-policy version.

## WP-08 - Canonical Data And Feature Contracts

**Priority:** P0 for research resumption
**Dependencies:** WP-01

### Build tasks

1. Make `fetch_universe_dataset` or a successor the only scheduled-retrain ingress. Remove per-symbol concatenation with `ignore_index=True`.
2. Enforce sorted unique UTC MultiIndex `['timestamp', 'symbol']` at every stage.
3. Build labels inside each symbol using grouped forward shifts. Explicitly remove the final horizon rows per symbol.
4. Create a `DatasetManifest` containing:
   - requested/fetched/failed symbols;
   - requested and actual time ranges per symbol;
   - row count and unique timestamps per symbol;
   - missing, duplicate, stale, and gap statistics;
   - funding/open-interest/order-book coverage;
   - source and retrieval timestamps;
   - raw and transformed schema versions;
   - content digests.
5. Fail candidate construction when required symbols, horizons, or feature-source coverage fall below approved per-symbol thresholds. Aggregate row count is not sufficient.
6. Define a versioned feature schema: name, dtype, units, lookback, source, valid range, missing policy, and training/inference implementation identity.
7. Eliminate blanket `fillna(0.0)`. Each feature must use one approved behavior: naturally valid zero, bounded imputation with a missingness indicator, symbol quarantine, or full candidate failure.
8. Add point-in-time checks for cross-pair, regime, funding, and open-interest features.
9. Persist immutable Parquet snapshots or content-addressed references sufficient to reproduce each training run.

### Primary files

- `quant_v2/data/multi_symbol_dataset.py`
- `quant_v2/research/scheduled_retrain.py`
- `quant_v2/research/group_validation.py`
- `quant/features/pipeline.py`
- feature modules and data storage

### Required tests

- Shuffled symbol fetch order produces identical canonical data and labels.
- No label references another symbol or a non-future timestamp.
- A one-symbol gap cannot be hidden by aggregate rows from other symbols.
- Missing supplementary data triggers the declared feature policy.
- Feature computation is point-in-time under a future-data perturbation test.

### Exit gate

Every candidate has a reproducible dataset manifest, and all panel, label, completeness, and point-in-time invariants pass.

## WP-09 - Defensible Validation And Model Selection

**Priority:** P0 for research resumption
**Dependencies:** WP-08, WP-07 for cost assumptions

### Build tasks

1. Replace scheduled `TimeSeriesSplit` over flattened rows with explicit timestamp folds.
2. Primary temporal folds must satisfy `max(train_timestamp) < min(test_timestamp)` and purge at least the maximum label horizon plus any feature/lookahead overlap.
3. Use rolling and expanding variants as declared experiments. Compare 3, 6, 9, and 12-month windows plus an expanding baseline; do not assume shorter or longer is superior.
4. Apply recency weighting from preserved timestamps and record effective sample size. Compare weighting half-lives only inside nested development folds.
5. Keep symbol-cluster holdout as a separate robustness report. Do not substitute unseen-symbol validation for forward temporal validation on the traded universe.
6. Reserve a final contiguous temporal holdout before configuration search. No threshold, feature, hyperparameter, window, or cost choice may use it.
7. Track every attempted configuration in an experiment ledger. Multiple-testing evidence must use the actual trial count, not a rough fold-count proxy.
8. Report at minimum:
   - net expectancy and total return after costs;
   - drawdown and time under water;
   - turnover, exposure, concentration, and capacity proxy;
   - hit rate and calibration by horizon/regime/symbol;
   - probabilistic and deflated Sharpe evidence;
   - PBO or equivalent selection-risk diagnostics;
   - dispersion across temporal folds and symbols;
   - adverse-cost sensitivity.
9. Accuracy remains a diagnostic, never the promotion objective.
10. Define failure rules for empty folds, class collapse, data drift, singular metrics, or concentration-driven results.

### Research rationale

- Bailey et al. show that repeated strategy selection increases the probability of a backtest false discovery.
- The Deflated Sharpe Ratio adjusts evidence for multiple trials and non-normal returns.
- Time-series performance estimation must preserve temporal dependence and reflect non-stationarity.
- Overlapping horizons require block-aware inference; raw row count is not effective sample size.

Primary references are retained in `audit_report.md`.

### Primary files

- `quant_v2/validation/purged_group_cpcv.py`
- new temporal splitter under `quant_v2/validation/`
- `quant_v2/research/group_validation.py`
- `quant_v2/research/experiment_score.py`
- `quant/validation/metrics.py`
- `quant_v2/research/scheduled_retrain.py`

### Exit gate

A synthetic leakage test fails the old splitter and passes the new one; the selected candidate has a frozen, untouched holdout result and trial-aware robustness report.

## WP-10 - Training, Refit, Artifact, And Inference Parity

**Priority:** P0 for research resumption
**Dependencies:** WP-08, WP-09, WP-02

### Build tasks

1. Separate configuration selection from final fitting.
2. Generate out-of-fold development predictions for calibration and threshold selection.
3. After configuration freeze, refit the estimator on the complete approved development window. Fit calibration without contaminating the final holdout, using out-of-fold predictions or a declared calibration slice.
4. Evaluate once on the untouched holdout; do not iterate after viewing it without creating a new experiment family and holdout.
5. Save an artifact manifest with:
   - exact rows/time range and dataset digest;
   - symbols and horizons actually used;
   - feature schema digest and ordered names;
   - estimator, calibration, threshold, and sample-weight configuration;
   - Python and dependency versions;
   - Git SHA and image digest;
   - validation/cost policy versions;
   - all artifact checksums.
6. Treat feature order, dtype, missingness, and freshness as a runtime contract. The existing strict predictor behavior should be retained, and upstream zero-insertion paths removed.
7. Quarantine an individual symbol on missing/stale inputs where the policy permits; otherwise fail the whole cycle closed. Never silently fabricate model inputs.
8. Run a golden-row parity test: the training image and inference image must produce identical transformed features and probabilities within declared numeric tolerance.
9. Reject model load on dependency, schema, code, checksum, or manifest mismatch.

### Primary files

- `quant_v2/models/trainer.py`
- `quant_v2/models/predictor.py`
- `quant_v2/research/scheduled_retrain.py`
- `quant_v2/model_registry.py`
- signal-generation feature assembly

### Exit gate

The registered artifact is the declared full-development fit, loads only in a compatible runtime, and passes golden feature/prediction parity.

## WP-11 - Executable Portfolio Replay

**Priority:** P1
**Dependencies:** WP-03, WP-05, WP-06, WP-07, WP-08

### Build tasks

1. Build a deterministic event-driven replay that invokes the production contracts for signal thresholds, confirmation, allocation, risk policy, planner, order simulation, ledger, and reconciliation.
2. Use synchronized as-of market events across symbols. Define deterministic ordering for bars sharing a timestamp.
3. Carry positions, cash, outstanding orders, cooldowns, deadbands, funding, and lifecycle state across timestamps.
4. Prevent repeated hourly predictions for an 8-hour horizon from being counted as independent round trips. They may alter a position only through production rebalance rules.
5. Support candidate, incumbent, and transparent benchmark on the same event stream with isolated state.
6. Produce equity curve, fills, costs, turnover, exposures, drawdown, concentration, blocked intents, and risk-state transitions.
7. Add scenario injection for spread widening, latency, partial fills, data gaps, venue rejection, mark jumps, and Redis/process restart.
8. Make replay output content-addressed and linked to model, dataset, code, and policy manifests.

### Benchmark set

- cash/no-trade baseline;
- current incumbent;
- simple transparent low-turnover directional baseline using fixed documented rules;
- candidate.

The candidate must be positive after costs in absolute terms and must improve on relevant deployable alternatives. Beating a negative incumbent is insufficient.

### Exit gate

The same event bundle and manifests reproduce byte-stable fills and numerically stable portfolio metrics, and replay exercises the same policy contracts as production.

## WP-12 - Shadow Evaluation And Statistical Decision Policy

**Priority:** P1
**Dependencies:** WP-09, WP-10, WP-11

### Build tasks

1. Replace directional-row aggregation in `model_evaluator.py` with the executable replay/ledger engine.
2. Pair candidate, incumbent, and benchmark on identical timestamps and market inputs.
3. Count effective evidence by independent calendar blocks, position episodes, and non-overlapping outcome units. Retain raw decisions only as diagnostics.
4. Use block bootstrap or an equivalent dependence-aware method for confidence intervals on candidate-minus-benchmark net return and drawdown.
5. Require minimum calendar duration and market coverage, not only row count. The initial policy should require at least 30 calendar days, 20 distinct trading days, complete required-symbol coverage, and documented volatility/regime coverage; revise only through a versioned policy change.
6. Require all of the following:
   - positive absolute net expectancy under base and adverse costs;
   - positive paired edge over incumbent and transparent benchmark;
   - drawdown and concentration within approved limits;
   - no unresolved data, risk, accounting, or runtime blocker;
   - trial-aware robustness evidence;
   - stable behavior across symbols and regimes rather than one-symbol dominance.
7. Separate threshold tuning evidence from promotion evidence. Tuning consumes development data; promotion consumes subsequent untouched shadow data.
8. Record failures and rejected candidates permanently so future trial counts and selection bias are not reset.

### Exit gate

An evaluator decision can be reproduced from immutable inputs and cannot pass through duplicated decisions, overlap-inflated counts, negative absolute expectancy, or operational blockers.

## WP-13 - Registry And Promotion Governance

**Priority:** P0 before any activation
**Dependencies:** WP-02, WP-10, WP-12

### Build tasks

1. Replace mutable JSON status as authority with append-only registry events and an atomically derived active pointer.
2. Define legal transitions:

```text
BUILT -> VALIDATED -> SHADOW -> ELIGIBLE -> APPROVED -> CANARY -> ACTIVE
   \-> REJECTED      \-> EXPIRED       \-> ROLLED_BACK
```

3. Ensure top-level status and tags are projections from the same events so contradictory `active`/`paper_quarantine` state is impossible.
4. Keep automatic promotion code disabled and remove it from production configuration. Reintroduction requires a separate security and governance review after stable live operation.
5. Require two distinct authorized approvals for live activation. Approvals include evidence-bundle digest, expiry, scope, reason, and identity.
6. Prohibit an override from changing quantitative metrics or eligibility. Emergency actions can reject, pause, rollback, or extend evaluation only.
7. Replace direct file/DB patch helpers with authenticated control-plane commands that validate transition preconditions and append audit events.
8. Make rollback select only previously compatible, attested artifacts and execute a readiness check before pointer change.
9. Bind every active pointer to image compatibility and feature/data contracts.

### Primary files

- `quant_v2/model_registry.py`
- `quant_v2/research/model_evaluator.py`
- Telegram administration handlers
- registry storage/migrations
- `MODEL_PROMOTION_RUNBOOK.md`

### Exit gate

No single operator or mutable flag can activate an ineligible model, and every transition is attributable, reproducible, and reversible.

## WP-14 - Observability, Data Health, And Performance Diagnosis

**Priority:** P1
**Dependencies:** Cross-cutting; finalize after WP-03 through WP-13 contracts stabilize

### Build tasks

1. Emit structured metrics for each stage: data fetch, feature build, inference, allocation, planning, routing, fill, ledger commit, reconciliation, and notification.
2. Track data freshness, per-symbol gaps, duplicate bars, supplementary-source coverage, DNS errors, API latency/rate limits, feature missingness, and quarantined symbols.
3. Track model probability/calibration drift, feature drift, turnover, realized cost, rejected intents, risk headroom, concentration, and candidate/incumbent paired performance.
4. Track system CPU, memory, disk, I/O wait, network/DNS latency, container restarts, queue lag, Redis memory, database lock time, and WAL/reconciliation lag.
5. Add correlation IDs spanning signal, intent, order, fill, ledger, and notification without including credentials.
6. Send security, lifecycle, and accounting logs to an append-only destination outside the trading container with bounded retention and access controls.
7. Define service-level indicators and initial objectives from measured baseline rather than attributing performance to Ubuntu without evidence.
8. Add synthetic market/data-provider probes and stale-data circuit breakers.

### Exit gate

A single incident timeline can be reconstructed from correlated structured events without reading secrets or inferring state from free-form logs.

## WP-15 - Deployment, Migration, And Runtime Readiness

**Priority:** P0 before restart
**Dependencies:** WP-02 through WP-14 as applicable

### Build tasks

1. Create an idempotent deployment command that accepts an image digest and environment manifest, not a mutable tag.
2. Preflight:
   - clean/attested image;
   - database backup and migration dry-run;
   - compatible active model;
   - secrets and Redis ACL health;
   - data-provider/DNS health;
   - disk and volume permissions;
   - evaluator auto-promotion disabled;
   - no unmanaged open positions.
3. Run database migrations with forward and rollback rehearsals against a production-sized copy.
4. Start dependencies, accounting/reconciliation, supervisor, signal source, and user interface in that order. Readiness must reflect dependency and model health.
5. Run post-deploy smoke replay and compare expected deterministic output.
6. Record image digest, configuration digest, model digest, migration version, smoke result, and operator approvals.
7. Roll back by digest and compatible model pointer. Never rebuild during rollback.
8. Keep old state volumes read-only until reconciliation and retention gates pass.

### Exit gate

The exact tested digest can be deployed and rolled back repeatably, with state migration and model compatibility proven before traffic or trading activation.

## 6. Pull Request And Dependency Sequence

Keep pull requests narrow enough to review and test independently. A practical sequence is:

| Order | Pull request | Depends on | Primary proof |
|---:|---|---|---|
| 1 | Containment configuration and credential-safe logging | none | token canary and Redis reachability tests |
| 2 | Canonical source reconciliation and repository hygiene | incident evidence | clean source manifest |
| 3 | Dependency lock, image test stage, and attestations | PR 2 | exact-image full suite |
| 4 | Hard/dynamic/operating risk policy types | PR 2 | cap property tests |
| 5 | Projected exposure, headroom, and reduction bypass | PR 4 | incident replay |
| 6 | Durable risk lifecycle and independent supervisor | PR 5 | chaos lifecycle suite |
| 7 | Execution outcome taxonomy and idempotent fill schema | PR 2 | duplicate delivery tests |
| 8 | Ledger, projections, and reconciliation | PR 7 | deterministic rebuild tests |
| 9 | Shared cost policy and paper adapter | PR 8 | cost attribution golden tests |
| 10 | Canonical panel, data manifest, and feature contract | PR 2 | panel/leakage tests |
| 11 | Temporal validation and trial ledger | PR 9, PR 10 | synthetic leakage and holdout proof |
| 12 | Final refit, artifact manifest, runtime compatibility | PR 3, PR 11 | golden parity and mismatch rejection |
| 13 | Executable portfolio replay | PR 5, PR 8, PR 9, PR 10 | deterministic replay |
| 14 | Shadow evaluator statistical policy | PR 11, PR 12, PR 13 | paired block-aware report |
| 15 | Append-only registry and two-person promotion | PR 12, PR 14 | illegal-transition tests |
| 16 | Structured observability and external audit sink | contracts stabilized | correlated incident test |
| 17 | Deployment/migration automation and readiness harness | all restart-critical PRs | staging deploy/rollback rehearsal |

PRs 4-6, 7-9, and 10-12 form three engineering lanes after the canonical baseline. They may proceed in parallel, but integration into replay and promotion waits for all three.

## 7. Test And Verification Program

### Test layers

1. **Unit tests:** typed contracts, formulas, transition guards, schema validation, cost components, and statistics.
2. **Property tests:** risk monotonicity, exposure reduction, idempotency, ledger balance, replay invariance, and shuffled panel order.
3. **Contract tests:** adapter outcomes, feature schema, model manifest, registry transitions, Redis ACL, and database migrations.
4. **Integration tests:** signal through fill and ledger; hard breach through supervisor; candidate through replay and evaluator.
5. **Golden tests:** June incident state, fixed market bundle, expected ledger, and expected model parity rows.
6. **Leakage tests:** future perturbation, symbol-boundary labels, chronological folds, threshold/holdout separation, and as-of supplementary data.
7. **Chaos tests:** process kill, Redis restart, database lock/failure, stale data, provider DNS failure, partial fill, venue rejection, and network partition.
8. **Image tests:** full suite and smoke tests run inside the exact release image.
9. **Security tests:** secret scan, token log canary, network exposure, least privilege, dependency scan, and image-layer inspection.
10. **Staging rehearsals:** migration, shadow operation, forced breach, flatten, rollback, and evidence bundle generation.

### Required CI gates

- Formatting/linting and static checks.
- All tests pass with no newly unreviewed warnings.
- Coverage thresholds on risk, accounting, data, validation, registry, and deployment control modules.
- Database migration upgrade/downgrade rehearsal.
- Deterministic replay comparison.
- No secrets or prohibited production artifacts.
- Clean source tree and manifest match.
- Image vulnerability policy passes.
- Model compatibility and feature parity pass.
- Audit finding traceability check has no unowned or untested P0/P1 item.

## 8. Data And Schema Migration Strategy

1. Freeze and hash the current SQLite database as incident evidence.
2. Introduce explicit migration tooling before new ledger or lifecycle tables.
3. Rehearse against a copy at production size and measure lock duration.
4. Add new tables without deleting legacy route/state tables.
5. Import only events supported by evidence. Attach provenance and confidence to imported records.
6. Establish a cutover sequence number and opening balance/position event approved from reconciled state.
7. Mark earlier lifetime metrics as `LEGACY_UNVERIFIABLE`; retain them for reference but exclude them from promotion and current performance.
8. Run dual projections during shadow: legacy state and ledger-derived state. Any difference blocks cutover.
9. Switch reads to the ledger projection only after a complete soak with zero unexplained differences.
10. Retire legacy writes after rollback window expiry; retain read-only evidence per policy.

## 9. Rollout Stages And Stop Conditions

### Stage A - Offline recovery

- No network order routing.
- Complete WP-00 through WP-10 and deterministic historical replay.
- Reproduce the June incident and demonstrate corrected behavior.

**Stop immediately if:** any risk cap can expand beyond hard limits, labels cross symbols/time, ledger replay differs, or artifacts load under mismatched runtime.

### Stage B - Live-data shadow

- Read live market data; generate isolated candidate/incumbent/benchmark state.
- No executable adapter credentials.
- Run at least 30 calendar days under the initial evidence policy.

**Stop or reset evidence window if:** data coverage fails, code/model/policy changes, accounting diverges, or runtime identity changes. Security fixes may proceed, but their deployment must be recorded.

### Stage C - Paper execution soak

- Use realistic cost and partial-fill simulation.
- Exercise hard breach, supervisor, restart, migration, and rollback drills.
- Require zero unexplained reconciliation differences for the full acceptance window.

**Stop immediately if:** exposure becomes unmanaged, reductions are blocked, lifecycle ownership is ambiguous, or PnL cannot be rebuilt.

### Stage D - Restricted canary

- Requires explicit separate approval after all restart gates pass.
- One account, minimal capital, reduced universe, conservative operating caps, no auto-promotion, and automatic rollback/flatten thresholds.
- Increase only after a predeclared observation period and review; never scale solely because short-run PnL is positive.

**Stop immediately if:** any hard risk, reconciliation, stale-data, model compatibility, security, or provenance gate fails.

### Stage E - Staged scale

- Expand capital and universe in discrete approved steps.
- Re-estimate cost/capacity and concentration at each step.
- Preserve benchmark and incumbent shadow comparisons.

## 10. Restart Acceptance Checklist

No paper or live execution restart is approved until the exact candidate image demonstrates:

- [ ] Source branch is clean and reconciled against retained production behavior.
- [ ] Image digest is tied to Git SHA, source manifest, lock, tests, and SBOM.
- [ ] Telegram credential is rotated and absent from new logs and image layers.
- [ ] Redis is not publicly exposed and uses approved ACL/secrets configuration.
- [ ] Auto-promotion is disabled by code and deployment policy.
- [ ] No direct DB/JSON helper can clear pauses or promote models in production.
- [ ] Dynamic limits never exceed immutable hard limits.
- [ ] Operating targets include tested headroom.
- [ ] Projected post-trade exposure is checked at conservative current marks.
- [ ] Reductions and flattens bypass every normal alpha gate.
- [ ] Hard breach reaches flat confirmation or a supervised incident with continuous heartbeat.
- [ ] Final marks, positions, equity, and PnL commit before session stop.
- [ ] Fill replay creates no duplicate economic event.
- [ ] Ledger, adapter, projections, and marked equity reconcile exactly within declared precision.
- [ ] Paper/replay PnL includes fees, spread, slippage, funding, latency, and partial-fill behavior.
- [ ] Dataset manifests pass per-symbol completeness and point-in-time checks.
- [ ] Labels and folds preserve symbol and chronological identity.
- [ ] Final holdout was untouched by model, feature, threshold, window, and cost selection.
- [ ] Candidate was refit on its declared complete development set.
- [ ] Artifact runtime and feature schema match the release image exactly.
- [ ] Missing/stale required features fail closed or quarantine under explicit policy.
- [ ] Executable replay is positive after approved adverse costs and within risk limits.
- [ ] Shadow evidence is paired, dependence-aware, trial-aware, and positive in absolute terms.
- [ ] Candidate beats incumbent and transparent benchmark without unacceptable concentration.
- [ ] Promotion has two unexpired approvals bound to the evidence digest.
- [ ] Staging deployment, forced breach, flatten, migration, and rollback rehearsals pass.
- [ ] Operations has an owner, alerts, runbooks, and rollback digest for the release.

Any failed box returns the release to the relevant work package. A verbal exception is not an acceptance artifact.

## 11. Definition Of Done By Discipline

### Risk and execution

Risk behavior is encoded as invariants, exposure reduction cannot be blocked by alpha controls, and lifecycle ownership persists through failures until flat confirmation.

### Accounting

Every economic event is immutable and replayable; current state is a projection; reconciliations are continuous and unexplained differences stop new risk.

### Research

Datasets and trials are immutable, point-in-time, symbol-aware, chronological, reproducible, and evaluated on untouched time with realistic costs and selection-bias evidence.

### Model operations

Artifacts are compatible, feature-complete, content-addressed, shadowed through executable portfolio logic, and activated only through append-only two-person governance.

### Platform and security

Secrets do not leak, Redis is least-privilege and private, images are clean and attested, and the tested digest is the deployed digest.

### Operations

An operator can reconstruct state and causality from structured events, detect data/runtime drift, force a supervised reduction, and roll back without rebuilding.

## 12. Effort And Staffing Shape

This is a multi-workstream recovery, not a one-sprint patch. Relative sizing for planning:

| Lane | Scope | Estimated focused engineering effort |
|---|---|---:|
| Containment, source, build, security | WP-00 to WP-02 | 8-13 engineer-days |
| Risk lifecycle | WP-03 to WP-04 | 10-16 engineer-days |
| Execution/accounting/costs | WP-05 to WP-07 | 15-24 engineer-days |
| Data/validation/artifacts | WP-08 to WP-10 | 15-24 engineer-days |
| Replay/evaluator/governance | WP-11 to WP-13 | 15-25 engineer-days |
| Observability/deployment/readiness | WP-14 to WP-15 | 10-16 engineer-days |

These are implementation and verification ranges, not calendar promises. Parallel work is possible after WP-01, but the same person should not be the sole author and approver for risk policy, accounting invariants, or promotion governance. Shadow and canary observation periods add elapsed time that cannot be responsibly compressed with more engineers.

## 13. Governance And Ownership

Assign named owners before implementation begins:

| Area | Required owner | Independent reviewer |
|---|---|---|
| Incident containment and secrets | Operations/security | Engineering lead |
| Risk policy and lifecycle | Execution engineer | Risk owner |
| Ledger and reconciliation | Accounting/execution engineer | Independent reviewer |
| Dataset and validation | Research engineer | Quant reviewer not selecting the candidate |
| Artifact and promotion registry | ML/platform engineer | Operations/risk owner |
| Build and deployment | Platform engineer | Security/engineering lead |
| Restart decision | Risk owner | Operations plus second approver |

The candidate researcher must not be the only approver of the candidate's validation policy or promotion evidence.

## 14. Open Decisions To Resolve During Design Review

These decisions do not block containment or canonical-source work, but they must be resolved before their work packages merge:

1. Durable accounting database target for production growth and concurrency: hardened SQLite with strict single-writer ownership versus PostgreSQL.
2. External append-only telemetry destination and retention policy.
3. Exact immutable hard limits and headroom range by paper/canary/live mode.
4. Exchange precision and residual-position policy below minimum notional.
5. Approved cost-policy sources and conservative fallbacks by symbol.
6. Final holdout length, rolling-window candidates, and minimum regime coverage based on available history.
7. Transparent benchmark definition that is simple, low-turnover, and fixed before candidate evaluation.
8. Initial shadow and canary capital/duration gates after effective sample-size analysis.
9. Two-person identity and authorization mechanism for promotion and pause-clear workflows.

Each decision must be recorded as a versioned architecture or policy decision, not an undocumented environment-variable change.

## 15. First Implementation Milestone

The first milestone should contain only WP-00 through WP-03 foundations:

1. Complete containment and rotate exposed credentials.
2. Establish the clean canonical recovery branch.
3. Produce an exact-image locked build with tests and attestations.
4. Centralize immutable risk limits and prove dynamic-policy/headroom invariants.

At that milestone the system remains paused. The deliverable is trustworthy source and bounded risk mathematics, not resumed trading. Lifecycle, ledger, research, evaluation, and staged rollout then build on a stable base.

## 16. Final Build Direction

Recover the system by making correctness and evidence part of the product, not surrounding documentation. Keep the useful existing architecture, but establish one source of truth at each layer:

- immutable policy for hard risk;
- durable supervisor for exposure ownership;
- append-only ledger for economic state;
- canonical panel and manifests for research data;
- executable replay for performance evidence;
- append-only registry for model governance;
- immutable image digest for deployment identity.

Only after those truths agree should model performance determine whether the system trades. Until then, training more often, shortening the history window, or adding adaptive models would increase confidence faster than it increases knowledge.
