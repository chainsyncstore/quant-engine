# Production Resume Gate Recovery Spec

**Spec date:** 2026-06-27
**Implementer profile:** GPT-5.4 mini orchestrating spawned implementation agents
**Audit owner:** GPT-5.5 lead/audit agent
**Current disposition:** **DO NOT RESUME PRODUCTION TRADING**
**Goal:** Open the production resume gate only after source/runtime parity, replay performance, model evidence, paper-soak evidence, and operational safety gates all pass.

## 1. Operating Rules

This spec is the implementation authority for the next recovery sequence. It does not authorize live trading, manual model promotion, threshold weakening, credential changes, or hard-risk bypasses.

Hard constraints:

- Keep production trading disabled until Phase 6 passes.
- Do not lower `RETRAIN_MIN_ACCURACY`, holdout accuracy gates, cost-adjusted expectancy gates, drawdown gates, symbol-coverage gates, or benchmark-comparison gates to force a pass.
- Do not promote a model unless this spec explicitly reaches the paper-soak and resume-approval phases.
- Do not clear, edit, or fabricate risk/accounting state to make a gate pass.
- Do not commit secrets, private IPs, account IDs, Tailscale hostnames, bot tokens, exchange keys, chat IDs, or database dumps.
- Treat 4arm host code, running container code, local Windows code, and canonical Git source as separate artifacts until proven identical.
- Every spawned implementation agent must produce an end note using the required format in Section 3.

## 2. Required Preflight

Before any phase implementation:

1. Record current local and 4arm state:

   ```bash
   git status --short
   git rev-parse HEAD
   git remote -v
   ssh 4arm-ubuntu "cd /home/admin-4arm/hypothesis-research-engine && git status --short && git rev-parse HEAD"
   ssh 4arm-ubuntu "docker ps --format '{{.Names}} {{.Image}} {{.Status}}'"
   ```

2. Confirm runtime remains no-trade:

   ```bash
   ssh 4arm-ubuntu "docker logs --tail 200 quant_telegram | grep -Ei 'No production model|no_active_model|hard_risk|paper|live' || true"
   ssh 4arm-ubuntu "docker logs --tail 200 quant_retrain | grep -Ei 'candidate|registered|published|promoted|failed|accuracy' || true"
   ```

3. Confirm recovery snapshot availability:

   ```bash
   ssh 4arm-ubuntu "ls -lh /home/admin-4arm/hypothesis-research-engine/datasets/v2/snapshots/*20260626*.parquet"
   ```

If preflight shows live trading is active, a model was promoted, or accounts are not controlled by the expected no-trade state, stop and report to the audit owner.

## 3. Sub-Agent Orchestration Contract

The 5.4 mini lead may spawn implementation agents, but each phase must be accepted by the lead before the next phase mutates code or runtime state.

Each implementation agent must receive:

- The exact phase name.
- Files it may edit.
- Files it must not edit.
- Commands it must run.
- Acceptance evidence it must produce.

Each implementation agent must return this end note:

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

The 5.4 mini lead must then audit the phase and write:

```text
LEAD AUDIT DECISION: accept | reject
ACCEPTANCE EVIDENCE:
REJECTION REASON:
CORRECTION INSTRUCTIONS:
AUDIT_REPORT_UPDATE:
```

The phase is not complete until `audit_report.md` is updated with a dated checkpoint.

## 4. Phase 1 - Remote Source And Runtime Reconciliation

### Objective

Make local source, canonical Git source, 4arm host source, and running container source consistent enough that tests and experiments mean the same thing everywhere.

### Work Items

1. Identify the canonical remote:

   ```bash
   git remote -v
   ssh 4arm-ubuntu "cd /home/admin-4arm/hypothesis-research-engine && git remote -v"
   ```

2. Compare the following artifacts:

   - Local Windows working tree.
   - Local `origin/main`.
   - 4arm host working tree.
   - `quant_retrain` container `/app`.

3. Produce a reconciliation table in `docs/runtime_reconciliation/source_runtime_reconciliation.md` with columns:

   ```text
   artifact | git_sha | dirty_state | missing_files | extra_files | decision | reason
   ```

4. Fix the known remote test drift:

   - `tests/quant_v2/test_scheduled_retrain_candidates.py` expects `quant_v2.research.scheduled_retrain.fetch_symbol_dataset`.
   - Either restore/export that function from the canonical implementation or update tests and implementation together if the function was intentionally renamed.
   - Do not mark the test obsolete without proving the replacement contract.

5. Ensure these files exist on 4arm host and inside the runtime image or mounted `/app`:

   - `quant_v2/research/model_recovery_experiments.py`
   - `quant_v2/research/portfolio_replay.py`
   - `quant_v2/research/scheduled_retrain.py`
   - `scripts/run_model_recovery_experiments.py`
   - `tests/quant_v2/test_model_recovery_experiments.py`
   - `tests/quant_v2/test_scheduled_retrain_candidates.py`

6. Rebuild or restart only if needed to make container source match the accepted host source. Do not activate trading.

### Acceptance Commands

Run locally:

```powershell
& 'C:\Users\Bloomington\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m pytest tests/quant_v2/test_model_recovery_experiments.py tests/quant_v2/test_model_quality_recovery.py tests/quant_v2/test_scheduled_retrain_candidates.py -q
```

Run on 4arm host:

```bash
cd /home/admin-4arm/hypothesis-research-engine
.venv/bin/python -m pytest tests/quant_v2/test_model_recovery_experiments.py tests/quant_v2/test_model_quality_recovery.py tests/quant_v2/test_scheduled_retrain_candidates.py -q
```

Run inside `quant_retrain` if the tests are present in `/app`; otherwise compile the relevant modules:

```bash
docker exec quant_retrain python - <<'PY'
import py_compile
for path in [
    "/app/quant_v2/research/model_recovery_experiments.py",
    "/app/quant_v2/research/portfolio_replay.py",
    "/app/quant_v2/research/scheduled_retrain.py",
]:
    py_compile.compile(path, cfile=f"/tmp/{path.rsplit('/', 1)[-1]}.pyc", doraise=True)
print("container_compile_ok")
PY
```

### Phase 1 Pass Criteria

- Local targeted tests pass.
- 4arm host targeted tests pass.
- Container compile passes.
- Reconciliation doc exists and explains any remaining intentional differences.
- `audit_report.md` has a dated Phase 1 checkpoint.

## 5. Phase 2 - Portfolio Replay Performance And Determinism

### Objective

Make replay fast enough to run broad recovery sweeps on the real 10-symbol 4arm snapshot without changing replay decisions or weakening gates.

### Required Changes

Profile before editing. Capture timings for:

- `_benchmark_replay_report`
- `_candidate_replay_report`
- `run_portfolio_replay`
- `_replay_actor`

Required implementation targets:

1. Replace repeated per-timestamp `pd.concat` history growth in replay with deterministic indexed history views or precomputed per-symbol arrays.
2. Preserve exact signal, planner, order, fill, fee, position, and equity semantics on existing replay fixtures.
3. Add a research-mode log throttle so replay does not emit thousands of allocation INFO lines during offline experiments. The default production logging behavior must not be silenced unless explicitly configured.
4. Keep compact replay artifacts:
   - No full embedded `equity_curve` in top-level recovery report by default.
   - No uncapped fill list beyond `DEFAULT_REPORT_FILL_LIMIT`.
   - Replay reports must include digests and truncation metadata.

### Performance Targets

On the real 10-symbol 4arm snapshot `datasets/v2/snapshots/model_recovery_real_1h_20260626_20260626_141210.parquet`:

- Top-1 recovery run must complete in **under 6 minutes**.
- Top-12 recovery run must complete in **under 75 minutes**.
- If hardware load prevents these targets, report measured CPU load and wall time; do not claim pass.

### Acceptance Commands

```bash
cd /home/admin-4arm/hypothesis-research-engine
PYTHONWARNINGS=ignore .venv/bin/python scripts/run_model_recovery_experiments.py \
  --snapshot-path datasets/v2/snapshots/model_recovery_real_1h_20260626_20260626_141210.parquet \
  --output-root /tmp/model_recovery_perf_top1 \
  --docs-output-dir /tmp/model_recovery_perf_docs_top1 \
  --max-candidates 1 \
  --no-production-registry

PYTHONWARNINGS=ignore .venv/bin/python scripts/run_model_recovery_experiments.py \
  --snapshot-path datasets/v2/snapshots/model_recovery_real_1h_20260626_20260626_141210.parquet \
  --output-root /tmp/model_recovery_perf_top12 \
  --docs-output-dir /tmp/model_recovery_perf_docs_top12 \
  --max-candidates 12 \
  --no-production-registry
```

### Phase 2 Pass Criteria

- Replay unit/regression tests pass.
- Recovery tests pass.
- Top-1 runtime is under 6 minutes.
- Top-12 runtime is under 75 minutes or the phase is marked blocked with profiler evidence.
- No quality gate is weakened.
- `audit_report.md` has a dated Phase 2 checkpoint.

## 6. Phase 3 - Broad 10-Symbol Recovery Sweep

### Objective

Run a real candidate search on the current 4arm snapshot and decide whether a model is eligible for paper quarantine.

### Work Items

1. Run at least top 24 candidates if Phase 2 performance allows it. If top-24 cannot finish within 150 minutes, run top-12 and mark top-24 as blocked with runtime evidence.
2. Preserve all candidate artifacts:

   - `candidate_config.json`
   - `dataset_manifest.json`
   - `feature_manifest.json`
   - `label_audit.json`
   - `fold_ledger.json`
   - `threshold_policy.json`
   - `holdout_report.json`
   - `replay_report.json`
   - `selection_risk_summary.json`
   - `manifest.json`

3. Summarize ranked candidates in `docs/model_quality/latest_recovery_sweep_summary.md`.
4. Do not publish any model to the production registry from this phase.

### Acceptance Commands

```bash
cd /home/admin-4arm/hypothesis-research-engine
PYTHONWARNINGS=ignore .venv/bin/python scripts/run_model_recovery_experiments.py \
  --snapshot-path datasets/v2/snapshots/model_recovery_real_1h_20260626_20260626_141210.parquet \
  --output-root /tmp/model_recovery_sweep_top24 \
  --docs-output-dir docs/model_quality/recovery_sweep_top24 \
  --max-candidates 24 \
  --no-production-registry
```

### Phase 3 Decision Rules

If at least one candidate passes all gates:

- Mark recommendation `paper_quarantine_candidate`.
- Record selected candidate ID, artifact path, data digest, feature digest, code SHA, and replay digest.
- Proceed to Phase 5 only after the audit owner accepts Phase 3.

If no candidate passes:

- Keep recommendation `remain_no_trade`.
- Proceed to Phase 4.

### Phase 3 Pass Criteria

- Sweep artifacts exist.
- Summary doc exists.
- No production active pointer changed.
- No model was promoted.
- Decision follows the exact rules above.
- `audit_report.md` has a dated Phase 3 checkpoint.

## 7. Phase 4 - Research Input Repair If Sweep Fails

### Objective

If broad recovery still fails, improve the research contract rather than lowering gates.

### Required Diagnostics

Produce `docs/model_quality/research_input_diagnostics.md` containing:

1. Prediction distribution by candidate, horizon, symbol, and month.
2. Explicit check for all-buy or all-sell collapse.
3. Label distribution by symbol and month.
4. Cost floor by symbol.
5. Feature missingness and constant-feature report.
6. Feature importance or permutation-importance report for the best failed candidates.
7. Regime split diagnostics:
   - high/low volatility
   - BTC trend up/down
   - funding positive/negative
   - drawdown/recovery windows
8. Benchmark comparison by regime.

### Allowed Research Variants

Implement no more than four variants per pass:

1. Cost-aware ternary target with explicit no-trade class.
2. Horizon-specific feature set and threshold policy.
3. Symbol-group calibration or per-symbol threshold calibration trained only on development folds.
4. Regime-gated candidate where the model can explicitly abstain.

Do not use final holdout results to choose thresholds. Thresholds must be selected inside development folds only.

### Required Outputs

For every variant:

- Config JSON.
- Dataset manifest.
- Feature manifest.
- Fold ledger.
- Threshold policy.
- Holdout report.
- Replay report.
- Selection risk summary.
- Clear pass/fail decision.

### Phase 4 Pass Criteria

- Diagnostics doc exists.
- At least one implemented variant addresses the observed failure mode.
- No final holdout leakage is introduced.
- Tests prove labels remain grouped by symbol and chronological.
- If a variant passes all gates, return to Phase 3 decision rules.
- If no variant passes, keep production gate closed and report `remain_no_trade`.
- `audit_report.md` has a dated Phase 4 checkpoint.

## 8. Phase 5 - Paper Quarantine And Soak

### Entry Condition

Phase 5 may start only if Phase 3 or Phase 4 produces a candidate that passes every offline gate and the audit owner accepts it.

### Objective

Validate profitability and runtime behavior in paper/demo only before production consideration.

### Setup Requirements

1. Start from flat demo state for both users.
2. Confirm no live mode is enabled:

   ```bash
   ssh 4arm-ubuntu "docker logs --tail 300 quant_telegram | grep -Ei 'live_mode=1|LIVE|production trade' && exit 1 || true"
   ```

3. Register candidate only as paper quarantine or shadow candidate.
4. Keep auto-promotion disabled unless the audit owner explicitly approves enabling it for paper-only evaluation.
5. Persist paper-soak evidence to `docs/model_quality/paper_soak/<candidate_id>/`.

### Minimum Soak Duration

Run for at least:

- 168 hours, or
- 100 paper decisions,

whichever takes longer.

### Paper Pass Criteria

The paper candidate must show:

- Positive absolute cost-adjusted PnL.
- Positive PnL versus flat.
- Positive PnL versus best transparent benchmark.
- No hard-risk breach.
- No unowned open positions.
- No stale-data trading.
- No missing-feature runtime warnings.
- No schema migration errors.
- No credential-bearing URLs in logs.
- Max drawdown within approved gate.
- Symbol concentration within approved gate.

### Phase 5 Pass Criteria

- Paper-soak report exists.
- Logs and database evidence support the report.
- Both users remain demo/paper only.
- Candidate either passes paper gate or is rejected.
- `audit_report.md` has a dated Phase 5 checkpoint.

## 9. Phase 6 - Production Resume Gate Review

### Objective

Decide whether production trading can resume.

### Required Evidence Bundle

Create `docs/production_resume/resume_evidence_bundle.md` containing:

1. Canonical Git commit SHA.
2. Image digest.
3. Dependency lock digest.
4. Dataset snapshot digest.
5. Feature schema digest.
6. Candidate model digest.
7. Offline sweep report path.
8. Paper-soak report path.
9. Risk/accounting reconciliation report.
10. Container health report.
11. Secret/log-redaction report.
12. Open-position report for both users.
13. Explicit operator action required to resume.

### Production Resume Gates

All must be true:

- Canonical source, 4arm host, and runtime image match the evidence bundle.
- No hard-risk breach is active.
- Both users are flat or positions are owned by a tested reduce-only supervisor.
- Active model artifact matches its manifest and runtime dependency contract.
- Paper-soak candidate passed Phase 5.
- Auto-promotion policy is explicitly documented.
- Redis is not exposed without required auth controls.
- Logs contain no credential-bearing URLs.
- Data-provider health checks pass on host and in container.
- Telegram/operator commands report expected state.
- Audit owner accepts the evidence bundle.

### Phase 6 Decision Rules

If every gate passes:

- Audit owner may mark production resume gate open.
- Resume should still begin with the smallest approved canary exposure.

If any gate fails:

- Keep production trading disabled.
- Write the failed gate and required corrective phase to `audit_report.md`.

## 10. Final Deliverables

At completion, the 5.4 mini lead must provide:

- Phase-by-phase implementation notes.
- All sub-agent end notes.
- Lead accept/reject decisions.
- Test command outputs summarized.
- Artifact paths.
- Runtime/source reconciliation status.
- Final recommendation:

  ```text
  production_resume_gate = open | closed
  reason = ...
  next_required_action = ...
  ```

The 5.5 audit owner will independently inspect artifacts, rerun selected commands, and either accept the gate decision or reject it with correction instructions.
