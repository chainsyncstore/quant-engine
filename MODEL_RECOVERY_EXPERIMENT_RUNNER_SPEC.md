# Model Recovery Experiment Runner Spec

## Purpose

This spec defines the next implementation step after model-quality recovery: build an experiment runner that diagnoses why fresh retrains are failing quality gates and searches for a safer candidate model/configuration without activating trading automatically.

The implementing agent must follow this document exactly. Do not lower quality thresholds, force trades, activate a model, or resume paper/live trading as part of this spec.

## Background

Recent recovery retrains failed to produce an acceptable model. Known observed results from the recovery attempt:

- 2h horizon latest accuracy: `0.5242`, below required `0.6000`
- 4h horizon latest accuracy: `0.5216`, below required `0.6000`
- 8h horizon latest accuracy: `0.5437`, below required `0.6000`
- No acceptable candidate model was registered.
- No active production model should be created from those failed attempts.

The correct response is not to reduce the threshold. The correct response is to audit labels, costs, features, training windows, recency weighting, and benchmark replay performance, then only produce a quarantined candidate if evidence supports it.

## Required Reading Before Editing

Before changing code, read these files if present:

- `MODEL_QUALITY_RECOVERY_SPEC.md`
- `quant_v2/research/model_quality_recovery.py`
- `scripts/model_quality_recovery.py`
- `quant_v2/research/scheduled_retrain.py`
- `quant_v2/models/trainer.py`
- `quant_v2/research/portfolio_replay.py`
- Any existing cost, execution, portfolio, registry, or replay modules under `quant_v2/`
- Existing tests under `tests/quant_v2/` for retrain, registry, model quality, portfolio replay, and risk controls

If any listed file is missing, inspect the nearest equivalent module and note the substitution in the end report.

## Non-Goals

Do not do any of the following:

- Do not lower `RETRAIN_MIN_ACCURACY` or equivalent gates to make a model pass.
- Do not enable live trading.
- Do not enable paper trading automatically.
- Do not clear hard-risk breach state.
- Do not send Telegram commands.
- Do not open, close, or flatten positions.
- Do not modify exchange credentials, API keys, bot tokens, chat IDs, or secrets.
- Do not write hostnames, IP addresses, private paths, credentials, account IDs, or Tailscale details into docs.
- Do not write to the active production model pointer.
- Do not register an experiment candidate as an active production model.
- Do not tune on final holdout or paper-soak results.
- Do not deploy unless the user separately asks for deployment.

## Implementation Summary

Add a model recovery experiment system with three outputs:

1. A dataset and label audit report.
2. A benchmark replay report.
3. A ranked experiment report that either selects a quarantined candidate configuration or explicitly recommends remaining in no-trade mode.

The runner must create experiment artifacts only under:

```text
models/experiments/model_recovery/<run_id>/
```

The runner must not write to:

```text
models/production/active.json
models/production/registry/active.json
```

The runner must not append production-promotion events to:

```text
models/production/registry/registry_events.jsonl
```

## Files To Add Or Change

Add:

- `quant_v2/research/model_recovery_experiments.py`
- `scripts/run_model_recovery_experiments.py`
- `tests/quant_v2/test_model_recovery_experiments.py`
- `docs/model_quality/experiment_policy_v1.md`

Change only if needed:

- `quant_v2/research/model_quality_recovery.py`
- `scripts/model_quality_recovery.py`
- Existing test helpers under `tests/quant_v2/`

Do not edit `.env` for this spec.

Do not edit deployment scripts for this spec.

## Run ID Contract

Each experiment run must use a deterministic, readable run id:

```text
YYYYMMDDTHHMMSSZ_<git_short_sha>
```

If git metadata is unavailable, use:

```text
YYYYMMDDTHHMMSSZ_unknown
```

Example:

```text
20260626T084200Z_b159364
```

## CLI Contract

Create this CLI:

```bash
python scripts/run_model_recovery_experiments.py \
  --snapshot-path <snapshot_path> \
  --output-root models/experiments/model_recovery \
  --docs-output-dir docs/model_quality \
  --max-candidates 144 \
  --no-production-registry
```

Required arguments:

- `--snapshot-path`: path to the training snapshot or dataset.

Optional arguments:

- `--output-root`: default `models/experiments/model_recovery`
- `--docs-output-dir`: default `docs/model_quality`
- `--max-candidates`: default `144`
- `--seed`: default `1337`
- `--min-accuracy`: default must come from existing quality policy; fallback `0.60`
- `--min-actionable-decisions`: default `100`
- `--max-drawdown`: default must come from existing replay/risk policy if available
- `--no-production-registry`: required safety flag; when set, production registry writes are forbidden

The CLI must fail fast with a non-zero exit code if `--no-production-registry` is absent.

The CLI must print:

- run id
- output directory
- number of candidates evaluated
- number of candidates passing gates
- final recommendation: `remain_no_trade` or `paper_quarantine_candidate`

## Dataset Contract

Use existing dataset loaders where available. Do not invent a parallel data format if the codebase already has one.

The experiment runner must support a multi-symbol time-series dataset containing, at minimum:

- timestamp
- symbol
- open, high, low, close, volume or equivalent OHLCV columns

If additional columns exist, preserve them for feature experiments:

- funding
- open interest
- spread
- order book derived features
- volatility features
- regime features

Tests must not require real production data. Use synthetic fixtures.

## Label Audit Requirements

Implement a label audit grid across:

- horizons: `2h`, `4h`, `8h`
- dead-zone policies:
  - `cost_floor`
  - `0.001`
  - `0.0015`
  - `0.002`
  - `0.003`
  - `0.005`
- training windows in months: `3`, `6`, `9`, `12`
- recency half-life days: `30`, `60`, `90`

For each grid cell, compute:

- sample count
- symbol count
- label counts by class
- label percentage by class
- ambiguous or no-trade label rate
- forward-return mean
- forward-return median
- forward-return standard deviation
- forward-return 25th and 75th percentile
- estimated cost floor
- earliest timestamp
- latest timestamp
- per-symbol label counts
- per-month label counts

Write:

```text
<run_dir>/label_audit.json
<run_dir>/label_audit.md
docs/model_quality/latest_label_audit.md
```

## Benchmark Replay Requirements

Implement or reuse transparent benchmark strategies:

- `flat`: never trade
- `momentum`: simple trend-following baseline
- `mean_reversion`: simple reversal baseline
- `volatility_filtered`: trades only when volatility conditions pass

Benchmarks must use the same:

- dataset snapshot
- cost model
- replay engine
- position sizing assumptions
- risk constraints
- train/validation/holdout splits

For each benchmark, compute:

- total return
- cost-adjusted return
- win rate
- number of decisions
- number of trades
- max drawdown
- average trade return
- median trade return
- exposure time
- per-symbol performance if symbols are available

Write:

```text
<run_dir>/benchmark_replay.json
<run_dir>/benchmark_replay.md
docs/model_quality/latest_benchmark_replay.md
```

## Candidate Experiment Grid

Evaluate candidate configurations across:

- horizon: `2h`, `4h`, `8h`
- training window months: `3`, `6`, `9`, `12`
- recency half-life days: `30`, `60`, `90`
- dead-zone policy: use only policies produced by the label audit grid
- feature set:
  - `full`
  - `price_volume_funding`
  - `no_open_interest`
  - `no_orderbook_placeholders`

Limit the total candidate count to `--max-candidates`.

If the raw grid exceeds `--max-candidates`, select candidates deterministically by prioritizing:

1. healthier label balance
2. lower ambiguous/no-trade label rate
3. sufficient sample count
4. shorter and medium recency half-lives before longer half-lives
5. all horizons represented if possible

Do not randomly discard candidates unless a fixed seed is used and the selection is recorded.

## Candidate Artifacts

Each candidate must write its own directory:

```text
<run_dir>/candidates/<candidate_id>/
```

Each candidate directory must contain:

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

`manifest.json` must include:

- run id
- candidate id
- git short SHA or `unknown`
- created timestamp in UTC
- snapshot path hash or stable identifier
- horizon
- training window
- recency half-life
- dead-zone policy
- feature set
- model type
- quality gates
- pass/fail status
- failure reasons

## Fold And Holdout Rules

Use time-series validation only.

Do not shuffle time-series rows.

Use at least three validation folds if the dataset size allows it.

The final holdout must be chronologically after all train/validation folds.

The threshold policy must be selected from train/validation folds only.

The final holdout can only be used once for reporting. It must not be used to change thresholds, features, dead zones, or model selection rules inside the same run.

## Ranking Rules

A candidate can pass only if all conditions are true:

- latest or holdout accuracy meets the existing minimum accuracy policy, default fallback `0.60`
- cost-adjusted expectancy is positive
- replay result beats the `flat` benchmark
- replay result beats at least one non-flat transparent benchmark
- max drawdown is within the configured bound
- actionable decision count is at least `--min-actionable-decisions`
- no single symbol dominates performance in a way that violates existing concentration policy
- fold results do not show severe instability

If no candidate passes, write:

```json
{
  "recommendation": "remain_no_trade",
  "reason": "No candidate passed all quality and replay gates."
}
```

If one or more candidates pass, write:

```json
{
  "recommendation": "paper_quarantine_candidate",
  "candidate_id": "<candidate_id>",
  "reason": "Candidate passed all quality and replay gates and is eligible for paper quarantine only."
}
```

## Quarantine Rules

Passing candidates are not production active models.

Passing candidates may only be marked:

```text
paper_quarantine
```

The implementation must not set:

```text
active
production
live
auto_promoted
```

Auto-promotion must remain disabled for this recovery flow unless a later spec explicitly enables it after paper-soak validation.

## Summary Reports

Write:

```text
<run_dir>/experiment_summary.json
<run_dir>/experiment_summary.md
docs/model_quality/latest_experiment_summary.md
```

The markdown summary must include:

- run id
- dataset date range
- symbol count
- candidate count evaluated
- candidate count passed
- benchmark summary table
- top five candidate table
- rejected candidate reason summary
- final recommendation
- next operator action

The next operator action must be one of:

- `remain_no_trade_and_collect_more_data`
- `adjust_research_spec_before_retrain`
- `paper_quarantine_selected_candidate`

## Safety Checks

Add explicit guardrails in code:

- If the CLI is invoked without `--no-production-registry`, exit non-zero.
- If code attempts to write an active production pointer, raise an exception.
- If code attempts to append a production promotion event, raise an exception.
- If a candidate is below quality gates, prevent model artifact export except as an experiment report.
- If the dataset has fewer than two symbols, add a warning to all summary reports.
- If any single symbol contributes more than 50 percent of positive replay PnL, mark concentration risk as failed unless existing policy uses a stricter threshold.

## Testing Requirements

Add unit tests that prove:

1. CLI requires `--no-production-registry`.
2. Label audit computes class counts, ambiguous rate, per-symbol counts, and per-month counts.
3. Candidate grid generation is deterministic.
4. Candidate limiting honors `--max-candidates`.
5. Holdout rows are chronologically after train and validation rows.
6. Threshold selection does not use final holdout rows.
7. A failing candidate cannot be marked active.
8. A passing candidate can only be marked `paper_quarantine`.
9. Summary recommendation is `remain_no_trade` when no candidates pass.
10. Summary recommendation is `paper_quarantine_candidate` when at least one candidate passes all gates.
11. No test writes to real `models/production`.
12. No test requires network access.

Use synthetic data fixtures. Do not use production snapshots in tests.

## Validation Commands

Run these commands before reporting completion:

```bash
python -m py_compile quant_v2/research/model_recovery_experiments.py scripts/run_model_recovery_experiments.py
python -m pytest tests/quant_v2/test_model_recovery_experiments.py tests/quant_v2/test_model_quality_recovery.py -q
```

If the repository has additional scheduled retrain tests, also run:

```bash
python -m pytest tests/quant_v2/test_scheduled_retrain_candidates.py -q
```

If a command cannot be run, report the exact command and the exact reason.

## Manual Dry Run

After tests pass, run a dry run against a non-production synthetic or local snapshot:

```bash
python scripts/run_model_recovery_experiments.py \
  --snapshot-path <snapshot_path> \
  --output-root models/experiments/model_recovery \
  --docs-output-dir docs/model_quality \
  --max-candidates 12 \
  --no-production-registry
```

Verify:

- A run directory is created under `models/experiments/model_recovery/`.
- `experiment_summary.json` exists.
- `experiment_summary.md` exists.
- `docs/model_quality/latest_experiment_summary.md` exists.
- No active production pointer is created or modified.
- No production promotion registry event is appended.

## Definition Of Done

This spec is complete only when all of the following are true:

- Required files are implemented.
- Tests pass.
- Py-compile passes.
- Dry run creates the required artifacts.
- Production registry and active pointers remain untouched.
- The final report states whether the system should remain no-trade or has a paper-quarantine candidate.
- The end note includes changed files, validation commands, outputs created, and any risks or follow-up recommendations.

## End Note Format

The implementing agent must finish with this exact structure:

```text
Implementation complete: yes/no

Changed files:
- <file>

Validation:
- <command>: pass/fail/not run

Artifacts:
- <path>

Final recommendation:
- remain_no_trade / paper_quarantine_candidate

Notes:
- <risk or follow-up>
```
