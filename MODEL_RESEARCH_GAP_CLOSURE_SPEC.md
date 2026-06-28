# Model Research Gap Closure Spec

**Spec date:** 2026-06-27
**Implementer profile:** GPT-5.4 mini orchestrating spawned implementation agents
**Audit owner:** GPT-5.5 lead/audit agent
**Source audit:** `COMPARATIVE_QUANT_RESEARCH_AUDIT.md`
**Current runtime disposition:** **NO PRODUCTION TRADING**

## 1. Objective

Close the research-pipeline gaps that prevented a valid production model from being generated.

The implementation must change the recovery/retrain question from:

```text
Can the model predict whether close[t+horizon] is up or down?
```

to:

```text
Can the model select trades that make money after costs under an explicit trade lifecycle?
```

Production trading remains disabled until a candidate passes offline economic gates and then paper quarantine.

## 2. Non-Negotiable Constraints

- Do not resume live trading.
- Do not weaken accuracy, expectancy, drawdown, benchmark, or paper-soak gates to force a pass.
- Do not promote a model from this work unless a later audit phase explicitly opens paper quarantine.
- Do not remove the existing directional-return label path; keep it as a legacy/diagnostic mode.
- Do not delete old models or registry records as part of this spec.
- Do not commit secrets, exchange keys, chat IDs, IPs, Tailscale hostnames, database dumps, or `.env` files.
- Preserve deterministic behavior where tests currently depend on deterministic output.
- Keep edits scoped to the files named in each phase unless the implementing agent documents why an extra file is required.

## 3. Required Sub-Agent Contract

Each implementation sub-agent must receive only one phase at a time and must return this exact end note:

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

The lead/audit agent must then write:

```text
LEAD AUDIT DECISION: accept | reject
ACCEPTANCE EVIDENCE:
REJECTION REASON:
CORRECTION INSTRUCTIONS:
AUDIT_REPORT_UPDATE:
```

No phase is accepted until tests pass and `audit_report.md` has a dated checkpoint.

## 4. Preflight

Run before Phase 1:

```powershell
git status --short
git rev-parse HEAD
rg -n "def _build_labels|def _select_threshold_from_oof_predictions|def _evaluate_candidate|def run_model_recovery_experiments" quant_v2\research\model_recovery_experiments.py quant_v2\research\scheduled_retrain.py
rg -n "ReplayActorConfig|ExecutionCostPolicy|funding|slippage|spread" quant_v2\research\portfolio_replay.py quant_v2\execution\cost_policy.py
```

Run on 4arm only if deploying or validating host parity:

```bash
cd /home/admin-4arm/hypothesis-research-engine
git status --short
docker ps --format '{{.Names}} {{.Image}} {{.Status}}'
docker logs --tail 200 quant_telegram | grep -Ei 'No production model|no_active_model|hard_risk|paper|live' || true
```

Stop if production trading is active or a model was promoted unexpectedly.

## 5. Phase 1 - Trade-Outcome Label Engine

### Goal

Add a reusable label engine that labels candidate trades by cost-adjusted lifecycle outcomes instead of only directional future returns.

### Files May Edit

- Add `quant_v2/research/trade_outcome_labels.py`
- Add `tests/quant_v2/test_trade_outcome_labels.py`
- May edit `quant_v2/research/__init__.py` only to export stable helpers if local package style requires it.

### Files Must Not Edit

- `quant_v2/research/model_recovery_experiments.py`
- `quant_v2/research/scheduled_retrain.py`
- registry, execution, and Telegram runtime files

### Required Public API

Implement these dataclasses and functions exactly:

```python
@dataclass(frozen=True)
class TradeOutcomeLabelConfig:
    horizon_bars: int = 4
    profit_target_bps: float = 20.0
    stop_loss_bps: float = 30.0
    dead_zone_bps: float = 5.0
    round_trip_cost_bps: float = 8.0
    funding_bps_per_8h: float = 0.0
    prefer_stop_on_same_bar: bool = True
    label_mode: str = "binary"

@dataclass(frozen=True)
class TradeOutcomeRecord:
    timestamp: str
    symbol: str
    side: str
    label: float
    gross_return_bps: float
    net_return_bps: float
    first_barrier: str
    holding_bars: int
    max_adverse_excursion_bps: float
    max_favorable_excursion_bps: float
    exit_price: float
    exit_timestamp: str | None

def build_trade_outcome_labels(
    frame: pd.DataFrame,
    *,
    config: TradeOutcomeLabelConfig,
    side: Literal["long", "short"] | str = "long",
) -> pd.Series:
    ...

def build_trade_outcome_report(
    frame: pd.DataFrame,
    *,
    config: TradeOutcomeLabelConfig,
) -> dict[str, Any]:
    ...
```

### Input Contract

`frame` must be a pandas DataFrame with:

- MultiIndex named exactly `["timestamp", "symbol"]`
- UTC timezone-aware timestamp level
- required columns: `open`, `high`, `low`, `close`
- optional columns: `funding_rate_bps`, `funding_bps`, `spread_bps`

If required conditions fail, raise `ValueError` with a message naming the violated condition.

### Label Algorithm

For each `(timestamp, symbol)` row and each side:

1. Entry price is current row `close`.
2. Look forward up to `horizon_bars` rows for the same symbol.
3. For long:
   - favorable return uses `future_high / entry_close - 1`
   - adverse return uses `future_low / entry_close - 1`
   - terminal return uses `future_close / entry_close - 1`
4. For short:
   - favorable return uses `entry_close / future_low - 1`
   - adverse return uses `entry_close / future_high - 1`
   - terminal return uses `entry_close / future_close - 1`
5. Convert returns to basis points.
6. Subtract total estimated costs from gross return:
   - `round_trip_cost_bps`
   - plus funding cost prorated by holding bars:
     `abs(funding_bps_per_8h) * holding_bars / 8`
   - plus optional row-level spread/funding columns if present
7. Determine first barrier:
   - `take_profit` if favorable excursion reaches `profit_target_bps`
   - `stop_loss` if adverse excursion reaches `stop_loss_bps`
   - `same_bar_stop_loss` if both are touched in the same future bar and `prefer_stop_on_same_bar=True`
   - `same_bar_take_profit` if both touched and `prefer_stop_on_same_bar=False`
   - `time_exit` if neither barrier is touched by `horizon_bars`
   - `insufficient_lookahead` if not enough future bars exist
8. Label:
   - `1.0` if net outcome is greater than `dead_zone_bps`
   - `0.0` if net outcome is less than `-dead_zone_bps`
   - `NaN` otherwise
   - `NaN` for `insufficient_lookahead`

For binary training, `1.0` means "trade is worth taking for this side"; `0.0` means "trade is not worth taking." Do not encode short as `0` and long as `1`.

### Required Report Fields

`build_trade_outcome_report` must return:

```python
{
    "policy_version": "trade_outcome_labels_v1",
    "config": {...},
    "dataset": {
        "rows": int,
        "symbols": list[str],
        "start": str | None,
        "end": str | None,
    },
    "long": {
        "label_counts": {"take": int, "skip": int, "ambiguous": int},
        "barrier_counts": {...},
        "net_return_bps": {"mean": float, "median": float, "p25": float, "p75": float},
        "holding_bars": {"mean": float, "median": float},
        "mae_bps": {"mean": float, "p95": float},
        "mfe_bps": {"mean": float, "p95": float},
        "by_symbol": {...},
    },
    "short": {same shape as long},
}
```

### Required Tests

Add tests for:

- long take-profit label
- long stop-loss label
- short take-profit label
- short stop-loss label
- same-bar conflict resolves to stop loss by default
- same-bar conflict can prefer take profit when configured
- time-exit profitable and unprofitable labels
- insufficient lookahead returns `NaN`
- MultiIndex validation errors
- report contains required by-symbol and barrier fields
- funding/cost increase can flip a positive gross trade into skip/ambiguous

### Acceptance Command

```powershell
python -m pytest tests/quant_v2/test_trade_outcome_labels.py -q
```

## 6. Phase 2 - Label Mode Integration In Recovery Experiments

### Goal

Wire trade-outcome labels into model recovery without deleting legacy directional labels.

### Files May Edit

- `quant_v2/research/model_recovery_experiments.py`
- `scripts/model_quality_recovery.py`
- `scripts/run_model_recovery_experiments.py`
- `tests/quant_v2/test_model_recovery_experiments.py`

### Required Changes

1. Extend `RecoveryCandidateConfig` with:

```python
label_mode: str = "directional_return"
trade_outcome_profit_target_bps: float = 20.0
trade_outcome_stop_loss_bps: float = 30.0
trade_outcome_round_trip_cost_bps: float = 8.0
```

2. Add helper:

```python
def _build_candidate_labels(
    frame: pd.DataFrame,
    config: RecoveryCandidateConfig,
) -> tuple[pd.Series, dict[str, Any]]:
    ...
```

Behavior:

- If `config.label_mode == "directional_return"`, call existing `_build_labels` and return existing label audit shape.
- If `config.label_mode == "trade_outcome"`, call `build_trade_outcome_labels(..., side="long")` for Phase 2. Do not train short-specific or multi-output models yet.
- Return a label report containing `label_mode`, `policy_version`, and the underlying label diagnostics.
- Raise `ValueError("unsupported label_mode=...")` for unknown modes.

3. Replace direct calls to `_build_labels` inside `_evaluate_candidate` with `_build_candidate_labels`.

4. Preserve existing label audit docs for directional mode.

5. Add CLI args to both recovery scripts:

```text
--label-mode directional_return|trade_outcome
--trade-profit-target-bps FLOAT
--trade-stop-loss-bps FLOAT
--trade-round-trip-cost-bps FLOAT
```

6. Default behavior:

- Existing tests that do not specify `label_mode` must continue using `directional_return`.
- New recovery recommendations should document that production recovery should use `--label-mode trade_outcome`.

### Required Tests

Add or update tests to prove:

- default config remains `directional_return`
- `_build_candidate_labels` returns legacy labels for directional mode
- `_build_candidate_labels` returns trade-outcome labels and report for trade-outcome mode
- unsupported label mode raises
- candidate manifest includes `label_mode`
- CLI parser accepts the new label args

### Acceptance Commands

```powershell
python -m pytest tests/quant_v2/test_trade_outcome_labels.py tests/quant_v2/test_model_recovery_experiments.py -q
```

## 7. Phase 3 - Economic Threshold Selection

### Goal

Stop selecting thresholds by classification accuracy alone. Select by economic utility while retaining accuracy as a diagnostic.

### Files May Edit

- Add `quant_v2/research/economic_thresholds.py`
- Add `tests/quant_v2/test_economic_thresholds.py`
- Edit `quant_v2/research/model_recovery_experiments.py`
- Edit `quant_v2/research/scheduled_retrain.py` only after recovery integration passes

### Required Public API

```python
@dataclass(frozen=True)
class EconomicThresholdConfig:
    threshold_min: float = 0.50
    threshold_max: float = 0.80
    threshold_step: float = 0.05
    min_actionable: int = 20
    round_trip_cost_bps: float = 8.0
    drawdown_penalty_weight: float = 1.0
    turnover_penalty_bps: float = 0.0
    concentration_penalty_weight: float = 25.0
    instability_penalty_weight: float = 10.0

def select_threshold_by_utility(
    probabilities: np.ndarray,
    labels: np.ndarray,
    forward_return_bps: np.ndarray,
    *,
    symbols: Sequence[str] | None = None,
    fold_ids: Sequence[str] | None = None,
    config: EconomicThresholdConfig | None = None,
) -> dict[str, Any]:
    ...
```

### Utility Algorithm

For each threshold in `[threshold_min, threshold_max]`:

1. `action_mask = probabilities >= threshold`
2. If actionable count is below `min_actionable`, mark threshold as ineligible.
3. For acted rows:
   - if label is `1`, realized utility return is `forward_return_bps - round_trip_cost_bps`
   - if label is `0`, realized utility return is `-abs(forward_return_bps) - round_trip_cost_bps`
   - For Phase 3, this simple convention is accepted because trade-outcome labels already encode good/bad outcomes.
4. Compute:
   - mean net expectancy bps
   - cumulative return curve over acted rows
   - max drawdown bps
   - action rate
   - symbol concentration share
   - fold expectancy standard deviation
5. Score:

```text
score =
  mean_net_expectancy_bps
  - abs(max_drawdown_bps) * drawdown_penalty_weight / 100
  - turnover_penalty_bps * action_rate
  - max(0, concentration_share - 0.50) * concentration_penalty_weight
  - fold_expectancy_std * instability_penalty_weight / 100
```

6. Select the eligible threshold with highest score. Tie-breakers:
   - higher mean net expectancy
   - lower drawdown
   - lower threshold

Return a dict containing:

```python
{
    "source": "economic_utility",
    "selected_threshold": float,
    "selected_score": float,
    "selected_expectancy_bps": float,
    "selected_actionable": int,
    "accuracy_at_selected": float,
    "accuracy_optimal_threshold": float,
    "accuracy_optimal_accuracy": float,
    "thresholds": [per-threshold diagnostics],
    "config": {...},
}
```

### Integration Requirements

- In recovery experiments, replace `selected_threshold_policy = _select_threshold_from_oof_predictions(...)` with the economic selector when enough forward-return evidence is available.
- Preserve old accuracy selector as fallback with source `accuracy_fallback`.
- Candidate reports must include both policies:
  - `threshold_policy`
  - `accuracy_threshold_policy`
- Candidate manifest must include selected threshold source.

### Required Tests

- economic selector prefers a higher-utility threshold over higher-accuracy threshold
- insufficient actionable rows marks threshold ineligible
- symbol concentration penalty changes selected threshold when one symbol dominates
- fold instability penalty is included when fold IDs are provided
- fallback to accuracy selector still works when forward returns are unavailable

### Acceptance Commands

```powershell
python -m pytest tests/quant_v2/test_economic_thresholds.py tests/quant_v2/test_model_recovery_experiments.py tests/quant_v2/test_scheduled_retrain_candidates.py -q
```

## 8. Phase 4 - Selection-Risk Report

### Goal

Prevent a lucky candidate from passing only because a broad sweep tried many combinations.

### Files May Edit

- Add `quant_v2/research/selection_risk.py`
- Add `tests/quant_v2/test_selection_risk.py`
- Edit `quant_v2/research/model_recovery_experiments.py`
- Edit `COMPARATIVE_QUANT_RESEARCH_AUDIT.md` only if documenting implementation completion

### Required Public API

```python
def build_selection_risk_report(
    candidates: Sequence[Mapping[str, Any]],
    *,
    selected_candidate_id: str | None,
    trial_count: int,
) -> dict[str, Any]:
    ...
```

Required output:

```python
{
    "policy_version": "selection_risk_v1",
    "trial_count": int,
    "candidate_count": int,
    "selected_candidate_id": str | None,
    "selected_rank_by_score": int | None,
    "selected_rank_by_holdout": int | None,
    "pbo_proxy": float,
    "fold_instability": float,
    "overfit_risk": "low|medium|high",
    "blockers": list[str],
}
```

Minimum blocker rules:

- `trial_count > 50` and selected holdout rank is worse than top 25 percent: blocker `selected_holdout_rank_weak_after_broad_search`
- fold accuracy or fold expectancy standard deviation above existing candidate gate: blocker `fold_instability`
- selected candidate has no fold ledger: blocker `missing_fold_ledger`

### Integration Requirements

- Recovery summary must include `selection_risk_report`.
- A candidate cannot pass if selection-risk report has blockers.
- Existing no-pass behavior must remain no-pass.

### Acceptance Commands

```powershell
python -m pytest tests/quant_v2/test_selection_risk.py tests/quant_v2/test_model_recovery_experiments.py -q
```

## 9. Phase 5 - Expanded Baselines

### Goal

Require ML candidates to beat a richer set of simple alternatives after realistic costs.

### Files May Edit

- `quant_v2/research/model_recovery_experiments.py`
- `quant_v2/research/portfolio_replay.py` only if a reusable baseline signal belongs there
- `tests/quant_v2/test_model_recovery_experiments.py`
- `tests/quant_v2/test_portfolio_replay.py`

### Required New Baseline Actors

Add benchmark actor configs for:

- `long_only`
- `short_only`
- `moving_average_trend`
- `volatility_breakout`
- `adverse_excursion_exit`
- `funding_aware_abstain`

If a baseline cannot be fully implemented because required data is absent, implement a deterministic no-trade fallback and mark `available=false` in the benchmark report. Do not silently omit it.

### Required Report Additions

Benchmark report must include:

```python
{
    "actor_summaries": {...},
    "availability": {
        "long_only": {"available": bool, "reason": str},
        ...
    },
    "by_symbol_best_actor": {...},
    "by_regime_best_actor": {...},
}
```

### Gate Change

Candidate must beat:

- flat
- best available non-flat baseline
- best available same-regime baseline when regime report is available

### Acceptance Commands

```powershell
python -m pytest tests/quant_v2/test_portfolio_replay.py tests/quant_v2/test_model_recovery_experiments.py -q
```

## 10. Phase 6 - AWS-to-4arm Lineage Forensics

### Goal

Determine whether profitability disappeared because of model/data/runtime drift rather than market decay.

### Files May Edit

- Add `scripts/model_lineage_forensics.py`
- Add `docs/runtime_reconciliation/aws_4arm_model_lineage.md`
- Add `tests/quant_v2/test_model_lineage_forensics.py` if script exposes pure functions

### Required Script Behavior

Script args:

```text
--current-registry-root PATH
--current-model-root PATH
--legacy-model-root PATH optional
--legacy-registry-root PATH optional
--snapshot-path PATH optional
--output docs/runtime_reconciliation/aws_4arm_model_lineage.md
```

The script must:

1. Inventory current registry versions.
2. Inventory legacy model roots if provided.
3. Identify active, previous-active, candidate, paper-quarantine, rejected, and archived versions.
4. Extract for each model:
   - version id
   - artifact path
   - created/promoted timestamps
   - horizons
   - feature catalog digest
   - model manifest digest
   - threshold policy
   - training window
   - symbols if available
5. Produce a markdown table.
6. If legacy artifacts are absent, explicitly write:

```text
Legacy AWS artifacts were not found at the provided paths; lineage reconstruction is incomplete.
```

Do not fabricate AWS findings.

### Acceptance Commands

```powershell
python scripts/model_lineage_forensics.py --current-registry-root models/production/registry --current-model-root models/production --output docs/runtime_reconciliation/aws_4arm_model_lineage.md
python -m pytest tests/quant_v2/test_model_lineage_forensics.py -q
```

If no production model root exists locally, the script must still produce an incomplete-but-valid report and exit 0.

## 11. Phase 7 - End-To-End Recovery Rerun

### Goal

Run the revised research pipeline without reopening production.

### Required Local Command

```powershell
python scripts/run_model_recovery_experiments.py `
  --snapshot-path datasets/v2/snapshots/model_recovery_real_1h_20260626_20260626_141210.parquet `
  --docs-output-dir docs/model_quality `
  --max-candidates 12 `
  --label-mode trade_outcome `
  --trade-profit-target-bps 20 `
  --trade-stop-loss-bps 30 `
  --trade-round-trip-cost-bps 8 `
  --no-production-registry
```

### Required 4arm Command

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

### Pass Criteria

The phase passes implementation if:

- command completes without crashing
- summary contains `label_mode=trade_outcome`
- threshold policy source is `economic_utility` or explicitly documented fallback
- selection-risk report exists
- benchmark report includes expanded baseline availability
- production registry is not mutated

The phase opens paper quarantine only if:

- at least one candidate passes all offline gates
- selected candidate beats flat and best available non-flat baseline after costs
- selection-risk report has no blockers
- model artifact is exported only to a research/staging location
- audit owner explicitly approves paper quarantine

## 12. Required Final Validation Suite

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

Run on 4arm after sync/deploy:

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

## 13. Expected Audit Outcome

At completion, update:

- `audit_report.md`
- `COMPARATIVE_QUANT_RESEARCH_AUDIT.md` if any recommendation changed
- generated model-quality docs under `docs/model_quality/`

The expected near-term outcome may still be `remain_no_trade`. That is acceptable. The goal of this spec is not to force a model to pass; it is to ensure any future passing model is aligned with trade profitability rather than directional prediction alone.
