# Comparative Quant Research Audit

**Date:** 2026-06-27
**System:** hypothesis-research-engine / 4arm production-resume recovery
**Disposition:** production trading should remain disabled until at least one candidate passes an economic, walk-forward, paper-soak gate.

## Executive Conclusion

The current failure to produce a valid model is not best explained as "we need one more retrain." The stronger interpretation is that the research definition of a good model is still misaligned with the live trading problem.

The repo has meaningful safeguards: temporal validation, recency weighting, LightGBM-based candidate training, cost-aware replay, flat/non-flat baselines, paper quarantine, runtime blockers, and fail-closed promotion controls. Those are good foundations.

The main gap is that the model is trained primarily on directional future-return labels, while production PnL depends on a full trade lifecycle: entry, holding time, stop/take-profit behavior, fees, spread, slippage, funding, drawdown, adverse excursion, symbol concentration, and exit policy. This mismatch can create a model that appears directionally reasonable but still holds losing trades and bleeds equity in live or paper execution.

The recommended next move is therefore a research-pipeline refactor, not another purge-and-retrain cycle:

1. Add trade-outcome labels, preferably triple-barrier style labels with cost/funding-aware profit target, stop loss, and max holding horizon.
2. Select thresholds and models by net expectancy / risk-adjusted utility, not classification accuracy alone.
3. Replace the current selection-risk proxy with real multiple-testing controls: purged walk-forward plus CPCV/PBO/deflated-Sharpe-style reporting.
4. Expand baselines and require the ML candidate to beat simple, regime-aware, per-symbol baselines after costs.
5. Run an AWS-era versus 4arm-era data/feature/model-forensics comparison before assuming the market edge vanished.
6. Keep no-trade mode until a candidate passes offline gates and then a paper quarantine window.

## External Reference Baseline

### Peer-Reviewed / Academic Practices

| Source | Relevant practice | Audit implication |
| --- | --- | --- |
| Gu, Kelly, Xiu, "Empirical Asset Pricing via Machine Learning" | ML can add value in asset pricing when nonlinearities and interactions are exploited and evaluated economically, not merely as raw classification accuracy. | Our LightGBM stack is directionally compatible, but candidate acceptance must emphasize economic value and robustness rather than a headline accuracy gate. |
| Bailey, Borwein, Lopez de Prado, Zhu, "The Probability of Backtest Overfitting" | Multiple trials and strategy selection can create false discoveries; strategy selection needs overfit controls. | Our broad sweeps need stronger selection-risk accounting than the current percentile-style proxy. |
| Lopez de Prado-style financial ML workflow | Purging, embargoing, meta-labeling, and triple-barrier labels reduce leakage and better represent trade outcomes. | Our temporal purging exists, but the labels are still directional future-return labels rather than trade-outcome labels. |
| DeepLOB | Deep order-book methods can model microstructure, but require order-book state and high-frequency event data. | This is not the next move unless we collect/order-book data; forcing DeepLOB-style architecture onto hourly OHLCV bars would be cargo-culting. |

### Open-Source Quant System Patterns

| Project | Relevant pattern | Audit implication |
| --- | --- | --- |
| Microsoft Qlib | Treats data, features, model training, backtesting, analysis, and workflow as one reproducible research platform. | We are moving in this direction, but still need stronger experiment lineage and comparable benchmark suites. |
| Freqtrade/FreqAI | Emphasizes retraining lifecycle, backtesting, dry/live separation, and model operational controls. | Our quarantine/promotion controls are aligned, but no valid candidate should be force-promoted through runtime controls. |
| FinRL | Useful reference for environment/reward framing, but RL needs careful validation and should not be the first recovery step. | We should borrow the "trading as an environment/reward" framing for labels and scoring before considering RL. |
| vectorbt/backtesting-style engines | Fast replay, vectorized sensitivity, and transparent portfolio accounting are central to strategy iteration. | Our replay work is useful, but research reports should expose more trade lifecycle diagnostics such as MAE/MFE/hold time and exit reasons. |

## Current Pipeline Evidence

### What The System Already Does Well

- **Fail-closed behavior:** The latest 4arm repair run completed with `recommendation=remain_no_trade`, `passed_candidates=0`, and `passed_variants=0`. This is the right operational outcome when no candidate has proven edge.
- **Temporal validation exists:** `build_temporal_validation_plan` builds monthly folds, a final holdout, and purge windows. See `quant_v2/validation/temporal_validation.py`.
- **Candidate grid exists:** The recovery runner sweeps horizons `(2, 4, 8)`, dead zones, training windows, and recency half-lives. See `quant_v2/research/model_recovery_experiments.py`.
- **Cost-aware replay exists:** Candidates must beat flat and non-flat benchmarks after cost-adjusted replay, not just score on labels.
- **Paper quarantine exists:** `model_evaluator.decide_promotion` blocks promotion on insufficient window, insufficient decisions, weak edge, worse drawdown, concentration, hard-risk pauses, and active sessions.
- **Model bundles carry provenance:** `trainer.save_model_bundle` and the registry capture artifact metadata and promotion state.

These are real strengths. The problem is not absence of controls; it is incomplete alignment between the supervised label, the threshold selection objective, and the live trading objective.

## Severity-Ranked Findings

### P0 - Label Target Does Not Encode The Trading Lifecycle

**Evidence in repo**

- `quant_v2/research/scheduled_retrain.py` builds labels from `close.shift(-horizon) / close - 1.0`.
- `quant_v2/research/model_recovery_experiments.py` uses the same future-return directional labeling.
- Labels become binary up/down when forward return exceeds a dead zone; ambiguous rows are dropped.

**Why this matters**

The label asks: "Was the close higher or lower after N bars by more than a dead zone?"

The bot needs to answer: "If I enter now, under the production planner, cost model, stop/take-profit/hold policy, risk constraints, and realistic exit behavior, is this trade worth taking?"

Those are different problems. A model can correctly predict a positive 4-hour forward return but still be a bad trade because of interim drawdown, fees, funding, spread, volatility expansion, position concentration, or exit timing.

**Likely live symptom**

This is consistent with the SOLUSD behavior the user observed: the bot can hold through repeated losing cycles if the signal/exit lifecycle does not explicitly encode adverse excursion, stale confidence, max hold decay, or trade invalidation.

**Recommended fix**

Create a trade-outcome labeler:

- Inputs: OHLCV bars, spread/slippage/funding assumptions, horizon, profit target, stop loss, max hold, side, symbol, timestamp.
- Outputs:
  - primary label: profitable trade outcome after costs;
  - meta label: whether to act on a directional signal;
  - side label: long/short/abstain if needed;
  - diagnostics: first barrier touched, holding bars, max adverse excursion, max favorable excursion, realized return after costs.

Acceptance should require that candidate training and recovery experiments can choose between legacy directional labels and trade-outcome labels, with trade-outcome labels becoming the production default.

### P0 - Threshold Selection Still Optimizes Accuracy Before Economics

**Evidence in repo**

- `_select_threshold_from_oof_predictions` in both scheduled retrain and recovery experiments searches thresholds by classification accuracy.
- Candidate rejection later checks economic metrics, but threshold choice itself is not selected by net expectancy or drawdown-aware utility.

**Why this matters**

The best classification threshold is often not the best trading threshold. In trading, abstaining is valuable. A 54 percent accurate model can be profitable if it trades only high-conviction, high-payoff cases, while a 60 percent accurate model can lose money if the payoff/cost profile is poor.

**Recommended fix**

Replace or supplement threshold selection with utility-based selection:

```text
threshold_score =
  net_expectancy_bps
  - drawdown_penalty
  - turnover_cost_penalty
  - concentration_penalty
  - instability_penalty
```

Candidate reports should still include accuracy, but the selected threshold should come from expected net return and risk constraints.

### P0 - Model Search Needs Stronger Multiple-Testing Controls

**Evidence in repo**

- The recovery grid can evaluate many combinations: horizons, dead zones, training windows, half-lives, feature sets, and variants.
- `_summarize_selection_risk` reports a percentile-style proxy rather than a true PBO/CPCV/deflated-Sharpe style estimate.

**Why this matters**

Broad sweeps are useful, but every extra trial increases the chance of selecting noise. The current gate refused all candidates, which protected us. But once a marginal pass appears, the system needs stronger proof that it is not just the lucky survivor of a large search.

**Recommended fix**

Add a selection-risk report with:

- number of candidate trials actually evaluated;
- fold-by-fold economic scores;
- selected candidate rank by train/dev versus holdout;
- PBO-style probability that selected strategy is overfit;
- deflated-Sharpe-style adjustment where returns are available;
- pass/fail blockers if the selected candidate is fragile or selected only after excessive trials.

### P1 - Baselines Exist But Are Too Narrow For Research Recovery

**Evidence in repo**

- Recovery includes flat, momentum, mean-reversion, and volatility-filtered benchmark actors.
- Candidate gates compare against flat and best non-flat benchmark.

**Why this matters**

This is good, but a recovery audit needs more baselines:

- per-symbol buy/hold or passive exposure;
- long-only/short-only variants;
- volatility breakout;
- moving-average trend;
- funding-aware carry/no-trade filters;
- regime-specific baselines;
- naive "close after N bars" and "reduce after adverse excursion" baselines.

If simple baselines cannot make money after costs on the current data, an ML model has no foundation. If simple baselines can make money, the ML model should be required to beat them per symbol/regime, not only aggregate.

**Recommended fix**

Expand the benchmark suite and include a "baseline viability report" before training ML candidates.

### P1 - AWS-to-4arm Performance Regression Has Not Been Fully Forensically Reconstructed

**Evidence in repo/session**

- The system became non-profitable after the AWS-to-Ubuntu move.
- Current audits repaired source/runtime parity and data connectivity, but the last-known-good AWS model path is not yet reconstructed.

**Why this matters**

If the AWS deployment had a real edge, we need to know whether it came from:

- a specific model artifact;
- a different data provider or bar construction;
- a different feature schema;
- a different threshold;
- a different symbol universe;
- a different execution path;
- a different cost assumption;
- an unintentional bug that happened to help backtests but not live trading.

Without this comparison, new research may optimize around the wrong failure mode.

**Recommended fix**

Run a forensic lineage task:

- locate the last AWS-era active model artifact and registry record if available;
- reconstruct its training window, features, thresholds, symbols, and data source;
- compare feature distributions AWS-era versus 4arm-era;
- compare prediction distribution and action rate;
- replay old model on current 4arm snapshot and current model on AWS-era data if available.

### P1 - Validation Is Temporal And Purged, But Not Yet Research-Grade CPCV

**Evidence in repo**

- `build_temporal_validation_plan` applies monthly folds, final holdout, and purge bars.
- This is materially better than random train/test splitting.

**Why this matters**

Financial labels overlap in time. A rolling monthly plan helps, but candidate search across multiple windows/features/thresholds still needs a stronger combinatorial or nested validation story when making production decisions.

**Recommended fix**

Keep the existing temporal plan, then add:

- nested threshold/model selection inside development folds;
- untouched final holdout;
- optional combinatorial purged cross-validation for final candidate families;
- separate "research selection" and "production acceptance" reports.

### P2 - Deep Learning / LOB Methods Are Not The Immediate Fix

**Evidence**

- The current system is hourly-bar, multi-symbol, feature-based.
- DeepLOB-style approaches depend on limit-order-book event structure.

**Recommendation**

Do not pivot to DeepLOB or reinforcement learning until:

- the label mismatch is fixed;
- simple baselines are understood;
- data lineage is stable;
- economic scoring is in place.

Deep learning can be revisited if we start collecting order-book snapshots, trades, and microstructure features with robust storage and replay.

## Comparison Matrix

| Practice | Research/open-source baseline | Current repo state | Gap | Severity |
| --- | --- | --- | --- | --- |
| Trade labels | Triple-barrier/meta-labeling or outcome labels where feasible | Directional future-return labels with dead zone | Does not encode stop/take-profit/max-hold/adverse excursion/funding | P0 |
| Validation | Purged/embargoed CV; preferably nested/CPCV for selection-heavy research | Temporal monthly folds with purge and holdout | Good start, but selection-risk controls are incomplete | P1 |
| Threshold selection | Utility/economic objective with abstention | Accuracy-selected OOF threshold | Accuracy can select poor trading threshold | P0 |
| Backtest realism | Costs, slippage, turnover, drawdown, benchmark comparisons | Cost-aware replay exists | Needs richer lifecycle diagnostics and sensitivity | P1 |
| Baselines | Multiple simple strategies by symbol/regime | Flat, momentum, mean-reversion, vol-filtered | Good start, too narrow for recovery | P1 |
| Model family | Start simple; tree models common for tabular bar features | LightGBM classifier stack | Reasonable; model family is not the first bottleneck | P2 |
| Lineage | Full data/model/config provenance | Much improved, still dirty multi-runtime state | Need AWS-era model reconstruction | P1 |
| Promotion | Forward/paper quarantine and operational blockers | Present and currently fail-closed | Keep; do not weaken | P0 guardrail |

## Recommended Build Plan

### Workstream A - Trade-Outcome Labeling

Deliverables:

- `quant_v2/research/trade_outcome_labels.py`
- tests for long/short barrier hits, no-hit max hold, cost-adjusted outcomes, missing bars, funding/slippage sensitivity
- recovery runner option: `--label-mode directional_return|trade_outcome`
- default production recovery mode: `trade_outcome`

Acceptance:

- reports include label balance, barrier-hit distribution, average hold bars, MAE/MFE, net return bps, and per-symbol label health.

### Workstream B - Economic Threshold Selection

Deliverables:

- replace accuracy-only threshold selection with `select_threshold_by_utility`
- utility must account for net expectancy, turnover/costs, drawdown, symbol concentration, action count, and fold stability
- preserve accuracy metrics as diagnostics, not the primary threshold objective

Acceptance:

- threshold report shows both accuracy-optimal and utility-optimal thresholds and explains which one was selected.

### Workstream C - Selection-Risk And Robustness Report

Deliverables:

- `selection_risk_report.json`
- PBO-style rank instability estimate
- deflated/probabilistic Sharpe proxy where replay returns exist
- trial count and candidate-family count
- pass/fail blockers for overfit selection

Acceptance:

- candidate cannot pass if selection-risk report flags high overfit probability or insufficient independent fold evidence.

### Workstream D - Baseline Expansion

Deliverables:

- baseline actors for long-only, short-only, buy/hold, moving-average trend, volatility breakout, adverse-excursion exit, funding-aware abstain
- per-symbol and per-regime benchmark tables
- "baseline viability" report before ML training

Acceptance:

- ML candidate must beat flat, best simple baseline, and relevant per-regime baseline after costs.

### Workstream E - AWS-to-4arm Forensics

Deliverables:

- `docs/runtime_reconciliation/aws_4arm_model_lineage.md`
- last-known-good model inventory, if available
- feature distribution drift report
- prediction/action-rate drift report
- old-versus-new replay matrix

Acceptance:

- either the last AWS-good lineage is reconstructed, or the report proves the required artifact/data no longer exists and documents the confidence limit.

## Recommended Immediate Next Step

Implement Workstream A and B first. They attack the central mismatch: the model currently learns directional future returns, but the business outcome is cost-adjusted trade profitability under a lifecycle policy.

After A and B, rerun the recovery sweep on the same 4arm snapshot. If no model passes, we will have higher confidence that the current data/feature universe lacks exploitable edge under realistic costs. If a model passes, it should enter paper quarantine, not live trading.

## Sources

- Gu, Shihao; Kelly, Bryan; Xiu, Dacheng. "Empirical Asset Pricing via Machine Learning." *Review of Financial Studies*, 2020. https://academic.oup.com/rfs/article/33/5/2223/5758276
- Bailey, David H.; Borwein, Jonathan; Lopez de Prado, Marcos; Zhu, Qiji Jim. "The Probability of Backtest Overfitting." https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2326253
- Zhang, Zihao; Zohren, Stefan; Roberts, Stephen. "DeepLOB: Deep Convolutional Neural Networks for Limit Order Books." https://arxiv.org/abs/1808.03668
- Microsoft Qlib. https://github.com/microsoft/qlib and https://qlib.readthedocs.io/
- Freqtrade FreqAI documentation. https://www.freqtrade.io/en/stable/freqai/
- AI4Finance FinRL. https://github.com/AI4Finance-Foundation/FinRL
- vectorbt documentation. https://vectorbt.dev/
