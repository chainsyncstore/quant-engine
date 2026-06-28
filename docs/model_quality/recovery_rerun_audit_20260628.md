# Trade-Outcome Recovery Rerun Audit - 2026-06-28

## Executive Decision

**Disposition:** production trading remains blocked.

The no-production-registry rerun completed successfully on 4arm, but it did not produce a paper-quarantine candidate.

```text
run_id: 20260628T051218Z_b1593641
snapshot: datasets/v2/snapshots/model_recovery_real_1h_20260626_20260626_141210.parquet
label_mode: trade_outcome
evaluated_candidates: 12
passed_candidates: 0
selected_candidate_id: null
recommendation: remain_no_trade
```

The rerun is useful evidence. It confirms the trade-outcome side-semantics repair is working, and it gives a cleaner explanation for why the production resume gate remains closed.

## Acceptance Evidence

The rerun evaluated both side-specific trade-outcome families:

- `_sidelong` candidates were evaluated.
- `_sideshort` candidates were evaluated.
- Long-side replays emitted only `BUY` fills.
- Short-side replays emitted only `SELL` fills.
- Every candidate holdout report includes `predicted_hold_count`.
- Every candidate threshold policy used `economic_utility`.
- Expanded benchmark actors were present, including `short_only`, `long_only`, `moving_average_trend`, `adverse_excursion_exit`, `volatility_breakout`, and `funding_aware_abstain`.
- The run summary includes a selection-risk report.

This means the previous invalid class-`0` inverse-side behavior is no longer the reason candidates fail or pass.

## Benchmark Context

The expanded benchmark replay was strongly dominated by `short_only`:

| actor | cost_adjusted_net_pnl_usd | max_drawdown_frac | fills |
|---|---:|---:|---:|
| short_only | 1929.28 | -0.5282 | 1580 |
| moving_average_trend | 313.24 | -0.3852 | 1535 |
| adverse_excursion_exit | 219.90 | -0.1246 | 604 |
| volatility_breakout | 25.89 | -0.1068 | 84 |
| flat | 0.00 | 0.0000 | 0 |
| mean_reversion | -236.48 | -0.6277 | 1660 |
| volatility_filtered | -441.73 | -0.7082 | 1501 |
| momentum | -463.64 | -0.7230 | 1680 |
| funding_aware_abstain | -456.12 | -0.7163 | 1676 |
| long_only | -1690.51 | -2.1547 | 2390 |

The market slice in this snapshot is not merely "hard for the model." It is specifically a regime where simple always-short exposure dominates the tested candidate family after costs. Any model that cannot beat that transparent baseline is not ready for paper quarantine.

## Candidate Failure Pattern

All 12 candidates failed `candidate_did_not_beat_nonflat_benchmark`.

Long-side candidates:

- All long candidates emitted only `BUY` fills, as intended.
- They generally met or exceeded the `0.60` accuracy gate.
- They had positive row-level expectancy in the holdout report.
- They still lost money in replay after execution dynamics.
- Several had too few actionable decisions.

Short-side candidates:

- All short candidates emitted only `SELL` fills, as intended.
- They generated positive replay PnL, but far below `short_only`.
- They failed the accuracy gate, with holdout accuracy around `0.567` to `0.569`.
- They failed `non_positive_cost_adjusted_expectancy`.
- They failed `candidate_did_not_beat_nonflat_benchmark`.

Top candidate by replay PnL:

| candidate | side | candidate_pnl_usd | accuracy | expectancy_bps | take_share | failures |
|---|---|---:|---:|---:|---:|---|
| h8_tw3m_hl30d_dz0p0010_fsfull_sideshort | short | 317.55 | 0.5667 | -2.7822 | 0.1366 | holdout_accuracy, expectancy, benchmark |
| h8_tw3m_hl30d_dz0p0010_fsno_orderbook_placeholders_sideshort | short | 317.55 | 0.5667 | -2.7822 | 0.1366 | holdout_accuracy, expectancy, benchmark |
| h8_tw3m_hl30d_dz0p0010_fsprice_volume_funding_sideshort | short | 300.28 | 0.5689 | -2.5875 | 0.1259 | holdout_accuracy, expectancy, benchmark |
| h8_tw3m_hl60d_dz0p0010_fsfull_sideshort | short | 115.82 | 0.5690 | -0.5609 | 0.0348 | holdout_accuracy, expectancy, benchmark |
| h8_tw3m_hl30d_dz0p0010_fsno_open_interest_sideshort | short | 107.77 | 0.5689 | -0.4778 | 0.0316 | holdout_accuracy, expectancy, benchmark |

Best long-side row-level expectancy:

| candidate | candidate_pnl_usd | accuracy | expectancy_bps | take_share | failures |
|---|---:|---:|---:|---:|---|
| h8_tw3m_hl60d_dz0p0010_fsfull_sidelong | -33.32 | 0.6333 | 1.1692 | 0.0133 | flat, benchmark, actionable |
| h4_tw3m_hl30d_dz0p0010_fsfull_sidelong | -49.74 | 0.6352 | 1.1141 | 0.0162 | flat, benchmark, actionable |
| h8_tw3m_hl30d_dz0p0010_fsprice_volume_funding_sidelong | -32.50 | 0.6331 | 0.9373 | 0.0135 | flat, benchmark, actionable |

## Interpretation

The repair changed the evidence quality. Before the repair, a trade-outcome `0` could accidentally become an inverse-side trade in parts of the evaluation path. After the repair, candidates only trade their configured side and skip otherwise.

That exposed the actual blocker:

1. The candidate family is not learning a superior short-selection policy.
2. The long-side model family is too sparse and loses money in replay despite positive row-level expectancy.
3. The short-side candidates capture some bearish regime value but underperform a simple transparent `short_only` baseline by a wide margin.
4. The candidate gate is behaving correctly by rejecting models that are weaker than a simple benchmark.

## Audit Decision

**Paper quarantine:** rejected.

The run does not satisfy the paper-quarantine gate because:

- `passed_candidates = 0`.
- No selected candidate exists.
- No candidate beat the best non-flat benchmark.
- Long candidates did not beat flat after costs in replay.
- Short candidates did not clear holdout accuracy or expectancy gates.

## Recommended Next Work

The next implementation should not loosen gates. The evidence points to model and objective quality work:

1. Add a benchmark-relative objective/reporting pass that measures candidate edge against `short_only`, `moving_average_trend`, and `adverse_excursion_exit` per regime and per symbol.
2. Add regime-conditional candidate families so the model does not compete with `short_only` globally when the entire snapshot is bearish; it should learn when to be short, when to abstain, and when transparent baselines already dominate.
3. Investigate the long-side replay gap: positive row-level expectancy but negative replay PnL suggests execution timing, fill clustering, symbol concentration, or threshold calibration is erasing the apparent edge.
4. Add candidate diagnostics by symbol for take rate, PnL, and benchmark delta so weak symbols can be excluded before replay.
5. Keep production and paper quarantine blocked until a candidate beats flat, `short_only`, `moving_average_trend`, `adverse_excursion_exit`, and the best non-flat benchmark after costs.

## Artifact Locations

```text
models/experiments/model_recovery/20260628T051218Z_b1593641/experiment_summary.json
models/experiments/model_recovery/20260628T051218Z_b1593641/benchmark_replay.json
models/experiments/model_recovery/20260628T051218Z_b1593641/candidates/
docs/model_quality/latest_experiment_summary.json
docs/model_quality/latest_recovery_sweep_summary.json
```
