# Model Recovery Experiment Summary

- run_id: `20260628T113942Z_b1593641`
- recommendation: `remain_no_trade`
- selected_candidate_id: ``
- evaluated_candidates: `4`
- passed_candidates: `0`

## Benchmarks

| benchmark              | cost_adj_pnl_usd | max_dd  | fills |
| ---------------------- | ---------------- | ------- | ----- |
| adverse_excursion_exit | 219.90           | -0.1246 | 604   |
| flat                   | 0.00             | 0.0000  | 0     |
| funding_aware_abstain  | -456.12          | -0.7163 | 1676  |
| long_only              | -1690.51         | -2.1547 | 2390  |
| mean_reversion         | -236.48          | -0.6277 | 1660  |
| momentum               | -463.64          | -0.7230 | 1680  |
| moving_average_trend   | 313.24           | -0.3852 | 1535  |
| short_only             | 1929.28          | -0.5282 | 1580  |
| volatility_breakout    | 25.89            | -0.1068 | 84    |
| volatility_filtered    | -441.73          | -0.7082 | 1501  |

## Candidate Quality

| metric               | value |
| -------------------- | ----- |
| evaluated_candidates | 4     |
| passed_quality       | 0     |
| watch_quality        | 0     |
| failed_quality       | 4     |

### Top Failure Reasons

- `candidate_pnl_usd<=0`: 4
- `candidate_minus_best_nonflat_pnl_usd<=threshold`: 4
- `candidate_minus_same_side_pnl_usd<=threshold`: 4
- `candidate_vs_benchmarks_margin_below_threshold`: 4
- `insufficient_allowed_symbols_or_take_count`: 4
- `regime_coverage_incomplete`: 4

## Top Candidates

| candidate_id                            | horizon | window_m | half_life_d | dead_zone | feature_set | holdout_acc | score  | status | quality_decision | quality_evidence_digest                                          | quality_hard_fails                                                                  | best_nonflat_bps | same_side_bps | allowed_symbols | rejected_symbols |
| --------------------------------------- | ------- | -------- | ----------- | --------- | ----------- | ----------- | ------ | ------ | ---------------- | ---------------------------------------------------------------- | ----------------------------------------------------------------------------------- | ---------------- | ------------- | --------------- | ---------------- |
| h2_tw3m_hl30d_dz0p0010_fsfull_sidelong  | 2       | 3        | 30          | 0.0010    | full        | 0.6396      | 639.77 | fail   | fail             | 2ceb55d128291eea3663fd9d502f570e9b46f0b2809e47643d6501269fa15a33 | positive_absolute_expectancy,best_nonflat_benchmark_delta,same_side_benchmark_delta | 0.00             | 0.00          | 0               | 10               |
| h4_tw3m_hl30d_dz0p0010_fsfull_sidelong  | 4       | 3        | 30          | 0.0010    | full        | 0.6352      | 636.29 | fail   | fail             | 58b76db9d67cce5559849c4b6e7a835818055ef4aedb1f37686eefac5f40a33f | positive_absolute_expectancy,best_nonflat_benchmark_delta,same_side_benchmark_delta | 0.00             | 0.00          | 0               | 10               |
| h8_tw3m_hl30d_dz0p0010_fsfull_sidelong  | 8       | 3        | 30          | 0.0010    | full        | 0.6344      | 635.26 | fail   | fail             | c031940b94181b2395dcbe9fdedf5c98b98108580a9fb870cc7e704f078ec132 | positive_absolute_expectancy,best_nonflat_benchmark_delta,same_side_benchmark_delta | 0.00             | 0.00          | 0               | 10               |
| h8_tw3m_hl30d_dz0p0010_fsfull_sideshort | 8       | 3        | 30          | 0.0010    | full        | 0.5667      | 563.94 | fail   | fail             | 7a114920a3fee5702a3cfc7e67eaa571c00b605666fdcf2ce01b1b16462b89be | positive_absolute_expectancy,best_nonflat_benchmark_delta,same_side_benchmark_delta | 0.00             | 0.00          | 0               | 10               |

## Next Operator Action

`remain_no_trade_and_collect_more_data`