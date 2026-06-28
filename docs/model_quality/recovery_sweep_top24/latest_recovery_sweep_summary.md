# Model Recovery Experiment Summary

- run_id: `20260627T073405Z_b1593641`
- recommendation: `remain_no_trade`
- selected_candidate_id: ``
- evaluated_candidates: `24`
- passed_candidates: `0`

## Benchmarks

| benchmark           | cost_adj_pnl_usd | max_dd  | fills |
| ------------------- | ---------------- | ------- | ----- |
| flat                | 0.00             | 0.0000  | 0     |
| mean_reversion      | -236.48          | -0.6277 | 1660  |
| momentum            | -463.64          | -0.7230 | 1680  |
| volatility_filtered | -441.73          | -0.7082 | 1501  |

## Top Candidates

| candidate_id                                  | horizon | window_m | half_life_d | dead_zone | feature_set          | holdout_acc | score    | status |
| --------------------------------------------- | ------- | -------- | ----------- | --------- | -------------------- | ----------- | -------- | ------ |
| h2_tw3m_hl30d_dz0p0010_fsfull                 | 2       | 3        | 30          | 0.0010    | full                 | 0.5356      | 540.05   | fail   |
| h4_tw6m_hl30d_dz0p0010_fsfull                 | 4       | 6        | 30          | 0.0010    | full                 | 0.5505      | 563.23   | fail   |
| h8_tw3m_hl30d_dz0p0010_fsfull                 | 8       | 3        | 30          | 0.0010    | full                 | 0.5621      | 11400.15 | fail   |
| h2_tw3m_hl30d_dz0p0010_fsprice_volume_funding | 2       | 3        | 30          | 0.0010    | price_volume_funding | 0.5356      | 540.05   | fail   |
| h2_tw3m_hl30d_dz0p0010_fsno_open_interest     | 2       | 3        | 30          | 0.0010    | no_open_interest     | 0.5356      | 540.05   | fail   |

## Next Operator Action

`remain_no_trade_and_collect_more_data`