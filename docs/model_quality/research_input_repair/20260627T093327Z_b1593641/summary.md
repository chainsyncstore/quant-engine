# Model Recovery Experiment Summary

- run_id: `20260627T093327Z_b1593641`
- recommendation: `remain_no_trade`
- selected_candidate_id: ``
- evaluated_candidates: `4`
- passed_candidates: `0`

## Benchmarks

| benchmark           | cost_adj_pnl_usd | max_dd  | fills |
| ------------------- | ---------------- | ------- | ----- |
| flat                | 0.00             | 0.0000  | 0     |
| mean_reversion      | -236.48          | -0.6277 | 1660  |
| momentum            | -463.64          | -0.7230 | 1680  |
| volatility_filtered | -441.73          | -0.7082 | 1501  |

## Top Candidates

| candidate_id                                       | horizon | window_m | half_life_d | dead_zone | feature_set               | holdout_acc | score | status |
| -------------------------------------------------- | ------- | -------- | ----------- | --------- | ------------------------- | ----------- | ----- | ------ |
| h2_tw3m_hl30d_dz5p2699_fsprice_volume_funding      | 2       | 3        | 30          | 5.2699    | price_volume_funding      | 0.0000      | -inf  | fail   |
| h2_tw3m_hl30d_dz5p2699_fsno_open_interest          | 2       | 3        | 30          | 5.2699    | no_open_interest          | 0.0000      | -inf  | fail   |
| h2_tw3m_hl30d_dz5p2699_fsfull                      | 2       | 3        | 30          | 5.2699    | full                      | 0.0000      | -inf  | fail   |
| h2_tw3m_hl30d_dz5p2699_fsno_orderbook_placeholders | 2       | 3        | 30          | 5.2699    | no_orderbook_placeholders | 0.0000      | -inf  | fail   |

## Next Operator Action

`remain_no_trade_and_repair_research_inputs`

## Warnings

- Use the best failed variant as the next repair target.
- Do not resume trading until a variant passes offline gates and paper soak.