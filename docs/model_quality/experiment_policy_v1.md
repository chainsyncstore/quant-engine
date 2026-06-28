# Experiment Policy v1

This policy governs the model recovery experiment runner.

Rules:

1. Evaluate candidate configurations only under `models/experiments/model_recovery/<run_id>/`.
2. Never write to the active production pointer or production registry from this workflow.
3. Keep threshold selection on development folds only.
4. Use the final holdout once for reporting only.
5. Save a model artifact only when the candidate passes every quality gate.
6. If no candidate passes, the correct recommendation is `remain_no_trade`.
7. If a candidate passes, the correct recommendation is `paper_quarantine_candidate` only.
8. The final recommendation must not auto-enable paper or live trading.
9. Candidate validation may cap development folds and LightGBM estimator counts for experiment throughput, but must record those bounds in each candidate ledger and manifest.
