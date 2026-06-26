# Validation Policy v1

The model-quality recovery policy remains strict until benchmark replay proves positive cost-adjusted expectancy.

Rules:

1. A fresh candidate must beat flat and at least one transparent benchmark after costs.
2. No-trade is the correct state when every benchmark loses after costs.
3. The validation policy version is `model_quality_validation_policy_v1`.
4. `RETRAIN_REQUIRE_ALL_HORIZONS=1` remains a hard requirement.
5. Promotion cannot rely on a final holdout or paper-soak result that was used to tune the threshold.
