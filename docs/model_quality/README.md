# Model Quality Recovery

This directory holds the recovery evidence for the post-lineage-reset retraining path.

Use `scripts/model_quality_recovery.py` to regenerate:

- failed retrain diagnostics
- label audits
- benchmark replay summaries
- candidate selection summaries
- the frozen validation policy note

The current recovery posture is intentionally conservative:

- no-trade remains the default until a candidate beats flat and a simple benchmark after costs
- retrain output is quarantined until the manifest and replay gates pass
- validation policy versions are attached to saved model metadata
