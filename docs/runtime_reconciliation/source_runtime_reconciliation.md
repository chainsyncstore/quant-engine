# Source Runtime Reconciliation

**Date:** 2026-06-27
**Scope:** Local Windows working tree, `origin/main`, 4arm host checkout, and `quant_retrain` runtime source.

## Reconciliation Table

| artifact | git_sha | dirty_state | missing_files | extra_files | decision | reason |
|---|---|---|---|---|---|---|
| local Windows working tree | `65fc9e959335ccd6de819259a469ce97d6fdcbcd` | dirty | none in the reconciliation set | untracked recovery/spec artifacts and many pending implementation edits | keep as active implementation tree | local tree contains the in-flight recovery work, so it is not a release candidate yet |
| `origin/main` | `b15936417643fb5f1ddbf06483b9245e8bfc1100` | clean reference | none | none | canonical baseline | this is the clean remote baseline currently matching the 4arm host HEAD |
| 4arm host checkout | `b15936417643fb5f1ddbf06483b9245e8bfc1100` | dirty | none in the reconciliation set after syncing `scheduled_retrain.py` | modified `audit_report.md` and untracked recovery artifacts | acceptable staging checkout, not releaseable | host source now carries the scheduled retrain export fix and the recovery artifacts, but the tree is still dirty by design |
| `quant_retrain` runtime source (`/app`) | unavailable via git metadata | runtime source present but image metadata unavailable | `scripts/run_model_recovery_experiments.py` not present in `/app` | none observed from the limited runtime listing | runtime compile-validated, but not a full source mirror | the runtime container has the core modules needed for recovery tests, but the CLI wrapper is host-only at the moment |

## Notes

1. The scheduled retrain export drift was restored by re-exporting `fetch_symbol_dataset` from `quant_v2.research.scheduled_retrain`.
2. The runtime container is sufficient for module-level validation of the recovery runner, but the recovery CLI wrapper is not present in `/app`.
3. No trading state was modified as part of this reconciliation.
