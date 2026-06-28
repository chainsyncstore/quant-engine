# Model Candidate Lifecycle

Scheduled retrain no longer activates a model by default. It trains, validates, writes artifacts, and registers a candidate in the model registry.

## Candidate Registration

Default behavior:

```bash
BOT_RETRAIN_AUTO_PROMOTE=0
python -m quant_v2.research.scheduled_retrain
```

The retrain job writes a registry record under:

```text
models/production/registry/versions/<version_id>.json
```

Candidate metadata includes:

- `status`
- `validation_scores`
- `symbols_requested`
- `symbols_fetched`
- `symbols_failed`
- `train_rows`
- `horizons_trained`
- `created_at`
- `promotion_eligible`
- `promotion_eligibility_notes`

`registry/active.json` is not updated unless `BOT_RETRAIN_AUTO_PROMOTE=1` is explicitly set.

## Inspect Candidates

Telegram admin commands:

```text
/model_active
/model_versions
/model_candidates
/model_events
```

`/model_candidates` lists candidate and paper-quarantine versions with key metrics.
`/model_events` shows the append-only registry event history and current active
pointer.
`/model_active` now also shows the active artifact's manifest image reference,
feature-schema digest, and dataset digest.

## Paper Quarantine

Mark a candidate for paper-only evaluation:

```text
/model_quarantine <version_id>
```

This records `status=paper_quarantine` in the registry. Run paper-only evaluation against that artifact without activating it for live users. Full automated quarantine scoring is intentionally left as follow-up; the registry metadata now supports recording the quarantine state.

## Terminal Candidate States

Reject a candidate:

```text
/model_reject <version_id> [reason...]
```

Expire a candidate:

```text
/model_expire <version_id> [reason...]
```

These commands mark the registry record as terminal and prevent the version
from being promoted or reactivated.

## Manual Promotion

Stop active sessions first, then promote by exact version id:

```text
/stop
/model_promote <version_id> [evidence_digest] [reason...]
```

Promotion checks:

- version id exists,
- artifact directory exists,
- artifact has `config.json` or `model_*m.pkl`,
- registry metadata does not mark `promotion_eligible=false`.
- if the two-person gate is enabled, the same evidence digest must have two
  distinct unexpired approvals in the live scope.

The promotion records an append-only registry event, derives the active pointer
from event history, marks the promoted record `status=active`, records
`promoted_at`, `promoted_by`, evidence digest, and promotion notes, then
reloads the runtime signal source. If runtime loading fails, the bot attempts
to restore the prior active pointer.

## Emergency Auto-Promote

Legacy auto-promotion is available only by explicit opt-in:

```bash
BOT_RETRAIN_AUTO_PROMOTE=1
```

Leave this unset or set to `0` for normal production operation.

## Rollback

Rollback to the previous active version:

```text
/model_rollback
```

Rollback to a specific registered version:

```text
/model_rollback <version_id>
```

Rollback uses the registry pointer and does not require the target version to be a current candidate.
