# WP-01 Source Reconciliation

**Collection date:** 2026-06-22

**Production access:** read-only over the configured `4arm-ubuntu` SSH alias

**Status:** evidence baseline complete; WP-00 repository controls accepted; canonical recovery and production operational closure remain open

## Result

WP-01 now retains enough raw evidence to reproduce the source-provenance conclusions that were previously supported only by aggregate counts. The evidence binds the Ubuntu host, all running application containers, and a filtered source snapshot to complete identities without collecting environment contents, credentials, logs, databases, user state, model artifacts, or datasets.

The collection proves:

- Ubuntu Git HEAD is `6234aff58092458683125c8abcba333bbda99388` on branch `main`.
- The working tree had 251 porcelain status entries at collection time.
- `quant_telegram`, `quant_model_eval`, and `quant_retrain` all used image ID `sha256:e85886744eaf85cb275c6cd1bd344b56fc1609482152743bb1d616ecbb0c7d58`.
- The inspected image reports the same repository digest and was created at `2026-06-12T18:38:15.67783002Z` for Linux/amd64.
- Every retained critical runtime file has the same SHA-256 digest on the Ubuntu host and all three application containers.
- The normalized source reconciliation covers 141 unique files: 121 verified matches, 14 host-only deployment/build files, two files superseded by accepted WP-00 repository controls, and four untracked production-only files excluded from canonical recovery.

This evidence establishes what ran. It does **not** make the dirty production tree canonical.

## Evidence Layout

| Artifact | Purpose |
|---|---|
| `ubuntu_audit_20260622/source_provenance/production_metadata.txt` | Full Git identity, status filenames, diff stats, whitelisted Docker identity/labels, and critical hashes |
| `ubuntu_audit_20260622/source_provenance/host_source.tar.gz` | Deterministic filtered Ubuntu host source snapshot |
| `ubuntu_audit_20260622/source_provenance/container_source.tar.gz` | Deterministic filtered `/app` source snapshot from `quant_telegram` |
| `ubuntu_audit_20260622/source_provenance/*_archive_manifest.json` | Per-file raw SHA-256 and sizes for each snapshot |
| `docs/wp01/identities.json` | Parsed Git, container, image, and hash-binding identities |
| `docs/wp01/source_reconciliation.json` | Local HEAD/worktree/host/container raw and normalized hashes for every retained source path |
| `docs/wp01/critical_hashes.json` | Host and per-container critical source hashes |
| `docs/wp01/evidence_inventory.json` | SHA-256 inventory of the retained WP-01 evidence directory |
| `docs/wp01/dispositions.json` | Reviewed keep/replace/discard/unresolved decisions with finding and test traceability |
| `docs/wp01/canonical_exclusions.txt` | Recovery exclusion policy |

`tools/collect_wp01_source_provenance.py` performs the production collection. `tools/verify_wp01.py` is offline and deterministically regenerates the machine-readable documents from retained evidence.

## Collection Boundary

The source snapshots include `quant/`, `quant_v2/`, the Dockerfile, Compose YAML, `pyproject.toml`, `bootstrap_registry.py`, dependency input/lock files present on Ubuntu, and deployment shell/service/YAML files. They reject path traversal, duplicate members, symlinks, device entries, and non-regular files.

The collector excludes environment files, credentials, keys, databases, state, logs, models and registry artifacts, datasets, experiments, archives, caches, bytecode, and generated files. Docker inspection is field-whitelisted; container environment variables are never requested.

Production commands were limited to read-only forms of:

- `git rev-parse`, `git branch`, `git status --porcelain`, and `git diff --stat`;
- `docker inspect` and `docker image inspect` with explicit non-environment fields;
- `sha256sum` over the defined source allowlist;
- `find` and streaming `tar` over exact source roots;
- `docker exec` running only `find`, `sha256sum`, and streaming `tar` inside application containers.

No remote temporary file was created. No process, container, file, permission, package, credential, model, registry, database, session, pause, or deployment state was changed.

## Reconciliation Decisions

### Keep

The 121 files classified `keep_verified_runtime_behavior` match the current local worktree, Ubuntu host, and image after deterministic line-ending normalization. Host-only Dockerfile, `docker-compose.prod.yml`, and six deployment files also match the local worktree and are retained as build/deployment source rather than runtime filesystem content.

### Replace with accepted WP-00 repository controls

- `quant/telebot/main.py`: replace the credential-leaking runtime version with the accepted transport-suppression and final-output-redaction implementation.
- `quant_v2/research/model_evaluator.py`: replace the runtime version with the accepted deployment-policy cap on persistent/direct auto-promotion controls.
- `docker-compose.yml`: replace the audited Redis host publication with the accepted authenticated, non-published Redis configuration.
- `docker-compose.override.yml`: apply the accepted evaluator auto-promotion deployment-policy enforcement.

The WP-00 repository controls were accepted after 642 passing tests. They have not been deployed to production; operational closure, credential rotation, and Compose runtime validation remain open.

### Discard from canonical recovery

- `quant/telebot/redaction.py` is an untracked, unused production-only helper superseded by the pending containment implementation.
- `quant_v2/models/confirmation.py` and `quant_v2/models/confirmation_trainer.py` are untracked experimental modules with no clean commit or retained test binding.
- `quant_v2/research/confirmation_shadow_export.py` directly repeats the audited `concat(ignore_index=True)`, blanket zero imputation, and naive 80/20 split. It must not be imported into the recovery baseline.

The production copies remain in immutable evidence archives; discard means do not port them into canonical source.

### Unresolved

Ubuntu contains four host-only `requirements*.in`/`requirements*.lock` files that are absent from the local checkout and runtime source snapshot. They remain evidence only. WP-02 must create a reviewed Python 3.11 lock and bind its digest to the tested image rather than adopting these files by inference.

## Line Endings

`.gitattributes` now pins source and documentation formats to LF while marking binary evidence and model formats binary. Reconciliation records both raw and LF-normalized hashes, so line-ending-only changes cannot masquerade as behavioral drift.

## Remaining Gates

WP-01 cannot yet claim its plan-level exit gate because:

1. The shared working tree is intentionally dirty with accepted WP-00 and forensic work; no clean recovery branch has been created.
2. WP-00 production operational closure remains open, including deployment, credential rotation, and Compose/Redis runtime validation.
3. The full suite must pass against the eventual canonical recovery commit and, in WP-02, the exact image.
4. A second reviewer has not independently regenerated and signed the baseline manifest.
5. Quarantine tagging of the retained incident image is a production mutation and was prohibited for this read-only acquisition.

Until those gates close, the correct status is **evidence-complete, canonicalization pending**.
