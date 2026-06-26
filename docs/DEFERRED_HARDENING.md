# Deferred Hardening

This document records hardening work that is intentionally not part of the P0 live-deployment gate. Deferred work must not weaken the production-grade invariants in `PRODUCTION_REFACTOR_ROADMAP.md`.

## Not Deferred

These are never deferrable for live enablement:

- Live execution disabled by default.
- Explicit live go/no-go and rollback-clear checks.
- No normal post-only entry fallback to market orders.
- Missing or stale marks for open live positions fail closed.
- Truthful cancel, flatten, and shutdown reporting.
- Durable WAL/idempotency for live order intent.
- Redis command authentication and private Redis exposure.
- Credential redaction and release artifact scanning.
- Trusted model manifests, checksums, complete horizons, and fail-closed runtime selection.
- Explicit production compose topology and hardened runtime mounts.
- Reproducible locked dependencies and digest-pinned base images.
- Production readiness evidence, including Docker build, compose config, dependency audit, SBOM, and scanned release artifact.

## Deferred Items

| Item | Rationale | Guardrail |
| --- | --- | --- |
| Redis TLS for same-host Docker network | Redis is private to the Docker network and command messages are authenticated; TLS can be added when cross-host Redis is introduced. | Do not expose Redis on a host or public interface without TLS/auth review. |
| External artifact signing service | Current model trust requires rooted paths, manifests, checksums, and load validation. A signing service improves provenance but is not required to prevent unvalidated activation. | Do not activate artifacts without manifest/checksum/load validation. |
| Docker secrets migration | Current runbooks keep secrets outside the repo in root-owned host env files. Docker secrets or cloud secret managers are preferred for later platform maturity. | Do not commit, archive, or upload env files; keep `/etc/quant-bot/quant.env` locked down. |
| Full dependency modernization | Locked/audited dependencies are required now; broader modernization can happen after production readiness is proven. | `pip-audit` and hash-lock checks must stay clean. |
| Redis TLS for local diagnostics | Diagnostics should use `docker exec` or internal network access rather than host-published Redis. | Do not publish Redis by default. |

## Review Rule

Any new deferral must answer:

- Why it is not a P0 live-safety requirement.
- Which existing guardrail prevents the deferred risk from becoming live exposure.
- Which test, readiness check, or runbook would fail if the deferral became unsafe.
