# Production Refactor Roadmap

Created: 2026-06-03

Source audit: `SECURITY_AUDIT_ROADMAP.md`

Focused child roadmap: `docs/NATIVE_TORCH_CONFIRMATION_ROADMAP.md`

## Objective

Refactor this repository into a production-grade trading runtime. The target is not cosmetic cleanup; it is a system that fails closed for live trading, protects credentials and model artifacts, survives restart/redeploy paths safely, and can prove those guarantees through tests and deployment checks.

Until the P0 gates below are complete, live deployment should be treated as blocked.

## Production-Grade Definition

A production-grade version of this repo must satisfy these invariants:

- Live trading is disabled by default in every runtime path and requires explicit, verified enablement.
- Normal entry orders never silently degrade into market/taker orders.
- Redis, WAL, idempotency, and shutdown paths cannot duplicate orders, lose safety state, or falsely report exposure as flat.
- Missing or stale market prices for open live positions become a risk condition, not an invisible zero-risk state.
- Credentials, tokens, DB state, PEMs, logs, and debug output are kept out of tracked files, release artifacts, and routine logs.
- Model artifacts are trusted, complete, validated, and loadable before activation; invalid registry state fails closed.
- Production Docker/compose configuration is explicit, least-privilege, reproducible, and testable.
- Every dangerous historical behavior has a regression test or deployment assertion that fails if it returns.

## Refactor Strategy

Recommended sequence:

1. **Seal live-order fail-open paths first.**
   - Fix live defaults, post-only fallback, missing-price exposure, cancel failure reporting, and shutdown sentinel behavior before any broader polish.

2. **Make command processing durable and authenticated.**
   - Redis/WAL/idempotency work must come before live recovery claims; otherwise restarts can still replay or lose dangerous side effects.

3. **Remove credential/artifact leakage paths.**
   - This includes code/log redaction, diagnostic script quarantine, tracked archive removal planning, and release artifact scanning.

4. **Lock model trust boundaries.**
   - Promotion, rollback, runtime selection, and unsafe deserialization need an explicit trust model before retrain automation is production-safe.

5. **Harden deployment/build last, but verify continuously.**
   - Compose topology, dependency locks, container restrictions, docs, and CI gates should reinforce the code-level fixes.

This order keeps the sharpest live-risk edges from being hidden under larger refactors.

## P0 Deployment-Blocking Gates

| Gate | Related audit IDs | Required outcome | Proof |
| --- | --- | --- | --- |
| P0-1 Live runtime fails closed | D3, D5, T2 | Standalone, Telegram bridge, and compose/prod paths all reject live sessions unless live enablement and go/no-go gates are explicitly true. | Unit tests plus `docker compose config` process/command assertion. |
| P0-2 No unintended market entry fallback | E1 | Post-only/entry limit failures reject rather than fallback to `MARKET`; emergency flatten paths remain explicit and reduce-only where possible. | Inverted Binance adapter tests and service routing tests. |
| P0-3 Open live positions require fresh marks | E2 | Missing/non-finite/stale prices for non-zero positions trigger pause/hard-risk handling and do not permit new entries. | Service/reconciler tests with open position and missing mark. |
| P0-4 Cancel/flatten truthfulness | E3, R4, D6 | Cancel failures are surfaced; shutdown only emits `positions_flat` after verified flat/canceled state; scripts target actual compose services. | Stale-feed, shutdown, and script smoke tests. |
| P0-5 Durable idempotency and WAL recovery | R1, R2, R3, R5 | Redelivered commands cannot duplicate live orders; live replay enters safe recovery; lifecycle/kill-switch state survives restart; ACK follows durable safety recording. | Crash-replay, live WAL replay, lifecycle replay, and WAL-failure tests. |
| P0-6 Redis command trust | D1, D10, R6 | Redis is not publicly exposed by default; commands are authenticated/signed before execution. | Compose/config tests, host firewall check, forged command rejection test. |
| P0-7 Credential leakage controls | D2, D4, T1, T3, D7, H4 | Raw Telegram updates, exception text, diagnostics, and release artifacts do not expose credentials or state; key rotation/removal plan is documented. | Log-capture tests, artifact scans, secret scans. |
| P0-8 Model activation fails closed | M1, M2, M3 | Runtime does not load untrusted/out-of-root artifacts; promotion requires explicit eligibility, complete horizons, and loadability; invalid active pointer does not fallback in production. | Model registry and signal-manager tests. |

## Implementation Pass Order

### Pass 1: Live-Order Safety Refactor

Scope:
- `quant_v2/execution/main.py`
- `quant_v2/execution/service.py`
- `quant_v2/execution/binance_adapter.py`
- `quant_v2/execution/reconciler.py`
- `quant/data/binance_client.py`
- `quant/telebot/main.py`
- `docker-compose*.yml`
- tests under `tests/quant_v2` and `tests/infra`

Targets:
- Make standalone execution parse `BOT_V2_ALLOW_LIVE_EXECUTION` with default false.
- Make compose/prod command selection explicit.
- Remove market fallback for normal post-only entry failures.
- Treat missing marks for open positions as fail-closed risk state.
- Make cancel failures visible.
- Fix shutdown flatten helper and `positions_flat` sentinel.

Acceptance:
- New or inverted tests fail on the current behavior and pass after refactor.
- No live runtime path starts live with unset live env.
- No normal post-only entry path can call `MARKET`.

### Pass 2: Redis, WAL, Idempotency, and Recovery

Scope:
- `quant_v2/execution/redis_bus.py`
- `quant_v2/execution/state_wal.py`
- `quant_v2/execution/idempotency.py`
- `quant_v2/execution/main.py`
- `quant_v2/execution/service.py`
- Redis bus/WAL tests

Targets:
- Persist idempotency/order intent before exchange placement.
- Include deterministic command/message IDs in idempotency keys.
- Add command envelope auth or HMAC validation.
- Rework ACK semantics for safety-critical persistence failures.
- Replay live sessions into explicit paused/flatten-only recovery unless secure credentials are available.
- Persist/replay lifecycle and kill-switch state.

Acceptance:
- Redelivered `route_signals` cannot place a second live order.
- Unsigned/tampered Redis commands are rejected before service mutation.
- WAL replay restores or safely pauses all safety-relevant state.

### Pass 3: Credential, Logging, and Artifact Hygiene

Scope:
- `quant/telebot/main.py`
- `quant/telebot/manager.py`
- `quant/telebot/engine.py`
- `debug_credentials.py`
- `test_capital_auth.py`
- local analyzer scripts
- `.gitignore`, `.dockerignore`, deploy docs
- release/scan scripts to be added

Targets:
- Remove or redact catch-all Telegram update logging.
- Sanitize exception replies/logs/crash notifications.
- Require approved user status for `/setup`.
- Remove hardcoded credentials and quarantine diagnostic scripts.
- Add artifact/secret scan tooling and release allowlist.
- Document key rotation and history-cleanup decision points.

Acceptance:
- Sentinel secrets never appear in logs, replies, or crash notifications.
- Pending/banned users cannot persist credentials.
- Release artifact scan rejects env, PEM, DB, archives, diagnostics, and logs unless explicitly allowlisted.

### Pass 4: Model Registry and Artifact Trust

Scope:
- `quant_v2/model_registry.py`
- `quant_v2/models/*`
- `quant_v2/telebot/signal_manager.py`
- `quant/telebot/model_selection.py`
- `quant_v2/research/scheduled_retrain.py`
- `bootstrap_registry.py`
- registry/model tests

Targets:
- Enforce trusted-root and symlink/out-of-root rejection.
- Require explicit `promotion_eligible is True`.
- Require manifest/checksum, complete horizon set, and loadability validation.
- Make active pointer updates transactional enough for crash/partial-write safety.
- Disable latest-filesystem fallback in production.
- Parse auto-promote flags fail-closed.

Acceptance:
- Placeholder/corrupt/out-of-root/symlink artifacts cannot be promoted or loaded.
- Invalid active registry state fails closed in production.
- Unknown auto-promote env values do not promote.

### Pass 5: Deployment, Build, and Runtime Hardening

Scope:
- `Dockerfile`
- `docker-compose*.yml`
- `pyproject.toml`
- dependency lock/constraints files
- `DEPLOY.md`, `AWS_DEPLOY.md`, deploy scripts
- `tests/infra/*`

Targets:
- Remove host Redis publish by default or bind to localhost only.
- Add Redis auth/ACL support where host access is required.
- Add compose hardening: `read_only`, `cap_drop`, `security_opt`, healthchecks, explicit writable mounts.
- Lock dependencies and pin base images.
- Replace dirty-tree upload docs with clean release artifact workflow.
- Align shutdown/monitoring script container names with compose topology.

Acceptance:
- `docker compose config` passes no-public-Redis and hardening assertions.
- Dependency installation is reproducible from locked inputs.
- Deployment docs no longer recommend `0.0.0.0/0` SSH or recursive local repo upload.

### Pass 6: CI, Regression, and Production Readiness Gate

Scope:
- `tests/**`
- new scripts under an agreed tooling path
- CI config if present or newly introduced
- production runbooks

Targets:
- Convert audit findings into automated tests/checks.
- Add secret/artifact scan to local and CI workflow.
- Add dependency audit/SBOM generation.
- Add production readiness checklist and rollback/disable procedure.
- Run focused test suites for each refactor area, then a broader suite.

Acceptance:
- All P0 gates have evidence.
- Unsafe historical tests are inverted.
- Production readiness checklist can be executed from a clean checkout.

## Delegated Refactor Notes Template

Sub-agents should end with:

```text
END NOTES
Scope reviewed:
- ...

Recommended implementation:
- [Priority] Title
  Files to touch: ...
  Code changes: concrete changes
  Tests/checks: concrete tests or commands
  Risks/tradeoffs: compatibility or migration notes
  Blocks live deployment: yes/no

Sequencing notes:
- ...

Open questions:
- ...
```

## Implementation Ledger

Use this section as delegated implementation notes arrive. Keep it focused on actions, not restating the audit.

| ID | Pass | Priority | Status | Work item | Files | Proof required |
| --- | --- | --- | --- | --- | --- | --- |
| L1 | Live-Order Safety Refactor | P0 | Verified 2026-06-04 | Make every live runtime fail closed by default. Standalone execution must parse `BOT_V2_ALLOW_LIVE_EXECUTION` with default false; `RoutedExecutionService` default should not permit live; prod compose command/env must be explicit. | `quant_v2/execution/main.py`, `quant_v2/execution/service.py`, `docker-compose.yml`, `docker-compose.prod.yml`, `tests/quant_v2/test_execution_service.py`, `tests/infra/test_docker_compose_services.py` | Verified by focused pytest: default service shadows live, standalone unset env disables live, and prod compose does not inherit Dockerfile CMD. |
| L2 | Live-Order Safety Refactor | P0 | Verified 2026-06-04 | Make v2 Telegram session startup atomic on live gate failure. Cleanup newly-started source or bridge sessions when the paired start raises/fails, while preserving pre-existing sessions. | `quant/telebot/main.py`, `tests/quant/test_telebot_main_v2_handlers.py` | Verified by fake source plus bridge raising `go_no_go_failed`; newly-started source is stopped and no half-session remains. |
| L3 | Live-Order Safety Refactor | P0 | Verified 2026-06-04 | Remove normal post-only entry fallback to `MARKET`. Post-only/limit entry rejection should return rejected result or typed adapter failure; emergency reduce/flatten paths remain separate. | `quant_v2/execution/binance_adapter.py`, `quant/data/binance_client.py`, `tests/quant_v2/test_binance_adapter.py`, `tests/quant_v2/test_execution_service.py` | Verified by inverted adapter test: failed post-only limit returns rejected result and no `MARKET` call; Binance client post-only maps to `GTX`. |
| L4 | Live-Order Safety Refactor | P0 | Verified 2026-06-04 | Represent missing, zero, NaN, or infinite open-position prices as fail-closed stale-price state for live sessions. Do not let open positions disappear from risk math. | `quant_v2/execution/service.py`, `quant_v2/execution/reconciler.py`, `tests/quant_v2/test_execution_service.py`, `tests/quant_v2/test_execution_infra.py` | Verified by live open-position routing test recording `missing_live_mark` pause and reconciler test rejecting missing price for non-zero current position. |
| L5 | Live-Order Safety Refactor | P0 | Verified 2026-06-04 | Make cancel/flatten failures observable and truthful. `cancel_all_orders` should return structured status or raise after failed cancels/open-order residue; circuit breaker should not mark symbols canceled or flattened without confirmation. | `quant_v2/execution/binance_adapter.py`, `quant_v2/execution/service.py`, `quant_v2/execution/main.py`, `tests/quant_v2/test_binance_adapter.py`, `tests/quant_v2/test_live_readiness.py` | Verified by cancel-failure circuit-breaker test: false `canceled_symbols`/`flattened=True` are prevented and errors remain visible. |
| L6 | Live-Order Safety Refactor | P0 | Verified 2026-06-04 | Fix graceful shutdown flatten helper and `positions_flat` sentinel. Replace nonexistent helper call, factor shutdown flatten into testable helper, and log success sentinel only after every live session is confirmed flat/canceled. | `quant_v2/execution/main.py`, `tests/quant_v2/test_live_readiness.py` or `tests/quant_v2/test_execution_shutdown.py` | Verified by shutdown helper test: failed flatten returns false and does not emit `positions_flat`; `run_server` now logs `positions_not_flat` on failure. |
| L7 | Live-Order Safety Refactor | P1 | Verified 2026-06-04 | Align shutdown/monitoring scripts with actual compose service names or require `QUANT_CONTAINER_NAME` during install. | `deploy/flatten_on_shutdown.sh`, `deploy/dead_mans_switch.sh`, `deploy/setup_cloudwatch.sh`, `tests/infra/test_docker_compose_services.py` | Infra test loads compose container names and asserts deploy script defaults reference an existing container or fail closed when unset. |
| L8 | Live-Order Safety Refactor | P1 | Verified 2026-06-04 | Add compose/prod topology assertions. Choose explicit production command and live env defaults; do not rely on Dockerfile CMD for prod behavior. | `docker-compose.yml`, `docker-compose.prod.yml`, `tests/infra/test_docker_compose_services.py` | Default and prod compose use explicit `telegram_bot` commands and fail-closed live env; prod no longer inherits Dockerfile CMD or uses singleton `quant_bot`. |
| L9 | Live-Order Safety Refactor | P2 | Verified 2026-06-04 | Keep Binance client changes narrow: add post-only support to limit orders without broadly removing legacy market order helpers. | `quant/data/binance_client.py`, `tests/quant_v2/test_binance_client_phase4.py` | Verified by Binance client test asserting post-only limit sends `timeInForce=GTX`. |
| RWI1 | Redis, WAL, Idempotency, and Recovery | P0 | Verified 2026-06-04 | Add durable order-intent WAL events before any exchange placement. Record `order_intent_created`, `order_intent_completed`, and `order_intent_failed` with command/intent/client-order metadata and non-secret context. | `quant_v2/execution/state_wal.py`, `quant_v2/execution/main.py`, `quant_v2/execution/service.py`, `quant_v2/execution/idempotency.py`, `tests/quant_v2/test_live_readiness.py`, `tests/quant_v2/test_execution_infra.py` | Accepted route appends intent before adapter call and completion after result; failed intent append prevents adapter call; failed completion append after accepted order leaves stream unACKed. |
| RWI2 | Redis, WAL, Idempotency, and Recovery | P0 | Verified 2026-06-04 | Replace minute-only idempotency with command-scoped deterministic IDs. Use correlation ID or stream entry ID plus normalized order-intent hash; current time must not affect redelivered command keys. | `quant_v2/execution/idempotency.py`, `quant_v2/execution/main.py`, `quant_v2/execution/service.py`, `quant_v2/contracts.py`, `tests/quant_v2/test_execution_infra.py` | Same stream entry/correlation/order intent produces same key across retry; different command produces different key; minute changes do not affect redelivery. |
| RWI3 | Redis, WAL, Idempotency, and Recovery | P0 | Verified 2026-06-04 | Pass deterministic exchange client order IDs through the adapter. Derive sanitized/truncated IDs from durable intent and reuse them on crash retry. | `quant_v2/execution/adapters.py`, `quant_v2/execution/binance_adapter.py`, `quant/data/binance_client.py`, `tests/quant_v2/test_binance_adapter.py`, `tests/quant_v2/test_binance_client_phase4.py` | Fake Binance receives identical `newClientOrderId` on redelivery; duplicate client-order response is treated as already submitted rather than a second placement. |
| RWI4 | Redis, WAL, Idempotency, and Recovery | P0 | Verified 2026-06-04 | Make stream ACK depend on durable safety outcome rather than user-facing error publication. Validation errors may ACK; safety persistence failures must re-raise after publishing error event. | `quant_v2/execution/main.py`, `quant_v2/execution/redis_bus.py`, `tests/quant_v2/test_live_readiness.py`, `tests/quant_v2/test_session_lock.py` | WAL outcome append failure sends `route_signals_error` but leaves message unACKed; malformed user command is ACKed after error event; successful handler ACK tests remain. |
| RWI5 | Redis, WAL, Idempotency, and Recovery | P0 | Verified 2026-06-04 | Make live WAL replay fail closed without credentials. Rebuild live entries into explicit `recovery_paused`/`flatten_only` state unless a secure credential source is available. | `quant_v2/execution/main.py`, `quant_v2/execution/service.py`, `quant_v2/execution/state_wal.py`, `tests/quant_v2/test_live_readiness.py`, `tests/quant_v2/test_reconciliation.py` | Live WAL entry without credentials does not call live adapter factory; service reports paused recovery; routing returns no entries; alert event is emitted. |
| RWI6 | Redis, WAL, Idempotency, and Recovery | P0 | Verified 2026-06-04 | Persist and replay lifecycle and kill-switch state before consuming commands. Store absolute UTC deadlines/thresholds, not relative hours. | `quant_v2/execution/state_wal.py`, `quant_v2/execution/main.py`, `quant_v2/execution/service.py`, `tests/quant_v2/test_live_readiness.py`, `tests/quant_v2/test_session_lock.py` | WAL replay restores horizon deadline, stop-loss threshold, and kill-switch reasons; route after replay remains blocked until `kill_switch_cleared` is replayed. |
| RWI7 | Redis, WAL, Idempotency, and Recovery | P0 | Verified 2026-06-04 | Add simple HMAC command envelopes. Extend bus messages with producer/signature/schema version and verify canonical signed command data before dispatch when auth is required. | `quant_v2/execution/redis_bus.py`, `quant_v2/execution/main.py`, Telegram publisher/bridge tests, `tests/quant_v2/test_live_readiness.py` | Signed command accepted; tampered payload rejected before service mutation; unsigned command rejected with `BOT_REDIS_REQUIRE_COMMAND_AUTH=1`; legacy unsigned tests pass when auth is disabled. |
| RWI8 | Redis, WAL, Idempotency, and Recovery | P1 | Verified 2026-06-04 | Add stream command metadata to handler with minimal bus churn. Expose stream entry ID to routing/idempotency as fallback command identity. | `quant_v2/execution/redis_bus.py`, `quant_v2/execution/main.py`, `tests/quant_v2/test_live_readiness.py` | `_process_entry` gives handler access to exact stream ID and idempotency uses it when correlation ID is absent. |
| RWI9 | Redis, WAL, Idempotency, and Recovery | P1 | Verified 2026-06-04 | Reconcile cancel/flatten truthfulness with recovery WAL events after Pass 1 structured results exist. Replay unresolved flatten failures into paused recovery. | `quant_v2/execution/main.py`, `quant_v2/execution/state_wal.py`, `quant_v2/execution/service.py`, `tests/quant_v2/test_reconciliation.py`, `tests/quant_v2/test_live_readiness.py` | Cancel failure WAL entry replays into paused state; no flat/success state exists until verified flat. |
| RWI10 | Redis, WAL, Idempotency, and Recovery | P2 | Verified 2026-06-04 | Keep in-memory idempotency only for paper/demo convenience. Live safety must come from durable WAL intent state plus exchange client order IDs. | `quant_v2/execution/idempotency.py`, `quant_v2/execution/adapters.py`, `quant_v2/execution/binance_adapter.py`, `tests/quant_v2/test_execution_infra.py`, `tests/quant_v2/test_binance_adapter.py`, `tests/quant_v2/test_bounded_liquidation.py` | Paper adapter idempotency remains in-memory; Binance live adapter no longer consults an in-memory journal; entry and bounded-exit retries carry deterministic client order IDs and duplicate venue responses are reported as already submitted. |
| CLH1 | Credential, Logging, and Artifact Hygiene | P0 | Verified 2026-06-04 | Remove raw Telegram update logging. Delete the catch-all raw `Update` logger or gate it behind sanitized debug metadata only. | `quant/telebot/main.py`, Telegram handler tests | `/setup SENTINEL_API_KEY SENTINEL_SECRET` never appears in `caplog`; debug handler is not registered by default or logs only safe metadata. |
| CLH2 | Credential, Logging, and Artifact Hygiene | P0 | Verified 2026-06-04 | Add central redaction and safe error handling. Redact secret-bearing keys, auth headers, bot-token shaped strings, PEM blocks, and long credential-like values before logs/replies/crash notifications. | `quant/telebot/redaction.py` or `quant/security/redaction.py`, `quant/telebot/main.py`, `quant/telebot/manager.py`, `quant/telebot/engine.py`, tests | Exceptions containing sentinel secrets produce generic public replies and redacted logs/crash notifications with no raw secret text. |
| CLH3 | Credential, Logging, and Artifact Hygiene | P0 | Verified 2026-06-04 | Require approved users for `/setup` and clear credentials on revoke/ban. Pending, banned, or inactive users must not persist Binance keys. | `quant/telebot/main.py`, credential/auth tests | Pending/banned users cannot save credentials; approved users can; revoke clears encrypted credential fields and disables active runtime flags. |
| CLH4 | Credential, Logging, and Artifact Hygiene | P1 | Verified 2026-06-04 | Quarantine or convert unsafe diagnostic scripts. Remove hardcoded credentials and raw prints of API/session headers, account payloads, DB rows, Redis sessions, and response bodies from deployable artifacts. | `debug_credentials.py`, `test_capital_auth.py`, `redis_analyzer_local.py`, `analyze_sqlite_local.py`, `trade_analyzer_local.py`, `tools/security/scan_artifacts.py`, `tests/infra/test_release_artifact_scan.py` | Scanner rejects old unsafe script names and unsafe diagnostic content patterns before release; if any diagnostic is intentionally retained later, it must be rewritten with redacted output and covered by a separate safe-output test. |
| CLH5 | Credential, Logging, and Artifact Hygiene | P1 | Verified 2026-06-04 | Add release artifact and secret scan tooling. Non-extractive scanner should reject env files, PEM/key files, DBs, nested archives, logs, debug scripts, signal logs, audit dumps, model artifacts, and high-confidence tracked-file secrets unless allowlisted. | `tools/security/scan_artifacts.py`, `tools/security/scan_tracked_files.py`, `tests/infra/test_release_artifact_scan.py`, deployment scripts/docs | Temp archive tests reject `.env`, PEM, DB, nested archive entries, diagnostic scripts, model artifacts, and accept a clean release tree without printing file contents; tracked-file tests reject denied paths and high-confidence token/private-key signatures without printing secret values. |
| CLH6 | Credential, Logging, and Artifact Hygiene | P1 | Verified 2026-06-04 | Remove tracked archive from future releases and document rotation. Do non-history-rewriting cleanup first and document key/token rotation requirements. | `docs/KEY_ROTATION_AND_ARCHIVE_CLEANUP.md`, release docs, git index in implementation pass | `git ls-files -- deploy_optimized.tar.gz` returns empty; tracked secret/state/archive blocklist returns empty; rotation and history-cleanup runbook exists. |
| CLH7 | Credential, Logging, and Artifact Hygiene | P1 | Verified 2026-06-04 | Rewrite deployment docs away from dirty-tree upload and broad SSH. Use clean scanned release artifacts or signed/tagged host clone; keep secrets host-local or in a secret manager. | `DEPLOY.md`, `AWS_DEPLOY.md`, `deploy/*`, `tests/infra/test_deployment_hardening_docs.py` | Doc checks reject `scp -r`, `0.0.0.0/0`, repo-root PEM guidance, `.env` upload wording, and concrete host IP inventory; release flow requires scanner pass. |
| MAT1 | Model Registry and Artifact Trust | P0 | Verified 2026-06-04 | Enforce trusted model roots before registration, promotion, activation, or runtime load. Resolve artifact paths strictly under the configured model root, keep registry state separate, and reject out-of-root/symlinked artifacts before `joblib.load`. | `quant_v2/model_registry.py`, `quant/telebot/model_selection.py`, `quant_v2/telebot/signal_manager.py`, `quant_v2/models/ensemble.py`, `bootstrap_registry.py`, `quant_v2/research/scheduled_retrain.py`, tests | Out-of-root artifact dirs, symlinked artifact dirs, and untrusted active model paths are rejected; ensemble validator test proves rejected model files never reach `joblib.load`. |
| MAT2 | Model Registry and Artifact Trust | P0 | Verified 2026-06-04 | Require artifact manifests and checksums for active models. Add `model_manifest.json` with schema/version/files/horizons/hash/size/metrics/source metadata and verify it before promotion/load. | `quant_v2/model_registry.py`, `quant_v2/research/scheduled_retrain.py`, tests | Missing manifest, corrupt checksum, placeholder files, and wrong-horizon files cannot promote or load; scheduled retrain writes manifests for candidate artifacts. |
| MAT3 | Model Registry and Artifact Trust | P0 | Verified 2026-06-04 | Make promotion fail closed and prove runtime loadability. Require explicit `promotion_eligible is True`, complete horizons, validation score fields, manifest verification, schema validation, and smoke-load for required horizons. | `quant_v2/model_registry.py`, `quant_v2/models/ensemble.py`, `quant_v2/telebot/signal_manager.py`, `quant/telebot/main.py`, `tests/quant_v2/test_model_registry.py`, `tests/quant/test_telebot_main_v2_handlers.py` | Missing/false eligibility, missing validation scores, placeholder/corrupt `.pkl`, incomplete horizons, and wrong `TrainedModel.horizon` are rejected by promotion without active pointer change. |
| MAT4 | Model Registry and Artifact Trust | P0 | Verified 2026-06-04 | Disable production filesystem fallback. Production runtime should be registry-only; invalid/missing/quarantined/corrupt active pointer returns unresolved/fatal unless explicit dev/bootstrap fallback flag is set. | `quant/telebot/model_selection.py`, `quant/telebot/main.py`, `tests/quant/test_model_selection.py` | Invalid active registry pointer and missing active pointer return no runtime model by default; latest-filesystem fallback works only with `BOT_MODEL_ALLOW_FILESYSTEM_FALLBACK=1`. |
| MAT5 | Model Registry and Artifact Trust | P1 | Verified 2026-06-04 | Make active pointer changes atomic enough for file-backed registry. Validate and smoke-load before switching, write active pointer last, keep previous pointer on validation/write failure, and make rollback run the same checks. | `quant_v2/model_registry.py`, `quant/telebot/main.py`, tests | Simulated active-pointer write failure and pre-pointer status-write failure preserve the old active version and restore version statuses. |
| MAT6 | Model Registry and Artifact Trust | P1 | Verified 2026-06-04 | Restrict unsafe activation APIs. Make direct activation policy-checked; route admin promotion/rollback through the same validation; require bootstrap/dev paths to pass promotion policy. | `quant_v2/model_registry.py`, `bootstrap_registry.py`, `quant/telebot/main.py`, tests | Direct `set_active_version` cannot activate a manifest-valid but promotion-ineligible record; bootstrap uses `promote_version` and cannot activate placeholder/latest artifacts by default. |
| MAT7 | Model Registry and Artifact Trust | P1 | Verified 2026-06-04 | Remove silent partial-horizon fallback in production. Required horizons come from registry/manifest; partial horizon use requires explicit dev/paper flag. | `quant_v2/telebot/signal_manager.py`, `quant_v2/models/ensemble.py`, `tests/quant_v2/test_signal_manager.py` | Missing configured horizon with another horizon available fails closed by default; partial fallback works only with `BOT_MODEL_ALLOW_PARTIAL_HORIZONS=1`. |
| MAT8 | Model Registry and Artifact Trust | P1 | Verified 2026-06-04 | Parse retrain auto-promote fail-closed. Unknown boolean env values log and return safe default; `BOT_RETRAIN_AUTO_PROMOTE=maybe` must not promote. | `quant_v2/research/scheduled_retrain.py`, `tests/quant_v2/test_scheduled_retrain_candidates.py` | Candidate is registered but not promoted when `BOT_RETRAIN_AUTO_PROMOTE=maybe`; only explicit true values enable auto-promotion. |
| MAT9 | Model Registry and Artifact Trust | P2 | Verified 2026-06-04 | Fix legacy retrain tail labels. Preserve `NaN` for unlabeled tail rows and cast labels only after filtering valid rows. | `quant_v2/research/retrain_pipeline.py`, `tests/quant_v2/test_retrain_pipeline.py` | Final `horizon` rows are excluded from training rather than labeled class `0`. |
| MAT10 | Model Registry and Artifact Trust | P1 | Verified 2026-06-04 | Split writable model volumes from runtime read-only artifacts. Runtime mounts production artifacts read-only; retrain writes only to candidate/staging paths; registry state is separate. | `docker-compose.yml`, `docker-compose.prod.yml`, `tests/infra/test_docker_compose_services.py` | Runtime model artifact mounts are `:ro`; registry state is mounted separately at `/app/model_registry`; retrain is the only default service with writable production model artifacts. |
| DBH1 | Deployment, Build, and Runtime Hardening | P0 | Verified 2026-06-04 | Pick one explicit production topology. Recommended current production shape is `telegram_bot + private redis + retrain_scheduler`; standalone `quant_v2.execution.main` stays out of prod until live gates, WAL/idempotency, and command auth are complete. | `docker-compose.yml`, `docker-compose.prod.yml`, `DEPLOY.md`, `AWS_DEPLOY.md`, `tests/infra/test_docker_compose_services.py` | Default and prod compose use the same `redis`, `telegram_bot`, `retrain_scheduler` service topology with explicit app commands and fail-closed live env. |
| DBH2 | Deployment, Build, and Runtime Hardening | P0 | Verified 2026-06-04 | Make Redis private by default. Remove host `6379:6379`; use Docker-network-only Redis in default/prod, with localhost-only dev override plus ACL/auth if host access is needed. | `docker-compose.yml`, deployment docs, infra tests | Default compose Redis has no host `ports` publish and includes a healthcheck; runbooks require private Redis. |
| DBH3 | Deployment, Build, and Runtime Hardening | P0 | Verified 2026-06-04 | Harden compose runtime. Add read-only filesystem, dropped capabilities, no-new-privileges, explicit non-root user, healthchecks, resource limits, tmpfs, bounded logging, and explicit writable mounts. | `docker-compose.yml`, `docker-compose.prod.yml`, `tests/infra/test_docker_compose_services.py` | Infra tests assert app services have non-root user, `read_only`, `cap_drop`, `security_opt`, healthcheck, tmpfs, CPU limit, memory limit, and bounded logging. |
| DBH4 | Deployment, Build, and Runtime Hardening | P0 | Verified 2026-06-04 | Split writable mounts by trust boundary. App code and production model artifacts are read-only; DB/state, logs, registry state, and retrain staging are separate writable mounts; root `signal_log.json` bind is removed. | `docker-compose.yml`, `docker-compose.prod.yml`, deploy docs, model registry tests | Compose tests reject root `quant_bot.db`/`signal_log.json` binds and `/app/quant_bot.db`/`/app/signal_log.json`; runtime state uses `/state`, registry uses `/app/model_registry`, model artifacts are read-only for runtime. |
| DBH5 | Deployment, Build, and Runtime Hardening | P0 | Verified 2026-06-04 | Replace prod compose mismatch. Make `docker-compose.prod.yml` use the same base service names, not singleton `quant_bot`; include explicit command, private Redis, hardened mounts, and fail-closed live env. | `docker-compose.prod.yml`, `tests/infra/test_docker_compose_services.py` | Prod config includes `telegram_bot`, `redis`, and `retrain_scheduler`; infra test rejects accidental singleton `quant_bot`. |
| DBH6 | Deployment, Build, and Runtime Hardening | P1 | Verified 2026-06-04 | Align shutdown and monitoring scripts with production service names or fail closed when `QUANT_CONTAINER_NAME` is unset. Remove hardcoded `quant_execution`. | `deploy/flatten_on_shutdown.sh`, `deploy/dead_mans_switch.sh`, `deploy/setup_cloudwatch.sh`, `deploy/ec2_bootstrap.sh`, `tests/infra/test_docker_compose_services.py` | Infra test loads compose service/container names and asserts script defaults reference existing `quant_telegram`; no deploy script references stale `quant_execution`. |
| DBH7 | Deployment, Build, and Runtime Hardening | P1 | Verified 2026-06-04 | Add reproducible dependency locking. Recommended minimal path is `pip-tools` with `requirements.lock`/hashes and Docker install via `pip install --require-hashes`. | `pyproject.toml`, `requirements.in`, `requirements.lock`, `Dockerfile`, deploy docs, `tests/infra/test_docker_compose_services.py`, `tools/security/production_readiness.py` | Dockerfile copies `requirements.lock` and installs with `pip install --require-hashes`; lock dry-run passes; Linux CPython 3.11 wheel/hash availability passes; infra test rejects direct unpinned Dockerfile package installs. |
| DBH8 | Deployment, Build, and Runtime Hardening | P1 | Verified 2026-06-04 | Pin Docker base images and tighten package installs. Pin Python and Redis images by digest; remove direct unpinned package installs from Dockerfile. | `Dockerfile`, compose files, lock files, deploy docs, `tests/infra/test_docker_compose_services.py` | Python base stages and Redis compose images are digest pinned; infra test rejects un-digested pulled images and direct unpinned `pip install numpy pandas ...`. |
| DBH9 | Deployment, Build, and Runtime Hardening | P1 | Verified 2026-06-04 | Add release packaging and artifact scan workflow. Generate releases from tracked allowlisted files only and run artifact scanner before deploy. | `tools/security/build_release.py`, `tools/security/scan_artifacts.py`, `tests/infra/test_release_artifact_scan.py`, `DEPLOY.md`, `AWS_DEPLOY.md` | Release builder refuses dirty trees by default, archives tracked files with `git archive`, scans the archive, removes unsafe output on scan failure, and is part of readiness checks. |
| DBH10 | Deployment, Build, and Runtime Hardening | P1 | Verified 2026-06-04 | Rewrite deployment runbooks. Remove active server/key inventory, broad SSH, recursive upload, sample secrets, and repo-root private key handling; recommend SSM or restricted SSH and secrets outside repo root. | `DEPLOY.md`, `AWS_DEPLOY.md`, `docs/PRODUCTION_READINESS.md`, `tests/infra/test_deployment_hardening_docs.py` | Doc tests reject `0.0.0.0/0`, `scp -r`, repo-root PEM guidance, `.env` upload wording, and concrete host IP inventory; runbooks require clean scanned release flow and restricted operator access. |
| DBH11 | Deployment, Build, and Runtime Hardening | P2 | Verified 2026-06-04 | Add SBOM and dependency audit hooks in CI/readiness pass. Generate SBOM and run `pip-audit` or OSV scanner against locked dependencies. | `.github/workflows/production-readiness.yml`, `tools/security/production_readiness.py`, `tools/security/generate_sbom.py`, `tests/infra/test_production_readiness.py` | Readiness wrapper runs `pip-audit` and SBOM generation; vulnerable locked packages were bumped until audit returned no known vulnerabilities; workflow uploads `build/security/sbom.cdx.json`. |
| CRG1 | CI, Regression, and Production Readiness Gate | P0 | Verified 2026-06-04 | Add failing P0 regressions before code changes. Cover P0-1 through P0-8: live defaults, atomic live startup cleanup, no post-only market fallback, missing mark pause, truthful cancel/flatten, Redis redelivery idempotency, Redis command auth, credential redaction, model fail-closed behavior, and prod compose explicitness. | `tools/security/production_readiness.py`, `tests/infra/test_production_readiness.py`, focused P0 test files | Focused P0 suite includes all deployment-blocking gate files and passes locally; static readiness test asserts required gate files stay in `FOCUSED_P0_TESTS`. |
| CRG2 | CI, Regression, and Production Readiness Gate | P0 | Verified 2026-06-04 | Invert unsafe historical tests. Replace tests expecting `fallback_to_market`, invalid-active latest fallback, placeholder promotion, and bare model path acceptance with fail-closed assertions. | `tests/quant_v2/test_binance_adapter.py`, `tests/quant/test_model_selection.py`, `tests/quant_v2/test_model_registry.py`, `tests/quant/test_telebot_main_v2_handlers.py`, `tests/quant_v2/test_signal_manager.py`, `tests/infra/test_production_readiness.py` | Static readiness test proves unsafe historical expectations are inverted: no post-only market fallback, latest filesystem fallback requires explicit flag, placeholder/corrupt/ineligible promotion is rejected, and partial-horizon fallback is blocked by default. |
| CRG3 | CI, Regression, and Production Readiness Gate | P0 | Verified 2026-06-06 | Add production readiness CI workflow. Jobs should run focused P0 tests, full tests, compose validation, artifact scan, secret scan, dependency audit, SBOM generation, and unsafe-doc grep. | `.github/workflows/production-readiness.yml`, `pyproject.toml`, `requirements-ci.in`, `requirements-ci.lock`, `tests/infra/test_production_readiness.py`, `tests/infra/test_release_artifact_scan.py`, `docs/CRG3_CI_EVIDENCE.md`, `tools/security/scan_tracked_files.py`, `tools/security/finalize_crg3.py` | Production Readiness workflow passed with exact completed checks recorded in CRG3 evidence JSON: `focused P0 regression suite`, `full pytest suite`, `unsafe deployment docs grep`, `tracked-file secret scan`, `hashed CI tool lock dry-run`, `Linux CPython 3.11 CI tool wheel availability`, `hashed dependency lock dry-run`, `Linux CPython 3.11 locked wheel availability`, `dependency vulnerability audit`, `CycloneDX SBOM generation`, `CRG3 evidence schema validation`, `roadmap P0 ledger evidence`, `scanned release artifact packaging`, `default compose config`, `production compose config`, `production image build`, `production-sbom artifact upload`, `downloaded CycloneDX SBOM artifact validation`; artifact ID/download URL binding, canonical artifact URL validation, and cross-host redirect authorization stripping; repo: `chainsyncstore/hypothesis-research-engine`; repo binding: `origin fetch remote`; run: https://github.com/chainsyncstore/hypothesis-research-engine/actions/runs/27053934293; commit: `c6ee071cb628cb7a48b3d2a42f38093b4f6c2677`; run id: `27053934293`; run attempt: `1`; artifact: `production-sbom`. |
| CRG4 | CI, Regression, and Production Readiness Gate | P0 | Verified 2026-06-04 | Add local readiness command wrapper. One command should run deploy-blocking checks from a clean checkout and print pass/fail evidence without hiding failures. | `tools/security/production_readiness.py`, `docs/PRODUCTION_READINESS.md`, `tests/infra/test_production_readiness.py` | Wrapper defines focused P0 tests, hash-lock dry run, compose config, Docker build, and tracked release scan; live profile rejects Docker skips; local dry-run prints evidence commands without hiding failures. |
| CRG5 | CI, Regression, and Production Readiness Gate | P0 | Verified 2026-06-04 | Track ledger status with evidence links. Add status values and evidence fields so no P0 item can be called complete without implementation and verification proof. | `tools/security/check_roadmap.py`, `tools/security/production_readiness.py`, `tests/infra/test_production_readiness.py`, `PRODUCTION_REFACTOR_ROADMAP.md` | Roadmap validator is part of readiness; it fails if any P0 row is not `Verified YYYY-MM-DD` or lacks proof scope. Current validator blocks live readiness on remaining open `CRG3`. |
| CRG6 | CI, Regression, and Production Readiness Gate | P0 | Verified 2026-06-04 | Define the live enablement checklist. Require all P0 gates verified, private Redis, command auth where applicable, explicit live env, go/no-go true, rollback clear, valid model manifest, fresh mark checks, key rotation record, scanned release artifact, restricted access, and rehearsed rollback/disable. | `docs/PRODUCTION_READINESS.md`, `DEPLOY.md`, `AWS_DEPLOY.md`, `tests/infra/test_deployment_hardening_docs.py` | Docs test requires live enablement checklist terms: P0 ledger, readiness wrapper, CI workflow, private/authenticated Redis, go/no-go true, rollback clear, valid manifests/checksums, fresh marks, restricted operator access, rehearsed rollback, and rotation. |
| CRG7 | CI, Regression, and Production Readiness Gate | P0 | Verified 2026-06-04 | Establish "do not deploy live unless these pass" commands. Document the focused pytest suite, compose config, artifact scan, secret scan, dependency audit, SBOM, and readiness wrapper commands. | `docs/PRODUCTION_READINESS.md`, `.github/workflows/production-readiness.yml`, `tools/security/production_readiness.py`, `requirements-ci.lock`, `tests/infra/test_production_readiness.py` | Readiness wrapper and docs include focused P0 suite, compose config, Docker build, release scan, CI/runtime hash dry-runs, dependency audit, SBOM generation, and CI workflow/SBOM artifact requirements. |
| CRG8 | CI, Regression, and Production Readiness Gate | P1 | Verified 2026-06-04 | Add artifact, secret, docs, and compose infra tests. Reject sensitive artifacts, public Redis, missing hardening, stale `quant_execution`, singleton prod compose mismatch, unpinned Docker images, unsafe docs strings, and release-sensitive files. | `tests/infra/test_release_artifact_scan.py`, `tests/infra/test_docker_compose_services.py`, `tests/infra/test_deployment_hardening_docs.py`, `tests/infra/test_production_readiness.py` | Infra suite rejects sensitive release members/content, public Redis, missing hardening, stale `quant_execution`, singleton prod compose mismatch, unpinned Docker images, unsafe deployment docs, and missing readiness workflow evidence. |
| CRG9 | CI, Regression, and Production Readiness Gate | P1 | Verified 2026-06-04 | Add implementation report template after each pass. Reports must include changed ledger IDs, files, invariant proven, tests added/inverted, commands run, residual risk, rollback plan, and live-block status. | `docs/IMPLEMENTATION_REPORT_TEMPLATE.md`, `tests/infra/test_production_readiness.py`, `PRODUCTION_REFACTOR_ROADMAP.md` | Template test requires ledger IDs, files, invariants, tests/checks, commands, results, rollback plan, residual risk, live-block status, dated verification, and external run evidence fields. |
| CRG10 | CI, Regression, and Production Readiness Gate | P2 | Verified 2026-06-04 | Defer non-blocking hardening with explicit rationale. Long-term signing, Redis TLS for same-host Docker, broad dependency modernization, and non-critical polish can wait; live defaults, Redis privacy/auth, WAL/idempotency, credential redaction, artifact scans, model trust, compose explicitness, and readiness evidence cannot. | `docs/DEFERRED_HARDENING.md`, `tests/infra/test_production_readiness.py`, `PRODUCTION_REFACTOR_ROADMAP.md` | Deferred-hardening test proves P0 guardrails are listed as non-deferrable and allowed deferrals carry rationale plus guardrails. |

### Pass 1 Sequencing Notes

1. Invert unsafe tests first: market fallback, live defaults, and compose command inheritance.
2. Implement live default/env parsing before deeper execution changes, so accidental live starts are blocked while work continues.
3. Remove post-only market fallback next; it is narrow and should not disturb paper/demo.
4. Handle missing live marks and cancel/flatten truthfulness together around the invariant: do not lie about exposure.
5. Fix shutdown sentinel after circuit-breaker results become trustworthy.
6. Update deploy script names and compose assertions after the intended production topology is explicit.

Open topology questions for implementation:

- Is Pass 1 targeting Telegram-only `quant_telegram`, standalone execution service, or both with Redis bridge?
- Should missing live marks immediately trigger emergency flatten, or pause entries and alert while allowing only explicit operator reduce-only flatten?
- Should `sync_positions` require an explicit mode such as `rebalance`, `flatten`, or `maintenance_restore` before it can place live orders?
- Confirm Binance Futures post-only support should use `timeInForce="GTX"` for this account/API surface.

### Pass 2 Sequencing Notes

1. Add command metadata and deterministic key helpers first; each command needs a stable identity before durable idempotency can work.
2. Add WAL order-intent events and write intent before placement, then completion/failure after adapter result.
3. Change `_handle_command` failure semantics so safety-critical WAL failures keep stream entries unACKed.
4. Pass deterministic `client_order_id` into Binance so crash-after-placement retries reuse the same exchange identity.
5. Implement live replay fail-closed behavior and lifecycle/kill-switch replay.
6. Add HMAC envelopes after command metadata exists; it uses the same canonical message shape but is conceptually separate.
7. Wire cancel/flatten recovery events after Pass 1 returns truthful cancel/flatten result objects.

Open Redis/WAL questions for implementation:

- Is there an approved secure credential store for live replay, or should Pass 2 implement only `recovery_paused`/`flatten_only` without credentials?
- Should correlation ID be required from Telegram publishers, or should stream entry ID remain the canonical fallback?
- Should unresolved `order_intent_created` with no outcome query the exchange by client order ID before retrying placement?
- Confirm Binance Futures `newClientOrderId` length and character constraints for this account mode.
- Should Redis HMAC be mandatory in every environment, or only when `BOT_REDIS_REQUIRE_COMMAND_AUTH=1` is set?

### Pass 3 Sequencing Notes

1. Start with Telegram redaction and `/setup` authorization because they protect credentials already flowing through the bot.
2. Quarantine diagnostics and add artifact scanning next, so cleanup is mechanically enforced.
3. Remove tracked archives from the index only after scan tooling exists; do not rewrite git history without explicit approval.
4. Update deployment docs/scripts to use clean, scanned artifacts rather than dirty-tree upload.
5. Finish with sentinel regression tests across logs, replies, crash notifications, diagnostics, and release bundles.

Open credential/artifact questions for implementation:

- Should revoked users always lose stored credentials, or should a separate temporary suspension state preserve encrypted credentials while blocking runtime use?
- Should sanitized diagnostics live under `tools/security/`, or split scanners into `tools/security/` and runtime inspection into `tools/diagnostics/`?
- Is history rewriting in scope after key rotation, or should this pass only remove tracked artifacts going forward and document historical exposure?

### Pass 4 Sequencing Notes

1. Invert unsafe tests first: invalid active fallback, placeholder promotion, unknown auto-promote, and partial-horizon fallback.
2. Add trusted-root and manifest verification before changing admin promotion flow.
3. Route promotion and rollback through one policy-checked activation path.
4. Demote `set_active_version` to internal/dev-only so it cannot bypass production policy.
5. Split compose model mounts so the runtime cannot mutate artifacts it is about to deserialize.

Open model-trust questions for implementation:

- Should legacy pre-existing models be bootstrapped with generated manifests, or treated as dev-only until retrained?
- Should partial-horizon artifacts ever be allowed for paper quarantine, or only for offline research?
- Is an unsigned checksum manifest sufficient for Pass 4, or should external signing be pulled forward before live deployment?

### Pass 5 Sequencing Notes

1. Fix topology and Redis exposure first; they define the rest of the production deployment shape.
2. Replace `docker-compose.prod.yml` with an explicit hardened prod override and align deploy scripts to that shape.
3. Split writable mounts, especially runtime model artifacts versus retrain staging.
4. Introduce dependency locking and Docker digest pinning.
5. Rewrite deployment docs around clean scanned releases.
6. Leave SBOM/dependency audit automation for Pass 6 CI wiring unless time permits earlier.

Open deployment/build questions for implementation:

- Should production keep retrain running continuously, or should retrain be operator-triggered/job-style until model promotion trust is complete?
- Is Redis host access needed for any approved ops workflow, or can diagnostics use `docker exec`/internal network access?
- Should secrets use Docker Compose secrets, AWS SSM/Secrets Manager, or a locked host-local env file for the first production hardening pass?

### Pass 6 Sequencing Notes

1. Invert unsafe tests and add missing P0 regression tests first, even though they will fail.
2. Implement Passes 1-5 against those tests, updating each ledger row from `Ready` to `Verified` only when evidence exists.
3. Add CI after the focused P0 suite exists, then wire the readiness wrapper to run the same checks locally.
4. Only after every P0 gate is verified should runbooks allow setting live env flags.
5. Full test suite and SBOM/dependency audit should run before release; focused P0 tests are the fastest stop-the-bleeding loop during implementation.

Open CI/readiness questions for implementation:

- Should CI use GitHub Actions specifically, or another provider?
- Should `production_readiness.py` be Python-only for portability, or should there also be a PowerShell wrapper for Windows operators?
- Who owns final signoff for live enablement after all P0 gates pass: code owner, operator, or a two-person checklist?

## Implementation Reports

### 2026-06-04 Pass 1 Initial Slice

Ledger IDs changed:
- `L1` verified for fail-closed service/standalone defaults and explicit prod command/env.
- `L2` verified for cleanup when bridge startup raises after source startup.
- `L3` verified for no normal post-only entry fallback to `MARKET`.
- `L8` moved to in progress for initial prod command/env assertion; full topology hardening remains under `DBH1`/`DBH5`.
- `L9` verified for Binance post-only `GTX` support.

Files changed:
- `quant_v2/execution/service.py`
- `quant_v2/execution/main.py`
- `quant_v2/execution/binance_adapter.py`
- `quant/data/binance_client.py`
- `quant/telebot/main.py`
- `docker-compose.prod.yml`
- `tests/quant_v2/test_binance_adapter.py`
- `tests/quant_v2/test_binance_client_phase4.py`
- `tests/quant_v2/test_execution_service.py`
- `tests/quant_v2/test_live_readiness.py`
- `tests/quant/test_telebot_main_v2_handlers.py`
- `tests/infra/test_docker_compose_services.py`

Safety invariants proven:
- Unset standalone live env disables live execution.
- `RoutedExecutionService()` defaults live requests to paper-shadow rather than live.
- Prod compose no longer inherits the Dockerfile standalone execution command.
- A bridge start failure after source start stops the newly-created source session.
- A failed post-only limit entry returns a rejected result and does not call `MARKET`.
- Binance client post-only limit orders use `timeInForce=GTX`.

Verification command:

```powershell
python -m pytest tests/quant_v2/test_binance_adapter.py tests/quant_v2/test_binance_client_phase4.py tests/quant_v2/test_execution_service.py tests/quant_v2/test_live_readiness.py tests/quant/test_telebot_main_v2_handlers.py tests/infra/test_docker_compose_services.py -q
```

Result:
- `91 passed in 5.54s`

Residual risk:
- At this slice boundary, missing-price exposure handling (`L4`), cancel/flatten truthfulness (`L5`), shutdown sentinel correctness (`L6`), script service-name alignment (`L7`), and full prod topology hardening (`DBH1`/`DBH5`) were still open. Later reports below supersede the `L4`-`L7` status.

### 2026-06-04 Pass 1 Exposure Truthfulness Slice

Ledger IDs changed:
- `L4` verified for live missing-mark fail-closed routing and reconciler missing-price rejection.
- `L5` verified for observable cancel/flatten failures and no false canceled/flattened result.
- `L6` verified for shutdown flatten helper and `positions_flat` sentinel gating.

Files changed:
- `quant_v2/execution/reconciler.py`
- `quant_v2/execution/service.py`
- `quant_v2/execution/adapters.py`
- `quant_v2/execution/binance_adapter.py`
- `quant_v2/execution/main.py`
- `tests/quant_v2/test_execution_infra.py`
- `tests/quant_v2/test_execution_service.py`
- `tests/quant_v2/test_live_readiness.py`

Safety invariants proven:
- A live session with an open position and no fresh finite mark for that position pauses routing with `missing_live_mark`.
- Reconciliation cannot silently omit a non-zero current position because its price is missing.
- `cancel_all_orders` surfaces incomplete cancels or remaining open orders.
- Circuit breaker results include remaining positions/orders and do not report `flattened=True` when cancel/flatten verification fails.
- Shutdown flattening uses `_execute_stale_feed_circuit_breaker` and only permits the `positions_flat` sentinel after verified success.

Verification commands:

```powershell
python -m pytest tests/quant_v2/test_execution_infra.py tests/quant_v2/test_execution_service.py tests/quant_v2/test_live_readiness.py tests/quant_v2/test_binance_adapter.py tests/quant_v2/test_reconciliation.py -q
python -m pytest tests/quant_v2/test_binance_adapter.py tests/quant_v2/test_binance_client_phase4.py tests/quant_v2/test_execution_service.py tests/quant_v2/test_live_readiness.py tests/quant_v2/test_execution_infra.py tests/quant_v2/test_reconciliation.py tests/quant/test_telebot_main_v2_handlers.py tests/infra/test_docker_compose_services.py -q
```

Results:
- Focused L4-L6 suite: `68 passed in 4.44s`
- Broader Pass 1 suite: `104 passed in 4.44s`

Residual risk:
- At this slice boundary, `L7` service-name alignment and full deployment hardening were still open, and Pass 2 Redis/WAL/idempotency P0 work had not started. Later reports below supersede the `L7` and Pass 2 status.

### 2026-06-04 Pass 1 Script Alignment Slice

Ledger IDs changed:
- `L7` verified for shutdown, dead-man switch, and CloudWatch heartbeat script alignment with current compose container names.

Files changed:
- `deploy/flatten_on_shutdown.sh`
- `deploy/dead_mans_switch.sh`
- `deploy/setup_cloudwatch.sh`
- `tests/infra/test_docker_compose_services.py`

Safety invariants proven:
- Deploy scripts no longer default to the removed `quant_execution` container.
- `quant_telegram` is the default monitored/flattened container for `docker-compose.yml`.
- Operators can still override the target with `QUANT_CONTAINER_NAME`, including `quant_bot` for the current singleton prod compose.
- CloudWatch heartbeat uses an exact container-name match rather than a loose substring match.

Verification commands:

```powershell
python -m pytest tests/infra/test_docker_compose_services.py -q
python -m pytest tests/quant_v2/test_binance_adapter.py tests/quant_v2/test_binance_client_phase4.py tests/quant_v2/test_execution_service.py tests/quant_v2/test_live_readiness.py tests/quant_v2/test_execution_infra.py tests/quant_v2/test_reconciliation.py tests/quant/test_telebot_main_v2_handlers.py tests/infra/test_docker_compose_services.py -q
```

Results:
- Infra script suite: `3 passed in 0.18s`
- Broader Pass 1 suite: `105 passed in 5.26s`

Residual risk:
- Live deployment remains blocked. Full production topology/container hardening remains tracked under `L8`, `DBH1`, and `DBH5`; Pass 2 Redis/WAL/idempotency P0 work has not started.

### 2026-06-04 Pass 2 Command Identity and Auth Slice

Ledger IDs changed:
- `RWI7` verified for optional HMAC command envelopes on Redis stream commands.
- `RWI8` verified for stream entry metadata propagation into command handling.
- `RWI2` moved to in progress for command-scoped idempotency keys when a correlation ID or stream entry ID is available.

Files changed:
- `quant_v2/execution/redis_bus.py`
- `quant_v2/execution/idempotency.py`
- `quant_v2/execution/main.py`
- `quant_v2/execution/service.py`
- `tests/quant_v2/test_live_readiness.py`
- `tests/quant_v2/test_execution_infra.py`

Safety invariants proven:
- Redis stream messages carry `producer`, `schema_version`, and `signature` envelope fields.
- Stream publishers auto-sign messages when `BOT_REDIS_COMMAND_AUTH_SECRET` or an injected command secret is configured.
- Auth-required stream consumers reject unsigned or tampered commands before invoking the command handler, then ACK/skip them to avoid poison replay.
- `_process_entry` stamps the exact Redis stream ID onto `BusMessage.stream_id` before dispatch.
- `ExecutionEngineServer` uses `correlation_id` or stream ID as the private routing `_command_id`.
- `build_idempotency_key` ignores epoch-minute drift when `command_id` is present, so redelivered stream commands keep the same deterministic key.

Verification commands:

```powershell
python -m pytest tests/quant_v2/test_live_readiness.py tests/quant_v2/test_execution_infra.py tests/quant_v2/test_day2_infra_patches.py tests/quant_v2/test_session_lock.py -q
python -m pytest tests/quant_v2/test_binance_adapter.py tests/quant_v2/test_binance_client_phase4.py tests/quant_v2/test_execution_service.py tests/quant_v2/test_live_readiness.py tests/quant_v2/test_execution_infra.py tests/quant_v2/test_reconciliation.py tests/quant_v2/test_day2_infra_patches.py tests/quant_v2/test_session_lock.py tests/quant/test_telebot_main_v2_handlers.py tests/infra/test_docker_compose_services.py -q
```

Results:
- Focused Pass 2 command-auth suite: `40 passed in 4.45s`
- Broader Pass 1/2 safety suite: `127 passed in 5.72s`

Residual risk:
- At this slice boundary, `RWI1` and `RWI3` still needed durable order intent WAL records and deterministic exchange client order IDs. Later reports below supersede the `RWI1`/`RWI3` status; `RWI5`/`RWI6` recovery replay state remains open.

### 2026-06-04 Pass 2 ACK Durability Slice

Ledger IDs changed:
- `RWI4` verified for accepted-order WAL append failure requeue behavior.

Files changed:
- `quant_v2/execution/main.py`
- `tests/quant_v2/test_live_readiness.py`

Safety invariants proven:
- Accepted order results must be durably logged with `log_order_executed`; if that WAL append fails, the command handler publishes `route_signals_error` and re-raises `DurableSafetyError`.
- `RedisStreamCommandBus._process_entry` does not `XACK` when the command handler re-raises the durable safety error, preserving stream redelivery.
- Existing successful handler ACK tests still pass.

Verification commands:

```powershell
python -m pytest tests/quant_v2/test_live_readiness.py tests/quant_v2/test_execution_infra.py tests/quant_v2/test_day2_infra_patches.py tests/quant_v2/test_session_lock.py -q
python -m pytest tests/quant_v2/test_binance_adapter.py tests/quant_v2/test_binance_client_phase4.py tests/quant_v2/test_execution_service.py tests/quant_v2/test_live_readiness.py tests/quant_v2/test_execution_infra.py tests/quant_v2/test_reconciliation.py tests/quant_v2/test_day2_infra_patches.py tests/quant_v2/test_session_lock.py tests/quant/test_telebot_main_v2_handlers.py tests/infra/test_docker_compose_services.py -q
```

Results:
- Focused Pass 2 ACK suite: `41 passed in 4.43s`
- Broader Pass 1/2 safety suite: `128 passed in 5.55s`

Residual risk:
- At this slice boundary, `RWI1` and `RWI3` still needed pre-placement durable order intents and deterministic exchange client order IDs. Later reports below supersede the `RWI1`/`RWI3` status; `RWI5`/`RWI6` recovery replay state remains open.

### 2026-06-04 Pass 2 Durable Intent and Client Order Slice

Ledger IDs changed:
- `RWI1` verified for pre-placement durable order-intent WAL events and completion/failure intent recording.
- `RWI2` verified for command-scoped idempotency now flowing from Redis stream IDs/correlation IDs into service order keys.
- `RWI3` verified for deterministic Binance client order IDs and duplicate client-order retry handling.

Files changed:
- `quant_v2/execution/state_wal.py`
- `quant_v2/execution/idempotency.py`
- `quant_v2/execution/service.py`
- `quant_v2/execution/main.py`
- `quant_v2/execution/binance_adapter.py`
- `quant/data/binance_client.py`
- `tests/quant_v2/test_execution_service.py`
- `tests/quant_v2/test_live_readiness.py`
- `tests/quant_v2/test_execution_infra.py`
- `tests/quant_v2/test_binance_adapter.py`
- `tests/quant_v2/test_binance_client_phase4.py`

Safety invariants proven:
- `order_intent_created`, `order_intent_completed`, and `order_intent_failed` WAL events exist and are scrubbed through the existing WAL serializer.
- `RoutedExecutionService` records `order_intent_created` before adapter placement; if that write fails, the adapter is not called.
- Accepted order intent completion failures are wrapped as `DurableSafetyError`; stream processing publishes an error event but leaves the Redis message unACKed.
- Deterministic client order IDs are derived from idempotency keys, bounded to Binance's client-order length, and attached to routed `OrderPlan`s.
- Binance limit orders send `newClientOrderId`; duplicate client-order errors are treated as `already_submitted` rather than market fallback or ordinary rejection.

Verification commands:

```powershell
python -m pytest tests/quant_v2/test_execution_service.py tests/quant_v2/test_live_readiness.py tests/quant_v2/test_execution_infra.py tests/quant_v2/test_binance_adapter.py tests/quant_v2/test_binance_client_phase4.py -q
python -m pytest tests/quant_v2/test_binance_adapter.py tests/quant_v2/test_binance_client_phase4.py tests/quant_v2/test_execution_service.py tests/quant_v2/test_live_readiness.py tests/quant_v2/test_execution_infra.py tests/quant_v2/test_reconciliation.py tests/quant_v2/test_day2_infra_patches.py tests/quant_v2/test_session_lock.py tests/quant_v2/test_credential_scrubbing.py tests/quant/test_telebot_main_v2_handlers.py tests/infra/test_docker_compose_services.py -q
```

Results:
- Focused durable-intent/client-order suite: `90 passed in 4.79s`
- Broader Pass 1/2 safety suite: `140 passed in 5.61s`

Residual risk:
- At this slice boundary, `RWI5` and `RWI6` still needed live WAL replay recovery-paused behavior and lifecycle/kill-switch persistence/replay. Later reports below supersede the `RWI5` status; `RWI6` and `RWI9` remain open.

### 2026-06-04 Pass 2 Live Replay Recovery Slice

Ledger IDs changed:
- `RWI5` verified for fail-closed live WAL replay without credentials.

Files changed:
- `quant_v2/execution/main.py`
- `quant_v2/execution/service.py`
- `tests/quant_v2/test_live_readiness.py`

Safety invariants proven:
- A live `session_started` WAL entry no longer attempts to rebuild a live Binance adapter without credentials.
- Replayed live sessions enter `recovery_paused` mode with a `live_wal_replay_without_credentials` kill-switch reason.
- Recovery-paused routing returns no orders and preserves the pause state.
- The execution server emits a `live_recovery_paused` event for control-plane visibility.
- `get_live_session_ids()` excludes recovery-paused sessions, so reconciliation/heartbeat code does not treat them as safely restored live adapters.

Verification commands:

```powershell
python -m pytest tests/quant_v2/test_live_readiness.py tests/quant_v2/test_reconciliation.py -q
python -m pytest tests/quant_v2/test_binance_adapter.py tests/quant_v2/test_binance_client_phase4.py tests/quant_v2/test_execution_service.py tests/quant_v2/test_live_readiness.py tests/quant_v2/test_execution_infra.py tests/quant_v2/test_reconciliation.py tests/quant_v2/test_day2_infra_patches.py tests/quant_v2/test_session_lock.py tests/quant_v2/test_credential_scrubbing.py tests/quant/test_telebot_main_v2_handlers.py tests/infra/test_docker_compose_services.py -q
```

Results:
- Focused live replay suite: `23 passed in 3.97s`
- Broader Pass 1/2 safety suite: `141 passed in 5.54s`

Residual risk:
- At this slice boundary, `RWI6` still needed lifecycle/kill-switch persistence and replay. Later reports below supersede the `RWI6` status; `RWI9` still needs cancel/flatten recovery WAL replay.

### 2026-06-04 Pass 2 Lifecycle and Kill-Switch Replay Slice

Ledger IDs changed:
- `RWI6` verified for lifecycle persistence/replay and kill-switch persistence/replay.

Files changed:
- `quant_v2/execution/state_wal.py`
- `quant_v2/execution/main.py`
- `quant_v2/execution/service.py`
- `tests/quant_v2/test_live_readiness.py`

Safety invariants proven:
- `set_lifecycle` writes `lifecycle_changed` WAL events with absolute UTC horizon deadlines and absolute stop-loss equity thresholds.
- WAL replay restores watchdog horizon deadline, stop-loss threshold, and service lifecycle rules before command consumption.
- WAL replay restores `kill_switch_triggered` reasons into service state, and normal routing remains blocked after replay.
- A later `kill_switch_cleared` WAL event clears the replayed pause.
- Persisted replay pauses are sticky and are not cleared by ordinary monitoring recomputation.

Verification commands:

```powershell
python -m pytest tests/quant_v2/test_live_readiness.py tests/quant_v2/test_session_lock.py tests/quant_v2/test_watchdog_flatten_retry.py -q
python -m pytest tests/quant_v2/test_binance_adapter.py tests/quant_v2/test_binance_client_phase4.py tests/quant_v2/test_execution_service.py tests/quant_v2/test_live_readiness.py tests/quant_v2/test_execution_infra.py tests/quant_v2/test_reconciliation.py tests/quant_v2/test_day2_infra_patches.py tests/quant_v2/test_session_lock.py tests/quant_v2/test_watchdog_flatten_retry.py tests/quant_v2/test_credential_scrubbing.py tests/quant/test_telebot_main_v2_handlers.py tests/infra/test_docker_compose_services.py -q
```

Results:
- Focused lifecycle/kill-switch suite: `29 passed in 4.15s`
- Broader Pass 1/2 safety suite: `148 passed in 5.69s`

Residual risk:
- At this slice boundary, `RWI9` still needed cancel/flatten recovery WAL replay. Later reports below supersede the `RWI9` status.

### 2026-06-04 Pass 2 Flatten Recovery Replay Slice

Ledger IDs changed:
- `RWI9` verified for cancel/flatten recovery WAL events and unresolved recovery replay.

Files changed:
- `quant_v2/execution/state_wal.py`
- `quant_v2/execution/main.py`
- `tests/quant_v2/test_live_readiness.py`

Safety invariants proven:
- Circuit breaker flatten paths log `flatten_attempted` before cancel/flatten work.
- Circuit breaker results log `flatten_completed` only when verified flat/canceled, otherwise `flatten_failed`.
- Cancel failure WAL entries include structured failure details and replay into a paused `unresolved_flatten_recovery` state.
- A crash after `flatten_attempted` but before a result is treated as unresolved and replays into pause.
- A later verified `flatten_completed(flattened=True)` entry is not replayed as a failure.

Verification commands:

```powershell
python -m pytest tests/quant_v2/test_live_readiness.py tests/quant_v2/test_reconciliation.py -q
python -m pytest tests/quant_v2/test_binance_adapter.py tests/quant_v2/test_binance_client_phase4.py tests/quant_v2/test_execution_service.py tests/quant_v2/test_live_readiness.py tests/quant_v2/test_execution_infra.py tests/quant_v2/test_reconciliation.py tests/quant_v2/test_day2_infra_patches.py tests/quant_v2/test_session_lock.py tests/quant_v2/test_watchdog_flatten_retry.py tests/quant_v2/test_credential_scrubbing.py tests/quant/test_telebot_main_v2_handlers.py tests/infra/test_docker_compose_services.py -q
```

Results:
- Focused flatten recovery suite: `29 passed in 3.70s`
- Broader Pass 1/2 safety suite: `151 passed in 4.74s`

Residual risk:
- Pass 2 recovery/idempotency rows `RWI1`-`RWI9` are verified. Live deployment remains blocked by later P0 gates in credential/logging hygiene, model trust, and deployment/build hardening.

### 2026-06-04 Pass 3 Telegram Credential Hygiene Slice

Ledger IDs changed:
- `CLH1` verified for removing default raw Telegram update logging.
- `CLH2` verified for central redaction and sanitized public error replies on audited sensitive paths.
- `CLH3` verified for approved-user-only `/setup` and credential clearing on revoke/ban.

Files changed:
- `quant/telebot/redaction.py`
- `quant/telebot/main.py`
- `quant/telebot/manager.py`
- `quant/telebot/engine.py`
- `tests/quant/telebot/test_security_hygiene.py`

Safety invariants proven:
- The catch-all `MessageHandler(filters.ALL, debug_log)` is not registered by default; optional debug mode logs only safe metadata.
- Redaction removes secret-like assignments, bot-token-shaped values, long credential-like values, and PEM blocks from diagnostic strings.
- Start, maintenance-continue, maintenance-prepare, legacy manager auth, and engine crash paths no longer reflect raw exception text to users.
- Pending and banned users cannot persist Binance credentials through `/setup`.
- Approved users can still save encrypted credentials.
- `/revoke` clears stored Binance credential fields, disables live mode, and marks the user inactive.

Verification commands:

```powershell
python -m pytest tests/quant/telebot/test_security_hygiene.py tests/quant/test_telebot_main_v2_handlers.py -q
python -m pytest tests/quant/telebot/test_security_hygiene.py tests/quant/test_telebot_main_v2_handlers.py tests/quant/telebot/test_hard_risk_pause.py tests/quant/telebot/test_main_lifetime_metrics.py tests/quant/telebot/test_main_model_rotation.py tests/quant_v2/test_binance_adapter.py tests/quant_v2/test_binance_client_phase4.py tests/quant_v2/test_execution_service.py tests/quant_v2/test_live_readiness.py tests/quant_v2/test_execution_infra.py tests/quant_v2/test_reconciliation.py tests/quant_v2/test_day2_infra_patches.py tests/quant_v2/test_session_lock.py tests/quant_v2/test_watchdog_flatten_retry.py tests/quant_v2/test_credential_scrubbing.py tests/infra/test_docker_compose_services.py -q
```

Results:
- Focused hygiene suite: `26 passed in 4.88s`
- Broader safety/hygiene suite: `167 passed in 6.83s`

Residual risk:
- Live deployment remains blocked. `CLH4`-`CLH7` still need diagnostic script quarantine, release artifact scanning, tracked archive cleanup/rotation docs, and deployment doc hardening. Model trust and deployment/build hardening P0 gates also remain open.

### 2026-06-04 Pass 3 Artifact and Diagnostic Scan Slice

Ledger IDs changed:
- `CLH4` verified through release quarantine: known unsafe diagnostic script names and unsafe diagnostic content patterns are rejected before deploy/release packaging.
- `CLH5` verified for non-extractive release artifact scanning.

Files changed:
- `tools/security/scan_artifacts.py`
- `tests/infra/test_release_artifact_scan.py`
- `PRODUCTION_REFACTOR_ROADMAP.md`

Safety invariants proven:
- Clean release trees pass the scanner.
- Secret/state paths are rejected, including `.env`, PEM/key files, SQLite/DB files, signal logs, logs, local diagnostic scripts, root archives, nested archive members, and model artifacts.
- Archive scanning inspects tar/zip member names without extracting archive contents.
- Unsafe diagnostic content patterns such as `API_KEY =`, `PASSWORD =`, Capital session headers, `hgetall(...)`, `SELECT *`, and raw response-body prints are rejected.
- Scanner findings report only paths and rule names; tests assert sentinel file contents are not printed in findings.

Verification commands:

```powershell
python -m pytest tests/infra/test_release_artifact_scan.py -q
python -m pytest tests/infra/test_release_artifact_scan.py tests/quant/telebot/test_security_hygiene.py tests/quant/test_telebot_main_v2_handlers.py tests/quant/telebot/test_hard_risk_pause.py tests/quant/telebot/test_main_lifetime_metrics.py tests/quant/telebot/test_main_model_rotation.py tests/quant_v2/test_binance_adapter.py tests/quant_v2/test_binance_client_phase4.py tests/quant_v2/test_execution_service.py tests/quant_v2/test_live_readiness.py tests/quant_v2/test_execution_infra.py tests/quant_v2/test_reconciliation.py tests/quant_v2/test_day2_infra_patches.py tests/quant_v2/test_session_lock.py tests/quant_v2/test_watchdog_flatten_retry.py tests/quant_v2/test_credential_scrubbing.py tests/infra/test_docker_compose_services.py -q
```

Results:
- Focused artifact scanner suite: `5 passed in 0.21s`
- Broader safety/hygiene suite: `172 passed in 6.28s`

Residual risk:
- Live deployment remains blocked. `CLH6` and `CLH7` still need tracked archive cleanup/rotation documentation and deployment doc hardening. Model trust and deployment/build hardening P0 gates also remain open.
- This slice quarantines unsafe diagnostics from release artifacts; it does not rewrite ignored local diagnostics into safe operational tools. Any diagnostic script intentionally retained for production use still needs a redacted-output implementation and focused tests.

### 2026-06-04 Pass 3 Archive Cleanup and Deployment Docs Slice

Ledger IDs changed:
- `CLH6` verified for removing the tracked deployment archive from future releases and documenting key/token rotation plus history-cleanup decision points.
- `CLH7` verified for deployment docs that reject dirty-tree upload, broad SSH, repo-root private key handling, and active infrastructure inventory.

Files changed:
- `DEPLOY.md`
- `AWS_DEPLOY.md`
- `docs/KEY_ROTATION_AND_ARCHIVE_CLEANUP.md`
- `tests/infra/test_deployment_hardening_docs.py`
- `PRODUCTION_REFACTOR_ROADMAP.md`
- Git index: `deploy_optimized.tar.gz` removed from tracking while the local ignored file remains on disk.

Safety invariants proven:
- `deploy_optimized.tar.gz` is no longer tracked by Git.
- Tracked env/key/archive/database paths matching the release blocklist are absent from `git ls-files`.
- Deployment docs no longer contain broad SSH guidance, recursive local tree upload, repo-root PEM guidance, `.env` upload wording, or active host IP inventory.
- Deployment docs require clean tracked source or a `git archive` release that passes `tools/security/scan_artifacts.py`.
- Runtime secrets are directed to host-local storage outside the repository checkout.
- The rotation runbook treats exposed archives as permanently exposed if pushed/shared, requires key rotation before relying on history cleanup, and keeps live deployment blocked until release scans and readiness checks pass.

Verification commands:

```powershell
git ls-files -- deploy_optimized.tar.gz
git ls-files | rg "\.(tar\.gz|tgz|zip|pem|key|db|sqlite|sqlite3)$|(^|/)\.env($|[./])"
python -m pytest tests/infra/test_release_artifact_scan.py tests/infra/test_deployment_hardening_docs.py tests/infra/test_docker_compose_services.py -q
python -m pytest tests/infra/test_release_artifact_scan.py tests/infra/test_deployment_hardening_docs.py tests/infra/test_docker_compose_services.py tests/quant/telebot/test_security_hygiene.py tests/quant/test_telebot_main_v2_handlers.py tests/quant/telebot/test_hard_risk_pause.py tests/quant/telebot/test_main_lifetime_metrics.py tests/quant/telebot/test_main_model_rotation.py tests/quant_v2/test_binance_adapter.py tests/quant_v2/test_binance_client_phase4.py tests/quant_v2/test_execution_service.py tests/quant_v2/test_live_readiness.py tests/quant_v2/test_execution_infra.py tests/quant_v2/test_reconciliation.py tests/quant_v2/test_day2_infra_patches.py tests/quant_v2/test_session_lock.py tests/quant_v2/test_watchdog_flatten_retry.py tests/quant_v2/test_credential_scrubbing.py -q
```

Results:
- `git ls-files -- deploy_optimized.tar.gz`: empty output
- Tracked release blocklist grep: empty output
- Focused infra/security suite: `11 passed in 0.31s`
- Broader safety/hygiene suite: `175 passed in 6.46s`

Residual risk:
- Pass 3 credential/logging/artifact hygiene rows `CLH1`-`CLH7` are verified. Live deployment remains blocked by model trust (`MAT*`), deployment/build hardening (`DBH*`), and CI/readiness gates (`CRG*`).
- Local ignored artifacts still exist in the working tree and should be deleted or moved after any needed evidence capture; they are no longer eligible for tracked-file releases and are rejected by the artifact scanner.

### 2026-06-04 Pass 4 Trusted Model Root Slice

Ledger IDs changed:
- `MAT1` verified for trusted-root enforcement before model registration, promotion/activation, runtime selection, and signal-manager/ensemble deserialization.

Files changed:
- `quant_v2/model_registry.py`
- `quant/telebot/model_selection.py`
- `quant_v2/telebot/signal_manager.py`
- `quant_v2/models/ensemble.py`
- `quant/telebot/main.py`
- `bootstrap_registry.py`
- `quant_v2/research/scheduled_retrain.py`
- `tests/quant_v2/test_model_registry.py`
- `tests/quant_v2/test_signal_manager.py`
- `tests/quant/test_model_selection.py`
- `PRODUCTION_REFACTOR_ROADMAP.md`

Safety invariants proven:
- `ModelRegistry` now has an explicit trusted model root and validates artifact dirs before registration, promotion eligibility checks, and active pointer writes.
- Out-of-root artifact dirs are rejected.
- Symlinked artifact dirs are rejected when the OS allows symlink creation in tests.
- Runtime model selection uses the registry's trusted-root validation for active pointers and filters latest-model discovery through non-symlink, in-root directories.
- `V2SignalManager` validates active artifact dirs and candidate model files before calling `load_model`.
- `HorizonEnsemble.from_directory` accepts a model-file validator, and the signal manager passes the registry validator through the ensemble path.
- A monkeypatched ensemble test proves a rejected model file does not reach `joblib.load`.

Verification commands:

```powershell
python -m pytest tests/quant_v2/test_model_registry.py tests/quant_v2/test_signal_manager.py tests/quant/test_model_selection.py -q
python -m pytest tests/quant/test_model_selection.py tests/quant_v2/test_model_registry.py tests/quant_v2/test_signal_manager.py tests/quant_v2/test_full_ensemble.py tests/quant_v2/test_scheduled_retrain_candidates.py tests/infra/test_release_artifact_scan.py tests/infra/test_deployment_hardening_docs.py tests/infra/test_docker_compose_services.py tests/quant/telebot/test_security_hygiene.py tests/quant/test_telebot_main_v2_handlers.py tests/quant_v2/test_binance_adapter.py tests/quant_v2/test_binance_client_phase4.py tests/quant_v2/test_execution_service.py tests/quant_v2/test_live_readiness.py tests/quant_v2/test_execution_infra.py tests/quant_v2/test_reconciliation.py tests/quant_v2/test_day2_infra_patches.py tests/quant_v2/test_session_lock.py tests/quant_v2/test_watchdog_flatten_retry.py tests/quant_v2/test_credential_scrubbing.py -q
```

Results:
- Focused model trust suite: `39 passed in 24.92s`
- Broader safety/model suite: `215 passed in 26.78s`

Residual risk:
- Later Pass 4 reports below supersede the original `MAT2`-`MAT10` residuals from this slice.
- Deployment/build hardening (`DBH*`) and CI/readiness gates (`CRG*`) remain open.

### 2026-06-04 Pass 4 Manifest and Promotion Gate Slice

Ledger IDs changed:
- `MAT2` verified for manifest/checksum enforcement on active model artifacts.
- `MAT3` verified for fail-closed promotion policy, explicit eligibility, required validation fields, complete horizons, and smoke-load checks.
- `MAT4` verified: invalid or missing active pointers now fail closed by default; latest-filesystem fallback requires explicit `BOT_MODEL_ALLOW_FILESYSTEM_FALLBACK=1`.

Files changed:
- `quant_v2/model_registry.py`
- `quant_v2/research/scheduled_retrain.py`
- `bootstrap_registry.py`
- `quant/telebot/model_selection.py`
- `quant_v2/telebot/signal_manager.py`
- `tests/quant_v2/test_model_registry.py`
- `tests/quant_v2/test_scheduled_retrain_candidates.py`
- `tests/quant_v2/test_signal_manager.py`
- `tests/quant/test_model_selection.py`
- `tests/quant/test_telebot_main_v2_handlers.py`
- `PRODUCTION_REFACTOR_ROADMAP.md`

Safety invariants proven:
- `model_manifest.json` records schema version, version ID, source, required horizons, file paths, file sizes, SHA-256 checksums, and metrics.
- Promotion rejects missing manifests, checksum drift, missing explicit `promotion_eligible is True`, missing validation scores, missing horizon metadata, and wrong loaded model horizon.
- Promotion smoke-loads each required horizon model and verifies `TrainedModel.horizon` matches the manifest entry before writing the active pointer.
- `set_active_version` can no longer activate a record whose artifact fails trusted-root or manifest validation.
- Scheduled retrain writes a manifest before registering candidates and only marks candidates promotion-eligible when all configured horizons pass.
- Bootstrap registration now routes through `promote_version` instead of direct active pointer writes, so legacy placeholder/latest artifacts cannot activate by default.
- Runtime model selection refuses latest-filesystem fallback when an active registry pointer exists but is invalid.
- Runtime model selection also refuses latest-filesystem fallback when no active pointer exists unless `BOT_MODEL_ALLOW_FILESYSTEM_FALLBACK=1` is set.
- Telegram admin promotion test now uses a manifest-backed, smoke-loadable artifact and still updates the active pointer successfully.

Verification commands:

```powershell
python -m pytest tests/quant_v2/test_model_registry.py tests/quant/test_model_selection.py tests/quant_v2/test_signal_manager.py tests/quant_v2/test_scheduled_retrain_candidates.py -q
python -m pytest tests/quant/test_model_selection.py tests/quant_v2/test_model_registry.py tests/quant_v2/test_signal_manager.py tests/quant_v2/test_full_ensemble.py tests/quant_v2/test_scheduled_retrain_candidates.py tests/infra/test_release_artifact_scan.py tests/infra/test_deployment_hardening_docs.py tests/infra/test_docker_compose_services.py tests/quant/telebot/test_security_hygiene.py tests/quant/test_telebot_main_v2_handlers.py tests/quant_v2/test_binance_adapter.py tests/quant_v2/test_binance_client_phase4.py tests/quant_v2/test_execution_service.py tests/quant_v2/test_live_readiness.py tests/quant_v2/test_execution_infra.py tests/quant_v2/test_reconciliation.py tests/quant_v2/test_day2_infra_patches.py tests/quant_v2/test_session_lock.py tests/quant_v2/test_watchdog_flatten_retry.py tests/quant_v2/test_credential_scrubbing.py -q
```

Results:
- Focused model manifest/promotion suite: `44 passed in 20.14s`
- Focused model-selection fallback suite: `6 passed in 1.94s`
- Broader safety/model suite: `220 passed in 17.46s`

Residual risk:
- Later Pass 4 reports below supersede the original `MAT5`-`MAT10` residuals from this slice.
- Deployment/build hardening (`DBH*`) and CI/readiness gates (`CRG*`) remain open.

### 2026-06-04 Pass 4 Activation Transaction and Partial Horizon Slice

Ledger IDs changed:
- `MAT5` verified for transaction-safer active pointer switching.
- `MAT6` verified for policy-checked direct activation and bootstrap activation.
- `MAT7` verified for production fail-closed partial-horizon behavior.

Files changed:
- `quant_v2/model_registry.py`
- `quant_v2/telebot/signal_manager.py`
- `tests/quant_v2/test_model_registry.py`
- `tests/quant_v2/test_signal_manager.py`
- `PRODUCTION_REFACTOR_ROADMAP.md`

Safety invariants proven:
- Activation validates and smoke-loads before changing active state.
- Version status metadata is written before the active pointer, and the active pointer is written last.
- If active-pointer write fails, old active pointer remains and version statuses are restored.
- If a status write fails before the active pointer write, old active pointer remains and version statuses are restored.
- `set_active_version` can no longer activate a manifest-valid but promotion-ineligible record.
- `rollback_to_previous_version` uses the same promotion/manifest validation path as other active switching.
- Bootstrap activation is already routed through `promote_version`, so legacy/latest artifacts cannot activate without passing production policy.
- `V2SignalManager` no longer falls back from a missing configured horizon to another horizon by default.
- Partial-horizon fallback is available only when `BOT_MODEL_ALLOW_PARTIAL_HORIZONS=1`.

Verification commands:

```powershell
python -m pytest tests/quant_v2/test_model_registry.py tests/quant_v2/test_signal_manager.py tests/quant/test_model_selection.py tests/quant/test_telebot_main_v2_handlers.py -q
python -m pytest tests/quant/test_model_selection.py tests/quant_v2/test_model_registry.py tests/quant_v2/test_signal_manager.py tests/quant_v2/test_full_ensemble.py tests/quant_v2/test_scheduled_retrain_candidates.py tests/infra/test_release_artifact_scan.py tests/infra/test_deployment_hardening_docs.py tests/infra/test_docker_compose_services.py tests/quant/telebot/test_security_hygiene.py tests/quant/test_telebot_main_v2_handlers.py tests/quant_v2/test_binance_adapter.py tests/quant_v2/test_binance_client_phase4.py tests/quant_v2/test_execution_service.py tests/quant_v2/test_live_readiness.py tests/quant_v2/test_execution_infra.py tests/quant_v2/test_reconciliation.py tests/quant_v2/test_day2_infra_patches.py tests/quant_v2/test_session_lock.py tests/quant_v2/test_watchdog_flatten_retry.py tests/quant_v2/test_credential_scrubbing.py -q
```

Results:
- Focused activation/partial-horizon suite: `70 passed in 13.86s`
- Broader safety/model suite: `225 passed in 24.31s`

Residual risk:
- Later Pass 4 reports below supersede the original `MAT8`-`MAT10` residuals from this slice.
- Deployment/build hardening (`DBH*`) and CI/readiness gates (`CRG*`) remain open.
- `set_active_version` remains as a policy-checked compatibility method; a future cleanup can rename it, but the production bypass behavior is closed by tests.

### 2026-06-04 Pass 4 Retrain Boolean Parsing Slice

Ledger IDs changed:
- `MAT8` verified for fail-closed retrain boolean parsing.

Files changed:
- `quant_v2/research/scheduled_retrain.py`
- `tests/quant_v2/test_scheduled_retrain_candidates.py`
- `PRODUCTION_REFACTOR_ROADMAP.md`

Safety invariants proven:
- `_env_flag` now enables a flag only for explicit true values: `1`, `true`, `yes`, or `on`.
- Explicit false values remain disabled: `0`, `false`, `no`, or `off`.
- Unknown values log a warning and use the caller's safe default.
- `BOT_RETRAIN_AUTO_PROMOTE=maybe` registers a candidate but does not promote it.

Verification commands:

```powershell
python -m pytest tests/quant_v2/test_scheduled_retrain_candidates.py -q
python -m pytest tests/quant/test_model_selection.py tests/quant_v2/test_model_registry.py tests/quant_v2/test_signal_manager.py tests/quant_v2/test_full_ensemble.py tests/quant_v2/test_scheduled_retrain_candidates.py tests/infra/test_release_artifact_scan.py tests/infra/test_deployment_hardening_docs.py tests/infra/test_docker_compose_services.py tests/quant/telebot/test_security_hygiene.py tests/quant/test_telebot_main_v2_handlers.py tests/quant_v2/test_binance_adapter.py tests/quant_v2/test_binance_client_phase4.py tests/quant_v2/test_execution_service.py tests/quant_v2/test_live_readiness.py tests/quant_v2/test_execution_infra.py tests/quant_v2/test_reconciliation.py tests/quant_v2/test_day2_infra_patches.py tests/quant_v2/test_session_lock.py tests/quant_v2/test_watchdog_flatten_retry.py tests/quant_v2/test_credential_scrubbing.py -q
```

Results:
- Focused retrain candidate suite: `3 passed in 4.09s`
- Broader safety/model suite: `226 passed in 27.14s`

Residual risk:
- Later Pass 4 reports below supersede the original `MAT9`-`MAT10` residuals from this slice.
- Deployment/build hardening (`DBH*`) and CI/readiness gates (`CRG*`) remain open.

### 2026-06-04 Pass 4 Legacy Label and Model Mount Slice

Ledger IDs changed:
- `MAT9` verified for preserving unlabeled tail rows as `NaN` in the legacy retrain pipeline.
- `MAT10` verified for splitting runtime read-only model artifacts from writable registry/model staging paths.

Files changed:
- `quant_v2/research/retrain_pipeline.py`
- `tests/quant_v2/test_retrain_pipeline.py`
- `docker-compose.yml`
- `docker-compose.prod.yml`
- `tests/infra/test_docker_compose_services.py`
- `PRODUCTION_REFACTOR_ROADMAP.md`

Safety invariants proven:
- Legacy retrain labels no longer turn the final unlabeled horizon rows into class `0`.
- Runtime app services do not mount broad writable `/app/models`.
- Runtime production artifacts mount at `/app/models/production:ro`.
- Runtime registry state is a separate writable mount at `/app/model_registry`.
- The retrain scheduler is the only default service with writable production model artifacts.
- Prod compose uses the same read-only runtime model artifact and separate registry-state mount pattern.

Verification commands:

```powershell
python -m pytest tests/quant_v2/test_retrain_pipeline.py -q
python -m pytest tests/infra/test_docker_compose_services.py -q
python -m pytest tests/quant_v2/test_retrain_pipeline.py tests/infra/test_docker_compose_services.py tests/quant/test_model_selection.py tests/quant_v2/test_model_registry.py tests/quant_v2/test_signal_manager.py tests/quant_v2/test_full_ensemble.py tests/quant_v2/test_scheduled_retrain_candidates.py tests/infra/test_release_artifact_scan.py tests/infra/test_deployment_hardening_docs.py tests/quant/telebot/test_security_hygiene.py tests/quant/test_telebot_main_v2_handlers.py tests/quant_v2/test_binance_adapter.py tests/quant_v2/test_binance_client_phase4.py tests/quant_v2/test_execution_service.py tests/quant_v2/test_live_readiness.py tests/quant_v2/test_execution_infra.py tests/quant_v2/test_reconciliation.py tests/quant_v2/test_day2_infra_patches.py tests/quant_v2/test_session_lock.py tests/quant_v2/test_watchdog_flatten_retry.py tests/quant_v2/test_credential_scrubbing.py -q
```

Results:
- Focused legacy retrain suite: `1 passed in 2.85s`
- Focused compose model-mount suite: `6 passed in 0.17s`
- Broader safety/model/deploy suite: `230 passed in 23.71s`

Residual risk:
- Pass 4 model trust rows `MAT1`-`MAT10` are verified. Live deployment remains blocked by deployment/build hardening (`DBH*`) and CI/readiness gates (`CRG*`).

### 2026-06-04 Pass 5 Redis Privacy and Compose Hardening Slice

Ledger IDs changed:
- `DBH2` verified for private default Redis exposure.
- `DBH3` verified for compose runtime hardening basics.

Files changed:
- `docker-compose.yml`
- `docker-compose.prod.yml`
- `tests/infra/test_docker_compose_services.py`
- `PRODUCTION_REFACTOR_ROADMAP.md`

Safety invariants proven:
- Default Redis no longer publishes host port `6379`.
- Redis has a healthcheck.
- Default app services run as the non-root `quantbot` user.
- Default app services have `read_only: true`, `cap_drop: ["ALL"]`, `security_opt: ["no-new-privileges:true"]`, tmpfs mounts, healthchecks, CPU limits, and memory limits.
- Prod app service has the same hardening basics.
- Existing model trust mount assertions still pass: runtime model artifacts are read-only and registry state is mounted separately.

Verification commands:

```powershell
python -m pytest tests/infra/test_docker_compose_services.py -q
python -m pytest tests/quant_v2/test_retrain_pipeline.py tests/infra/test_docker_compose_services.py tests/quant/test_model_selection.py tests/quant_v2/test_model_registry.py tests/quant_v2/test_signal_manager.py tests/quant_v2/test_full_ensemble.py tests/quant_v2/test_scheduled_retrain_candidates.py tests/infra/test_release_artifact_scan.py tests/infra/test_deployment_hardening_docs.py tests/quant/telebot/test_security_hygiene.py tests/quant/test_telebot_main_v2_handlers.py tests/quant_v2/test_binance_adapter.py tests/quant_v2/test_binance_client_phase4.py tests/quant_v2/test_execution_service.py tests/quant_v2/test_live_readiness.py tests/quant_v2/test_execution_infra.py tests/quant_v2/test_reconciliation.py tests/quant_v2/test_day2_infra_patches.py tests/quant_v2/test_session_lock.py tests/quant_v2/test_watchdog_flatten_retry.py tests/quant_v2/test_credential_scrubbing.py -q
```

Results:
- Focused compose hardening suite: `9 passed in 0.37s`
- Broader safety/model/deploy suite: `233 passed in 18.86s`

Residual risk:
- Later Pass 5 reports below supersede the original `DBH1`/`DBH4`/`DBH5` residuals from this slice.
- Later Pass 5 reports below supersede the dependency/base-image locking residual from this slice.
- Live deployment remains blocked by CI/readiness gates (`CRG*`).

### 2026-06-04 Pass 5 Production Topology and Root Bind Cleanup Slice

Ledger IDs changed:
- `L8` verified for full compose/prod topology assertions.
- `DBH1` verified for explicit production topology.
- `DBH4` verified for root DB/log bind cleanup and separated writable state.
- `DBH5` verified for replacing singleton prod compose with the shared service topology.
- `DBH6` verified for deploy script target alignment with the shared production container name.

Files changed:
- `docker-compose.yml`
- `docker-compose.prod.yml`
- `deploy/flatten_on_shutdown.sh`
- `deploy/dead_mans_switch.sh`
- `tests/infra/test_docker_compose_services.py`
- `PRODUCTION_REFACTOR_ROADMAP.md`

Safety invariants proven:
- Prod compose no longer defines singleton `quant_bot`.
- Default and prod compose both define exactly `redis`, `telegram_bot`, and `retrain_scheduler`.
- Prod `telegram_bot` uses explicit `python -m quant.telebot.main`, not Dockerfile `CMD`.
- Prod live execution defaults fail closed with `BOT_V2_ALLOW_LIVE_EXECUTION=${BOT_V2_ALLOW_LIVE_EXECUTION:-0}`.
- Default and prod compose reject root `./quant_bot.db` and `./signal_log.json` binds.
- Default and prod compose reject `/app/quant_bot.db` and `/app/signal_log.json` container targets.
- Runtime state is under `/state`, registry state is under `/app/model_registry`, and runtime model artifacts remain read-only.
- Shutdown/dead-man comments now match the shared `quant_telegram` production target.
- Deploy script tests prove stale `quant_execution` is absent and `QUANT_CONTAINER_NAME` defaults reference `quant_telegram`.

Verification commands:

```powershell
python -m pytest tests/infra/test_docker_compose_services.py tests/infra/test_deployment_hardening_docs.py -q
python -m pytest tests/quant_v2/test_retrain_pipeline.py tests/infra/test_docker_compose_services.py tests/infra/test_deployment_hardening_docs.py tests/quant/test_model_selection.py tests/quant_v2/test_model_registry.py tests/quant_v2/test_signal_manager.py tests/quant_v2/test_full_ensemble.py tests/quant_v2/test_scheduled_retrain_candidates.py tests/infra/test_release_artifact_scan.py tests/quant/telebot/test_security_hygiene.py tests/quant/test_telebot_main_v2_handlers.py tests/quant_v2/test_binance_adapter.py tests/quant_v2/test_binance_client_phase4.py tests/quant_v2/test_execution_service.py tests/quant_v2/test_live_readiness.py tests/quant_v2/test_execution_infra.py tests/quant_v2/test_reconciliation.py tests/quant_v2/test_day2_infra_patches.py tests/quant_v2/test_session_lock.py tests/quant_v2/test_watchdog_flatten_retry.py tests/quant_v2/test_credential_scrubbing.py -q
```

Results:
- Focused infra/docs suite: `14 passed in 0.40s`
- Broader safety/deploy suite: `235 passed in 31.57s`

Residual risk:
- Live deployment remains blocked by CI/readiness gates (`CRG*`).

### 2026-06-04 Pass 5 Dependency Locking and Image Pinning Slice

Ledger IDs changed:
- `DBH7` verified for hashed production dependency locking.
- `DBH8` verified for digest-pinned base/Redis images and removal of direct unpinned Dockerfile package installs.

Files changed:
- `Dockerfile`
- `docker-compose.yml`
- `docker-compose.prod.yml`
- `pyproject.toml`
- `requirements.in`
- `requirements.lock`
- `DEPLOY.md`
- `AWS_DEPLOY.md`
- `tests/infra/test_docker_compose_services.py`
- `PRODUCTION_REFACTOR_ROADMAP.md`

Safety invariants proven:
- Dockerfile copies `requirements.lock` and installs runtime Python dependencies with `pip install --require-hashes`.
- Dockerfile no longer performs direct unpinned `pip install numpy pandas ...` style installs or uses the PyTorch extra index in the default production image.
- `requirements.lock` contains pinned `==` requirements and SHA-256 hashes.
- Runtime dependencies listed in `pyproject.toml` must be represented in `requirements.in`, and `requirements.in` sources must be represented in `requirements.lock`.
- The Python builder and runtime stages are pinned to `python:3.11-slim@sha256:a3ab0b966bc4e91546a033e22093cb840908979487a9fc0e6e38295747e49ac0`.
- Default and prod Redis are pinned to `redis:7-alpine@sha256:6ab0b6e7381779332f97b8ca76193e45b0756f38d4c0dcda72dbb3c32061ab99`.
- Torch now uses a reviewed CPU-only wheel URL in `requirements.in`; do not replace it with the default PyPI Torch package, which can pull a CUDA dependency stack on Linux. Chronos package enablement remains blocked until its compatible dependency path passes audit without the current `transformers<5` advisory exposure.
- Deploy docs require hash-lock checks and warn against direct Dockerfile package installs.

Verification commands:

```powershell
python -m pytest tests/infra/test_docker_compose_services.py tests/infra/test_deployment_hardening_docs.py -q
python -m pip install --dry-run --require-hashes -r requirements.lock
```

Results:
- Focused infra/docs hardening suite: `17 passed in 0.60s`
- Hash-lock dry run: passed for `requirements.lock`

Additional notes:
- Docker CLI is not installed in this environment, so `docker compose config` and a full `docker build` could not be run locally.
- Later readiness work added a Linux CPython 3.11 wheel/hash availability check and corrected an incompatible local `greenlet` lock candidate to `greenlet==3.2.5`; that check now passes and is part of the CRG3 readiness wrapper.

Residual risk:
- Live deployment remains blocked by CI/readiness gates (`CRG*`), including production readiness workflow, local readiness wrapper, dependency audit, SBOM, and final Docker build evidence.

### 2026-06-04 Pass 6 Local Production Readiness Wrapper Slice

Ledger IDs changed:
- `CRG4` verified for the local production readiness wrapper.
- `CRG7` moved to in progress; readiness commands are documented, while dependency audit, SBOM, and CI wiring remain open.

Files changed:
- `tools/security/production_readiness.py`
- `docs/PRODUCTION_READINESS.md`
- `DEPLOY.md`
- `AWS_DEPLOY.md`
- `tests/infra/test_production_readiness.py`
- `PRODUCTION_REFACTOR_ROADMAP.md`

Safety invariants proven:
- `python tools/security/production_readiness.py --profile live` is the canonical live-readiness command.
- Live profile requires a clean working tree and does not allow `--skip-docker`.
- The wrapper includes focused P0 regression tests, hash-lock dry run, default/prod compose config checks, production Docker build, and tracked release artifact scan.
- Developer-only local dry run can print the evidence commands in environments without Docker, but clearly states that skip-based output is not live deployment evidence.
- Deployment docs point operators to the readiness wrapper before live enablement.

Verification commands:

```powershell
python -m pytest tests/infra/test_production_readiness.py tests/infra/test_docker_compose_services.py tests/infra/test_deployment_hardening_docs.py -q
python tools/security/production_readiness.py --profile local --skip-clean-check --skip-docker --dry-run
python -m pytest tests/quant_v2/test_retrain_pipeline.py tests/infra/test_docker_compose_services.py tests/infra/test_deployment_hardening_docs.py tests/infra/test_production_readiness.py tests/quant/test_model_selection.py tests/quant_v2/test_model_registry.py tests/quant_v2/test_signal_manager.py tests/quant_v2/test_full_ensemble.py tests/quant_v2/test_scheduled_retrain_candidates.py tests/infra/test_release_artifact_scan.py tests/quant/telebot/test_security_hygiene.py tests/quant/test_telebot_main_v2_handlers.py tests/quant_v2/test_binance_adapter.py tests/quant_v2/test_binance_client_phase4.py tests/quant_v2/test_execution_service.py tests/quant_v2/test_live_readiness.py tests/quant_v2/test_execution_infra.py tests/quant_v2/test_reconciliation.py tests/quant_v2/test_day2_infra_patches.py tests/quant_v2/test_session_lock.py tests/quant_v2/test_watchdog_flatten_retry.py tests/quant_v2/test_credential_scrubbing.py -q
```

Results:
- Focused readiness/infra/docs suite: `19 passed in 1.07s`
- Local readiness dry run: passed and printed deploy-blocking command plan
- Broader focused P0 suite: `240 passed, 4 warnings in 25.48s`

Additional notes:
- Docker CLI is not installed in this environment; live profile is intentionally not proven here because Docker compose config and image build evidence require Docker.

Residual risk:
- Live deployment remains blocked by remaining CI/readiness gates, especially `CRG3` CI workflow, `CRG7` dependency-audit/SBOM coverage, and final Docker build evidence from a Docker-capable clean checkout.

### 2026-06-04 Pass 6 CI Audit and SBOM Slice

Ledger IDs changed:
- `DBH11` verified for dependency audit and SBOM hooks.
- `CRG7` verified for documented "do not deploy live unless these pass" commands.
- `CRG3` moved to in progress; workflow is added and statically tested, but an actual GitHub/Docker-capable run remains required before live enablement.

Files changed:
- `.github/workflows/production-readiness.yml`
- `tools/security/production_readiness.py`
- `tools/security/generate_sbom.py`
- `docs/PRODUCTION_READINESS.md`
- `DEPLOY.md`
- `AWS_DEPLOY.md`
- `requirements.in`
- `requirements.lock`
- `pyproject.toml`
- `tests/infra/test_production_readiness.py`
- `PRODUCTION_REFACTOR_ROADMAP.md`

Safety invariants proven:
- The readiness wrapper includes dependency vulnerability audit with `pip-audit`.
- The readiness wrapper generates a CycloneDX SBOM at `build/security/sbom.cdx.json`.
- Live profile cannot skip dependency audit or Docker checks.
- The production readiness workflow installs locked runtime dependencies, installs pinned CI tools, runs `python tools/security/production_readiness.py --profile live`, and uploads the SBOM artifact.
- Static infra tests verify the workflow keeps the live wrapper, audit tool, Docker compose/build path, release scan, and SBOM artifact wiring.
- Initial dependency audit found vulnerabilities in `cryptography`, `idna`, `pyarrow`, `python-dotenv`, `requests`, and `urllib3`; the lock was bumped to fixed versions and audit now reports no known vulnerabilities.

Patched dependency versions:
- `cryptography==48.0.0`
- `idna==3.18`
- `pyarrow==24.0.0`
- `python-dotenv==1.2.2`
- `requests==2.34.2`
- `urllib3==2.7.0`

Verification commands:

```powershell
python -m pytest tests/infra/test_production_readiness.py tests/infra/test_docker_compose_services.py tests/infra/test_deployment_hardening_docs.py -q
python -m pip install --dry-run --require-hashes -r requirements.lock
python -m pip_audit -r requirements.lock --progress-spinner off
python tools/security/generate_sbom.py --requirements requirements.lock --output build/security/sbom.cdx.json
python -m pytest tests/quant_v2/test_retrain_pipeline.py tests/infra/test_docker_compose_services.py tests/infra/test_deployment_hardening_docs.py tests/infra/test_production_readiness.py tests/quant/test_model_selection.py tests/quant_v2/test_model_registry.py tests/quant_v2/test_signal_manager.py tests/quant_v2/test_full_ensemble.py tests/quant_v2/test_scheduled_retrain_candidates.py tests/infra/test_release_artifact_scan.py tests/quant/telebot/test_security_hygiene.py tests/quant/test_telebot_main_v2_handlers.py tests/quant_v2/test_binance_adapter.py tests/quant_v2/test_binance_client_phase4.py tests/quant_v2/test_execution_service.py tests/quant_v2/test_live_readiness.py tests/quant_v2/test_execution_infra.py tests/quant_v2/test_reconciliation.py tests/quant_v2/test_day2_infra_patches.py tests/quant_v2/test_session_lock.py tests/quant_v2/test_watchdog_flatten_retry.py tests/quant_v2/test_credential_scrubbing.py -q
```

Results:
- Focused readiness/infra/docs suite: `22 passed in 1.79s`
- Hash-lock dry run: passed
- Dependency audit: `No known vulnerabilities found`
- SBOM generation: `wrote build\security\sbom.cdx.json with 35 components`
- Broader focused P0 suite: `243 passed, 4 warnings in 29.82s`

Additional notes:
- Docker CLI is not installed in this local environment, so the workflow and wrapper are statically tested here but the final live-readiness proof still needs `python tools/security/production_readiness.py --profile live` on a clean Docker-capable checkout or a successful GitHub Actions run.

Residual risk:
- Live deployment remains blocked by `CRG3` until the production readiness workflow runs successfully with Docker compose config and production image build evidence.

### 2026-06-04 Pass 6 Ledger and Live Checklist Slice

Ledger IDs changed:
- `CRG1` verified for focused P0 regression suite coverage.
- `CRG2` verified for inverted unsafe historical tests.
- `CRG5` verified for automated P0 ledger validation.
- `CRG6` verified for live enablement checklist coverage.

Files changed:
- `tools/security/check_roadmap.py`
- `tools/security/production_readiness.py`
- `docs/PRODUCTION_READINESS.md`
- `DEPLOY.md`
- `tests/infra/test_production_readiness.py`
- `tests/infra/test_deployment_hardening_docs.py`
- `PRODUCTION_REFACTOR_ROADMAP.md`

Safety invariants proven:
- Readiness now includes a roadmap ledger validator.
- P0 ledger rows must be dated `Verified YYYY-MM-DD` and carry non-placeholder files/proof before live readiness can pass.
- The current roadmap validator blocks live readiness only on `CRG3`, the remaining Docker-capable CI evidence gate.
- The GitHub Actions workflow uses the `ci` readiness profile, which allows only `CRG3` to remain open so the workflow can bootstrap its own evidence; operator live readiness still uses the strict `live` profile.
- The focused P0 suite list includes all deployment-blocking test files, including readiness and infra checks.
- Static tests prove historical unsafe expectations are inverted: no post-only market fallback, filesystem fallback requires explicit flag, placeholder/corrupt/ineligible model promotion is rejected, and partial-horizon fallback is blocked by default.
- Live enablement docs require P0 verification, production readiness wrapper, CI workflow, private/authenticated Redis, go/no-go true, rollback clear, valid manifests/checksums, fresh marks, restricted operator access, rehearsed rollback, and key/token rotation.

Verification commands:

```powershell
python -m pytest tests/infra/test_production_readiness.py tests/infra/test_deployment_hardening_docs.py tests/infra/test_docker_compose_services.py -q
python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md
python -m pytest tests/quant_v2/test_retrain_pipeline.py tests/infra/test_docker_compose_services.py tests/infra/test_deployment_hardening_docs.py tests/infra/test_production_readiness.py tests/quant/test_model_selection.py tests/quant_v2/test_model_registry.py tests/quant_v2/test_signal_manager.py tests/quant_v2/test_full_ensemble.py tests/quant_v2/test_scheduled_retrain_candidates.py tests/infra/test_release_artifact_scan.py tests/quant/telebot/test_security_hygiene.py tests/quant/test_telebot_main_v2_handlers.py tests/quant_v2/test_binance_adapter.py tests/quant_v2/test_binance_client_phase4.py tests/quant_v2/test_execution_service.py tests/quant_v2/test_live_readiness.py tests/quant_v2/test_execution_infra.py tests/quant_v2/test_reconciliation.py tests/quant_v2/test_day2_infra_patches.py tests/quant_v2/test_session_lock.py tests/quant_v2/test_watchdog_flatten_retry.py tests/quant_v2/test_credential_scrubbing.py -q
```

Results:
- Focused readiness/docs/compose suite: `27 passed in 2.03s`
- Roadmap validator: failed only on `CRG3`, intentionally keeping live deployment blocked until Docker-capable CI evidence exists.
- Broader focused P0 suite after CI-profile self-bootstrap fix: `249 passed, 4 warnings in 24.67s`
- Roadmap validator with CI bootstrap allowance: passed with `--allow-open-id CRG3`

Residual risk:
- Live deployment remains blocked solely by `CRG3`: the production readiness workflow must run successfully in a clean Docker-capable environment and prove compose config plus production image build.

### 2026-06-04 Pass 6 CRG3 Evidence Handoff Slice

Ledger IDs changed:
- `CRG3` remains in progress, but the final evidence procedure is now documented and test-covered.

Files changed:
- `docs/CRG3_CI_EVIDENCE.md`
- `docs/PRODUCTION_READINESS.md`
- `tests/infra/test_production_readiness.py`
- `PRODUCTION_REFACTOR_ROADMAP.md`

Safety invariants proven:
- The CRG3 handoff doc requires a successful `Production Readiness` GitHub Actions run.
- Required evidence includes CI readiness profile, default/prod Compose config, production Docker build, dependency audit, SBOM artifact, commit SHA/run URL, and final strict live readiness.
- The final live-enable step still requires `python tools/security/production_readiness.py --profile live`; CI bootstrap evidence alone is not enough to enable live trading.
- Tests assert the evidence doc names the required Docker, audit, SBOM, roadmap, and strict-live proof items.

Verification commands:

```powershell
python -m pytest tests/infra/test_production_readiness.py tests/infra/test_deployment_hardening_docs.py tests/infra/test_docker_compose_services.py -q
python -m pytest tests/quant_v2/test_retrain_pipeline.py tests/infra/test_docker_compose_services.py tests/infra/test_deployment_hardening_docs.py tests/infra/test_production_readiness.py tests/quant/test_model_selection.py tests/quant_v2/test_model_registry.py tests/quant_v2/test_signal_manager.py tests/quant_v2/test_full_ensemble.py tests/quant_v2/test_scheduled_retrain_candidates.py tests/infra/test_release_artifact_scan.py tests/quant/telebot/test_security_hygiene.py tests/quant/test_telebot_main_v2_handlers.py tests/quant_v2/test_binance_adapter.py tests/quant_v2/test_binance_client_phase4.py tests/quant_v2/test_execution_service.py tests/quant_v2/test_live_readiness.py tests/quant_v2/test_execution_infra.py tests/quant_v2/test_reconciliation.py tests/quant_v2/test_day2_infra_patches.py tests/quant_v2/test_session_lock.py tests/quant_v2/test_watchdog_flatten_retry.py tests/quant_v2/test_credential_scrubbing.py -q
```

Results:
- Focused readiness/docs/compose suite: `29 passed in 2.13s`
- Broader focused P0 suite: `250 passed, 4 warnings in 33.61s`

Residual risk:
- Live deployment remains blocked by `CRG3` until the documented GitHub Actions run succeeds and the roadmap row is updated with run evidence.

### 2026-06-04 Pass 6 Release Packaging and Infra Regression Slice

Ledger IDs changed:
- `DBH9` verified for scanned release packaging.
- `DBH10` verified for hardened deployment runbooks.
- `CRG8` verified for artifact, docs, compose, and readiness infra regression coverage.

Files changed:
- `tools/security/build_release.py`
- `tools/security/production_readiness.py`
- `DEPLOY.md`
- `AWS_DEPLOY.md`
- `docs/PRODUCTION_READINESS.md`
- `docs/CRG3_CI_EVIDENCE.md`
- `tests/infra/test_release_artifact_scan.py`
- `tests/infra/test_deployment_hardening_docs.py`
- `tests/infra/test_production_readiness.py`
- `PRODUCTION_REFACTOR_ROADMAP.md`

Safety invariants proven:
- Release packaging uses `git archive` from tracked files only.
- Release builder refuses dirty trees by default.
- Release builder scans the output archive and removes it if secret/state/model/archive findings are present.
- Readiness now includes scanned release artifact packaging, not just an ad hoc artifact scan.
- Deployment docs point operators to the release builder and continue to reject broad SSH, recursive local upload, repo-root PEM, `.env` upload, and concrete host inventory patterns.
- Infra tests cover release artifact scanning, deployment doc safety, compose hardening, digest-pinned images, readiness workflow wiring, and CRG3 evidence capture.

Verification commands:

```powershell
python -m pytest tests/infra/test_release_artifact_scan.py -q
python -m pytest tests/infra/test_production_readiness.py tests/infra/test_deployment_hardening_docs.py tests/infra/test_docker_compose_services.py -q
python tools/security/production_readiness.py --profile local --skip-clean-check --skip-docker --skip-audit --dry-run
python -m pytest tests/quant_v2/test_retrain_pipeline.py tests/infra/test_docker_compose_services.py tests/infra/test_deployment_hardening_docs.py tests/infra/test_production_readiness.py tests/infra/test_release_artifact_scan.py tests/quant/test_model_selection.py tests/quant_v2/test_model_registry.py tests/quant_v2/test_signal_manager.py tests/quant_v2/test_full_ensemble.py tests/quant_v2/test_scheduled_retrain_candidates.py tests/quant/telebot/test_security_hygiene.py tests/quant/test_telebot_main_v2_handlers.py tests/quant_v2/test_binance_adapter.py tests/quant_v2/test_binance_client_phase4.py tests/quant_v2/test_execution_service.py tests/quant_v2/test_live_readiness.py tests/quant_v2/test_execution_infra.py tests/quant_v2/test_reconciliation.py tests/quant_v2/test_day2_infra_patches.py tests/quant_v2/test_session_lock.py tests/quant_v2/test_watchdog_flatten_retry.py tests/quant_v2/test_credential_scrubbing.py -q
```

Results:
- Focused release scan suite: `8 passed in 0.34s`
- Focused readiness/docs/compose suite: `29 passed in 2.80s`
- Local readiness dry run includes scanned release artifact packaging.
- Broader focused P0 suite: `253 passed, 4 warnings in 20.34s`

Residual risk:
- Live deployment remains blocked by `CRG3` until the documented GitHub Actions run succeeds and strict live readiness passes.

### 2026-06-04 Pass 6 Report Template and Deferral Controls Slice

Ledger IDs changed:
- `CRG9` verified for implementation report template coverage.
- `CRG10` verified for deferred-hardening rationale and non-deferrable P0 guardrails.

Files changed:
- `docs/IMPLEMENTATION_REPORT_TEMPLATE.md`
- `docs/DEFERRED_HARDENING.md`
- `tests/infra/test_production_readiness.py`
- `PRODUCTION_REFACTOR_ROADMAP.md`

Safety invariants proven:
- Future implementation reports have a required structure covering ledger IDs, files, safety invariants, tests/checks, commands, results, rollback, residual risk, and live-block status.
- External evidence must include run URL or run ID, commit SHA, and artifacts when relevant.
- Deferred hardening explicitly cannot include live defaults, market fallback prevention, missing-mark handling, WAL/idempotency, Redis command auth, credential redaction, model trust, dependency locking, or production readiness evidence.
- Allowed deferrals carry rationale and guardrails.

Verification commands:

```powershell
python -m pytest tests/infra/test_production_readiness.py -q
python -m pytest tests/quant_v2/test_retrain_pipeline.py tests/infra/test_docker_compose_services.py tests/infra/test_deployment_hardening_docs.py tests/infra/test_production_readiness.py tests/infra/test_release_artifact_scan.py tests/quant/test_model_selection.py tests/quant_v2/test_model_registry.py tests/quant_v2/test_signal_manager.py tests/quant_v2/test_full_ensemble.py tests/quant_v2/test_scheduled_retrain_candidates.py tests/quant/telebot/test_security_hygiene.py tests/quant/test_telebot_main_v2_handlers.py tests/quant_v2/test_binance_adapter.py tests/quant_v2/test_binance_client_phase4.py tests/quant_v2/test_execution_service.py tests/quant_v2/test_live_readiness.py tests/quant_v2/test_execution_infra.py tests/quant_v2/test_reconciliation.py tests/quant_v2/test_day2_infra_patches.py tests/quant_v2/test_session_lock.py tests/quant_v2/test_watchdog_flatten_retry.py tests/quant_v2/test_credential_scrubbing.py -q
```

Results:
- Focused readiness docs/control suite: `13 passed in 1.41s`
- Broader focused P0 suite: `255 passed, 4 warnings in 30.39s`

Residual risk:
- Live deployment remains blocked by `CRG3` until the documented GitHub Actions run succeeds and strict live readiness passes.

### 2026-06-04 Pass 6 CRG3 Finalizer Slice

Ledger IDs changed:
- `CRG3` remains in progress; final evidence update is now scripted and test-covered.

Files changed:
- `tools/security/finalize_crg3.py`
- `docs/CRG3_CI_EVIDENCE.md`
- `tests/infra/test_production_readiness.py`
- `PRODUCTION_REFACTOR_ROADMAP.md`

Safety invariants proven:
- CRG3 cannot be finalized without a GitHub Actions run URL, 40-character commit SHA, and SBOM artifact name.
- Valid CRG3 evidence updates the roadmap row to dated `Verified`, records run evidence, and appends an implementation report.
- The finalizer output roadmap passes the P0 ledger validator in tests.

Verification commands:

```powershell
python -m pytest tests/infra/test_production_readiness.py -q
python -m pytest tests/quant_v2/test_retrain_pipeline.py tests/infra/test_docker_compose_services.py tests/infra/test_deployment_hardening_docs.py tests/infra/test_production_readiness.py tests/infra/test_release_artifact_scan.py tests/quant/test_model_selection.py tests/quant_v2/test_model_registry.py tests/quant_v2/test_signal_manager.py tests/quant_v2/test_full_ensemble.py tests/quant_v2/test_scheduled_retrain_candidates.py tests/quant/telebot/test_security_hygiene.py tests/quant/test_telebot_main_v2_handlers.py tests/quant_v2/test_binance_adapter.py tests/quant_v2/test_binance_client_phase4.py tests/quant_v2/test_execution_service.py tests/quant_v2/test_live_readiness.py tests/quant_v2/test_execution_infra.py tests/quant_v2/test_reconciliation.py tests/quant_v2/test_day2_infra_patches.py tests/quant_v2/test_session_lock.py tests/quant_v2/test_watchdog_flatten_retry.py tests/quant_v2/test_credential_scrubbing.py -q
```

Results:
- Focused readiness/finalizer suite: `15 passed in 1.89s`
- Broader focused P0 suite: `257 passed, 4 warnings in 27.09s`

Residual risk:
- Live deployment remains blocked by `CRG3` until the documented GitHub Actions run succeeds, `tools/security/finalize_crg3.py` records evidence, and strict live readiness passes.

### 2026-06-04 Pass 6 Python 3.11 Lock Compatibility Slice

Ledger IDs changed:
- `DBH7` evidence tightened for CI-target Python 3.11 dependency compatibility.
- `CRG3` evidence requirements expanded to include Linux CPython 3.11 wheel/hash availability.

Files changed:
- `requirements.in`
- `requirements.lock`
- `tools/security/production_readiness.py`
- `tests/infra/test_production_readiness.py`
- `docs/PRODUCTION_READINESS.md`
- `docs/CRG3_CI_EVIDENCE.md`
- `PRODUCTION_REFACTOR_ROADMAP.md`

Safety invariants proven:
- The hash lock is not only valid on the local Python 3.14 workstation; it also resolves downloadable binary wheels for Linux CPython 3.11 using the relevant manylinux tags.
- The readiness wrapper now runs the full pytest suite in addition to the focused P0 suite, matching the CRG3 workflow requirement.
- The readiness wrapper has an explicit `unsafe deployment docs grep` check, not only an implicit docs test inside broader pytest runs.
- `greenlet` is explicitly pinned to `3.2.5` because the locally resolved `3.5.1` wheel set was not available for the CI/Docker Python 3.11 target.
- The readiness wrapper now runs a Python 3.11 wheel/hash availability check before dependency audit, SBOM generation, Compose validation, and image build.
- The Python 3.11 wheel/hash check removes its generated download directory before each run, so stale wheels cannot mask a broken lock.
- The GitHub Actions workflow installs pinned CI tools before the final hash-locked runtime install, so CI tool dependencies cannot silently leave the test environment off the production lock.
- The CRG3 evidence runbook requires this wheel/hash check in addition to Docker, Compose, audit, release scan, and SBOM evidence.

Verification commands:

```powershell
python -m piptools compile requirements.in --generate-hashes --output-file requirements.lock
python -m pytest -q
python -m pip download --only-binary=:all: --dest build\pip-download-py311 --python-version 3.11 --implementation cp --abi cp311 --platform manylinux_2_28_x86_64 --platform manylinux2014_x86_64 --require-hashes -r requirements.lock
python -m pytest tests/infra/test_production_readiness.py -q
python -m pytest tests/infra/test_production_readiness.py tests/infra/test_docker_compose_services.py -q
python -m pytest tests/infra/test_production_readiness.py -q
python -m pytest tests/infra/test_deployment_hardening_docs.py -q
python -m pip install --dry-run --require-hashes -r requirements.lock
python -m pip_audit -r requirements.lock --progress-spinner off
python tools/security/generate_sbom.py --requirements requirements.lock --output build/security/sbom.cdx.json
python tools/security/production_readiness.py --profile local --skip-clean-check --skip-docker --skip-audit --dry-run
```

Results:
- Initial Python 3.11 wheel simulation caught `greenlet==3.5.1` as unavailable for the target.
- Recompiled lock with `greenlet==3.2.5`.
- Full pytest suite: `709 passed, 27 warnings in 92.45s`
- Linux CPython 3.11 wheel/hash download: passed.
- Readiness wrapper cache cleanup test: `16 passed in 2.34s`
- Workflow order guard: `16 passed in 2.53s`
- Unsafe deployment docs grep: `4 passed`
- Focused readiness/compose suite: `29 passed in 2.62s`
- Hash-lock dry run: passed.
- Dependency audit: `No known vulnerabilities found`
- SBOM generation: `wrote build\security\sbom.cdx.json with 35 components`
- Local readiness dry run: passed and listed the new Linux CPython 3.11 wheel availability check.

Residual risk:
- Live deployment remains blocked by `CRG3` until the documented GitHub Actions run succeeds, `tools/security/finalize_crg3.py` records evidence, and strict live readiness passes from a clean Docker-capable checkout.

### 2026-06-04 Pass 2 RWI10 Live Idempotency Cleanup Slice

Ledger IDs changed:
- `RWI10` verified for live adapter independence from process-local idempotency state.
- `CRG1` focused-suite coverage tightened to include bounded reduce-only liquidation and chase safety tests.

Files changed:
- `quant_v2/execution/binance_adapter.py`
- `tools/security/production_readiness.py`
- `tests/quant_v2/test_binance_adapter.py`
- `tests/quant_v2/test_bounded_liquidation.py`
- `requirements.lock`
- `tests/infra/test_production_readiness.py`
- `PRODUCTION_REFACTOR_ROADMAP.md`

Safety invariants proven:
- The Binance live adapter no longer imports, consults, or records `InMemoryIdempotencyJournal`.
- Paper/demo idempotency remains covered by the in-memory paper adapter tests.
- Live entry retries across a fresh adapter instance reuse the same deterministic client order ID and treat a duplicate venue response as `already_submitted`.
- Bounded reduce-only exits now pass deterministic per-attempt client order IDs, so restart retries do not rely on process memory.
- The focused P0 readiness suite includes bounded liquidation and chase tests, so reduce-only exit safety remains part of the deploy-blocking command surface.
- The regenerated dependency lock remains hash-backed and audit-clean; SBOM tests now compare generated component versions to the lockfile source of truth.

Verification commands:

```powershell
python -m pytest tests/quant_v2/test_execution_infra.py tests/quant_v2/test_binance_adapter.py tests/quant_v2/test_bounded_liquidation.py tests/quant_v2/test_chase_logic.py -q
python -m pytest tests/quant_v2/test_execution_service.py tests/quant_v2/test_live_readiness.py tests/quant_v2/test_execution_infra.py tests/quant_v2/test_binance_adapter.py tests/quant_v2/test_bounded_liquidation.py tests/quant_v2/test_chase_logic.py -q
python -m pytest tests/infra/test_docker_compose_services.py tests/infra/test_production_readiness.py -q
python -m pip install --dry-run --require-hashes -r requirements.lock
python -m pip_audit -r requirements.lock --progress-spinner off
python tools/security/generate_sbom.py --requirements requirements.lock --output build/security/sbom.cdx.json
python tools/security/production_readiness.py --profile local --skip-clean-check --skip-docker --skip-audit --dry-run
python -m pytest tests/infra/test_production_readiness.py -q
python -m pytest tests/quant_v2/test_retrain_pipeline.py tests/infra/test_docker_compose_services.py tests/infra/test_deployment_hardening_docs.py tests/infra/test_production_readiness.py tests/infra/test_release_artifact_scan.py tests/quant/test_model_selection.py tests/quant_v2/test_model_registry.py tests/quant_v2/test_signal_manager.py tests/quant_v2/test_full_ensemble.py tests/quant_v2/test_scheduled_retrain_candidates.py tests/quant/telebot/test_security_hygiene.py tests/quant/test_telebot_main_v2_handlers.py tests/quant_v2/test_binance_adapter.py tests/quant_v2/test_binance_client_phase4.py tests/quant_v2/test_bounded_liquidation.py tests/quant_v2/test_chase_logic.py tests/quant_v2/test_execution_service.py tests/quant_v2/test_live_readiness.py tests/quant_v2/test_execution_infra.py tests/quant_v2/test_reconciliation.py tests/quant_v2/test_day2_infra_patches.py tests/quant_v2/test_session_lock.py tests/quant_v2/test_watchdog_flatten_retry.py tests/quant_v2/test_credential_scrubbing.py -q
```

Results:
- Focused idempotency/adapter suite: `30 passed in 4.19s`
- Broader execution/WAL/adapter suite: `93 passed in 5.96s`
- Focused dependency/infra suite: `29 passed in 1.98s`
- Hash-lock dry run: passed
- Dependency audit: `No known vulnerabilities found`
- SBOM generation: `wrote build\security\sbom.cdx.json with 35 components`
- Focused readiness guard suite: `15 passed in 1.95s`
- Local readiness dry run: passed with developer-only Docker/audit skips and listed bounded liquidation/chase tests; not live deployment evidence
- Broader focused P0 suite: `267 passed, 4 warnings in 32.14s`

Residual risk:
- Live deployment remains blocked by `CRG3` until the documented GitHub Actions run succeeds, `tools/security/finalize_crg3.py` records evidence, and strict live readiness passes.

### 2026-06-04 Pass 6 Tracked Secret Scan Slice

Ledger IDs changed:
- `CLH5` evidence expanded from release-artifact scanning to include tracked-file secret scanning.
- `CRG3` evidence requirements expanded to include an explicit tracked-file secret scan in the readiness wrapper.

Files changed:
- `.github/workflows/production-readiness.yml`
- `.gitignore`
- `tools/security/scan_tracked_files.py`
- `tools/security/production_readiness.py`
- `tests/infra/test_release_artifact_scan.py`
- `tests/infra/test_production_readiness.py`
- `docs/PRODUCTION_READINESS.md`
- `docs/CRG3_CI_EVIDENCE.md`
- `PRODUCTION_REFACTOR_ROADMAP.md`
- Removed tracked local/debug artifacts: `audit_report.md`, `final_7_percent_audit_report.md`, `independent_audit_report.md`, `v2_live_audit_report.md`, `debug_market_data.py`, `debug_market_data_v2.py`

Safety invariants proven:
- Production readiness now runs `tools/security/scan_tracked_files.py` as a separate gate from release artifact packaging.
- The tracked-file scanner rejects committed env/key/DB/archive/log/model-artifact/diagnostic paths plus high-confidence token/private-key signatures.
- The scanner reports only path and rule names; tests assert secret-like values are not emitted.
- Historical root audit/debug artifacts were removed from the future tracked tree and ignored so they do not re-enter release commits.
- Source packages named `models` are not treated as deployed model artifacts; actual tracked `.pkl`, `.pickle`, and `.joblib` artifacts remain denied.
- The production readiness workflow can be manually dispatched for CRG3 evidence capture and runs with read-only repository permissions.

Verification commands:

```powershell
python -m pytest tests/infra/test_release_artifact_scan.py tests/infra/test_production_readiness.py -q
python tools/security/scan_tracked_files.py
python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md --allow-open-id CRG3
python tools/security/production_readiness.py --profile local --skip-clean-check --skip-docker --skip-audit --dry-run
python -c "import sys, pytest; from tools.security.production_readiness import FOCUSED_P0_TESTS; sys.exit(pytest.main([*FOCUSED_P0_TESTS, '-q']))"
```

Results:
- Focused release/readiness tests: `26 passed in 2.34s`
- Tracked-file secret scan: passed.
- Roadmap validator with CI bootstrap allowance: passed.
- Local readiness dry run: passed and listed `tracked-file secret scan`; Docker/audit skips mean this is not live evidence.
- Broad focused P0 suite: `270 passed, 4 warnings in 29.70s`

Residual risk:
- Live deployment remains blocked by `CRG3` until the documented GitHub Actions run succeeds, `tools/security/finalize_crg3.py` records evidence, and strict live readiness passes.

### 2026-06-04 Pass 6 CRG3 Finalizer Evidence Specificity Slice

Ledger IDs changed:
- `CRG3` remains in progress; finalizer-generated proof now records the full mandatory workflow gate set.

Files changed:
- `tools/security/finalize_crg3.py`
- `tests/infra/test_production_readiness.py`
- `PRODUCTION_REFACTOR_ROADMAP.md`

Safety invariants proven:
- The CRG3 finalizer does not produce vague verification text after a workflow pass.
- Finalized CRG3 proof records focused P0 tests, full pytest suite, unsafe deployment docs grep, tracked-file secret scan, Python 3.11 wheel/hash availability, dependency audit, scanned release packaging, Docker compose config, production image build, SBOM upload, run URL, commit SHA, and artifact name.
- The finalizer output roadmap still passes the P0 ledger validator in tests.

Verification commands:

```powershell
python -m pytest tests/infra/test_production_readiness.py -q
python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md --allow-open-id CRG3
```

Results:
- Readiness/finalizer tests: `16 passed in 2.13s`
- Roadmap validator with CI bootstrap allowance: passed.

Residual risk:
- Live deployment remains blocked by `CRG3` until the documented GitHub Actions run succeeds, `tools/security/finalize_crg3.py` records evidence, and strict live readiness passes.

### 2026-06-04 Pass 6 Workflow Credential and Artifact Evidence Slice

Ledger IDs changed:
- `CRG3` remains in progress; workflow credential exposure and finalizer artifact-name evidence are now statically guarded.

Files changed:
- `.github/workflows/production-readiness.yml`
- `tools/security/finalize_crg3.py`
- `docs/CRG3_CI_EVIDENCE.md`
- `tests/infra/test_production_readiness.py`
- `PRODUCTION_REFACTOR_ROADMAP.md`

Safety invariants proven:
- The Production Readiness workflow checkout uses `persist-credentials: false`, so the job does not leave GitHub credentials in the local checkout.
- The CRG3 finalizer rejects any SBOM artifact name other than the workflow-required `production-sbom`.
- Static readiness tests prove the workflow and evidence runbook both preserve these requirements.

Verification commands:

```powershell
python -m pytest tests/infra/test_production_readiness.py -q
python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md --allow-open-id CRG3
```

Results:
- Readiness/finalizer tests: `17 passed in 2.06s`
- Roadmap validator with CI bootstrap allowance: passed.
- Combined release/readiness infra tests: `27 passed in 2.47s`
- Local readiness dry run: passed and listed the full deploy-blocking chain; Docker/audit skips mean this is not live evidence.
- Tracked-file secret scan: passed.

Residual risk:
- Live deployment remains blocked by `CRG3` until the documented GitHub Actions run succeeds, `tools/security/finalize_crg3.py` records evidence, and strict live readiness passes.

### 2026-06-04 Pass 6 Workflow Action Pinning Slice

Ledger IDs changed:
- `CRG3` remains in progress; Production Readiness workflow actions are now pinned to immutable commits.

Files changed:
- `.github/workflows/production-readiness.yml`
- `docs/CRG3_CI_EVIDENCE.md`
- `tests/infra/test_production_readiness.py`
- `PRODUCTION_REFACTOR_ROADMAP.md`

Safety invariants proven:
- `actions/checkout`, `actions/setup-python`, and `actions/upload-artifact` are pinned to specific upstream commit SHAs instead of mutable major-version tags.
- Static readiness tests reject `uses: actions/...@vN` in the production readiness workflow.
- CRG3 evidence procedure requires SHA-pinned actions as part of the workflow proof.

Verification commands:

```powershell
python -m pytest tests/infra/test_production_readiness.py -q
python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md --allow-open-id CRG3
```

Results:
- Readiness workflow pinning tests: `17 passed in 2.04s`
- Roadmap validator with CI bootstrap allowance: passed.
- Combined release/readiness infra tests: `27 passed in 2.19s`
- Local readiness dry run: passed and listed the full deploy-blocking chain; Docker/audit skips mean this is not live evidence.
- Tracked-file secret scan: passed.
- Broad focused P0 suite: `271 passed, 4 warnings in 35.43s`

Residual risk:
- Live deployment remains blocked by `CRG3` until the documented GitHub Actions run succeeds, `tools/security/finalize_crg3.py` records evidence, and strict live readiness passes.

### 2026-06-04 Pass 6 CI Tool Hash Lock Slice

Ledger IDs changed:
- `CRG3` remains in progress; CI helper tool installation is now hash-locked.
- `CRG7` evidence expanded from runtime hash dry-run to CI and runtime hash dry-runs.

Files changed:
- `.github/workflows/production-readiness.yml`
- `requirements-ci.in`
- `requirements-ci.lock`
- `tools/security/production_readiness.py`
- `tools/security/finalize_crg3.py`
- `docs/PRODUCTION_READINESS.md`
- `docs/CRG3_CI_EVIDENCE.md`
- `tests/infra/test_production_readiness.py`
- `PRODUCTION_REFACTOR_ROADMAP.md`

Safety invariants proven:
- CI helper tools are installed from `requirements-ci.lock` with `pip --require-hashes`, not from plain pinned versions.
- The readiness wrapper runs a deploy-blocking `hashed CI tool lock dry-run`.
- The readiness wrapper simulates Linux CPython 3.11 CI tool wheel/hash availability and removes the generated download cache before each check.
- CRG3 finalizer proof records the CI tool lock gate before marking the row verified.

Verification commands:

```powershell
python -m piptools compile requirements-ci.in --generate-hashes --output-file requirements-ci.lock
python -m pip install --dry-run --require-hashes -r requirements-ci.lock
python -m pip download --only-binary=:all: --dest build/pip-download-ci-py311 --python-version 3.11 --implementation cp --abi cp311 --platform manylinux_2_28_x86_64 --platform manylinux2014_x86_64 --require-hashes -r requirements-ci.lock
python -m pytest tests/infra/test_production_readiness.py -q
python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md --allow-open-id CRG3
```

Results:
- CI tool lock regenerated with `--allow-unsafe` after the first Python 3.11 wheel simulation caught an unpinned transitive `pip` requirement.
- Hashed CI tool lock dry-run: passed.
- Linux CPython 3.11 CI tool wheel/hash download: passed.
- Combined release/readiness infra tests: `27 passed in 2.76s`
- Local readiness dry run: passed and listed `hashed CI tool lock dry-run` plus `Linux CPython 3.11 CI tool wheel availability`; Docker/audit skips mean this is not live evidence.
- Roadmap validator with CI bootstrap allowance: passed.
- Broad focused P0 suite: `271 passed, 4 warnings in 36.59s`

Residual risk:
- Live deployment remains blocked by `CRG3` until the documented GitHub Actions run succeeds, `tools/security/finalize_crg3.py` records evidence, and strict live readiness passes.

### 2026-06-04 Pass 6 CRG3 Repository-Bound Evidence Slice

Ledger IDs changed:
- `CRG3` remains in progress; finalizer evidence is now bound to configured GitHub remotes.

Files changed:
- `tools/security/finalize_crg3.py`
- `docs/CRG3_CI_EVIDENCE.md`
- `tests/infra/test_production_readiness.py`
- `PRODUCTION_REFACTOR_ROADMAP.md`

Safety invariants proven:
- `tools/security/finalize_crg3.py` rejects GitHub Actions run URLs from repositories that are not configured GitHub remotes for the checkout.
- Finalized CRG3 evidence records the approved repository alongside the run URL, commit SHA, and `production-sbom` artifact.
- Tests cover both rejection of an unapproved repository and successful roadmap finalization with repository evidence.

Verification commands:

```powershell
python -m pytest tests/infra/test_production_readiness.py -q
python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md --allow-open-id CRG3
```

Results:
- Readiness/finalizer tests: `18 passed in 2.66s`
- Roadmap validator with CI bootstrap allowance: passed.
- Combined release/readiness infra tests: `28 passed in 2.91s`
- Local readiness dry run: passed and still lists the full deploy-blocking chain; Docker/audit skips mean this is not live evidence.
- Tracked-file secret scan: passed.

Residual risk:
- Live deployment remains blocked by `CRG3` until the documented GitHub Actions run succeeds, `tools/security/finalize_crg3.py` records evidence, and strict live readiness passes.

### 2026-06-04 Pass 6 GitHub API Evidence Verification Slice

Ledger IDs changed:
- `CRG3` remains in progress; finalizer now verifies GitHub run metadata and artifacts before updating the roadmap.

Files changed:
- `tools/security/finalize_crg3.py`
- `docs/CRG3_CI_EVIDENCE.md`
- `tests/infra/test_production_readiness.py`
- `PRODUCTION_REFACTOR_ROADMAP.md`

Safety invariants proven:
- `tools/security/finalize_crg3.py` validates the GitHub run through the GitHub API before marking `CRG3` verified.
- The finalizer rejects failed, incomplete, wrong-workflow, or wrong-SHA runs.
- The finalizer rejects runs that lack a non-expired `production-sbom` artifact.
- For private repositories, operators must provide `GITHUB_TOKEN` so the finalizer can read workflow run and artifact metadata.

Verification commands:

```powershell
python -m pytest tests/infra/test_production_readiness.py -q
python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md --allow-open-id CRG3
```

Results:
- Readiness/finalizer tests: `21 passed in 2.82s`
- Roadmap validator with CI bootstrap allowance: passed.
- Combined release/readiness infra tests: `31 passed in 2.41s`
- Local readiness dry run: passed and still lists the full deploy-blocking chain; Docker/audit skips mean this is not live evidence.
- Tracked-file secret scan: passed.
- Broad focused P0 suite: `275 passed, 4 warnings in 29.34s`

Residual risk:
- Live deployment remains blocked by `CRG3` until the documented GitHub Actions run succeeds, `tools/security/finalize_crg3.py` records evidence from GitHub API metadata, and strict live readiness passes.

### 2026-06-04 Pass 6 Release-Branch Workflow Metadata Slice

Ledger IDs changed:
- `CRG3` remains in progress; finalizer metadata verification now rejects same-named wrong workflow files, pull request runs, and non-release branches.

Files changed:
- `tools/security/finalize_crg3.py`
- `docs/CRG3_CI_EVIDENCE.md`
- `tests/infra/test_production_readiness.py`
- `PRODUCTION_REFACTOR_ROADMAP.md`

Safety invariants proven:
- CRG3 finalization requires GitHub run `path` to be `.github/workflows/production-readiness.yml`.
- CRG3 finalization accepts only `push` or `workflow_dispatch` events.
- CRG3 finalization accepts only `main` or `master` as the run `head_branch`; pull request runs remain useful probes but cannot finalize live readiness.
- Tests reject wrong workflow path, pull request event, and non-release branch metadata.

Verification commands:

```powershell
python -m pytest tests/infra/test_production_readiness.py -q
python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md --allow-open-id CRG3
```

Results:
- Readiness/finalizer tests: `24 passed in 2.24s`
- Roadmap validator with CI bootstrap allowance: passed.
- Combined release/readiness infra tests: `34 passed in 3.02s`
- Local readiness dry run: passed and still lists the full deploy-blocking chain; Docker/audit skips mean this is not live evidence.
- Tracked-file secret scan: passed.

Residual risk:
- Live deployment remains blocked by `CRG3` until the documented GitHub Actions run succeeds, `tools/security/finalize_crg3.py` records evidence from GitHub API metadata, and strict live readiness passes.

### 2026-06-04 Pass 6 Local Checkout SHA Binding Slice

Ledger IDs changed:
- `CRG3` remains in progress; finalizer now requires local checkout `HEAD` to match the successful workflow commit SHA.

Files changed:
- `tools/security/finalize_crg3.py`
- `docs/CRG3_CI_EVIDENCE.md`
- `tests/infra/test_production_readiness.py`
- `PRODUCTION_REFACTOR_ROADMAP.md`

Safety invariants proven:
- CRG3 finalization rejects a checkout whose local `git rev-parse HEAD` does not match the supplied workflow commit SHA.
- Finalized evidence records the local checkout `HEAD` alongside the GitHub run URL, repository, commit SHA, and SBOM artifact.
- Tests reject mismatched local checkout HEAD and still verify successful roadmap finalization with matching local HEAD.

Verification commands:

```powershell
python -m pytest tests/infra/test_production_readiness.py -q
python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md --allow-open-id CRG3
```

Results:
- Readiness/finalizer tests: `25 passed in 2.22s`
- Roadmap validator with CI bootstrap allowance: passed.
- Combined release/readiness infra tests: `35 passed in 2.29s`
- Local readiness dry run: passed and still lists the full deploy-blocking chain; Docker/audit skips mean this is not live evidence.
- Tracked-file secret scan: passed.

Residual risk:
- Live deployment remains blocked by `CRG3` until the documented GitHub Actions run succeeds, `tools/security/finalize_crg3.py` records evidence from GitHub API metadata, and strict live readiness passes.

### 2026-06-04 Pass 6 Clean Finalizer Checkout Slice

Ledger IDs changed:
- `CRG3` remains in progress; finalizer now requires a clean checkout before writing CRG3 verification evidence.

Files changed:
- `tools/security/finalize_crg3.py`
- `docs/CRG3_CI_EVIDENCE.md`
- `tests/infra/test_production_readiness.py`
- `PRODUCTION_REFACTOR_ROADMAP.md`

Safety invariants proven:
- CRG3 finalization rejects dirty working trees before updating `PRODUCTION_REFACTOR_ROADMAP.md`.
- Finalized evidence records `Local working tree: clean`.
- Tests reject dirty finalizer state and still verify successful roadmap finalization from a matching clean checkout.

Verification commands:

```powershell
python -m pytest tests/infra/test_production_readiness.py -q
python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md --allow-open-id CRG3
```

Results:
- Readiness/finalizer tests: `26 passed in 2.38s`
- Roadmap validator with CI bootstrap allowance: passed.
- Combined release/readiness infra tests: `36 passed in 2.29s`
- Local readiness dry run: passed and still lists the full deploy-blocking chain; Docker/audit skips mean this is not live evidence.
- Tracked-file secret scan: passed.

Residual risk:
- Live deployment remains blocked by `CRG3` until the documented GitHub Actions run succeeds, `tools/security/finalize_crg3.py` records evidence from GitHub API metadata, and strict live readiness passes.

### 2026-06-04 Pass 6 CRG3 API Run Binding Slice

Ledger IDs changed:
- `CRG3` remains in progress; finalizer evidence is now bound to the exact GitHub Actions run URL returned by the GitHub API.

Files changed:
- `tools/security/finalize_crg3.py`
- `docs/CRG3_CI_EVIDENCE.md`
- `tests/infra/test_production_readiness.py`
- `PRODUCTION_REFACTOR_ROADMAP.md`

Safety invariants proven:
- `tools/security/finalize_crg3.py` rejects API metadata whose `html_url` does not match the submitted workflow run URL.
- Artifact verification requests the run artifacts endpoint with `per_page=100`, including when GitHub supplies the run-level `artifacts_url`.
- The CRG3 runbook documents both exact run URL binding and expanded artifact lookup.

Verification commands:

```powershell
python -m pytest tests/infra/test_production_readiness.py -q
python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md --allow-open-id CRG3
python -c "import sys, pytest; from tools.security.production_readiness import FOCUSED_P0_TESTS; sys.exit(pytest.main([*FOCUSED_P0_TESTS, '-q']))"
python tools/security/scan_tracked_files.py
python tools/security/production_readiness.py --profile local --skip-clean-check --skip-docker --skip-audit --dry-run
```

Results:
- `python -m pytest tests/infra/test_production_readiness.py -q` passed: 27 passed in 2.14s.
- `python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md --allow-open-id CRG3` passed.
- Focused P0 suite passed: 281 passed, 4 warnings in 23.04s.
- `python tools/security/scan_tracked_files.py` passed.
- Local dry-run readiness wrapper passed with explicit developer-only skips for clean-check, Docker, and dependency audit.

Rollback plan:
- Revert this slice if GitHub API response shape changes; keep `CRG3` in progress until a replacement finalizer check proves exact run identity and SBOM artifact presence.

Residual risk:
- This is still not live deployment evidence because Docker and dependency audit were skipped locally.

Live-block status:
- Live deployment remains blocked by `CRG3` until the documented GitHub Actions run succeeds, `tools/security/finalize_crg3.py` records evidence from GitHub API metadata, and strict live readiness passes.

### 2026-06-04 Pass 6 CRG3 Reverification Guard Slice

Ledger IDs changed:
- `CRG3` remains in progress; finalizer now protects existing verified evidence from accidental overwrite.

Files changed:
- `tools/security/finalize_crg3.py`
- `docs/CRG3_CI_EVIDENCE.md`
- `tests/infra/test_production_readiness.py`
- `PRODUCTION_REFACTOR_ROADMAP.md`

Safety invariants proven:
- `tools/security/finalize_crg3.py` refuses to replace an already `Verified` CRG3 row unless `--allow-reverify` is explicitly passed.
- Tests cover both rejection without override and successful re-verification with explicit override.
- The CRG3 runbook documents the override requirement.

Verification commands:

```powershell
python -m pytest tests/infra/test_production_readiness.py -q
python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md --allow-open-id CRG3
python -c "import sys, pytest; from tools.security.production_readiness import FOCUSED_P0_TESTS; sys.exit(pytest.main([*FOCUSED_P0_TESTS, '-q']))"
```

Results:
- `python -m pytest tests/infra/test_production_readiness.py -q` passed: 29 passed in 1.94s.
- `python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md --allow-open-id CRG3` passed.
- Focused P0 suite passed: 283 passed, 4 warnings in 33.86s.

Rollback plan:
- Remove the re-verification guard only if release operations need automatic evidence replacement; keep exact-run URL/API validation in place.

Residual risk:
- This is still local finalizer hardening, not Docker-capable CRG3 evidence.

Live-block status:
- Live deployment remains blocked by `CRG3` until the documented GitHub Actions run succeeds, `tools/security/finalize_crg3.py` records evidence from GitHub API metadata, and strict live readiness passes.

### 2026-06-04 Pass 6 CRG3 Artifact Pagination Slice

Ledger IDs changed:
- `CRG3` remains in progress; finalizer artifact proof now searches paged GitHub artifact results instead of only the first page.

Files changed:
- `tools/security/finalize_crg3.py`
- `docs/CRG3_CI_EVIDENCE.md`
- `tests/infra/test_production_readiness.py`
- `PRODUCTION_REFACTOR_ROADMAP.md`

Safety invariants proven:
- Artifact verification requests pages with `per_page=100` and follows GitHub `total_count` pagination.
- A workflow run with `production-sbom` only on a later artifact page can still finalize CRG3.
- Missing or expired `production-sbom` artifacts remain rejected.
- The CRG3 ledger proof and runbook now describe exact run URL binding, paged artifact lookup, and explicit re-verification override.

Verification commands:

```powershell
python -m pytest tests/infra/test_production_readiness.py -q
python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md --allow-open-id CRG3
python -c "import sys, pytest; from tools.security.production_readiness import FOCUSED_P0_TESTS; sys.exit(pytest.main([*FOCUSED_P0_TESTS, '-q']))"
```

Results:
- `python -m pytest tests/infra/test_production_readiness.py -q` passed: 30 passed in 2.19s.
- `python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md --allow-open-id CRG3` passed.
- Focused P0 suite passed: 284 passed, 4 warnings in 30.14s.

Rollback plan:
- Revert this pagination slice only if GitHub artifact API behavior changes; keep the finalizer fail-closed on missing non-expired `production-sbom`.

Residual risk:
- This remains local finalizer hardening; Docker-capable GitHub Actions evidence is still required.

Live-block status:
- Live deployment remains blocked by `CRG3` until the documented GitHub Actions run succeeds, `tools/security/finalize_crg3.py` records evidence from GitHub API metadata, and strict live readiness passes.

### 2026-06-04 Pass 6 CI Versus Live Ledger Strictness Slice

Ledger IDs changed:
- `CRG3` remains in progress; readiness wrapper contract now proves only CI may tolerate open `CRG3` while collecting workflow evidence.

Files changed:
- `tests/infra/test_production_readiness.py`
- `PRODUCTION_REFACTOR_ROADMAP.md`

Safety invariants proven:
- CI readiness can run with `--allow-open-id CRG3` so the workflow can produce the evidence needed to finalize `CRG3`.
- Live readiness uses the strict roadmap validator and does not include `--allow-open-id CRG3`.
- Future wrapper edits that make live readiness permissive will fail the infra test.

Verification commands:

```powershell
python -m pytest tests/infra/test_production_readiness.py -q
python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md --allow-open-id CRG3
python -c "import sys, pytest; from tools.security.production_readiness import FOCUSED_P0_TESTS; sys.exit(pytest.main([*FOCUSED_P0_TESTS, '-q']))"
```

Results:
- `python -m pytest tests/infra/test_production_readiness.py -q` passed: 31 passed in 2.27s.
- `python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md --allow-open-id CRG3` passed.
- Focused P0 suite passed: 285 passed, 4 warnings in 21.51s.

Rollback plan:
- If the CI workflow changes shape, preserve the invariant: CI may only allow the named evidence-bootstrap row, while live readiness must remain strict.

Residual risk:
- This is test coverage for the readiness contract; it does not replace the Docker-capable GitHub Actions evidence run.

Live-block status:
- Live deployment remains blocked by `CRG3` until the documented GitHub Actions run succeeds, `tools/security/finalize_crg3.py` records evidence from GitHub API metadata, and strict live readiness passes.

### 2026-06-04 Pass 6 Local and Live Ledger Strictness Slice

Ledger IDs changed:
- `CRG3` remains in progress; readiness contract now explicitly proves local and live runs stay strict while only CI may bootstrap CRG3 evidence.

Files changed:
- `tests/infra/test_production_readiness.py`
- `PRODUCTION_REFACTOR_ROADMAP.md`

Safety invariants proven:
- CI readiness can include `--allow-open-id CRG3` so GitHub Actions can produce the external evidence needed to finalize `CRG3`.
- Local developer readiness checks and live readiness checks do not include `--allow-open-id CRG3`.
- Dependency audit tooling is available locally and the locked runtime dependency audit reports no known vulnerabilities.

Verification commands:

```powershell
python -m pip_audit -r requirements.lock --progress-spinner off
python -m pytest tests/infra/test_production_readiness.py -q
python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md --allow-open-id CRG3
python -c "import sys, pytest; from tools.security.production_readiness import FOCUSED_P0_TESTS; sys.exit(pytest.main([*FOCUSED_P0_TESTS, '-q']))"
```

Results:
- `python -m pip_audit -r requirements.lock --progress-spinner off` passed: no known vulnerabilities found.
- `python -m pytest tests/infra/test_production_readiness.py -q` passed: 31 passed in 2.25s.
- `python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md --allow-open-id CRG3` passed.
- Focused P0 suite passed: 285 passed, 4 warnings in 33.35s.

Rollback plan:
- If CI bootstrapping changes, preserve the invariant that only CI can allow the named evidence-bootstrap row; local and live readiness must remain strict.

Residual risk:
- Docker is still unavailable locally, so Docker compose config and image build proof still require the GitHub Actions runner.

Live-block status:
- Live deployment remains blocked by `CRG3` until the documented GitHub Actions run succeeds, `tools/security/finalize_crg3.py` records evidence from GitHub API metadata, and strict live readiness passes.

### 2026-06-04 Pass 6 Release Builder Script Invocation Slice

Ledger IDs changed:
- `CRG3` remains in progress; scanned release artifact tooling now works when invoked by the same script path used by the readiness wrapper.

Files changed:
- `tools/security/build_release.py`
- `tests/infra/test_production_readiness.py`
- `PRODUCTION_REFACTOR_ROADMAP.md`

Safety invariants proven:
- `tools/security/build_release.py` adds the repository root to `sys.path` before importing `tools.security.scan_artifacts`.
- Direct script-path invocation no longer fails with `ModuleNotFoundError: No module named 'tools'`.
- Infra tests cover the release-builder CLI path so the CI readiness wrapper cannot silently drift back to an unimportable helper.

Verification commands:

```powershell
python tools/security/build_release.py --help
python -m pytest tests/infra/test_production_readiness.py tests/infra/test_release_artifact_scan.py -q
python -c "import sys, pytest; from tools.security.production_readiness import FOCUSED_P0_TESTS; sys.exit(pytest.main([*FOCUSED_P0_TESTS, '-q']))"
python -m pytest -q
python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md --allow-open-id CRG3
```

Results:
- Pre-fix probe `python tools/security/build_release.py --output build/release/quant-release.tar.gz` failed with `ModuleNotFoundError: No module named 'tools'`.
- `python tools/security/build_release.py --help` passed after the fix.
- Combined infra suite passed: 42 passed in 2.60s.
- Focused P0 suite passed: 286 passed, 4 warnings in 34.35s.
- Full pytest suite passed: 727 passed, 27 warnings in 94.06s.

Rollback plan:
- Revert only if the release helper is converted to module-only invocation everywhere; otherwise keep the script-path smoke test.

Residual risk:
- A clean release archive build from `HEAD` still depends on committing/removing tracked secret/archive artifacts before the GitHub Actions run; Docker remains unavailable locally.

Live-block status:
- Live deployment remains blocked by `CRG3` until the documented GitHub Actions run succeeds, `tools/security/finalize_crg3.py` records evidence from GitHub API metadata, and strict live readiness passes.

### 2026-06-04 Pass 6 Clean-Checkout Strictness Slice

Ledger IDs changed:
- `CRG3` remains in progress; readiness wrapper now keeps clean-check bypass strictly local while preserving developer release-scan ergonomics.

Files changed:
- `tools/security/production_readiness.py`
- `tests/infra/test_production_readiness.py`
- `PRODUCTION_REFACTOR_ROADMAP.md`

Safety invariants proven:
- `--skip-clean-check` is rejected for `live` and `ci` readiness profiles.
- Local developer runs with `--skip-clean-check` pass `--allow-dirty` to `tools/security/build_release.py`, so release artifact scanning can run against `HEAD` in an implementation worktree.
- Normal live/CI release artifact commands do not include `--allow-dirty`.
- Local readiness remains strict on the roadmap ledger; open `CRG3` still prevents accidental live-style success before final workflow evidence exists.

Verification commands:

```powershell
python -m pytest tests/infra/test_production_readiness.py -q
python tools/security/production_readiness.py --profile live --skip-clean-check --dry-run
python tools/security/production_readiness.py --profile ci --skip-clean-check --dry-run
python tools/security/production_readiness.py --profile local --skip-clean-check --skip-docker --skip-audit --dry-run
python -c "import sys, pytest; from tools.security.production_readiness import FOCUSED_P0_TESTS; sys.exit(pytest.main([*FOCUSED_P0_TESTS, '-q']))"
```

Results:
- `python -m pytest tests/infra/test_production_readiness.py -q` passed: 35 passed in 2.88s.
- Live and CI clean-check bypass probes rejected `--skip-clean-check` as expected.
- Local dry-run showed `tools/security/build_release.py --output build/release/quant-release.tar.gz --allow-dirty`.
- Focused P0 suite passed: 289 passed, 4 warnings in 33.22s.

Rollback plan:
- Revert only if release artifact packaging no longer needs dirty-tree developer support; keep live and CI clean-check bypass rejection.

Residual risk:
- Docker is still unavailable locally, and open `CRG3` still prevents strict live readiness from passing until GitHub Actions evidence is recorded.

Live-block status:
- Live deployment remains blocked by `CRG3` until the documented GitHub Actions run succeeds, `tools/security/finalize_crg3.py` records evidence from GitHub API metadata, and strict live readiness passes.

### 2026-06-04 Pass 6 Release Scanner Source Model Package Slice

Ledger IDs changed:
- `CRG3` remains in progress; release artifact scanner no longer rejects source-code packages named `models`.

Files changed:
- `tools/security/scan_artifacts.py`
- `tests/infra/test_release_artifact_scan.py`
- `PRODUCTION_REFACTOR_ROADMAP.md`

Safety invariants proven:
- Root-level `models/*` release artifacts and serialized model files such as `.joblib` remain rejected.
- Source-code packages such as `quant_v2/models/trainer.py` are allowed, avoiding a clean-release false positive.
- Dirty `HEAD` release archive probe now fails only on genuine legacy tracked artifacts: audit reports, debug scripts, and `deploy_optimized.tar.gz`.

Verification commands:

```powershell
python -m pytest tests/infra/test_release_artifact_scan.py tests/infra/test_production_readiness.py -q
python tools/security/build_release.py --output build/release/quant-release.tar.gz --allow-dirty
python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md --allow-open-id CRG3
python -c "import sys, pytest; from tools.security.production_readiness import FOCUSED_P0_TESTS; sys.exit(pytest.main([*FOCUSED_P0_TESTS, '-q']))"
```

Results:
- Combined release/readiness infra suite passed: 46 passed in 3.11s.
- Dirty `HEAD` release archive probe failed, as expected, only on genuine legacy tracked artifacts and no longer on `quant/models` or `quant_v2/models` source files.
- `python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md --allow-open-id CRG3` passed.
- Focused P0 suite passed: 290 passed, 4 warnings in 27.85s.

Rollback plan:
- Revert only if the repo moves source code out of package paths named `models`; keep `.pkl`, `.pickle`, `.joblib`, and root `models/*` artifact denies.

Residual risk:
- A scanned release archive still requires the existing tracked legacy artifacts to be removed from the release commit/history path.

Live-block status:
- Live deployment remains blocked by `CRG3` until the documented GitHub Actions run succeeds, `tools/security/finalize_crg3.py` records evidence from GitHub API metadata, and strict live readiness passes.

### 2026-06-04 Pass 6 Clean Commit Release Archive Integration Slice

Ledger IDs changed:
- `CRG3` remains in progress; release-builder tests now exercise a real clean git commit and `git archive` path.

Files changed:
- `tools/security/build_release.py`
- `tests/infra/test_release_artifact_scan.py`
- `PRODUCTION_REFACTOR_ROADMAP.md`

Safety invariants proven:
- `tools/security/build_release.py` can build and scan an archive from a clean git commit using the real `git archive` command.
- A clean tracked source package under `quant_v2/models/` is accepted in the produced archive.
- Serialized model artifacts and root release artifact paths remain denied by separate scanner tests.

Verification commands:

```powershell
python -m pytest tests/infra/test_release_artifact_scan.py -q
python -m pytest tests/infra/test_production_readiness.py tests/infra/test_release_artifact_scan.py -q
python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md --allow-open-id CRG3
python -c "import sys, pytest; from tools.security.production_readiness import FOCUSED_P0_TESTS; sys.exit(pytest.main([*FOCUSED_P0_TESTS, '-q']))"
```

Results:
- `python -m pytest tests/infra/test_release_artifact_scan.py -q` passed: 12 passed in 1.15s.
- Combined release/readiness infra suite passed: 47 passed in 3.49s.
- `python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md --allow-open-id CRG3` passed.
- Focused P0 suite passed: 291 passed, 4 warnings in 28.74s.

Rollback plan:
- Revert only if release artifact packaging no longer uses `git archive`; preserve scanner coverage for source model package paths.

Residual risk:
- Current dirty `HEAD` archive probing still fails on legacy tracked artifacts until their existing removals are committed.

Live-block status:
- Live deployment remains blocked by `CRG3` until the documented GitHub Actions run succeeds, `tools/security/finalize_crg3.py` records evidence from GitHub API metadata, and strict live readiness passes.

### 2026-06-04 Pass 6 Archive Member Content Scan Slice

Ledger IDs changed:
- `CRG3` remains in progress; release artifact scanner now inspects text member contents inside archives, not only member names.

Files changed:
- `tools/security/scan_artifacts.py`
- `tests/infra/test_release_artifact_scan.py`
- `PRODUCTION_REFACTOR_ROADMAP.md`

Safety invariants proven:
- Archive member names are still checked without filesystem extraction.
- Text-like tar/zip members are scanned in memory for unsafe credential/debug content patterns.
- Unsafe content inside a harmless-looking archive member such as `ops_probe.py` is rejected without printing secret values.
- Oversized text members fail closed with `text_too_large_for_content_scan`.

Verification commands:

```powershell
python -m pytest tests/infra/test_release_artifact_scan.py -q
python -m pytest tests/infra/test_production_readiness.py tests/infra/test_release_artifact_scan.py -q
python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md --allow-open-id CRG3
python tools/security/scan_artifacts.py build/security/sbom.cdx.json
python -c "import sys, pytest; from tools.security.production_readiness import FOCUSED_P0_TESTS; sys.exit(pytest.main([*FOCUSED_P0_TESTS, '-q']))"
```

Results:
- `python -m pytest tests/infra/test_release_artifact_scan.py -q` passed: 13 passed in 0.77s.
- Combined release/readiness infra suite passed: 48 passed in 3.28s.
- `python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md --allow-open-id CRG3` passed.
- `python tools/security/scan_artifacts.py build/security/sbom.cdx.json` passed.
- Focused P0 suite passed: 292 passed, 4 warnings in 27.35s.

Rollback plan:
- Revert only if archive content scanning creates unavoidable false positives; preserve path-only archive scanning at minimum.

Residual risk:
- Archive content scanning is bounded to text-like members and fails closed on oversized text members; binary payload provenance remains covered by denied path/extension rules and release allowlisting.

Live-block status:
- Live deployment remains blocked by `CRG3` until the documented GitHub Actions run succeeds, `tools/security/finalize_crg3.py` records evidence from GitHub API metadata, and strict live readiness passes.

### 2026-06-04 Pass 6 Proposed Worktree Release Archive Slice

Ledger IDs changed:
- `CRG3` remains in progress; local developer release preflight can now scan a temporary-index archive of the proposed worktree commit.

Files changed:
- `tools/security/build_release.py`
- `tools/security/production_readiness.py`
- `tools/security/scan_artifacts.py`
- `tests/infra/test_production_readiness.py`
- `tests/infra/test_release_artifact_scan.py`
- `.gitignore`
- `PRODUCTION_REFACTOR_ROADMAP.md`

Safety invariants proven:
- `tools/security/build_release.py --allow-dirty --from-worktree` copies the current git index into a temporary index, applies `git add -A` there, writes a temporary tree, and archives that tree without mutating the real index.
- Staged deletions are honored even when the denied file still exists on disk and is ignored.
- Local readiness dirty mode now uses `--allow-dirty --from-worktree`; live and CI release archive commands remain clean-check strict.
- Release artifact content scanning is calibrated to avoid source-code/test/Markdown false positives while still rejecting unsafe diagnostic script content and denied paths.
- Local audit log folders matching `ubuntu_audit_*/` are ignored and no longer enter proposed release archives.

Verification commands:

```powershell
python -m pytest tests/infra/test_release_artifact_scan.py -q
python tools/security/build_release.py --output build/release/quant-release.tar.gz --allow-dirty --from-worktree
python -m pytest tests/infra/test_production_readiness.py tests/infra/test_release_artifact_scan.py -q
python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md --allow-open-id CRG3
python -c "import sys, pytest; from tools.security.production_readiness import FOCUSED_P0_TESTS; sys.exit(pytest.main([*FOCUSED_P0_TESTS, '-q']))"
```

Results:
- `python -m pytest tests/infra/test_release_artifact_scan.py -q` passed: 17 passed in 2.52s.
- `python tools/security/build_release.py --output build/release/quant-release.tar.gz --allow-dirty --from-worktree` passed and produced a scanned release archive.
- Combined release/readiness infra suite passed: 52 passed in 4.60s.
- `python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md --allow-open-id CRG3` passed.
- Focused P0 suite passed: 296 passed, 4 warnings in 29.01s.

Rollback plan:
- If temporary-index archiving proves confusing operationally, remove `--from-worktree` from local developer readiness only; keep live/CI clean-check strict release packaging unchanged.

Residual risk:
- The passing proposed-worktree archive is local developer evidence, not a replacement for the clean GitHub Actions release archive and Docker image evidence required by `CRG3`.

Live-block status:
- Live deployment remains blocked by `CRG3` until the documented GitHub Actions run succeeds, `tools/security/finalize_crg3.py` records evidence from GitHub API metadata, and strict live readiness passes.

### 2026-06-04 Pass 6 Local Non-Docker Readiness Preflight Slice

Ledger IDs changed:
- `CRG3` remains in progress; local non-Docker readiness now fails only on the intentionally open CRG3 ledger row.

Files changed:
- `PRODUCTION_REFACTOR_ROADMAP.md`

Safety invariants proven:
- Focused P0 suite, full pytest suite, unsafe deployment docs grep, tracked-file secret scan, CI/runtime hash dry-runs, Linux CPython 3.11 wheel availability, dependency audit, SBOM generation, and proposed-worktree release packaging all execute successfully on this workstation.
- Local developer readiness with `--skip-clean-check --skip-docker` still does not produce live evidence and still fails on strict roadmap validation while `CRG3` is open.
- Release packaging now succeeds through the readiness wrapper using `--allow-dirty --from-worktree`.

Verification command:

```powershell
python tools/security/production_readiness.py --profile local --skip-clean-check --skip-docker
```

Results:
- Focused P0 regression suite passed: 296 passed, 4 warnings in 29.42s.
- Full pytest suite passed: 737 passed, 27 warnings in 100.65s.
- Unsafe deployment docs grep passed: 4 passed in 0.08s.
- Tracked-file secret scan passed.
- Hashed CI tool lock dry-run passed.
- Linux CPython 3.11 CI tool wheel availability passed.
- Hashed runtime dependency lock dry-run passed.
- Linux CPython 3.11 runtime wheel availability passed.
- Dependency vulnerability audit passed: no known vulnerabilities found.
- CycloneDX SBOM generation passed: 35 components.
- Proposed-worktree scanned release artifact packaging passed.
- Default compose config, production compose config, and production image build were skipped by developer flag because Docker is unavailable locally.
- Final wrapper result failed only on `roadmap P0 ledger evidence` because `CRG3` remains `In progress`.

Rollback plan:
- If local readiness starts masking open P0 rows, restore strict local roadmap validation immediately; only CI may bootstrap with `--allow-open-id CRG3`.

Residual risk:
- This is strong local preflight evidence, not live evidence. Docker compose config, Docker image build, and final CRG3 verification still require the GitHub Actions runner.

Live-block status:
- Live deployment remains blocked by `CRG3` until the documented GitHub Actions run succeeds, `tools/security/finalize_crg3.py` records evidence from GitHub API metadata, and strict live readiness passes.

### 2026-06-04 Pass 6 Proposed Worktree Preflight Documentation Slice

Ledger IDs changed:
- `CRG3` remains in progress; production readiness docs now distinguish local proposed-worktree preflight from clean CI/live release evidence.

Files changed:
- `docs/PRODUCTION_READINESS.md`
- `tests/infra/test_production_readiness.py`
- `PRODUCTION_REFACTOR_ROADMAP.md`

Safety invariants proven:
- Operator docs state that local dirty probes use `tools/security/build_release.py --output build/release/quant-release.tar.gz --allow-dirty --from-worktree`.
- Docs explain the command builds a temporary-index archive of the proposed worktree without mutating the real git index.
- Docs still state local probe output is not live deployment evidence, while CI/live release packaging remains clean-check strict.

Verification commands:

```powershell
python -m pytest tests/infra/test_production_readiness.py tests/infra/test_deployment_hardening_docs.py -q
python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md --allow-open-id CRG3
python -c "import sys, pytest; from tools.security.production_readiness import FOCUSED_P0_TESTS; sys.exit(pytest.main([*FOCUSED_P0_TESTS, '-q']))"
```

Results:
- Readiness/deployment docs suite passed: 39 passed in 3.22s.
- `python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md --allow-open-id CRG3` passed.
- Focused P0 suite passed: 296 passed, 4 warnings in 31.08s.

Rollback plan:
- Revert only if local dirty readiness no longer uses temporary-index worktree archives; keep the local-vs-live evidence distinction documented.

Residual risk:
- Documentation does not replace the required Docker-capable GitHub Actions evidence.

Live-block status:
- Live deployment remains blocked by `CRG3` until the documented GitHub Actions run succeeds, `tools/security/finalize_crg3.py` records evidence from GitHub API metadata, and strict live readiness passes.

### 2026-06-04 Pass 6 SBOM Artifact Payload Validation Slice

Ledger IDs changed:
- `CRG3` remains in progress; finalizer now validates the downloaded `production-sbom` artifact payload.

Files changed:
- `tools/security/finalize_crg3.py`
- `docs/CRG3_CI_EVIDENCE.md`
- `tests/infra/test_production_readiness.py`
- `PRODUCTION_REFACTOR_ROADMAP.md`

Safety invariants proven:
- `tools/security/finalize_crg3.py` follows GitHub artifact pagination, finds the non-expired `production-sbom` artifact, downloads its artifact zip, and verifies it contains `sbom.cdx.json`.
- The SBOM JSON must be an object with `bomFormat: CycloneDX` and a non-empty `components` list.
- Malformed or non-CycloneDX SBOM artifacts are rejected before the CRG3 ledger row can be marked verified.
- CRG3 runbook and finalizer-generated proof text now mention downloaded CycloneDX SBOM artifact validation.

Verification commands:

```powershell
python -m pytest tests/infra/test_production_readiness.py -q
python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md --allow-open-id CRG3
python -c "import sys, pytest; from tools.security.production_readiness import FOCUSED_P0_TESTS; sys.exit(pytest.main([*FOCUSED_P0_TESTS, '-q']))"
```

Results:
- `python -m pytest tests/infra/test_production_readiness.py -q` passed: 36 passed in 3.34s.
- `python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md --allow-open-id CRG3` passed.
- Focused P0 suite passed: 297 passed, 4 warnings in 33.81s.

Rollback plan:
- If GitHub artifact download semantics change, keep the artifact-name and non-expired checks while replacing the payload fetcher with the new API flow.

Residual risk:
- Final CRG3 evidence still depends on an actual successful GitHub Actions run and Docker-capable checks.

Live-block status:
- Live deployment remains blocked by `CRG3` until the documented GitHub Actions run succeeds, `tools/security/finalize_crg3.py` records evidence from GitHub API metadata, and strict live readiness passes.

### 2026-06-04 Pass 6 SBOM Component Contract Slice

Ledger IDs changed:
- `CRG3` remains in progress; finalizer now validates that downloaded SBOM artifact components match the generated package schema.

Files changed:
- `tools/security/finalize_crg3.py`
- `docs/CRG3_CI_EVIDENCE.md`
- `tests/infra/test_production_readiness.py`
- `PRODUCTION_REFACTOR_ROADMAP.md`

Safety invariants proven:
- `production-sbom` artifact payload must contain CycloneDX `sbom.cdx.json`.
- SBOM components must be non-empty objects with `type`, `name`, `version`, and `purl`, matching `tools/security/generate_sbom.py` output.
- Incomplete SBOM component payloads are rejected before CRG3 finalization.
- CRG3 runbook documents the typed component requirement.

Verification commands:

```powershell
python -m pytest tests/infra/test_production_readiness.py -q
python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md --allow-open-id CRG3
python -c "import sys, pytest; from tools.security.production_readiness import FOCUSED_P0_TESTS; sys.exit(pytest.main([*FOCUSED_P0_TESTS, '-q']))"
```

Results:
- `python -m pytest tests/infra/test_production_readiness.py -q` passed: 37 passed in 2.95s.
- `python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md --allow-open-id CRG3` passed.
- Focused P0 suite passed: 298 passed, 4 warnings in 32.27s.

Rollback plan:
- If the SBOM generator schema changes, update the finalizer component contract and runbook together rather than weakening artifact payload validation.

Residual risk:
- Final CRG3 evidence still depends on the actual Docker-capable GitHub Actions run and finalizer execution.

Live-block status:
- Live deployment remains blocked by `CRG3` until the documented GitHub Actions run succeeds, `tools/security/finalize_crg3.py` records evidence from GitHub API metadata, and strict live readiness passes.

### 2026-06-04 Pass 6 SBOM Artifact Size Bounds Slice

Ledger IDs changed:
- `CRG3` remains in progress; finalizer SBOM artifact validation now rejects oversized downloaded payloads before parsing.

Files changed:
- `tools/security/finalize_crg3.py`
- `docs/CRG3_CI_EVIDENCE.md`
- `tests/infra/test_production_readiness.py`
- `PRODUCTION_REFACTOR_ROADMAP.md`

Safety invariants proven:
- Downloaded `production-sbom` artifact zips are bounded before zip parsing.
- Embedded `sbom.cdx.json` is read with a hard byte limit before JSON decoding.
- Oversized artifact downloads and oversized SBOM JSON payloads are rejected before `CRG3` finalization can update roadmap evidence.
- The CRG3 runbook now documents bounded artifact and SBOM payload validation.

Verification commands:

```powershell
python -m pytest tests/infra/test_production_readiness.py -q
python -m pytest tests/infra/test_production_readiness.py tests/infra/test_deployment_hardening_docs.py -q
python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md --allow-open-id CRG3
python -c "import sys, pytest; from tools.security.production_readiness import FOCUSED_P0_TESTS; sys.exit(pytest.main([*FOCUSED_P0_TESTS, '-q']))"
```

Results:
- `python -m pytest tests/infra/test_production_readiness.py -q` passed: 39 passed in 2.84s.
- Readiness/deployment docs suite passed after runbook update: 43 passed in 3.15s.
- `python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md --allow-open-id CRG3` passed.
- Focused P0 suite passed after roadmap update: 300 passed, 4 warnings in 31.92s.

Rollback plan:
- If GitHub artifact sizes need to grow, raise the explicit constants in `tools/security/finalize_crg3.py` and keep the size tests aligned rather than removing bounds.

Residual risk:
- Final CRG3 evidence still depends on the actual Docker-capable GitHub Actions run and finalizer execution.

Live-block status:
- Live deployment remains blocked by `CRG3` until the documented GitHub Actions run succeeds, `tools/security/finalize_crg3.py` records evidence from GitHub API metadata, and strict live readiness passes.

### 2026-06-04 Pass 6 CRG3 Workflow Concurrency Slice

Ledger IDs changed:
- `CRG3` remains in progress; the production readiness workflow now serializes same-ref evidence runs.

Files changed:
- `.github/workflows/production-readiness.yml`
- `docs/CRG3_CI_EVIDENCE.md`
- `tests/infra/test_production_readiness.py`
- `PRODUCTION_REFACTOR_ROADMAP.md`

Safety invariants proven:
- Production readiness runs use a same-ref concurrency group: `production-readiness-${{ github.ref }}`.
- `cancel-in-progress: false` prevents a later run from silently canceling an earlier release evidence run on the same ref.
- Static readiness tests assert workflow timeout, concurrency group, and non-canceling behavior.
- CRG3 evidence handoff docs require the timeout/concurrency behavior to be true for final evidence.

Verification commands:

```powershell
python -m pytest tests/infra/test_production_readiness.py -q
python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md --allow-open-id CRG3
python -m pytest tests/infra/test_production_readiness.py tests/infra/test_deployment_hardening_docs.py -q
python -c "import sys, pytest; from tools.security.production_readiness import FOCUSED_P0_TESTS; sys.exit(pytest.main([*FOCUSED_P0_TESTS, '-q']))"
```

Results:
- `python -m pytest tests/infra/test_production_readiness.py -q` passed: 39 passed in 2.76s.
- `python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md --allow-open-id CRG3` passed.
- Readiness/deployment docs suite passed: 43 passed in 3.25s.
- Focused P0 suite passed: 300 passed, 4 warnings in 31.63s.

Rollback plan:
- If repository policy later prefers canceling duplicate PR probes, keep release-branch final evidence non-canceling and split PR-only behavior explicitly instead of removing concurrency.

Residual risk:
- Workflow concurrency reduces ambiguous evidence runs but does not replace the required Docker-capable GitHub Actions run.

Live-block status:
- Live deployment remains blocked by `CRG3` until the documented GitHub Actions run succeeds, `tools/security/finalize_crg3.py` records evidence from GitHub API metadata, and strict live readiness passes.

### 2026-06-04 Pass 6 CRG3 Artifact URL Binding Slice

Ledger IDs changed:
- `CRG3` remains in progress; finalizer now binds the downloaded SBOM artifact URL to the same GitHub repository as the verified workflow run.

Files changed:
- `tools/security/finalize_crg3.py`
- `docs/CRG3_CI_EVIDENCE.md`
- `tests/infra/test_production_readiness.py`
- `PRODUCTION_REFACTOR_ROADMAP.md`

Safety invariants proven:
- `production-sbom` artifact downloads must use `https://api.github.com`.
- Artifact download paths must match `/repos/<verified-run-repo>/actions/artifacts/<id>/zip`.
- Cross-repository artifact download URLs are rejected before the finalizer downloads or parses SBOM payloads.
- CRG3 evidence docs now state the artifact URL must belong to the same GitHub repository.

Verification commands:

```powershell
python -m pytest tests/infra/test_production_readiness.py -q
python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md --allow-open-id CRG3
python -m pytest tests/infra/test_production_readiness.py tests/infra/test_deployment_hardening_docs.py -q
python -c "import sys, pytest; from tools.security.production_readiness import FOCUSED_P0_TESTS; sys.exit(pytest.main([*FOCUSED_P0_TESTS, '-q']))"
```

Results:
- `python -m pytest tests/infra/test_production_readiness.py -q` passed: 40 passed in 3.21s.
- `python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md --allow-open-id CRG3` passed.
- Readiness/deployment docs suite passed: 44 passed in 3.11s.
- Focused P0 suite passed: 301 passed, 4 warnings in 51.79s.

Rollback plan:
- If GitHub changes artifact download URL shape, update the repository-binding parser to the new documented URL format rather than allowing arbitrary download URLs.

Residual risk:
- Repository-bound artifact URL validation strengthens finalizer evidence but does not replace the required Docker-capable GitHub Actions run.

Live-block status:
- Live deployment remains blocked by `CRG3` until the documented GitHub Actions run succeeds, `tools/security/finalize_crg3.py` records evidence from GitHub API metadata, and strict live readiness passes.

### 2026-06-04 Pass 6 CRG3 Run Artifacts URL Binding Slice

Ledger IDs changed:
- `CRG3` remains in progress; finalizer now binds GitHub run `artifacts_url` to the exact submitted workflow run.

Files changed:
- `tools/security/finalize_crg3.py`
- `docs/CRG3_CI_EVIDENCE.md`
- `tests/infra/test_production_readiness.py`
- `PRODUCTION_REFACTOR_ROADMAP.md`

Safety invariants proven:
- GitHub run metadata must include `artifacts_url`.
- The run `artifacts_url` must equal `https://api.github.com/repos/<repo>/actions/runs/<run-id>/artifacts` for the verified run URL.
- Mismatched artifact listing URLs are rejected before artifact pagination, download, or SBOM parsing.
- CRG3 evidence docs now require run `artifacts_url` to match the submitted workflow run URL.

Verification commands:

```powershell
python -m pytest tests/infra/test_production_readiness.py -q
python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md --allow-open-id CRG3
python -m pytest tests/infra/test_production_readiness.py tests/infra/test_deployment_hardening_docs.py -q
python -c "import sys, pytest; from tools.security.production_readiness import FOCUSED_P0_TESTS; sys.exit(pytest.main([*FOCUSED_P0_TESTS, '-q']))"
```

Results:
- `python -m pytest tests/infra/test_production_readiness.py -q` passed: 41 passed in 2.96s.
- `python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md --allow-open-id CRG3` passed.
- Readiness/deployment docs suite passed: 45 passed in 3.15s.
- Focused P0 suite passed: 302 passed, 4 warnings in 29.72s.

Rollback plan:
- If GitHub run metadata changes, update the exact URL derivation in `tools/security/finalize_crg3.py`; keep artifact-listing provenance bound to the verified workflow run.

Residual risk:
- Run artifact URL binding strengthens finalizer evidence but does not replace the required Docker-capable GitHub Actions run.

Live-block status:
- Live deployment remains blocked by `CRG3` until the documented GitHub Actions run succeeds, `tools/security/finalize_crg3.py` records evidence from GitHub API metadata, and strict live readiness passes.

### 2026-06-04 Pass 6 CRG3 Artifact Expiry Flag Slice

Ledger IDs changed:
- `CRG3` remains in progress; finalizer now requires the SBOM artifact metadata to explicitly prove `expired: false`.

Files changed:
- `tools/security/finalize_crg3.py`
- `docs/CRG3_CI_EVIDENCE.md`
- `tests/infra/test_production_readiness.py`
- `PRODUCTION_REFACTOR_ROADMAP.md`

Safety invariants proven:
- A matching `production-sbom` artifact is accepted only when GitHub metadata includes explicit `expired: false`.
- Missing `expired` metadata is treated as insufficient evidence and rejected before artifact download.
- CRG3 handoff docs now require explicit artifact expiry metadata, not implicit non-expiration.

Verification commands:

```powershell
python -m pytest tests/infra/test_production_readiness.py -q
python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md --allow-open-id CRG3
python -m pytest tests/infra/test_production_readiness.py tests/infra/test_deployment_hardening_docs.py -q
python -c "import sys, pytest; from tools.security.production_readiness import FOCUSED_P0_TESTS; sys.exit(pytest.main([*FOCUSED_P0_TESTS, '-q']))"
```

Results:
- `python -m pytest tests/infra/test_production_readiness.py -q` passed: 42 passed in 2.90s.
- `python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md --allow-open-id CRG3` passed.
- Readiness/deployment docs suite passed: 46 passed in 3.13s.
- Focused P0 suite passed: 303 passed, 4 warnings in 32.39s.

Rollback plan:
- If GitHub artifact metadata changes, replace this check with the new explicit non-expiration field rather than accepting missing expiry state.

Residual risk:
- Artifact expiry metadata hardening strengthens finalizer evidence but does not replace the required Docker-capable GitHub Actions run.

Live-block status:
- Live deployment remains blocked by `CRG3` until the documented GitHub Actions run succeeds, `tools/security/finalize_crg3.py` records evidence from GitHub API metadata, and strict live readiness passes.

### 2026-06-04 Pass 6 CRG3 Bounded GitHub Fetch Slice

Ledger IDs changed:
- `CRG3` remains in progress; finalizer GitHub API and artifact HTTP reads are now bounded before parsing.

Files changed:
- `tools/security/finalize_crg3.py`
- `docs/CRG3_CI_EVIDENCE.md`
- `tests/infra/test_production_readiness.py`
- `PRODUCTION_REFACTOR_ROADMAP.md`

Safety invariants proven:
- GitHub JSON API responses are read with a 1 MiB maximum before JSON decoding.
- GitHub artifact downloads are read with the existing 5 MiB `production-sbom` artifact maximum before zip parsing.
- Oversized GitHub API and artifact responses are rejected by the low-level fetch path, before finalizer metadata or SBOM parsing.
- CRG3 handoff docs now state that finalizer GitHub API metadata and artifact downloads use bounded response reads.

Verification commands:

```powershell
python -m pytest tests/infra/test_production_readiness.py -q
python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md --allow-open-id CRG3
python -m pytest tests/infra/test_production_readiness.py tests/infra/test_deployment_hardening_docs.py -q
python -c "import sys, pytest; from tools.security.production_readiness import FOCUSED_P0_TESTS; sys.exit(pytest.main([*FOCUSED_P0_TESTS, '-q']))"
```

Results:
- `python -m pytest tests/infra/test_production_readiness.py -q` passed: 44 passed in 3.67s.
- `python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md --allow-open-id CRG3` passed.
- Readiness/deployment docs suite passed: 48 passed in 2.93s.
- Focused P0 suite passed: 305 passed, 4 warnings in 37.17s.

Rollback plan:
- If GitHub response payloads legitimately exceed these bounds, raise the explicit constants and keep the over-size rejection tests aligned rather than returning to unbounded reads.

Residual risk:
- Bounded fetches strengthen finalizer robustness but do not replace the required Docker-capable GitHub Actions run.

Live-block status:
- Live deployment remains blocked by `CRG3` until the documented GitHub Actions run succeeds, `tools/security/finalize_crg3.py` records evidence from GitHub API metadata, and strict live readiness passes.

### 2026-06-04 Pass 6 CRG3 UTF-8 Payload Validation Slice

Ledger IDs changed:
- `CRG3` remains in progress; finalizer now rejects non-UTF-8 GitHub JSON and SBOM JSON payloads with controlled errors.

Files changed:
- `tools/security/finalize_crg3.py`
- `docs/CRG3_CI_EVIDENCE.md`
- `tests/infra/test_production_readiness.py`
- `PRODUCTION_REFACTOR_ROADMAP.md`

Safety invariants proven:
- GitHub API JSON responses must decode as UTF-8 before JSON parsing.
- Downloaded `production-sbom` `sbom.cdx.json` must decode as UTF-8 before CycloneDX JSON parsing.
- Non-UTF-8 API and SBOM payloads are rejected as `ValueError`s instead of surfacing raw codec exceptions.
- CRG3 handoff docs now require a bounded-size UTF-8 valid CycloneDX SBOM payload.

Verification commands:

```powershell
python -m pytest tests/infra/test_production_readiness.py -q
python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md --allow-open-id CRG3
python -m pytest tests/infra/test_production_readiness.py tests/infra/test_deployment_hardening_docs.py -q
python -c "import sys, pytest; from tools.security.production_readiness import FOCUSED_P0_TESTS; sys.exit(pytest.main([*FOCUSED_P0_TESTS, '-q']))"
```

Results:
- `python -m pytest tests/infra/test_production_readiness.py -q` passed: 46 passed in 3.65s.
- `python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md --allow-open-id CRG3` passed.
- Readiness/deployment docs suite passed: 50 passed in 3.34s.
- Focused P0 suite passed: 307 passed, 4 warnings in 34.36s.

Rollback plan:
- If GitHub or SBOM tooling ever emits a different encoding, convert it explicitly at the fetch boundary; do not allow implicit decoder exceptions in the finalizer path.

Residual risk:
- UTF-8 validation strengthens finalizer robustness but does not replace the required Docker-capable GitHub Actions run.

Live-block status:
- Live deployment remains blocked by `CRG3` until the documented GitHub Actions run succeeds, `tools/security/finalize_crg3.py` records evidence from GitHub API metadata, and strict live readiness passes.

### 2026-06-04 Pass 6 CRG3 Single SBOM Artifact Slice

Ledger IDs changed:
- `CRG3` remains in progress; finalizer now rejects ambiguous `production-sbom` artifacts containing multiple `sbom.cdx.json` files.

Files changed:
- `tools/security/finalize_crg3.py`
- `docs/CRG3_CI_EVIDENCE.md`
- `tests/infra/test_production_readiness.py`
- `PRODUCTION_REFACTOR_ROADMAP.md`

Safety invariants proven:
- The downloaded `production-sbom` artifact must contain exactly one `sbom.cdx.json`.
- Artifacts with zero or multiple SBOM JSON members are rejected before SBOM parsing and CRG3 finalization.
- CRG3 handoff docs and finalizer-generated reports now describe exactly-one SBOM evidence.

Verification commands:

```powershell
python -m pytest tests/infra/test_production_readiness.py -q
python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md --allow-open-id CRG3
python -m pytest tests/infra/test_production_readiness.py tests/infra/test_deployment_hardening_docs.py -q
python -c "import sys, pytest; from tools.security.production_readiness import FOCUSED_P0_TESTS; sys.exit(pytest.main([*FOCUSED_P0_TESTS, '-q']))"
```

Results:
- `python -m pytest tests/infra/test_production_readiness.py -q` passed: 47 passed in 2.81s.
- `python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md --allow-open-id CRG3` passed.
- Readiness/deployment docs suite passed: 51 passed in 3.03s.
- Focused P0 suite passed: 308 passed, 4 warnings in 29.19s.

Rollback plan:
- If GitHub artifact packaging later wraps files differently, preserve the exactly-one logical SBOM invariant while updating the member path matcher.

Residual risk:
- Single-SBOM validation removes artifact ambiguity but does not replace the required Docker-capable GitHub Actions run.

Live-block status:
- Live deployment remains blocked by `CRG3` until the documented GitHub Actions run succeeds, `tools/security/finalize_crg3.py` records evidence from GitHub API metadata, and strict live readiness passes.

### 2026-06-04 Pass 6 CRG3 Artifact Pagination Limit Slice

Ledger IDs changed:
- `CRG3` remains in progress; finalizer now caps GitHub artifact pagination while still allowing later-page SBOM discovery.

Files changed:
- `tools/security/finalize_crg3.py`
- `docs/CRG3_CI_EVIDENCE.md`
- `tests/infra/test_production_readiness.py`
- `PRODUCTION_REFACTOR_ROADMAP.md`

Safety invariants proven:
- Artifact pagination requests use `per_page=100` and are capped at 10 pages.
- `production-sbom` on page 2 still finalizes CRG3 in the tested happy path.
- A run that would require page 11 is rejected before issuing the page-11 API request.
- CRG3 handoff docs now state the finalizer follows `total_count` pagination for at most 10 pages.

Verification commands:

```powershell
python -m pytest tests/infra/test_production_readiness.py -q
python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md --allow-open-id CRG3
python -m pytest tests/infra/test_production_readiness.py tests/infra/test_deployment_hardening_docs.py -q
python -c "import sys, pytest; from tools.security.production_readiness import FOCUSED_P0_TESTS; sys.exit(pytest.main([*FOCUSED_P0_TESTS, '-q']))"
```

Results:
- `python -m pytest tests/infra/test_production_readiness.py -q` passed: 48 passed in 2.99s.
- `python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md --allow-open-id CRG3` passed.
- Readiness/deployment docs suite passed: 52 passed in 3.02s.
- Focused P0 suite passed: 309 passed, 4 warnings in 30.46s.

Rollback plan:
- If production workflows legitimately produce more than 1000 artifacts, raise `MAX_GITHUB_ARTIFACT_PAGES` explicitly and keep the over-limit regression aligned.

Residual risk:
- Pagination limiting prevents runaway evidence lookup but does not replace the required Docker-capable GitHub Actions run.

Live-block status:
- Live deployment remains blocked by `CRG3` until the documented GitHub Actions run succeeds, `tools/security/finalize_crg3.py` records evidence from GitHub API metadata, and strict live readiness passes.

### 2026-06-04 Pass 6 CRG3 SBOM Zip Member Limit Slice

Ledger IDs changed:
- `CRG3` remains in progress; finalizer now rejects noisy `production-sbom` artifact zips with excessive member counts.

Files changed:
- `tools/security/finalize_crg3.py`
- `docs/CRG3_CI_EVIDENCE.md`
- `tests/infra/test_production_readiness.py`
- `PRODUCTION_REFACTOR_ROADMAP.md`

Safety invariants proven:
- The downloaded `production-sbom` artifact zip must contain at most 64 members.
- Zip member-count validation runs before selecting and parsing `sbom.cdx.json`.
- Artifacts with excessive zip members are rejected before CRG3 finalization.
- CRG3 handoff docs now state both the zip member cap and the exactly-one SBOM requirement.

Verification commands:

```powershell
python -m pytest tests/infra/test_production_readiness.py -q
python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md --allow-open-id CRG3
python -m pytest tests/infra/test_production_readiness.py tests/infra/test_deployment_hardening_docs.py -q
python -c "import sys, pytest; from tools.security.production_readiness import FOCUSED_P0_TESTS; sys.exit(pytest.main([*FOCUSED_P0_TESTS, '-q']))"
```

Results:
- `python -m pytest tests/infra/test_production_readiness.py -q` passed: 49 passed in 3.51s.
- `python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md --allow-open-id CRG3` passed.
- Readiness/deployment docs suite passed: 53 passed in 3.43s.
- Focused P0 suite passed: 310 passed, 4 warnings in 33.34s.

Rollback plan:
- If the artifact uploader starts adding required metadata files, raise `MAX_SBOM_ZIP_MEMBERS` explicitly and keep the over-limit regression aligned.

Residual risk:
- Zip member limiting hardens finalizer artifact parsing but does not replace the required Docker-capable GitHub Actions run.

Live-block status:
- Live deployment remains blocked by `CRG3` until the documented GitHub Actions run succeeds, `tools/security/finalize_crg3.py` records evidence from GitHub API metadata, and strict live readiness passes.

### 2026-06-04 Pass 6 CRG3 Single SBOM Artifact Name Slice

Ledger IDs changed:
- `CRG3` remains in progress; finalizer now rejects workflow runs with duplicate non-expired `production-sbom` artifacts.

Files changed:
- `tools/security/finalize_crg3.py`
- `docs/CRG3_CI_EVIDENCE.md`
- `tests/infra/test_production_readiness.py`
- `PRODUCTION_REFACTOR_ROADMAP.md`

Safety invariants proven:
- Artifact pagination is scanned within the existing page cap before selecting the SBOM artifact.
- CRG3 finalization requires exactly one non-expired GitHub artifact named `production-sbom`.
- Duplicate non-expired `production-sbom` artifacts are rejected before any artifact download or SBOM parsing.
- CRG3 handoff docs now state both artifact-name uniqueness and exactly-one SBOM file inside the artifact.

Verification commands:

```powershell
python -m pytest tests/infra/test_production_readiness.py -q
python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md --allow-open-id CRG3
python -m pytest tests/infra/test_production_readiness.py tests/infra/test_deployment_hardening_docs.py -q
python -c "import sys, pytest; from tools.security.production_readiness import FOCUSED_P0_TESTS; sys.exit(pytest.main([*FOCUSED_P0_TESTS, '-q']))"
```

Results:
- `python -m pytest tests/infra/test_production_readiness.py -q` passed: 50 passed in 3.85s.
- `python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md --allow-open-id CRG3` passed.
- Readiness/deployment docs suite passed: 54 passed in 3.00s.
- Focused P0 suite passed: 311 passed, 4 warnings in 42.13s.

Rollback plan:
- If GitHub artifact naming semantics change, preserve a deterministic single-artifact selection rule and keep duplicate-artifact rejection covered by tests.

Residual risk:
- Artifact-name uniqueness hardens finalizer evidence but does not replace the required Docker-capable GitHub Actions run.

Live-block status:
- Live deployment remains blocked by `CRG3` until the documented GitHub Actions run succeeds, `tools/security/finalize_crg3.py` records evidence from GitHub API metadata, and strict live readiness passes.

### 2026-06-04 Pass 6 CRG3 Artifact Page Shape Slice

Ledger IDs changed:
- `CRG3` remains in progress; finalizer now validates GitHub artifact page metadata before accepting SBOM evidence.

Files changed:
- `tools/security/finalize_crg3.py`
- `docs/CRG3_CI_EVIDENCE.md`
- `tests/infra/test_production_readiness.py`
- `PRODUCTION_REFACTOR_ROADMAP.md`

Safety invariants proven:
- Artifact page requests use the shared `MAX_GITHUB_ARTIFACTS_PER_PAGE` value of 100.
- Each GitHub artifact page must include non-negative integer `total_count`.
- Each artifact page must contain no more than 100 artifacts.
- Malformed `total_count` metadata or overfull artifact pages are rejected before artifact selection, download, or SBOM parsing.

Verification commands:

```powershell
python -m pytest tests/infra/test_production_readiness.py -q
python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md --allow-open-id CRG3
python -m pytest tests/infra/test_production_readiness.py tests/infra/test_deployment_hardening_docs.py -q
python -c "import sys, pytest; from tools.security.production_readiness import FOCUSED_P0_TESTS; sys.exit(pytest.main([*FOCUSED_P0_TESTS, '-q']))"
```

Results:
- `python -m pytest tests/infra/test_production_readiness.py -q` passed: 52 passed in 3.38s.
- `python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md --allow-open-id CRG3` passed.
- Readiness/deployment docs suite passed: 56 passed in 3.03s.
- Focused P0 suite passed: 313 passed, 4 warnings in 37.59s.

Rollback plan:
- If GitHub artifact API pagination semantics change, update the shared per-page constant and response validator together rather than accepting malformed page metadata.

Residual risk:
- Artifact page shape validation hardens finalizer evidence but does not replace the required Docker-capable GitHub Actions run.

Live-block status:
- Live deployment remains blocked by `CRG3` until the documented GitHub Actions run succeeds, `tools/security/finalize_crg3.py` records evidence from GitHub API metadata, and strict live readiness passes.

### 2026-06-04 Pass 6 CRG3 Artifact Entry Shape Slice

Ledger IDs changed:
- `CRG3` remains in progress; finalizer now rejects malformed GitHub artifact list entries instead of silently skipping them.

Files changed:
- `tools/security/finalize_crg3.py`
- `docs/CRG3_CI_EVIDENCE.md`
- `tests/infra/test_production_readiness.py`
- `PRODUCTION_REFACTOR_ROADMAP.md`

Safety invariants proven:
- Every entry in a GitHub artifact page must be an object.
- Non-object artifact entries are rejected before artifact matching, download, or SBOM parsing.
- CRG3 handoff docs now state artifact pages contain at most 100 object entries.

Verification commands:

```powershell
python -m pytest tests/infra/test_production_readiness.py -q
python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md --allow-open-id CRG3
python -m pytest tests/infra/test_production_readiness.py tests/infra/test_deployment_hardening_docs.py -q
python -c "import sys, pytest; from tools.security.production_readiness import FOCUSED_P0_TESTS; sys.exit(pytest.main([*FOCUSED_P0_TESTS, '-q']))"
```

Results:
- `python -m pytest tests/infra/test_production_readiness.py -q` passed: 53 passed in 3.47s.
- `python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md --allow-open-id CRG3` passed.
- Readiness/deployment docs suite passed: 57 passed in 3.47s.
- Focused P0 suite passed: 314 passed, 4 warnings in 34.30s.

Rollback plan:
- If GitHub API response shape changes, replace the object-entry validator with the new documented schema rather than silently skipping malformed entries.

Residual risk:
- Artifact entry shape validation hardens finalizer evidence but does not replace the required Docker-capable GitHub Actions run.

Live-block status:
- Live deployment remains blocked by `CRG3` until the documented GitHub Actions run succeeds, `tools/security/finalize_crg3.py` records evidence from GitHub API metadata, and strict live readiness passes.

### 2026-06-04 Pass 6 CRG3 Artifact Size Metadata Slice

Ledger IDs changed:
- `CRG3` remains in progress; finalizer now validates selected `production-sbom` artifact `size_in_bytes` metadata before download.

Files changed:
- `tools/security/finalize_crg3.py`
- `docs/CRG3_CI_EVIDENCE.md`
- `tests/infra/test_production_readiness.py`
- `PRODUCTION_REFACTOR_ROADMAP.md`

Safety invariants proven:
- The selected non-expired `production-sbom` artifact must include integer `size_in_bytes` metadata.
- Artifact `size_in_bytes` must be non-negative and no larger than `MAX_SBOM_ARTIFACT_BYTES`.
- Invalid or oversized artifact size metadata is rejected before artifact download or SBOM parsing.
- CRG3 handoff docs now require valid `size_in_bytes` metadata as part of final artifact evidence.

Verification commands:

```powershell
python -m pytest tests/infra/test_production_readiness.py -q
python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md --allow-open-id CRG3
python -m pytest tests/infra/test_production_readiness.py tests/infra/test_deployment_hardening_docs.py -q
python -c "import sys, pytest; from tools.security.production_readiness import FOCUSED_P0_TESTS; sys.exit(pytest.main([*FOCUSED_P0_TESTS, '-q']))"
```

Results:
- `python -m pytest tests/infra/test_production_readiness.py -q` passed: 55 passed in 3.87s.
- `python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md --allow-open-id CRG3` passed.
- Readiness/deployment docs suite passed: 59 passed in 3.12s.
- Focused P0 suite passed: 316 passed, 4 warnings in 30.02s.

Rollback plan:
- If GitHub artifact metadata changes, replace this check with the new documented size field while preserving pre-download artifact-size validation.

Residual risk:
- Size metadata validation hardens finalizer evidence but does not replace the required Docker-capable GitHub Actions run.

Live-block status:
- Live deployment remains blocked by `CRG3` until the documented GitHub Actions run succeeds, `tools/security/finalize_crg3.py` records evidence from GitHub API metadata, and strict live readiness passes.

### 2026-06-04 Pass 6 CRG3 Artifact ID Binding Slice

Ledger IDs changed:
- `CRG3` remains in progress; finalizer now binds selected GitHub `production-sbom` artifact metadata to the artifact ID embedded in `archive_download_url`.

Files changed:
- `tools/security/finalize_crg3.py`
- `tests/infra/test_production_readiness.py`
- `docs/CRG3_CI_EVIDENCE.md`

Safety invariants proven:
- The selected non-expired `production-sbom` artifact must include positive integer GitHub `id` metadata.
- Boolean or otherwise malformed artifact IDs are rejected before artifact download.
- Artifact metadata `id` must match the numeric artifact ID in the repository-bound `archive_download_url`.
- Later-page artifact discovery still succeeds when the matching artifact has coherent `id`, `expired`, `size_in_bytes`, and download URL metadata.
- CRG3 runbook requirements now document artifact ID/download URL binding.

Tests added or inverted:
- Added invalid artifact ID metadata rejection.
- Added metadata/download URL artifact ID mismatch rejection.
- Updated CRG3 evidence runbook phrase enforcement.
- Updated successful fake GitHub artifact responses to include explicit artifact IDs.

Verification commands:

```bash
python -m pytest tests/infra/test_production_readiness.py -q
python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md --allow-open-id CRG3
python -m pytest tests/infra/test_production_readiness.py tests/infra/test_deployment_hardening_docs.py -q
python -c "import sys, pytest; from tools.security.production_readiness import FOCUSED_P0_TESTS; sys.exit(pytest.main([*FOCUSED_P0_TESTS, '-q']))"
python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md
```

Results:
- `python -m pytest tests/infra/test_production_readiness.py -q` passed: 57 passed in 2.95s.
- `python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md --allow-open-id CRG3` passed.
- `python -m pytest tests/infra/test_production_readiness.py tests/infra/test_deployment_hardening_docs.py -q` passed: 61 passed in 3.34s.
- Focused P0 suite passed: 318 passed, 4 warnings in 32.42s.
- Strict roadmap validation failed only because `CRG3` is still `In progress`; a Docker-capable GitHub Actions evidence run is still required.

Rollback plan:
- Revert this slice only if GitHub artifact metadata no longer exposes stable numeric `id`; replace it with the new documented artifact identity field while preserving deterministic binding between selected artifact metadata and download URL.

Residual risk:
- Artifact ID binding strengthens finalizer provenance, but it does not replace the required clean, Docker-capable GitHub Actions `Production Readiness` run and finalizer execution.

Live-block status:
- `CRG3` remains the only open P0 ledger item; production live enablement remains blocked until strict roadmap validation passes with `CRG3` marked `Verified YYYY-MM-DD`.

### 2026-06-04 Pass 6 CRG3 Canonical Artifact URL Slice

Ledger IDs changed:
- `CRG3` remains in progress; finalizer now requires selected `production-sbom` artifact download URLs to be canonical GitHub API artifact ZIP URLs.

Files changed:
- `tools/security/finalize_crg3.py`
- `tests/infra/test_production_readiness.py`
- `docs/CRG3_CI_EVIDENCE.md`

Safety invariants proven:
- `archive_download_url` must use the already-validated `https://api.github.com/repos/<verified-run-repo>/actions/artifacts/<id>/zip` path with no query string.
- `archive_download_url` must have no URL fragment.
- Canonical URL validation runs before artifact ID binding and before any artifact download.
- CRG3 handoff docs now require canonical artifact download URLs with no query string or fragment.

Tests added or inverted:
- Added query-string artifact download URL rejection.
- Added fragment artifact download URL rejection.
- Updated CRG3 evidence runbook phrase enforcement.

Verification commands:

```bash
python -m pytest tests/infra/test_production_readiness.py -q
python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md --allow-open-id CRG3
python -m pytest tests/infra/test_production_readiness.py tests/infra/test_deployment_hardening_docs.py -q
python -c "import sys, pytest; from tools.security.production_readiness import FOCUSED_P0_TESTS; sys.exit(pytest.main([*FOCUSED_P0_TESTS, '-q']))"
python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md
```

Results:
- `python -m pytest tests/infra/test_production_readiness.py -q` passed: 59 passed in 3.61s.
- `python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md --allow-open-id CRG3` passed.
- `python -m pytest tests/infra/test_production_readiness.py tests/infra/test_deployment_hardening_docs.py -q` passed: 63 passed in 3.38s.
- Focused P0 suite passed: 320 passed, 4 warnings in 32.51s.
- Strict roadmap validation failed only because `CRG3` is still `In progress`; a Docker-capable GitHub Actions evidence run is still required.

Rollback plan:
- Revert this slice only if GitHub starts returning required artifact download URLs with query parameters; if that happens, explicitly parse and validate the documented parameters instead of allowing arbitrary query or fragment values.

Residual risk:
- Canonical artifact URL validation tightens evidence provenance but does not replace the required clean, Docker-capable GitHub Actions `Production Readiness` run and finalizer execution.

Live-block status:
- `CRG3` remains the only open P0 ledger item; production live enablement remains blocked until strict roadmap validation passes with `CRG3` marked `Verified YYYY-MM-DD`.

### 2026-06-04 Pass 6 CRG3 Redirect Token Containment Slice

Ledger IDs changed:
- `CRG3` remains in progress; finalizer GitHub fetches now strip `Authorization` on cross-host redirects before following artifact download responses.

Files changed:
- `tools/security/finalize_crg3.py`
- `tests/infra/test_production_readiness.py`
- `docs/CRG3_CI_EVIDENCE.md`

Safety invariants proven:
- GitHub API requests may use `GITHUB_TOKEN`, but redirected artifact downloads do not forward `Authorization` to a different host.
- Same-host GitHub API redirects preserve normal headers.
- Bounded GitHub JSON and artifact fetch tests still exercise the finalizer fetch wrapper.
- CRG3 handoff docs now state that finalizer artifact download redirects strip `Authorization` cross-host.

Tests added or inverted:
- Added direct redirect-handler test proving cross-host `Authorization` removal.
- Added direct redirect-handler test proving same-host `Authorization` preservation.
- Updated bounded-fetch tests to patch the new finalizer fetch wrapper.
- Updated CRG3 evidence runbook phrase enforcement.

Verification commands:

```bash
python -m pytest tests/infra/test_production_readiness.py -q
python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md --allow-open-id CRG3
python -m pytest tests/infra/test_production_readiness.py tests/infra/test_deployment_hardening_docs.py -q
python -c "import sys, pytest; from tools.security.production_readiness import FOCUSED_P0_TESTS; sys.exit(pytest.main([*FOCUSED_P0_TESTS, '-q']))"
python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md
```

Results:
- `python -m pytest tests/infra/test_production_readiness.py -q` passed: 61 passed in 3.52s.
- `python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md --allow-open-id CRG3` passed.
- `python -m pytest tests/infra/test_production_readiness.py tests/infra/test_deployment_hardening_docs.py -q` passed: 65 passed in 3.80s.
- Focused P0 suite passed: 322 passed, 4 warnings in 33.83s.
- Strict roadmap validation failed only because `CRG3` is still `In progress`; a Docker-capable GitHub Actions evidence run is still required.

Rollback plan:
- Revert this slice only if GitHub artifact download behavior no longer uses redirectable URLs; preserve explicit token-containment tests for any replacement fetcher.

Residual risk:
- Redirect token containment protects CRG3 finalizer credential handling but does not replace the required clean, Docker-capable GitHub Actions `Production Readiness` run and finalizer execution.

Live-block status:
- `CRG3` remains the only open P0 ledger item; production live enablement remains blocked until strict roadmap validation passes with `CRG3` marked `Verified YYYY-MM-DD`.

### 2026-06-04 Pass 6 CRG3 Finalizer Proof Alignment Slice

Ledger IDs changed:
- `CRG3` remains in progress; finalizer-generated roadmap evidence now names the same artifact provenance controls enforced by the finalizer and CRG3 runbook.

Files changed:
- `tools/security/finalize_crg3.py`
- `tests/infra/test_production_readiness.py`
- `PRODUCTION_REFACTOR_ROADMAP.md`

Safety invariants proven:
- Finalizer ledger proof text names artifact ID/download URL binding.
- Finalizer ledger proof text names canonical artifact URL validation.
- Finalizer ledger proof text names cross-host redirect authorization stripping.
- Finalizer report text names explicit `expired: false`, bounded `size_in_bytes`, artifact `id` matching `archive_download_url`, canonical URL shape, bounded reads, and cross-host redirect token stripping.
- Successful finalization tests assert these proof phrases before accepting a roadmap as P0-valid.

Tests added or inverted:
- Expanded `test_finalize_crg3_updates_roadmap_and_allows_p0_validation` to require the new provenance proof phrases in generated roadmap output.

Verification commands:

```bash
python -m pytest tests/infra/test_production_readiness.py -q
python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md --allow-open-id CRG3
python -m pytest tests/infra/test_production_readiness.py tests/infra/test_deployment_hardening_docs.py -q
python -c "import sys, pytest; from tools.security.production_readiness import FOCUSED_P0_TESTS; sys.exit(pytest.main([*FOCUSED_P0_TESTS, '-q']))"
python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md
```

Results:
- `python -m pytest tests/infra/test_production_readiness.py -q` passed: 61 passed in 3.63s.
- `python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md --allow-open-id CRG3` passed.
- `python -m pytest tests/infra/test_production_readiness.py tests/infra/test_deployment_hardening_docs.py -q` passed: 65 passed in 3.69s.
- Focused P0 suite passed: 322 passed, 4 warnings in 38.34s.
- Strict roadmap validation failed only because `CRG3` is still `In progress`; a Docker-capable GitHub Actions evidence run is still required.

Rollback plan:
- Revert this slice only if CRG3 finalizer output is moved out of the roadmap; preserve a generated evidence artifact that explicitly records the enforced artifact provenance controls.

Residual risk:
- Proof-text alignment reduces audit drift, but it does not replace the required clean, Docker-capable GitHub Actions `Production Readiness` run and finalizer execution.

Live-block status:
- `CRG3` remains the only open P0 ledger item; production live enablement remains blocked until strict roadmap validation passes with `CRG3` marked `Verified YYYY-MM-DD`.

### 2026-06-04 Pass 6 CRG3 Run Attempt Provenance Slice

Ledger IDs changed:
- `CRG3` remains in progress; finalizer evidence is now bound to GitHub `run_attempt` metadata from the verified workflow run.

Files changed:
- `tools/security/finalize_crg3.py`
- `tests/infra/test_production_readiness.py`
- `docs/CRG3_CI_EVIDENCE.md`
- `PRODUCTION_REFACTOR_ROADMAP.md`

Safety invariants proven:
- GitHub run metadata must include positive integer `run_attempt`.
- Boolean or nonpositive attempt values are rejected before artifact pagination, artifact download, or roadmap mutation.
- Finalizer-generated CRG3 ledger proof records the workflow run attempt.
- Finalizer-generated CRG3 report records the workflow run attempt.
- CRG3 handoff docs now require `run_attempt` evidence.

Tests added or inverted:
- Added invalid `run_attempt` type rejection.
- Added nonpositive `run_attempt` rejection.
- Extended successful finalization test to assert run attempt evidence in generated roadmap text.
- Updated runbook phrase enforcement for final CRG3 evidence.

Verification commands:

```bash
python -m pytest tests/infra/test_production_readiness.py -q
python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md --allow-open-id CRG3
python -m pytest tests/infra/test_production_readiness.py tests/infra/test_deployment_hardening_docs.py -q
python -c "import sys, pytest; from tools.security.production_readiness import FOCUSED_P0_TESTS; sys.exit(pytest.main([*FOCUSED_P0_TESTS, '-q']))"
python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md
```

Results:
- `python -m pytest tests/infra/test_production_readiness.py -q` passed: 63 passed in 3.58s.
- `python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md --allow-open-id CRG3` passed.
- `python -m pytest tests/infra/test_production_readiness.py tests/infra/test_deployment_hardening_docs.py -q` passed: 67 passed in 3.18s.
- Focused P0 suite passed: 324 passed, 4 warnings in 36.90s.
- Strict roadmap validation failed only because `CRG3` is still `In progress`; a Docker-capable GitHub Actions evidence run is still required.

Rollback plan:
- Revert this slice only if GitHub removes `run_attempt` from workflow run metadata; replace it with the documented attempt identity field rather than finalizing ambiguous rerun evidence.

Residual risk:
- Run attempt provenance tightens evidence identity but does not replace the required clean, Docker-capable GitHub Actions `Production Readiness` run and finalizer execution.

Live-block status:
- `CRG3` remains the only open P0 ledger item; production live enablement remains blocked until strict roadmap validation passes with `CRG3` marked `Verified YYYY-MM-DD`.

### 2026-06-04 Pass 6 CRG3 Run API Self-Link Slice

Ledger IDs changed:
- `CRG3` remains in progress; finalizer now validates the GitHub run metadata `url` self-link against the submitted workflow run URL.

Files changed:
- `tools/security/finalize_crg3.py`
- `tests/infra/test_production_readiness.py`
- `docs/CRG3_CI_EVIDENCE.md`
- `PRODUCTION_REFACTOR_ROADMAP.md`

Safety invariants proven:
- GitHub run metadata must include `url` matching `https://api.github.com/repos/<repo>/actions/runs/<run-id>` for the submitted workflow run.
- Mismatched API self-links are rejected before status, branch, artifact, SBOM, or roadmap mutation checks can pass.
- Existing `html_url`, `artifacts_url`, `head_sha`, and `run_attempt` bindings remain intact.
- CRG3 handoff docs now require the run API self-link check.

Tests added or inverted:
- Added API `url` mismatch rejection for GitHub workflow run metadata.
- Updated fake GitHub run fixtures with explicit API self-links.
- Updated CRG3 evidence runbook phrase enforcement.

Verification commands:

```bash
python -m pytest tests/infra/test_production_readiness.py -q
python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md --allow-open-id CRG3
python -m pytest tests/infra/test_production_readiness.py tests/infra/test_deployment_hardening_docs.py -q
python -c "import sys, pytest; from tools.security.production_readiness import FOCUSED_P0_TESTS; sys.exit(pytest.main([*FOCUSED_P0_TESTS, '-q']))"
python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md
```

Results:
- `python -m pytest tests/infra/test_production_readiness.py -q` passed: 64 passed in 3.69s.
- `python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md --allow-open-id CRG3` passed.
- `python -m pytest tests/infra/test_production_readiness.py tests/infra/test_deployment_hardening_docs.py -q` passed: 68 passed in 3.54s.
- Focused P0 suite passed: 325 passed, 4 warnings in 34.67s.
- Strict roadmap validation failed only because `CRG3` is still `In progress`; a Docker-capable GitHub Actions evidence run is still required.

Rollback plan:
- Revert this slice only if GitHub removes the workflow run `url` self-link from API responses; replace it with the documented self-link identity field rather than accepting unbound run metadata.

Residual risk:
- Run API self-link validation tightens evidence identity but does not replace the required clean, Docker-capable GitHub Actions `Production Readiness` run and finalizer execution.

Live-block status:
- `CRG3` remains the only open P0 ledger item; production live enablement remains blocked until strict roadmap validation passes with `CRG3` marked `Verified YYYY-MM-DD`.

### 2026-06-04 Pass 6 CRG3 Workflow Path Ref Compatibility Slice

Ledger IDs changed:
- `CRG3` remains in progress; finalizer now accepts GitHub workflow-run path metadata that includes a safe release-branch `@ref` suffix.

Files changed:
- `tools/security/finalize_crg3.py`
- `tests/infra/test_production_readiness.py`
- `docs/CRG3_CI_EVIDENCE.md`
- `PRODUCTION_REFACTOR_ROADMAP.md`

Safety invariants proven:
- The workflow file must still be `.github/workflows/production-readiness.yml`.
- GitHub metadata path may be bare or suffixed as `.github/workflows/production-readiness.yml@main` / `@master`.
- The optional workflow path suffix must match the validated release `head_branch`.
- Feature-branch or mismatched workflow path refs are rejected before artifact and roadmap mutation checks.
- CRG3 handoff docs now describe the optional safe release-ref suffix.

Tests added or inverted:
- Added successful finalization with GitHub-style `.github/workflows/production-readiness.yml@main`.
- Added rejection for `.github/workflows/production-readiness.yml@feature/live-gates`.
- Updated CRG3 evidence runbook phrase enforcement.

Verification commands:

```bash
python -m pytest tests/infra/test_production_readiness.py -q
python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md --allow-open-id CRG3
python -m pytest tests/infra/test_production_readiness.py tests/infra/test_deployment_hardening_docs.py -q
python -c "import sys, pytest; from tools.security.production_readiness import FOCUSED_P0_TESTS; sys.exit(pytest.main([*FOCUSED_P0_TESTS, '-q']))"
python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md
```

Results:
- `python -m pytest tests/infra/test_production_readiness.py -q` passed: 66 passed in 3.24s.
- `python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md --allow-open-id CRG3` passed.
- `python -m pytest tests/infra/test_production_readiness.py tests/infra/test_deployment_hardening_docs.py -q` passed: 70 passed in 3.09s.
- Focused P0 suite passed: 327 passed, 4 warnings in 37.40s.
- Strict roadmap validation failed only because `CRG3` is still `In progress`; a Docker-capable GitHub Actions evidence run is still required.

Rollback plan:
- Revert this slice only if GitHub guarantees workflow run `path` is always bare; keep exact workflow-file validation and release-branch binding intact.

Residual risk:
- Workflow path ref compatibility prevents false CRG3 finalizer rejection but does not replace the required clean, Docker-capable GitHub Actions `Production Readiness` run and finalizer execution.

Live-block status:
- `CRG3` remains the only open P0 ledger item; production live enablement remains blocked until strict roadmap validation passes with `CRG3` marked `Verified YYYY-MM-DD`.

### 2026-06-04 Pass 6 CRG3 Run ID Metadata Binding Slice

Ledger IDs changed:
- `CRG3` remains in progress; finalizer now validates GitHub run metadata `id` against the submitted workflow run ID.

Files changed:
- `tools/security/finalize_crg3.py`
- `tests/infra/test_production_readiness.py`
- `docs/CRG3_CI_EVIDENCE.md`
- `PRODUCTION_REFACTOR_ROADMAP.md`

Safety invariants proven:
- GitHub run metadata must include positive integer `id`.
- Boolean or otherwise malformed run IDs are rejected before artifact lookup or roadmap mutation.
- GitHub run metadata `id` must equal the numeric run ID embedded in the submitted GitHub Actions run URL.
- Finalizer-generated CRG3 ledger proof and report now record workflow run ID explicitly.
- CRG3 handoff docs now require GitHub API run `id` binding.

Tests added or inverted:
- Added invalid GitHub run `id` metadata rejection.
- Added mismatched GitHub run `id` metadata rejection.
- Extended successful finalization test to assert run ID evidence in generated roadmap text.
- Updated CRG3 evidence runbook phrase enforcement.

Verification commands:

```bash
python -m pytest tests/infra/test_production_readiness.py -q
python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md --allow-open-id CRG3
python -m pytest tests/infra/test_production_readiness.py tests/infra/test_deployment_hardening_docs.py -q
python -c "import sys, pytest; from tools.security.production_readiness import FOCUSED_P0_TESTS; sys.exit(pytest.main([*FOCUSED_P0_TESTS, '-q']))"
python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md
```

Results:
- `python -m pytest tests/infra/test_production_readiness.py -q` passed: 68 passed in 3.26s.
- `python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md --allow-open-id CRG3` passed.
- `python -m pytest tests/infra/test_production_readiness.py tests/infra/test_deployment_hardening_docs.py -q` passed: 72 passed in 3.26s.
- Focused P0 suite passed: 329 passed, 4 warnings in 38.38s.
- Strict roadmap validation failed only because `CRG3` is still `In progress`; a Docker-capable GitHub Actions evidence run is still required.

Rollback plan:
- Revert this slice only if GitHub removes workflow run `id` from API responses; replace it with the documented run identity field rather than accepting unbound run metadata.

Residual risk:
- Run ID metadata binding tightens evidence identity but does not replace the required clean, Docker-capable GitHub Actions `Production Readiness` run and finalizer execution.

Live-block status:
- `CRG3` remains the only open P0 ledger item; production live enablement remains blocked until strict roadmap validation passes with `CRG3` marked `Verified YYYY-MM-DD`.

### 2026-06-04 Pass 6 CRG3 Dry-Run Evidence Clarity Slice

Ledger IDs changed:
- `CRG3` remains in progress; production readiness dry-runs now explicitly report that they are previews and not deployment evidence.

Files changed:
- `tools/security/production_readiness.py`
- `tests/infra/test_production_readiness.py`
- `docs/PRODUCTION_READINESS.md`
- `PRODUCTION_REFACTOR_ROADMAP.md`

Safety invariants proven:
- `--dry-run` prints planned checks without executing them and does not print `Production readiness checks passed.`
- Live-profile `--dry-run` can preview checks without requiring Docker CLI availability.
- Local dry-runs with developer skip flags still print the developer-only skip warning.
- Production readiness docs state that dry-runs are never deployment evidence.

Tests added or inverted:
- Updated local dry-run test to require non-evidence wording and reject success wording.
- Added live dry-run non-evidence regression.
- Updated production readiness docs enforcement.

Verification commands:

```bash
python -m pytest tests/infra/test_production_readiness.py -q
python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md --allow-open-id CRG3
python -m pytest tests/infra/test_production_readiness.py tests/infra/test_deployment_hardening_docs.py -q
python -c "import sys, pytest; from tools.security.production_readiness import FOCUSED_P0_TESTS; sys.exit(pytest.main([*FOCUSED_P0_TESTS, '-q']))"
python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md
```

Results:
- `python -m pytest tests/infra/test_production_readiness.py -q` passed: 69 passed in 3.36s.
- `python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md --allow-open-id CRG3` passed.
- `python -m pytest tests/infra/test_production_readiness.py tests/infra/test_deployment_hardening_docs.py -q` passed: 73 passed in 3.35s.
- Focused P0 suite passed: 330 passed, 4 warnings in 32.03s.
- Strict roadmap validation failed only because `CRG3` is still `In progress`; a Docker-capable GitHub Actions evidence run is still required.

Rollback plan:
- Revert this slice only if the readiness wrapper removes dry-run mode entirely; otherwise preserve explicit non-evidence wording for any preview mode.

Residual risk:
- Dry-run wording prevents operator confusion but does not replace the required clean, Docker-capable GitHub Actions `Production Readiness` run and finalizer execution.

Live-block status:
- `CRG3` remains the only open P0 ledger item; production live enablement remains blocked until strict roadmap validation passes with `CRG3` marked `Verified YYYY-MM-DD`.

### 2026-06-04 Pass 6 CRG3 Run Repository Metadata Binding Slice

Ledger IDs changed:
- `CRG3` remains in progress; finalizer now validates GitHub run `repository.full_name` metadata against the submitted workflow run repository.

Files changed:
- `tools/security/finalize_crg3.py`
- `tests/infra/test_production_readiness.py`
- `docs/CRG3_CI_EVIDENCE.md`
- `PRODUCTION_REFACTOR_ROADMAP.md`

Safety invariants proven:
- GitHub run metadata must include repository metadata as an object.
- `repository.full_name` must be a string equal to the owner/repo in the submitted run URL.
- Invalid or mismatched repository metadata is rejected before workflow status, artifact lookup, SBOM parsing, or roadmap mutation.
- CRG3 handoff docs now require run repository metadata binding.

Tests added or inverted:
- Added invalid `repository.full_name` metadata rejection.
- Added mismatched `repository.full_name` metadata rejection.
- Updated fake GitHub run fixtures with explicit repository metadata.
- Updated CRG3 evidence runbook phrase enforcement.

Verification commands:

```bash
python -m pytest tests/infra/test_production_readiness.py -q
python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md --allow-open-id CRG3
python -m pytest tests/infra/test_production_readiness.py tests/infra/test_deployment_hardening_docs.py -q
python -c "import sys, pytest; from tools.security.production_readiness import FOCUSED_P0_TESTS; sys.exit(pytest.main([*FOCUSED_P0_TESTS, '-q']))"
python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md
```

Results:
- `python -m pytest tests/infra/test_production_readiness.py -q` passed: 71 passed in 5.00s.
- `python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md --allow-open-id CRG3` passed.
- `python -m pytest tests/infra/test_production_readiness.py tests/infra/test_deployment_hardening_docs.py -q` passed: 75 passed in 4.63s.
- Focused P0 suite passed: 332 passed, 4 warnings in 44.74s.
- Strict roadmap validation failed only because `CRG3` is still `In progress`; a Docker-capable GitHub Actions evidence run is still required.

Rollback plan:
- Revert this slice only if GitHub removes repository metadata from workflow run responses; replace it with the documented repository identity field rather than accepting unbound run metadata.

Residual risk:
- Run repository metadata binding tightens evidence identity but does not replace the required clean, Docker-capable GitHub Actions `Production Readiness` run and finalizer execution.

Live-block status:
- `CRG3` remains the only open P0 ledger item; production live enablement remains blocked until strict roadmap validation passes with `CRG3` marked `Verified YYYY-MM-DD`.

### 2026-06-04 Pass 6 CRG3 Dry-Run PLAN Label Slice

Ledger IDs changed:
- `CRG3` remains in progress; production readiness dry-run check labels now use `PLAN` instead of `RUN`.

Files changed:
- `tools/security/production_readiness.py`
- `tests/infra/test_production_readiness.py`
- `docs/PRODUCTION_READINESS.md`
- `PRODUCTION_REFACTOR_ROADMAP.md`

Safety invariants proven:
- Dry-run output labels planned checks as `PLAN <check name>`.
- Dry-run output does not label planned checks as `RUN <check name>`.
- Dry-run output still echoes the exact command preview for each planned check.
- Production readiness docs now describe `PLAN` labels for dry-run previews.

Tests added or inverted:
- Updated local dry-run test to require `PLAN focused P0 regression suite` and reject `RUN focused P0 regression suite`.
- Updated live dry-run test with the same label distinction.
- Updated production readiness docs enforcement for dry-run `PLAN` labels.

Verification commands:

```bash
python -m pytest tests/infra/test_production_readiness.py -q
python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md --allow-open-id CRG3
python -m pytest tests/infra/test_production_readiness.py tests/infra/test_deployment_hardening_docs.py -q
python -c "import sys, pytest; from tools.security.production_readiness import FOCUSED_P0_TESTS; sys.exit(pytest.main([*FOCUSED_P0_TESTS, '-q']))"
python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md
```

Results:
- `python -m pytest tests/infra/test_production_readiness.py -q` passed: 71 passed in 3.73s.
- `python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md --allow-open-id CRG3` passed.
- `python -m pytest tests/infra/test_production_readiness.py tests/infra/test_deployment_hardening_docs.py -q` passed: 75 passed in 3.60s.
- Focused P0 suite passed: 332 passed, 4 warnings in 32.57s.
- Strict roadmap validation failed only because `CRG3` is still `In progress`; a Docker-capable GitHub Actions evidence run is still required.

Rollback plan:
- Revert this slice only if dry-run mode is removed; otherwise keep preview labels visually distinct from executed readiness checks.

Residual risk:
- Dry-run `PLAN` labels reduce operator confusion but do not replace the required clean, Docker-capable GitHub Actions `Production Readiness` run and finalizer execution.

Live-block status:
- `CRG3` remains the only open P0 ledger item; production live enablement remains blocked until strict roadmap validation passes with `CRG3` marked `Verified YYYY-MM-DD`.

### 2026-06-04 Pass 6 CRG3 Local Profile Evidence Wording Slice

Ledger IDs changed:
- `CRG3` remains in progress; successful local-profile readiness runs now use local-probe wording instead of production-readiness evidence wording.

Files changed:
- `tools/security/production_readiness.py`
- `tests/infra/test_production_readiness.py`
- `docs/PRODUCTION_READINESS.md`
- `PRODUCTION_REFACTOR_ROADMAP.md`

Safety invariants proven:
- `--profile local` success prints `Local readiness checks passed; this is not live deployment evidence.`
- `--profile local` success does not print `Production readiness checks passed.`
- `--profile live` success still prints `Production readiness checks passed.`
- Production readiness docs state that local probe output is not live deployment evidence even when the local profile completes without skips.

Tests added or inverted:
- Added local-profile success wording test using a stub readiness gate.
- Added live-profile success wording test using a stub readiness gate.
- Updated production readiness docs enforcement for local-profile non-live evidence semantics.

Verification commands:

```bash
python -m pytest tests/infra/test_production_readiness.py -q
python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md --allow-open-id CRG3
python -m pytest tests/infra/test_production_readiness.py tests/infra/test_deployment_hardening_docs.py -q
python -c "import sys, pytest; from tools.security.production_readiness import FOCUSED_P0_TESTS; sys.exit(pytest.main([*FOCUSED_P0_TESTS, '-q']))"
python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md
```

Results:
- `python -m pytest tests/infra/test_production_readiness.py -q` passed: 73 passed in 3.55s.
- `python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md --allow-open-id CRG3` passed.
- `python -m pytest tests/infra/test_production_readiness.py tests/infra/test_deployment_hardening_docs.py -q` passed: 77 passed in 3.91s.
- Focused P0 suite passed: 334 passed, 4 warnings in 44.13s.
- Strict roadmap validation failed only because `CRG3` is still `In progress`; a Docker-capable GitHub Actions evidence run is still required.

Rollback plan:
- Revert this slice only if the local profile is removed; otherwise keep local-profile success wording distinct from live deployment evidence.

Residual risk:
- Local-profile wording reduces operator confusion but does not replace the required clean, Docker-capable GitHub Actions `Production Readiness` run and finalizer execution.

Live-block status:
- `CRG3` remains the only open P0 ledger item; production live enablement remains blocked until strict roadmap validation passes with `CRG3` marked `Verified YYYY-MM-DD`.

### 2026-06-04 Pass 6 CRG3 Repository Evidence Report Slice

Ledger IDs changed:
- `CRG3` remains in progress; finalizer-generated evidence now records the GitHub API repository metadata value that was validated.

Files changed:
- `tools/security/finalize_crg3.py`
- `tests/infra/test_production_readiness.py`
- `PRODUCTION_REFACTOR_ROADMAP.md`

Safety invariants proven:
- Generated CRG3 evidence report includes `GitHub API repository.full_name`.
- Successful finalization tests assert the reported repository metadata value matches the verified run repository.
- Repository metadata validation and generated proof output now stay aligned.

Tests added or inverted:
- Extended successful finalization test to assert `GitHub API repository.full_name` appears in generated roadmap evidence.

Verification commands:

```bash
python -m pytest tests/infra/test_production_readiness.py -q
python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md --allow-open-id CRG3
python -m pytest tests/infra/test_production_readiness.py tests/infra/test_deployment_hardening_docs.py -q
python -c "import sys, pytest; from tools.security.production_readiness import FOCUSED_P0_TESTS; sys.exit(pytest.main([*FOCUSED_P0_TESTS, '-q']))"
python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md
```

Results:
- `python -m pytest tests/infra/test_production_readiness.py -q` passed: 73 passed in 6.77s.
- `python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md --allow-open-id CRG3` passed.
- `python -m pytest tests/infra/test_production_readiness.py tests/infra/test_deployment_hardening_docs.py -q` passed: 77 passed in 4.62s.
- Focused P0 suite passed: 334 passed, 4 warnings in 39.16s.
- Strict roadmap validation failed only because `CRG3` is still `In progress`; a Docker-capable GitHub Actions evidence run is still required.

Rollback plan:
- Revert this slice only if generated CRG3 evidence moves out of the roadmap; preserve explicit proof of repository metadata binding in the replacement evidence artifact.

Residual risk:
- Repository evidence reporting improves auditability but does not replace the required clean, Docker-capable GitHub Actions `Production Readiness` run and finalizer execution.

Live-block status:
- `CRG3` remains the only open P0 ledger item; production live enablement remains blocked until strict roadmap validation passes with `CRG3` marked `Verified YYYY-MM-DD`.

### 2026-06-04 Pass 6 CRG3 Failure Output Evidence Clarity Slice

Ledger IDs changed:
- `CRG3` remains in progress; failed production readiness runs now explicitly state that they are not live deployment evidence.

Files changed:
- `tools/security/production_readiness.py`
- `tests/infra/test_production_readiness.py`
- `docs/PRODUCTION_READINESS.md`
- `PRODUCTION_REFACTOR_ROADMAP.md`

Safety invariants proven:
- Failed readiness runs list failed checks and print `Deployment readiness failed; do not use this run as live deployment evidence.`
- Failed readiness runs do not print `Production readiness checks passed.`
- Production readiness docs state that any failed readiness check keeps live deployment blocked.

Tests added or inverted:
- Added stubbed failure-output regression for live readiness profile.
- Updated production readiness docs enforcement for failed-run non-evidence wording.

Verification commands:

```bash
python -m pytest tests/infra/test_production_readiness.py -q
python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md --allow-open-id CRG3
python -m pytest tests/infra/test_production_readiness.py tests/infra/test_deployment_hardening_docs.py -q
python -c "import sys, pytest; from tools.security.production_readiness import FOCUSED_P0_TESTS; sys.exit(pytest.main([*FOCUSED_P0_TESTS, '-q']))"
python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md
```

Results:
- `python -m pytest tests/infra/test_production_readiness.py -q` passed: 74 passed in 3.92s.
- `python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md --allow-open-id CRG3` passed.
- `python -m pytest tests/infra/test_production_readiness.py tests/infra/test_deployment_hardening_docs.py -q` passed: 78 passed in 4.11s.
- Focused P0 suite passed: 335 passed, 4 warnings in 43.47s.
- Strict roadmap validation failed only because `CRG3` is still `In progress`; a Docker-capable GitHub Actions evidence run is still required.

Rollback plan:
- Revert this slice only if readiness failure reporting is replaced by structured output; preserve an explicit failed-run non-evidence field or message.

Residual risk:
- Failure output wording reduces operator confusion but does not replace the required clean, Docker-capable GitHub Actions `Production Readiness` run and finalizer execution.

Live-block status:
- `CRG3` remains the only open P0 ledger item; production live enablement remains blocked until strict roadmap validation passes with `CRG3` marked `Verified YYYY-MM-DD`.

### 2026-06-04 Pass 6 CRG3 CI Live-Disabled Env Slice

Ledger IDs changed:
- `CRG3` remains in progress; the production-readiness workflow now has an explicit live-disabled job environment and docs/tests require that proof.

Files changed:
- `.github/workflows/production-readiness.yml`
- `tests/infra/test_production_readiness.py`
- `docs/CRG3_CI_EVIDENCE.md`
- `docs/PRODUCTION_READINESS.md`
- `PRODUCTION_REFACTOR_ROADMAP.md`

Safety invariants proven:
- The `Production Readiness` workflow job pins `BOT_V2_ALLOW_LIVE_EXECUTION: "0"` before running the readiness wrapper.
- Workflow regression tests fail if the live-disable env pin is removed or appears after the wrapper step.
- CRG3 evidence docs require operators to confirm the workflow job env pin.
- Production readiness docs state that CI readiness checks cannot enable live execution.

Tests added or inverted:
- Extended the workflow regression test to assert the job-level live-disabled env is present before the readiness wrapper.
- Extended CRG3 and production readiness docs enforcement for live-disabled CI evidence wording.

Verification commands:

```bash
python -m pytest tests/infra/test_production_readiness.py -q
python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md --allow-open-id CRG3
python -m pytest tests/infra/test_production_readiness.py tests/infra/test_deployment_hardening_docs.py -q
python -c "import sys, pytest; from tools.security.production_readiness import FOCUSED_P0_TESTS; sys.exit(pytest.main([*FOCUSED_P0_TESTS, '-q']))"
python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md
```

Results:
- `python -m pytest tests/infra/test_production_readiness.py -q` passed: 74 passed in 3.78s.
- `python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md --allow-open-id CRG3` passed.
- `python -m pytest tests/infra/test_production_readiness.py tests/infra/test_deployment_hardening_docs.py -q` passed: 78 passed in 3.30s.
- Focused P0 suite passed: 335 passed, 4 warnings in 28.58s.
- Strict roadmap validation failed only because `CRG3` is still `In progress`; a Docker-capable GitHub Actions evidence run is still required.

Rollback plan:
- Revert this slice only if the readiness workflow is replaced by a stronger isolated CI runner contract; preserve explicit live-disabled CI evidence in the replacement.

Residual risk:
- The workflow env pin prevents accidental CI live enablement during readiness checks but does not replace the required clean, Docker-capable GitHub Actions `Production Readiness` run and finalizer execution.

Live-block status:
- `CRG3` remains the only open P0 ledger item; production live enablement remains blocked until strict roadmap validation passes with `CRG3` marked `Verified YYYY-MM-DD`.

### 2026-06-04 Pass 6 CRG3 Finalizer Live-Disabled Workflow Guard Slice

Ledger IDs changed:
- `CRG3` remains in progress; finalizer execution now refuses to verify CRG3 if the local readiness workflow no longer pins live execution disabled.

Files changed:
- `tools/security/finalize_crg3.py`
- `tests/infra/test_production_readiness.py`
- `docs/CRG3_CI_EVIDENCE.md`
- `PRODUCTION_REFACTOR_ROADMAP.md`

Safety invariants proven:
- CRG3 finalization rejects a local production-readiness workflow that lacks `BOT_V2_ALLOW_LIVE_EXECUTION=0`.
- CRG3 finalization rejects the env pin if it appears after the readiness wrapper step.
- Generated finalizer evidence reports that the local workflow pinned live execution disabled before running readiness.
- The CRG3 evidence runbook documents the finalizer's local workflow live-disable requirement.

Tests added or inverted:
- Added a direct finalizer guard regression for a workflow missing the live-disabled env pin.
- Extended successful finalizer evidence assertions to require the live-disabled workflow proof line.
- Extended CRG3 runbook enforcement for the local workflow live-disable finalizer check.

Verification commands:

```bash
python -m pytest tests/infra/test_production_readiness.py -q
python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md --allow-open-id CRG3
python -m pytest tests/infra/test_production_readiness.py tests/infra/test_deployment_hardening_docs.py -q
python -c "import sys, pytest; from tools.security.production_readiness import FOCUSED_P0_TESTS; sys.exit(pytest.main([*FOCUSED_P0_TESTS, '-q']))"
python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md
```

Results:
- `python -m pytest tests/infra/test_production_readiness.py -q` passed: 76 passed in 4.12s.
- `python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md --allow-open-id CRG3` passed.
- `python -m pytest tests/infra/test_production_readiness.py tests/infra/test_deployment_hardening_docs.py -q` passed: 80 passed in 4.58s.
- Focused P0 suite passed: 337 passed, 4 warnings in 35.57s.
- Strict roadmap validation failed only because `CRG3` is still `In progress`; a Docker-capable GitHub Actions evidence run is still required.

Rollback plan:
- Revert this slice only if CRG3 finalization is moved to a stronger signed CI evidence system; preserve a hard requirement that readiness evidence cannot come from a workflow that enables live execution.

Residual risk:
- The finalizer can verify the local workflow contract at the submitted commit only after a clean checkout is available; this still does not replace the required Docker-capable GitHub Actions run and SBOM artifact validation.

Live-block status:
- `CRG3` remains the only open P0 ledger item; production live enablement remains blocked until strict roadmap validation passes with `CRG3` marked `Verified YYYY-MM-DD`.

### 2026-06-04 Pass 6 CRG3 Readiness Wrapper Live-Env Preflight Slice

Ledger IDs changed:
- `CRG3` remains in progress; the readiness wrapper now refuses to run checks while live execution is enabled or ambiguously configured.

Files changed:
- `tools/security/production_readiness.py`
- `tests/infra/test_production_readiness.py`
- `docs/CRG3_CI_EVIDENCE.md`
- `docs/PRODUCTION_READINESS.md`
- `PRODUCTION_REFACTOR_ROADMAP.md`

Safety invariants proven:
- Readiness checks fail before planning or executing gates when `BOT_V2_ALLOW_LIVE_EXECUTION=1`.
- Readiness checks fail before planning or executing gates when `BOT_V2_ALLOW_LIVE_EXECUTION` is an unknown value such as `maybe`.
- Only unset or explicit false-ish values are accepted while readiness checks run.
- CRG3 and production readiness docs explain the wrapper-level live-env preflight.

Tests added or inverted:
- Added CI-profile fail-fast test for `BOT_V2_ALLOW_LIVE_EXECUTION=1`.
- Added live-profile fail-fast test for unknown live-env typos.
- Extended docs enforcement for wrapper-level live-env preflight wording.

Verification commands:

```bash
python -m pytest tests/infra/test_production_readiness.py -q
python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md --allow-open-id CRG3
python -m pytest tests/infra/test_production_readiness.py tests/infra/test_deployment_hardening_docs.py -q
python -c "import sys, pytest; from tools.security.production_readiness import FOCUSED_P0_TESTS; sys.exit(pytest.main([*FOCUSED_P0_TESTS, '-q']))"
python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md
```

Results:
- `python -m pytest tests/quant_v2/test_retrain_pipeline.py tests/infra/test_docker_compose_services.py tests/infra/test_deployment_hardening_docs.py tests/infra/test_production_readiness.py::test_production_readiness_local_dry_run_lists_deploy_blocking_checks -q` passed: 20 passed in 7.04s.
- `python -m pytest tests/infra/test_production_readiness.py -q` passed: 78 passed in 5.41s.
- `python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md --allow-open-id CRG3` passed.
- `python -m pytest tests/infra/test_production_readiness.py tests/infra/test_deployment_hardening_docs.py -q` passed: 82 passed in 3.51s.
- Focused P0 suite passed: 339 passed, 4 warnings in 37.57s.
- Strict roadmap validation failed only because `CRG3` is still `In progress`; a Docker-capable GitHub Actions evidence run is still required.

Rollback plan:
- Revert this slice only if readiness execution moves into a runner that cryptographically proves live execution cannot be enabled during checks; preserve fail-fast behavior in any replacement wrapper.

Residual risk:
- This preflight prevents readiness checks from running under live-enabled env, but CRG3 still requires the Docker-capable GitHub Actions evidence run and finalizer validation.

Live-block status:
- `CRG3` remains the only open P0 ledger item; production live enablement remains blocked until strict roadmap validation passes with `CRG3` marked `Verified YYYY-MM-DD`.

### 2026-06-04 Pass 6 CRG3 Structured Workflow Contract Slice

Ledger IDs changed:
- `CRG3` remains in progress; finalizer workflow validation now parses the local GitHub Actions YAML instead of relying only on string presence/order.

Files changed:
- `tools/security/finalize_crg3.py`
- `tests/infra/test_production_readiness.py`
- `docs/CRG3_CI_EVIDENCE.md`
- `PRODUCTION_REFACTOR_ROADMAP.md`

Safety invariants proven:
- CRG3 finalization rejects comment-only `BOT_V2_ALLOW_LIVE_EXECUTION: "0"` spoofing.
- CRG3 finalization rejects non-string live-disabled env values such as unquoted YAML integer `0`.
- CRG3 finalization rejects wrapper-step env overrides that enable live execution.
- CRG3 finalization rejects wrapper command drift away from `python tools/security/production_readiness.py --profile ci`.
- CRG3 evidence docs now describe structured workflow validation instead of raw-text-only proof.

Tests added or inverted:
- Added structured workflow finalizer regressions for comment-only env spoofing, non-string env value, wrapper-step live override, and wrapper command drift.
- Extended CRG3 evidence runbook enforcement for the structured workflow validation contract.

Verification commands:

```bash
python -m pytest tests/infra/test_production_readiness.py -q
python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md --allow-open-id CRG3
python -m pytest tests/infra/test_production_readiness.py tests/infra/test_deployment_hardening_docs.py -q
python -c "import sys, pytest; from tools.security.production_readiness import FOCUSED_P0_TESTS; sys.exit(pytest.main([*FOCUSED_P0_TESTS, '-q']))"
python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md
```

Results:
- `python -m pytest tests/infra/test_production_readiness.py -q` passed: 82 passed in 5.38s.
- `python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md --allow-open-id CRG3` passed.
- `python -m pytest tests/infra/test_production_readiness.py tests/infra/test_deployment_hardening_docs.py -q` passed: 86 passed in 4.35s.
- Focused P0 suite passed: 343 passed, 4 warnings in 38.53s.
- Strict roadmap validation failed only because `CRG3` is still `In progress`; a Docker-capable GitHub Actions evidence run is still required.

Rollback plan:
- Revert this slice only if workflow evidence validation moves to a stronger signed or GitHub-native policy evaluator; preserve structural validation of the readiness job env and wrapper step in any replacement.

Residual risk:
- Structured local workflow validation reduces finalizer spoofing risk but still depends on the required Docker-capable GitHub Actions run and SBOM artifact validation to close CRG3.

Live-block status:
- `CRG3` remains the only open P0 ledger item; production live enablement remains blocked until strict roadmap validation passes with `CRG3` marked `Verified YYYY-MM-DD`.

### 2026-06-04 Pass 6 CRG3 Structured SBOM Upload Contract Slice

Ledger IDs changed:
- `CRG3` remains in progress; finalizer workflow validation now proves the local readiness workflow uploads the expected SBOM artifact name/path.

Files changed:
- `tools/security/finalize_crg3.py`
- `tests/infra/test_production_readiness.py`
- `docs/CRG3_CI_EVIDENCE.md`
- `PRODUCTION_REFACTOR_ROADMAP.md`

Safety invariants proven:
- CRG3 finalization rejects a workflow without exactly one `Upload SBOM` step.
- CRG3 finalization rejects a wrong SBOM artifact name.
- CRG3 finalization rejects a wrong SBOM artifact path.
- CRG3 finalization rejects SBOM upload configuration that does not fail on missing files.
- Generated finalizer evidence reports the local workflow SBOM upload contract.

Tests added or inverted:
- Added finalizer regressions for missing SBOM upload, wrong artifact name, wrong path, and non-failing missing-file handling.
- Extended CRG3 evidence runbook enforcement for structured SBOM upload validation.
- Extended successful finalizer evidence assertions to include local workflow SBOM upload proof.

Verification commands:

```bash
python -m pytest tests/infra/test_production_readiness.py -q
python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md --allow-open-id CRG3
python -m pytest tests/infra/test_production_readiness.py tests/infra/test_deployment_hardening_docs.py -q
python -c "import sys, pytest; from tools.security.production_readiness import FOCUSED_P0_TESTS; sys.exit(pytest.main([*FOCUSED_P0_TESTS, '-q']))"
python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md
```

Results:
- `python -m pytest tests/infra/test_production_readiness.py -q` passed: 86 passed in 4.61s.
- `python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md --allow-open-id CRG3` passed.
- `python -m pytest tests/infra/test_production_readiness.py tests/infra/test_deployment_hardening_docs.py -q` passed: 90 passed in 3.93s.
- Focused P0 suite passed: 347 passed, 4 warnings in 47.03s.
- Strict roadmap validation failed only because `CRG3` is still `In progress`; a Docker-capable GitHub Actions evidence run is still required.

Rollback plan:
- Revert this slice only if SBOM evidence moves to a stronger signed provenance mechanism; preserve exact artifact name/path validation in any replacement.

Residual risk:
- Structured SBOM upload validation proves the local workflow contract but still requires the Docker-capable GitHub Actions run and downloaded artifact validation to close CRG3.

Live-block status:
- `CRG3` remains the only open P0 ledger item; production live enablement remains blocked until strict roadmap validation passes with `CRG3` marked `Verified YYYY-MM-DD`.

### 2026-06-04 Pass 6 CRG3 Structured Install Chain Contract Slice

Ledger IDs changed:
- `CRG3` remains in progress; finalizer workflow validation now proves the local readiness workflow uses the expected checkout, Python setup, and hash-locked install chain before running readiness.

Files changed:
- `tools/security/finalize_crg3.py`
- `tests/infra/test_production_readiness.py`
- `docs/CRG3_CI_EVIDENCE.md`
- `PRODUCTION_REFACTOR_ROADMAP.md`

Safety invariants proven:
- CRG3 finalization rejects checkout steps that persist credentials.
- CRG3 finalization rejects floating setup-python actions.
- CRG3 finalization rejects runtime dependency installs without `--require-hashes`.
- CRG3 finalization rejects runtime dependency install before CI tool install.
- Generated finalizer evidence reports pinned checkout/setup actions and hash-locked install ordering.

Tests added or inverted:
- Added finalizer regressions for persisted checkout credentials, floating setup-python action, unhashed runtime install, and wrong install order.
- Extended CRG3 evidence runbook enforcement for the structured setup/install chain.
- Extended successful finalizer evidence assertions to include local workflow setup/install proof.

Verification commands:

```bash
python -m pytest tests/infra/test_production_readiness.py -q
python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md --allow-open-id CRG3
python -m pytest tests/infra/test_production_readiness.py tests/infra/test_deployment_hardening_docs.py -q
python -c "import sys, pytest; from tools.security.production_readiness import FOCUSED_P0_TESTS; sys.exit(pytest.main([*FOCUSED_P0_TESTS, '-q']))"
python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md
```

Results:
- `python -m pytest tests/infra/test_production_readiness.py -q` passed: 90 passed in 4.80s.
- `python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md --allow-open-id CRG3` passed.
- `python -m pytest tests/infra/test_production_readiness.py tests/infra/test_deployment_hardening_docs.py -q` passed: 94 passed in 3.89s.
- Focused P0 suite passed: 351 passed, 4 warnings in 38.53s.
- Strict roadmap validation failed only because `CRG3` is still `In progress`; a Docker-capable GitHub Actions evidence run is still required.

Rollback plan:
- Revert this slice only if setup/install proof moves to a stronger workflow policy engine; preserve pinned action, no persisted checkout credentials, hash-locked install, and ordering checks in any replacement.

Residual risk:
- Structured install-chain validation proves the local workflow contract but still requires the Docker-capable GitHub Actions run and downloaded artifact validation to close CRG3.

Live-block status:
- `CRG3` remains the only open P0 ledger item; production live enablement remains blocked until strict roadmap validation passes with `CRG3` marked `Verified YYYY-MM-DD`.

### 2026-06-04 Pass 6 CRG3 Structured Execution Controls Slice

Ledger IDs changed:
- `CRG3` remains in progress; finalizer workflow validation now proves the local readiness workflow execution controls before accepting CI evidence.

Files changed:
- `tools/security/finalize_crg3.py`
- `tests/infra/test_production_readiness.py`
- `docs/CRG3_CI_EVIDENCE.md`
- `PRODUCTION_REFACTOR_ROADMAP.md`

Safety invariants proven:
- CRG3 finalization rejects incomplete release-branch push triggers.
- CRG3 finalization rejects broad workflow permissions.
- CRG3 finalization rejects concurrency that cancels in-progress evidence runs.
- CRG3 finalization rejects readiness job timeout drift away from 45 minutes.
- Generated finalizer evidence reports manual dispatch, pull-request probes, release-branch push triggers, read-only permissions, same-ref concurrency without cancellation, Ubuntu runner, and timeout proof.

Tests added or inverted:
- Added finalizer regressions for incomplete push branches, broad permissions, canceling concurrency, and timeout drift.
- Extended CRG3 evidence runbook enforcement for structured workflow execution controls.
- Extended successful finalizer evidence assertions to include local workflow execution-control proof.

Verification commands:

```bash
python -m pytest tests/infra/test_production_readiness.py -q
python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md --allow-open-id CRG3
python -m pytest tests/infra/test_production_readiness.py tests/infra/test_deployment_hardening_docs.py -q
python -c "import sys, pytest; from tools.security.production_readiness import FOCUSED_P0_TESTS; sys.exit(pytest.main([*FOCUSED_P0_TESTS, '-q']))"
python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md
```

Results:
- `python -m pytest tests/infra/test_production_readiness.py -q` passed: 94 passed in 4.24s.
- `python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md --allow-open-id CRG3` passed.
- `python -m pytest tests/infra/test_production_readiness.py tests/infra/test_deployment_hardening_docs.py -q` passed: 98 passed in 6.00s.
- Focused P0 suite passed: 355 passed, 4 warnings in 41.00s.
- Strict roadmap validation failed only because `CRG3` is still `In progress`; a Docker-capable GitHub Actions evidence run is still required.

Rollback plan:
- Revert this slice only if workflow execution controls move to a stronger policy engine; preserve trigger, permission, concurrency, runner, and timeout validation in any replacement.

Residual risk:
- Structured execution-control validation proves the local workflow contract but still requires the Docker-capable GitHub Actions run and downloaded artifact validation to close CRG3.

Live-block status:
- `CRG3` remains the only open P0 ledger item; production live enablement remains blocked until strict roadmap validation passes with `CRG3` marked `Verified YYYY-MM-DD`.

### 2026-06-04 Pass 6 CRG3 CI Success Wording Slice

Ledger IDs changed:
- `CRG3` remains in progress; the readiness wrapper now gives CI-profile success distinct non-live wording instead of live-readiness wording.

Files changed:
- `tools/security/production_readiness.py`
- `tests/infra/test_production_readiness.py`
- `docs/PRODUCTION_READINESS.md`
- `PRODUCTION_REFACTOR_ROADMAP.md`

Safety invariants proven:
- `--profile ci` success prints `CI readiness checks passed; finalize CRG3 from GitHub Actions evidence before live deployment.`
- `--profile ci` success does not print `Production readiness checks passed.`
- `--profile live` success remains the only profile that prints `Production readiness checks passed.`
- Production readiness docs describe the distinct CI success wording.

Tests added or inverted:
- Added stubbed CI-profile success wording regression.
- Extended production readiness docs enforcement for CI-profile non-live success wording.

Verification commands:

```bash
python -m pytest tests/infra/test_production_readiness.py -q
python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md --allow-open-id CRG3
python -m pytest tests/infra/test_production_readiness.py tests/infra/test_deployment_hardening_docs.py -q
python -c "import sys, pytest; from tools.security.production_readiness import FOCUSED_P0_TESTS; sys.exit(pytest.main([*FOCUSED_P0_TESTS, '-q']))"
python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md
```

Results:
- `python -m pytest tests/infra/test_production_readiness.py -q` passed: 95 passed in 9.23s.
- `python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md --allow-open-id CRG3` passed.
- `python -m pytest tests/infra/test_production_readiness.py tests/infra/test_deployment_hardening_docs.py -q` passed: 99 passed in 9.22s.
- Focused P0 suite passed: 356 passed, 4 warnings in 46.52s.
- Strict roadmap validation failed only because `CRG3` is still `In progress`; a Docker-capable GitHub Actions evidence run is still required.

Rollback plan:
- Revert this slice only if CI and live profiles are merged; preserve distinct output for any readiness mode that allows CRG3 to remain open.

Residual risk:
- CI wording reduces operator confusion but still depends on the Docker-capable GitHub Actions run and finalizer validation to close CRG3.

Live-block status:
- `CRG3` remains the only open P0 ledger item; production live enablement remains blocked until strict roadmap validation passes with `CRG3` marked `Verified YYYY-MM-DD`.

### 2026-06-04 Pass 6 CRG3 Finalizer Ledger Self-Validation Slice

Ledger IDs changed:
- `CRG3` remains in progress; finalizer now validates the full updated P0 ledger before writing roadmap evidence.

Files changed:
- `tools/security/check_roadmap.py`
- `tools/security/finalize_crg3.py`
- `tests/infra/test_production_readiness.py`
- `docs/CRG3_CI_EVIDENCE.md`
- `PRODUCTION_REFACTOR_ROADMAP.md`

Safety invariants proven:
- Finalizer builds the updated roadmap text in memory before writing.
- Finalizer validates every P0 row in the updated ledger before mutating `PRODUCTION_REFACTOR_ROADMAP.md`.
- Finalizer rejects and preserves the original roadmap when another P0 row remains open.
- CRG3 evidence runbook documents pre-write full-ledger validation.

Tests added or inverted:
- Added finalizer regression proving another open P0 row blocks write and leaves roadmap contents unchanged.
- Added text-level ledger parser so finalizer can validate would-be roadmap content before file mutation.
- Extended CRG3 evidence runbook enforcement for pre-write ledger validation.

Verification commands:

```bash
python -m pytest tests/infra/test_production_readiness.py -q
python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md --allow-open-id CRG3
python -m pytest tests/infra/test_production_readiness.py tests/infra/test_deployment_hardening_docs.py -q
python -c "import sys, pytest; from tools.security.production_readiness import FOCUSED_P0_TESTS; sys.exit(pytest.main([*FOCUSED_P0_TESTS, '-q']))"
python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md
```

Results:
- `python -m pytest tests/infra/test_production_readiness.py -q` passed: 96 passed in 9.28s.
- `python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md --allow-open-id CRG3` passed.
- `python -m pytest tests/infra/test_production_readiness.py tests/infra/test_deployment_hardening_docs.py -q` passed: 100 passed in 6.85s.
- Focused P0 suite passed: 357 passed, 4 warnings in 50.38s.
- Strict roadmap validation failed only because `CRG3` is still `In progress`; a Docker-capable GitHub Actions evidence run is still required.

Rollback plan:
- Revert this slice only if roadmap writes move to a transactional external store; preserve pre-write full P0 ledger validation in any replacement.

Residual risk:
- Ledger self-validation prevents bad roadmap mutation but still requires the Docker-capable GitHub Actions run and artifact validation to close CRG3.

Live-block status:
- `CRG3` remains the only open P0 ledger item; production live enablement remains blocked until strict roadmap validation passes with `CRG3` marked `Verified YYYY-MM-DD`.

### 2026-06-04 Pass 6 CRG3 Finalizer Dry-Run Slice

Ledger IDs changed:
- `CRG3` remains in progress; finalizer now supports validating CRG3 evidence and the would-be roadmap update without mutating the roadmap.

Files changed:
- `tools/security/finalize_crg3.py`
- `tests/infra/test_production_readiness.py`
- `docs/CRG3_CI_EVIDENCE.md`
- `PRODUCTION_REFACTOR_ROADMAP.md`

Safety invariants proven:
- Finalizer `dry_run=True` performs evidence validation and in-memory P0 ledger validation without writing roadmap changes.
- CLI `--dry-run` passes the dry-run flag through to finalization.
- CLI dry-run prints `CRG3 finalizer dry run passed; roadmap not modified.`
- CRG3 evidence runbook documents dry-run rehearsal before final write.

Tests added or inverted:
- Added direct finalizer dry-run no-write regression.
- Added CLI dry-run output/plumbing regression.
- Extended CRG3 evidence runbook enforcement for finalizer dry-run behavior.

Verification commands:

```bash
python -m pytest tests/infra/test_production_readiness.py -q
python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md --allow-open-id CRG3
python -m pytest tests/infra/test_production_readiness.py tests/infra/test_deployment_hardening_docs.py -q
python -c "import sys, pytest; from tools.security.production_readiness import FOCUSED_P0_TESTS; sys.exit(pytest.main([*FOCUSED_P0_TESTS, '-q']))"
python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md
```

Results:
- `python -m pytest tests/infra/test_production_readiness.py -q` passed: 98 passed in 7.26s.
- `python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md --allow-open-id CRG3` passed.
- `python -m pytest tests/infra/test_production_readiness.py tests/infra/test_deployment_hardening_docs.py -q` passed: 102 passed in 7.11s.
- Focused P0 suite passed: 359 passed, 4 warnings in 43.96s.
- Strict roadmap validation failed only because `CRG3` is still `In progress`; a Docker-capable GitHub Actions evidence run is still required.

Rollback plan:
- Revert this slice only if finalization moves to an external transaction system; preserve a no-write validation mode in any replacement.

Residual risk:
- Dry-run finalization helps rehearse CRG3 closure but does not replace the actual finalizer write after a Docker-capable GitHub Actions evidence run.

Live-block status:
- `CRG3` remains the only open P0 ledger item; production live enablement remains blocked until strict roadmap validation passes with `CRG3` marked `Verified YYYY-MM-DD`.

### 2026-06-04 Pass 6 CRG3 Machine-Readable Evidence Slice

Ledger IDs changed:
- `CRG3` remains in progress; finalizer can now emit an optional machine-readable evidence JSON alongside the roadmap update.

Files changed:
- `tools/security/finalize_crg3.py`
- `tests/infra/test_production_readiness.py`
- `docs/CRG3_CI_EVIDENCE.md`
- `PRODUCTION_REFACTOR_ROADMAP.md`

Safety invariants proven:
- Finalizer `--evidence-json` records the validated CRG3 proof as deterministic JSON with ledger ID, verification date, repository, run URL, run ID, run attempt, commit SHA, clean local HEAD, SBOM artifact, workflow contract, and completed check names.
- The JSON proof is generated from the same validated inputs used for the roadmap update.
- Finalizer dry-run remains no-write: it does not mutate the roadmap or create an evidence JSON file.
- CRG3 evidence runbook documents the optional evidence JSON handoff path.

Tests added or inverted:
- Added finalizer regression proving successful finalization writes the machine-readable evidence JSON.
- Added dry-run regression proving `dry_run=True` does not create the evidence JSON file.
- Extended CLI plumbing regression to cover `--evidence-json`.
- Extended CRG3 evidence runbook enforcement for optional evidence JSON and dry-run no-write behavior.

Verification commands:

```bash
python -m pytest tests/infra/test_production_readiness.py -q
python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md --allow-open-id CRG3
python -m pytest tests/infra/test_production_readiness.py tests/infra/test_deployment_hardening_docs.py -q
python -c "import sys, pytest; from tools.security.production_readiness import FOCUSED_P0_TESTS; sys.exit(pytest.main([*FOCUSED_P0_TESTS, '-q']))"
python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md
```

Results:
- `python -m pytest tests/infra/test_production_readiness.py -q` passed: 100 passed in 10.07s.
- `python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md --allow-open-id CRG3` passed.
- `python -m pytest tests/infra/test_production_readiness.py tests/infra/test_deployment_hardening_docs.py -q` passed: 104 passed in 8.82s.
- Focused P0 suite passed: 361 passed, 4 warnings in 50.00s.
- Strict roadmap validation failed only because `CRG3` is still `In progress`; a Docker-capable GitHub Actions evidence run is still required.

Rollback plan:
- Revert this slice only if CRG3 evidence is moved to a stronger signed artifact format; preserve a machine-readable proof path in any replacement.

Residual risk:
- Machine-readable local evidence improves handoff/auditability but still requires the Docker-capable GitHub Actions run and downloaded artifact validation to close CRG3.

Live-block status:
- `CRG3` remains the only open P0 ledger item; production live enablement remains blocked until strict roadmap validation passes with `CRG3` marked `Verified YYYY-MM-DD`.

### 2026-06-04 Pass 6 CRG3 Atomic Finalizer Write Slice

Ledger IDs changed:
- `CRG3` remains in progress; finalizer roadmap and optional evidence JSON writes are now file-atomic.

Files changed:
- `tools/security/finalize_crg3.py`
- `tests/infra/test_production_readiness.py`
- `docs/CRG3_CI_EVIDENCE.md`
- `PRODUCTION_REFACTOR_ROADMAP.md`

Safety invariants proven:
- Finalizer writes use same-directory temporary files followed by `os.replace`.
- A failed replace leaves the previous roadmap/evidence file content intact.
- Temporary files are cleaned up on replace failure.
- CRG3 evidence runbook documents atomic finalizer writes as part of the handoff contract.

Tests added or inverted:
- Added atomic write regression proving existing file content is replaced on success.
- Added replace-failure regression proving the original file remains intact and temporary files are removed.
- Extended CRG3 evidence runbook enforcement for atomic finalizer write behavior.

Verification commands:

```bash
python -m pytest tests/infra/test_production_readiness.py -q
python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md --allow-open-id CRG3
python -m pytest tests/infra/test_production_readiness.py tests/infra/test_deployment_hardening_docs.py -q
python -c "import sys, pytest; from tools.security.production_readiness import FOCUSED_P0_TESTS; sys.exit(pytest.main([*FOCUSED_P0_TESTS, '-q']))"
python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md
```

Results:
- `python -m pytest tests/infra/test_production_readiness.py -q` passed: 102 passed in 7.95s.
- `python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md --allow-open-id CRG3` passed.
- `python -m pytest tests/infra/test_production_readiness.py tests/infra/test_deployment_hardening_docs.py -q` passed: 106 passed in 9.19s.
- Focused P0 suite passed: 363 passed, 4 warnings in 37.83s.
- Strict roadmap validation failed only because `CRG3` is still `In progress`; a Docker-capable GitHub Actions evidence run is still required.

Rollback plan:
- Revert this slice only if finalizer writes move to a stronger transaction manager; preserve partial-write protection in any replacement.

Residual risk:
- File-level atomic writes reduce local mutation risk but still do not replace the required Docker-capable GitHub Actions run and downloaded artifact validation needed to close CRG3.

Live-block status:
- `CRG3` remains the only open P0 ledger item; production live enablement remains blocked until strict roadmap validation passes with `CRG3` marked `Verified YYYY-MM-DD`.

### 2026-06-04 Pass 6 CRG3 Evidence Artifact Metadata Slice

Ledger IDs changed:
- `CRG3` remains in progress; machine-readable finalizer evidence now records the validated SBOM artifact binding metadata.

Files changed:
- `tools/security/finalize_crg3.py`
- `tests/infra/test_production_readiness.py`
- `docs/CRG3_CI_EVIDENCE.md`
- `PRODUCTION_REFACTOR_ROADMAP.md`

Safety invariants proven:
- GitHub evidence verification returns the validated `production-sbom` artifact ID, bounded size, and canonical download URL instead of discarding them after validation.
- Optional CRG3 evidence JSON preserves the artifact name and adds artifact metadata for audit/diff/archive workflows.
- Finalizer-generated roadmap reports include artifact ID, artifact size, and canonical download URL evidence.
- CRG3 evidence runbook documents artifact metadata in the optional JSON proof.

Tests added or inverted:
- Extended machine-readable evidence JSON regression to assert artifact ID, size, and download URL.
- Extended finalizer-generated roadmap evidence assertions for artifact ID, size, and download URL.
- Extended CRG3 evidence runbook enforcement for artifact metadata proof.

Verification commands:

```bash
python -m pytest tests/infra/test_production_readiness.py -q
python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md --allow-open-id CRG3
python -m pytest tests/infra/test_production_readiness.py tests/infra/test_deployment_hardening_docs.py -q
python -c "import sys, pytest; from tools.security.production_readiness import FOCUSED_P0_TESTS; sys.exit(pytest.main([*FOCUSED_P0_TESTS, '-q']))"
python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md
```

Results:
- `python -m pytest tests/infra/test_production_readiness.py -q` passed: 102 passed in 11.66s.
- `python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md --allow-open-id CRG3` passed.
- `python -m pytest tests/infra/test_production_readiness.py tests/infra/test_deployment_hardening_docs.py -q` passed: 106 passed in 3.79s.
- Focused P0 suite passed: 363 passed, 4 warnings in 33.01s.
- Strict roadmap validation failed only because `CRG3` is still `In progress`; a Docker-capable GitHub Actions evidence run is still required.

Rollback plan:
- Revert this slice only if artifact evidence moves to a stronger signed provenance format; preserve validated artifact identity/size/URL recording in any replacement.

Residual risk:
- Artifact metadata recording improves auditability but still does not replace the required Docker-capable GitHub Actions run and downloaded artifact validation needed to close CRG3.

Live-block status:
- `CRG3` remains the only open P0 ledger item; production live enablement remains blocked until strict roadmap validation passes with `CRG3` marked `Verified YYYY-MM-DD`.

### 2026-06-04 Pass 6 CRG3 Evidence Write-Ordering Slice

Ledger IDs changed:
- `CRG3` remains in progress; finalizer write ordering now prevents orphaned optional evidence JSON when roadmap mutation fails.

Files changed:
- `tools/security/finalize_crg3.py`
- `tests/infra/test_production_readiness.py`
- `docs/CRG3_CI_EVIDENCE.md`
- `PRODUCTION_REFACTOR_ROADMAP.md`

Safety invariants proven:
- Finalizer writes the roadmap before optional machine-readable evidence JSON.
- A simulated roadmap write failure leaves the original roadmap content intact.
- A simulated roadmap write failure does not create the optional CRG3 evidence JSON.
- CRG3 evidence runbook documents the roadmap-before-JSON ordering guarantee.

Tests added or inverted:
- Added finalizer regression that fails if optional evidence JSON is written before a failing roadmap update.
- Extended CRG3 evidence runbook enforcement for the ordering guarantee.

Verification commands:

```bash
python -m pytest tests/infra/test_production_readiness.py -q
python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md --allow-open-id CRG3
python -m pytest tests/infra/test_production_readiness.py tests/infra/test_deployment_hardening_docs.py -q
python -c "import sys, pytest; from tools.security.production_readiness import FOCUSED_P0_TESTS; sys.exit(pytest.main([*FOCUSED_P0_TESTS, '-q']))"
python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md
```

Results:
- `python -m pytest tests/infra/test_production_readiness.py -q` passed: 103 passed in 5.15s.
- `python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md --allow-open-id CRG3` passed.
- `python -m pytest tests/infra/test_production_readiness.py tests/infra/test_deployment_hardening_docs.py -q` passed: 107 passed in 3.96s.
- Focused P0 suite passed: 364 passed, 4 warnings in 32.97s.
- Strict roadmap validation failed only because `CRG3` is still `In progress`; a Docker-capable GitHub Actions evidence run is still required.

Rollback plan:
- Revert this slice only if finalizer outputs move to a true multi-file transaction mechanism; preserve protection against orphaned evidence JSON in any replacement.

Residual risk:
- Write ordering reduces local evidence mutation risk but still does not replace the required Docker-capable GitHub Actions run and downloaded artifact validation needed to close CRG3.

Live-block status:
- `CRG3` remains the only open P0 ledger item; production live enablement remains blocked until strict roadmap validation passes with `CRG3` marked `Verified YYYY-MM-DD`.

### 2026-06-04 Pass 6 CRG3 Evidence Path Alias Guard Slice

Ledger IDs changed:
- `CRG3` remains in progress; finalizer now rejects optional evidence JSON paths that alias the roadmap.

Files changed:
- `tools/security/finalize_crg3.py`
- `tests/infra/test_production_readiness.py`
- `docs/CRG3_CI_EVIDENCE.md`
- `PRODUCTION_REFACTOR_ROADMAP.md`

Safety invariants proven:
- `--evidence-json` cannot point at the same resolved path as `PRODUCTION_REFACTOR_ROADMAP.md`.
- The finalizer rejects the alias before mutating roadmap content.
- CRG3 evidence runbook documents the alias guard.

Tests added or inverted:
- Added finalizer regression proving `--evidence-json` aliasing the roadmap is rejected and leaves the roadmap unchanged.
- Extended CRG3 evidence runbook enforcement for the alias guard.

Verification commands:

```bash
python -m pytest tests/infra/test_production_readiness.py -q
python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md --allow-open-id CRG3
python -m pytest tests/infra/test_production_readiness.py tests/infra/test_deployment_hardening_docs.py -q
python -c "import sys, pytest; from tools.security.production_readiness import FOCUSED_P0_TESTS; sys.exit(pytest.main([*FOCUSED_P0_TESTS, '-q']))"
python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md
```

Results:
- `python -m pytest tests/infra/test_production_readiness.py -q` passed: 104 passed in 4.36s.
- `python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md --allow-open-id CRG3` passed.
- `python -m pytest tests/infra/test_production_readiness.py tests/infra/test_deployment_hardening_docs.py -q` passed: 108 passed in 3.80s.
- Focused P0 suite passed: 365 passed, 4 warnings in 31.73s.
- Strict roadmap validation failed only because `CRG3` is still `In progress`; a Docker-capable GitHub Actions evidence run is still required.

Rollback plan:
- Revert this slice only if finalizer output paths move to a stronger typed destination model; preserve roadmap/evidence path separation in any replacement.

Residual risk:
- Path alias rejection prevents a local finalizer footgun but still does not replace the required Docker-capable GitHub Actions run and downloaded artifact validation needed to close CRG3.

Live-block status:
- `CRG3` remains the only open P0 ledger item; production live enablement remains blocked until strict roadmap validation passes with `CRG3` marked `Verified YYYY-MM-DD`.

### 2026-06-04 Pass 6 CRG3 Evidence JSON Suffix Guard Slice

Ledger IDs changed:
- `CRG3` remains in progress; finalizer now requires optional machine-readable evidence output to use a `.json` suffix.

Files changed:
- `tools/security/finalize_crg3.py`
- `tests/infra/test_production_readiness.py`
- `docs/CRG3_CI_EVIDENCE.md`
- `PRODUCTION_REFACTOR_ROADMAP.md`

Safety invariants proven:
- `--evidence-json` rejects non-JSON file suffixes before GitHub evidence checks or roadmap mutation.
- Rejected non-JSON evidence output paths leave the roadmap unchanged.
- CRG3 evidence runbook documents the `.json` suffix requirement.

Tests added or inverted:
- Added finalizer regression proving `--evidence-json crg3-evidence.txt` is rejected and leaves the roadmap unchanged.
- Extended CRG3 evidence runbook enforcement for the `.json` suffix requirement.

Verification commands:

```bash
python -m pytest tests/infra/test_production_readiness.py -q
python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md --allow-open-id CRG3
python -m pytest tests/infra/test_production_readiness.py tests/infra/test_deployment_hardening_docs.py -q
python -c "import sys, pytest; from tools.security.production_readiness import FOCUSED_P0_TESTS; sys.exit(pytest.main([*FOCUSED_P0_TESTS, '-q']))"
python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md
```

Results:
- `python -m pytest tests/infra/test_production_readiness.py -q` passed: 105 passed in 4.28s.
- `python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md --allow-open-id CRG3` passed.
- `python -m pytest tests/infra/test_production_readiness.py tests/infra/test_deployment_hardening_docs.py -q` passed: 109 passed in 3.86s.
- Focused P0 suite passed: 366 passed, 4 warnings in 41.04s.
- Strict roadmap validation failed only because `CRG3` is still `In progress`; a Docker-capable GitHub Actions evidence run is still required.

Rollback plan:
- Revert this slice only if evidence output paths move to a stronger typed artifact store; preserve clear machine-readable evidence file typing in any replacement.

Residual risk:
- JSON suffix validation prevents misleading local evidence destinations but still does not replace the required Docker-capable GitHub Actions run and downloaded artifact validation needed to close CRG3.

Live-block status:
- `CRG3` remains the only open P0 ledger item; production live enablement remains blocked until strict roadmap validation passes with `CRG3` marked `Verified YYYY-MM-DD`.

### 2026-06-04 Pass 6 CRG3 Evidence Schema Slice

Ledger IDs changed:
- `CRG3` remains in progress; optional machine-readable evidence now has a repo-local schema-backed proof shape.

Files changed:
- `tools/security/finalize_crg3.py`
- `docs/CRG3_EVIDENCE_SCHEMA.json`
- `docs/CRG3_CI_EVIDENCE.md`
- `tests/infra/test_production_readiness.py`
- `PRODUCTION_REFACTOR_ROADMAP.md`

Safety invariants proven:
- CRG3 evidence JSON self-identifies `docs/CRG3_EVIDENCE_SCHEMA.json` and `schema_version: 1`.
- The schema requires the validated run metadata, clean local checkout evidence, artifact metadata, workflow contract, and completed check list.
- The generated evidence JSON is tested against the repo-local schema shape.
- CRG3 evidence runbook documents the schema-backed proof.

Tests added or inverted:
- Added schema shape validation for generated CRG3 evidence JSON.
- Extended machine-readable evidence regression to assert the schema identifier.
- Extended CRG3 evidence runbook enforcement for schema-backed JSON proof.

Verification commands:

```bash
python -m pytest tests/infra/test_production_readiness.py -q
python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md --allow-open-id CRG3
python -m pytest tests/infra/test_production_readiness.py tests/infra/test_deployment_hardening_docs.py -q
python -c "import sys, pytest; from tools.security.production_readiness import FOCUSED_P0_TESTS; sys.exit(pytest.main([*FOCUSED_P0_TESTS, '-q']))"
python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md
```

Results:
- `python -m pytest tests/infra/test_production_readiness.py -q` passed: 105 passed in 4.74s.
- `python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md --allow-open-id CRG3` passed.
- `python -m pytest tests/infra/test_production_readiness.py tests/infra/test_deployment_hardening_docs.py -q` passed: 109 passed in 3.92s.
- Focused P0 suite passed: 366 passed, 4 warnings in 33.17s.
- Strict roadmap validation failed only because `CRG3` is still `In progress`; a Docker-capable GitHub Actions evidence run is still required.

Rollback plan:
- Revert this slice only if CRG3 evidence moves to a stronger signed provenance format; preserve a stable machine-readable schema contract in any replacement.

Residual risk:
- Schema-backed local evidence improves auditability but still does not replace the required Docker-capable GitHub Actions run and downloaded artifact validation needed to close CRG3.

Live-block status:
- `CRG3` remains the only open P0 ledger item; production live enablement remains blocked until strict roadmap validation passes with `CRG3` marked `Verified YYYY-MM-DD`.

### 2026-06-04 Pass 6 CRG3 Finalizer Schema Validation Slice

Ledger IDs changed:
- `CRG3` remains in progress; finalizer now validates generated machine-readable evidence against the repo-local schema before writing.

Files changed:
- `tools/security/finalize_crg3.py`
- `tests/infra/test_production_readiness.py`
- `docs/CRG3_CI_EVIDENCE.md`
- `PRODUCTION_REFACTOR_ROADMAP.md`

Safety invariants proven:
- Generated CRG3 evidence JSON is validated against `docs/CRG3_EVIDENCE_SCHEMA.json` inside the finalizer before file mutation.
- Evidence schema mismatch prevents roadmap mutation.
- Evidence schema mismatch prevents optional evidence JSON creation.
- CRG3 evidence runbook documents finalizer-side schema validation.

Tests added or inverted:
- Added finalizer regression with a temporary stricter schema proving schema mismatch leaves roadmap and evidence JSON untouched.
- Extended CRG3 evidence runbook enforcement for finalizer-side schema validation.

Verification commands:

```bash
python -m pytest tests/infra/test_production_readiness.py -q
python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md --allow-open-id CRG3
python -m pytest tests/infra/test_production_readiness.py tests/infra/test_deployment_hardening_docs.py -q
python -c "import sys, pytest; from tools.security.production_readiness import FOCUSED_P0_TESTS; sys.exit(pytest.main([*FOCUSED_P0_TESTS, '-q']))"
python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md
```

Results:
- `python -m pytest tests/infra/test_production_readiness.py -q` passed: 106 passed in 5.22s.
- `python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md --allow-open-id CRG3` passed.
- `python -m pytest tests/infra/test_production_readiness.py tests/infra/test_deployment_hardening_docs.py -q` passed: 110 passed in 3.99s.
- Focused P0 suite passed: 367 passed, 4 warnings in 31.33s.
- Strict roadmap validation failed only because `CRG3` is still `In progress`; a Docker-capable GitHub Actions evidence run is still required.

Rollback plan:
- Revert this slice only if finalizer validation moves to a full JSON Schema engine; preserve pre-write evidence-shape validation in any replacement.

Residual risk:
- Finalizer-side schema validation improves local evidence integrity but still does not replace the required Docker-capable GitHub Actions run and downloaded artifact validation needed to close CRG3.

Live-block status:
- `CRG3` remains the only open P0 ledger item; production live enablement remains blocked until strict roadmap validation passes with `CRG3` marked `Verified YYYY-MM-DD`.

### 2026-06-04 Pass 6 CRG3 Schema Readiness Gate Slice

Ledger IDs changed:
- `CRG3` remains in progress; production readiness now directly validates the CRG3 evidence schema artifact.

Files changed:
- `tools/security/check_crg3_evidence_schema.py`
- `tools/security/production_readiness.py`
- `tests/infra/test_production_readiness.py`
- `docs/PRODUCTION_READINESS.md`
- `docs/CRG3_CI_EVIDENCE.md`
- `PRODUCTION_REFACTOR_ROADMAP.md`

Safety invariants proven:
- Production readiness includes `CRG3 evidence schema validation` as a named gate.
- The schema checker validates a representative CRG3 evidence record through the same finalizer-side schema validator.
- The CRG3 evidence runbook and production readiness docs document the schema checker.

Tests added or inverted:
- Added readiness wrapper regression proving the schema validation gate is listed in dry-run output.
- Added `_checks()` regression proving the schema validation command is wired into readiness.
- Added script-level regression proving `tools/security/check_crg3_evidence_schema.py` passes.
- Extended CRG3 evidence and production readiness docs enforcement for the schema checker.

Verification commands:

```bash
python -m pytest tests/infra/test_production_readiness.py -q
python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md --allow-open-id CRG3
python -m pytest tests/infra/test_production_readiness.py tests/infra/test_deployment_hardening_docs.py -q
python -c "import sys, pytest; from tools.security.production_readiness import FOCUSED_P0_TESTS; sys.exit(pytest.main([*FOCUSED_P0_TESTS, '-q']))"
python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md
```

Results:
- `python tools/security/check_crg3_evidence_schema.py` passed.
- `python -m pytest tests/infra/test_production_readiness.py -q` passed: 108 passed in 4.43s.
- `python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md --allow-open-id CRG3` passed.
- `python -m pytest tests/infra/test_production_readiness.py tests/infra/test_deployment_hardening_docs.py -q` passed: 112 passed in 4.92s.
- Focused P0 suite passed: 369 passed, 4 warnings in 28.20s.
- Strict roadmap validation failed only because `CRG3` is still `In progress`; a Docker-capable GitHub Actions evidence run is still required.

Rollback plan:
- Revert this slice only if schema validation is folded into a stronger dedicated provenance validator; preserve an explicit readiness gate for CRG3 evidence-shape integrity.

Residual risk:
- Schema readiness validation improves local and CI evidence integrity but still does not replace the required Docker-capable GitHub Actions run and downloaded artifact validation needed to close CRG3.

Live-block status:
- `CRG3` remains the only open P0 ledger item; production live enablement remains blocked until strict roadmap validation passes with `CRG3` marked `Verified YYYY-MM-DD`.

### 2026-06-04 Pass 6 CRG3 Finalizer Schema Gate Evidence Slice

Ledger IDs changed:
- `CRG3` remains in progress; finalizer-generated CRG3 evidence now records the schema readiness gate.

Files changed:
- `tools/security/finalize_crg3.py`
- `tests/infra/test_production_readiness.py`
- `PRODUCTION_REFACTOR_ROADMAP.md`

Safety invariants proven:
- Optional machine-readable evidence JSON records `CRG3 evidence schema validation` in its completed check list.
- Finalizer-generated roadmap proof text records the schema validation gate.
- Finalizer-generated implementation report records schema validation as completed readiness evidence.

Tests added or inverted:
- Extended generated evidence JSON regression to assert the schema validation gate is present.
- Extended finalizer-generated roadmap/report assertions for schema validation evidence.

Verification commands:

```bash
python -m pytest tests/infra/test_production_readiness.py -q
python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md --allow-open-id CRG3
python -m pytest tests/infra/test_production_readiness.py tests/infra/test_deployment_hardening_docs.py -q
python -c "import sys, pytest; from tools.security.production_readiness import FOCUSED_P0_TESTS; sys.exit(pytest.main([*FOCUSED_P0_TESTS, '-q']))"
python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md
```

Results:
- `python -m pytest tests/infra/test_production_readiness.py -q` passed: 108 passed in 4.86s.
- `python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md --allow-open-id CRG3` passed.
- `python -m pytest tests/infra/test_production_readiness.py tests/infra/test_deployment_hardening_docs.py -q` passed: 112 passed in 4.76s.
- Focused P0 suite passed: 369 passed, 4 warnings in 36.47s.
- Strict roadmap validation failed only because `CRG3` is still `In progress`; a Docker-capable GitHub Actions evidence run is still required.

Rollback plan:
- Revert this slice only if schema validation is removed from readiness; otherwise preserve schema gate evidence in generated finalizer artifacts.

Residual risk:
- Finalizer evidence alignment improves audit completeness but still does not replace the required Docker-capable GitHub Actions run and downloaded artifact validation needed to close CRG3.

Live-block status:
- `CRG3` remains the only open P0 ledger item; production live enablement remains blocked until strict roadmap validation passes with `CRG3` marked `Verified YYYY-MM-DD`.

### 2026-06-04 Pass 6 CRG3 Evidence Check Drift Guard Slice

Ledger IDs changed:
- `CRG3` remains in progress; finalizer-generated machine-readable evidence now tracks the readiness wrapper check names directly.

Files changed:
- `tools/security/finalize_crg3.py`
- `tests/infra/test_production_readiness.py`
- `docs/CRG3_CI_EVIDENCE.md`
- `docs/PRODUCTION_READINESS.md`
- `PRODUCTION_REFACTOR_ROADMAP.md`

Safety invariants proven:
- CRG3 evidence JSON cannot drift away from `tools/security/production_readiness.py` check names without a regression failure.
- Finalizer evidence includes every CI readiness gate by exact wrapper check name.
- Workflow-only evidence remains explicit for `production-sbom` artifact upload and downloaded CycloneDX SBOM artifact validation.

Tests added or inverted:
- Added a finalizer regression that compares generated CRG3 evidence check names against the current readiness wrapper gates.
- Extended generated evidence JSON assertions to require exact readiness wrapper check coverage.
- Extended CRG3 evidence and production readiness docs enforcement for the check-name tracking contract.

Verification commands:

```bash
python -m pytest tests/infra/test_production_readiness.py -q
python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md --allow-open-id CRG3
python -m pytest tests/infra/test_production_readiness.py tests/infra/test_deployment_hardening_docs.py -q
python -c "import sys, pytest; from tools.security.production_readiness import FOCUSED_P0_TESTS; sys.exit(pytest.main([*FOCUSED_P0_TESTS, '-q']))"
python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md
```

Results:
- `python -m pytest tests/infra/test_production_readiness.py -q` passed: 109 passed in 5.13s.
- `python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md --allow-open-id CRG3` passed.
- `python -m pytest tests/infra/test_production_readiness.py tests/infra/test_deployment_hardening_docs.py -q` passed: 113 passed in 4.38s.
- Focused P0 suite passed: 370 passed, 4 warnings in 36.05s.
- Strict roadmap validation failed only because `CRG3` is still `In progress`; a Docker-capable GitHub Actions evidence run is still required.

Rollback plan:
- Revert this slice only if finalizer evidence moves to a stronger generated provenance bundle; preserve automated coverage that detects drift between readiness gates and final evidence.

Residual risk:
- Check-name drift is now guarded, but CRG3 still needs Docker-capable GitHub Actions evidence and downloaded artifact validation before it can be verified.

Live-block status:
- `CRG3` remains the only open P0 ledger item; production live enablement remains blocked until strict roadmap validation passes with `CRG3` marked `Verified YYYY-MM-DD`.

### 2026-06-04 Pass 6 CRG3 Semantic Evidence Validation Slice

Ledger IDs changed:
- `CRG3` remains in progress; CRG3 evidence validation now requires the exact readiness gate proof set.

Files changed:
- `tools/security/finalize_crg3.py`
- `tools/security/check_crg3_evidence_schema.py`
- `docs/CRG3_EVIDENCE_SCHEMA.json`
- `docs/CRG3_CI_EVIDENCE.md`
- `tests/infra/test_production_readiness.py`
- `PRODUCTION_REFACTOR_ROADMAP.md`

Safety invariants proven:
- CRG3 evidence JSON must contain each production readiness wrapper check exactly once.
- CRG3 evidence JSON must include workflow-only proof for `production-sbom` artifact upload and downloaded CycloneDX SBOM artifact validation.
- CRG3 evidence validation rejects missing, duplicate, or unexpected check names before writing roadmap or evidence files.
- The schema checker validates a full semantically valid sample instead of a placeholder check list.

Tests added or inverted:
- Added finalizer validation regression for missing, duplicate, and unexpected CRG3 evidence check names.
- Extended the lightweight schema assertion to enforce `uniqueItems`.
- Extended CRG3 evidence runbook tests for the exact unique check-name contract.

Verification commands:

```bash
python tools/security/check_crg3_evidence_schema.py
python -m pytest tests/infra/test_production_readiness.py -q
python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md --allow-open-id CRG3
python -m pytest tests/infra/test_production_readiness.py tests/infra/test_deployment_hardening_docs.py -q
python -c "import sys, pytest; from tools.security.production_readiness import FOCUSED_P0_TESTS; sys.exit(pytest.main([*FOCUSED_P0_TESTS, '-q']))"
python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md
```

Results:
- `python tools/security/check_crg3_evidence_schema.py` passed.
- `python -m pytest tests/infra/test_production_readiness.py -q` passed: 110 passed in 4.64s.
- `python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md --allow-open-id CRG3` passed.
- `python -m pytest tests/infra/test_production_readiness.py tests/infra/test_deployment_hardening_docs.py -q` passed: 114 passed in 4.29s.
- Focused P0 suite passed: 371 passed, 4 warnings in 36.03s.
- Strict roadmap validation failed only because `CRG3` is still `In progress`; a Docker-capable GitHub Actions evidence run is still required.

Rollback plan:
- Revert this slice only if CRG3 evidence moves to a stronger signed provenance bundle; preserve semantic validation that prevents missing, duplicate, or unexpected completed-check claims.

Residual risk:
- Semantic evidence validation closes a local proof-integrity gap, but CRG3 still needs Docker-capable GitHub Actions evidence and downloaded artifact validation before it can be verified.

Live-block status:
- `CRG3` remains the only open P0 ledger item; production live enablement remains blocked until strict roadmap validation passes with `CRG3` marked `Verified YYYY-MM-DD`.

### 2026-06-04 Pass 6 CRG3 Human Proof Alignment Slice

Ledger IDs changed:
- `CRG3` remains in progress; finalizer-generated roadmap/report proof now uses the same exact completed check names as machine-readable CRG3 evidence.

Files changed:
- `tools/security/finalize_crg3.py`
- `docs/CRG3_CI_EVIDENCE.md`
- `tests/infra/test_production_readiness.py`
- `PRODUCTION_REFACTOR_ROADMAP.md`

Safety invariants proven:
- Human-readable CRG3 proof cannot silently use paraphrased readiness gate names while the evidence JSON uses exact check names.
- Generated CRG3 roadmap proof includes the canonical completed-check list from the finalizer.
- Generated CRG3 implementation report includes a completed-checks section using the same canonical list.

Tests added or inverted:
- Updated finalizer-generated proof regression to assert every canonical CRG3 evidence check appears in the generated roadmap/report text.
- Extended CRG3 evidence runbook enforcement for exact human-proof/evidence check-name alignment.

Verification commands:

```bash
python tools/security/check_crg3_evidence_schema.py
python -m pytest tests/infra/test_production_readiness.py -q
python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md --allow-open-id CRG3
python -m pytest tests/infra/test_production_readiness.py tests/infra/test_deployment_hardening_docs.py -q
python -c "import sys, pytest; from tools.security.production_readiness import FOCUSED_P0_TESTS; sys.exit(pytest.main([*FOCUSED_P0_TESTS, '-q']))"
python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md
```

Results:
- `python tools/security/check_crg3_evidence_schema.py` passed.
- `python -m pytest tests/infra/test_production_readiness.py -q` passed: 110 passed in 4.53s.
- `python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md --allow-open-id CRG3` passed.
- `python -m pytest tests/infra/test_production_readiness.py tests/infra/test_deployment_hardening_docs.py -q` passed: 114 passed in 12.40s.
- Focused P0 suite passed: 371 passed, 4 warnings in 40.38s.
- Strict roadmap validation failed only because `CRG3` is still `In progress`; a Docker-capable GitHub Actions evidence run is still required.

Rollback plan:
- Revert this slice only if final CRG3 proof moves to a generated signed bundle; preserve one canonical check-name source for human and machine evidence.

Residual risk:
- Human proof alignment reduces finalizer drift risk, but CRG3 still needs Docker-capable GitHub Actions evidence and downloaded artifact validation before it can be verified.

Live-block status:
- `CRG3` remains the only open P0 ledger item; production live enablement remains blocked until strict roadmap validation passes with `CRG3` marked `Verified YYYY-MM-DD`.

### 2026-06-04 Pass 6 CRG3 SBOM Lockfile Semantic Validation Slice

Ledger IDs changed:
- `CRG3` remains in progress; downloaded `production-sbom` artifacts now must prove they match the locked runtime dependency set.

Files changed:
- `tools/security/finalize_crg3.py`
- `tests/infra/test_production_readiness.py`
- `docs/CRG3_CI_EVIDENCE.md`
- `PRODUCTION_REFACTOR_ROADMAP.md`

Safety invariants proven:
- Finalizer validation rejects a downloaded CycloneDX SBOM that omits a package pinned in `requirements.lock`.
- Finalizer validation rejects a downloaded CycloneDX SBOM that includes an unlocked package.
- Generated CRG3 roadmap/report proof now states that SBOM components match `requirements.lock` exactly.
- Valid test SBOM artifacts are built from the same lockfile component set used by finalizer validation.

Tests added or inverted:
- Added finalizer regression for missing locked SBOM components.
- Added finalizer regression for unexpected unlocked SBOM components.
- Extended CRG3 runbook and finalizer-generated proof assertions for exact `requirements.lock` SBOM matching.

Verification commands:

```bash
python tools/security/check_crg3_evidence_schema.py
python -m pytest tests/infra/test_production_readiness.py -q
python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md --allow-open-id CRG3
python -m pytest tests/infra/test_production_readiness.py tests/infra/test_deployment_hardening_docs.py -q
python -c "import sys, pytest; from tools.security.production_readiness import FOCUSED_P0_TESTS; sys.exit(pytest.main([*FOCUSED_P0_TESTS, '-q']))"
python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md
```

Results:
- `python tools/security/check_crg3_evidence_schema.py` passed.
- `python -m pytest tests/infra/test_production_readiness.py -q` passed: 112 passed in 5.75s.
- `python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md --allow-open-id CRG3` passed.
- `python -m pytest tests/infra/test_production_readiness.py tests/infra/test_deployment_hardening_docs.py -q` passed: 116 passed in 6.09s.
- Focused P0 suite passed: 373 passed, 4 warnings in 43.02s.
- Strict roadmap validation failed only because `CRG3` is still `In progress`; a Docker-capable GitHub Actions evidence run is still required.

Rollback plan:
- Revert this slice only if SBOM provenance moves to a stronger signed lockfile attestation; preserve a finalizer check that downloaded SBOM content corresponds to the locked runtime dependencies.

Residual risk:
- SBOM lockfile matching improves downloaded artifact semantics, but CRG3 still needs Docker-capable GitHub Actions evidence and downloaded artifact validation before it can be verified.

Live-block status:
- `CRG3` remains the only open P0 ledger item; production live enablement remains blocked until strict roadmap validation passes with `CRG3` marked `Verified YYYY-MM-DD`.

### 2026-06-04 Pass 6 CRG3 SBOM Component Identity Slice

Ledger IDs changed:
- `CRG3` remains in progress; downloaded SBOM validation now treats component `type` and duplicate identities as part of exact lockfile matching.

Files changed:
- `tools/security/finalize_crg3.py`
- `tests/infra/test_production_readiness.py`
- `docs/CRG3_CI_EVIDENCE.md`
- `PRODUCTION_REFACTOR_ROADMAP.md`

Safety invariants proven:
- SBOM component identity includes `type`, `name`, `version`, and `purl`.
- A component with the right package name/version/purl but wrong CycloneDX type no longer satisfies lockfile matching.
- Duplicate SBOM component identities are rejected before final CRG3 evidence can be written.

Tests added or inverted:
- Added finalizer regression for wrong SBOM component type.
- Added finalizer regression for duplicate SBOM component identity.
- Extended CRG3 evidence runbook enforcement for unique `type`/`name`/`version`/`purl` lockfile matching.

Verification commands:

```bash
python tools/security/check_crg3_evidence_schema.py
python -m pytest tests/infra/test_production_readiness.py -q
python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md --allow-open-id CRG3
python -m pytest tests/infra/test_production_readiness.py tests/infra/test_deployment_hardening_docs.py -q
python -c "import sys, pytest; from tools.security.production_readiness import FOCUSED_P0_TESTS; sys.exit(pytest.main([*FOCUSED_P0_TESTS, '-q']))"
python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md
```

Results:
- `python tools/security/check_crg3_evidence_schema.py` passed.
- `python -m pytest tests/infra/test_production_readiness.py -q` passed: 114 passed in 5.89s.
- `python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md --allow-open-id CRG3` passed.
- `python -m pytest tests/infra/test_production_readiness.py tests/infra/test_deployment_hardening_docs.py -q` passed: 118 passed in 7.98s.
- Focused P0 suite passed: 375 passed, 4 warnings in 38.78s.
- Strict roadmap validation failed only because `CRG3` is still `In progress`; a Docker-capable GitHub Actions evidence run is still required.

Rollback plan:
- Revert this slice only if SBOM validation is replaced with a stronger CycloneDX validator that still preserves exact unique component identity checks.

Residual risk:
- SBOM component identity is now stricter, but CRG3 still needs Docker-capable GitHub Actions evidence and downloaded artifact validation before it can be verified.

Live-block status:
- `CRG3` remains the only open P0 ledger item; production live enablement remains blocked until strict roadmap validation passes with `CRG3` marked `Verified YYYY-MM-DD`.

### 2026-06-04 Pass 6 CRG3 Runtime Lock Digest Evidence Slice

Ledger IDs changed:
- `CRG3` remains in progress; machine-readable evidence now records the runtime `requirements.lock` SHA-256 digest used for SBOM matching.

Files changed:
- `tools/security/finalize_crg3.py`
- `tools/security/check_crg3_evidence_schema.py`
- `docs/CRG3_EVIDENCE_SCHEMA.json`
- `docs/CRG3_CI_EVIDENCE.md`
- `tests/infra/test_production_readiness.py`
- `PRODUCTION_REFACTOR_ROADMAP.md`

Safety invariants proven:
- CRG3 evidence JSON records `runtime_lock.path == requirements.lock`.
- CRG3 evidence JSON records a SHA-256 digest of the exact `requirements.lock` bytes used for SBOM component validation.
- Evidence schema validation rejects missing or malformed runtime lock digest metadata.
- Finalizer-generated human proof includes the lockfile path and SHA-256 digest.

Tests added or inverted:
- Extended generated evidence JSON regression to assert runtime lock metadata and digest shape.
- Added evidence validation regression for missing and malformed runtime lock digest metadata.
- Extended CRG3 evidence runbook and generated-report assertions for runtime lock digest proof.

Verification commands:

```bash
python tools/security/check_crg3_evidence_schema.py
python -m pytest tests/infra/test_production_readiness.py -q
python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md --allow-open-id CRG3
python -m pytest tests/infra/test_production_readiness.py tests/infra/test_deployment_hardening_docs.py -q
python -c "import sys, pytest; from tools.security.production_readiness import FOCUSED_P0_TESTS; sys.exit(pytest.main([*FOCUSED_P0_TESTS, '-q']))"
python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md
```

Results:
- `python tools/security/check_crg3_evidence_schema.py` passed.
- `python -m pytest tests/infra/test_production_readiness.py -q` passed: 115 passed in 5.99s.
- `python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md --allow-open-id CRG3` passed.
- `python -m pytest tests/infra/test_production_readiness.py tests/infra/test_deployment_hardening_docs.py -q` passed: 119 passed in 5.40s.
- Focused P0 suite passed: 376 passed, 4 warnings in 68.78s.
- Strict roadmap validation failed only because `CRG3` is still `In progress`; a Docker-capable GitHub Actions evidence run is still required.

Rollback plan:
- Revert this slice only if runtime lock identity moves to a stronger signed attestation; preserve a digest-backed link between CRG3 evidence and the lockfile used for SBOM validation.

Residual risk:
- Runtime lock digest evidence improves audit reproducibility, but CRG3 still needs Docker-capable GitHub Actions evidence and downloaded artifact validation before it can be verified.

Live-block status:
- `CRG3` remains the only open P0 ledger item; production live enablement remains blocked until strict roadmap validation passes with `CRG3` marked `Verified YYYY-MM-DD`.

### 2026-06-04 Pass 6 CRG3 Evidence Schema Version Slice

Ledger IDs changed:
- `CRG3` remains in progress; CRG3 evidence schema version now reflects the required runtime-lock digest shape.

Files changed:
- `tools/security/finalize_crg3.py`
- `tools/security/check_crg3_evidence_schema.py`
- `docs/CRG3_EVIDENCE_SCHEMA.json`
- `docs/CRG3_CI_EVIDENCE.md`
- `tests/infra/test_production_readiness.py`
- `PRODUCTION_REFACTOR_ROADMAP.md`

Safety invariants proven:
- Finalizer-generated evidence uses a named `CRG3_EVIDENCE_SCHEMA_VERSION` constant.
- `docs/CRG3_EVIDENCE_SCHEMA.json` requires `schema_version: 2`.
- Schema version 2 documents and validates the runtime lockfile digest evidence field.
- A regression test fails if the schema file and finalizer constant drift.

Tests added or inverted:
- Added CRG3 evidence schema-version consistency regression.
- Updated generated evidence JSON and schema checker sample tests to use the finalizer schema-version constant.
- Extended CRG3 evidence runbook enforcement for schema version 2.

Verification commands:

```bash
python tools/security/check_crg3_evidence_schema.py
python -m pytest tests/infra/test_production_readiness.py -q
python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md --allow-open-id CRG3
python -m pytest tests/infra/test_production_readiness.py tests/infra/test_deployment_hardening_docs.py -q
python -c "import sys, pytest; from tools.security.production_readiness import FOCUSED_P0_TESTS; sys.exit(pytest.main([*FOCUSED_P0_TESTS, '-q']))"
python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md
```

Results:
- `python tools/security/check_crg3_evidence_schema.py` passed.
- `python -m pytest tests/infra/test_production_readiness.py -q` passed: 116 passed in 10.21s.
- `python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md --allow-open-id CRG3` passed.
- `python -m pytest tests/infra/test_production_readiness.py tests/infra/test_deployment_hardening_docs.py -q` passed: 120 passed in 8.56s.
- Focused P0 suite passed: 377 passed, 4 warnings in 43.89s.
- Strict roadmap validation failed only because `CRG3` is still `In progress`; a Docker-capable GitHub Actions evidence run is still required.

Rollback plan:
- Revert this slice only if the evidence schema is replaced by a stronger externally versioned provenance format; preserve explicit schema version drift checks.

Residual risk:
- Schema versioning is now explicit, but CRG3 still needs Docker-capable GitHub Actions evidence and downloaded artifact validation before it can be verified.

Live-block status:
- `CRG3` remains the only open P0 ledger item; production live enablement remains blocked until strict roadmap validation passes with `CRG3` marked `Verified YYYY-MM-DD`.

### 2026-06-04 Pass 6 CRG3 Downloaded SBOM Digest Evidence Slice

Ledger IDs changed:
- `CRG3` remains in progress; machine-readable evidence now records the accepted downloaded `sbom.cdx.json` digest and component count.

Files changed:
- `tools/security/finalize_crg3.py`
- `tools/security/check_crg3_evidence_schema.py`
- `docs/CRG3_EVIDENCE_SCHEMA.json`
- `docs/CRG3_CI_EVIDENCE.md`
- `tests/infra/test_production_readiness.py`
- `PRODUCTION_REFACTOR_ROADMAP.md`

Safety invariants proven:
- Finalizer validation returns the SHA-256 digest of the exact downloaded `sbom.cdx.json` bytes it accepted.
- CRG3 evidence JSON records the accepted SBOM digest and component count under `sbom_artifact_metadata`.
- Evidence schema version 3 requires `sbom_sha256` and positive `component_count` metadata.
- Finalizer-generated human proof records the accepted SBOM digest and component count.

Tests added or inverted:
- Extended generated evidence JSON regression to assert accepted SBOM SHA-256 and component count.
- Extended evidence schema validation regression to reject missing/malformed SBOM digest metadata.
- Extended CRG3 runbook and generated-report assertions for downloaded SBOM digest evidence.

Verification commands:

```bash
python tools/security/check_crg3_evidence_schema.py
python -m pytest tests/infra/test_production_readiness.py -q
python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md --allow-open-id CRG3
python -m pytest tests/infra/test_production_readiness.py tests/infra/test_deployment_hardening_docs.py -q
python -c "import sys, pytest; from tools.security.production_readiness import FOCUSED_P0_TESTS; sys.exit(pytest.main([*FOCUSED_P0_TESTS, '-q']))"
python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md
```

Results:
- `python tools/security/check_crg3_evidence_schema.py` passed.
- `python -m pytest tests/infra/test_production_readiness.py -q` passed: 117 passed in 5.10s.
- `python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md --allow-open-id CRG3` passed.
- `python -m pytest tests/infra/test_production_readiness.py tests/infra/test_deployment_hardening_docs.py -q` passed: 121 passed in 4.90s.
- Focused P0 suite passed: 378 passed, 4 warnings in 32.12s.
- Strict roadmap validation failed only because `CRG3` is still `In progress`; a Docker-capable GitHub Actions evidence run is still required.

Rollback plan:
- Revert this slice only if SBOM artifact proof moves to a stronger signed attestation; preserve a digest-backed record of the downloaded SBOM accepted by the finalizer.

Residual risk:
- Downloaded SBOM digest evidence improves audit reproducibility, but CRG3 still needs Docker-capable GitHub Actions evidence and downloaded artifact validation before it can be verified.

Live-block status:
- `CRG3` remains the only open P0 ledger item; production live enablement remains blocked until strict roadmap validation passes with `CRG3` marked `Verified YYYY-MM-DD`.

### 2026-06-04 Pass 6 CRG3 Downloaded Artifact Digest Evidence Slice

Ledger IDs changed:
- `CRG3` remains in progress; machine-readable evidence now records the SHA-256 digest of the downloaded `production-sbom` artifact ZIP as well as the SBOM JSON inside it.

Files changed:
- `tools/security/finalize_crg3.py`
- `tools/security/check_crg3_evidence_schema.py`
- `docs/CRG3_EVIDENCE_SCHEMA.json`
- `docs/CRG3_CI_EVIDENCE.md`
- `tests/infra/test_production_readiness.py`
- `PRODUCTION_REFACTOR_ROADMAP.md`

Safety invariants proven:
- Finalizer validation records the SHA-256 digest of the exact artifact ZIP payload downloaded from GitHub.
- CRG3 evidence JSON records `artifact_sha256`, `sbom_sha256`, and positive `component_count` under `sbom_artifact_metadata`.
- Evidence schema version 4 requires artifact ZIP digest evidence.
- Finalizer-generated human proof records both artifact ZIP and SBOM JSON digests.

Tests added or inverted:
- Extended generated evidence JSON regression to assert artifact ZIP digest metadata.
- Extended evidence schema validation regression to reject missing/malformed artifact digest metadata.
- Extended CRG3 runbook assertions for schema version 4 and artifact ZIP digest proof.

Verification commands:

```bash
python tools/security/check_crg3_evidence_schema.py
python -m pytest tests/infra/test_production_readiness.py -q
python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md --allow-open-id CRG3
python -m pytest tests/infra/test_production_readiness.py tests/infra/test_deployment_hardening_docs.py -q
python -c "import sys, pytest; from tools.security.production_readiness import FOCUSED_P0_TESTS; sys.exit(pytest.main([*FOCUSED_P0_TESTS, '-q']))"
python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md
```

Results:
- `python tools/security/check_crg3_evidence_schema.py` passed.
- `python -m pytest tests/infra/test_production_readiness.py -q` passed: 117 passed in 5.36s.
- `python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md --allow-open-id CRG3` passed.
- `python -m pytest tests/infra/test_production_readiness.py tests/infra/test_deployment_hardening_docs.py -q` passed: 121 passed in 4.98s.
- Focused P0 suite passed: 378 passed, 4 warnings in 40.00s.
- Strict roadmap validation failed only because `CRG3` is still `In progress`; a Docker-capable GitHub Actions evidence run is still required.

Rollback plan:
- Revert this slice only if GitHub artifact proof moves to a stronger signed artifact attestation; preserve a digest-backed record of the downloaded artifact payload accepted by the finalizer.

Residual risk:
- Artifact digest evidence improves audit reproducibility, but CRG3 still needs Docker-capable GitHub Actions evidence and downloaded artifact validation before it can be verified.

Live-block status:
- `CRG3` remains the only open P0 ledger item; production live enablement remains blocked until strict roadmap validation passes with `CRG3` marked `Verified YYYY-MM-DD`.

### 2026-06-04 Pass 6 CRG3 Artifact Download Size Evidence Slice

Ledger IDs changed:
- `CRG3` remains in progress; machine-readable evidence now records the actual downloaded artifact payload length separately from GitHub artifact metadata size.

Files changed:
- `tools/security/finalize_crg3.py`
- `tools/security/check_crg3_evidence_schema.py`
- `docs/CRG3_EVIDENCE_SCHEMA.json`
- `docs/CRG3_CI_EVIDENCE.md`
- `tests/infra/test_production_readiness.py`
- `PRODUCTION_REFACTOR_ROADMAP.md`

Safety invariants proven:
- Finalizer validation records the byte length of the artifact ZIP payload it actually downloaded and accepted.
- CRG3 evidence JSON records `download_size_in_bytes` in `sbom_artifact_metadata`.
- Evidence schema version 5 requires non-negative downloaded-size metadata.
- Finalizer-generated human proof records the downloaded artifact size separately from GitHub API `size_in_bytes`.

Tests added or inverted:
- Extended generated evidence JSON regression to assert downloaded artifact byte length.
- Extended evidence schema validation regression to reject negative downloaded-size metadata.
- Extended CRG3 runbook and generated-report assertions for downloaded artifact byte-length proof.

Verification commands:

```bash
python tools/security/check_crg3_evidence_schema.py
python -m pytest tests/infra/test_production_readiness.py -q
python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md --allow-open-id CRG3
python -m pytest tests/infra/test_production_readiness.py tests/infra/test_deployment_hardening_docs.py -q
python -c "import sys, pytest; from tools.security.production_readiness import FOCUSED_P0_TESTS; sys.exit(pytest.main([*FOCUSED_P0_TESTS, '-q']))"
python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md
```

Results:
- `python tools/security/check_crg3_evidence_schema.py` passed.
- `python -m pytest tests/infra/test_production_readiness.py -q` passed: 117 passed in 5.02s.
- `python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md --allow-open-id CRG3` passed.
- `python -m pytest tests/infra/test_production_readiness.py tests/infra/test_deployment_hardening_docs.py -q` passed: 121 passed in 4.82s.
- Focused P0 suite passed: 378 passed, 4 warnings in 43.06s.
- Strict roadmap validation failed only because `CRG3` is still `In progress`; a Docker-capable GitHub Actions evidence run is still required.

Rollback plan:
- Revert this slice only if artifact evidence moves to a stronger signed attestation; preserve recording of the actual downloaded artifact payload accepted by the finalizer.

Residual risk:
- Downloaded-size evidence improves audit reproducibility, but CRG3 still needs Docker-capable GitHub Actions evidence and downloaded artifact validation before it can be verified.

Live-block status:
- `CRG3` remains the only open P0 ledger item; production live enablement remains blocked until strict roadmap validation passes with `CRG3` marked `Verified YYYY-MM-DD`.

### 2026-06-04 Pass 6 CRG3 Provenance Metadata Semantic Validation Slice

Ledger IDs changed:
- `CRG3` remains in progress; CRG3 evidence validation now rejects provenance metadata that is schema-valid but stale relative to the current lockfile.

Files changed:
- `tools/security/finalize_crg3.py`
- `tests/infra/test_production_readiness.py`
- `docs/CRG3_CI_EVIDENCE.md`
- `PRODUCTION_REFACTOR_ROADMAP.md`

Safety invariants proven:
- CRG3 evidence JSON `runtime_lock` metadata must match the current clean-checkout `requirements.lock` path and SHA-256 digest.
- CRG3 evidence JSON `sbom_artifact_metadata.component_count` must match the runtime component set derived from `requirements.lock`.
- Evidence validation rejects valid-shaped but semantically stale runtime-lock digest and component-count metadata before writing CRG3 evidence.

Tests added or inverted:
- Added evidence validation regression for a syntactically valid but wrong `runtime_lock.sha256`.
- Added evidence validation regression for a positive but wrong SBOM component count.
- Extended CRG3 runbook enforcement for finalizer-side provenance metadata validation.

Verification commands:

```bash
python tools/security/check_crg3_evidence_schema.py
python -m pytest tests/infra/test_production_readiness.py -q
python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md --allow-open-id CRG3
python -m pytest tests/infra/test_production_readiness.py tests/infra/test_deployment_hardening_docs.py -q
python -c "import sys, pytest; from tools.security.production_readiness import FOCUSED_P0_TESTS; sys.exit(pytest.main([*FOCUSED_P0_TESTS, '-q']))"
python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md
```

Results:
- `python tools/security/check_crg3_evidence_schema.py` passed.
- `python -m pytest tests/infra/test_production_readiness.py -q` passed: 117 passed in 9.02s.
- `python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md --allow-open-id CRG3` passed.
- `python -m pytest tests/infra/test_production_readiness.py tests/infra/test_deployment_hardening_docs.py -q` passed: 121 passed in 4.81s.
- Focused P0 suite passed: 378 passed, 4 warnings in 31.46s.
- Strict roadmap validation failed only because `CRG3` is still `In progress`; a Docker-capable GitHub Actions evidence run is still required.

Rollback plan:
- Revert this slice only if CRG3 evidence moves to a signed external attestation that still proves the runtime lockfile digest and SBOM component-count binding.

Residual risk:
- Semantic provenance validation improves local evidence integrity, but CRG3 still needs Docker-capable GitHub Actions evidence and downloaded artifact validation before it can be verified.

Live-block status:
- `CRG3` remains the only open P0 ledger item; production live enablement remains blocked until strict roadmap validation passes with `CRG3` marked `Verified YYYY-MM-DD`.

### 2026-06-04 Pass 6 CRG3 Evidence Schema Bounded Size Slice

Ledger IDs changed:
- `CRG3` remains in progress; CRG3 evidence schema now bounds artifact size metadata to the same limit enforced by finalizer artifact download validation.

Files changed:
- `tools/security/finalize_crg3.py`
- `docs/CRG3_EVIDENCE_SCHEMA.json`
- `docs/CRG3_CI_EVIDENCE.md`
- `tests/infra/test_production_readiness.py`
- `PRODUCTION_REFACTOR_ROADMAP.md`

Safety invariants proven:
- CRG3 evidence schema version 6 requires `sbom_artifact_metadata.size_in_bytes <= 5242880`.
- CRG3 evidence schema version 6 requires `sbom_artifact_metadata.download_size_in_bytes <= 5242880`.
- Finalizer schema validation now enforces integer `maximum` constraints, so oversized evidence metadata is rejected before writing evidence.

Tests added or inverted:
- Added evidence schema regression asserting artifact size and downloaded-size maximums match `MAX_SBOM_ARTIFACT_BYTES`.
- Added evidence validation regressions for oversized `size_in_bytes` and `download_size_in_bytes`.
- Extended CRG3 runbook enforcement for schema version 6 and bounded artifact size/download-size evidence.

Verification commands:

```bash
python tools/security/check_crg3_evidence_schema.py
python -m pytest tests/infra/test_production_readiness.py -q
python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md --allow-open-id CRG3
python -m pytest tests/infra/test_production_readiness.py tests/infra/test_deployment_hardening_docs.py -q
python -c "import sys, pytest; from tools.security.production_readiness import FOCUSED_P0_TESTS; sys.exit(pytest.main([*FOCUSED_P0_TESTS, '-q']))"
python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md
```

Results:
- `python tools/security/check_crg3_evidence_schema.py` passed.
- `python -m pytest tests/infra/test_production_readiness.py -q` passed: 117 passed in 7.62s.
- `python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md --allow-open-id CRG3` passed.
- `python -m pytest tests/infra/test_production_readiness.py tests/infra/test_deployment_hardening_docs.py -q` passed: 121 passed in 5.41s.
- Focused P0 suite passed: 378 passed, 4 warnings in 42.59s.
- Strict roadmap validation failed only because `CRG3` is still `In progress`; a Docker-capable GitHub Actions evidence run is still required.

Rollback plan:
- Revert this slice only if CRG3 evidence size bounds move to a stronger externally validated attestation; preserve a machine-readable limit on accepted artifact evidence sizes.

Residual risk:
- Schema-level size bounds improve CRG3 evidence integrity, but CRG3 still needs Docker-capable GitHub Actions evidence and downloaded artifact validation before it can be verified.

Live-block status:
- `CRG3` remains the only open P0 ledger item; production live enablement remains blocked until strict roadmap validation passes with `CRG3` marked `Verified YYYY-MM-DD`.

### 2026-06-04 Pass 6 CRG3 Evidence Identity Semantic Validation Slice

Ledger IDs changed:
- `CRG3` remains in progress; CRG3 evidence validation now rejects records whose identity fields disagree with the workflow run and artifact URLs.

Files changed:
- `tools/security/finalize_crg3.py`
- `docs/CRG3_CI_EVIDENCE.md`
- `tests/infra/test_production_readiness.py`
- `PRODUCTION_REFACTOR_ROADMAP.md`

Safety invariants proven:
- CRG3 evidence `repository` must match the owner/repo embedded in `run_url`.
- CRG3 evidence `run_id` must match the run ID embedded in `run_url`.
- CRG3 evidence `local_head` must match `commit_sha`.
- CRG3 evidence `sbom_artifact_metadata.download_url` and `sbom_artifact_metadata.id` must belong to the same workflow repository and artifact ID.

Tests added or inverted:
- Added evidence validation regressions for mismatched repository, run ID, local head, artifact download repository, and artifact ID.
- Extended CRG3 runbook enforcement for finalizer-side identity consistency validation.

Verification commands:

```bash
python tools/security/check_crg3_evidence_schema.py
python -m pytest tests/infra/test_production_readiness.py -q
python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md --allow-open-id CRG3
python -m pytest tests/infra/test_production_readiness.py tests/infra/test_deployment_hardening_docs.py -q
python -c "import sys, pytest; from tools.security.production_readiness import FOCUSED_P0_TESTS; sys.exit(pytest.main([*FOCUSED_P0_TESTS, '-q']))"
python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md
```

Results:
- `python tools/security/check_crg3_evidence_schema.py` passed.
- `python -m pytest tests/infra/test_production_readiness.py -q` passed: 118 passed in 5.38s.
- `python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md --allow-open-id CRG3` passed.
- `python -m pytest tests/infra/test_production_readiness.py tests/infra/test_deployment_hardening_docs.py -q` passed: 122 passed in 5.03s.
- Focused P0 suite passed: 379 passed, 4 warnings in 43.99s.
- Strict roadmap validation failed only because `CRG3` is still `In progress`; a Docker-capable GitHub Actions evidence run is still required.

Rollback plan:
- Revert this slice only if CRG3 evidence identity binding moves to a stronger externally signed attestation; preserve finalizer rejection of mismatched repo/run/artifact proof.

Residual risk:
- Identity consistency validation improves local evidence integrity, but CRG3 still needs Docker-capable GitHub Actions evidence and downloaded artifact validation before it can be verified.

Live-block status:
- `CRG3` remains the only open P0 ledger item; production live enablement remains blocked until strict roadmap validation passes with `CRG3` marked `Verified YYYY-MM-DD`.

### 2026-06-04 Pass 6 CRG3 Verified Date Semantic Validation Slice

Ledger IDs changed:
- `CRG3` remains in progress; CRG3 evidence validation now rejects impossible verification dates that only match the schema regex.

Files changed:
- `tools/security/finalize_crg3.py`
- `docs/CRG3_CI_EVIDENCE.md`
- `tests/infra/test_production_readiness.py`
- `PRODUCTION_REFACTOR_ROADMAP.md`

Safety invariants proven:
- CRG3 evidence `verified_date` must parse as a real ISO calendar date before roadmap or evidence JSON writes proceed.
- Regex-valid but impossible dates such as `2026-02-31` cannot be used as final CRG3 evidence.

Tests added or inverted:
- Added evidence validation regression for an impossible but schema-shaped `verified_date`.
- Extended CRG3 runbook enforcement for finalizer-side calendar-date validation.

Verification commands:

```bash
python tools/security/check_crg3_evidence_schema.py
python -m pytest tests/infra/test_production_readiness.py -q
python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md --allow-open-id CRG3
python -m pytest tests/infra/test_production_readiness.py tests/infra/test_deployment_hardening_docs.py -q
python -c "import sys, pytest; from tools.security.production_readiness import FOCUSED_P0_TESTS; sys.exit(pytest.main([*FOCUSED_P0_TESTS, '-q']))"
python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md
```

Results:
- `python tools/security/check_crg3_evidence_schema.py` passed.
- `python -m pytest tests/infra/test_production_readiness.py -q` passed: 119 passed in 5.72s.
- `python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md --allow-open-id CRG3` passed.
- `python -m pytest tests/infra/test_production_readiness.py tests/infra/test_deployment_hardening_docs.py -q` passed: 123 passed in 4.50s.
- Focused P0 suite passed: 380 passed, 4 warnings in 43.31s.
- Strict roadmap validation failed only because `CRG3` is still `In progress`; a Docker-capable GitHub Actions evidence run is still required.

Rollback plan:
- Revert this slice only if CRG3 evidence dates move to a stronger signed timestamp attestation; preserve rejection of impossible verification dates.

Residual risk:
- Calendar-date validation improves local evidence integrity, but CRG3 still needs Docker-capable GitHub Actions evidence and downloaded artifact validation before it can be verified.

Live-block status:
- `CRG3` remains the only open P0 ledger item; production live enablement remains blocked until strict roadmap validation passes with `CRG3` marked `Verified YYYY-MM-DD`.

### 2026-06-04 Pass 6 CRG3 Non-Future Verified Date Slice

Ledger IDs changed:
- `CRG3` remains in progress; CRG3 evidence validation now rejects future-dated verification evidence.

Files changed:
- `tools/security/finalize_crg3.py`
- `docs/CRG3_CI_EVIDENCE.md`
- `tests/infra/test_production_readiness.py`
- `PRODUCTION_REFACTOR_ROADMAP.md`

Safety invariants proven:
- CRG3 evidence `verified_date` must be a real ISO calendar date that is not later than the current UTC date.
- Finalizer validation rejects far-future evidence dates before roadmap or evidence JSON writes proceed.

Tests added or inverted:
- Added evidence validation regression for a future `verified_date`.
- Extended CRG3 runbook enforcement for finalizer-side non-future date validation.

Verification commands:

```bash
python tools/security/check_crg3_evidence_schema.py
python -m pytest tests/infra/test_production_readiness.py -q
python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md --allow-open-id CRG3
python -m pytest tests/infra/test_production_readiness.py tests/infra/test_deployment_hardening_docs.py -q
python -c "import sys, pytest; from tools.security.production_readiness import FOCUSED_P0_TESTS; sys.exit(pytest.main([*FOCUSED_P0_TESTS, '-q']))"
python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md
```

Results:
- `python tools/security/check_crg3_evidence_schema.py` passed.
- `python -m pytest tests/infra/test_production_readiness.py -q` passed: 119 passed in 5.55s.
- `python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md --allow-open-id CRG3` passed.
- `python -m pytest tests/infra/test_production_readiness.py tests/infra/test_deployment_hardening_docs.py -q` passed: 123 passed in 4.78s.
- Focused P0 suite passed: 380 passed, 4 warnings in 42.44s.
- Strict roadmap validation failed only because `CRG3` is still `In progress`; a Docker-capable GitHub Actions evidence run is still required.

Rollback plan:
- Revert this slice only if CRG3 evidence dates move to a stronger trusted timestamp source; preserve rejection of future-dated verification evidence.

Residual risk:
- Non-future date validation improves local evidence integrity, but CRG3 still needs Docker-capable GitHub Actions evidence and downloaded artifact validation before it can be verified.

Live-block status:
- `CRG3` remains the only open P0 ledger item; production live enablement remains blocked until strict roadmap validation passes with `CRG3` marked `Verified YYYY-MM-DD`.

### 2026-06-04 Pass 6 CRG3 Full-String Schema Pattern Slice

Ledger IDs changed:
- `CRG3` remains in progress; CRG3 evidence schema validation now rejects newline-tainted or suffix-tainted string fields.

Files changed:
- `tools/security/finalize_crg3.py`
- `docs/CRG3_CI_EVIDENCE.md`
- `tests/infra/test_production_readiness.py`
- `PRODUCTION_REFACTOR_ROADMAP.md`

Safety invariants proven:
- CRG3 evidence schema pattern checks use full-string matching instead of prefix matching.
- Newline-tainted values such as a commit SHA with a trailing newline are rejected before roadmap or evidence JSON writes proceed.

Tests added or inverted:
- Added evidence validation regression for a newline-tainted `commit_sha`.
- Updated the CRG3 schema helper assertion to use full-string matching.
- Extended CRG3 runbook enforcement for full-string schema pattern validation.

Verification commands:

```bash
python tools/security/check_crg3_evidence_schema.py
python -m pytest tests/infra/test_production_readiness.py -q
python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md --allow-open-id CRG3
python -m pytest tests/infra/test_production_readiness.py tests/infra/test_deployment_hardening_docs.py -q
python -c "import sys, pytest; from tools.security.production_readiness import FOCUSED_P0_TESTS; sys.exit(pytest.main([*FOCUSED_P0_TESTS, '-q']))"
python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md
```

Results:
- `python tools/security/check_crg3_evidence_schema.py` passed.
- `python -m pytest tests/infra/test_production_readiness.py -q` passed: 120 passed in 6.45s.
- `python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md --allow-open-id CRG3` passed.
- `python -m pytest tests/infra/test_production_readiness.py tests/infra/test_deployment_hardening_docs.py -q` passed: 124 passed in 6.11s.
- Focused P0 suite passed: 381 passed, 4 warnings in 44.72s.
- Strict roadmap validation failed only because `CRG3` is still `In progress`; a Docker-capable GitHub Actions evidence run is still required.

Rollback plan:
- Revert this slice only if CRG3 evidence validation is replaced by a standards-complete JSON Schema validator that preserves full-string pattern rejection.

Residual risk:
- Full-string pattern validation improves local evidence integrity, but CRG3 still needs Docker-capable GitHub Actions evidence and downloaded artifact validation before it can be verified.

Live-block status:
- `CRG3` remains the only open P0 ledger item; production live enablement remains blocked until strict roadmap validation passes with `CRG3` marked `Verified YYYY-MM-DD`.

### 2026-06-04 Pass 6 CRG3 Roadmap Date Finalizer Guard Slice

Ledger IDs changed:
- `CRG3` remains in progress; roadmap-only finalizer updates now reject future verification dates just like machine-readable evidence JSON.

Files changed:
- `tools/security/finalize_crg3.py`
- `docs/CRG3_CI_EVIDENCE.md`
- `tests/infra/test_production_readiness.py`
- `PRODUCTION_REFACTOR_ROADMAP.md`

Safety invariants proven:
- Finalizer `verified_date` cannot be in the future even when `--evidence-json` is not used.
- Future-dated roadmap finalization fails before mutating `PRODUCTION_REFACTOR_ROADMAP.md`.
- Machine-readable evidence and roadmap-only finalization share the same non-future verification-date helper.

Tests added or inverted:
- Added finalizer regression for a future `verified_date` without evidence JSON.
- Extended CRG3 runbook enforcement for non-future date validation before writing evidence or updating the roadmap.

Verification commands:

```bash
python tools/security/check_crg3_evidence_schema.py
python -m pytest tests/infra/test_production_readiness.py -q
python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md --allow-open-id CRG3
python -m pytest tests/infra/test_production_readiness.py tests/infra/test_deployment_hardening_docs.py -q
python -c "import sys, pytest; from tools.security.production_readiness import FOCUSED_P0_TESTS; sys.exit(pytest.main([*FOCUSED_P0_TESTS, '-q']))"
python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md
```

Results:
- `python tools/security/check_crg3_evidence_schema.py` passed.
- `python -m pytest tests/infra/test_production_readiness.py -q` passed: 121 passed in 5.75s.
- `python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md --allow-open-id CRG3` passed.
- `python -m pytest tests/infra/test_production_readiness.py tests/infra/test_deployment_hardening_docs.py -q` passed: 125 passed in 7.89s.
- Focused P0 suite passed: 382 passed, 4 warnings in 42.92s.
- Strict roadmap validation failed only because `CRG3` is still `In progress`; a Docker-capable GitHub Actions evidence run is still required.

Rollback plan:
- Revert this slice only if CRG3 finalization moves to a stronger trusted timestamp source that still prevents future-dated roadmap verification.

Residual risk:
- Roadmap date validation improves finalizer integrity, but CRG3 still needs Docker-capable GitHub Actions evidence and downloaded artifact validation before it can be verified.

Live-block status:
- `CRG3` remains the only open P0 ledger item; production live enablement remains blocked until strict roadmap validation passes with `CRG3` marked `Verified YYYY-MM-DD`.

### 2026-06-04 Pass 6 CRG3 Future-Date Early Rejection Slice

Ledger IDs changed:
- `CRG3` remains in progress; finalizer future-date rejection now runs before GitHub API and artifact verification work.

Files changed:
- `tools/security/finalize_crg3.py`
- `docs/CRG3_CI_EVIDENCE.md`
- `tests/infra/test_production_readiness.py`
- `PRODUCTION_REFACTOR_ROADMAP.md`

Safety invariants proven:
- Future-dated CRG3 finalizer requests fail before GitHub run metadata fetches.
- Future-dated CRG3 finalizer requests fail before SBOM artifact downloads.
- The roadmap remains unchanged when future-date validation rejects finalization.

Tests added or inverted:
- Strengthened the future-date finalizer regression with API/artifact fetchers that fail if called.
- Extended CRG3 runbook enforcement for early future-date rejection before GitHub verification work starts.

Verification commands:

```bash
python tools/security/check_crg3_evidence_schema.py
python -m pytest tests/infra/test_production_readiness.py -q
python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md --allow-open-id CRG3
python -m pytest tests/infra/test_production_readiness.py tests/infra/test_deployment_hardening_docs.py -q
python -c "import sys, pytest; from tools.security.production_readiness import FOCUSED_P0_TESTS; sys.exit(pytest.main([*FOCUSED_P0_TESTS, '-q']))"
python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md
```

Results:
- `python tools/security/check_crg3_evidence_schema.py` passed.
- `python -m pytest tests/infra/test_production_readiness.py -q` passed: 121 passed in 11.26s.
- `python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md --allow-open-id CRG3` passed.
- `python -m pytest tests/infra/test_production_readiness.py tests/infra/test_deployment_hardening_docs.py -q` passed: 125 passed in 8.25s.
- Focused P0 suite passed: 382 passed, 4 warnings in 35.06s.
- Strict roadmap validation failed only because `CRG3` is still `In progress`; a Docker-capable GitHub Actions evidence run is still required.

Rollback plan:
- Revert this slice only if CRG3 date validation moves to a stronger trusted timestamp source that still rejects invalid dates before external evidence fetches.

Residual risk:
- Early future-date rejection improves finalizer failure ordering, but CRG3 still needs Docker-capable GitHub Actions evidence and downloaded artifact validation before it can be verified.

Live-block status:
- `CRG3` remains the only open P0 ledger item; production live enablement remains blocked until strict roadmap validation passes with `CRG3` marked `Verified YYYY-MM-DD`.

### 2026-06-04 Pass 6 CRG3 Direct Input Full-String Validation Slice

Ledger IDs changed:
- `CRG3` remains in progress; direct finalizer URL and SHA inputs now reject newline-tainted or suffix-tainted values before external evidence fetches.

Files changed:
- `tools/security/finalize_crg3.py`
- `docs/CRG3_CI_EVIDENCE.md`
- `tests/infra/test_production_readiness.py`
- `PRODUCTION_REFACTOR_ROADMAP.md`

Safety invariants proven:
- Finalizer `--run-url` validation uses full-string matching before GitHub API fetches.
- Finalizer `--commit-sha` validation uses full-string matching before GitHub API fetches.
- Remote URL parsing uses full-string matching for GitHub remote recognition.
- Newline-tainted direct finalizer inputs leave the roadmap unchanged and do not call GitHub API or artifact fetchers.

Tests added or inverted:
- Added finalizer regression for newline-tainted `run_url` and `commit_sha` with fetchers that fail if called.
- Updated generated-evidence digest assertions to use full-string matching.
- Extended CRG3 runbook enforcement for direct `--run-url` and `--commit-sha` full-string validation.

Verification commands:

```bash
python tools/security/check_crg3_evidence_schema.py
python -m pytest tests/infra/test_production_readiness.py -q
python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md --allow-open-id CRG3
python -m pytest tests/infra/test_production_readiness.py tests/infra/test_deployment_hardening_docs.py -q
python -c "import sys, pytest; from tools.security.production_readiness import FOCUSED_P0_TESTS; sys.exit(pytest.main([*FOCUSED_P0_TESTS, '-q']))"
python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md
```

Results:
- `python tools/security/check_crg3_evidence_schema.py` passed.
- `python -m pytest tests/infra/test_production_readiness.py -q` passed: 122 passed in 4.94s.
- `python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md --allow-open-id CRG3` passed.
- `python -m pytest tests/infra/test_production_readiness.py tests/infra/test_deployment_hardening_docs.py -q` passed: 126 passed in 5.51s.
- Focused P0 suite passed: 383 passed, 4 warnings in 37.62s.
- Strict roadmap validation failed only because `CRG3` is still `In progress`; a Docker-capable GitHub Actions evidence run is still required.

Rollback plan:
- Revert this slice only if direct finalizer input parsing is replaced by a stronger typed parser that still rejects suffix-tainted workflow URLs and commit SHAs.

Residual risk:
- Direct input validation improves finalizer integrity, but CRG3 still needs Docker-capable GitHub Actions evidence and downloaded artifact validation before it can be verified.

Live-block status:
- `CRG3` remains the only open P0 ledger item; production live enablement remains blocked until strict roadmap validation passes with `CRG3` marked `Verified YYYY-MM-DD`.

### 2026-06-04 Pass 6 CRG3 GitHub Slug Pattern Hardening Slice

Ledger IDs changed:
- `CRG3` remains in progress; finalizer and evidence schema now reject whitespace/control-tainted GitHub owner/repo slug values.

Files changed:
- `tools/security/finalize_crg3.py`
- `docs/CRG3_EVIDENCE_SCHEMA.json`
- `docs/CRG3_CI_EVIDENCE.md`
- `tests/infra/test_production_readiness.py`
- `PRODUCTION_REFACTOR_ROADMAP.md`

Safety invariants proven:
- CRG3 evidence schema version 7 requires slug-safe GitHub owner/repo patterns for `repository`, `run_url`, and `sbom_artifact_metadata.download_url`.
- Direct finalizer workflow run URL parsing rejects whitespace-tainted GitHub owner/repo slugs before GitHub API or artifact fetchers run.
- GitHub remote URL parsing uses the same slug-safe owner/repo patterns.

Tests added or inverted:
- Added evidence schema regression asserting slug-safe GitHub owner/repo patterns are present.
- Added evidence validation regression for whitespace-tainted repository values.
- Extended direct finalizer input regression for whitespace-tainted workflow run URLs.
- Extended CRG3 runbook enforcement for schema version 7 and slug-safe GitHub owner/repo values.

Verification commands:

```bash
python tools/security/check_crg3_evidence_schema.py
python -m pytest tests/infra/test_production_readiness.py -q
python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md --allow-open-id CRG3
python -m pytest tests/infra/test_production_readiness.py tests/infra/test_deployment_hardening_docs.py -q
python -c "import sys, pytest; from tools.security.production_readiness import FOCUSED_P0_TESTS; sys.exit(pytest.main([*FOCUSED_P0_TESTS, '-q']))"
python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md
```

Results:
- `python tools/security/check_crg3_evidence_schema.py` passed.
- `python -m pytest tests/infra/test_production_readiness.py -q` passed: 122 passed in 6.53s.
- `python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md --allow-open-id CRG3` passed.
- `python -m pytest tests/infra/test_production_readiness.py tests/infra/test_deployment_hardening_docs.py -q` passed: 126 passed in 5.96s.
- Focused P0 suite passed: 383 passed, 4 warnings in 38.13s.
- Strict roadmap validation failed only because `CRG3` is still `In progress`; a Docker-capable GitHub Actions evidence run is still required.

Rollback plan:
- Revert this slice only if finalizer GitHub evidence binding moves to a stronger typed URL parser that still rejects whitespace/control-tainted owner/repo values.

Residual risk:
- GitHub slug pattern hardening improves finalizer and evidence integrity, but CRG3 still needs Docker-capable GitHub Actions evidence and downloaded artifact validation before it can be verified.

Live-block status:
- `CRG3` remains the only open P0 ledger item; production live enablement remains blocked until strict roadmap validation passes with `CRG3` marked `Verified YYYY-MM-DD`.

### 2026-06-04 Pass 6 CRG3 Evidence Output Containment Slice

Ledger IDs changed:
- `CRG3` remains in progress; optional CRG3 evidence JSON output is now constrained to the roadmap directory tree.

Files changed:
- `tools/security/finalize_crg3.py`
- `docs/CRG3_CI_EVIDENCE.md`
- `tests/infra/test_production_readiness.py`
- `PRODUCTION_REFACTOR_ROADMAP.md`

Safety invariants proven:
- `--evidence-json` output must stay under the roadmap directory, which is the repository root in the standard workflow.
- Outside-directory evidence JSON destinations are rejected before roadmap or evidence JSON writes.
- Rejected outside-directory evidence paths leave both the roadmap and outside destination unchanged.

Tests added or inverted:
- Added finalizer regression for an evidence JSON path outside the roadmap directory.
- Extended CRG3 runbook enforcement for evidence output containment.

Verification commands:

```bash
python tools/security/check_crg3_evidence_schema.py
python -m pytest tests/infra/test_production_readiness.py -q
python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md --allow-open-id CRG3
python -m pytest tests/infra/test_production_readiness.py tests/infra/test_deployment_hardening_docs.py -q
python -c "import sys, pytest; from tools.security.production_readiness import FOCUSED_P0_TESTS; sys.exit(pytest.main([*FOCUSED_P0_TESTS, '-q']))"
python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md
```

Results:
- `python tools/security/check_crg3_evidence_schema.py` passed.
- `python -m pytest tests/infra/test_production_readiness.py -q` passed: 123 passed in 9.84s.
- `python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md --allow-open-id CRG3` passed.
- `python -m pytest tests/infra/test_production_readiness.py tests/infra/test_deployment_hardening_docs.py -q` passed: 127 passed in 6.47s.
- Focused P0 suite passed: 384 passed, 4 warnings in 44.77s.
- Strict roadmap validation failed only because `CRG3` is still `In progress`; a Docker-capable GitHub Actions evidence run is still required.

Rollback plan:
- Revert this slice only if CRG3 evidence output moves to a dedicated artifact writer that still prevents accidental writes outside the verified repository/roadmap tree.

Residual risk:
- Evidence output containment improves local finalizer safety, but CRG3 still needs Docker-capable GitHub Actions evidence and downloaded artifact validation before it can be verified.

Live-block status:
- `CRG3` remains the only open P0 ledger item; production live enablement remains blocked until strict roadmap validation passes with `CRG3` marked `Verified YYYY-MM-DD`.

### 2026-06-04 Pass 6 CRG3 Evidence Output Git Directory Guard Slice

Ledger IDs changed:
- `CRG3` remains in progress; optional CRG3 evidence JSON output can no longer target `.git`.

Files changed:
- `tools/security/finalize_crg3.py`
- `docs/CRG3_CI_EVIDENCE.md`
- `tests/infra/test_production_readiness.py`
- `PRODUCTION_REFACTOR_ROADMAP.md`

Safety invariants proven:
- `--evidence-json` output paths containing a `.git` segment are rejected.
- Rejected `.git` evidence paths leave both the roadmap and intended evidence output unchanged.
- Evidence output remains constrained to ordinary repository artifact paths rather than repository internals.

Tests added or inverted:
- Added finalizer regression for an evidence JSON path under `.git`.
- Extended CRG3 runbook enforcement for `.git` evidence output rejection.

Verification commands:

```bash
python tools/security/check_crg3_evidence_schema.py
python -m pytest tests/infra/test_production_readiness.py -q
python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md --allow-open-id CRG3
python -m pytest tests/infra/test_production_readiness.py tests/infra/test_deployment_hardening_docs.py -q
python -c "import sys, pytest; from tools.security.production_readiness import FOCUSED_P0_TESTS; sys.exit(pytest.main([*FOCUSED_P0_TESTS, '-q']))"
python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md
```

Results:
- `python tools/security/check_crg3_evidence_schema.py` passed.
- `python -m pytest tests/infra/test_production_readiness.py -q` passed: 124 passed in 6.74s.
- `python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md --allow-open-id CRG3` passed.
- `python -m pytest tests/infra/test_production_readiness.py tests/infra/test_deployment_hardening_docs.py -q` passed: 128 passed in 6.25s.
- Focused P0 suite passed: 385 passed, 4 warnings in 48.22s.
- Strict roadmap validation failed only because `CRG3` is still `In progress`; a Docker-capable GitHub Actions evidence run is still required.

Rollback plan:
- Revert this slice only if CRG3 evidence output moves to a dedicated artifact writer that still rejects writes into repository internals.

Residual risk:
- `.git` output rejection improves local finalizer safety, but CRG3 still needs Docker-capable GitHub Actions evidence and downloaded artifact validation before it can be verified.

Live-block status:
- `CRG3` remains the only open P0 ledger item; production live enablement remains blocked until strict roadmap validation passes with `CRG3` marked `Verified YYYY-MM-DD`.

### 2026-06-04 Pass 6 CRG3 Evidence Output Hidden Directory Guard Slice

Ledger IDs changed:
- `CRG3` remains in progress; optional CRG3 evidence JSON output can no longer target hidden directories.

Files changed:
- `tools/security/finalize_crg3.py`
- `docs/CRG3_CI_EVIDENCE.md`
- `tests/infra/test_production_readiness.py`
- `PRODUCTION_REFACTOR_ROADMAP.md`

Safety invariants proven:
- `--evidence-json` output paths under hidden directories such as `.github` are rejected.
- Rejected hidden-directory evidence paths leave both the roadmap and intended evidence output unchanged.
- Evidence output remains constrained to normal artifact paths inside the roadmap/repository tree.

Tests added or inverted:
- Added finalizer regression for an evidence JSON path under `.github`.
- Extended CRG3 runbook enforcement for hidden-directory evidence output rejection.

Verification commands:

```bash
python tools/security/check_crg3_evidence_schema.py
python -m pytest tests/infra/test_production_readiness.py -q
python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md --allow-open-id CRG3
python -m pytest tests/infra/test_production_readiness.py tests/infra/test_deployment_hardening_docs.py -q
python -c "import sys, pytest; from tools.security.production_readiness import FOCUSED_P0_TESTS; sys.exit(pytest.main([*FOCUSED_P0_TESTS, '-q']))"
python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md
```

Results:
- `python tools/security/check_crg3_evidence_schema.py` passed.
- `python -m pytest tests/infra/test_production_readiness.py -q` passed: 125 passed in 11.28s.
- `python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md --allow-open-id CRG3` passed.
- `python -m pytest tests/infra/test_production_readiness.py tests/infra/test_deployment_hardening_docs.py -q` passed: 129 passed in 7.29s.
- Focused P0 suite passed: 386 passed, 4 warnings in 45.43s.
- Strict roadmap validation failed only because `CRG3` is still `In progress`; a Docker-capable GitHub Actions evidence run is still required.

Rollback plan:
- Revert this slice only if CRG3 evidence output moves to a dedicated artifact writer that still rejects writes into repository/tool internals.

Residual risk:
- Hidden-directory output rejection improves local finalizer safety, but CRG3 still needs Docker-capable GitHub Actions evidence and downloaded artifact validation before it can be verified.

Live-block status:
- `CRG3` remains the only open P0 ledger item; production live enablement remains blocked until strict roadmap validation passes with `CRG3` marked `Verified YYYY-MM-DD`.

### 2026-06-04 Pass 6 CRG3 Evidence Schema Output Alias Guard Slice

Ledger IDs changed:
- `CRG3` remains in progress; optional CRG3 evidence JSON output can no longer target the CRG3 evidence schema file.

Files changed:
- `tools/security/finalize_crg3.py`
- `docs/CRG3_CI_EVIDENCE.md`
- `tests/infra/test_production_readiness.py`
- `PRODUCTION_REFACTOR_ROADMAP.md`

Safety invariants proven:
- `--evidence-json docs/CRG3_EVIDENCE_SCHEMA.json` is rejected before roadmap or evidence writes.
- Rejected schema-alias evidence output leaves the evidence schema unchanged.
- CRG3 proof output cannot overwrite the validation contract used to verify that proof output.

Tests added or inverted:
- Added finalizer regression for an evidence JSON path aliasing `docs/CRG3_EVIDENCE_SCHEMA.json`.
- Extended CRG3 runbook enforcement for evidence schema output alias rejection.

Verification commands:

```bash
python tools/security/check_crg3_evidence_schema.py
python -m pytest tests/infra/test_production_readiness.py -q
python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md --allow-open-id CRG3
python -m pytest tests/infra/test_production_readiness.py tests/infra/test_deployment_hardening_docs.py -q
python -c "import sys, pytest; from tools.security.production_readiness import FOCUSED_P0_TESTS; sys.exit(pytest.main([*FOCUSED_P0_TESTS, '-q']))"
python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md
```

Results:
- `python tools/security/check_crg3_evidence_schema.py` passed.
- `python -m pytest tests/infra/test_production_readiness.py -q` passed: 126 passed in 6.43s.
- `python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md --allow-open-id CRG3` passed.
- `python -m pytest tests/infra/test_production_readiness.py tests/infra/test_deployment_hardening_docs.py -q` passed: 130 passed in 8.03s.
- Focused P0 suite passed: 387 passed, 4 warnings in 44.10s.
- Strict roadmap validation failed only because `CRG3` is still `In progress`; a Docker-capable GitHub Actions evidence run is still required.

Rollback plan:
- Revert this slice only if CRG3 evidence output moves to a dedicated artifact writer that still rejects overwriting validation contracts.

Residual risk:
- Schema alias rejection improves local finalizer safety, but CRG3 still needs Docker-capable GitHub Actions evidence and downloaded artifact validation before it can be verified.

Live-block status:
- `CRG3` remains the only open P0 ledger item; production live enablement remains blocked until strict roadmap validation passes with `CRG3` marked `Verified YYYY-MM-DD`.

### 2026-06-04 Pass 6 CRG3 Evidence Output SBOM Filename Guard Slice

Ledger IDs changed:
- `CRG3` remains in progress; optional CRG3 evidence JSON output can no longer use the workflow SBOM filename.

Files changed:
- `tools/security/finalize_crg3.py`
- `docs/CRG3_CI_EVIDENCE.md`
- `tests/infra/test_production_readiness.py`
- `PRODUCTION_REFACTOR_ROADMAP.md`

Safety invariants proven:
- `--evidence-json .../sbom.cdx.json` is rejected before roadmap or evidence writes.
- Rejected SBOM-filename evidence output leaves the roadmap and intended SBOM path unchanged.
- CRG3 proof output cannot overwrite the workflow SBOM artifact filename.

Tests added or inverted:
- Added finalizer regression for an evidence JSON path named `sbom.cdx.json`.
- Extended CRG3 runbook enforcement for SBOM filename collision rejection.

Verification commands:

```bash
python tools/security/check_crg3_evidence_schema.py
python -m pytest tests/infra/test_production_readiness.py -q
python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md --allow-open-id CRG3
python -m pytest tests/infra/test_production_readiness.py tests/infra/test_deployment_hardening_docs.py -q
python -c "import sys, pytest; from tools.security.production_readiness import FOCUSED_P0_TESTS; sys.exit(pytest.main([*FOCUSED_P0_TESTS, '-q']))"
python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md
```

Results:
- `python tools/security/check_crg3_evidence_schema.py` passed.
- `python -m pytest tests/infra/test_production_readiness.py -q` passed: 127 passed in 6.99s.
- `python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md --allow-open-id CRG3` passed.
- `python -m pytest tests/infra/test_production_readiness.py tests/infra/test_deployment_hardening_docs.py -q` passed: 131 passed in 5.63s.
- Focused P0 suite passed: 388 passed, 4 warnings in 41.21s.
- Strict roadmap validation failed only because `CRG3` is still `In progress`; a Docker-capable GitHub Actions evidence run is still required.

Rollback plan:
- Revert this slice only if CRG3 proof and SBOM artifact generation move to a dedicated artifact writer that still keeps their output filenames distinct.

Residual risk:
- SBOM filename collision rejection improves local finalizer safety, but CRG3 still needs Docker-capable GitHub Actions evidence and downloaded artifact validation before it can be verified.

Live-block status:
- `CRG3` remains the only open P0 ledger item; production live enablement remains blocked until strict roadmap validation passes with `CRG3` marked `Verified YYYY-MM-DD`.

### 2026-06-04 Pass 6 CRG3 Evidence Canonical Filename Slice

Ledger IDs changed:
- `CRG3` remains in progress; optional CRG3 evidence JSON output now requires the canonical `crg3-evidence.json` filename.

Files changed:
- `tools/security/finalize_crg3.py`
- `docs/CRG3_CI_EVIDENCE.md`
- `tests/infra/test_production_readiness.py`
- `PRODUCTION_REFACTOR_ROADMAP.md`

Safety invariants proven:
- `--evidence-json` output must be named `crg3-evidence.json`.
- Arbitrary JSON destinations inside the roadmap/repository tree are rejected before roadmap or evidence writes.
- Rejected non-canonical evidence filenames leave both the roadmap and intended output path unchanged.

Tests added or inverted:
- Added finalizer regression for a same-directory arbitrary JSON evidence filename.
- Extended CRG3 runbook enforcement for canonical `crg3-evidence.json` output naming.

Verification commands:

```bash
python tools/security/check_crg3_evidence_schema.py
python -m pytest tests/infra/test_production_readiness.py -q
python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md --allow-open-id CRG3
python -m pytest tests/infra/test_production_readiness.py tests/infra/test_deployment_hardening_docs.py -q
python -c "import sys, pytest; from tools.security.production_readiness import FOCUSED_P0_TESTS; sys.exit(pytest.main([*FOCUSED_P0_TESTS, '-q']))"
python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md
```

Results:
- `python tools/security/check_crg3_evidence_schema.py` passed.
- `python -m pytest tests/infra/test_production_readiness.py -q` passed: 128 passed in 4.95s.
- `python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md --allow-open-id CRG3` passed.
- `python -m pytest tests/infra/test_production_readiness.py tests/infra/test_deployment_hardening_docs.py -q` passed: 132 passed in 5.12s.
- Focused P0 suite passed: 389 passed, 4 warnings in 42.33s.
- Strict roadmap validation failed only because `CRG3` is still `In progress`; a Docker-capable GitHub Actions evidence run is still required.

Rollback plan:
- Revert this slice only if CRG3 proof output moves to a dedicated artifact writer that still prevents overwriting arbitrary JSON files.

Residual risk:
- Canonical evidence filename enforcement improves local finalizer safety, but CRG3 still needs Docker-capable GitHub Actions evidence and downloaded artifact validation before it can be verified.

Live-block status:
- `CRG3` remains the only open P0 ledger item; production live enablement remains blocked until strict roadmap validation passes with `CRG3` marked `Verified YYYY-MM-DD`.

### 2026-06-04 Pass 6 CRG3 GitHub Owner Slug Ending Slice

Ledger IDs changed:
- `CRG3` remains in progress; GitHub owner slug validation now rejects owner values ending in `-`.

Files changed:
- `tools/security/finalize_crg3.py`
- `docs/CRG3_EVIDENCE_SCHEMA.json`
- `docs/CRG3_CI_EVIDENCE.md`
- `tests/infra/test_production_readiness.py`
- `PRODUCTION_REFACTOR_ROADMAP.md`

Safety invariants proven:
- Direct finalizer workflow run URL parsing requires GitHub owner values to start and end alphanumeric.
- CRG3 evidence schema version 7 requires alphanumeric-ended GitHub owner values in `repository`, `run_url`, and `sbom_artifact_metadata.download_url`.
- Trailing-hyphen owner values are rejected before GitHub API or artifact fetchers run.

Tests added or inverted:
- Added evidence validation regression for a trailing-hyphen owner in `repository`.
- Extended direct finalizer input regression for a trailing-hyphen owner in `run_url`.
- Extended CRG3 runbook enforcement for alphanumeric-ended GitHub owner values.

Verification commands:

```bash
python tools/security/check_crg3_evidence_schema.py
python -m pytest tests/infra/test_production_readiness.py -q
python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md --allow-open-id CRG3
python -m pytest tests/infra/test_production_readiness.py tests/infra/test_deployment_hardening_docs.py -q
python -c "import sys, pytest; from tools.security.production_readiness import FOCUSED_P0_TESTS; sys.exit(pytest.main([*FOCUSED_P0_TESTS, '-q']))"
python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md
```

Results:
- `python tools/security/check_crg3_evidence_schema.py` passed.
- `python -m pytest tests/infra/test_production_readiness.py -q` passed: 128 passed in 8.84s.
- `python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md --allow-open-id CRG3` passed.
- `python -m pytest tests/infra/test_production_readiness.py tests/infra/test_deployment_hardening_docs.py -q` passed: 132 passed in 6.32s.
- Focused P0 suite passed: 389 passed, 4 warnings in 46.30s.
- Strict roadmap validation failed only because `CRG3` is still `In progress`; a Docker-capable GitHub Actions evidence run is still required.

Rollback plan:
- Revert this slice only if finalizer GitHub evidence binding moves to a stronger typed URL parser that still rejects invalid GitHub owner slugs.

Residual risk:
- GitHub owner slug hardening improves finalizer and evidence integrity, but CRG3 still needs Docker-capable GitHub Actions evidence and downloaded artifact validation before it can be verified.

Live-block status:
- `CRG3` remains the only open P0 ledger item; production live enablement remains blocked until strict roadmap validation passes with `CRG3` marked `Verified YYYY-MM-DD`.

### 2026-06-04 Pass 6 CRG3 Non-Zero Artifact Size Evidence Slice

Ledger IDs changed:
- `CRG3` remains in progress; machine-readable CRG3 evidence now rejects zero-byte artifact size and downloaded-size fields.

Files changed:
- `tools/security/finalize_crg3.py`
- `docs/CRG3_EVIDENCE_SCHEMA.json`
- `docs/CRG3_CI_EVIDENCE.md`
- `tests/infra/test_production_readiness.py`
- `PRODUCTION_REFACTOR_ROADMAP.md`

Safety invariants proven:
- CRG3 evidence schema version 8 requires `sbom_artifact_metadata.size_in_bytes >= 1`.
- CRG3 evidence schema version 8 requires `sbom_artifact_metadata.download_size_in_bytes >= 1`.
- The CRG3 runbook now documents non-zero artifact metadata and non-zero downloaded byte length as part of final evidence.

Tests added or inverted:
- Added evidence validation regressions for zero `download_size_in_bytes` and zero `size_in_bytes`.
- Extended CRG3 runbook enforcement for schema version 8 and non-zero artifact-size wording.

Verification commands:

```bash
python tools/security/check_crg3_evidence_schema.py
python -m pytest tests/infra/test_production_readiness.py -q
python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md --allow-open-id CRG3
python -m pytest tests/infra/test_production_readiness.py tests/infra/test_deployment_hardening_docs.py -q
python -c "import sys, pytest; from tools.security.production_readiness import FOCUSED_P0_TESTS; sys.exit(pytest.main([*FOCUSED_P0_TESTS, '-q']))"
python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md
```

Results:
- `python tools/security/check_crg3_evidence_schema.py` passed.
- `python -m pytest tests/infra/test_production_readiness.py -q` passed: 128 passed in 5.12s.
- `python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md --allow-open-id CRG3` passed.
- `python -m pytest tests/infra/test_production_readiness.py tests/infra/test_deployment_hardening_docs.py -q` passed: 132 passed in 4.61s.
- Focused P0 suite passed: 389 passed, 4 warnings in 37.51s.
- Strict roadmap validation failed only because `CRG3` is still `In progress`; a Docker-capable GitHub Actions evidence run is still required.

Rollback plan:
- Revert this slice only if CRG3 evidence records artifact byte counts through a stronger downloaded-artifact proof that still rejects impossible zero-byte artifacts.

Residual risk:
- Non-zero artifact size evidence improves machine-readable CRG3 proof quality, but CRG3 still needs Docker-capable GitHub Actions evidence and downloaded artifact validation before it can be verified.

Live-block status:
- `CRG3` remains the only open P0 ledger item; production live enablement remains blocked until strict roadmap validation passes with `CRG3` marked `Verified YYYY-MM-DD`.

### 2026-06-04 Pass 6 CRG3 Zero-Size Artifact Metadata Guard Slice

Ledger IDs changed:
- `CRG3` remains in progress; GitHub artifact metadata with zero `size_in_bytes` is now rejected before artifact download.

Files changed:
- `tools/security/finalize_crg3.py`
- `tests/infra/test_production_readiness.py`
- `PRODUCTION_REFACTOR_ROADMAP.md`

Safety invariants proven:
- The CRG3 finalizer no longer accepts `production-sbom` artifact metadata where `size_in_bytes == 0`.
- Zero-size GitHub artifact metadata is rejected before the artifact fetcher/download path runs.
- Finalizer source validation now matches schema version 8's non-zero artifact-size evidence contract.

Tests added or inverted:
- Added a finalizer regression for `artifact_size_in_bytes=0` with an artifact fetcher that fails if called.

Verification commands:

```bash
python tools/security/check_crg3_evidence_schema.py
python -m pytest tests/infra/test_production_readiness.py -q
python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md --allow-open-id CRG3
python -m pytest tests/infra/test_production_readiness.py tests/infra/test_deployment_hardening_docs.py -q
python -c "import sys, pytest; from tools.security.production_readiness import FOCUSED_P0_TESTS; sys.exit(pytest.main([*FOCUSED_P0_TESTS, '-q']))"
python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md
```

Results:
- `python tools/security/check_crg3_evidence_schema.py` passed.
- `python -m pytest tests/infra/test_production_readiness.py -q` passed: 129 passed in 5.87s.
- `python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md --allow-open-id CRG3` passed.
- `python -m pytest tests/infra/test_production_readiness.py tests/infra/test_deployment_hardening_docs.py -q` passed: 133 passed in 4.56s.
- Focused P0 suite passed: 390 passed, 4 warnings in 41.73s.
- Strict roadmap validation failed only because `CRG3` is still `In progress`; a Docker-capable GitHub Actions evidence run is still required.

Rollback plan:
- Revert this slice only if GitHub artifact metadata validation moves to a dedicated evidence parser that still rejects impossible zero-byte production SBOM artifacts before download.

Residual risk:
- Zero-size artifact metadata rejection aligns finalizer source validation with the evidence schema, but CRG3 still needs Docker-capable GitHub Actions evidence and downloaded artifact validation before it can be verified.

Live-block status:
- `CRG3` remains the only open P0 ledger item; production live enablement remains blocked until strict roadmap validation passes with `CRG3` marked `Verified YYYY-MM-DD`.

### 2026-06-04 Pass 6 CRG3 Artifact Pagination Count Consistency Slice

Ledger IDs changed:
- `CRG3` remains in progress; GitHub artifact pagination metadata is now rejected when accumulated entries exceed `total_count`.

Files changed:
- `tools/security/finalize_crg3.py`
- `docs/CRG3_CI_EVIDENCE.md`
- `tests/infra/test_production_readiness.py`
- `PRODUCTION_REFACTOR_ROADMAP.md`

Safety invariants proven:
- The CRG3 finalizer rejects impossible GitHub artifact pages whose entries exceed the reported `total_count`.
- The finalizer no longer treats inconsistent pagination metadata as a coherent artifact listing.
- The CRG3 evidence runbook documents `total_count` consistency as part of final artifact evidence.

Tests added or inverted:
- Added a finalizer regression for artifact entries exceeding `total_count`.
- Extended CRG3 runbook enforcement for the pagination count-consistency requirement.

Verification commands:

```bash
python tools/security/check_crg3_evidence_schema.py
python -m pytest tests/infra/test_production_readiness.py -q
python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md --allow-open-id CRG3
python -m pytest tests/infra/test_production_readiness.py tests/infra/test_deployment_hardening_docs.py -q
python -c "import sys, pytest; from tools.security.production_readiness import FOCUSED_P0_TESTS; sys.exit(pytest.main([*FOCUSED_P0_TESTS, '-q']))"
python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md
```

Results:
- `python tools/security/check_crg3_evidence_schema.py` passed.
- `python -m pytest tests/infra/test_production_readiness.py -q` passed: 130 passed in 5.03s.
- `python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md --allow-open-id CRG3` passed.
- `python -m pytest tests/infra/test_production_readiness.py tests/infra/test_deployment_hardening_docs.py -q` passed: 134 passed in 4.73s.
- Focused P0 suite passed: 391 passed, 4 warnings in 33.81s.
- Strict roadmap validation failed only because `CRG3` is still `In progress`; a Docker-capable GitHub Actions evidence run is still required.

Rollback plan:
- Revert this slice only if GitHub artifact pagination validation moves to a dedicated API response parser that still rejects entry counts that exceed `total_count`.

Residual risk:
- Pagination count consistency improves CRG3 artifact listing integrity, but CRG3 still needs Docker-capable GitHub Actions evidence and downloaded artifact validation before it can be verified.

Live-block status:
- `CRG3` remains the only open P0 ledger item; production live enablement remains blocked until strict roadmap validation passes with `CRG3` marked `Verified YYYY-MM-DD`.

### 2026-06-04 Pass 6 CRG3 Artifact Total Count Type Guard Slice

Ledger IDs changed:
- `CRG3` remains in progress; GitHub artifact `total_count` metadata now requires an exact integer and rejects booleans.

Files changed:
- `tools/security/finalize_crg3.py`
- `docs/CRG3_CI_EVIDENCE.md`
- `tests/infra/test_production_readiness.py`
- `PRODUCTION_REFACTOR_ROADMAP.md`

Safety invariants proven:
- Boolean `total_count` values are rejected before artifact download.
- GitHub artifact pagination metadata now uses exact integer validation instead of Python's bool-compatible `isinstance(..., int)` behavior.
- The CRG3 evidence runbook documents non-boolean integer `total_count` metadata as part of final artifact evidence.

Tests added or inverted:
- Added a finalizer regression for `total_count: true` with an artifact fetcher that fails if called.
- Extended CRG3 runbook enforcement for non-boolean integer `total_count` wording.

Verification commands:

```bash
python tools/security/check_crg3_evidence_schema.py
python -m pytest tests/infra/test_production_readiness.py -q
python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md --allow-open-id CRG3
python -m pytest tests/infra/test_production_readiness.py tests/infra/test_deployment_hardening_docs.py -q
python -c "import sys, pytest; from tools.security.production_readiness import FOCUSED_P0_TESTS; sys.exit(pytest.main([*FOCUSED_P0_TESTS, '-q']))"
python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md
```

Results:
- `python tools/security/check_crg3_evidence_schema.py` passed.
- `python -m pytest tests/infra/test_production_readiness.py -q` passed: 131 passed in 5.11s.
- `python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md --allow-open-id CRG3` passed.
- `python -m pytest tests/infra/test_production_readiness.py tests/infra/test_deployment_hardening_docs.py -q` passed: 135 passed in 4.45s.
- Focused P0 suite passed: 392 passed, 4 warnings in 38.84s.
- Strict roadmap validation failed only because `CRG3` is still `In progress`; a Docker-capable GitHub Actions evidence run is still required.

Rollback plan:
- Revert this slice only if artifact pagination metadata validation moves to a stricter typed parser that still rejects boolean `total_count`.

Residual risk:
- Exact integer `total_count` validation improves CRG3 artifact metadata integrity, but CRG3 still needs Docker-capable GitHub Actions evidence and downloaded artifact validation before it can be verified.

Live-block status:
- `CRG3` remains the only open P0 ledger item; production live enablement remains blocked until strict roadmap validation passes with `CRG3` marked `Verified YYYY-MM-DD`.

### 2026-06-04 Pass 6 CRG3 Empty Artifact Page Guard Slice

Ledger IDs changed:
- `CRG3` remains in progress; GitHub artifact pagination now rejects empty pages before the reported `total_count` is reached.

Files changed:
- `tools/security/finalize_crg3.py`
- `docs/CRG3_CI_EVIDENCE.md`
- `tests/infra/test_production_readiness.py`
- `PRODUCTION_REFACTOR_ROADMAP.md`

Safety invariants proven:
- A GitHub artifact page with `artifacts: []` and remaining reported `total_count` is treated as malformed metadata.
- The finalizer no longer converts an incoherent artifact listing into a generic missing-artifact failure.
- The CRG3 evidence runbook documents empty-page rejection as part of pagination integrity.

Tests added or inverted:
- Added a finalizer regression for an empty artifact page before `total_count` is reached.
- Extended CRG3 runbook enforcement for empty-page pagination rejection.

Verification commands:

```bash
python tools/security/check_crg3_evidence_schema.py
python -m pytest tests/infra/test_production_readiness.py -q
python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md --allow-open-id CRG3
python -m pytest tests/infra/test_production_readiness.py tests/infra/test_deployment_hardening_docs.py -q
python -c "import sys, pytest; from tools.security.production_readiness import FOCUSED_P0_TESTS; sys.exit(pytest.main([*FOCUSED_P0_TESTS, '-q']))"
python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md
```

Results:
- `python tools/security/check_crg3_evidence_schema.py` passed.
- `python -m pytest tests/infra/test_production_readiness.py -q` passed: 132 passed in 4.97s.
- `python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md --allow-open-id CRG3` passed.
- `python -m pytest tests/infra/test_production_readiness.py tests/infra/test_deployment_hardening_docs.py -q` passed: 136 passed in 4.75s.
- Focused P0 suite passed: 393 passed, 4 warnings in 37.68s.
- Strict roadmap validation failed only because `CRG3` is still `In progress`; a Docker-capable GitHub Actions evidence run is still required.

Rollback plan:
- Revert this slice only if artifact pagination validation moves to a dedicated GitHub response parser that still rejects empty pages before the reported total is reached.

Residual risk:
- Empty-page rejection improves CRG3 artifact pagination integrity, but CRG3 still needs Docker-capable GitHub Actions evidence and downloaded artifact validation before it can be verified.

Live-block status:
- `CRG3` remains the only open P0 ledger item; production live enablement remains blocked until strict roadmap validation passes with `CRG3` marked `Verified YYYY-MM-DD`.

### 2026-06-04 Pass 6 CRG3 Explicit Artifact List Guard Slice

Ledger IDs changed:
- `CRG3` remains in progress; GitHub artifact page responses now require an explicit `artifacts` list field.

Files changed:
- `tools/security/finalize_crg3.py`
- `docs/CRG3_CI_EVIDENCE.md`
- `tests/infra/test_production_readiness.py`
- `PRODUCTION_REFACTOR_ROADMAP.md`

Safety invariants proven:
- A GitHub artifact page missing the `artifacts` key is rejected as malformed metadata.
- The finalizer no longer treats omitted artifact lists as empty artifact lists.
- The CRG3 evidence runbook documents an explicit `artifacts` list as part of final artifact evidence.

Tests added or inverted:
- Added a finalizer regression for artifact page JSON with `total_count` but no `artifacts` field.
- Extended CRG3 runbook enforcement for explicit artifact-list wording.

Verification commands:

```bash
python tools/security/check_crg3_evidence_schema.py
python -m pytest tests/infra/test_production_readiness.py -q
python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md --allow-open-id CRG3
python -m pytest tests/infra/test_production_readiness.py tests/infra/test_deployment_hardening_docs.py -q
python -c "import sys, pytest; from tools.security.production_readiness import FOCUSED_P0_TESTS; sys.exit(pytest.main([*FOCUSED_P0_TESTS, '-q']))"
python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md
```

Results:
- `python tools/security/check_crg3_evidence_schema.py` passed.
- `python -m pytest tests/infra/test_production_readiness.py -q` passed: 133 passed in 5.19s.
- `python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md --allow-open-id CRG3` passed.
- `python -m pytest tests/infra/test_production_readiness.py tests/infra/test_deployment_hardening_docs.py -q` passed: 137 passed in 4.64s.
- Focused P0 suite passed: 394 passed, 4 warnings in 35.10s.
- Strict roadmap validation failed only because `CRG3` is still `In progress`; a Docker-capable GitHub Actions evidence run is still required.

Rollback plan:
- Revert this slice only if artifact page parsing moves to a dedicated GitHub API schema validator that still rejects omitted artifact-list fields.

Residual risk:
- Explicit artifact-list validation improves CRG3 artifact metadata integrity, but CRG3 still needs Docker-capable GitHub Actions evidence and downloaded artifact validation before it can be verified.

Live-block status:
- `CRG3` remains the only open P0 ledger item; production live enablement remains blocked until strict roadmap validation passes with `CRG3` marked `Verified YYYY-MM-DD`.

### 2026-06-04 Pass 6 CRG3 Workflow Duplicate YAML Key Guard Slice

Ledger IDs changed:
- `CRG3` remains in progress; local production-readiness workflow parsing now rejects duplicate YAML mapping keys.

Files changed:
- `tools/security/finalize_crg3.py`
- `docs/CRG3_CI_EVIDENCE.md`
- `tests/infra/test_production_readiness.py`
- `PRODUCTION_REFACTOR_ROADMAP.md`

Safety invariants proven:
- Duplicate workflow YAML keys fail closed before workflow structure validation.
- The finalizer no longer relies on `yaml.safe_load` behavior that silently keeps the last duplicate key value.
- The CRG3 evidence runbook documents duplicate YAML mapping key rejection as part of workflow validation.

Tests added or inverted:
- Added a finalizer regression for duplicate `env` keys in the readiness job.
- Extended CRG3 runbook enforcement for duplicate YAML mapping key rejection.

Verification commands:

```bash
python tools/security/check_crg3_evidence_schema.py
python -m pytest tests/infra/test_production_readiness.py -q
python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md --allow-open-id CRG3
python -m pytest tests/infra/test_production_readiness.py tests/infra/test_deployment_hardening_docs.py -q
python -c "import sys, pytest; from tools.security.production_readiness import FOCUSED_P0_TESTS; sys.exit(pytest.main([*FOCUSED_P0_TESTS, '-q']))"
python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md
```

Results:
- `python tools/security/check_crg3_evidence_schema.py` passed.
- `python -m pytest tests/infra/test_production_readiness.py -q` passed: 134 passed in 5.48s.
- `python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md --allow-open-id CRG3` passed.
- `python -m pytest tests/infra/test_production_readiness.py tests/infra/test_deployment_hardening_docs.py -q` passed: 138 passed in 4.80s.
- Focused P0 suite passed: 395 passed, 4 warnings in 34.48s.
- Strict roadmap validation failed only because `CRG3` is still `In progress`; a Docker-capable GitHub Actions evidence run is still required.

Rollback plan:
- Revert this slice only if workflow validation moves to a dedicated YAML schema parser that still rejects duplicate mapping keys.

Residual risk:
- Duplicate YAML key rejection improves local workflow evidence integrity, but CRG3 still needs Docker-capable GitHub Actions evidence and downloaded artifact validation before it can be verified.

Live-block status:
- `CRG3` remains the only open P0 ledger item; production live enablement remains blocked until strict roadmap validation passes with `CRG3` marked `Verified YYYY-MM-DD`.

### 2026-06-04 Pass 6 CRG3 Origin Fetch Remote Binding Slice

Ledger IDs changed:
- `CRG3` remains in progress; automatic repository approval now binds final evidence to the configured GitHub `origin` fetch remote only.

Files changed:
- `tools/security/finalize_crg3.py`
- `docs/CRG3_CI_EVIDENCE.md`
- `tests/infra/test_production_readiness.py`
- `PRODUCTION_REFACTOR_ROADMAP.md`

Safety invariants proven:
- Extra GitHub remotes such as `new-origin` are ignored for CRG3 run URL approval.
- Missing or non-GitHub `origin` fetch remotes fail closed before final evidence can be accepted.
- The CRG3 evidence runbook now documents `origin` fetch remote binding instead of broad configured-remote matching.

Tests added or inverted:
- Added a regression proving `_allowed_github_repos()` uses only `origin ... (fetch)` even when another GitHub remote exists.
- Added a regression proving a checkout without a GitHub `origin` fetch remote is rejected.
- Updated CRG3 runbook enforcement for `origin` fetch remote wording.

Verification commands:

```bash
python tools/security/check_crg3_evidence_schema.py
python -m pytest tests/infra/test_production_readiness.py -q
python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md --allow-open-id CRG3
python -m pytest tests/infra/test_production_readiness.py tests/infra/test_deployment_hardening_docs.py -q
python -c "import sys, pytest; from tools.security.production_readiness import FOCUSED_P0_TESTS; sys.exit(pytest.main([*FOCUSED_P0_TESTS, '-q']))"
python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md
```

Results:
- `python tools/security/check_crg3_evidence_schema.py` passed.
- `python -m pytest tests/infra/test_production_readiness.py -q` passed: 136 passed in 5.22s.
- `python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md --allow-open-id CRG3` passed.
- `python -m pytest tests/infra/test_production_readiness.py tests/infra/test_deployment_hardening_docs.py -q` passed: 140 passed in 4.79s.
- Focused P0 suite passed: 397 passed, 4 warnings in 28.98s.
- Strict roadmap validation failed only because `CRG3` is still `In progress`; a Docker-capable GitHub Actions evidence run is still required.

Rollback plan:
- Revert this slice only if repository binding moves to a signed release provenance source that is stronger than local `origin` fetch remote matching.

Residual risk:
- Origin fetch binding improves CRG3 repository identity checks, but CRG3 still needs Docker-capable GitHub Actions evidence and downloaded artifact validation before it can be verified.

Live-block status:
- `CRG3` remains the only open P0 ledger item; production live enablement remains blocked until strict roadmap validation passes with `CRG3` marked `Verified YYYY-MM-DD`.

### 2026-06-04 Pass 6 CRG3 Repository Binding Source Report Slice

Ledger IDs changed:
- `CRG3` remains in progress; finalizer-generated roadmap proof now records the repository binding source.

Files changed:
- `tools/security/finalize_crg3.py`
- `docs/CRG3_CI_EVIDENCE.md`
- `tests/infra/test_production_readiness.py`
- `PRODUCTION_REFACTOR_ROADMAP.md`

Safety invariants proven:
- CRG3 finalizer output records whether repository approval came from normal `origin` fetch remote binding or an explicit injected allowed-repo set.
- The CRG3 ledger proof text includes `repo binding`.
- The CRG3 finalization report includes `Repository binding source`.

Tests added or inverted:
- Extended finalization report regression to assert `repo binding: explicit allowed repository set`.
- Extended finalization report regression to assert `Repository binding source: explicit allowed repository set`.
- Extended CRG3 runbook enforcement for repository binding source wording.

Verification commands:

```bash
python tools/security/check_crg3_evidence_schema.py
python -m pytest tests/infra/test_production_readiness.py -q
python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md --allow-open-id CRG3
python -m pytest tests/infra/test_production_readiness.py tests/infra/test_deployment_hardening_docs.py -q
python -c "import sys, pytest; from tools.security.production_readiness import FOCUSED_P0_TESTS; sys.exit(pytest.main([*FOCUSED_P0_TESTS, '-q']))"
python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md
```

Results:
- `python tools/security/check_crg3_evidence_schema.py` passed.
- `python -m pytest tests/infra/test_production_readiness.py -q` passed: 136 passed in 5.20s.
- `python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md --allow-open-id CRG3` passed.
- `python -m pytest tests/infra/test_production_readiness.py tests/infra/test_deployment_hardening_docs.py -q` passed: 140 passed in 4.55s.
- Focused P0 suite passed: 397 passed, 4 warnings in 37.86s.
- Strict roadmap validation failed only because `CRG3` is still `In progress`; a Docker-capable GitHub Actions evidence run is still required.

Rollback plan:
- Revert this slice only if finalization evidence moves to a richer structured provenance document that still records repository binding source.

Residual risk:
- Repository binding source visibility improves CRG3 auditability, but CRG3 still needs Docker-capable GitHub Actions evidence and downloaded artifact validation before it can be verified.

Live-block status:
- `CRG3` remains the only open P0 ledger item; production live enablement remains blocked until strict roadmap validation passes with `CRG3` marked `Verified YYYY-MM-DD`.

### 2026-06-04 Pass 6 CRG3 Repository Binding Source Schema Slice

Ledger IDs changed:
- `CRG3` remains in progress; machine-readable CRG3 evidence now records and schema-validates the repository binding source.

Files changed:
- `docs/CRG3_EVIDENCE_SCHEMA.json`
- `tools/security/check_crg3_evidence_schema.py`
- `tests/infra/test_production_readiness.py`
- `PRODUCTION_REFACTOR_ROADMAP.md`

Safety invariants proven:
- Generated CRG3 evidence JSON includes `repository_binding_source`.
- CRG3 evidence schema version 9 accepts only `origin fetch remote` or `explicit allowed repository set` as repository binding sources.
- The standalone CRG3 schema validator sample exercises the repository binding source field.

Tests added or inverted:
- Added schema-version regression coverage for the repository binding source pattern.
- Added evidence validation regression proving an unrecognized repository binding source such as `new-origin fetch remote` is rejected.

Verification commands:

```bash
python tools/security/check_crg3_evidence_schema.py
python -m pytest tests/infra/test_production_readiness.py -q
python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md --allow-open-id CRG3
python -m pytest tests/infra/test_production_readiness.py tests/infra/test_deployment_hardening_docs.py -q
python -c "import sys, pytest; from tools.security.production_readiness import FOCUSED_P0_TESTS; sys.exit(pytest.main([*FOCUSED_P0_TESTS, '-q']))"
python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md
```

Results:
- `python tools/security/check_crg3_evidence_schema.py` passed.
- `python -m pytest tests/infra/test_production_readiness.py -q` passed: 136 passed in 5.34s.
- `python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md --allow-open-id CRG3` passed.
- `python -m pytest tests/infra/test_production_readiness.py tests/infra/test_deployment_hardening_docs.py -q` passed: 140 passed in 4.66s.
- Focused P0 suite passed: 397 passed, 4 warnings in 34.17s.
- Strict roadmap validation failed only because `CRG3` is still `In progress`; a Docker-capable GitHub Actions evidence run is still required.

Rollback plan:
- Revert this slice only if CRG3 evidence validation moves to a richer provenance schema that still records and bounds repository binding source.

Residual risk:
- Repository binding source schema validation improves CRG3 evidence integrity, but CRG3 still needs Docker-capable GitHub Actions evidence and downloaded artifact validation before it can be verified.

Live-block status:
- `CRG3` remains the only open P0 ledger item; production live enablement remains blocked until strict roadmap validation passes with `CRG3` marked `Verified YYYY-MM-DD`.

### 2026-06-04 Pass 6 CRG3 Local Readiness Probe Slice

Ledger IDs changed:
- `CRG3` remains in progress; the local readiness profile now allows only `CRG3` to remain open so workstation probes can complete while external GitHub Actions evidence is still pending.

Files changed:
- `tools/security/production_readiness.py`
- `docs/PRODUCTION_READINESS.md`
- `tests/infra/test_production_readiness.py`
- `PRODUCTION_REFACTOR_ROADMAP.md`

Safety invariants proven:
- `--profile local` passes `--allow-open-id CRG3` to roadmap validation, matching the non-live developer probe contract.
- `--profile ci` still allows only `CRG3` to bootstrap workflow evidence.
- `--profile live` remains strict and allows no open P0 ledger rows.
- Local profile success remains explicitly non-deployment evidence.

Tests added or inverted:
- Updated readiness command coverage to prove local and CI roadmap checks allow only `CRG3` open while live remains strict.
- Extended the local-profile `main()` test to assert the CRG3 allowance is passed into generated checks.
- Extended production readiness docs enforcement for local-probe CRG3-only wording.

Verification commands:

```bash
python -m pytest tests/infra/test_production_readiness.py tests/infra/test_deployment_hardening_docs.py -q
python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md --allow-open-id CRG3
python tools/security/production_readiness.py --profile local --skip-clean-check --skip-docker --skip-audit
```

Results:
- `python -m pytest tests/infra/test_production_readiness.py tests/infra/test_deployment_hardening_docs.py -q` passed: 140 passed in 5.15s.
- `python tools/security/check_roadmap.py PRODUCTION_REFACTOR_ROADMAP.md --allow-open-id CRG3` passed.
- `python tools/security/production_readiness.py --profile local --skip-clean-check --skip-docker --skip-audit` passed.
- Local readiness included focused P0 suite: 397 passed, 4 warnings in 33.32s.
- Local readiness included full pytest suite: 838 passed, 27 warnings in 150.54s.
- Local readiness also passed tracked-file secret scan, hashed CI/runtime lock dry-runs, Linux CPython 3.11 CI/runtime wheel availability, CycloneDX SBOM generation, CRG3 evidence schema validation, roadmap validation with `CRG3` allowed open, and scanned release artifact packaging from the dirty worktree temporary index.
- Local readiness skipped Docker and dependency audit by developer flag; this is not live deployment evidence.

Rollback plan:
- Revert this slice only if local implementation probes are intentionally required to fail until CRG3 external evidence exists; preserve live-profile strictness either way.

Residual risk:
- Local readiness probes are now more useful during implementation, but CRG3 still needs Docker-capable GitHub Actions evidence, dependency audit, Docker compose/build checks, and downloaded SBOM artifact validation before it can be verified.

Live-block status:
- `CRG3` remains the only open P0 ledger item; production live enablement remains blocked until strict roadmap validation passes with `CRG3` marked `Verified YYYY-MM-DD`.

## Lead Refactor Rules

- Preserve user work in the dirty tree; do not revert unrelated changes.
- Prefer narrow safety refactors over broad rewrites.
- Implement tests before or with each fix when the current test suite encodes unsafe behavior.
- Do not mark a pass complete without proof from tests, config output, or explicit manual verification.
- Keep live deployment blocked until every P0 gate is proven.

### 2026-06-06 Pass 6 CRG3 Workflow Evidence Finalization

Ledger IDs changed:
- `CRG3` verified from successful Docker-capable GitHub Actions evidence.

Files changed:
- `PRODUCTION_REFACTOR_ROADMAP.md`

Safety invariants proven:
- Production readiness workflow ran on a clean GitHub Actions runner.
- Finalizer-generated roadmap and report proof use the same exact completed check names as CRG3 evidence JSON.
- Downloaded `production-sbom` artifact contains exactly one valid CycloneDX `sbom.cdx.json` whose unique `type`/`name`/`version`/`purl` components match `requirements.lock` exactly.
- CRG3 evidence records the `requirements.lock` SHA-256 digest used for SBOM matching.
- `production-sbom` artifact metadata proves explicit `expired: false`, bounded `size_in_bytes`, and artifact `id` matching `archive_download_url`.
- Artifact download URL is repository-bound and canonical with no query string or fragment.
- GitHub API/artifact response reads are bounded, and cross-host artifact download redirects strip `Authorization`.
- Local production readiness workflow keeps manual dispatch, pull-request probes, release-branch push triggers, read-only permissions, same-ref concurrency without cancellation, Ubuntu runner, and 45-minute timeout.
- Local production readiness workflow pins job-level `BOT_V2_ALLOW_LIVE_EXECUTION=0`.
- Local production readiness workflow uses pinned checkout/setup actions, disables persisted checkout credentials, installs hash-locked CI tools before hash-locked runtime dependencies, and runs the wrapper before SBOM upload.
- Local production readiness workflow uploads `build/security/sbom.cdx.json` as exactly one `production-sbom` artifact with missing-file failure enabled.

Completed checks:
- `focused P0 regression suite`
- `full pytest suite`
- `unsafe deployment docs grep`
- `tracked-file secret scan`
- `hashed CI tool lock dry-run`
- `Linux CPython 3.11 CI tool wheel availability`
- `hashed dependency lock dry-run`
- `Linux CPython 3.11 locked wheel availability`
- `dependency vulnerability audit`
- `CycloneDX SBOM generation`
- `CRG3 evidence schema validation`
- `roadmap P0 ledger evidence`
- `scanned release artifact packaging`
- `default compose config`
- `production compose config`
- `production image build`
- `production-sbom artifact upload`
- `downloaded CycloneDX SBOM artifact validation`

Verification evidence:
- Repository: `chainsyncstore/hypothesis-research-engine`
- Repository binding source: `origin fetch remote`
- GitHub API repository.full_name: `chainsyncstore/hypothesis-research-engine`
- Workflow run: https://github.com/chainsyncstore/hypothesis-research-engine/actions/runs/27053934293
- Workflow run ID: `27053934293`
- Workflow run attempt: `1`
- Commit SHA: `c6ee071cb628cb7a48b3d2a42f38093b4f6c2677`
- Local checkout HEAD: `c6ee071cb628cb7a48b3d2a42f38093b4f6c2677`
- Local working tree: clean
- SBOM artifact: `production-sbom`
- SBOM artifact ID: `7451606425`
- SBOM artifact size: `1091`
- SBOM artifact downloaded size: `1091`
- SBOM artifact download URL: `https://api.github.com/repos/chainsyncstore/hypothesis-research-engine/actions/artifacts/7451606425/zip`
- SBOM artifact SHA-256: `e1bbd232a21224b4564579795f1187645f6825d5ae66f8530f080cba8e826127`
- SBOM JSON SHA-256: `cd45dd0904fda96ed1b454a5bdcb818bc75b24c2f73b2939015dbd3a8d9fcecb`
- SBOM component count: `35`
- Runtime lockfile: `requirements.lock`
- Runtime lockfile SHA-256: `b88a2a692b7e33ebd94c5338df75acfff020d9e84a0f98272d6a1fba439ddac3`

Rollback plan:
- If this evidence is later invalidated, set `CRG3` back to `In progress`, keep `BOT_V2_ALLOW_LIVE_EXECUTION=0`, and rerun `python tools/security/production_readiness.py --profile ci` on a fixed commit.

Residual risk:
- Strict live readiness must still pass from a clean Docker-capable checkout before live enablement.

Live-block status:
- Live remains blocked until `python tools/security/production_readiness.py --profile live` passes after this ledger update.
