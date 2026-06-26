# Security Audit Roadmap

Created: 2026-06-03

## Objective

Run a delegated security audit after the deployment switch, with emphasis on bugs that could expose live trading, credentials, state integrity, or operational controls.

The audit lead owns final triage. Sub-agents should produce end notes only: concise findings, evidence, exploit/impact reasoning, and recommended verification. Do not paste secrets or raw credential values in notes.

## Current Risk Surface

- Active runtime is a multi-user Telegram trading platform with paper/live sessions.
- Live execution is controlled by env gates such as `BOT_V2_ALLOW_LIVE_EXECUTION`, go/no-go flags, rollback flags, and runtime lifecycle settings.
- Deployment uses Docker Compose with Redis, Telegram bot, retrain scheduler, mounted model/state volumes, and `.env`-sourced secrets.
- Redis is part of the command/event path, WAL replay, watchdogs, reconciliation, and Telegram/execution coordination.
- User credentials are encrypted through `BOT_MASTER_KEY`; Telegram admin and user controls are authorization-critical.
- Model registry and retrain promotion can affect live decisions and must be treated as part of the control plane.
- Repository contains historical deploy archives, local DBs, logs, keys, and audit artifacts that may affect deployment hygiene even when not copied into the Docker image.

## Severity Rubric

- **Critical**: realistic path to unauthorized live order placement, credential disclosure, account takeover, irreversible fund loss, or bypass of kill-switch/risk gates.
- **High**: live-trading safety control bypass, durable state corruption, unauthenticated command injection, replay/double-execution, or deployment exposure with clear attack path.
- **Medium**: privilege confusion, insecure defaults, fragile recovery behavior, secret/log leakage with constrained impact, or unsafe operational ambiguity.
- **Low**: hardening gap, missing regression coverage, weak documentation, or low-probability issue needing additional assumptions.

## Pass Order

1. **Deployment, Secrets, and Runtime Exposure**
   - Files: `docker-compose*.yml`, `Dockerfile`, `.dockerignore`, `.gitignore`, `DEPLOY.md`, `AWS_DEPLOY.md`, `deploy/*`, root artifacts.
   - Questions: What changed in deployment? Are services exposed unnecessarily? Are secrets or state mounted/logged/copied? Are live defaults safe?

2. **Telegram Control Plane and Authorization**
   - Files: `quant/telebot/main.py`, `quant/telebot/auth.py`, `quant/telebot/manager.py`, `quant/telebot/models.py`, related tests.
   - Questions: Can non-admin users reach admin actions? Are user sessions isolated? Are credentials encrypted, rotated, and scrubbed? Are live starts gated correctly?

3. **Execution Service, Risk Gates, and Live Order Safety**
   - Files: `quant_v2/execution/*`, `quant_v2/portfolio/*`, `quant_v2/monitoring/*`, `quant_v2/contracts.py`.
   - Questions: Can stale state, bad signals, or env flags route unsafe orders? Are kill-switches, go/no-go checks, lifecycle rules, and exposure caps enforced under live mode?

4. **Redis Bus, WAL, Idempotency, and Concurrency**
   - Files: `quant_v2/execution/redis_bus.py`, `state_wal.py`, `idempotency.py`, `watchdog.py`, server startup/shutdown paths.
   - Questions: Can commands be replayed, duplicated, forged, or lost? Are pending Redis stream messages handled? Can concurrent watchdog/reconciliation/user commands double-flatten or double-enter?

5. **Model Registry, Retraining, and Data Integrity**
   - Files: `quant_v2/model_registry.py`, `quant_v2/models/*`, `quant_v2/research/*`, `bootstrap_registry.py`, model volume paths.
   - Questions: Can retrain or registry state promote untrusted models? Are models loaded safely? Can poisoned/missing data disable safety checks or alter live decisions?

6. **Dependency, Build, and Artifact Hygiene**
   - Files: `pyproject.toml`, `Dockerfile`, archives, logs, DBs, local scripts, ignored/unignored files.
   - Questions: Are dependency versions pinned enough for production? Are private keys, DBs, archives, logs, or local debug scripts deployable/leakable? Is the Docker build reproducible?

7. **Regression and Verification Plan**
   - Files: `tests/**` plus targeted new/updated tests proposed by prior passes.
   - Questions: Which findings need tests? Which checks can be automated in CI? Which deployment assertions should fail closed?

## Sub-Agent End Notes Template

Each sub-agent must end with:

```text
END NOTES
Scope reviewed:
- ...

Findings:
- [Severity] Title
  Evidence: file:line or command result summary
  Why it matters: concise exploit/failure path
  Suggested fix: concrete change
  Verification: test/command/manual check

No-finding areas:
- ...

Open questions:
- ...
```

## Findings Ledger

Use this section as sub-agent notes arrive. Keep raw notes outside final chat unless needed; triage here into confirmed, needs-proof, duplicate, or rejected.

| ID | Pass | Severity | Status | Finding | Evidence | Next action |
| --- | --- | --- | --- | --- | --- | --- |
| D1 | Deployment, Secrets, and Runtime Exposure | High | Confirmed | Redis is published on all host interfaces by default without auth/ACLs. | `docker-compose.yml:9-13` maps `6379:6379` and runs `redis-server --appendonly yes`; no Redis password/TLS config found. | Remove host publish or bind to `127.0.0.1`; add Redis ACL/auth if host access is required; verify with `docker compose config` and host listener checks. |
| D2 | Deployment, Secrets, and Runtime Exposure | High | Confirmed | A tracked deployment archive contains files excluded by current ignore rules. | `git ls-files` shows `deploy_optimized.tar.gz`; archive listing includes `quant-key.pem`, `debug_credentials.py`, production model paths, and `signal_log.json`. | Remove archive from tracked history if exposed, rotate referenced key material, and add CI/pre-commit archive/secret scanning. |
| D3 | Deployment, Secrets, and Runtime Exposure | High | Confirmed | Standalone execution engine is live-capable by default and bypasses Telegram's fail-closed live env gate. | `Dockerfile:54` defaults to `python -m quant_v2.execution.main`; `quant_v2/execution/main.py:72` passes `allow_live_execution=True`; Telegram bridge defaults `BOT_V2_ALLOW_LIVE_EXECUTION=0` at `quant/telebot/main.py:83`. | Make standalone execution parse `BOT_V2_ALLOW_LIVE_EXECUTION` with default false; add a regression asserting unset live env rejects or shadows live sessions. |
| D4 | Deployment, Secrets, and Runtime Exposure | High | Confirmed | Local root artifacts create a deploy-time secret/state leakage path. | Root inventory includes `.env`, `quant-key.pem`, `quant_bot.db`, `quant_bot_remote.db`, and multiple archives; `AWS_DEPLOY.md:66-70` documents `scp -r` of the local repo tree. | Deploy from clean git/release artifacts only; move host secrets outside repo root; add an artifact scan before deploy for `.env`, `*.pem`, `*.db`, and `*.tar.gz`. |
| D5 | Deployment, Secrets, and Runtime Exposure | Medium | Confirmed | Production compose does not match documented/default runtime and can select the standalone live-capable path. | `docker-compose.prod.yml:4-14` defines only `quant_bot`, no Redis, no command override; `Dockerfile:54` supplies standalone execution CMD; `AWS_DEPLOY.md:96` documents default compose instead. | Pick one production topology; make commands explicit; include private Redis if required; smoke-test process identity and Redis connectivity. |
| D6 | Deployment, Secrets, and Runtime Exposure | Medium | Confirmed | Shutdown and monitoring scripts watch `quant_execution`, which does not match compose container names. | Compose names include `quant_redis`, `quant_telegram`, `quant_retrain`, and `quant_bot`; `deploy/flatten_on_shutdown.sh:16`, `deploy/dead_mans_switch.sh:16`, and `deploy/setup_cloudwatch.sh:92` default to `quant_execution`. | Align container names or require `QUANT_CONTAINER_NAME` in installed service/cron environment; run script smoke tests with default env. |
| D7 | Deployment, Secrets, and Runtime Exposure | Medium | Confirmed | Debug/local scripts can disclose credentials or runtime state if run or archived. | `debug_credentials.py` loads credential env vars and prints failed response bodies; local analyzers inspect Redis/SQLite state; tracked archive includes `debug_credentials.py`. | Remove credential scripts from deployable trees, redact all secret/session fields, and gate ops analyzers outside release artifacts. |
| D8 | Deployment, Secrets, and Runtime Exposure | Medium | Confirmed | Runtime container hardening is incomplete for production. | Compose has writable host mounts for models/state/logs and lacks `read_only`, `cap_drop`, `security_opt`, healthchecks, and secrets support; Dockerfile does run as non-root. | Add least-privilege compose hardening where practical and separate writable state/tmpfs mounts from read-only app/model mounts. |
| D9 | Deployment, Secrets, and Runtime Exposure | Medium | Confirmed | AWS guide encourages broad SSH exposure and includes active infrastructure details. | `AWS_DEPLOY.md:3-4` lists active server/key metadata; `AWS_DEPLOY.md:27` says SSH from `0.0.0.0/0`; `AWS_DEPLOY.md:41` includes a concrete SSH target. | Restrict SSH to admin IPs or SSM Session Manager; remove active IP/key-path inventory from repo docs. |
| D10 | Deployment, Secrets, and Runtime Exposure | Medium | Needs proof | Redis host exposure impact depends on actual EC2/VPS firewall rules. | Compose publishes Redis, but cloud firewall state was not inspected in this repo-only pass. | Check EC2/VPS inbound rules and host firewall; if port 6379 is internet-reachable, escalate D1 impact toward critical operational exposure. |
| T1 | Telegram Control Plane and Authorization | High | Confirmed | Catch-all Telegram debug handler can log raw credential setup messages. | `quant/telebot/main.py:3809-3814` logs full `Update` objects via `MessageHandler(filters.ALL, debug_log)`; `/setup` accepts API key/secret at `quant/telebot/main.py:3438-3464`. | Remove production catch-all logging or redact message text; add log-capture regression that `/setup` secrets are absent from logs. |
| T2 | Telegram Control Plane and Authorization | Medium | Confirmed | Failed live starts can leave the v2 signal source running after rollout gates block execution. | `_start_v2_primary_sessions` starts source before bridge at `quant/telebot/main.py:1789-1801`; cleanup only runs after both calls return at `quant/telebot/main.py:1803-1812`; live gate failure raises at `quant_v2/execution/service.py:1805-1812`. | Wrap bridge start in `try/finally` to stop newly-started source sessions, or gate execution before source startup; add fake bridge/source regression. |
| T3 | Telegram Control Plane and Authorization | Medium | Confirmed | Sensitive runtime exception text is reflected to users, logs, or crash notifications. | Raw exception strings are logged/replied at `quant/telebot/main.py:2407-2408` and `quant/telebot/main.py:2754-2760`; legacy auth wraps client exception text at `quant/telebot/manager.py:73-77`; crash signal includes `last_error` at `quant/telebot/engine.py:93-96`. | Use fixed user-facing errors, sanitized logs with correlation IDs, and centralized redaction for keys/tokens/headers; test with sentinel secret exceptions. |
| T4 | Telegram Control Plane and Authorization | Medium | Confirmed | Pending or banned users can persist Binance credentials. | New users default to `pending` at `quant/telebot/main.py:2134-2144`; start blocks pending/banned users at `quant/telebot/main.py:2270-2271`; `/setup` only checks that a user row exists before encrypted credential save at `quant/telebot/main.py:3458-3465`. | Require active/approved status for `/setup`; clear credentials on revoke if retention is unnecessary; add pending/banned setup tests. |
| T5 | Telegram Control Plane and Authorization | Low | Confirmed | Unbounded lifecycle horizon can crash a user's routing cycle. | `_normalize_lifecycle_horizon` only enforces non-negative values at `quant/telebot/main.py:528-537`; `/set_horizon` persists the value at `quant/telebot/main.py:2473-2488`; routing builds `timedelta(hours=...)` at `quant_v2/execution/service.py:1741-1744`. | Cap horizon to an operational range in Telegram validation and `LifecycleRules`; test huge inputs are rejected and routing does not raise. |
| E1 | Execution Service, Risk Gates, and Live Order Safety | High | Confirmed | Post-only live entries can fall back to market orders. | Service routes ordinary plans with `post_only=True` at `quant_v2/execution/service.py:1027-1042`; Binance adapter catches limit failure and calls `place_order(..., "MARKET")` at `quant_v2/execution/binance_adapter.py:95-116`; existing test asserts fallback success at `tests/quant_v2/test_binance_adapter.py:203-217`. | Propagate venue post-only semantics and fail closed on post-only entry rejection; reserve taker fallback only for explicit emergency reduce/flatten paths; add a no-market-fallback regression for `post_only=True`. |
| E2 | Execution Service, Risk Gates, and Live Order Safety | High | Confirmed | Missing prices make open positions invisible to risk checks and ordinary flatten planning. | Reconciler skips symbols with missing/non-positive price at `quant_v2/execution/reconciler.py:23-29`; snapshot risk omits such positions from notional/gross/net at `quant_v2/execution/service.py:2443-2487`; hard-risk breach depends on those fields at `quant_v2/execution/service.py:1547-1569`. | Treat missing or non-finite marks for non-zero live positions as hard-risk/stale-price state; require fresh finite marks before entries; generate reduce-only flatten where possible. |
| E3 | Execution Service, Risk Gates, and Live Order Safety | High | Confirmed | Circuit breaker can report orders canceled even when Binance cancels fail. | Adapter `cancel_all_orders` logs and swallows cancel exceptions at `quant_v2/execution/binance_adapter.py:373-383`; stale-feed circuit breaker appends symbols to `canceled_symbols` when the method returns at `quant_v2/execution/main.py:253-259`. | Make cancel failures observable via raised/per-order status, verify `get_open_orders` is empty before declaring success, and keep alert/retry state unresolved on failure. |
| E4 | Execution Service, Risk Gates, and Live Order Safety | Medium | Duplicate of T5 | Lifecycle horizon accepts values that can crash routing before forced exits. | Service `LifecycleRules` only rejects negative horizon at `quant_v2/execution/service.py:96-109`; `set_lifecycle_rules` casts to int at `quant_v2/execution/service.py:1441-1467`; routing converts to `timedelta(hours=...)` at `quant_v2/execution/service.py:1741-1744`. | Resolve with T5 by enforcing a bounded operational horizon at both Telegram and service boundaries; add huge/NaN/inf tests. |
| R1 | Redis Bus, WAL, Idempotency, and Concurrency | High | Confirmed | Stream retries can duplicate live orders after crash/restart. | Pending stream entries are redelivered on boot at `quant_v2/execution/redis_bus.py:403-405` and ACKed only after handler return at `quant_v2/execution/redis_bus.py:475-480`; idempotency keys use current minute at `quant_v2/execution/idempotency.py:25-39`; journal is in-memory only at `quant_v2/execution/idempotency.py:42-60`; adapter checks only that in-memory journal. | Persist idempotency/order-intent state before exchange placement, include stream message/correlation ID in deterministic keys, and pass deterministic client order IDs to Binance where supported. |
| R2 | Redis Bus, WAL, Idempotency, and Concurrency | High | Confirmed | Live WAL replay fails without credentials, leaving live exposure unmanaged after restart. | `_rebuild_state_from_wal` rebuilds `SessionRequest` without credentials at `quant_v2/execution/main.py:325-334`; WAL session-start payload stores live/profile/universe only at `quant_v2/execution/state_wal.py:168-184`; live adapter creation rejects missing credentials at `quant_v2/execution/service.py:2076-2079`. | Rebuild live sessions into explicit paused/flatten-only recovery or load credentials from a secure credential store; add live WAL replay tests for safe restore or fail-closed recovery. |
| R3 | Redis Bus, WAL, Idempotency, and Concurrency | High | Confirmed | Lifecycle and kill-switch state are not restored on WAL replay. | WAL defines lifecycle/kill-switch event types at `quant_v2/execution/state_wal.py:69-71`; `_cmd_set_lifecycle` updates watchdog without WAL append at `quant_v2/execution/main.py:679-697`; `_rebuild_state_from_wal` handles only `session_started`, `session_stopped`, and `state_checkpoint` at `quant_v2/execution/main.py:325-358`. | WAL-append lifecycle changes, replay lifecycle and kill-switch events before command consumption, and add restart tests for horizon, stop-loss, and pause state. |
| R4 | Redis Bus, WAL, Idempotency, and Concurrency | High | Confirmed | Graceful shutdown flatten path calls a nonexistent method and still emits success sentinel. | Shutdown calls `server._circuit_breaker_flatten(...)` at `quant_v2/execution/main.py:826-838`, but the available helper is `_execute_stale_feed_circuit_breaker`; `positions_flat` is logged unconditionally after `server.stop()` at `quant_v2/execution/main.py:844-845`. | Call the existing flatten helper, verify every live session result, and emit `positions_flat` only after confirmed flat/canceled state. |
| R5 | Redis Bus, WAL, Idempotency, and Concurrency | Medium | Confirmed | Command handler can ACK stream commands after safety side effects fail. | `_handle_command` catches exceptions and sends an error event without re-raise at `quant_v2/execution/main.py:499-534`; stream processing ACKs whenever handler returns at `quant_v2/execution/redis_bus.py:475-480`; `_cmd_route_signals` places orders before WAL order logging at `quant_v2/execution/main.py:609-626`. | Distinguish user-facing error events from stream success; re-raise unrecoverable persistence failures or make order placement and WAL append durably compensating before ACK. |
| R6 | Redis Bus, WAL, Idempotency, and Concurrency | Medium | Confirmed | Redis stream commands are trusted without message-level authorization. | `BusMessage.from_json` accepts action/payload/correlation ID directly at `quant_v2/execution/redis_bus.py:37-44`; `_handle_command` dispatches privileged actions at `quant_v2/execution/main.py:499-520`; `_cmd_start_session` accepts `live` and `credentials` from payload at `quant_v2/execution/main.py:536-544`. | Add signed/HMAC message envelopes or authenticated producer identity, enforce allowed actions per producer, and reject unsigned/tampered commands before dispatch. |
| M1 | Model Registry, Retraining, and Data Integrity | High | Confirmed | Active model artifacts are loaded with unsafe deserialization from weakly trusted paths. | `quant_v2/models/trainer.py:184-187` uses `joblib.load`; `quant_v2/models/ensemble.py:47-61` loads horizon model files from artifact directories; `quant_v2/telebot/signal_manager.py:730-738` loads the active registry model; registry validation only checks shallow artifact existence at `quant_v2/model_registry.py:303-319`. | Restrict artifacts to trusted roots, reject symlinks/out-of-root paths, require signed/hash-checked manifests, and validate-load artifacts in isolation before activation. |
| M2 | Model Registry, Retraining, and Data Integrity | High | Confirmed | Promotion checks fail open for incomplete or unvalidated models. | `_promotion_checks_pass` rejects only explicit false eligibility and accepts `config.json` or any `model_*m.pkl` at `quant_v2/model_registry.py:303-319`; bootstrap activates with `set_active_version` at `bootstrap_registry.py:81-90`; tests promote placeholder model files. | Require `promotion_eligible is True`, complete scorecard/required horizons, loadability checks, and prevent `set_active_version` from bypassing production promotion policy. |
| M3 | Model Registry, Retraining, and Data Integrity | High | Confirmed | Runtime can fall back to the latest filesystem model when the active registry pointer is invalid. | `resolve_model_dir` falls back to `find_latest_model` when active artifact is invalid at `quant/telebot/model_selection.py:39-71`; latest discovery accepts model-looking dirs at `quant/telebot/model_selection.py:21-36`; runtime uses this resolution at `quant/telebot/main.py:310-320`. | Fail closed in production when the active registry pointer is missing/invalid/quarantined; allow filesystem fallback only under explicit dev/bootstrap mode. |
| M4 | Model Registry, Retraining, and Data Integrity | Medium | Confirmed | Promotion and rollback are not transactional and do not prove the next runtime model is loadable. | `promote_version` writes active pointer before status update at `quant_v2/model_registry.py:161-184`; `_mark_active_status` updates version statuses one file at a time at `quant_v2/model_registry.py:321-329`; runtime load happens later in `quant_v2/telebot/signal_manager.py:730-738`. | Validate-load the full model set before switching, use a registry lock/transactional state file, and roll back only after confirming runtime load succeeded. |
| M5 | Model Registry, Retraining, and Data Integrity | Medium | Confirmed | Partial or wrong-horizon models can be used silently. | Scheduled retrain can register partial candidates when `RETRAIN_REQUIRE_ALL_HORIZONS` is false at `quant_v2/research/scheduled_retrain.py:355-369`; ensemble loads any available horizon at `quant_v2/models/ensemble.py:47-61`; signal manager falls back to any `2/4/8m` model at `quant_v2/telebot/signal_manager.py:1817-1834`. | Encode required horizons in registry metadata and reject activation/runtime loading unless configured horizons are complete. |
| M6 | Model Registry, Retraining, and Data Integrity | Medium | Confirmed | Auto-promote flag parsing treats unknown non-empty values as enabled. | `_env_flag` returns true for any value not in `0/false/no/off` at `quant_v2/research/scheduled_retrain.py:48-52`; auto-promotion is read at `quant_v2/research/scheduled_retrain.py:370-372` and executed at `quant_v2/research/scheduled_retrain.py:416-421`. | Parse booleans fail-closed: enable only `1/true/yes/on`, log unknown values, and add `BOT_RETRAIN_AUTO_PROMOTE=maybe` regression. |
| M7 | Model Registry, Retraining, and Data Integrity | Low | Confirmed | Legacy retrain labels unlabeled tail rows as class 0. | `_build_labels` casts `(future_return > 0)` directly to int at `quant_v2/research/retrain_pipeline.py:22-27`, so NaN future returns become false labels before the documented `notna()` drop at `quant_v2/research/retrain_pipeline.py:50-54`. | Preserve NaN labels until filtering, then cast valid rows to integer; add a test proving final horizon rows are excluded. |
| H1 | Dependency, Build, and Artifact Hygiene | High | Confirmed | Production dependency resolution is not reproducible. | `pyproject.toml:15-29` uses lower-bound-only dependency ranges; `Dockerfile:14-21` installs direct unpinned packages with `--extra-index-url`; no lockfile, constraints file, SBOM, or dependency-audit config found. | Add locked constraints with hashes, install with `--require-hashes`, pin Docker base images by digest, and run dependency audit/SBOM generation in CI. |
| H2 | Dependency, Build, and Artifact Hygiene | High | Duplicate of D2 | Tracked deployment archive bypasses current ignore rules and contains deploy-sensitive artifacts. | `deploy_optimized.tar.gz` is tracked; `.gitignore`/`.dockerignore` now exclude archives; archive listing includes key/debug/model/log entries. | Resolve under D2 with history cleanup, key rotation, and archive/secret scanning. |
| H3 | Dependency, Build, and Artifact Hygiene | High | Duplicate of D4 | Local deployment archives and root files contain recurring secret/state leakage paths. | Root contains `.env`, `quant-key.pem`, DB files, and multiple `*.tar.gz` archives; recursive deploy docs can upload them. | Resolve under D4 with clean release artifacts, secrets outside repo root, stale archive removal, and artifact scans. |
| H4 | Dependency, Build, and Artifact Hygiene | High | Confirmed | Diagnostic scripts can expose credentials, session headers, account data, Redis state, and SQLite contents. | `test_capital_auth.py:11-12` contains hardcoded credential fields and prints API/session/account data at `test_capital_auth.py:32-62`; `debug_credentials.py:6-40` loads and prints credential/API response diagnostics; `redis_analyzer_local.py:10-14`, `analyze_sqlite_local.py:11-24`, and `trade_analyzer_local.py:11-15` print runtime/DB state. | Delete hardcoded credentials, move diagnostics outside deployable trees, redact all token/session/DB output, and add scans for unsafe diagnostics. |
| H5 | Dependency, Build, and Artifact Hygiene | Medium | Duplicate of D8 | Docker runtime hardening is incomplete. | Dockerfile runs non-root but makes `/app` writable; compose lacks `read_only`, `cap_drop`, `security_opt`, and mounts writable model/state/log files. | Resolve under D8 with least-privilege compose hardening and read-only model mounts where practical. |
| H6 | Dependency, Build, and Artifact Hygiene | Medium | Duplicate of D4/D9 | Deployment docs encourage unsafe host exposure and dirty-tree transfer. | `AWS_DEPLOY.md:27` allows SSH from `0.0.0.0/0`; `AWS_DEPLOY.md:69` documents recursive local repo upload; `DEPLOY.md` permits broad upload/env export patterns. | Resolve under D4 and D9 by restricting SSH and deploying only clean signed artifacts with external secret handling. |
| H7 | Dependency, Build, and Artifact Hygiene | Medium | Duplicate of D6 | Ops scripts target stale container names, weakening shutdown/heartbeat protections. | Compose names `quant_telegram`/`quant_retrain`; deploy scripts default/check `quant_execution`. | Resolve under D6 by aligning names or requiring `QUANT_CONTAINER_NAME` in installed service/cron env. |

## Final Verification Synthesis

The final verification pass grouped the ledger into deployment-blocking remediation themes:

| Group | Severity | Related IDs | Summary | Required proof before live deployment |
| --- | --- | --- | --- | --- |
| Live-order fail-closed controls | Critical | D3, D5, E1, E2, E3, R1, R2, R4 | Standalone/prod paths can be live-capable by default; normal post-only orders can fall back to market; missing prices hide exposure; restart/replay/shutdown paths can duplicate orders or falsely report flat. | Tests for unset live env, prod compose command identity, post-only reject/no-market fallback, missing-price hard-risk handling, Redis redelivery idempotency, live WAL recovery, and shutdown sentinel only after confirmed flat. |
| Redis control-plane trust | Critical if externally reachable; High otherwise | D1, D10, R1, R5, R6 | Redis is host-published and commands are unsigned; replay/ACK/idempotency behavior can duplicate or lose safety-critical effects. | `docker compose config` shows no public Redis port; host/firewall scan proves 6379 closed; unsigned stream commands are rejected; redelivered route commands do not place duplicate orders. |
| Credential/key leakage | Critical | D2, D4, T1, T3, D7, H2, H3, H4 | Tracked/local archives and diagnostics expose key/state paths; Telegram catch-all logging can capture `/setup` secrets; raw exception text can leak sensitive diagnostics. | Full-history and current-tree secret scans, release artifact scan, key rotation record, and log-capture tests proving setup credentials and sentinel exception secrets are redacted. |
| Model artifact trust | High | M1, M2, M3, M4, M5, M6 | Runtime uses unsafe deserialization; promotion validation is shallow; invalid registry pointers can fall back to latest filesystem model; auto-promote parsing fails open. | Tests reject out-of-root/symlink/placeholder/corrupt artifacts, require explicit eligibility and complete horizons, fail closed on invalid active pointer, and keep `BOT_RETRAIN_AUTO_PROMOTE=maybe` disabled. |
| Recovery, shutdown, and kill-switch persistence | High | D6, R2, R3, R4, E3 | Scripts target stale container names; WAL replay omits live credentials/lifecycle/kill-switch state; cancel/flatten failures can be reported as success. | Script smoke test against compose config, live-session restart recovery test, lifecycle/kill-switch replay test, and cancel-failure test that keeps alert/retry state unresolved. |
| Build and deployment hygiene | High | H1, D8, D9, H5, H6 | Dependencies are unpinned/unlocked; container hardening is incomplete; docs encourage broad SSH and dirty-tree transfer. | Locked hashed dependencies, base image digests, dependency audit/SBOM, hardened compose config, restricted SSH guidance, and clean allowlisted release artifact process. |

Clarifications from final verification:

- Collapse duplicate ledger rows in external tracking: E4 -> T5; H2 -> D2; H3 -> D4; H5 -> D8; H6 -> D4/D9; H7 -> D6.
- Keep D10 as `Needs proof` until actual cloud firewall/security group state is inspected.
- Treat H4 as confirmed local/artifact risk unless full-history scan proves committed exposure beyond `deploy_optimized.tar.gz`.
- Some existing tests encode unsafe behavior and must be inverted, especially market fallback in `tests/quant_v2/test_binance_adapter.py` and placeholder model promotion in registry tests.

## Lead Audit Rules

- Redact secret values and tokens. It is acceptable to name env vars and file paths.
- Treat live trading enablement, order routing, and credential handling as highest-risk paths.
- Separate confirmed vulnerabilities from hardening gaps and operational risks.
- Prefer file/line evidence and minimal repro steps over broad claims.
- Before the final report, map each confirmed issue to severity, impact, recommended fix, and verification status.
