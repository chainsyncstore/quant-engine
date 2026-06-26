# Quant Engine Operational README

This repository is the canonical source for the current 4arm-hosted trading
system.

Canonical Git remote:

```text
https://github.com/chainsyncstore/quant-engine.git
```

Production host:

```text
ssh 4arm-ubuntu
/home/admin-4arm/hypothesis-research-engine
```

The previous Ubuntu remote is retained on 4arm only as `legacy-hypothesis`:

```text
https://github.com/chainsyncstore/hypothesis-research-engine.git
```

## Current Operating State

As of 2026-06-26, the system is intentionally running in a guarded
no-active-model/no-trade recovery posture while model quality is rebuilt.

Current behavior:

- Telegram, evaluator, retrain, and Redis services are running on 4arm.
- Runtime source is bind-mounted from the 4arm repo into Docker containers.
- The retrain service runs from `quant_v2.research.scheduled_retrain`.
- Retrain does not directly activate a fresh model.
- Passing retrain output is registered for paper quarantine/review first.
- Trading should not resume until a candidate passes validation, benchmark
  replay, and forward paper-soak gates.

Important current environment policy:

```text
BOT_RETRAIN_AUTO_PROMOTE=0
RETRAIN_MIN_ACCURACY=0.60
RETRAIN_REQUIRE_ALL_HORIZONS=1
RETRAIN_REQUIRE_ALL_SYMBOLS=1
RETRAIN_STARTUP_DELAY_SECONDS=3600
RETRAIN_INTERVAL_HOURS=168
RETRAIN_TRAIN_MONTHS=6
```

This means:

- If retrain fails validation, the system remains no-trade.
- If retrain passes validation, the candidate is still not active by default.
- Paper trading does not automatically resume just because retrain completed.
- Manual/governed promotion is required after evidence review.

## Active Runtime Services

The 4arm deployment currently runs these containers:

```text
quant_telegram
quant_model_eval
quant_retrain
quant_redis
```

The live containers bind-mount these source/data paths:

```text
/home/admin-4arm/hypothesis-research-engine/quant    -> /app/quant
/home/admin-4arm/hypothesis-research-engine/quant_v2 -> /app/quant_v2
/home/admin-4arm/hypothesis-research-engine/models   -> /app/models
/home/admin-4arm/hypothesis-research-engine/state    -> /state
```

`scripts/` is not mounted into the long-running containers. Operator scripts
can be run on the host or in a disposable `quant_bot:latest` container with the
repo mounted.

## Model Quality Recovery Flow

The current model recovery path is documented in:

- `MODEL_QUALITY_RECOVERY_SPEC.md`
- `docs/model_quality/README.md`
- `docs/model_quality/validation_policy_v1.md`
- `scripts/model_quality_recovery.py`

The recovery tool produces:

- failed retrain diagnostics
- label/dead-zone audit
- transparent benchmark replay
- candidate selection report
- validation policy evidence

Run locally or on 4arm with an existing dataset snapshot:

```bash
python scripts/model_quality_recovery.py \
  --model-root models/production \
  --registry-root models/production/registry \
  --snapshot-path datasets/v2/snapshots/<snapshot>.parquet \
  --output-dir docs/model_quality
```

When running inside the production image:

```bash
docker run --rm --user 0 -e PYTHONDONTWRITEBYTECODE=1 \
  -v /home/admin-4arm/hypothesis-research-engine:/app:ro \
  -w /app quant_bot:latest \
  python scripts/model_quality_recovery.py --help
```

## Telegram Usage

User-facing commands include:

```text
/start
/start_demo
/start_live
/stop
/status
/stats
/reset_demo
/continue_demo
/continue_live
```

Admin/operator commands include model registry and promotion controls:

```text
/model_versions
/model_candidates
/model_eval_status
/model_eval_detail <version_id>
/model_auto_promote on|off
/model_approve <version_id> <evidence_digest>
/model_promote <version_id> [evidence_digest] [reason]
/model_quarantine <version_id>
/model_rollback
/prepare_update
/update_complete
```

Operational rule: do not use these commands to force production trading while
the system is in recovery/no-active-model mode. Promotion should follow the
current validation policy and paper-soak gates.

## Deployment To 4arm

Typical source deployment path:

```bash
ssh 4arm-ubuntu
cd /home/admin-4arm/hypothesis-research-engine
git fetch origin main
git status --short
```

For documentation or source-only changes, verify the target files and avoid
touching runtime state:

```bash
git diff --stat HEAD..origin/main
```

For runtime code changes, validate in the production image before restarting
affected services. Example:

```bash
docker run --rm --user 0 -e PYTHONDONTWRITEBYTECODE=1 \
  -v /home/admin-4arm/hypothesis-research-engine:/app:ro \
  -w /app quant_bot:latest \
  python -m compileall -q quant quant_v2 scripts
```

Restart only the affected service. For retrain-only changes:

```bash
docker restart quant_retrain
```

Avoid restarting `quant_telegram` unless the bot/runtime session layer changed
or a controlled maintenance window is intended.

## Local Development

Install:

```bash
pip install -e .
pip install -e ".[dev]"
```

Run focused validation:

```bash
pytest tests/quant_v2/test_model_quality_recovery.py -q
```

Run broader tests when touching runtime logic:

```bash
pytest tests/ -q
```

Required local environment for running the Telegram bot:

```text
TELEGRAM_TOKEN
ADMIN_ID
BOT_MASTER_KEY
```

Do not commit `.env`, database files, model artifacts, keys, state files, or
production backups.

## Safety Posture

Live trading remains blocked unless all production-resume gates are satisfied:

- fresh candidate artifact is complete and manifest-valid
- candidate passes the current validation policy
- candidate beats flat and transparent benchmark replay after costs
- candidate completes forward paper soak with positive expectancy
- paper books reconcile flat before live canary
- no active hard-risk pause exists
- source SHA, image, env policy, and model manifest match

The immediate objective is not to make the bot trade. It is to make the next
trade defensible.

## License

Research and internal operations use.
