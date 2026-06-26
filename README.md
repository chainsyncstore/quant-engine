# Quant Engine

Quant Engine is a Python research and execution framework for cryptocurrency
trading-system experiments. It combines market-data ingestion, feature
engineering, model training, portfolio/risk controls, Telegram-facing session
management, and operational safety tooling.

This repository is intended to be safe for public collaboration. It does not
contain production secrets, exchange credentials, user data, model artifacts,
runtime databases, or host-specific deployment state.

## What Is Included

- Market-data clients and feature pipelines for crypto futures research.
- Model-training and model-registry utilities under `quant_v2`.
- Portfolio, cost, risk, and execution-planning components.
- Telegram bot command handlers and paper/live session orchestration code.
- Model-quality recovery diagnostics and benchmark replay tooling.
- Tests and operational hardening utilities.

## Current Safety Posture

This codebase is designed to fail closed:

- A retrain can be rejected when quality gates are not met.
- Fresh model artifacts should not become active merely because training ran.
- Candidate models should pass validation, benchmark replay, and forward
  paper-soak evidence before production use.
- Live trading should require explicit operator approval and verified runtime
  state.

The repository may include tools for live execution, but public source code is
not proof that any deployment is currently approved to trade.

## Model Quality Recovery

The current recovery workflow is documented in:

- `MODEL_QUALITY_RECOVERY_SPEC.md`
- `docs/model_quality/README.md`
- `docs/model_quality/validation_policy_v1.md`
- `scripts/model_quality_recovery.py`

The recovery tool can produce:

- failed retrain diagnostics
- label/dead-zone audits
- transparent benchmark replay
- candidate-selection reports
- validation-policy evidence

Example:

```bash
python scripts/model_quality_recovery.py \
  --model-root models/production \
  --registry-root models/production/registry \
  --snapshot-path datasets/v2/snapshots/<snapshot>.parquet \
  --output-dir docs/model_quality
```

`models/`, `datasets/`, and generated reports are deployment/runtime artifacts
unless intentionally checked in as sanitized evidence.

## Telegram Bot

The Telegram layer supports user-facing session commands such as starting,
stopping, checking status, and resetting paper sessions. Administrative model
registry and promotion commands exist for governed operations.

Do not expose bot tokens, admin IDs, exchange credentials, user databases, or
operator-only runbooks in this public repository.

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

Run broader tests:

```bash
pytest tests/ -q
```

Run syntax validation:

```bash
python -m compileall -q quant quant_v2 scripts tools tests
```

## Configuration

Runtime configuration should be supplied through environment variables or
private deployment secret stores. Typical local variables include:

```text
TELEGRAM_TOKEN
ADMIN_ID
BOT_MASTER_KEY
BINANCE_API_KEY
BINANCE_API_SECRET
```

Never commit real values for these variables.

## Public Repository Guardrails

Before making or keeping this repository public:

- keep GitHub secret scanning and push protection enabled
- protect `main` with pull-request review and required checks
- enable Dependabot alerts and security updates
- enable CodeQL/code scanning for Python
- scan the full Git history before public release
- rotate any credential that has ever appeared in Git, logs, archives, or chat
- keep production hostnames, paths, IPs, runbooks, and current runtime state in
  private operations documentation outside this repository

See `SECURITY.md` and `docs/PUBLIC_RELEASE_CHECKLIST.md`.

## Files That Must Stay Out Of Git

Do not commit:

- `.env` or `.env.*`
- API keys, private keys, certificates, seed phrases, or passwords
- `quant_bot.db`, SQLite WAL/SHM files, Redis dumps, or user/account state
- `models/`, `state/`, `datasets/`, logs, backups, and deployment archives
- raw audit bundles or incident captures unless sanitized and explicitly
  approved

## Deployment

Deployment should be performed from a reviewed commit or signed release
artifact. Host-specific details belong in private operations runbooks rather
than this public README.

At minimum, deployment should verify:

- source commit SHA
- container image or build provenance
- environment-policy settings
- model manifest and registry pointer
- database/schema migration state
- no active hard-risk or kill-switch condition

## Disclaimer

This project is research and infrastructure software. It is not financial
advice, investment advice, or a guarantee of trading performance. Use at your
own risk.

## License

Research and internal operations use.
