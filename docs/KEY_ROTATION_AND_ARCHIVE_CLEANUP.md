# Key Rotation and Archive Cleanup Runbook

Use this runbook whenever a deployment archive, env file, key file, database, log, or diagnostic script may have been committed, uploaded, backed up, or shared.

## Immediate Containment

1. Stop using the exposed artifact.
2. Remove tracked archive/key/state files from the Git index while preserving local copies only if they are needed for investigation.
3. Run the release artifact scanner:

```bash
git ls-files | rg "\\.(tar\\.gz|tgz|zip|pem|key|db|sqlite)$|(^|/)\\.env"
python tools/security/scan_artifacts.py .
```

4. Delete stale local archives after any required evidence capture.

## Rotate Potentially Exposed Secrets

Rotate every credential that may appear in the exposed file or archive:

- EC2/operator key pairs.
- Telegram bot token.
- Exchange API keys and API secrets.
- Capital/Binance session credentials.
- `BOT_MASTER_KEY` if encrypted credential material may have been exposed with its key.
- Redis command-auth secret.
- Any third-party market-data API keys.

Do not reuse old keys after rotation. Record rotation completion in private operations inventory.

## Historical Git Exposure

If the artifact was pushed to any remote:

1. Treat it as permanently exposed.
2. Rotate affected keys before relying on history cleanup.
3. Decide whether to rewrite history with a tool such as `git filter-repo` or BFG Repo-Cleaner.
4. Coordinate force-push and reclone requirements with every collaborator and deployment host.

History cleanup reduces future accidental access, but it does not replace key rotation.

## Release Gate

A release candidate is blocked if any command below reports a tracked secret/state/archive path:

```bash
git ls-files | rg "\\.(tar\\.gz|tgz|zip|pem|key|db|sqlite)$|(^|/)\\.env"
```

A release candidate is blocked if the artifact scanner reports findings:

```bash
python tools/security/scan_artifacts.py <release-path>
```

Keep live trading disabled until the release artifact, deployment docs, model trust gate, and production readiness checklist all pass.
