# Public Release Checklist

Use this checklist before making the repository public and before major public
releases.

## Repository Settings

- [ ] GitHub secret scanning is enabled.
- [ ] Push protection is enabled for supported secret types.
- [ ] `main` is protected by a branch protection rule or ruleset.
- [ ] Pull requests are required before merge.
- [ ] Required status checks are configured.
- [ ] Force pushes to `main` are blocked.
- [ ] Dependabot alerts are enabled.
- [ ] Dependabot security updates are enabled.
- [ ] CodeQL/code scanning is enabled for Python.
- [ ] Private vulnerability reporting or GitHub Security Advisories are enabled.

## Source Hygiene

- [ ] `.env`, keys, certificates, databases, model artifacts, logs, datasets,
      backups, and audit bundles are ignored.
- [ ] No production hostnames, IPs, SSH aliases, account IDs, or private paths
      appear in public docs.
- [ ] Operator-only runbooks are stored outside the public repository.
- [ ] Example configuration uses placeholders only.
- [ ] Public docs do not disclose current live runtime state.

## History And Artifact Scan

- [ ] Scan current `HEAD` for secrets.
- [ ] Scan full Git history for secrets.
- [ ] Scan release artifacts and Docker build context.
- [ ] Scan archived audit bundles before publishing sanitized evidence.
- [ ] Rotate any credential that ever appeared in Git, logs, archives, chat, or
      build artifacts.

## Trading Safety

- [ ] Public docs state that the project is research/infrastructure software,
      not financial advice.
- [ ] Production/live trading procedures are private.
- [ ] Model promotion requires validation evidence and explicit approval.
- [ ] Live execution remains blocked unless production-resume gates pass.

## Maintenance

- [ ] `SECURITY.md` is current.
- [ ] README is public-safe.
- [ ] CI checks are green.
- [ ] Dependency lock files are refreshed intentionally.
- [ ] Release notes avoid private operational details.
