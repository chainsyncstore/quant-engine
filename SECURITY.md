# Security Policy

## Supported Versions

Security fixes are considered for the current `main` branch unless a maintainer
has explicitly published a supported release branch.

## Reporting A Vulnerability

Do not open a public issue for a suspected vulnerability.

Report privately through GitHub Security Advisories when available, or contact
the repository maintainers through the private channel used for this project.
Include:

- affected file or component
- impact and exploitability
- reproduction steps
- whether any credential, user data, or trading account may be exposed

## Secret Handling

This repository must not contain real credentials or production state.

Never commit:

- `.env` files
- Telegram bot tokens
- exchange API keys or secrets
- private keys or certificates
- user/account databases
- model artifacts containing private runtime data
- production backups, logs, or raw audit captures

If a secret is committed or shared accidentally:

1. Rotate the secret immediately.
2. Revoke old tokens or keys at the provider.
3. Check logs, artifacts, release bundles, and container images for copies.
4. Rewrite public Git history only after rotation and impact review.
5. Document the incident in private operations records.

## Public Release Requirements

Before publishing or accepting outside contributions:

- enable GitHub secret scanning and push protection
- protect `main` with pull requests and required checks
- enable Dependabot alerts and security updates
- enable CodeQL/code scanning
- run a full history secret scan
- keep host-specific operations documentation outside the public repository

See `docs/PUBLIC_RELEASE_CHECKLIST.md`.
