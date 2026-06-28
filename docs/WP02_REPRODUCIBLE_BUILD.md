# WP-02 Reproducible And Secure Build

## Status and boundary

This repository implementation addresses F-05, F-08, and F-18. It does not
authorize deployment, model activation, pause clearing, or production access.
Model artifact compatibility remains a fail-closed WP-10 deliverable; the WP-02
smoke records that gate as deferred and does not pretend to load an active model.

## Dependency contract

- Python: `3.11.9`, Linux `amd64`.
- Resolver: `uv 0.11.23 (3cdf50e09, 2026-06-19)`.
- Training and inference install the same `requirements/runtime.lock`.
- The build-only uv bootstrap is independently hashed in `requirements/build.lock`.
- Test tooling is isolated in `requirements/test.lock` and never enters runtime.
- Torch resolves through uv's CPU backend. There is no global PyTorch extra
  index and `unsafe-best-match` is prohibited.

Regenerate from the repository root:

```powershell
uv pip compile requirements/build.in --output-file requirements/build.lock --python-version 3.11 --python-platform x86_64-manylinux_2_28 --generate-hashes
uv pip compile requirements/runtime.in --output-file requirements/runtime.lock --python-version 3.11 --python-platform x86_64-manylinux_2_28 --torch-backend cpu --generate-hashes
uv pip compile requirements/test.in --output-file requirements/test.lock --python-version 3.11 --python-platform x86_64-manylinux_2_28 --generate-hashes
```

Review the diff and run the complete suite before accepting any lock update.

## Build contract

The Docker base is Python `3.11.9-slim-bookworm` pinned by manifest digest.
APT is pinned to Debian snapshot `20250201T000000Z`, with snapshot expiry
checking disabled only because the archive is intentionally immutable.
`libgomp1` remains pinned to `12.2.0-14+deb12u1`. The stages are:

1. `os-runtime`: exact operating-system runtime library.
2. `dependencies`: hashed uv installed into disposable `/opt/build-tools`; the
   runtime lock is installed separately into `/opt/runtime-venv`.
3. `runtime`: non-root application without tests, compilers, credentials, or evidence.
4. `test`: adds only the hashed test lock, runs all tests, and applies Ruff to
   infrastructure/build code before smoke checks. Repository-wide Ruff has known
   legacy debt and is not misrepresented as a passing release gate.
5. `release`: starts again from clean `runtime` and copies only a success marker
   from `test`, making bypass impossible without retaining test/build tooling.

Generate metadata first. Release mode requires a clean Git tree; development mode
records every dirty path and is never deployable:

```powershell
python tools/build_manifest.py --release --output .build/wp02-manifest.json
./scripts/build_release.ps1 -Image quant-bot:wp02
```

The manifest hashes only reviewed build context: `quant/`, `quant_v2/`, `tests/`,
Docker and all three Compose files, lock inputs, bootstrap code, and build/smoke tooling. Reports, docs,
archives, state, models, databases, and forensic evidence are excluded.

## Runtime and deployment

Compose does not build images and has no `latest` authority. Before rendering it:

```powershell
$env:QUANT_IMAGE = "registry.example/quant-bot@sha256:<64 lowercase hex>"
python scripts/verify_immutable_image.py
docker compose config
```

Application services run as UID/GID `10001`, with read-only roots, no added
capabilities, `no-new-privileges`, PID/CPU/memory bounds, and writable tmpfs only
for `/tmp` and cache. Host `models`, `state`, database, and signal-log mounts are
intentional writable exceptions and must be owned by `10001:10001` before startup.
Redis remains private and ACL-authenticated; its data volume and ACL `/tmp` are
the intentional writable exceptions.

Do not pass credentials as build arguments. `.env`, secrets, models, state,
reports, archives, and incident helpers are excluded from the build context.

## CI evidence and acceptance

The pinned workflow builds the `release` target once. That one local image is
smoked, inspected, inventoried with Syft `1.42.2`, and scanned with Trivy `0.71.2`
for vulnerabilities and repository secrets. Release checksums verify scanner
downloads. The retained bundle contains the build manifest, image inspection,
smoke output, CycloneDX SBOM, scan reports, and a digest manifest.

A local Docker image ID is not a registry digest. `exact_image_certified` remains
false until WP-15 pushes that same image, records its `@sha256` digest, signs the
attestation, and deploys exactly that reference. High or critical vulnerability
exceptions require a separate owner, rationale, compensating control, and expiry;
there is no permanent allowlist in this work package.

Syft `1.42.2` and Trivy `0.71.2` archives are checked against hardcoded reviewed
SHA-256 values; checksum files fetched beside the archives are not trusted.

Verify the reviewed Docker Hub tag manifests deterministically before a build:

```powershell
python tools/verify_external_artifacts.py --output evidence/external-images.json
```

Expected manifests are:

- `python:3.11.9-slim-bookworm` -> `sha256:8fb099199b9f2d70342674bd9dbccd3ed03a258f26bbd1d556822c6dfc60c317`
- `redis:7-alpine` -> `sha256:6ab0b6e7381779332f97b8ca76193e45b0756f38d4c0dcda72dbb3c32061ab99`

## Operator gates

1. Use a clean reviewed commit and regenerate both locks byte-for-byte.
2. Run CI and retain every evidence artifact.
3. Confirm zero secret findings and no unreviewed high/critical vulnerabilities.
4. Push the already tested image without rebuilding and capture its registry digest.
5. Verify host mount ownership and create the Redis secret out of band.
6. Keep auto-promotion disabled and trading paused.
7. Complete WP-10 artifact compatibility before any model load claim.
8. Complete WP-15 signing, exact-image attestation, and staged deployment gates.
