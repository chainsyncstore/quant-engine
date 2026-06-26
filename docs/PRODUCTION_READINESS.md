# Production Readiness

Live trading must remain disabled until the production readiness wrapper passes from a reviewed clean checkout on a host with Docker available.

## Required Command

```bash
python tools/security/production_readiness.py --profile live
```

The GitHub Actions workflow uses `--profile ci` to let the `CRG3` workflow row bootstrap its
own evidence. The local developer probe also allows only `CRG3` to remain open, because that
row requires external GitHub Actions artifact evidence. Operators must use `--profile live`;
the live profile allows no open P0 ledger rows.
The CI workflow pins `BOT_V2_ALLOW_LIVE_EXECUTION=0` while running readiness checks, so
readiness checks cannot enable live execution.
The readiness wrapper also fails fast if `BOT_V2_ALLOW_LIVE_EXECUTION` is set to `1` or any
unknown value while checks are running.
The CI profile success message is `CI readiness checks passed; finalize CRG3 from GitHub Actions evidence before live deployment.`
Only the live profile prints final production readiness success.
Use `docs/CRG3_CI_EVIDENCE.md` to capture the final CI evidence and convert `CRG3` from in-progress to verified.
`--dry-run` only prints the planned checks with `PLAN` labels; it does not execute them and is never deployment evidence.
Any failed readiness check keeps live deployment blocked; failed runs are not deployment evidence.

The live profile requires:

- Clean Git working tree.
- Focused P0 regression suite.
- Full pytest suite.
- Unsafe deployment docs grep.
- Tracked-file secret scan.
- Hashed CI tool lock dry run.
- Linux CPython 3.11 CI tool wheel availability check.
- Hashed dependency lock dry run.
- Linux CPython 3.11 locked wheel availability check.
- Dependency vulnerability audit with `pip-audit`.
- CycloneDX SBOM generation at `build/security/sbom.cdx.json`.
- CRG3 evidence schema validation via `tools/security/check_crg3_evidence_schema.py`.
- CRG3 finalizer evidence that tracks these exact readiness wrapper check names.
- Default and production Compose config validation.
- Production Docker image build.
- Scanned release artifact packaging via `tools/security/build_release.py`.

## Developer Probe

During implementation on a dirty workstation without Docker, use the local probe only to check non-Docker gates:

```bash
python tools/security/production_readiness.py --profile local --skip-clean-check --skip-docker --skip-audit
```

When `--skip-clean-check` is used with the local profile, the release packaging check runs:

```bash
python tools/security/build_release.py --output build/release/quant-release.tar.gz --allow-dirty --from-worktree
```

That command builds a temporary-index archive of the proposed worktree state, so pending tracked additions and deletions are scanned without mutating the real git index. Output from a local probe is not live deployment evidence, even when the local profile completes without skips.
The local probe permits only the still-external `CRG3` row to remain open; every other P0 row must already be verified.

## Dependency Locking

Runtime dependencies are sourced from `requirements.in` and installed from `requirements.lock` with `pip --require-hashes`.
CI helper tools are sourced from `requirements-ci.in` and installed from `requirements-ci.lock` with `pip --require-hashes`.

When dependencies change:

```bash
python -m piptools compile requirements-ci.in --generate-hashes --output-file requirements-ci.lock
python -m pip install --dry-run --require-hashes -r requirements-ci.lock
python -m pip download --only-binary=:all: --dest build/pip-download-ci-py311 --python-version 3.11 --implementation cp --abi cp311 --platform manylinux_2_28_x86_64 --platform manylinux2014_x86_64 --require-hashes -r requirements-ci.lock
python -m piptools compile requirements.in --allow-unsafe --generate-hashes --pip-args "--only-binary=:all:" --output-file requirements.lock
python -m pip install --dry-run --require-hashes -r requirements.lock
python -m pip download --only-binary=:all: --dest build/pip-download-py311 --python-version 3.11 --implementation cp --abi cp311 --platform manylinux_2_28_x86_64 --platform manylinux2014_x86_64 --require-hashes -r requirements.lock
python -m pip_audit -r requirements.lock --progress-spinner off
python tools/security/generate_sbom.py --requirements requirements.lock --output build/security/sbom.cdx.json
python tools/security/check_crg3_evidence_schema.py
python tools/security/scan_tracked_files.py
python -m pytest tests/infra/test_docker_compose_services.py -q
```

The production runtime lock includes CPU-only Torch support. Keep Torch pinned to the reviewed CPU
wheel URL in `requirements.in`; do not replace it with the default PyPI Torch package, which can
pull a CUDA dependency stack on Linux. The `chronos-forecasting` package remains optional and must
not be added to the default production lock while its compatible `transformers<5` dependency is
blocked by active dependency-audit advisories.

## Live Enablement

Before setting `BOT_V2_ALLOW_LIVE_EXECUTION=1`, confirm:

- Every P0 ledger row in `PRODUCTION_REFACTOR_ROADMAP.md` is verified.
- `python tools/security/production_readiness.py --profile live` passes.
- CI `Production Readiness` workflow passes and uploads the SBOM artifact.
- `CRG3` is marked verified using the evidence procedure in `docs/CRG3_CI_EVIDENCE.md`.
- Release artifacts are built with `tools/security/build_release.py` and pass `tools/security/scan_artifacts.py`.
- `tools/security/scan_tracked_files.py` passes on the release commit.
- Redis is private and command messages are authenticated.
- Production env sets `BOT_V2_ENFORCE_GO_NO_GO=true` and `BOT_V2_LIVE_GO_NO_GO=true`.
- `BOT_V2_FORCE_ROLLBACK` is unset or explicitly cleared.
- Active model artifacts are trusted and loadable.
- Active model artifacts have valid manifests and checksums.
- Live mark freshness checks pass for every open position.
- SSH or session access is restricted to approved operators.
- Shutdown flatten and rollback drills have been rehearsed.
- Exposed keys or tokens from historical/local archives have been rotated.
