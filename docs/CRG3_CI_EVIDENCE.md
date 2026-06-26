# CRG3 CI Evidence Procedure

`CRG3` is the final P0 blocker for live enablement. It cannot be verified from a workstation without Docker because the gate requires Compose config validation and a production image build.

## Required Evidence

Use a clean GitHub Actions run of `.github/workflows/production-readiness.yml`; the workflow supports `workflow_dispatch` for manual evidence capture.

The run must show:

- Workflow name: `Production Readiness`.
- Trigger: `workflow_dispatch` or `push` to `main`/`master`; pull request runs are useful probes but are not final CRG3 evidence.
- Run URL repository matches the configured GitHub `origin` fetch remote for this checkout.
- GitHub API run metadata `html_url` matches the submitted workflow run URL.
- GitHub Actions are pinned to immutable commit SHAs, not floating version tags.
- Workflow job env pins `BOT_V2_ALLOW_LIVE_EXECUTION=0`, so readiness checks cannot enable live execution.
- Readiness wrapper rejects `BOT_V2_ALLOW_LIVE_EXECUTION=1` or unknown values while checks are running.
- CI helper tools install with `python -m pip install --require-hashes -r requirements-ci.lock`.
- Command: `python tools/security/production_readiness.py --profile ci`.
- Checkout step uses `persist-credentials: false`.
- `full pytest suite` passed.
- `unsafe deployment docs grep` passed.
- `tracked-file secret scan` passed.
- `hashed CI tool lock dry-run` passed.
- `Linux CPython 3.11 CI tool wheel availability` passed.
- `docker compose -f docker-compose.yml config` passed.
- `docker compose -f docker-compose.prod.yml config` passed.
- `docker build --pull -t quant_bot:readiness .` passed.
- `Linux CPython 3.11 locked wheel availability` passed.
- `python -m pip_audit -r requirements.lock --progress-spinner off` reported no known vulnerabilities.
- `python tools/security/build_release.py --output build/release/quant-release.tar.gz` passed.
- `python tools/security/check_crg3_evidence_schema.py` passed.
- `build/security/sbom.cdx.json` uploaded as the `production-sbom` artifact.
- The workflow used the configured 45-minute job timeout and same-ref concurrency group with `cancel-in-progress: false`, so final evidence was not produced by competing overlapping readiness runs.

## After The Workflow Passes

Update `PRODUCTION_REFACTOR_ROADMAP.md`:

- Set `CRG3` status to `Verified YYYY-MM-DD`.
- Replace the proof text with the successful workflow run URL or run ID.
- Add an implementation report section with the run URL, commit SHA, SBOM artifact name, and residual risk.

Use the finalizer to make that update consistently:

```bash
python tools/security/finalize_crg3.py \
  --run-url https://github.com/<owner>/<repo>/actions/runs/<run-id> \
  --commit-sha <40-character-commit-sha> \
  --sbom-artifact production-sbom \
  --evidence-json build/security/crg3-evidence.json
```

The finalizer rejects run URLs from repositories that are not the configured GitHub `origin` fetch remote for this checkout, rejects any SBOM artifact name other than `production-sbom`, and verifies GitHub API metadata with bounded response reads before updating the roadmap:

- `--dry-run` verifies the CRG3 evidence and would-be roadmap update, then prints `CRG3 finalizer dry run passed; roadmap not modified.` without mutating `PRODUCTION_REFACTOR_ROADMAP.md`.
- `--evidence-json build/security/crg3-evidence.json` writes an optional machine-readable CRG3 evidence JSON with the ledger ID, verification date, repository, repository binding source, run URL, run ID, run attempt, commit SHA, clean local HEAD, runtime `requirements.lock` SHA-256 digest, SBOM artifact ZIP SHA-256 digest and downloaded byte length, accepted `sbom.cdx.json` SHA-256 digest and component count, validated artifact `id`, `size_in_bytes`, and canonical `archive_download_url`, workflow contract, and completed readiness wrapper check names.
- Finalizer-generated roadmap/report proof and machine-readable evidence JSON record the repository binding source, which is the GitHub `origin` fetch remote during normal CLI finalization.
- Machine-readable evidence must track the exact `tools/security/production_readiness.py` check names, plus workflow-only proof for the `production-sbom` artifact upload and downloaded CycloneDX SBOM artifact validation.
- Finalizer-generated roadmap/report proof must use the same exact completed check names as the machine-readable CRG3 evidence JSON.
- The finalizer rejects missing, duplicate, or unexpected check names before writing CRG3 evidence.
- `docs/CRG3_EVIDENCE_SCHEMA.json` defines the schema-backed proof shape; generated JSON includes a `schema` field pointing to that file and `schema_version: 9`.
- Schema version 9 includes repository binding source evidence, runtime lockfile digest evidence, non-zero bounded artifact size/download-size evidence, downloaded artifact digest evidence, downloaded SBOM digest evidence, and GitHub owner/repo slug patterns that reject whitespace/control characters and owner names ending in `-`.
- The finalizer validates generated evidence JSON against `docs/CRG3_EVIDENCE_SCHEMA.json` before writing the roadmap or evidence JSON file.
- Schema string patterns are validated with full-string matching, so newline-tainted or suffix-tainted evidence values are rejected before writing evidence.
- Direct finalizer `--run-url` and `--commit-sha` inputs are validated with full-string matching, so newline-tainted or suffix-tainted CLI values are rejected before GitHub API or artifact verification work starts.
- Direct finalizer and evidence JSON GitHub owner/repo values must use slug-safe characters and alphanumeric-ended owners; whitespace-tainted and trailing-hyphen-owner repository, run URL, and artifact URL values are rejected.
- The finalizer semantically validates that `runtime_lock.sha256` matches the current clean-checkout `requirements.lock` and `sbom_artifact_metadata.component_count` matches the lockfile-derived runtime component set before writing evidence.
- The finalizer semantically validates evidence identity: `repository` and `run_id` must match `run_url`, `local_head` must match `commit_sha`, and `sbom_artifact_metadata.download_url` plus `sbom_artifact_metadata.id` must belong to the same workflow repository.
- The finalizer semantically validates `verified_date` as a real ISO calendar date that is not in the future before writing evidence or updating the roadmap.
- Future-dated finalizer requests are rejected before GitHub API or artifact verification work starts.
- `--evidence-json` must use a `.json` suffix.
- `--evidence-json` must be named `crg3-evidence.json`.
- `--evidence-json` must stay under the roadmap directory, which is the repository root for the standard `PRODUCTION_REFACTOR_ROADMAP.md` workflow.
- `--evidence-json` must not be inside `.git`.
- `--evidence-json` must not be inside hidden directories such as `.github` or `.pytest_cache`.
- `--evidence-json` must not point at `docs/CRG3_EVIDENCE_SCHEMA.json`; the finalizer rejects paths that alias the evidence schema to prevent proof output from overwriting its own validation contract.
- `--evidence-json` must not be named `sbom.cdx.json`; the finalizer rejects proof output paths that could overwrite the workflow SBOM artifact.
- `--evidence-json` must not point at `PRODUCTION_REFACTOR_ROADMAP.md`; the finalizer rejects paths that alias the roadmap to prevent JSON proof output from overwriting ledger evidence.
- Dry runs do not write the roadmap or evidence JSON file.
- Finalizer writes use same-directory temporary files and atomic `os.replace`, so a failed replace does not leave a partially written roadmap or evidence JSON file.
- The finalizer writes `PRODUCTION_REFACTOR_ROADMAP.md` before optional evidence JSON, so a roadmap write failure cannot leave an orphaned CRG3 evidence JSON file.
- Local production readiness workflow file pins job-level `BOT_V2_ALLOW_LIVE_EXECUTION=0`.
- Finalizer validates workflow execution controls: manual dispatch, pull-request probes, `push` only for `main`/`master`, `permissions: contents: read`, same-ref concurrency with `cancel-in-progress: false`, `ubuntu-latest`, and 45-minute timeout.
- Finalizer validates the workflow structure, not comments or raw text alone: duplicate YAML mapping keys are rejected, the `readiness` job env must set `BOT_V2_ALLOW_LIVE_EXECUTION` to string `0`, and the wrapper step must run `python tools/security/production_readiness.py --profile ci` without enabling live execution.
- Finalizer validates the workflow setup/install chain: checkout and setup-python actions must be pinned, checkout must set `persist-credentials: false`, Python must be `3.11` with pip cache enabled, hashed CI tools must install before hashed runtime dependencies, and the readiness wrapper must run before SBOM upload.
- Finalizer validates the workflow SBOM upload structure: exactly one `Upload SBOM` step must use the pinned `actions/upload-artifact` action, run with `always()`, upload `build/security/sbom.cdx.json` as `production-sbom`, and fail with `if-no-files-found: error`.
- Workflow name is `Production Readiness`.
- Workflow path is `.github/workflows/production-readiness.yml`, optionally suffixed with `@main` or `@master` matching the release branch.
- Run event is `workflow_dispatch` or `push`.
- Run branch is `main` or `master`.
- Run status is `completed`.
- Run conclusion is `success`.
- Run `head_sha` matches `--commit-sha`.
- GitHub API run metadata `url` matches the submitted workflow run URL.
- GitHub API run metadata `id` matches the submitted workflow run ID.
- GitHub API run metadata `repository.full_name` matches the submitted workflow run repository.
- GitHub `run_attempt` is a positive integer and is recorded in final CRG3 evidence.
- GitHub run `artifacts_url` matches the submitted workflow run URL.
- Local checkout `HEAD` matches `--commit-sha`.
- Local working tree is clean before finalizer writes roadmap evidence.
- Exactly one `production-sbom` artifact exists on the run with explicit `expired: false`, valid non-zero `size_in_bytes` no larger than the bounded artifact download limit, and an artifact `id` that matches `archive_download_url`; the finalizer requests artifact pages with `per_page=100`, requires valid non-boolean integer GitHub `total_count` metadata and an explicit `artifacts` list, follows pagination for at most 10 pages while ensuring each page contains at most 100 object entries, rejects pages whose accumulated entries exceed `total_count`, rejects empty pages before the reported `total_count` is reached, verifies the artifact download URL belongs to the same GitHub repository and is canonical with no query string or fragment, downloads the artifact with a bounded response read, records a non-zero downloaded byte length, strips `Authorization` on cross-host redirects, and verifies it contains at most 64 zip members and exactly one bounded-size UTF-8 valid CycloneDX `sbom.cdx.json` whose unique `type`, `name`, `version`, and `purl` components match `requirements.lock` exactly; the evidence JSON records the `requirements.lock` SHA-256 digest, artifact ZIP SHA-256 digest, downloaded byte length, accepted `sbom.cdx.json` SHA-256 digest, and component count used for that comparison.
- Before writing, the finalizer validates the full updated P0 ledger in memory and refuses to mutate `PRODUCTION_REFACTOR_ROADMAP.md` if any P0 row would remain unverified or malformed.
- If `CRG3` is already verified, the finalizer refuses to replace existing evidence unless `--allow-reverify` is passed.

For private repositories, set `GITHUB_TOKEN` before running the finalizer so GitHub API verification can read the workflow run and artifacts.

Use `--dry-run` first when rehearsing finalization:

```bash
python tools/security/finalize_crg3.py \
  --run-url https://github.com/<owner>/<repo>/actions/runs/<run-id> \
  --commit-sha <40-character-commit-sha> \
  --sbom-artifact production-sbom \
  --dry-run
```

Then run strict live readiness from a clean Docker-capable checkout:

```bash
python tools/security/production_readiness.py --profile live
```

Strict live readiness must pass before setting `BOT_V2_ALLOW_LIVE_EXECUTION=1`.

## Failure Handling

If the workflow fails:

- Do not mark `CRG3` verified.
- Keep `BOT_V2_ALLOW_LIVE_EXECUTION=0`.
- Fix the failing Docker, Compose, audit, SBOM, tracked-file secret scan, artifact scan, or focused P0 test issue.
- Rerun the workflow on the fixed commit.
