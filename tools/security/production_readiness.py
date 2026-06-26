from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
FALSE_LIVE_ENV_VALUES = {"", "0", "false", "no", "off"}

FOCUSED_P0_TESTS = [
    "tests/quant_v2/test_retrain_pipeline.py",
    "tests/infra/test_docker_compose_services.py",
    "tests/infra/test_deployment_hardening_docs.py",
    "tests/infra/test_production_readiness.py",
    "tests/quant/test_model_selection.py",
    "tests/quant_v2/test_model_registry.py",
    "tests/quant_v2/test_signal_manager.py",
    "tests/quant_v2/test_full_ensemble.py",
    "tests/quant_v2/test_scheduled_retrain_candidates.py",
    "tests/infra/test_release_artifact_scan.py",
    "tests/quant/telebot/test_security_hygiene.py",
    "tests/quant/test_telebot_main_v2_handlers.py",
    "tests/quant_v2/test_binance_adapter.py",
    "tests/quant_v2/test_binance_client_phase4.py",
    "tests/quant_v2/test_bounded_liquidation.py",
    "tests/quant_v2/test_chase_logic.py",
    "tests/quant_v2/test_execution_service.py",
    "tests/quant_v2/test_live_readiness.py",
    "tests/quant_v2/test_execution_infra.py",
    "tests/quant_v2/test_reconciliation.py",
    "tests/quant_v2/test_day2_infra_patches.py",
    "tests/quant_v2/test_session_lock.py",
    "tests/quant_v2/test_watchdog_flatten_retry.py",
    "tests/quant_v2/test_credential_scrubbing.py",
]


@dataclass(frozen=True)
class Check:
    name: str
    command: list[str]
    requires_docker: bool = False
    requires_audit_tool: bool = False
    clean_paths: tuple[str, ...] = ()


def _clean_generated_path(raw_path: str) -> None:
    path = (REPO_ROOT / raw_path).resolve()
    if not path.is_relative_to(REPO_ROOT):
        raise ValueError(f"refusing to clean path outside repository: {raw_path}")
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def _run(check: Check, *, dry_run: bool) -> int:
    command = check.command
    print("+ " + " ".join(command), flush=True)
    if dry_run:
        return 0
    for path in check.clean_paths:
        _clean_generated_path(path)
    return subprocess.run(command, cwd=REPO_ROOT).returncode


def _git_status_clean() -> bool:
    result = subprocess.run(
        ["git", "status", "--short"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(result.stderr.strip(), file=sys.stderr)
        return False
    if result.stdout.strip():
        print("Working tree is not clean; live readiness must run from a reviewed clean checkout.")
        print(result.stdout)
        return False
    return True


def _live_execution_env_disabled() -> bool:
    raw_value = os.environ.get("BOT_V2_ALLOW_LIVE_EXECUTION")
    if raw_value is None:
        return True
    return raw_value.strip().lower() in FALSE_LIVE_ENV_VALUES


def _checks(
    *,
    allow_open_ledger_ids: list[str] | None = None,
    release_allow_dirty: bool = False,
) -> list[Check]:
    allow_open_ledger_ids = allow_open_ledger_ids or []
    roadmap_command = [
        sys.executable,
        "tools/security/check_roadmap.py",
        "PRODUCTION_REFACTOR_ROADMAP.md",
    ]
    for row_id in allow_open_ledger_ids:
        roadmap_command.extend(["--allow-open-id", row_id])
    release_command = [
        sys.executable,
        "tools/security/build_release.py",
        "--output",
        "build/release/quant-release.tar.gz",
    ]
    if release_allow_dirty:
        release_command.extend(["--allow-dirty", "--from-worktree"])

    return [
        Check(
            name="focused P0 regression suite",
            command=[sys.executable, "-m", "pytest", *FOCUSED_P0_TESTS, "-q"],
        ),
        Check(
            name="full pytest suite",
            command=[sys.executable, "-m", "pytest", "-q"],
        ),
        Check(
            name="unsafe deployment docs grep",
            command=[
                sys.executable,
                "-m",
                "pytest",
                "tests/infra/test_deployment_hardening_docs.py",
                "-q",
            ],
        ),
        Check(
            name="tracked-file secret scan",
            command=[sys.executable, "tools/security/scan_tracked_files.py"],
        ),
        Check(
            name="hashed CI tool lock dry-run",
            command=[
                sys.executable,
                "-m",
                "pip",
                "install",
                "--dry-run",
                "--require-hashes",
                "-r",
                "requirements-ci.lock",
            ],
        ),
        Check(
            name="Linux CPython 3.11 CI tool wheel availability",
            command=[
                sys.executable,
                "-m",
                "pip",
                "download",
                "--only-binary=:all:",
                "--dest",
                "build/pip-download-ci-py311",
                "--python-version",
                "3.11",
                "--implementation",
                "cp",
                "--abi",
                "cp311",
                "--platform",
                "manylinux_2_28_x86_64",
                "--platform",
                "manylinux2014_x86_64",
                "--require-hashes",
                "-r",
                "requirements-ci.lock",
            ],
            clean_paths=("build/pip-download-ci-py311",),
        ),
        Check(
            name="hashed dependency lock dry-run",
            command=[sys.executable, "-m", "pip", "install", "--dry-run", "--require-hashes", "-r", "requirements.lock"],
        ),
        Check(
            name="Linux CPython 3.11 locked wheel availability",
            command=[
                sys.executable,
                "-m",
                "pip",
                "download",
                "--only-binary=:all:",
                "--dest",
                "build/pip-download-py311",
                "--python-version",
                "3.11",
                "--implementation",
                "cp",
                "--abi",
                "cp311",
                "--platform",
                "manylinux_2_28_x86_64",
                "--platform",
                "manylinux2014_x86_64",
                "--require-hashes",
                "-r",
                "requirements.lock",
            ],
            clean_paths=("build/pip-download-py311",),
        ),
        Check(
            name="dependency vulnerability audit",
            command=[sys.executable, "-m", "pip_audit", "-r", "requirements.lock", "--progress-spinner", "off"],
            requires_audit_tool=True,
        ),
        Check(
            name="CycloneDX SBOM generation",
            command=[
                sys.executable,
                "tools/security/generate_sbom.py",
                "--requirements",
                "requirements.lock",
                "--output",
                "build/security/sbom.cdx.json",
            ],
        ),
        Check(
            name="CRG3 evidence schema validation",
            command=[sys.executable, "tools/security/check_crg3_evidence_schema.py"],
        ),
        Check(
            name="roadmap P0 ledger evidence",
            command=roadmap_command,
        ),
        Check(
            name="scanned release artifact packaging",
            command=release_command,
        ),
        Check(
            name="default compose config",
            command=["docker", "compose", "-f", "docker-compose.yml", "config"],
            requires_docker=True,
        ),
        Check(
            name="production compose config",
            command=["docker", "compose", "-f", "docker-compose.prod.yml", "config"],
            requires_docker=True,
        ),
        Check(
            name="production image build",
            command=["docker", "build", "--pull", "-t", "quant_bot:readiness", "."],
            requires_docker=True,
        ),
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run production deployment readiness checks.")
    parser.add_argument("--profile", choices=("live", "local", "ci"), default="live")
    parser.add_argument("--dry-run", action="store_true", help="Print checks without executing them.")
    parser.add_argument(
        "--skip-clean-check",
        action="store_true",
        help="Developer-only escape hatch for running checks in an implementation worktree.",
    )
    parser.add_argument(
        "--skip-docker",
        action="store_true",
        help="Developer-only escape hatch for environments without Docker. Never use for live enablement.",
    )
    parser.add_argument(
        "--skip-audit",
        action="store_true",
        help="Developer-only escape hatch for environments without pip-audit. Never use for live enablement.",
    )
    args = parser.parse_args(argv)

    strict_profile = args.profile in {"live", "ci"}

    if strict_profile and args.skip_docker:
        print("--skip-docker is not allowed with --profile live or --profile ci", file=sys.stderr)
        return 2
    if strict_profile and args.skip_audit:
        print("--skip-audit is not allowed with --profile live or --profile ci", file=sys.stderr)
        return 2
    if strict_profile and args.skip_clean_check:
        print("--skip-clean-check is not allowed with --profile live or --profile ci", file=sys.stderr)
        return 2

    if not _live_execution_env_disabled():
        print(
            "BOT_V2_ALLOW_LIVE_EXECUTION must be unset or false while running readiness checks.",
            file=sys.stderr,
        )
        return 2

    if not args.skip_clean_check and not args.dry_run and not _git_status_clean():
        return 1

    if not args.dry_run and not args.skip_docker and shutil.which("docker") is None:
        print("Docker CLI is required for production readiness checks.", file=sys.stderr)
        return 1

    failures: list[str] = []
    allow_open_ledger_ids = ["CRG3"] if args.profile in {"ci", "local"} else []
    release_allow_dirty = args.profile == "local" and args.skip_clean_check
    for check in _checks(
        allow_open_ledger_ids=allow_open_ledger_ids,
        release_allow_dirty=release_allow_dirty,
    ):
        if check.requires_docker and args.skip_docker:
            print(f"SKIP {check.name}: Docker checks skipped by developer flag.")
            continue
        if check.requires_audit_tool and args.skip_audit:
            print(f"SKIP {check.name}: dependency audit skipped by developer flag.")
            continue
        action = "PLAN" if args.dry_run else "RUN"
        print(f"{action} {check.name}")
        if _run(check, dry_run=args.dry_run):
            failures.append(check.name)

    if failures:
        print("FAILED production readiness checks:")
        for failure in failures:
            print(f"- {failure}")
        print("Deployment readiness failed; do not use this run as live deployment evidence.")
        return 1

    if args.skip_clean_check or args.skip_docker or args.skip_audit:
        print("Developer-only skips were used; this is not live deployment evidence.")
    if args.dry_run:
        print("Dry run only; no checks were executed and this is not deployment evidence.")
        return 0

    if args.profile == "local":
        print("Local readiness checks passed; this is not live deployment evidence.")
        return 0
    if args.profile == "ci":
        print(
            "CI readiness checks passed; finalize CRG3 from GitHub Actions evidence before live deployment."
        )
        return 0

    print("Production readiness checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
