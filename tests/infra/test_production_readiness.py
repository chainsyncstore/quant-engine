from __future__ import annotations

import hashlib
import io
import json
import re
import subprocess
import sys
import urllib.request
import zipfile
from datetime import date
from pathlib import Path

import pytest

from tools.security import production_readiness
from tools.security.production_readiness import FOCUSED_P0_TESTS
from tools.security.check_roadmap import parse_ledger, validate_p0_rows
from tools.security.finalize_crg3 import (
    CRG3_EVIDENCE_SCHEMA_VERSION,
    MAX_GITHUB_ARTIFACT_PAGES,
    MAX_GITHUB_ARTIFACTS_PER_PAGE,
    MAX_GITHUB_JSON_BYTES,
    MAX_SBOM_ARTIFACT_BYTES,
    MAX_SBOM_JSON_BYTES,
    MAX_SBOM_ZIP_MEMBERS,
    _CrossHostAuthStrippingRedirectHandler,
    _atomic_write_text,
    _allowed_github_repos,
    _expected_crg3_evidence_checks,
    _expected_runtime_sbom_components,
    _github_api_bytes,
    _github_api_json,
    _runtime_lock_metadata,
    _validate_crg3_evidence_record,
    _validate_local_workflow_live_disabled,
    finalize,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def _clear_live_execution_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("BOT_V2_ALLOW_LIVE_EXECUTION", raising=False)


def _workflow_text(
    *,
    workflow_name: str = "Production Readiness",
    push_branches: str = "- main\n      - master",
    permissions: str = "contents: read",
    concurrency_group: str = "production-readiness-${{ github.ref }}",
    cancel_in_progress: str = "false",
    runner: str = "ubuntu-latest",
    timeout_minutes: str = "45",
    checkout_uses: str = "actions/checkout@34e114876b0b11c390a56381ad16ebd13914f8d5",
    persist_credentials: str = "false",
    setup_uses: str = "actions/setup-python@a26af69be951a213d495a4c3e4e4022e16d87065",
    runtime_install: str = "python -m pip install --require-hashes -r requirements.lock",
    install_order: str = "ci-first",
    wrapper_env_override: str | None = None,
    wrapper_command: str = "python tools/security/production_readiness.py --profile ci",
    include_upload: bool = True,
    upload_name: str = "production-sbom",
    upload_path: str = "build/security/sbom.cdx.json",
    upload_missing: str = "error",
) -> str:
    ci_install = (
        "      - name: Install pinned CI tools\n"
        "        run: python -m pip install --require-hashes -r requirements-ci.lock\n"
    )
    runtime_install_step = (
        "      - name: Install locked runtime dependencies\n"
        f"        run: {runtime_install}\n"
    )
    install_steps = (
        ci_install + runtime_install_step
        if install_order == "ci-first"
        else runtime_install_step + ci_install
    )
    wrapper_env = (
        "        env:\n"
        f"          BOT_V2_ALLOW_LIVE_EXECUTION: \"{wrapper_env_override}\"\n"
        if wrapper_env_override is not None
        else ""
    )
    upload_step = (
        "      - name: Upload SBOM\n"
        "        uses: actions/upload-artifact@ea165f8d65b6e75b540449e92b4886f43607fa02\n"
        "        if: always()\n"
        "        with:\n"
        f"          name: {upload_name}\n"
        f"          path: {upload_path}\n"
        f"          if-no-files-found: {upload_missing}\n"
        if include_upload
        else ""
    )
    return (
        f"name: {workflow_name}\n"
        "\n"
        "on:\n"
        "  workflow_dispatch:\n"
        "  pull_request:\n"
        "  push:\n"
        "    branches:\n"
        f"      {push_branches}\n"
        "\n"
        "permissions:\n"
        f"  {permissions}\n"
        "\n"
        "concurrency:\n"
        f"  group: {concurrency_group}\n"
        f"  cancel-in-progress: {cancel_in_progress}\n"
        "\n"
        "jobs:\n"
        "  readiness:\n"
        f"    runs-on: {runner}\n"
        f"    timeout-minutes: {timeout_minutes}\n"
        "    env:\n"
        "      BOT_V2_ALLOW_LIVE_EXECUTION: \"0\"\n"
        "    steps:\n"
        "      - name: Check out repository\n"
        f"        uses: {checkout_uses}\n"
        "        with:\n"
        f"          persist-credentials: {persist_credentials}\n"
        "      - name: Set up Python\n"
        f"        uses: {setup_uses}\n"
        "        with:\n"
        "          python-version: \"3.11\"\n"
        "          cache: pip\n"
        f"{install_steps}"
        "      - name: Run production readiness wrapper\n"
        f"{wrapper_env}"
        f"        run: {wrapper_command}\n"
        f"{upload_step}"
    )


def _sbom_zip(*, bom_format: str = "CycloneDX", components: list[dict] | None = None) -> bytes:
    output = io.BytesIO()
    sbom = {
        "bomFormat": bom_format,
        "specVersion": "1.5",
        "components": components
        if components is not None
        else _expected_runtime_sbom_components(),
    }
    with zipfile.ZipFile(output, "w") as archive:
        archive.writestr("sbom.cdx.json", json.dumps(sbom))
    return output.getvalue()


def _default_sbom_artifact_metadata() -> dict[str, int | str]:
    artifact_payload = _sbom_zip()
    with zipfile.ZipFile(io.BytesIO(artifact_payload)) as archive:
        payload = archive.read("sbom.cdx.json")
        sbom = json.loads(payload.decode("utf-8"))
    return {
        "artifact_sha256": hashlib.sha256(artifact_payload).hexdigest(),
        "component_count": len(sbom["components"]),
        "download_size_in_bytes": len(artifact_payload),
        "sbom_sha256": hashlib.sha256(payload).hexdigest(),
    }


def _large_sbom_zip() -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        archive.writestr("sbom.cdx.json", b" " * (MAX_SBOM_JSON_BYTES + 1))
    return output.getvalue()


def _non_utf8_sbom_zip() -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        archive.writestr("sbom.cdx.json", b"\xff")
    return output.getvalue()


def _multiple_sbom_zip() -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        archive.writestr("sbom.cdx.json", json.dumps({"bomFormat": "CycloneDX"}))
        archive.writestr("nested/sbom.cdx.json", json.dumps({"bomFormat": "CycloneDX"}))
    return output.getvalue()


def _many_member_sbom_zip() -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        for index in range(MAX_SBOM_ZIP_MEMBERS + 1):
            archive.writestr(f"extra-{index}.txt", "")
        archive.writestr("sbom.cdx.json", json.dumps({"bomFormat": "CycloneDX"}))
    return output.getvalue()


def _artifact_fetcher(payload: bytes | None = None):
    def fetch(url: str) -> bytes:
        if url.endswith("/artifacts/42/zip"):
            return payload if payload is not None else _sbom_zip()
        raise AssertionError(f"unexpected GitHub artifact URL: {url}")

    return fetch


class _FakeGitHubResponse:
    def __init__(self, payload: bytes):
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self, size: int = -1) -> bytes:
        if size is None or size < 0:
            return self._payload
        return self._payload[:size]


def _github_fetcher(
    *,
    sha: str = "0123456789abcdef0123456789abcdef01234567",
    status: str = "completed",
    conclusion: str = "success",
    run_id=123456789,
    run_attempt=1,
    workflow_name: str = "Production Readiness",
    workflow_path: str = ".github/workflows/production-readiness.yml",
    event: str = "workflow_dispatch",
    head_branch: str = "main",
    artifact_name: str = "production-sbom",
    artifact_expired: bool = False,
    artifact_id: int = 42,
    artifact_size_in_bytes: int = 2048,
    archive_download_url: str = "https://api.github.com/repos/chainsyncstore/hypothesis-research-engine/actions/artifacts/42/zip",
    artifacts_url: str = "https://api.github.com/repos/chainsyncstore/hypothesis-research-engine/actions/runs/123456789/artifacts",
    repository_full_name="chainsyncstore/hypothesis-research-engine",
    html_url: str = "https://github.com/chainsyncstore/hypothesis-research-engine/actions/runs/123456789",
    api_url: str = "https://api.github.com/repos/chainsyncstore/hypothesis-research-engine/actions/runs/123456789",
):
    def fetch(url: str):
        if url.endswith("/actions/runs/123456789"):
            return {
                "url": api_url,
                "id": run_id,
                "repository": {"full_name": repository_full_name},
                "html_url": html_url,
                "name": workflow_name,
                "path": workflow_path,
                "event": event,
                "head_branch": head_branch,
                "status": status,
                "conclusion": conclusion,
                "run_attempt": run_attempt,
                "head_sha": sha,
                "artifacts_url": artifacts_url,
            }
        if url.endswith("/actions/runs/123456789/artifacts?per_page=100&page=1"):
            return {
                "total_count": 1,
                "artifacts": [
                    {
                        "name": artifact_name,
                        "expired": artifact_expired,
                        "id": artifact_id,
                        "size_in_bytes": artifact_size_in_bytes,
                        "archive_download_url": archive_download_url,
                    },
                ]
            }
        raise AssertionError(f"unexpected GitHub API URL: {url}")

    return fetch


def _assert_json_object_matches_schema(instance: dict, schema: dict) -> None:
    required = schema.get("required", [])
    assert set(required) <= set(instance)
    if schema.get("additionalProperties") is False:
        assert set(instance) <= set(schema.get("properties", {}))

    for key, value in instance.items():
        property_schema = schema["properties"][key]
        expected_type = property_schema.get("type")
        if expected_type == "object":
            assert isinstance(value, dict)
            _assert_json_object_matches_schema(value, property_schema)
        elif expected_type == "array":
            assert isinstance(value, list)
            assert len(value) >= property_schema.get("minItems", 0)
            if property_schema.get("uniqueItems") is True:
                assert len(value) == len(set(value))
            if property_schema.get("items", {}).get("type") == "string":
                assert all(isinstance(item, str) and item for item in value)
        elif expected_type == "integer":
            assert type(value) is int
            if "minimum" in property_schema:
                assert value >= property_schema["minimum"]
            if "maximum" in property_schema:
                assert value <= property_schema["maximum"]
        elif expected_type == "string":
            assert isinstance(value, str)
            if "const" in property_schema:
                assert value == property_schema["const"]
            if "pattern" in property_schema:
                assert re.fullmatch(property_schema["pattern"], value)
        if "const" in property_schema:
            assert value == property_schema["const"]


def test_production_readiness_local_dry_run_lists_deploy_blocking_checks() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "tools/security/production_readiness.py",
            "--profile",
            "local",
            "--skip-clean-check",
            "--skip-docker",
            "--dry-run",
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "PLAN focused P0 regression suite" in result.stdout
    assert "RUN focused P0 regression suite" not in result.stdout
    assert "full pytest suite" in result.stdout
    assert "unsafe deployment docs grep" in result.stdout
    assert "tracked-file secret scan" in result.stdout
    assert "hashed CI tool lock dry-run" in result.stdout
    assert "Linux CPython 3.11 CI tool wheel availability" in result.stdout
    assert "hashed dependency lock dry-run" in result.stdout
    assert "Linux CPython 3.11 locked wheel availability" in result.stdout
    assert "dependency vulnerability audit" in result.stdout
    assert "CycloneDX SBOM generation" in result.stdout
    assert "CRG3 evidence schema validation" in result.stdout
    assert "roadmap P0 ledger evidence" in result.stdout
    assert "scanned release artifact packaging" in result.stdout
    assert "Docker checks skipped by developer flag" in result.stdout
    assert "Developer-only skips were used" in result.stdout
    assert "Dry run only; no checks were executed and this is not deployment evidence." in result.stdout
    assert "Production readiness checks passed." not in result.stdout


def test_production_readiness_live_profile_cannot_skip_docker() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "tools/security/production_readiness.py",
            "--profile",
            "live",
            "--skip-docker",
            "--dry-run",
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "--skip-docker is not allowed with --profile live or --profile ci" in result.stderr


def test_production_readiness_live_profile_cannot_skip_audit() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "tools/security/production_readiness.py",
            "--profile",
            "live",
            "--skip-audit",
            "--dry-run",
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "--skip-audit is not allowed with --profile live or --profile ci" in result.stderr


def test_production_readiness_rejects_live_enabled_env_before_checks(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("BOT_V2_ALLOW_LIVE_EXECUTION", "1")

    assert production_readiness.main(["--profile", "ci", "--dry-run"]) == 2

    captured = capsys.readouterr()
    assert "BOT_V2_ALLOW_LIVE_EXECUTION must be unset or false" in captured.err
    assert "PLAN focused P0 regression suite" not in captured.out


def test_production_readiness_rejects_unknown_live_env_typo(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("BOT_V2_ALLOW_LIVE_EXECUTION", "maybe")

    assert production_readiness.main(["--profile", "live", "--dry-run"]) == 2

    captured = capsys.readouterr()
    assert "BOT_V2_ALLOW_LIVE_EXECUTION must be unset or false" in captured.err
    assert "PLAN focused P0 regression suite" not in captured.out


def test_production_readiness_live_dry_run_is_not_evidence() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "tools/security/production_readiness.py",
            "--profile",
            "live",
            "--dry-run",
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "PLAN focused P0 regression suite" in result.stdout
    assert "RUN focused P0 regression suite" not in result.stdout
    assert "Dry run only; no checks were executed and this is not deployment evidence." in result.stdout
    assert "Production readiness checks passed." not in result.stdout


def test_production_readiness_local_success_is_not_live_evidence(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    check = production_readiness.Check(
        name="stub readiness gate",
        command=[sys.executable, "-c", "pass"],
    )
    captured_kwargs = {}

    def fake_checks(**kwargs):
        captured_kwargs.update(kwargs)
        return [check]

    monkeypatch.setattr(production_readiness, "_checks", fake_checks)
    monkeypatch.setattr(production_readiness, "_git_status_clean", lambda: True)
    monkeypatch.setattr(production_readiness.shutil, "which", lambda name: "docker")
    monkeypatch.setattr(production_readiness, "_run", lambda check, dry_run: 0)

    assert production_readiness.main(["--profile", "local"]) == 0

    output = capsys.readouterr().out
    assert captured_kwargs["allow_open_ledger_ids"] == ["CRG3"]
    assert "RUN stub readiness gate" in output
    assert "Local readiness checks passed; this is not live deployment evidence." in output
    assert "Production readiness checks passed." not in output


def test_production_readiness_live_success_keeps_deployment_evidence_wording(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    check = production_readiness.Check(
        name="stub readiness gate",
        command=[sys.executable, "-c", "pass"],
    )

    def fake_checks(**kwargs):
        return [check]

    monkeypatch.setattr(production_readiness, "_checks", fake_checks)
    monkeypatch.setattr(production_readiness, "_git_status_clean", lambda: True)
    monkeypatch.setattr(production_readiness.shutil, "which", lambda name: "docker")
    monkeypatch.setattr(production_readiness, "_run", lambda check, dry_run: 0)

    assert production_readiness.main(["--profile", "live"]) == 0

    output = capsys.readouterr().out
    assert "RUN stub readiness gate" in output
    assert "Production readiness checks passed." in output
    assert "Local readiness checks passed" not in output


def test_production_readiness_ci_success_is_not_live_evidence(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    check = production_readiness.Check(
        name="stub readiness gate",
        command=[sys.executable, "-c", "pass"],
    )

    def fake_checks(**kwargs):
        return [check]

    monkeypatch.setattr(production_readiness, "_checks", fake_checks)
    monkeypatch.setattr(production_readiness, "_git_status_clean", lambda: True)
    monkeypatch.setattr(production_readiness.shutil, "which", lambda name: "docker")
    monkeypatch.setattr(production_readiness, "_run", lambda check, dry_run: 0)

    assert production_readiness.main(["--profile", "ci"]) == 0

    output = capsys.readouterr().out
    assert "RUN stub readiness gate" in output
    assert (
        "CI readiness checks passed; finalize CRG3 from GitHub Actions evidence before live deployment."
        in output
    )
    assert "Production readiness checks passed." not in output
    assert "Local readiness checks passed" not in output


def test_production_readiness_failure_is_not_deployment_evidence(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    check = production_readiness.Check(
        name="stub failing gate",
        command=[sys.executable, "-c", "raise SystemExit(1)"],
    )

    def fake_checks(**kwargs):
        return [check]

    monkeypatch.setattr(production_readiness, "_checks", fake_checks)
    monkeypatch.setattr(production_readiness, "_git_status_clean", lambda: True)
    monkeypatch.setattr(production_readiness.shutil, "which", lambda name: "docker")
    monkeypatch.setattr(production_readiness, "_run", lambda check, dry_run: 1)

    assert production_readiness.main(["--profile", "live"]) == 1

    output = capsys.readouterr().out
    assert "RUN stub failing gate" in output
    assert "FAILED production readiness checks:" in output
    assert "- stub failing gate" in output
    assert "Deployment readiness failed; do not use this run as live deployment evidence." in output
    assert "Production readiness checks passed." not in output


def test_production_readiness_live_profile_cannot_skip_clean_check() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "tools/security/production_readiness.py",
            "--profile",
            "live",
            "--skip-clean-check",
            "--dry-run",
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "--skip-clean-check is not allowed with --profile live or --profile ci" in result.stderr


def test_production_readiness_ci_profile_cannot_skip_clean_check() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "tools/security/production_readiness.py",
            "--profile",
            "ci",
            "--skip-clean-check",
            "--dry-run",
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "--skip-clean-check is not allowed with --profile live or --profile ci" in result.stderr


def test_sbom_generator_writes_cyclonedx_components(tmp_path: Path) -> None:
    lock_versions = {
        component["name"]: component["version"]
        for component in _expected_runtime_sbom_components()
    }

    output = tmp_path / "sbom.cdx.json"
    result = subprocess.run(
        [
            sys.executable,
            "tools/security/generate_sbom.py",
            "--requirements",
            "requirements.lock",
            "--output",
            str(output),
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "wrote" in result.stdout
    sbom = json.loads(output.read_text(encoding="utf-8"))
    assert sbom["bomFormat"] == "CycloneDX"
    assert sbom["specVersion"] == "1.5"
    components = {component["name"]: component for component in sbom["components"]}
    assert components["requests"]["version"] == lock_versions["requests"]
    assert components["redis"]["purl"] == f"pkg:pypi/redis@{lock_versions['redis']}"
    assert components["torch"]["version"] == "2.12.0+cpu"
    assert components["torch"]["purl"] == "pkg:pypi/torch@2.12.0+cpu"


def test_production_readiness_workflow_runs_live_gates_and_uploads_sbom() -> None:
    workflow = REPO_ROOT / ".github" / "workflows" / "production-readiness.yml"
    text = workflow.read_text(encoding="utf-8")

    assert "python -m pip install --require-hashes -r requirements-ci.lock" in text
    assert "python -m pip install --require-hashes -r requirements.lock" in text
    assert "pip-audit==2.9.0" not in text
    assert "python tools/security/production_readiness.py --profile ci" in text
    assert "actions/checkout@34e114876b0b11c390a56381ad16ebd13914f8d5" in text
    assert "actions/setup-python@a26af69be951a213d495a4c3e4e4022e16d87065" in text
    assert "actions/upload-artifact@ea165f8d65b6e75b540449e92b4886f43607fa02" in text
    assert not re.search(r"uses:\s+actions/[^@\s]+@v\d+\b", text)
    assert "build/security/sbom.cdx.json" in text
    assert "workflow_dispatch:" in text
    assert "permissions:" in text
    assert "contents: read" in text
    assert "timeout-minutes: 45" in text
    assert 'BOT_V2_ALLOW_LIVE_EXECUTION: "0"' in text
    assert "concurrency:" in text
    assert "group: production-readiness-${{ github.ref }}" in text
    assert "cancel-in-progress: false" in text
    assert "persist-credentials: false" in text
    assert text.index('BOT_V2_ALLOW_LIVE_EXECUTION: "0"') < text.index(
        "Run production readiness wrapper"
    )
    assert text.index("Install pinned CI tools") < text.index(
        "Install locked runtime dependencies"
    )
    assert text.index("Install locked runtime dependencies") < text.index(
        "Run production readiness wrapper"
    )

    wrapper = (REPO_ROOT / "tools" / "security" / "production_readiness.py").read_text(
        encoding="utf-8"
    )
    assert "pip_audit" in wrapper
    assert 'name="full pytest suite"' in wrapper
    assert '"pytest", "-q"' in wrapper
    assert 'name="unsafe deployment docs grep"' in wrapper
    assert "tests/infra/test_deployment_hardening_docs.py" in wrapper
    assert 'name="tracked-file secret scan"' in wrapper
    assert "tools/security/scan_tracked_files.py" in wrapper
    assert 'name="hashed CI tool lock dry-run"' in wrapper
    assert "requirements-ci.lock" in wrapper
    assert 'name="Linux CPython 3.11 CI tool wheel availability"' in wrapper
    assert "build/pip-download-ci-py311" in wrapper
    assert "clean_paths=(\"build/pip-download-py311\",)" in wrapper
    assert "clean_paths=(\"build/pip-download-ci-py311\",)" in wrapper
    assert "manylinux_2_28_x86_64" in wrapper
    assert "manylinux2014_x86_64" in wrapper
    assert '"--python-version"' in wrapper
    assert "tools/security/generate_sbom.py" in wrapper
    assert "tools/security/check_roadmap.py" in wrapper
    assert '"docker", "compose"' in wrapper
    assert '"docker", "build"' in wrapper
    assert "tools/security/build_release.py" in wrapper


def test_production_readiness_cleans_py311_download_cache_before_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = REPO_ROOT / "build" / "pip-download-py311"
    cache.mkdir(parents=True, exist_ok=True)
    stale = cache / "stale.whl"
    stale.write_text("stale\n", encoding="utf-8")

    calls: list[list[str]] = []

    def fake_run(command, **kwargs):
        calls.append(list(command))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(production_readiness.subprocess, "run", fake_run)

    check = production_readiness.Check(
        name="test",
        command=[sys.executable, "-c", "pass"],
        clean_paths=("build/pip-download-py311",),
    )

    assert production_readiness._run(check, dry_run=False) == 0
    assert not stale.exists()
    assert calls == [[sys.executable, "-c", "pass"]]


def test_production_readiness_local_and_ci_allow_only_crg3_open_while_live_is_strict() -> None:
    ci_checks = production_readiness._checks(allow_open_ledger_ids=["CRG3"])
    local_checks = production_readiness._checks(allow_open_ledger_ids=["CRG3"])
    live_checks = production_readiness._checks()

    ci_roadmap = next(check for check in ci_checks if check.name == "roadmap P0 ledger evidence")
    local_roadmap = next(
        check for check in local_checks if check.name == "roadmap P0 ledger evidence"
    )
    live_roadmap = next(
        check for check in live_checks if check.name == "roadmap P0 ledger evidence"
    )

    assert "--allow-open-id" in ci_roadmap.command
    assert "CRG3" in ci_roadmap.command
    assert "--allow-open-id" in local_roadmap.command
    assert "CRG3" in local_roadmap.command
    assert "--allow-open-id" not in live_roadmap.command
    assert "CRG3" not in live_roadmap.command


def test_production_readiness_includes_crg3_evidence_schema_validation() -> None:
    checks = production_readiness._checks()
    schema_check = next(check for check in checks if check.name == "CRG3 evidence schema validation")

    assert schema_check.command == [
        sys.executable,
        "tools/security/check_crg3_evidence_schema.py",
    ]


def test_crg3_evidence_schema_checker_passes() -> None:
    result = subprocess.run(
        [sys.executable, "tools/security/check_crg3_evidence_schema.py"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "CRG3 evidence schema validation passed." in result.stdout


def test_crg3_evidence_schema_version_matches_finalizer_constant() -> None:
    schema = json.loads((REPO_ROOT / "docs" / "CRG3_EVIDENCE_SCHEMA.json").read_text(
        encoding="utf-8"
    ))

    assert schema["properties"]["schema_version"]["const"] == CRG3_EVIDENCE_SCHEMA_VERSION
    assert "[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?" in (
        schema["properties"]["repository"]["pattern"]
    )
    assert "[A-Za-z0-9._-]+" in schema["properties"]["run_url"]["pattern"]
    assert "[A-Za-z0-9._-]+" in (
        schema["properties"]["sbom_artifact_metadata"]["properties"]["download_url"]["pattern"]
    )
    assert schema["properties"]["repository_binding_source"]["pattern"] == (
        "^(origin fetch remote|explicit allowed repository set)$"
    )
    sbom_metadata_schema = schema["properties"]["sbom_artifact_metadata"]["properties"]
    assert sbom_metadata_schema["size_in_bytes"]["maximum"] == MAX_SBOM_ARTIFACT_BYTES
    assert sbom_metadata_schema["download_size_in_bytes"]["maximum"] == MAX_SBOM_ARTIFACT_BYTES


def test_production_readiness_only_local_skip_clean_allows_dirty_release_archive() -> None:
    local_dirty_checks = production_readiness._checks(release_allow_dirty=True)
    default_checks = production_readiness._checks()

    local_release = next(
        check for check in local_dirty_checks if check.name == "scanned release artifact packaging"
    )
    default_release = next(
        check for check in default_checks if check.name == "scanned release artifact packaging"
    )

    assert "--allow-dirty" in local_release.command
    assert "--from-worktree" in local_release.command
    assert "--allow-dirty" not in default_release.command
    assert "--from-worktree" not in default_release.command


def test_release_builder_cli_runs_when_invoked_by_script_path() -> None:
    result = subprocess.run(
        [sys.executable, "tools/security/build_release.py", "--help"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Build a scanned release artifact" in result.stdout


def test_crg3_evidence_runbook_requires_docker_workflow_proof() -> None:
    text = (REPO_ROOT / "docs" / "CRG3_CI_EVIDENCE.md").read_text(encoding="utf-8")

    required_phrases = [
        "Production Readiness",
        "workflow_dispatch",
        "pull request runs are useful probes but are not final CRG3 evidence",
        "Run URL repository matches the configured GitHub `origin` fetch remote",
        "GitHub Actions are pinned to immutable commit SHAs",
        "Workflow job env pins `BOT_V2_ALLOW_LIVE_EXECUTION=0`",
        "readiness checks cannot enable live execution",
        "Readiness wrapper rejects `BOT_V2_ALLOW_LIVE_EXECUTION=1`",
        "unknown values while checks are running",
        "python -m pip install --require-hashes -r requirements-ci.lock",
        "persist-credentials: false",
        "python tools/security/production_readiness.py --profile ci",
        "docker compose -f docker-compose.yml config",
        "docker compose -f docker-compose.prod.yml config",
        "docker build --pull -t quant_bot:readiness .",
        "full pytest suite",
        "unsafe deployment docs grep",
        "tracked-file secret scan",
        "hashed CI tool lock dry-run",
        "Linux CPython 3.11 CI tool wheel availability",
        "python -m pip_audit -r requirements.lock --progress-spinner off",
        "Linux CPython 3.11 locked wheel availability",
        "python tools/security/build_release.py --output build/release/quant-release.tar.gz",
        "python tools/security/check_crg3_evidence_schema.py",
        "Exactly one `production-sbom` artifact",
        "rejects run URLs from repositories that are not the configured GitHub `origin` fetch remote",
        "rejects any SBOM artifact name other than `production-sbom`",
        "verifies GitHub API metadata",
        "bounded response reads",
        "`--dry-run` verifies the CRG3 evidence and would-be roadmap update",
        "CRG3 finalizer dry run passed; roadmap not modified.",
        "Local production readiness workflow file pins job-level `BOT_V2_ALLOW_LIVE_EXECUTION=0`",
        "Finalizer validates workflow execution controls",
        "`push` only for `main`/`master`",
        "`permissions: contents: read`",
        "same-ref concurrency with `cancel-in-progress: false`",
        "`ubuntu-latest`, and 45-minute timeout",
        "Finalizer validates the workflow structure, not comments or raw text alone",
        "duplicate YAML mapping keys are rejected",
        "the `readiness` job env must set `BOT_V2_ALLOW_LIVE_EXECUTION` to string `0`",
        "wrapper step must run `python tools/security/production_readiness.py --profile ci` without enabling live execution",
        "Finalizer validates the workflow setup/install chain",
        "checkout and setup-python actions must be pinned",
        "checkout must set `persist-credentials: false`",
        "hashed CI tools must install before hashed runtime dependencies",
        "readiness wrapper must run before SBOM upload",
        "Finalizer validates the workflow SBOM upload structure",
        "exactly one `Upload SBOM` step",
        "upload `build/security/sbom.cdx.json` as `production-sbom`",
        "if-no-files-found: error",
        "GitHub API run metadata `url` matches the submitted workflow run URL",
        "GitHub API run metadata `id` matches the submitted workflow run ID",
        "GitHub API run metadata `repository.full_name` matches the submitted workflow run repository",
        "GitHub API run metadata `html_url` matches the submitted workflow run URL",
        "GitHub run `artifacts_url` matches the submitted workflow run URL",
        "Workflow path is `.github/workflows/production-readiness.yml`, optionally suffixed with `@main` or `@master` matching the release branch",
        "Run event is `workflow_dispatch` or `push`",
        "Run branch is `main` or `master`",
        "Run conclusion is `success`",
        "Run `head_sha` matches `--commit-sha`",
        "GitHub `run_attempt` is a positive integer",
        "recorded in final CRG3 evidence",
        "optional machine-readable CRG3 evidence JSON",
        "repository binding source",
        "record the repository binding source",
        "GitHub `origin` fetch remote during normal CLI finalization",
        "runtime `requirements.lock` SHA-256 digest",
        "SBOM artifact ZIP SHA-256 digest and downloaded byte length",
        "accepted `sbom.cdx.json` SHA-256 digest and component count",
        "completed readiness wrapper check names",
        "must track the exact `tools/security/production_readiness.py` check names",
        "workflow-only proof for the `production-sbom` artifact upload",
        "roadmap/report proof must use the same exact completed check names",
        "rejects missing, duplicate, or unexpected check names",
        "`docs/CRG3_EVIDENCE_SCHEMA.json`",
        "schema-backed proof shape",
        "schema_version: 9",
        "Schema version 9 includes repository binding source evidence, runtime lockfile digest evidence, non-zero bounded artifact size/download-size evidence, downloaded artifact digest evidence, downloaded SBOM digest evidence, and GitHub owner/repo slug patterns that reject whitespace/control characters and owner names ending in `-`",
        "validates generated evidence JSON against `docs/CRG3_EVIDENCE_SCHEMA.json`",
        "before writing the roadmap or evidence JSON file",
        "Schema string patterns are validated with full-string matching",
        "newline-tainted or suffix-tainted evidence values are rejected",
        "Direct finalizer `--run-url` and `--commit-sha` inputs are validated with full-string matching",
        "newline-tainted or suffix-tainted CLI values are rejected before GitHub API or artifact verification work starts",
        "Direct finalizer and evidence JSON GitHub owner/repo values must use slug-safe characters and alphanumeric-ended owners",
        "whitespace-tainted and trailing-hyphen-owner repository, run URL, and artifact URL values are rejected",
        "validated artifact `id`, `size_in_bytes`, and canonical `archive_download_url`",
        "valid non-zero `size_in_bytes`",
        "records a non-zero downloaded byte length",
        "`--evidence-json build/security/crg3-evidence.json`",
        "`--evidence-json` must use a `.json` suffix",
        "`--evidence-json` must be named `crg3-evidence.json`",
        "`--evidence-json` must stay under the roadmap directory",
        "repository root for the standard `PRODUCTION_REFACTOR_ROADMAP.md` workflow",
        "`--evidence-json` must not be inside `.git`",
        "`--evidence-json` must not be inside hidden directories",
        "`.github` or `.pytest_cache`",
        "`--evidence-json` must not point at `docs/CRG3_EVIDENCE_SCHEMA.json`",
        "prevent proof output from overwriting its own validation contract",
        "`--evidence-json` must not be named `sbom.cdx.json`",
        "could overwrite the workflow SBOM artifact",
        "`--evidence-json` must not point at `PRODUCTION_REFACTOR_ROADMAP.md`",
        "rejects paths that alias the roadmap",
        "Dry runs do not write the roadmap or evidence JSON file",
        "same-directory temporary files and atomic `os.replace`",
        "does not leave a partially written roadmap or evidence JSON file",
        "writes `PRODUCTION_REFACTOR_ROADMAP.md` before optional evidence JSON",
        "roadmap write failure cannot leave an orphaned CRG3 evidence JSON file",
        "Local checkout `HEAD` matches `--commit-sha`",
        "Local working tree is clean",
        "explicit `expired: false`",
        "valid non-zero `size_in_bytes`",
        "valid non-boolean integer GitHub `total_count` metadata",
        "an explicit `artifacts` list",
        "artifact `id` that matches `archive_download_url`",
        "same GitHub repository",
        "canonical with no query string or fragment",
        "strips `Authorization` on cross-host redirects",
        f"{MAX_GITHUB_ARTIFACT_PAGES} pages",
        f"page contains at most {MAX_GITHUB_ARTIFACTS_PER_PAGE} object entries",
        "rejects pages whose accumulated entries exceed `total_count`",
        "rejects empty pages before the reported `total_count` is reached",
        "bounded response read",
        "exactly one",
        f"at most {MAX_SBOM_ZIP_MEMBERS} zip members",
        "UTF-8 valid CycloneDX",
        "per_page=100",
        "unique `type`, `name`, `version`, and `purl` components match `requirements.lock` exactly",
        "records the `requirements.lock` SHA-256 digest",
        "artifact ZIP SHA-256 digest",
        "downloaded byte length",
        "accepted `sbom.cdx.json` SHA-256 digest",
        "`runtime_lock.sha256` matches the current clean-checkout `requirements.lock`",
        "`sbom_artifact_metadata.component_count` matches the lockfile-derived runtime component set",
        "semantically validates evidence identity",
        "`repository` and `run_id` must match `run_url`",
        "`local_head` must match `commit_sha`",
        "`sbom_artifact_metadata.download_url` plus `sbom_artifact_metadata.id` must belong to the same workflow repository",
        "`verified_date` as a real ISO calendar date that is not in the future before writing evidence or updating the roadmap",
        "Future-dated finalizer requests are rejected before GitHub API or artifact verification work starts",
        "validates the full updated P0 ledger in memory",
        "refuses to mutate `PRODUCTION_REFACTOR_ROADMAP.md`",
        "--allow-reverify",
        "GITHUB_TOKEN",
        "python tools/security/finalize_crg3.py",
        "Set `CRG3` status to `Verified YYYY-MM-DD`",
        "python tools/security/production_readiness.py --profile live",
        "BOT_V2_ALLOW_LIVE_EXECUTION=1",
    ]

    for phrase in required_phrases:
        assert phrase in text

    readiness = (REPO_ROOT / "docs" / "PRODUCTION_READINESS.md").read_text(encoding="utf-8")
    assert "docs/CRG3_CI_EVIDENCE.md" in readiness
    assert "pins `BOT_V2_ALLOW_LIVE_EXECUTION=0`" in readiness
    assert "readiness checks cannot enable live execution" in readiness
    assert "fails fast if `BOT_V2_ALLOW_LIVE_EXECUTION` is set to `1`" in readiness
    assert "unknown value while checks are running" in readiness
    assert "CI readiness checks passed; finalize CRG3 from GitHub Actions evidence" in readiness
    assert "The local developer probe also allows only `CRG3` to remain open" in readiness
    assert "The local probe permits only the still-external `CRG3` row to remain open" in readiness
    assert "`--dry-run` only prints the planned checks with `PLAN` labels" in readiness
    assert "is never deployment evidence" in readiness
    assert "Any failed readiness check keeps live deployment blocked" in readiness
    assert "failed runs are not deployment evidence" in readiness
    assert "local probe is not live deployment evidence, even when the local profile completes without skips" in readiness
    assert "--allow-dirty --from-worktree" in readiness
    assert "temporary-index archive of the proposed worktree state" in readiness
    assert "without mutating the real git index" in readiness
    assert "CRG3 evidence schema validation via `tools/security/check_crg3_evidence_schema.py`" in readiness
    assert "CRG3 finalizer evidence that tracks these exact readiness wrapper check names" in readiness


def test_implementation_report_template_requires_evidence_and_rollback_sections() -> None:
    text = (REPO_ROOT / "docs" / "IMPLEMENTATION_REPORT_TEMPLATE.md").read_text(
        encoding="utf-8"
    )

    required_sections = [
        "Ledger IDs changed",
        "Files changed",
        "Safety invariants proven",
        "Tests added or inverted",
        "Verification commands",
        "Results",
        "Rollback plan",
        "Residual risk",
        "Live-block status",
    ]
    for section in required_sections:
        assert section in text

    assert "Verified YYYY-MM-DD" in text
    assert "run URL or run ID" in text


def test_deferred_hardening_keeps_p0_guardrails_non_deferred() -> None:
    text = (REPO_ROOT / "docs" / "DEFERRED_HARDENING.md").read_text(encoding="utf-8")

    non_deferrable = [
        "Live execution disabled by default",
        "No normal post-only entry fallback to market orders",
        "Missing or stale marks",
        "Durable WAL/idempotency",
        "Redis command authentication",
        "Credential redaction",
        "Trusted model manifests",
        "Reproducible locked dependencies",
        "Production readiness evidence",
    ]
    for phrase in non_deferrable:
        assert phrase in text

    allowed_deferrals = [
        "Redis TLS for same-host Docker network",
        "External artifact signing service",
        "Docker secrets migration",
        "Full dependency modernization",
    ]
    for phrase in allowed_deferrals:
        assert phrase in text

    assert "Any new deferral must answer" in text


def test_finalize_crg3_rejects_malformed_workflow_evidence(tmp_path: Path) -> None:
    roadmap = tmp_path / "roadmap.md"
    roadmap.write_text(
        "| ID | Pass | Priority | Status | Work item | Files | Proof required |\n"
        "| --- | --- | --- | --- | --- | --- | --- |\n"
        "| CRG3 | CI, Regression, and Production Readiness Gate | P0 | In progress | Add workflow | `.github/workflows/production-readiness.yml` | Awaiting workflow evidence. |\n",
        encoding="utf-8",
    )

    try:
        finalize(
            run_url="https://example.com/not-github",
            commit_sha="abc",
            sbom_artifact="production-sbom",
            verified_date=date(2026, 6, 4),
            roadmap_path=roadmap,
        )
    except ValueError as exc:
        assert "run URL" in str(exc)
    else:
        raise AssertionError("finalize accepted malformed workflow evidence")


def test_finalize_crg3_rejects_wrong_sbom_artifact(tmp_path: Path) -> None:
    roadmap = tmp_path / "roadmap.md"
    roadmap.write_text(
        "| ID | Pass | Priority | Status | Work item | Files | Proof required |\n"
        "| --- | --- | --- | --- | --- | --- | --- |\n"
        "| CRG3 | CI, Regression, and Production Readiness Gate | P0 | In progress | Add workflow | `.github/workflows/production-readiness.yml` | Awaiting workflow evidence. |\n",
        encoding="utf-8",
    )

    try:
        finalize(
            run_url="https://github.com/chainsyncstore/hypothesis-research-engine/actions/runs/123456789",
            commit_sha="0123456789abcdef0123456789abcdef01234567",
            sbom_artifact="some-other-artifact",
            verified_date=date(2026, 6, 4),
            roadmap_path=roadmap,
        )
    except ValueError as exc:
        assert "production-sbom" in str(exc)
    else:
        raise AssertionError("finalize accepted the wrong SBOM artifact name")


def test_finalize_crg3_github_json_fetch_is_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_urlopen(request, timeout):
        return _FakeGitHubResponse(b" " * (MAX_GITHUB_JSON_BYTES + 1))

    monkeypatch.setattr("tools.security.finalize_crg3._github_urlopen", fake_urlopen)

    try:
        _github_api_json("https://api.github.com/repos/chainsyncstore/hypothesis-research-engine/actions/runs/123")
    except ValueError as exc:
        assert "too large" in str(exc)
    else:
        raise AssertionError("GitHub JSON fetch accepted an oversized API response")


def test_finalize_crg3_github_json_fetch_rejects_non_utf8(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_urlopen(request, timeout):
        return _FakeGitHubResponse(b"\xff")

    monkeypatch.setattr("tools.security.finalize_crg3._github_urlopen", fake_urlopen)

    try:
        _github_api_json("https://api.github.com/repos/chainsyncstore/hypothesis-research-engine/actions/runs/123")
    except ValueError as exc:
        assert "UTF-8" in str(exc)
    else:
        raise AssertionError("GitHub JSON fetch accepted a non-UTF-8 API response")


def test_finalize_crg3_github_artifact_fetch_is_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_urlopen(request, timeout):
        return _FakeGitHubResponse(b"x" * (MAX_SBOM_ARTIFACT_BYTES + 1))

    monkeypatch.setattr("tools.security.finalize_crg3._github_urlopen", fake_urlopen)

    try:
        _github_api_bytes(
            "https://api.github.com/repos/chainsyncstore/hypothesis-research-engine/actions/artifacts/42/zip"
        )
    except ValueError as exc:
        assert "too large" in str(exc)
    else:
        raise AssertionError("GitHub artifact fetch accepted an oversized API response")


def test_finalize_crg3_redirect_drops_authorization_on_cross_host() -> None:
    handler = _CrossHostAuthStrippingRedirectHandler()
    request = urllib.request.Request(
        "https://api.github.com/repos/chainsyncstore/hypothesis-research-engine/actions/artifacts/42/zip",
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": "Bearer secret-token",
        },
    )

    redirected = handler.redirect_request(
        request,
        None,
        302,
        "Found",
        {},
        "https://objects.githubusercontent.com/github-production-release-asset/test.zip",
    )

    assert redirected is not None
    assert redirected.full_url.startswith("https://objects.githubusercontent.com/")
    assert "Authorization" not in redirected.headers
    assert redirected.headers["Accept"] == "application/vnd.github+json"


def test_finalize_crg3_redirect_preserves_authorization_on_same_host() -> None:
    handler = _CrossHostAuthStrippingRedirectHandler()
    request = urllib.request.Request(
        "https://api.github.com/repos/chainsyncstore/hypothesis-research-engine/actions/runs/123",
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": "Bearer secret-token",
        },
    )

    redirected = handler.redirect_request(
        request,
        None,
        302,
        "Found",
        {},
        "https://api.github.com/repositories/1/actions/runs/123",
    )

    assert redirected is not None
    assert redirected.full_url.startswith("https://api.github.com/")
    assert redirected.headers["Authorization"] == "Bearer secret-token"


def test_finalize_crg3_approval_uses_only_origin_fetch_remote(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(
            args=args[0],
            returncode=0,
            stdout=(
                "new-origin\thttps://github.com/attacker/not-this.git (fetch)\n"
                "new-origin\thttps://github.com/attacker/not-this.git (push)\n"
                "origin\thttps://github.com/chainsyncstore/hypothesis-research-engine.git (fetch)\n"
                "origin\thttps://github.com/chainsyncstore/hypothesis-research-engine.git (push)\n"
            ),
            stderr="",
        )

    monkeypatch.setattr("tools.security.finalize_crg3.subprocess.run", fake_run)

    assert _allowed_github_repos() == {"chainsyncstore/hypothesis-research-engine"}


def test_finalize_crg3_requires_github_origin_fetch_remote(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(
            args=args[0],
            returncode=0,
            stdout=(
                "new-origin\thttps://github.com/chainsyncstore/hypothesis-research-engine.git (fetch)\n"
                "origin\tgit@example.com:chainsyncstore/hypothesis-research-engine.git (fetch)\n"
            ),
            stderr="",
        )

    monkeypatch.setattr("tools.security.finalize_crg3.subprocess.run", fake_run)

    with pytest.raises(ValueError, match="origin fetch remote"):
        _allowed_github_repos()


def test_finalize_crg3_rejects_unapproved_github_repository(tmp_path: Path) -> None:
    roadmap = tmp_path / "roadmap.md"
    roadmap.write_text(
        "| ID | Pass | Priority | Status | Work item | Files | Proof required |\n"
        "| --- | --- | --- | --- | --- | --- | --- |\n"
        "| CRG3 | CI, Regression, and Production Readiness Gate | P0 | In progress | Add workflow | `.github/workflows/production-readiness.yml` | Awaiting workflow evidence. |\n",
        encoding="utf-8",
    )

    try:
        finalize(
            run_url="https://github.com/attacker/not-this-repo/actions/runs/123456789",
            commit_sha="0123456789abcdef0123456789abcdef01234567",
            sbom_artifact="production-sbom",
            verified_date=date(2026, 6, 4),
            roadmap_path=roadmap,
            allowed_repos={"chainsyncstore/hypothesis-research-engine"},
            local_head="0123456789abcdef0123456789abcdef01234567",
        )
    except ValueError as exc:
        assert "not approved" in str(exc)
        assert "chainsyncstore/hypothesis-research-engine" in str(exc)
    else:
        raise AssertionError("finalize accepted a run URL from an unapproved repo")


def test_finalize_crg3_rejects_failed_github_run(tmp_path: Path) -> None:
    roadmap = tmp_path / "roadmap.md"
    roadmap.write_text(
        "| ID | Pass | Priority | Status | Work item | Files | Proof required |\n"
        "| --- | --- | --- | --- | --- | --- | --- |\n"
        "| CRG3 | CI, Regression, and Production Readiness Gate | P0 | In progress | Add workflow | `.github/workflows/production-readiness.yml` | Awaiting workflow evidence. |\n",
        encoding="utf-8",
    )

    try:
        finalize(
            run_url="https://github.com/chainsyncstore/hypothesis-research-engine/actions/runs/123456789",
            commit_sha="0123456789abcdef0123456789abcdef01234567",
            sbom_artifact="production-sbom",
            verified_date=date(2026, 6, 4),
            roadmap_path=roadmap,
            allowed_repos={"chainsyncstore/hypothesis-research-engine"},
            github_api_fetcher=_github_fetcher(conclusion="failure"),
            local_head="0123456789abcdef0123456789abcdef01234567",
            git_status="",
        )
    except ValueError as exc:
        assert "conclusion" in str(exc)
    else:
        raise AssertionError("finalize accepted a failed GitHub run")


def test_finalize_crg3_rejects_invalid_github_run_attempt_type(tmp_path: Path) -> None:
    roadmap = tmp_path / "roadmap.md"
    roadmap.write_text(
        "| ID | Pass | Priority | Status | Work item | Files | Proof required |\n"
        "| --- | --- | --- | --- | --- | --- | --- |\n"
        "| CRG3 | CI, Regression, and Production Readiness Gate | P0 | In progress | Add workflow | `.github/workflows/production-readiness.yml` | Awaiting workflow evidence. |\n",
        encoding="utf-8",
    )

    try:
        finalize(
            run_url="https://github.com/chainsyncstore/hypothesis-research-engine/actions/runs/123456789",
            commit_sha="0123456789abcdef0123456789abcdef01234567",
            sbom_artifact="production-sbom",
            verified_date=date(2026, 6, 4),
            roadmap_path=roadmap,
            allowed_repos={"chainsyncstore/hypothesis-research-engine"},
            github_api_fetcher=_github_fetcher(run_attempt=True),
            local_head="0123456789abcdef0123456789abcdef01234567",
            git_status="",
        )
    except ValueError as exc:
        assert "run_attempt" in str(exc)
    else:
        raise AssertionError("finalize accepted invalid GitHub run_attempt metadata")


def test_finalize_crg3_rejects_nonpositive_github_run_attempt(tmp_path: Path) -> None:
    roadmap = tmp_path / "roadmap.md"
    roadmap.write_text(
        "| ID | Pass | Priority | Status | Work item | Files | Proof required |\n"
        "| --- | --- | --- | --- | --- | --- | --- |\n"
        "| CRG3 | CI, Regression, and Production Readiness Gate | P0 | In progress | Add workflow | `.github/workflows/production-readiness.yml` | Awaiting workflow evidence. |\n",
        encoding="utf-8",
    )

    try:
        finalize(
            run_url="https://github.com/chainsyncstore/hypothesis-research-engine/actions/runs/123456789",
            commit_sha="0123456789abcdef0123456789abcdef01234567",
            sbom_artifact="production-sbom",
            verified_date=date(2026, 6, 4),
            roadmap_path=roadmap,
            allowed_repos={"chainsyncstore/hypothesis-research-engine"},
            github_api_fetcher=_github_fetcher(run_attempt=0),
            local_head="0123456789abcdef0123456789abcdef01234567",
            git_status="",
        )
    except ValueError as exc:
        assert "run_attempt" in str(exc)
    else:
        raise AssertionError("finalize accepted a nonpositive GitHub run_attempt")


def test_finalize_crg3_rejects_wrong_workflow_path(tmp_path: Path) -> None:
    roadmap = tmp_path / "roadmap.md"
    roadmap.write_text(
        "| ID | Pass | Priority | Status | Work item | Files | Proof required |\n"
        "| --- | --- | --- | --- | --- | --- | --- |\n"
        "| CRG3 | CI, Regression, and Production Readiness Gate | P0 | In progress | Add workflow | `.github/workflows/production-readiness.yml` | Awaiting workflow evidence. |\n",
        encoding="utf-8",
    )

    try:
        finalize(
            run_url="https://github.com/chainsyncstore/hypothesis-research-engine/actions/runs/123456789",
            commit_sha="0123456789abcdef0123456789abcdef01234567",
            sbom_artifact="production-sbom",
            verified_date=date(2026, 6, 4),
            roadmap_path=roadmap,
            allowed_repos={"chainsyncstore/hypothesis-research-engine"},
            github_api_fetcher=_github_fetcher(workflow_path=".github/workflows/other.yml"),
            local_head="0123456789abcdef0123456789abcdef01234567",
            git_status="",
        )
    except ValueError as exc:
        assert "workflow path" in str(exc)
    else:
        raise AssertionError("finalize accepted the wrong workflow path")


def test_finalize_crg3_accepts_workflow_path_with_release_ref(tmp_path: Path) -> None:
    roadmap = tmp_path / "roadmap.md"
    roadmap.write_text(
        "| ID | Pass | Priority | Status | Work item | Files | Proof required |\n"
        "| --- | --- | --- | --- | --- | --- | --- |\n"
        "| CRG3 | CI, Regression, and Production Readiness Gate | P0 | In progress | Add workflow | `.github/workflows/production-readiness.yml` | Awaiting workflow evidence. |\n",
        encoding="utf-8",
    )
    sha = "0123456789abcdef0123456789abcdef01234567"

    finalize(
        run_url="https://github.com/chainsyncstore/hypothesis-research-engine/actions/runs/123456789",
        commit_sha=sha,
        sbom_artifact="production-sbom",
        verified_date=date(2026, 6, 4),
        roadmap_path=roadmap,
        allowed_repos={"chainsyncstore/hypothesis-research-engine"},
        github_api_fetcher=_github_fetcher(
            sha=sha,
            workflow_path=".github/workflows/production-readiness.yml@main",
            head_branch="main",
        ),
        github_artifact_fetcher=_artifact_fetcher(),
        local_head=sha,
        git_status="",
    )

    assert "Verified 2026-06-04" in roadmap.read_text(encoding="utf-8")


def test_finalize_crg3_rejects_workflow_path_with_non_release_ref(tmp_path: Path) -> None:
    roadmap = tmp_path / "roadmap.md"
    roadmap.write_text(
        "| ID | Pass | Priority | Status | Work item | Files | Proof required |\n"
        "| --- | --- | --- | --- | --- | --- | --- |\n"
        "| CRG3 | CI, Regression, and Production Readiness Gate | P0 | In progress | Add workflow | `.github/workflows/production-readiness.yml` | Awaiting workflow evidence. |\n",
        encoding="utf-8",
    )

    try:
        finalize(
            run_url="https://github.com/chainsyncstore/hypothesis-research-engine/actions/runs/123456789",
            commit_sha="0123456789abcdef0123456789abcdef01234567",
            sbom_artifact="production-sbom",
            verified_date=date(2026, 6, 4),
            roadmap_path=roadmap,
            allowed_repos={"chainsyncstore/hypothesis-research-engine"},
            github_api_fetcher=_github_fetcher(
                workflow_path=".github/workflows/production-readiness.yml@feature/live-gates",
                head_branch="main",
            ),
            local_head="0123456789abcdef0123456789abcdef01234567",
            git_status="",
        )
    except ValueError as exc:
        assert "workflow path" in str(exc)
    else:
        raise AssertionError("finalize accepted a workflow path from a non-release ref")


def test_finalize_crg3_rejects_pull_request_run(tmp_path: Path) -> None:
    roadmap = tmp_path / "roadmap.md"
    roadmap.write_text(
        "| ID | Pass | Priority | Status | Work item | Files | Proof required |\n"
        "| --- | --- | --- | --- | --- | --- | --- |\n"
        "| CRG3 | CI, Regression, and Production Readiness Gate | P0 | In progress | Add workflow | `.github/workflows/production-readiness.yml` | Awaiting workflow evidence. |\n",
        encoding="utf-8",
    )

    try:
        finalize(
            run_url="https://github.com/chainsyncstore/hypothesis-research-engine/actions/runs/123456789",
            commit_sha="0123456789abcdef0123456789abcdef01234567",
            sbom_artifact="production-sbom",
            verified_date=date(2026, 6, 4),
            roadmap_path=roadmap,
            allowed_repos={"chainsyncstore/hypothesis-research-engine"},
            github_api_fetcher=_github_fetcher(event="pull_request"),
            local_head="0123456789abcdef0123456789abcdef01234567",
            git_status="",
        )
    except ValueError as exc:
        assert "event" in str(exc)
    else:
        raise AssertionError("finalize accepted a pull_request run")


def test_finalize_crg3_rejects_non_release_branch(tmp_path: Path) -> None:
    roadmap = tmp_path / "roadmap.md"
    roadmap.write_text(
        "| ID | Pass | Priority | Status | Work item | Files | Proof required |\n"
        "| --- | --- | --- | --- | --- | --- | --- |\n"
        "| CRG3 | CI, Regression, and Production Readiness Gate | P0 | In progress | Add workflow | `.github/workflows/production-readiness.yml` | Awaiting workflow evidence. |\n",
        encoding="utf-8",
    )

    try:
        finalize(
            run_url="https://github.com/chainsyncstore/hypothesis-research-engine/actions/runs/123456789",
            commit_sha="0123456789abcdef0123456789abcdef01234567",
            sbom_artifact="production-sbom",
            verified_date=date(2026, 6, 4),
            roadmap_path=roadmap,
            allowed_repos={"chainsyncstore/hypothesis-research-engine"},
            github_api_fetcher=_github_fetcher(head_branch="feature/live-gates"),
            local_head="0123456789abcdef0123456789abcdef01234567",
            git_status="",
        )
    except ValueError as exc:
        assert "head_branch" in str(exc)
    else:
        raise AssertionError("finalize accepted a non-release branch run")


def test_finalize_crg3_rejects_github_run_api_url_mismatch(tmp_path: Path) -> None:
    roadmap = tmp_path / "roadmap.md"
    roadmap.write_text(
        "| ID | Pass | Priority | Status | Work item | Files | Proof required |\n"
        "| --- | --- | --- | --- | --- | --- | --- |\n"
        "| CRG3 | CI, Regression, and Production Readiness Gate | P0 | In progress | Add workflow | `.github/workflows/production-readiness.yml` | Awaiting workflow evidence. |\n",
        encoding="utf-8",
    )

    try:
        finalize(
            run_url="https://github.com/chainsyncstore/hypothesis-research-engine/actions/runs/123456789",
            commit_sha="0123456789abcdef0123456789abcdef01234567",
            sbom_artifact="production-sbom",
            verified_date=date(2026, 6, 4),
            roadmap_path=roadmap,
            allowed_repos={"chainsyncstore/hypothesis-research-engine"},
            github_api_fetcher=_github_fetcher(
                api_url="https://api.github.com/repos/chainsyncstore/hypothesis-research-engine/actions/runs/987654321"
            ),
            local_head="0123456789abcdef0123456789abcdef01234567",
            git_status="",
        )
    except ValueError as exc:
        assert "API url" in str(exc)
    else:
        raise AssertionError("finalize accepted API metadata for a different workflow run API URL")


def test_finalize_crg3_rejects_invalid_github_run_id_metadata(tmp_path: Path) -> None:
    roadmap = tmp_path / "roadmap.md"
    roadmap.write_text(
        "| ID | Pass | Priority | Status | Work item | Files | Proof required |\n"
        "| --- | --- | --- | --- | --- | --- | --- |\n"
        "| CRG3 | CI, Regression, and Production Readiness Gate | P0 | In progress | Add workflow | `.github/workflows/production-readiness.yml` | Awaiting workflow evidence. |\n",
        encoding="utf-8",
    )

    try:
        finalize(
            run_url="https://github.com/chainsyncstore/hypothesis-research-engine/actions/runs/123456789",
            commit_sha="0123456789abcdef0123456789abcdef01234567",
            sbom_artifact="production-sbom",
            verified_date=date(2026, 6, 4),
            roadmap_path=roadmap,
            allowed_repos={"chainsyncstore/hypothesis-research-engine"},
            github_api_fetcher=_github_fetcher(run_id=True),
            local_head="0123456789abcdef0123456789abcdef01234567",
            git_status="",
        )
    except ValueError as exc:
        assert "run id" in str(exc)
    else:
        raise AssertionError("finalize accepted invalid GitHub run id metadata")


def test_finalize_crg3_rejects_mismatched_github_run_id_metadata(tmp_path: Path) -> None:
    roadmap = tmp_path / "roadmap.md"
    roadmap.write_text(
        "| ID | Pass | Priority | Status | Work item | Files | Proof required |\n"
        "| --- | --- | --- | --- | --- | --- | --- |\n"
        "| CRG3 | CI, Regression, and Production Readiness Gate | P0 | In progress | Add workflow | `.github/workflows/production-readiness.yml` | Awaiting workflow evidence. |\n",
        encoding="utf-8",
    )

    try:
        finalize(
            run_url="https://github.com/chainsyncstore/hypothesis-research-engine/actions/runs/123456789",
            commit_sha="0123456789abcdef0123456789abcdef01234567",
            sbom_artifact="production-sbom",
            verified_date=date(2026, 6, 4),
            roadmap_path=roadmap,
            allowed_repos={"chainsyncstore/hypothesis-research-engine"},
            github_api_fetcher=_github_fetcher(run_id=987654321),
            local_head="0123456789abcdef0123456789abcdef01234567",
            git_status="",
        )
    except ValueError as exc:
        assert "run id" in str(exc)
    else:
        raise AssertionError("finalize accepted mismatched GitHub run id metadata")


def test_finalize_crg3_rejects_invalid_github_run_repository_metadata(
    tmp_path: Path,
) -> None:
    roadmap = tmp_path / "roadmap.md"
    roadmap.write_text(
        "| ID | Pass | Priority | Status | Work item | Files | Proof required |\n"
        "| --- | --- | --- | --- | --- | --- | --- |\n"
        "| CRG3 | CI, Regression, and Production Readiness Gate | P0 | In progress | Add workflow | `.github/workflows/production-readiness.yml` | Awaiting workflow evidence. |\n",
        encoding="utf-8",
    )

    try:
        finalize(
            run_url="https://github.com/chainsyncstore/hypothesis-research-engine/actions/runs/123456789",
            commit_sha="0123456789abcdef0123456789abcdef01234567",
            sbom_artifact="production-sbom",
            verified_date=date(2026, 6, 4),
            roadmap_path=roadmap,
            allowed_repos={"chainsyncstore/hypothesis-research-engine"},
            github_api_fetcher=_github_fetcher(repository_full_name=True),
            local_head="0123456789abcdef0123456789abcdef01234567",
            git_status="",
        )
    except ValueError as exc:
        assert "repository metadata" in str(exc)
    else:
        raise AssertionError("finalize accepted invalid GitHub run repository metadata")


def test_finalize_crg3_rejects_mismatched_github_run_repository_metadata(
    tmp_path: Path,
) -> None:
    roadmap = tmp_path / "roadmap.md"
    roadmap.write_text(
        "| ID | Pass | Priority | Status | Work item | Files | Proof required |\n"
        "| --- | --- | --- | --- | --- | --- | --- |\n"
        "| CRG3 | CI, Regression, and Production Readiness Gate | P0 | In progress | Add workflow | `.github/workflows/production-readiness.yml` | Awaiting workflow evidence. |\n",
        encoding="utf-8",
    )

    try:
        finalize(
            run_url="https://github.com/chainsyncstore/hypothesis-research-engine/actions/runs/123456789",
            commit_sha="0123456789abcdef0123456789abcdef01234567",
            sbom_artifact="production-sbom",
            verified_date=date(2026, 6, 4),
            roadmap_path=roadmap,
            allowed_repos={"chainsyncstore/hypothesis-research-engine"},
            github_api_fetcher=_github_fetcher(repository_full_name="attacker/not-this"),
            local_head="0123456789abcdef0123456789abcdef01234567",
            git_status="",
        )
    except ValueError as exc:
        assert "repository metadata" in str(exc)
    else:
        raise AssertionError("finalize accepted mismatched GitHub run repository metadata")


def test_finalize_crg3_rejects_github_run_html_url_mismatch(tmp_path: Path) -> None:
    roadmap = tmp_path / "roadmap.md"
    roadmap.write_text(
        "| ID | Pass | Priority | Status | Work item | Files | Proof required |\n"
        "| --- | --- | --- | --- | --- | --- | --- |\n"
        "| CRG3 | CI, Regression, and Production Readiness Gate | P0 | In progress | Add workflow | `.github/workflows/production-readiness.yml` | Awaiting workflow evidence. |\n",
        encoding="utf-8",
    )

    try:
        finalize(
            run_url="https://github.com/chainsyncstore/hypothesis-research-engine/actions/runs/123456789",
            commit_sha="0123456789abcdef0123456789abcdef01234567",
            sbom_artifact="production-sbom",
            verified_date=date(2026, 6, 4),
            roadmap_path=roadmap,
            allowed_repos={"chainsyncstore/hypothesis-research-engine"},
            github_api_fetcher=_github_fetcher(
                html_url="https://github.com/chainsyncstore/hypothesis-research-engine/actions/runs/987654321"
            ),
            local_head="0123456789abcdef0123456789abcdef01234567",
            git_status="",
        )
    except ValueError as exc:
        assert "html_url" in str(exc)
    else:
        raise AssertionError("finalize accepted API metadata for a different workflow run URL")


def test_finalize_crg3_rejects_github_run_sha_mismatch(tmp_path: Path) -> None:
    roadmap = tmp_path / "roadmap.md"
    roadmap.write_text(
        "| ID | Pass | Priority | Status | Work item | Files | Proof required |\n"
        "| --- | --- | --- | --- | --- | --- | --- |\n"
        "| CRG3 | CI, Regression, and Production Readiness Gate | P0 | In progress | Add workflow | `.github/workflows/production-readiness.yml` | Awaiting workflow evidence. |\n",
        encoding="utf-8",
    )

    try:
        finalize(
            run_url="https://github.com/chainsyncstore/hypothesis-research-engine/actions/runs/123456789",
            commit_sha="0123456789abcdef0123456789abcdef01234567",
            sbom_artifact="production-sbom",
            verified_date=date(2026, 6, 4),
            roadmap_path=roadmap,
            allowed_repos={"chainsyncstore/hypothesis-research-engine"},
            github_api_fetcher=_github_fetcher(
                sha="fedcba9876543210fedcba9876543210fedcba98"
            ),
            local_head="0123456789abcdef0123456789abcdef01234567",
            git_status="",
        )
    except ValueError as exc:
        assert "head_sha" in str(exc)
    else:
        raise AssertionError("finalize accepted a mismatched GitHub run SHA")


def test_finalize_crg3_rejects_github_run_artifacts_url_mismatch(tmp_path: Path) -> None:
    roadmap = tmp_path / "roadmap.md"
    roadmap.write_text(
        "| ID | Pass | Priority | Status | Work item | Files | Proof required |\n"
        "| --- | --- | --- | --- | --- | --- | --- |\n"
        "| CRG3 | CI, Regression, and Production Readiness Gate | P0 | In progress | Add workflow | `.github/workflows/production-readiness.yml` | Awaiting workflow evidence. |\n",
        encoding="utf-8",
    )

    try:
        finalize(
            run_url="https://github.com/chainsyncstore/hypothesis-research-engine/actions/runs/123456789",
            commit_sha="0123456789abcdef0123456789abcdef01234567",
            sbom_artifact="production-sbom",
            verified_date=date(2026, 6, 4),
            roadmap_path=roadmap,
            allowed_repos={"chainsyncstore/hypothesis-research-engine"},
            github_api_fetcher=_github_fetcher(
                artifacts_url="https://api.github.com/repos/chainsyncstore/hypothesis-research-engine/actions/runs/987654321/artifacts"
            ),
            local_head="0123456789abcdef0123456789abcdef01234567",
            git_status="",
        )
    except ValueError as exc:
        assert "artifacts_url" in str(exc)
    else:
        raise AssertionError("finalize accepted API metadata for a different artifacts_url")


def test_finalize_crg3_rejects_missing_sbom_artifact_in_github_run(tmp_path: Path) -> None:
    roadmap = tmp_path / "roadmap.md"
    roadmap.write_text(
        "| ID | Pass | Priority | Status | Work item | Files | Proof required |\n"
        "| --- | --- | --- | --- | --- | --- | --- |\n"
        "| CRG3 | CI, Regression, and Production Readiness Gate | P0 | In progress | Add workflow | `.github/workflows/production-readiness.yml` | Awaiting workflow evidence. |\n",
        encoding="utf-8",
    )

    try:
        finalize(
            run_url="https://github.com/chainsyncstore/hypothesis-research-engine/actions/runs/123456789",
            commit_sha="0123456789abcdef0123456789abcdef01234567",
            sbom_artifact="production-sbom",
            verified_date=date(2026, 6, 4),
            roadmap_path=roadmap,
            allowed_repos={"chainsyncstore/hypothesis-research-engine"},
            github_api_fetcher=_github_fetcher(artifact_name="not-production-sbom"),
            local_head="0123456789abcdef0123456789abcdef01234567",
            git_status="",
        )
    except ValueError as exc:
        assert "production-sbom" in str(exc)
    else:
        raise AssertionError("finalize accepted a run without production-sbom artifact")


def test_finalize_crg3_rejects_sbom_artifact_without_explicit_nonexpired_flag(
    tmp_path: Path,
) -> None:
    roadmap = tmp_path / "roadmap.md"
    roadmap.write_text(
        "| ID | Pass | Priority | Status | Work item | Files | Proof required |\n"
        "| --- | --- | --- | --- | --- | --- | --- |\n"
        "| CRG3 | CI, Regression, and Production Readiness Gate | P0 | In progress | Add workflow | `.github/workflows/production-readiness.yml` | Awaiting workflow evidence. |\n",
        encoding="utf-8",
    )

    def fetch(url: str):
        if url.endswith("/actions/runs/123456789"):
            return {
                "url": "https://api.github.com/repos/chainsyncstore/hypothesis-research-engine/actions/runs/123456789",
                "id": 123456789,
                "repository": {"full_name": "chainsyncstore/hypothesis-research-engine"},
                "html_url": "https://github.com/chainsyncstore/hypothesis-research-engine/actions/runs/123456789",
                "name": "Production Readiness",
                "path": ".github/workflows/production-readiness.yml",
                "event": "workflow_dispatch",
                "head_branch": "main",
                "status": "completed",
                "conclusion": "success",
                "run_attempt": 1,
                "head_sha": "0123456789abcdef0123456789abcdef01234567",
                "artifacts_url": "https://api.github.com/repos/chainsyncstore/hypothesis-research-engine/actions/runs/123456789/artifacts",
            }
        if url.endswith("/actions/runs/123456789/artifacts?per_page=100&page=1"):
            return {
                "total_count": 1,
                "artifacts": [
                    {
                        "name": "production-sbom",
                        "archive_download_url": "https://api.github.com/repos/chainsyncstore/hypothesis-research-engine/actions/artifacts/42/zip",
                    },
                ],
            }
        raise AssertionError(f"unexpected GitHub API URL: {url}")

    try:
        finalize(
            run_url="https://github.com/chainsyncstore/hypothesis-research-engine/actions/runs/123456789",
            commit_sha="0123456789abcdef0123456789abcdef01234567",
            sbom_artifact="production-sbom",
            verified_date=date(2026, 6, 4),
            roadmap_path=roadmap,
            allowed_repos={"chainsyncstore/hypothesis-research-engine"},
            github_api_fetcher=fetch,
            github_artifact_fetcher=_artifact_fetcher(),
            local_head="0123456789abcdef0123456789abcdef01234567",
            git_status="",
        )
    except ValueError as exc:
        assert "non-expired" in str(exc)
    else:
        raise AssertionError("finalize accepted a production-sbom artifact without expired=false")


def test_finalize_crg3_rejects_duplicate_nonexpired_sbom_artifacts(tmp_path: Path) -> None:
    roadmap = tmp_path / "roadmap.md"
    roadmap.write_text(
        "| ID | Pass | Priority | Status | Work item | Files | Proof required |\n"
        "| --- | --- | --- | --- | --- | --- | --- |\n"
        "| CRG3 | CI, Regression, and Production Readiness Gate | P0 | In progress | Add workflow | `.github/workflows/production-readiness.yml` | Awaiting workflow evidence. |\n",
        encoding="utf-8",
    )

    def fetch(url: str):
        if url.endswith("/actions/runs/123456789"):
            return {
                "url": "https://api.github.com/repos/chainsyncstore/hypothesis-research-engine/actions/runs/123456789",
                "id": 123456789,
                "repository": {"full_name": "chainsyncstore/hypothesis-research-engine"},
                "html_url": "https://github.com/chainsyncstore/hypothesis-research-engine/actions/runs/123456789",
                "name": "Production Readiness",
                "path": ".github/workflows/production-readiness.yml",
                "event": "workflow_dispatch",
                "head_branch": "main",
                "status": "completed",
                "conclusion": "success",
                "run_attempt": 1,
                "head_sha": "0123456789abcdef0123456789abcdef01234567",
                "artifacts_url": "https://api.github.com/repos/chainsyncstore/hypothesis-research-engine/actions/runs/123456789/artifacts",
            }
        if url.endswith("/actions/runs/123456789/artifacts?per_page=100&page=1"):
            return {
                "total_count": 2,
                "artifacts": [
                    {
                        "name": "production-sbom",
                        "expired": False,
                        "id": 42,
                        "size_in_bytes": 2048,
                        "archive_download_url": "https://api.github.com/repos/chainsyncstore/hypothesis-research-engine/actions/artifacts/42/zip",
                    },
                    {
                        "name": "production-sbom",
                        "expired": False,
                        "id": 43,
                        "size_in_bytes": 2048,
                        "archive_download_url": "https://api.github.com/repos/chainsyncstore/hypothesis-research-engine/actions/artifacts/43/zip",
                    },
                ],
            }
        raise AssertionError(f"unexpected GitHub API URL: {url}")

    try:
        finalize(
            run_url="https://github.com/chainsyncstore/hypothesis-research-engine/actions/runs/123456789",
            commit_sha="0123456789abcdef0123456789abcdef01234567",
            sbom_artifact="production-sbom",
            verified_date=date(2026, 6, 4),
            roadmap_path=roadmap,
            allowed_repos={"chainsyncstore/hypothesis-research-engine"},
            github_api_fetcher=fetch,
            github_artifact_fetcher=_artifact_fetcher(),
            local_head="0123456789abcdef0123456789abcdef01234567",
            git_status="",
        )
    except ValueError as exc:
        assert "exactly one" in str(exc)
    else:
        raise AssertionError("finalize accepted duplicate non-expired production-sbom artifacts")


def test_finalize_crg3_rejects_invalid_sbom_artifact_size_metadata(tmp_path: Path) -> None:
    roadmap = tmp_path / "roadmap.md"
    roadmap.write_text(
        "| ID | Pass | Priority | Status | Work item | Files | Proof required |\n"
        "| --- | --- | --- | --- | --- | --- | --- |\n"
        "| CRG3 | CI, Regression, and Production Readiness Gate | P0 | In progress | Add workflow | `.github/workflows/production-readiness.yml` | Awaiting workflow evidence. |\n",
        encoding="utf-8",
    )

    try:
        finalize(
            run_url="https://github.com/chainsyncstore/hypothesis-research-engine/actions/runs/123456789",
            commit_sha="0123456789abcdef0123456789abcdef01234567",
            sbom_artifact="production-sbom",
            verified_date=date(2026, 6, 4),
            roadmap_path=roadmap,
            allowed_repos={"chainsyncstore/hypothesis-research-engine"},
            github_api_fetcher=_github_fetcher(artifact_size_in_bytes=True),
            github_artifact_fetcher=_artifact_fetcher(),
            local_head="0123456789abcdef0123456789abcdef01234567",
            git_status="",
        )
    except ValueError as exc:
        assert "size_in_bytes" in str(exc)
    else:
        raise AssertionError("finalize accepted invalid production-sbom size_in_bytes metadata")


def test_finalize_crg3_rejects_zero_sbom_artifact_size_before_download(
    tmp_path: Path,
) -> None:
    roadmap = tmp_path / "roadmap.md"
    roadmap.write_text(
        "| ID | Pass | Priority | Status | Work item | Files | Proof required |\n"
        "| --- | --- | --- | --- | --- | --- | --- |\n"
        "| CRG3 | CI, Regression, and Production Readiness Gate | P0 | In progress | Add workflow | `.github/workflows/production-readiness.yml` | Awaiting workflow evidence. |\n",
        encoding="utf-8",
    )

    def fail_artifact_fetcher(url: str) -> bytes:
        raise AssertionError(f"zero-size metadata should be rejected before download: {url}")

    try:
        finalize(
            run_url="https://github.com/chainsyncstore/hypothesis-research-engine/actions/runs/123456789",
            commit_sha="0123456789abcdef0123456789abcdef01234567",
            sbom_artifact="production-sbom",
            verified_date=date(2026, 6, 4),
            roadmap_path=roadmap,
            allowed_repos={"chainsyncstore/hypothesis-research-engine"},
            github_api_fetcher=_github_fetcher(artifact_size_in_bytes=0),
            github_artifact_fetcher=fail_artifact_fetcher,
            local_head="0123456789abcdef0123456789abcdef01234567",
            git_status="",
        )
    except ValueError as exc:
        assert "size_in_bytes" in str(exc)
    else:
        raise AssertionError("finalize accepted zero production-sbom size_in_bytes metadata")


def test_finalize_crg3_rejects_oversized_sbom_artifact_size_metadata(tmp_path: Path) -> None:
    roadmap = tmp_path / "roadmap.md"
    roadmap.write_text(
        "| ID | Pass | Priority | Status | Work item | Files | Proof required |\n"
        "| --- | --- | --- | --- | --- | --- | --- |\n"
        "| CRG3 | CI, Regression, and Production Readiness Gate | P0 | In progress | Add workflow | `.github/workflows/production-readiness.yml` | Awaiting workflow evidence. |\n",
        encoding="utf-8",
    )

    try:
        finalize(
            run_url="https://github.com/chainsyncstore/hypothesis-research-engine/actions/runs/123456789",
            commit_sha="0123456789abcdef0123456789abcdef01234567",
            sbom_artifact="production-sbom",
            verified_date=date(2026, 6, 4),
            roadmap_path=roadmap,
            allowed_repos={"chainsyncstore/hypothesis-research-engine"},
            github_api_fetcher=_github_fetcher(
                artifact_size_in_bytes=MAX_SBOM_ARTIFACT_BYTES + 1
            ),
            github_artifact_fetcher=_artifact_fetcher(),
            local_head="0123456789abcdef0123456789abcdef01234567",
            git_status="",
        )
    except ValueError as exc:
        assert "too large" in str(exc)
    else:
        raise AssertionError("finalize accepted oversized production-sbom size_in_bytes metadata")


def test_finalize_crg3_rejects_invalid_sbom_artifact_id_metadata(tmp_path: Path) -> None:
    roadmap = tmp_path / "roadmap.md"
    roadmap.write_text(
        "| ID | Pass | Priority | Status | Work item | Files | Proof required |\n"
        "| --- | --- | --- | --- | --- | --- | --- |\n"
        "| CRG3 | CI, Regression, and Production Readiness Gate | P0 | In progress | Add workflow | `.github/workflows/production-readiness.yml` | Awaiting workflow evidence. |\n",
        encoding="utf-8",
    )

    try:
        finalize(
            run_url="https://github.com/chainsyncstore/hypothesis-research-engine/actions/runs/123456789",
            commit_sha="0123456789abcdef0123456789abcdef01234567",
            sbom_artifact="production-sbom",
            verified_date=date(2026, 6, 4),
            roadmap_path=roadmap,
            allowed_repos={"chainsyncstore/hypothesis-research-engine"},
            github_api_fetcher=_github_fetcher(artifact_id=True),
            github_artifact_fetcher=_artifact_fetcher(),
            local_head="0123456789abcdef0123456789abcdef01234567",
            git_status="",
        )
    except ValueError as exc:
        assert "invalid id" in str(exc)
    else:
        raise AssertionError("finalize accepted invalid production-sbom id metadata")


def test_finalize_crg3_rejects_mismatched_sbom_artifact_id_metadata(
    tmp_path: Path,
) -> None:
    roadmap = tmp_path / "roadmap.md"
    roadmap.write_text(
        "| ID | Pass | Priority | Status | Work item | Files | Proof required |\n"
        "| --- | --- | --- | --- | --- | --- | --- |\n"
        "| CRG3 | CI, Regression, and Production Readiness Gate | P0 | In progress | Add workflow | `.github/workflows/production-readiness.yml` | Awaiting workflow evidence. |\n",
        encoding="utf-8",
    )

    try:
        finalize(
            run_url="https://github.com/chainsyncstore/hypothesis-research-engine/actions/runs/123456789",
            commit_sha="0123456789abcdef0123456789abcdef01234567",
            sbom_artifact="production-sbom",
            verified_date=date(2026, 6, 4),
            roadmap_path=roadmap,
            allowed_repos={"chainsyncstore/hypothesis-research-engine"},
            github_api_fetcher=_github_fetcher(artifact_id=99),
            github_artifact_fetcher=_artifact_fetcher(),
            local_head="0123456789abcdef0123456789abcdef01234567",
            git_status="",
        )
    except ValueError as exc:
        assert "id does not match" in str(exc)
    else:
        raise AssertionError("finalize accepted mismatched production-sbom id metadata")


def test_finalize_crg3_rejects_sbom_artifact_download_url_query(
    tmp_path: Path,
) -> None:
    roadmap = tmp_path / "roadmap.md"
    roadmap.write_text(
        "| ID | Pass | Priority | Status | Work item | Files | Proof required |\n"
        "| --- | --- | --- | --- | --- | --- | --- |\n"
        "| CRG3 | CI, Regression, and Production Readiness Gate | P0 | In progress | Add workflow | `.github/workflows/production-readiness.yml` | Awaiting workflow evidence. |\n",
        encoding="utf-8",
    )

    try:
        finalize(
            run_url="https://github.com/chainsyncstore/hypothesis-research-engine/actions/runs/123456789",
            commit_sha="0123456789abcdef0123456789abcdef01234567",
            sbom_artifact="production-sbom",
            verified_date=date(2026, 6, 4),
            roadmap_path=roadmap,
            allowed_repos={"chainsyncstore/hypothesis-research-engine"},
            github_api_fetcher=_github_fetcher(
                archive_download_url=(
                    "https://api.github.com/repos/chainsyncstore/"
                    "hypothesis-research-engine/actions/artifacts/42/zip?unexpected=1"
                )
            ),
            github_artifact_fetcher=_artifact_fetcher(),
            local_head="0123456789abcdef0123456789abcdef01234567",
            git_status="",
        )
    except ValueError as exc:
        assert "canonical" in str(exc)
    else:
        raise AssertionError("finalize accepted a production-sbom artifact URL with a query")


def test_finalize_crg3_rejects_sbom_artifact_download_url_fragment(
    tmp_path: Path,
) -> None:
    roadmap = tmp_path / "roadmap.md"
    roadmap.write_text(
        "| ID | Pass | Priority | Status | Work item | Files | Proof required |\n"
        "| --- | --- | --- | --- | --- | --- | --- |\n"
        "| CRG3 | CI, Regression, and Production Readiness Gate | P0 | In progress | Add workflow | `.github/workflows/production-readiness.yml` | Awaiting workflow evidence. |\n",
        encoding="utf-8",
    )

    try:
        finalize(
            run_url="https://github.com/chainsyncstore/hypothesis-research-engine/actions/runs/123456789",
            commit_sha="0123456789abcdef0123456789abcdef01234567",
            sbom_artifact="production-sbom",
            verified_date=date(2026, 6, 4),
            roadmap_path=roadmap,
            allowed_repos={"chainsyncstore/hypothesis-research-engine"},
            github_api_fetcher=_github_fetcher(
                archive_download_url=(
                    "https://api.github.com/repos/chainsyncstore/"
                    "hypothesis-research-engine/actions/artifacts/42/zip#fragment"
                )
            ),
            github_artifact_fetcher=_artifact_fetcher(),
            local_head="0123456789abcdef0123456789abcdef01234567",
            git_status="",
        )
    except ValueError as exc:
        assert "canonical" in str(exc)
    else:
        raise AssertionError("finalize accepted a production-sbom artifact URL with a fragment")


def test_finalize_crg3_rejects_invalid_artifact_total_count(tmp_path: Path) -> None:
    roadmap = tmp_path / "roadmap.md"
    roadmap.write_text(
        "| ID | Pass | Priority | Status | Work item | Files | Proof required |\n"
        "| --- | --- | --- | --- | --- | --- | --- |\n"
        "| CRG3 | CI, Regression, and Production Readiness Gate | P0 | In progress | Add workflow | `.github/workflows/production-readiness.yml` | Awaiting workflow evidence. |\n",
        encoding="utf-8",
    )

    def fetch(url: str):
        if url.endswith("/actions/runs/123456789"):
            return {
                "url": "https://api.github.com/repos/chainsyncstore/hypothesis-research-engine/actions/runs/123456789",
                "id": 123456789,
                "repository": {"full_name": "chainsyncstore/hypothesis-research-engine"},
                "html_url": "https://github.com/chainsyncstore/hypothesis-research-engine/actions/runs/123456789",
                "name": "Production Readiness",
                "path": ".github/workflows/production-readiness.yml",
                "event": "workflow_dispatch",
                "head_branch": "main",
                "status": "completed",
                "conclusion": "success",
                "run_attempt": 1,
                "head_sha": "0123456789abcdef0123456789abcdef01234567",
                "artifacts_url": "https://api.github.com/repos/chainsyncstore/hypothesis-research-engine/actions/runs/123456789/artifacts",
            }
        if url.endswith("/actions/runs/123456789/artifacts?per_page=100&page=1"):
            return {
                "total_count": "1",
                "artifacts": [
                    {
                        "name": "production-sbom",
                        "expired": False,
                        "id": 42,
                        "size_in_bytes": 2048,
                        "archive_download_url": "https://api.github.com/repos/chainsyncstore/hypothesis-research-engine/actions/artifacts/42/zip",
                    },
                ],
            }
        raise AssertionError(f"unexpected GitHub API URL: {url}")

    try:
        finalize(
            run_url="https://github.com/chainsyncstore/hypothesis-research-engine/actions/runs/123456789",
            commit_sha="0123456789abcdef0123456789abcdef01234567",
            sbom_artifact="production-sbom",
            verified_date=date(2026, 6, 4),
            roadmap_path=roadmap,
            allowed_repos={"chainsyncstore/hypothesis-research-engine"},
            github_api_fetcher=fetch,
            github_artifact_fetcher=_artifact_fetcher(),
            local_head="0123456789abcdef0123456789abcdef01234567",
            git_status="",
        )
    except ValueError as exc:
        assert "total_count" in str(exc)
    else:
        raise AssertionError("finalize accepted invalid GitHub artifact total_count metadata")


def test_finalize_crg3_rejects_bool_artifact_total_count(tmp_path: Path) -> None:
    roadmap = tmp_path / "roadmap.md"
    roadmap.write_text(
        "| ID | Pass | Priority | Status | Work item | Files | Proof required |\n"
        "| --- | --- | --- | --- | --- | --- | --- |\n"
        "| CRG3 | CI, Regression, and Production Readiness Gate | P0 | In progress | Add workflow | `.github/workflows/production-readiness.yml` | Awaiting workflow evidence. |\n",
        encoding="utf-8",
    )

    def fetch(url: str):
        if url.endswith("/actions/runs/123456789"):
            return {
                "url": "https://api.github.com/repos/chainsyncstore/hypothesis-research-engine/actions/runs/123456789",
                "id": 123456789,
                "repository": {"full_name": "chainsyncstore/hypothesis-research-engine"},
                "html_url": "https://github.com/chainsyncstore/hypothesis-research-engine/actions/runs/123456789",
                "name": "Production Readiness",
                "path": ".github/workflows/production-readiness.yml",
                "event": "workflow_dispatch",
                "head_branch": "main",
                "status": "completed",
                "conclusion": "success",
                "run_attempt": 1,
                "head_sha": "0123456789abcdef0123456789abcdef01234567",
                "artifacts_url": "https://api.github.com/repos/chainsyncstore/hypothesis-research-engine/actions/runs/123456789/artifacts",
            }
        if url.endswith("/actions/runs/123456789/artifacts?per_page=100&page=1"):
            return {
                "total_count": True,
                "artifacts": [
                    {
                        "name": "production-sbom",
                        "expired": False,
                        "id": 42,
                        "size_in_bytes": 2048,
                        "archive_download_url": "https://api.github.com/repos/chainsyncstore/hypothesis-research-engine/actions/artifacts/42/zip",
                    },
                ],
            }
        raise AssertionError(f"unexpected GitHub API URL: {url}")

    def fail_artifact_fetcher(url: str) -> bytes:
        raise AssertionError(f"boolean total_count should be rejected before download: {url}")

    try:
        finalize(
            run_url="https://github.com/chainsyncstore/hypothesis-research-engine/actions/runs/123456789",
            commit_sha="0123456789abcdef0123456789abcdef01234567",
            sbom_artifact="production-sbom",
            verified_date=date(2026, 6, 4),
            roadmap_path=roadmap,
            allowed_repos={"chainsyncstore/hypothesis-research-engine"},
            github_api_fetcher=fetch,
            github_artifact_fetcher=fail_artifact_fetcher,
            local_head="0123456789abcdef0123456789abcdef01234567",
            git_status="",
        )
    except ValueError as exc:
        assert "total_count" in str(exc)
    else:
        raise AssertionError("finalize accepted boolean GitHub artifact total_count metadata")


def test_finalize_crg3_rejects_missing_artifacts_list(tmp_path: Path) -> None:
    roadmap = tmp_path / "roadmap.md"
    roadmap.write_text(
        "| ID | Pass | Priority | Status | Work item | Files | Proof required |\n"
        "| --- | --- | --- | --- | --- | --- | --- |\n"
        "| CRG3 | CI, Regression, and Production Readiness Gate | P0 | In progress | Add workflow | `.github/workflows/production-readiness.yml` | Awaiting workflow evidence. |\n",
        encoding="utf-8",
    )

    def fetch(url: str):
        if url.endswith("/actions/runs/123456789"):
            return {
                "url": "https://api.github.com/repos/chainsyncstore/hypothesis-research-engine/actions/runs/123456789",
                "id": 123456789,
                "repository": {"full_name": "chainsyncstore/hypothesis-research-engine"},
                "html_url": "https://github.com/chainsyncstore/hypothesis-research-engine/actions/runs/123456789",
                "name": "Production Readiness",
                "path": ".github/workflows/production-readiness.yml",
                "event": "workflow_dispatch",
                "head_branch": "main",
                "status": "completed",
                "conclusion": "success",
                "run_attempt": 1,
                "head_sha": "0123456789abcdef0123456789abcdef01234567",
                "artifacts_url": "https://api.github.com/repos/chainsyncstore/hypothesis-research-engine/actions/runs/123456789/artifacts",
            }
        if url.endswith("/actions/runs/123456789/artifacts?per_page=100&page=1"):
            return {"total_count": 0}
        raise AssertionError(f"unexpected GitHub API URL: {url}")

    try:
        finalize(
            run_url="https://github.com/chainsyncstore/hypothesis-research-engine/actions/runs/123456789",
            commit_sha="0123456789abcdef0123456789abcdef01234567",
            sbom_artifact="production-sbom",
            verified_date=date(2026, 6, 4),
            roadmap_path=roadmap,
            allowed_repos={"chainsyncstore/hypothesis-research-engine"},
            github_api_fetcher=fetch,
            github_artifact_fetcher=_artifact_fetcher(),
            local_head="0123456789abcdef0123456789abcdef01234567",
            git_status="",
        )
    except ValueError as exc:
        assert "missing artifacts" in str(exc)
    else:
        raise AssertionError("finalize accepted a GitHub artifact page missing artifacts")


def test_finalize_crg3_rejects_artifact_entries_exceeding_total_count(
    tmp_path: Path,
) -> None:
    roadmap = tmp_path / "roadmap.md"
    roadmap.write_text(
        "| ID | Pass | Priority | Status | Work item | Files | Proof required |\n"
        "| --- | --- | --- | --- | --- | --- | --- |\n"
        "| CRG3 | CI, Regression, and Production Readiness Gate | P0 | In progress | Add workflow | `.github/workflows/production-readiness.yml` | Awaiting workflow evidence. |\n",
        encoding="utf-8",
    )

    def fetch(url: str):
        if url.endswith("/actions/runs/123456789"):
            return {
                "url": "https://api.github.com/repos/chainsyncstore/hypothesis-research-engine/actions/runs/123456789",
                "id": 123456789,
                "repository": {"full_name": "chainsyncstore/hypothesis-research-engine"},
                "html_url": "https://github.com/chainsyncstore/hypothesis-research-engine/actions/runs/123456789",
                "name": "Production Readiness",
                "path": ".github/workflows/production-readiness.yml",
                "event": "workflow_dispatch",
                "head_branch": "main",
                "status": "completed",
                "conclusion": "success",
                "run_attempt": 1,
                "head_sha": "0123456789abcdef0123456789abcdef01234567",
                "artifacts_url": "https://api.github.com/repos/chainsyncstore/hypothesis-research-engine/actions/runs/123456789/artifacts",
            }
        if url.endswith("/actions/runs/123456789/artifacts?per_page=100&page=1"):
            return {
                "total_count": 1,
                "artifacts": [
                    {"name": "other-artifact-1", "expired": False},
                    {"name": "other-artifact-2", "expired": False},
                ],
            }
        raise AssertionError(f"unexpected GitHub API URL: {url}")

    try:
        finalize(
            run_url="https://github.com/chainsyncstore/hypothesis-research-engine/actions/runs/123456789",
            commit_sha="0123456789abcdef0123456789abcdef01234567",
            sbom_artifact="production-sbom",
            verified_date=date(2026, 6, 4),
            roadmap_path=roadmap,
            allowed_repos={"chainsyncstore/hypothesis-research-engine"},
            github_api_fetcher=fetch,
            github_artifact_fetcher=_artifact_fetcher(),
            local_head="0123456789abcdef0123456789abcdef01234567",
            git_status="",
        )
    except ValueError as exc:
        assert "exceed total_count" in str(exc)
    else:
        raise AssertionError("finalize accepted artifact entries exceeding total_count")


def test_finalize_crg3_rejects_empty_artifact_page_before_total_count(
    tmp_path: Path,
) -> None:
    roadmap = tmp_path / "roadmap.md"
    roadmap.write_text(
        "| ID | Pass | Priority | Status | Work item | Files | Proof required |\n"
        "| --- | --- | --- | --- | --- | --- | --- |\n"
        "| CRG3 | CI, Regression, and Production Readiness Gate | P0 | In progress | Add workflow | `.github/workflows/production-readiness.yml` | Awaiting workflow evidence. |\n",
        encoding="utf-8",
    )

    def fetch(url: str):
        if url.endswith("/actions/runs/123456789"):
            return {
                "url": "https://api.github.com/repos/chainsyncstore/hypothesis-research-engine/actions/runs/123456789",
                "id": 123456789,
                "repository": {"full_name": "chainsyncstore/hypothesis-research-engine"},
                "html_url": "https://github.com/chainsyncstore/hypothesis-research-engine/actions/runs/123456789",
                "name": "Production Readiness",
                "path": ".github/workflows/production-readiness.yml",
                "event": "workflow_dispatch",
                "head_branch": "main",
                "status": "completed",
                "conclusion": "success",
                "run_attempt": 1,
                "head_sha": "0123456789abcdef0123456789abcdef01234567",
                "artifacts_url": "https://api.github.com/repos/chainsyncstore/hypothesis-research-engine/actions/runs/123456789/artifacts",
            }
        if url.endswith("/actions/runs/123456789/artifacts?per_page=100&page=1"):
            return {"total_count": 1, "artifacts": []}
        raise AssertionError(f"unexpected GitHub API URL: {url}")

    try:
        finalize(
            run_url="https://github.com/chainsyncstore/hypothesis-research-engine/actions/runs/123456789",
            commit_sha="0123456789abcdef0123456789abcdef01234567",
            sbom_artifact="production-sbom",
            verified_date=date(2026, 6, 4),
            roadmap_path=roadmap,
            allowed_repos={"chainsyncstore/hypothesis-research-engine"},
            github_api_fetcher=fetch,
            github_artifact_fetcher=_artifact_fetcher(),
            local_head="0123456789abcdef0123456789abcdef01234567",
            git_status="",
        )
    except ValueError as exc:
        assert "ended before total_count" in str(exc)
    else:
        raise AssertionError("finalize accepted an empty artifact page before total_count")


def test_finalize_crg3_rejects_artifact_page_over_per_page_limit(tmp_path: Path) -> None:
    roadmap = tmp_path / "roadmap.md"
    roadmap.write_text(
        "| ID | Pass | Priority | Status | Work item | Files | Proof required |\n"
        "| --- | --- | --- | --- | --- | --- | --- |\n"
        "| CRG3 | CI, Regression, and Production Readiness Gate | P0 | In progress | Add workflow | `.github/workflows/production-readiness.yml` | Awaiting workflow evidence. |\n",
        encoding="utf-8",
    )

    def fetch(url: str):
        if url.endswith("/actions/runs/123456789"):
            return {
                "url": "https://api.github.com/repos/chainsyncstore/hypothesis-research-engine/actions/runs/123456789",
                "id": 123456789,
                "repository": {"full_name": "chainsyncstore/hypothesis-research-engine"},
                "html_url": "https://github.com/chainsyncstore/hypothesis-research-engine/actions/runs/123456789",
                "name": "Production Readiness",
                "path": ".github/workflows/production-readiness.yml",
                "event": "workflow_dispatch",
                "head_branch": "main",
                "status": "completed",
                "conclusion": "success",
                "run_attempt": 1,
                "head_sha": "0123456789abcdef0123456789abcdef01234567",
                "artifacts_url": "https://api.github.com/repos/chainsyncstore/hypothesis-research-engine/actions/runs/123456789/artifacts",
            }
        if url.endswith("/actions/runs/123456789/artifacts?per_page=100&page=1"):
            return {
                "total_count": MAX_GITHUB_ARTIFACTS_PER_PAGE + 1,
                "artifacts": [
                    {"name": f"other-artifact-{index}", "expired": False}
                    for index in range(MAX_GITHUB_ARTIFACTS_PER_PAGE + 1)
                ],
            }
        raise AssertionError(f"unexpected GitHub API URL: {url}")

    try:
        finalize(
            run_url="https://github.com/chainsyncstore/hypothesis-research-engine/actions/runs/123456789",
            commit_sha="0123456789abcdef0123456789abcdef01234567",
            sbom_artifact="production-sbom",
            verified_date=date(2026, 6, 4),
            roadmap_path=roadmap,
            allowed_repos={"chainsyncstore/hypothesis-research-engine"},
            github_api_fetcher=fetch,
            github_artifact_fetcher=_artifact_fetcher(),
            local_head="0123456789abcdef0123456789abcdef01234567",
            git_status="",
        )
    except ValueError as exc:
        assert "too many entries" in str(exc)
    else:
        raise AssertionError("finalize accepted an artifact page larger than per_page=100")


def test_finalize_crg3_rejects_non_object_artifact_entries(tmp_path: Path) -> None:
    roadmap = tmp_path / "roadmap.md"
    roadmap.write_text(
        "| ID | Pass | Priority | Status | Work item | Files | Proof required |\n"
        "| --- | --- | --- | --- | --- | --- | --- |\n"
        "| CRG3 | CI, Regression, and Production Readiness Gate | P0 | In progress | Add workflow | `.github/workflows/production-readiness.yml` | Awaiting workflow evidence. |\n",
        encoding="utf-8",
    )

    def fetch(url: str):
        if url.endswith("/actions/runs/123456789"):
            return {
                "url": "https://api.github.com/repos/chainsyncstore/hypothesis-research-engine/actions/runs/123456789",
                "id": 123456789,
                "repository": {"full_name": "chainsyncstore/hypothesis-research-engine"},
                "html_url": "https://github.com/chainsyncstore/hypothesis-research-engine/actions/runs/123456789",
                "name": "Production Readiness",
                "path": ".github/workflows/production-readiness.yml",
                "event": "workflow_dispatch",
                "head_branch": "main",
                "status": "completed",
                "conclusion": "success",
                "run_attempt": 1,
                "head_sha": "0123456789abcdef0123456789abcdef01234567",
                "artifacts_url": "https://api.github.com/repos/chainsyncstore/hypothesis-research-engine/actions/runs/123456789/artifacts",
            }
        if url.endswith("/actions/runs/123456789/artifacts?per_page=100&page=1"):
            return {
                "total_count": 1,
                "artifacts": ["not-an-artifact-object"],
            }
        raise AssertionError(f"unexpected GitHub API URL: {url}")

    try:
        finalize(
            run_url="https://github.com/chainsyncstore/hypothesis-research-engine/actions/runs/123456789",
            commit_sha="0123456789abcdef0123456789abcdef01234567",
            sbom_artifact="production-sbom",
            verified_date=date(2026, 6, 4),
            roadmap_path=roadmap,
            allowed_repos={"chainsyncstore/hypothesis-research-engine"},
            github_api_fetcher=fetch,
            github_artifact_fetcher=_artifact_fetcher(),
            local_head="0123456789abcdef0123456789abcdef01234567",
            git_status="",
        )
    except ValueError as exc:
        assert "entries must be objects" in str(exc)
    else:
        raise AssertionError("finalize accepted a non-object GitHub artifact entry")


def test_finalize_crg3_finds_sbom_artifact_on_later_github_page(tmp_path: Path) -> None:
    roadmap = tmp_path / "roadmap.md"
    roadmap.write_text(
        "| ID | Pass | Priority | Status | Work item | Files | Proof required |\n"
        "| --- | --- | --- | --- | --- | --- | --- |\n"
        "| CRG3 | CI, Regression, and Production Readiness Gate | P0 | In progress | Add workflow | `.github/workflows/production-readiness.yml` | Awaiting workflow evidence. |\n",
        encoding="utf-8",
    )
    sha = "0123456789abcdef0123456789abcdef01234567"
    fetched_urls: list[str] = []

    def fetch(url: str):
        fetched_urls.append(url)
        if url.endswith("/actions/runs/123456789"):
            return {
                "url": "https://api.github.com/repos/chainsyncstore/hypothesis-research-engine/actions/runs/123456789",
                "id": 123456789,
                "repository": {"full_name": "chainsyncstore/hypothesis-research-engine"},
                "html_url": "https://github.com/chainsyncstore/hypothesis-research-engine/actions/runs/123456789",
                "name": "Production Readiness",
                "path": ".github/workflows/production-readiness.yml",
                "event": "workflow_dispatch",
                "head_branch": "main",
                "status": "completed",
                "conclusion": "success",
                "run_attempt": 1,
                "head_sha": sha,
                "artifacts_url": "https://api.github.com/repos/chainsyncstore/hypothesis-research-engine/actions/runs/123456789/artifacts",
            }
        if url.endswith("/actions/runs/123456789/artifacts?per_page=100&page=1"):
            return {
                "total_count": 101,
                "artifacts": [
                    {"name": f"other-artifact-{index}", "expired": False}
                    for index in range(100)
                ],
            }
        if url.endswith("/actions/runs/123456789/artifacts?per_page=100&page=2"):
            return {
                "total_count": 101,
                "artifacts": [
                    {
                        "name": "production-sbom",
                        "expired": False,
                        "id": 42,
                        "size_in_bytes": 2048,
                        "archive_download_url": "https://api.github.com/repos/chainsyncstore/hypothesis-research-engine/actions/artifacts/42/zip",
                    },
                ],
            }
        raise AssertionError(f"unexpected GitHub API URL: {url}")

    finalize(
        run_url="https://github.com/chainsyncstore/hypothesis-research-engine/actions/runs/123456789",
        commit_sha=sha,
        sbom_artifact="production-sbom",
        verified_date=date(2026, 6, 4),
        roadmap_path=roadmap,
        allowed_repos={"chainsyncstore/hypothesis-research-engine"},
        github_api_fetcher=fetch,
        github_artifact_fetcher=_artifact_fetcher(),
        local_head=sha,
        git_status="",
    )

    assert any(url.endswith("per_page=100&page=1") for url in fetched_urls)
    assert any(url.endswith("per_page=100&page=2") for url in fetched_urls)
    assert "Verified 2026-06-04" in roadmap.read_text(encoding="utf-8")


def test_finalize_crg3_rejects_artifact_pagination_past_limit(tmp_path: Path) -> None:
    roadmap = tmp_path / "roadmap.md"
    roadmap.write_text(
        "| ID | Pass | Priority | Status | Work item | Files | Proof required |\n"
        "| --- | --- | --- | --- | --- | --- | --- |\n"
        "| CRG3 | CI, Regression, and Production Readiness Gate | P0 | In progress | Add workflow | `.github/workflows/production-readiness.yml` | Awaiting workflow evidence. |\n",
        encoding="utf-8",
    )
    sha = "0123456789abcdef0123456789abcdef01234567"
    fetched_urls: list[str] = []

    def fetch(url: str):
        fetched_urls.append(url)
        if url.endswith("/actions/runs/123456789"):
            return {
                "url": "https://api.github.com/repos/chainsyncstore/hypothesis-research-engine/actions/runs/123456789",
                "id": 123456789,
                "repository": {"full_name": "chainsyncstore/hypothesis-research-engine"},
                "html_url": "https://github.com/chainsyncstore/hypothesis-research-engine/actions/runs/123456789",
                "name": "Production Readiness",
                "path": ".github/workflows/production-readiness.yml",
                "event": "workflow_dispatch",
                "head_branch": "main",
                "status": "completed",
                "conclusion": "success",
                "run_attempt": 1,
                "head_sha": sha,
                "artifacts_url": "https://api.github.com/repos/chainsyncstore/hypothesis-research-engine/actions/runs/123456789/artifacts",
            }
        if "/actions/runs/123456789/artifacts?" in url:
            return {
                "total_count": (MAX_GITHUB_ARTIFACT_PAGES * 100) + 1,
                "artifacts": [
                    {"name": f"other-artifact-{index}", "expired": False}
                    for index in range(100)
                ],
            }
        raise AssertionError(f"unexpected GitHub API URL: {url}")

    try:
        finalize(
            run_url="https://github.com/chainsyncstore/hypothesis-research-engine/actions/runs/123456789",
            commit_sha=sha,
            sbom_artifact="production-sbom",
            verified_date=date(2026, 6, 4),
            roadmap_path=roadmap,
            allowed_repos={"chainsyncstore/hypothesis-research-engine"},
            github_api_fetcher=fetch,
            github_artifact_fetcher=_artifact_fetcher(),
            local_head=sha,
            git_status="",
        )
    except ValueError as exc:
        assert "page limit" in str(exc)
    else:
        raise AssertionError("finalize chased artifact pagination past the CRG3 page limit")

    assert any(f"page={MAX_GITHUB_ARTIFACT_PAGES}" in url for url in fetched_urls)
    assert not any(f"page={MAX_GITHUB_ARTIFACT_PAGES + 1}" in url for url in fetched_urls)


def test_finalize_crg3_rejects_malformed_sbom_artifact_payload(tmp_path: Path) -> None:
    roadmap = tmp_path / "roadmap.md"
    roadmap.write_text(
        "| ID | Pass | Priority | Status | Work item | Files | Proof required |\n"
        "| --- | --- | --- | --- | --- | --- | --- |\n"
        "| CRG3 | CI, Regression, and Production Readiness Gate | P0 | In progress | Add workflow | `.github/workflows/production-readiness.yml` | Awaiting workflow evidence. |\n",
        encoding="utf-8",
    )

    try:
        finalize(
            run_url="https://github.com/chainsyncstore/hypothesis-research-engine/actions/runs/123456789",
            commit_sha="0123456789abcdef0123456789abcdef01234567",
            sbom_artifact="production-sbom",
            verified_date=date(2026, 6, 4),
            roadmap_path=roadmap,
            allowed_repos={"chainsyncstore/hypothesis-research-engine"},
            github_api_fetcher=_github_fetcher(),
            github_artifact_fetcher=_artifact_fetcher(_sbom_zip(bom_format="not-cyclonedx")),
            local_head="0123456789abcdef0123456789abcdef01234567",
            git_status="",
        )
    except ValueError as exc:
        assert "CycloneDX" in str(exc)
    else:
        raise AssertionError("finalize accepted a malformed production-sbom artifact payload")


def test_finalize_crg3_rejects_cross_repo_sbom_artifact_download_url(tmp_path: Path) -> None:
    roadmap = tmp_path / "roadmap.md"
    roadmap.write_text(
        "| ID | Pass | Priority | Status | Work item | Files | Proof required |\n"
        "| --- | --- | --- | --- | --- | --- | --- |\n"
        "| CRG3 | CI, Regression, and Production Readiness Gate | P0 | In progress | Add workflow | `.github/workflows/production-readiness.yml` | Awaiting workflow evidence. |\n",
        encoding="utf-8",
    )

    try:
        finalize(
            run_url="https://github.com/chainsyncstore/hypothesis-research-engine/actions/runs/123456789",
            commit_sha="0123456789abcdef0123456789abcdef01234567",
            sbom_artifact="production-sbom",
            verified_date=date(2026, 6, 4),
            roadmap_path=roadmap,
            allowed_repos={"chainsyncstore/hypothesis-research-engine"},
            github_api_fetcher=_github_fetcher(
                archive_download_url="https://api.github.com/repos/attacker/not-this/actions/artifacts/42/zip"
            ),
            github_artifact_fetcher=_artifact_fetcher(),
            local_head="0123456789abcdef0123456789abcdef01234567",
            git_status="",
        )
    except ValueError as exc:
        assert "workflow repository" in str(exc)
    else:
        raise AssertionError("finalize accepted a cross-repo production-sbom artifact download URL")


def test_finalize_crg3_rejects_sbom_artifact_with_incomplete_components(
    tmp_path: Path,
) -> None:
    roadmap = tmp_path / "roadmap.md"
    roadmap.write_text(
        "| ID | Pass | Priority | Status | Work item | Files | Proof required |\n"
        "| --- | --- | --- | --- | --- | --- | --- |\n"
        "| CRG3 | CI, Regression, and Production Readiness Gate | P0 | In progress | Add workflow | `.github/workflows/production-readiness.yml` | Awaiting workflow evidence. |\n",
        encoding="utf-8",
    )

    try:
        finalize(
            run_url="https://github.com/chainsyncstore/hypothesis-research-engine/actions/runs/123456789",
            commit_sha="0123456789abcdef0123456789abcdef01234567",
            sbom_artifact="production-sbom",
            verified_date=date(2026, 6, 4),
            roadmap_path=roadmap,
            allowed_repos={"chainsyncstore/hypothesis-research-engine"},
            github_api_fetcher=_github_fetcher(),
            github_artifact_fetcher=_artifact_fetcher(_sbom_zip(components=[{"name": "requests"}])),
            local_head="0123456789abcdef0123456789abcdef01234567",
            git_status="",
        )
    except ValueError as exc:
        assert "type, name, version, and purl" in str(exc)
    else:
        raise AssertionError("finalize accepted an incomplete production-sbom component")


def test_finalize_crg3_rejects_sbom_artifact_missing_locked_component(
    tmp_path: Path,
) -> None:
    roadmap = tmp_path / "roadmap.md"
    roadmap.write_text(
        "| ID | Pass | Priority | Status | Work item | Files | Proof required |\n"
        "| --- | --- | --- | --- | --- | --- | --- |\n"
        "| CRG3 | CI, Regression, and Production Readiness Gate | P0 | In progress | Add workflow | `.github/workflows/production-readiness.yml` | Awaiting workflow evidence. |\n",
        encoding="utf-8",
    )
    components = _expected_runtime_sbom_components()

    with pytest.raises(ValueError, match="requirements\\.lock"):
        finalize(
            run_url="https://github.com/chainsyncstore/hypothesis-research-engine/actions/runs/123456789",
            commit_sha="0123456789abcdef0123456789abcdef01234567",
            sbom_artifact="production-sbom",
            verified_date=date(2026, 6, 4),
            roadmap_path=roadmap,
            allowed_repos={"chainsyncstore/hypothesis-research-engine"},
            github_api_fetcher=_github_fetcher(),
            github_artifact_fetcher=_artifact_fetcher(_sbom_zip(components=components[:-1])),
            local_head="0123456789abcdef0123456789abcdef01234567",
            git_status="",
        )


def test_finalize_crg3_rejects_sbom_artifact_with_unlocked_component(
    tmp_path: Path,
) -> None:
    roadmap = tmp_path / "roadmap.md"
    roadmap.write_text(
        "| ID | Pass | Priority | Status | Work item | Files | Proof required |\n"
        "| --- | --- | --- | --- | --- | --- | --- |\n"
        "| CRG3 | CI, Regression, and Production Readiness Gate | P0 | In progress | Add workflow | `.github/workflows/production-readiness.yml` | Awaiting workflow evidence. |\n",
        encoding="utf-8",
    )
    components = [
        *_expected_runtime_sbom_components(),
        {
            "type": "library",
            "name": "unexpected-package",
            "version": "1.0.0",
            "purl": "pkg:pypi/unexpected-package@1.0.0",
        },
    ]

    with pytest.raises(ValueError, match="unexpected components"):
        finalize(
            run_url="https://github.com/chainsyncstore/hypothesis-research-engine/actions/runs/123456789",
            commit_sha="0123456789abcdef0123456789abcdef01234567",
            sbom_artifact="production-sbom",
            verified_date=date(2026, 6, 4),
            roadmap_path=roadmap,
            allowed_repos={"chainsyncstore/hypothesis-research-engine"},
            github_api_fetcher=_github_fetcher(),
            github_artifact_fetcher=_artifact_fetcher(_sbom_zip(components=components)),
            local_head="0123456789abcdef0123456789abcdef01234567",
            git_status="",
        )


def test_finalize_crg3_rejects_sbom_artifact_with_wrong_component_type(
    tmp_path: Path,
) -> None:
    roadmap = tmp_path / "roadmap.md"
    roadmap.write_text(
        "| ID | Pass | Priority | Status | Work item | Files | Proof required |\n"
        "| --- | --- | --- | --- | --- | --- | --- |\n"
        "| CRG3 | CI, Regression, and Production Readiness Gate | P0 | In progress | Add workflow | `.github/workflows/production-readiness.yml` | Awaiting workflow evidence. |\n",
        encoding="utf-8",
    )
    components = _expected_runtime_sbom_components()
    components[0] = {**components[0], "type": "application"}

    with pytest.raises(ValueError, match="requirements\\.lock"):
        finalize(
            run_url="https://github.com/chainsyncstore/hypothesis-research-engine/actions/runs/123456789",
            commit_sha="0123456789abcdef0123456789abcdef01234567",
            sbom_artifact="production-sbom",
            verified_date=date(2026, 6, 4),
            roadmap_path=roadmap,
            allowed_repos={"chainsyncstore/hypothesis-research-engine"},
            github_api_fetcher=_github_fetcher(),
            github_artifact_fetcher=_artifact_fetcher(_sbom_zip(components=components)),
            local_head="0123456789abcdef0123456789abcdef01234567",
            git_status="",
        )


def test_finalize_crg3_rejects_sbom_artifact_with_duplicate_component_identity(
    tmp_path: Path,
) -> None:
    roadmap = tmp_path / "roadmap.md"
    roadmap.write_text(
        "| ID | Pass | Priority | Status | Work item | Files | Proof required |\n"
        "| --- | --- | --- | --- | --- | --- | --- |\n"
        "| CRG3 | CI, Regression, and Production Readiness Gate | P0 | In progress | Add workflow | `.github/workflows/production-readiness.yml` | Awaiting workflow evidence. |\n",
        encoding="utf-8",
    )
    components = _expected_runtime_sbom_components()
    components.append(dict(components[0]))

    with pytest.raises(ValueError, match="components must be unique"):
        finalize(
            run_url="https://github.com/chainsyncstore/hypothesis-research-engine/actions/runs/123456789",
            commit_sha="0123456789abcdef0123456789abcdef01234567",
            sbom_artifact="production-sbom",
            verified_date=date(2026, 6, 4),
            roadmap_path=roadmap,
            allowed_repos={"chainsyncstore/hypothesis-research-engine"},
            github_api_fetcher=_github_fetcher(),
            github_artifact_fetcher=_artifact_fetcher(_sbom_zip(components=components)),
            local_head="0123456789abcdef0123456789abcdef01234567",
            git_status="",
        )


def test_finalize_crg3_rejects_oversized_sbom_artifact_download(tmp_path: Path) -> None:
    roadmap = tmp_path / "roadmap.md"
    roadmap.write_text(
        "| ID | Pass | Priority | Status | Work item | Files | Proof required |\n"
        "| --- | --- | --- | --- | --- | --- | --- |\n"
        "| CRG3 | CI, Regression, and Production Readiness Gate | P0 | In progress | Add workflow | `.github/workflows/production-readiness.yml` | Awaiting workflow evidence. |\n",
        encoding="utf-8",
    )

    try:
        finalize(
            run_url="https://github.com/chainsyncstore/hypothesis-research-engine/actions/runs/123456789",
            commit_sha="0123456789abcdef0123456789abcdef01234567",
            sbom_artifact="production-sbom",
            verified_date=date(2026, 6, 4),
            roadmap_path=roadmap,
            allowed_repos={"chainsyncstore/hypothesis-research-engine"},
            github_api_fetcher=_github_fetcher(),
            github_artifact_fetcher=_artifact_fetcher(b"x" * (MAX_SBOM_ARTIFACT_BYTES + 1)),
            local_head="0123456789abcdef0123456789abcdef01234567",
            git_status="",
        )
    except ValueError as exc:
        assert "too large" in str(exc)
    else:
        raise AssertionError("finalize accepted an oversized production-sbom artifact")


def test_finalize_crg3_rejects_oversized_sbom_json_payload(tmp_path: Path) -> None:
    roadmap = tmp_path / "roadmap.md"
    roadmap.write_text(
        "| ID | Pass | Priority | Status | Work item | Files | Proof required |\n"
        "| --- | --- | --- | --- | --- | --- | --- |\n"
        "| CRG3 | CI, Regression, and Production Readiness Gate | P0 | In progress | Add workflow | `.github/workflows/production-readiness.yml` | Awaiting workflow evidence. |\n",
        encoding="utf-8",
    )

    try:
        finalize(
            run_url="https://github.com/chainsyncstore/hypothesis-research-engine/actions/runs/123456789",
            commit_sha="0123456789abcdef0123456789abcdef01234567",
            sbom_artifact="production-sbom",
            verified_date=date(2026, 6, 4),
            roadmap_path=roadmap,
            allowed_repos={"chainsyncstore/hypothesis-research-engine"},
            github_api_fetcher=_github_fetcher(),
            github_artifact_fetcher=_artifact_fetcher(_large_sbom_zip()),
            local_head="0123456789abcdef0123456789abcdef01234567",
            git_status="",
        )
    except ValueError as exc:
        assert "too large" in str(exc)
    else:
        raise AssertionError("finalize accepted an oversized sbom.cdx.json payload")


def test_finalize_crg3_rejects_non_utf8_sbom_json_payload(tmp_path: Path) -> None:
    roadmap = tmp_path / "roadmap.md"
    roadmap.write_text(
        "| ID | Pass | Priority | Status | Work item | Files | Proof required |\n"
        "| --- | --- | --- | --- | --- | --- | --- |\n"
        "| CRG3 | CI, Regression, and Production Readiness Gate | P0 | In progress | Add workflow | `.github/workflows/production-readiness.yml` | Awaiting workflow evidence. |\n",
        encoding="utf-8",
    )

    try:
        finalize(
            run_url="https://github.com/chainsyncstore/hypothesis-research-engine/actions/runs/123456789",
            commit_sha="0123456789abcdef0123456789abcdef01234567",
            sbom_artifact="production-sbom",
            verified_date=date(2026, 6, 4),
            roadmap_path=roadmap,
            allowed_repos={"chainsyncstore/hypothesis-research-engine"},
            github_api_fetcher=_github_fetcher(),
            github_artifact_fetcher=_artifact_fetcher(_non_utf8_sbom_zip()),
            local_head="0123456789abcdef0123456789abcdef01234567",
            git_status="",
        )
    except ValueError as exc:
        assert "UTF-8" in str(exc)
    else:
        raise AssertionError("finalize accepted a non-UTF-8 sbom.cdx.json payload")


def test_finalize_crg3_rejects_multiple_sbom_json_payloads(tmp_path: Path) -> None:
    roadmap = tmp_path / "roadmap.md"
    roadmap.write_text(
        "| ID | Pass | Priority | Status | Work item | Files | Proof required |\n"
        "| --- | --- | --- | --- | --- | --- | --- |\n"
        "| CRG3 | CI, Regression, and Production Readiness Gate | P0 | In progress | Add workflow | `.github/workflows/production-readiness.yml` | Awaiting workflow evidence. |\n",
        encoding="utf-8",
    )

    try:
        finalize(
            run_url="https://github.com/chainsyncstore/hypothesis-research-engine/actions/runs/123456789",
            commit_sha="0123456789abcdef0123456789abcdef01234567",
            sbom_artifact="production-sbom",
            verified_date=date(2026, 6, 4),
            roadmap_path=roadmap,
            allowed_repos={"chainsyncstore/hypothesis-research-engine"},
            github_api_fetcher=_github_fetcher(),
            github_artifact_fetcher=_artifact_fetcher(_multiple_sbom_zip()),
            local_head="0123456789abcdef0123456789abcdef01234567",
            git_status="",
        )
    except ValueError as exc:
        assert "exactly one" in str(exc)
    else:
        raise AssertionError("finalize accepted multiple sbom.cdx.json payloads")


def test_finalize_crg3_rejects_sbom_artifact_with_too_many_zip_members(tmp_path: Path) -> None:
    roadmap = tmp_path / "roadmap.md"
    roadmap.write_text(
        "| ID | Pass | Priority | Status | Work item | Files | Proof required |\n"
        "| --- | --- | --- | --- | --- | --- | --- |\n"
        "| CRG3 | CI, Regression, and Production Readiness Gate | P0 | In progress | Add workflow | `.github/workflows/production-readiness.yml` | Awaiting workflow evidence. |\n",
        encoding="utf-8",
    )

    try:
        finalize(
            run_url="https://github.com/chainsyncstore/hypothesis-research-engine/actions/runs/123456789",
            commit_sha="0123456789abcdef0123456789abcdef01234567",
            sbom_artifact="production-sbom",
            verified_date=date(2026, 6, 4),
            roadmap_path=roadmap,
            allowed_repos={"chainsyncstore/hypothesis-research-engine"},
            github_api_fetcher=_github_fetcher(),
            github_artifact_fetcher=_artifact_fetcher(_many_member_sbom_zip()),
            local_head="0123456789abcdef0123456789abcdef01234567",
            git_status="",
        )
    except ValueError as exc:
        assert "too many zip members" in str(exc)
    else:
        raise AssertionError("finalize accepted a production-sbom artifact with too many zip members")


def test_finalize_crg3_rejects_local_head_mismatch(tmp_path: Path) -> None:
    roadmap = tmp_path / "roadmap.md"
    roadmap.write_text(
        "| ID | Pass | Priority | Status | Work item | Files | Proof required |\n"
        "| --- | --- | --- | --- | --- | --- | --- |\n"
        "| CRG3 | CI, Regression, and Production Readiness Gate | P0 | In progress | Add workflow | `.github/workflows/production-readiness.yml` | Awaiting workflow evidence. |\n",
        encoding="utf-8",
    )

    try:
        finalize(
            run_url="https://github.com/chainsyncstore/hypothesis-research-engine/actions/runs/123456789",
            commit_sha="0123456789abcdef0123456789abcdef01234567",
            sbom_artifact="production-sbom",
            verified_date=date(2026, 6, 4),
            roadmap_path=roadmap,
            allowed_repos={"chainsyncstore/hypothesis-research-engine"},
            github_api_fetcher=_github_fetcher(),
            local_head="fedcba9876543210fedcba9876543210fedcba98",
        )
    except ValueError as exc:
        assert "local git HEAD" in str(exc)
    else:
        raise AssertionError("finalize accepted a mismatched local checkout HEAD")


def test_finalize_crg3_rejects_dirty_working_tree(tmp_path: Path) -> None:
    roadmap = tmp_path / "roadmap.md"
    roadmap.write_text(
        "| ID | Pass | Priority | Status | Work item | Files | Proof required |\n"
        "| --- | --- | --- | --- | --- | --- | --- |\n"
        "| CRG3 | CI, Regression, and Production Readiness Gate | P0 | In progress | Add workflow | `.github/workflows/production-readiness.yml` | Awaiting workflow evidence. |\n",
        encoding="utf-8",
    )

    try:
        finalize(
            run_url="https://github.com/chainsyncstore/hypothesis-research-engine/actions/runs/123456789",
            commit_sha="0123456789abcdef0123456789abcdef01234567",
            sbom_artifact="production-sbom",
            verified_date=date(2026, 6, 4),
            roadmap_path=roadmap,
            allowed_repos={"chainsyncstore/hypothesis-research-engine"},
            github_api_fetcher=_github_fetcher(),
            local_head="0123456789abcdef0123456789abcdef01234567",
            git_status=" M unreviewed.py\n",
        )
    except ValueError as exc:
        assert "not clean" in str(exc)
    else:
        raise AssertionError("finalize accepted a dirty working tree")


def test_finalize_crg3_requires_local_workflow_live_disabled_env(tmp_path: Path) -> None:
    workflow = tmp_path / "production-readiness.yml"
    workflow.write_text(
        _workflow_text().replace(
            "    env:\n      BOT_V2_ALLOW_LIVE_EXECUTION: \"0\"\n",
            "",
        ),
        encoding="utf-8",
    )

    try:
        _validate_local_workflow_live_disabled(workflow)
    except ValueError as exc:
        assert "readiness job must set" in str(exc)
    else:
        raise AssertionError("CRG3 finalizer accepted a workflow without live-disabled CI env")


def test_finalize_crg3_rejects_comment_only_live_disabled_env(tmp_path: Path) -> None:
    workflow = tmp_path / "production-readiness.yml"
    workflow.write_text(
        _workflow_text().replace(
            "    env:\n      BOT_V2_ALLOW_LIVE_EXECUTION: \"0\"\n",
            "    # BOT_V2_ALLOW_LIVE_EXECUTION: \"0\"\n",
        ),
        encoding="utf-8",
    )

    try:
        _validate_local_workflow_live_disabled(workflow)
    except ValueError as exc:
        assert "readiness job must set" in str(exc)
    else:
        raise AssertionError("CRG3 finalizer accepted a comment-only live-disabled env")


def test_finalize_crg3_rejects_duplicate_workflow_yaml_keys(tmp_path: Path) -> None:
    workflow = tmp_path / "production-readiness.yml"
    workflow.write_text(
        _workflow_text().replace(
            "    env:\n      BOT_V2_ALLOW_LIVE_EXECUTION: \"0\"\n",
            (
                "    env:\n"
                "      BOT_V2_ALLOW_LIVE_EXECUTION: \"1\"\n"
                "    env:\n"
                "      BOT_V2_ALLOW_LIVE_EXECUTION: \"0\"\n"
            ),
        ),
        encoding="utf-8",
    )

    try:
        _validate_local_workflow_live_disabled(workflow)
    except ValueError as exc:
        assert "YAML is invalid" in str(exc)
    else:
        raise AssertionError("CRG3 finalizer accepted duplicate workflow YAML keys")


def test_finalize_crg3_requires_string_zero_live_disabled_env(tmp_path: Path) -> None:
    workflow = tmp_path / "production-readiness.yml"
    workflow.write_text(
        _workflow_text().replace(
            "      BOT_V2_ALLOW_LIVE_EXECUTION: \"0\"\n",
            "      BOT_V2_ALLOW_LIVE_EXECUTION: 0\n",
        ),
        encoding="utf-8",
    )

    try:
        _validate_local_workflow_live_disabled(workflow)
    except ValueError as exc:
        assert "string '0'" in str(exc)
    else:
        raise AssertionError("CRG3 finalizer accepted a non-string live-disabled env")


def test_finalize_crg3_requires_job_level_live_disabled_env_not_step_only(
    tmp_path: Path,
) -> None:
    workflow = tmp_path / "production-readiness.yml"
    workflow.write_text(
        _workflow_text(wrapper_env_override="0").replace(
            "    env:\n      BOT_V2_ALLOW_LIVE_EXECUTION: \"0\"\n",
            "",
        ),
        encoding="utf-8",
    )

    try:
        _validate_local_workflow_live_disabled(workflow)
    except ValueError as exc:
        assert "readiness job must set" in str(exc)
    else:
        raise AssertionError(
            "CRG3 finalizer accepted a workflow with live-disable env only on wrapper step"
        )


def test_finalize_crg3_rejects_wrapper_step_live_env_override(tmp_path: Path) -> None:
    workflow = tmp_path / "production-readiness.yml"
    workflow.write_text(_workflow_text(wrapper_env_override="1"), encoding="utf-8")

    try:
        _validate_local_workflow_live_disabled(workflow)
    except ValueError as exc:
        assert "must not enable live execution" in str(exc)
    else:
        raise AssertionError("CRG3 finalizer accepted a wrapper step live env override")


def test_finalize_crg3_rejects_wrapper_command_drift(tmp_path: Path) -> None:
    workflow = tmp_path / "production-readiness.yml"
    workflow.write_text(
        _workflow_text(
            wrapper_command="python tools/security/production_readiness.py --profile local --skip-docker"
        ),
        encoding="utf-8",
    )

    try:
        _validate_local_workflow_live_disabled(workflow)
    except ValueError as exc:
        assert "wrapper step must run CI profile" in str(exc)
    else:
        raise AssertionError("CRG3 finalizer accepted wrapper command drift")


def test_finalize_crg3_requires_checkout_without_persisted_credentials(tmp_path: Path) -> None:
    workflow = tmp_path / "production-readiness.yml"
    workflow.write_text(_workflow_text(persist_credentials="true"), encoding="utf-8")

    try:
        _validate_local_workflow_live_disabled(workflow)
    except ValueError as exc:
        assert "persist-credentials false" in str(exc)
    else:
        raise AssertionError("CRG3 finalizer accepted persisted checkout credentials")


def test_finalize_crg3_rejects_floating_setup_python_action(tmp_path: Path) -> None:
    workflow = tmp_path / "production-readiness.yml"
    workflow.write_text(_workflow_text(setup_uses="actions/setup-python@v5"), encoding="utf-8")

    try:
        _validate_local_workflow_live_disabled(workflow)
    except ValueError as exc:
        assert "setup-python action must be pinned" in str(exc)
    else:
        raise AssertionError("CRG3 finalizer accepted a floating setup-python action")


def test_finalize_crg3_requires_hashed_runtime_install(tmp_path: Path) -> None:
    workflow = tmp_path / "production-readiness.yml"
    workflow.write_text(
        _workflow_text(runtime_install="python -m pip install -r requirements.lock"),
        encoding="utf-8",
    )

    try:
        _validate_local_workflow_live_disabled(workflow)
    except ValueError as exc:
        assert "hashed runtime dependencies" in str(exc)
    else:
        raise AssertionError("CRG3 finalizer accepted unhashed runtime dependency install")


def test_finalize_crg3_requires_ci_tools_before_runtime_dependencies(tmp_path: Path) -> None:
    workflow = tmp_path / "production-readiness.yml"
    workflow.write_text(_workflow_text(install_order="runtime-first"), encoding="utf-8")

    try:
        _validate_local_workflow_live_disabled(workflow)
    except ValueError as exc:
        assert "CI tools must install before runtime dependencies" in str(exc)
    else:
        raise AssertionError("CRG3 finalizer accepted runtime install before CI tools")


def test_finalize_crg3_requires_release_branch_push_triggers(tmp_path: Path) -> None:
    workflow = tmp_path / "production-readiness.yml"
    workflow.write_text(_workflow_text(push_branches="- main"), encoding="utf-8")

    try:
        _validate_local_workflow_live_disabled(workflow)
    except ValueError as exc:
        assert "push branches must be main and master" in str(exc)
    else:
        raise AssertionError("CRG3 finalizer accepted incomplete release push branches")


def test_finalize_crg3_requires_read_only_workflow_permissions(tmp_path: Path) -> None:
    workflow = tmp_path / "production-readiness.yml"
    workflow.write_text(
        _workflow_text(permissions="contents: write\n  actions: read"),
        encoding="utf-8",
    )

    try:
        _validate_local_workflow_live_disabled(workflow)
    except ValueError as exc:
        assert "permissions must be contents: read only" in str(exc)
    else:
        raise AssertionError("CRG3 finalizer accepted broad workflow permissions")


def test_finalize_crg3_requires_non_canceling_same_ref_concurrency(tmp_path: Path) -> None:
    workflow = tmp_path / "production-readiness.yml"
    workflow.write_text(_workflow_text(cancel_in_progress="true"), encoding="utf-8")

    try:
        _validate_local_workflow_live_disabled(workflow)
    except ValueError as exc:
        assert "must not cancel in-progress evidence" in str(exc)
    else:
        raise AssertionError("CRG3 finalizer accepted canceling workflow concurrency")


def test_finalize_crg3_requires_45_minute_timeout(tmp_path: Path) -> None:
    workflow = tmp_path / "production-readiness.yml"
    workflow.write_text(_workflow_text(timeout_minutes="5"), encoding="utf-8")

    try:
        _validate_local_workflow_live_disabled(workflow)
    except ValueError as exc:
        assert "timeout must be 45 minutes" in str(exc)
    else:
        raise AssertionError("CRG3 finalizer accepted a short readiness timeout")


def test_finalize_crg3_requires_structured_sbom_upload_step(tmp_path: Path) -> None:
    workflow = tmp_path / "production-readiness.yml"
    workflow.write_text(_workflow_text(include_upload=False), encoding="utf-8")

    try:
        _validate_local_workflow_live_disabled(workflow)
    except ValueError as exc:
        assert "SBOM upload step" in str(exc)
    else:
        raise AssertionError("CRG3 finalizer accepted a workflow without SBOM upload")


def test_finalize_crg3_requires_production_sbom_artifact_name(tmp_path: Path) -> None:
    workflow = tmp_path / "production-readiness.yml"
    workflow.write_text(_workflow_text(upload_name="wrong-sbom"), encoding="utf-8")

    try:
        _validate_local_workflow_live_disabled(workflow)
    except ValueError as exc:
        assert "artifact name must be production-sbom" in str(exc)
    else:
        raise AssertionError("CRG3 finalizer accepted a wrong SBOM artifact name")


def test_finalize_crg3_requires_expected_sbom_artifact_path(tmp_path: Path) -> None:
    workflow = tmp_path / "production-readiness.yml"
    workflow.write_text(_workflow_text(upload_path="build/security/other.json"), encoding="utf-8")

    try:
        _validate_local_workflow_live_disabled(workflow)
    except ValueError as exc:
        assert "artifact path is invalid" in str(exc)
    else:
        raise AssertionError("CRG3 finalizer accepted a wrong SBOM artifact path")


def test_finalize_crg3_requires_sbom_upload_missing_file_failure(tmp_path: Path) -> None:
    workflow = tmp_path / "production-readiness.yml"
    workflow.write_text(_workflow_text(upload_missing="warn"), encoding="utf-8")

    try:
        _validate_local_workflow_live_disabled(workflow)
    except ValueError as exc:
        assert "fail when missing" in str(exc)
    else:
        raise AssertionError("CRG3 finalizer accepted a non-failing missing SBOM upload")


def test_finalize_crg3_atomic_write_replaces_existing_file(tmp_path: Path) -> None:
    target = tmp_path / "roadmap.md"
    target.write_text("old roadmap\n", encoding="utf-8")

    _atomic_write_text(target, "new roadmap\n")

    assert target.read_text(encoding="utf-8") == "new roadmap\n"
    assert list(tmp_path.glob(".roadmap.md.*.tmp")) == []


def test_finalize_crg3_atomic_write_cleans_temp_file_on_replace_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    target = tmp_path / "roadmap.md"
    target.write_text("old roadmap\n", encoding="utf-8")

    def fail_replace(src, dst):
        raise OSError("simulated replace failure")

    monkeypatch.setattr("tools.security.finalize_crg3.os.replace", fail_replace)

    with pytest.raises(OSError, match="simulated replace failure"):
        _atomic_write_text(target, "new roadmap\n")

    assert target.read_text(encoding="utf-8") == "old roadmap\n"
    assert list(tmp_path.glob(".roadmap.md.*.tmp")) == []


def test_finalize_crg3_rejects_replacing_verified_row_without_explicit_override(
    tmp_path: Path,
) -> None:
    roadmap = tmp_path / "roadmap.md"
    roadmap.write_text(
        "| ID | Pass | Priority | Status | Work item | Files | Proof required |\n"
        "| --- | --- | --- | --- | --- | --- | --- |\n"
        "| CRG3 | CI, Regression, and Production Readiness Gate | P0 | Verified 2026-06-04 | Add workflow | `.github/workflows/production-readiness.yml` | Existing workflow evidence. |\n",
        encoding="utf-8",
    )
    sha = "0123456789abcdef0123456789abcdef01234567"

    try:
        finalize(
            run_url="https://github.com/chainsyncstore/hypothesis-research-engine/actions/runs/123456789",
            commit_sha=sha,
            sbom_artifact="production-sbom",
            verified_date=date(2026, 6, 4),
            roadmap_path=roadmap,
            allowed_repos={"chainsyncstore/hypothesis-research-engine"},
            github_api_fetcher=_github_fetcher(sha=sha),
            github_artifact_fetcher=_artifact_fetcher(),
            local_head=sha,
            git_status="",
        )
    except ValueError as exc:
        assert "already verified" in str(exc)
        assert "--allow-reverify" in str(exc)
    else:
        raise AssertionError("finalize replaced an existing CRG3 verification without override")


def test_finalize_crg3_validates_full_p0_ledger_before_writing(tmp_path: Path) -> None:
    roadmap = tmp_path / "roadmap.md"
    original_text = (
        "| ID | Pass | Priority | Status | Work item | Files | Proof required |\n"
        "| --- | --- | --- | --- | --- | --- | --- |\n"
        "| CRG3 | CI, Regression, and Production Readiness Gate | P0 | In progress | Add workflow | `.github/workflows/production-readiness.yml` | Awaiting workflow evidence. |\n"
        "| CRG9 | CI, Regression, and Production Readiness Gate | P0 | In progress | Other gate | `tests/other.py` | Awaiting other evidence. |\n"
    )
    roadmap.write_text(original_text, encoding="utf-8")
    sha = "0123456789abcdef0123456789abcdef01234567"

    try:
        finalize(
            run_url="https://github.com/chainsyncstore/hypothesis-research-engine/actions/runs/123456789",
            commit_sha=sha,
            sbom_artifact="production-sbom",
            verified_date=date(2026, 6, 4),
            roadmap_path=roadmap,
            allowed_repos={"chainsyncstore/hypothesis-research-engine"},
            github_api_fetcher=_github_fetcher(sha=sha),
            github_artifact_fetcher=_artifact_fetcher(),
            local_head=sha,
            git_status="",
        )
    except ValueError as exc:
        assert "updated roadmap P0 ledger validation failed" in str(exc)
        assert "CRG9" in str(exc)
    else:
        raise AssertionError("finalize wrote roadmap evidence while another P0 row was open")

    assert roadmap.read_text(encoding="utf-8") == original_text


def test_finalize_crg3_dry_run_validates_without_writing(tmp_path: Path) -> None:
    roadmap = tmp_path / "roadmap.md"
    original_text = (
        "| ID | Pass | Priority | Status | Work item | Files | Proof required |\n"
        "| --- | --- | --- | --- | --- | --- | --- |\n"
        "| CRG3 | CI, Regression, and Production Readiness Gate | P0 | In progress | Add workflow | `.github/workflows/production-readiness.yml` | Awaiting workflow evidence. |\n"
    )
    roadmap.write_text(original_text, encoding="utf-8")
    sha = "0123456789abcdef0123456789abcdef01234567"

    finalize(
        run_url="https://github.com/chainsyncstore/hypothesis-research-engine/actions/runs/123456789",
        commit_sha=sha,
        sbom_artifact="production-sbom",
        verified_date=date(2026, 6, 4),
        roadmap_path=roadmap,
        allowed_repos={"chainsyncstore/hypothesis-research-engine"},
        github_api_fetcher=_github_fetcher(sha=sha),
        github_artifact_fetcher=_artifact_fetcher(),
        local_head=sha,
        git_status="",
        dry_run=True,
    )

    assert roadmap.read_text(encoding="utf-8") == original_text


def test_finalize_crg3_writes_machine_readable_evidence_json(tmp_path: Path) -> None:
    roadmap = tmp_path / "roadmap.md"
    roadmap.write_text(
        "| ID | Pass | Priority | Status | Work item | Files | Proof required |\n"
        "| --- | --- | --- | --- | --- | --- | --- |\n"
        "| CRG3 | CI, Regression, and Production Readiness Gate | P0 | In progress | Add workflow | `.github/workflows/production-readiness.yml` | Awaiting workflow evidence. |\n",
        encoding="utf-8",
    )
    evidence_json = tmp_path / "crg3-evidence.json"
    sha = "0123456789abcdef0123456789abcdef01234567"

    finalize(
        run_url="https://github.com/chainsyncstore/hypothesis-research-engine/actions/runs/123456789",
        commit_sha=sha,
        sbom_artifact="production-sbom",
        verified_date=date(2026, 6, 4),
        roadmap_path=roadmap,
        allowed_repos={"chainsyncstore/hypothesis-research-engine"},
        github_api_fetcher=_github_fetcher(sha=sha),
        github_artifact_fetcher=_artifact_fetcher(),
        local_head=sha,
        git_status="",
        evidence_json_path=evidence_json,
    )

    proof = json.loads(evidence_json.read_text(encoding="utf-8"))
    schema = json.loads((REPO_ROOT / "docs" / "CRG3_EVIDENCE_SCHEMA.json").read_text(
        encoding="utf-8"
    ))
    _assert_json_object_matches_schema(proof, schema)
    assert proof["schema_version"] == CRG3_EVIDENCE_SCHEMA_VERSION
    assert proof["schema"] == "docs/CRG3_EVIDENCE_SCHEMA.json"
    assert proof["ledger_id"] == "CRG3"
    assert proof["verified_date"] == "2026-06-04"
    assert proof["repository"] == "chainsyncstore/hypothesis-research-engine"
    assert proof["repository_binding_source"] == "explicit allowed repository set"
    assert proof["run_url"].endswith("/actions/runs/123456789")
    assert proof["run_id"] == "123456789"
    assert proof["run_attempt"] == 1
    assert proof["commit_sha"] == sha
    assert proof["local_head"] == sha
    assert proof["local_working_tree"] == "clean"
    assert proof["runtime_lock"] == _runtime_lock_metadata()
    assert proof["runtime_lock"]["path"] == "requirements.lock"
    assert re.fullmatch(r"^[0-9a-f]{64}$", proof["runtime_lock"]["sha256"])
    assert proof["sbom_artifact"] == "production-sbom"
    assert proof["sbom_artifact_metadata"] == {
        **_default_sbom_artifact_metadata(),
        "download_url": (
            "https://api.github.com/repos/chainsyncstore/"
            "hypothesis-research-engine/actions/artifacts/42/zip"
        ),
        "id": 42,
        "size_in_bytes": 2048,
    }
    assert re.fullmatch(r"^[0-9a-f]{64}$", proof["sbom_artifact_metadata"]["artifact_sha256"])
    assert re.fullmatch(r"^[0-9a-f]{64}$", proof["sbom_artifact_metadata"]["sbom_sha256"])
    assert proof["sbom_artifact_metadata"]["download_size_in_bytes"] == len(_sbom_zip())
    assert proof["sbom_artifact_metadata"]["component_count"] == len(
        _expected_runtime_sbom_components()
    )
    assert proof["workflow"]["name"] == "Production Readiness"
    assert proof["workflow"]["job"] == "readiness"
    assert proof["workflow"]["live_execution_env"] == "0"
    assert proof["workflow"]["wrapper_command"] == (
        "python tools/security/production_readiness.py --profile ci"
    )
    readiness_check_names = {
        check.name for check in production_readiness._checks(allow_open_ledger_ids=["CRG3"])
    }
    assert readiness_check_names <= set(proof["checks"])
    assert "focused P0 regression suite" in proof["checks"]
    assert "production-sbom artifact upload" in proof["checks"]
    assert "CRG3 evidence schema validation" in proof["checks"]
    assert "downloaded CycloneDX SBOM artifact validation" in proof["checks"]
    assert "Verified 2026-06-04" in roadmap.read_text(encoding="utf-8")


def test_finalize_crg3_evidence_check_names_track_readiness_wrapper(tmp_path: Path) -> None:
    roadmap = tmp_path / "roadmap.md"
    roadmap.write_text(
        "| ID | Pass | Priority | Status | Work item | Files | Proof required |\n"
        "| --- | --- | --- | --- | --- | --- | --- |\n"
        "| CRG3 | CI, Regression, and Production Readiness Gate | P0 | In progress | Add workflow | `.github/workflows/production-readiness.yml` | Awaiting workflow evidence. |\n",
        encoding="utf-8",
    )
    evidence_json = tmp_path / "crg3-evidence.json"
    sha = "0123456789abcdef0123456789abcdef01234567"

    finalize(
        run_url="https://github.com/chainsyncstore/hypothesis-research-engine/actions/runs/123456789",
        commit_sha=sha,
        sbom_artifact="production-sbom",
        verified_date=date(2026, 6, 4),
        roadmap_path=roadmap,
        allowed_repos={"chainsyncstore/hypothesis-research-engine"},
        github_api_fetcher=_github_fetcher(sha=sha),
        github_artifact_fetcher=_artifact_fetcher(),
        local_head=sha,
        git_status="",
        evidence_json_path=evidence_json,
    )

    proof = json.loads(evidence_json.read_text(encoding="utf-8"))
    proof_checks = set(proof["checks"])
    readiness_check_names = [
        check.name for check in production_readiness._checks(allow_open_ledger_ids=["CRG3"])
    ]

    for check_name in readiness_check_names:
        assert check_name in proof_checks
        assert proof["checks"].count(check_name) == 1
    assert "production-sbom artifact upload" in proof_checks
    assert "downloaded CycloneDX SBOM artifact validation" in proof_checks


def _valid_crg3_evidence_record() -> dict:
    sha = "0123456789abcdef0123456789abcdef01234567"
    return {
        "checks": _expected_crg3_evidence_checks(),
        "commit_sha": sha,
        "ledger_id": "CRG3",
        "local_head": sha,
        "local_working_tree": "clean",
        "repository": "chainsyncstore/hypothesis-research-engine",
        "repository_binding_source": "explicit allowed repository set",
        "run_attempt": 1,
        "run_id": "123456789",
        "run_url": (
            "https://github.com/chainsyncstore/hypothesis-research-engine/actions/runs/123456789"
        ),
        "runtime_lock": _runtime_lock_metadata(),
        "schema": "docs/CRG3_EVIDENCE_SCHEMA.json",
        "schema_version": CRG3_EVIDENCE_SCHEMA_VERSION,
        "sbom_artifact": "production-sbom",
        "sbom_artifact_metadata": {
            **_default_sbom_artifact_metadata(),
            "download_url": (
                "https://api.github.com/repos/chainsyncstore/"
                "hypothesis-research-engine/actions/artifacts/42/zip"
            ),
            "id": 42,
            "size_in_bytes": 2048,
        },
        "verified_date": "2026-06-04",
        "workflow": {
            "job": "readiness",
            "live_execution_env": "0",
            "name": "Production Readiness",
            "path": ".github/workflows/production-readiness.yml",
            "sbom_path": "build/security/sbom.cdx.json",
            "wrapper_command": "python tools/security/production_readiness.py --profile ci",
        },
    }


def test_crg3_evidence_validation_requires_exact_unique_check_set() -> None:
    record = _valid_crg3_evidence_record()
    _validate_crg3_evidence_record(record)

    missing_check_record = {
        **record,
        "checks": [check for check in record["checks"] if check != "full pytest suite"],
    }
    with pytest.raises(ValueError, match="missing: full pytest suite"):
        _validate_crg3_evidence_record(missing_check_record)

    duplicate_check_record = {
        **record,
        "checks": [*record["checks"], "full pytest suite"],
    }
    with pytest.raises(ValueError, match="duplicates: 'full pytest suite'"):
        _validate_crg3_evidence_record(duplicate_check_record)

    unexpected_check_record = {
        **record,
        "checks": [*record["checks"], "invented readiness gate"],
    }
    with pytest.raises(ValueError, match="unexpected: invented readiness gate"):
        _validate_crg3_evidence_record(unexpected_check_record)


def test_crg3_evidence_validation_uses_full_string_schema_patterns() -> None:
    record = _valid_crg3_evidence_record()

    newline_tainted_sha_record = {
        **record,
        "commit_sha": f"{record['commit_sha']}\n",
    }
    with pytest.raises(ValueError, match="commit_sha does not match required pattern"):
        _validate_crg3_evidence_record(newline_tainted_sha_record)

    whitespace_tainted_repo_record = {
        **record,
        "repository": "chainsyncstore/hypothesis research engine",
    }
    with pytest.raises(ValueError, match="repository does not match required pattern"):
        _validate_crg3_evidence_record(whitespace_tainted_repo_record)

    trailing_hyphen_owner_record = {
        **record,
        "repository": "chainsyncstore-/hypothesis-research-engine",
    }
    with pytest.raises(ValueError, match="repository does not match required pattern"):
        _validate_crg3_evidence_record(trailing_hyphen_owner_record)

    unrecognized_binding_source_record = {
        **record,
        "repository_binding_source": "new-origin fetch remote",
    }
    with pytest.raises(
        ValueError,
        match="repository_binding_source does not match required pattern",
    ):
        _validate_crg3_evidence_record(unrecognized_binding_source_record)


def test_crg3_evidence_validation_requires_runtime_lock_digest() -> None:
    record = _valid_crg3_evidence_record()

    missing_lock_record = dict(record)
    missing_lock_record.pop("runtime_lock")
    with pytest.raises(ValueError, match="runtime_lock"):
        _validate_crg3_evidence_record(missing_lock_record)

    malformed_digest_record = {
        **record,
        "runtime_lock": {"path": "requirements.lock", "sha256": "not-a-digest"},
    }
    with pytest.raises(ValueError, match="runtime_lock.sha256"):
        _validate_crg3_evidence_record(malformed_digest_record)

    wrong_digest_record = {
        **record,
        "runtime_lock": {"path": "requirements.lock", "sha256": "f" * 64},
    }
    with pytest.raises(ValueError, match="runtime_lock does not match"):
        _validate_crg3_evidence_record(wrong_digest_record)


def test_crg3_evidence_validation_requires_consistent_identity_fields() -> None:
    record = _valid_crg3_evidence_record()
    _validate_crg3_evidence_record(record)

    wrong_repository_record = {
        **record,
        "repository": "chainsyncstore/other-repo",
    }
    with pytest.raises(ValueError, match="repository does not match run_url"):
        _validate_crg3_evidence_record(wrong_repository_record)

    wrong_run_id_record = {
        **record,
        "run_id": "987654321",
    }
    with pytest.raises(ValueError, match="run_id does not match run_url"):
        _validate_crg3_evidence_record(wrong_run_id_record)

    wrong_local_head_record = {
        **record,
        "local_head": "abcdef0123456789abcdef0123456789abcdef01",
    }
    with pytest.raises(ValueError, match="local_head does not match commit_sha"):
        _validate_crg3_evidence_record(wrong_local_head_record)

    wrong_download_repo_record = {
        **record,
        "sbom_artifact_metadata": {
            **record["sbom_artifact_metadata"],
            "download_url": (
                "https://api.github.com/repos/chainsyncstore/"
                "other-repo/actions/artifacts/42/zip"
            ),
        },
    }
    with pytest.raises(ValueError, match="download URL must belong"):
        _validate_crg3_evidence_record(wrong_download_repo_record)

    wrong_artifact_id_record = {
        **record,
        "sbom_artifact_metadata": {
            **record["sbom_artifact_metadata"],
            "id": 43,
        },
    }
    with pytest.raises(ValueError, match="id does not match download_url"):
        _validate_crg3_evidence_record(wrong_artifact_id_record)


def test_crg3_evidence_validation_requires_real_verified_date() -> None:
    record = _valid_crg3_evidence_record()
    _validate_crg3_evidence_record(record)

    impossible_date_record = {
        **record,
        "verified_date": "2026-02-31",
    }
    with pytest.raises(ValueError, match="verified_date must be a valid ISO calendar date"):
        _validate_crg3_evidence_record(impossible_date_record)

    future_date_record = {
        **record,
        "verified_date": "9999-12-31",
    }
    with pytest.raises(ValueError, match="verified_date must not be in the future"):
        _validate_crg3_evidence_record(future_date_record)


def test_crg3_evidence_validation_requires_sbom_digest_metadata() -> None:
    record = _valid_crg3_evidence_record()

    missing_digest_record = {
        **record,
        "sbom_artifact_metadata": {
            key: value
            for key, value in record["sbom_artifact_metadata"].items()
            if key != "sbom_sha256"
        },
    }
    with pytest.raises(ValueError, match="sbom_artifact_metadata"):
        _validate_crg3_evidence_record(missing_digest_record)

    missing_artifact_digest_record = {
        **record,
        "sbom_artifact_metadata": {
            key: value
            for key, value in record["sbom_artifact_metadata"].items()
            if key != "artifact_sha256"
        },
    }
    with pytest.raises(ValueError, match="sbom_artifact_metadata"):
        _validate_crg3_evidence_record(missing_artifact_digest_record)

    malformed_digest_record = {
        **record,
        "sbom_artifact_metadata": {
            **record["sbom_artifact_metadata"],
            "sbom_sha256": "not-a-digest",
        },
    }
    with pytest.raises(ValueError, match="sbom_artifact_metadata.sbom_sha256"):
        _validate_crg3_evidence_record(malformed_digest_record)

    malformed_artifact_digest_record = {
        **record,
        "sbom_artifact_metadata": {
            **record["sbom_artifact_metadata"],
            "artifact_sha256": "not-a-digest",
        },
    }
    with pytest.raises(ValueError, match="sbom_artifact_metadata.artifact_sha256"):
        _validate_crg3_evidence_record(malformed_artifact_digest_record)

    negative_download_size_record = {
        **record,
        "sbom_artifact_metadata": {
            **record["sbom_artifact_metadata"],
            "download_size_in_bytes": -1,
        },
    }
    with pytest.raises(ValueError, match="sbom_artifact_metadata.download_size_in_bytes"):
        _validate_crg3_evidence_record(negative_download_size_record)

    zero_download_size_record = {
        **record,
        "sbom_artifact_metadata": {
            **record["sbom_artifact_metadata"],
            "download_size_in_bytes": 0,
        },
    }
    with pytest.raises(ValueError, match="sbom_artifact_metadata.download_size_in_bytes"):
        _validate_crg3_evidence_record(zero_download_size_record)

    oversized_download_record = {
        **record,
        "sbom_artifact_metadata": {
            **record["sbom_artifact_metadata"],
            "download_size_in_bytes": MAX_SBOM_ARTIFACT_BYTES + 1,
        },
    }
    with pytest.raises(ValueError, match="sbom_artifact_metadata.download_size_in_bytes"):
        _validate_crg3_evidence_record(oversized_download_record)

    oversized_artifact_record = {
        **record,
        "sbom_artifact_metadata": {
            **record["sbom_artifact_metadata"],
            "size_in_bytes": MAX_SBOM_ARTIFACT_BYTES + 1,
        },
    }
    with pytest.raises(ValueError, match="sbom_artifact_metadata.size_in_bytes"):
        _validate_crg3_evidence_record(oversized_artifact_record)

    zero_artifact_record = {
        **record,
        "sbom_artifact_metadata": {
            **record["sbom_artifact_metadata"],
            "size_in_bytes": 0,
        },
    }
    with pytest.raises(ValueError, match="sbom_artifact_metadata.size_in_bytes"):
        _validate_crg3_evidence_record(zero_artifact_record)

    zero_component_record = {
        **record,
        "sbom_artifact_metadata": {
            **record["sbom_artifact_metadata"],
            "component_count": 0,
        },
    }
    with pytest.raises(ValueError, match="sbom_artifact_metadata.component_count"):
        _validate_crg3_evidence_record(zero_component_record)

    wrong_component_count_record = {
        **record,
        "sbom_artifact_metadata": {
            **record["sbom_artifact_metadata"],
            "component_count": len(_expected_runtime_sbom_components()) + 1,
        },
    }
    with pytest.raises(ValueError, match="component_count does not match"):
        _validate_crg3_evidence_record(wrong_component_count_record)


def test_finalize_crg3_dry_run_does_not_write_evidence_json(tmp_path: Path) -> None:
    roadmap = tmp_path / "roadmap.md"
    original_text = (
        "| ID | Pass | Priority | Status | Work item | Files | Proof required |\n"
        "| --- | --- | --- | --- | --- | --- | --- |\n"
        "| CRG3 | CI, Regression, and Production Readiness Gate | P0 | In progress | Add workflow | `.github/workflows/production-readiness.yml` | Awaiting workflow evidence. |\n"
    )
    roadmap.write_text(original_text, encoding="utf-8")
    evidence_json = tmp_path / "crg3-evidence.json"
    sha = "0123456789abcdef0123456789abcdef01234567"

    finalize(
        run_url="https://github.com/chainsyncstore/hypothesis-research-engine/actions/runs/123456789",
        commit_sha=sha,
        sbom_artifact="production-sbom",
        verified_date=date(2026, 6, 4),
        roadmap_path=roadmap,
        allowed_repos={"chainsyncstore/hypothesis-research-engine"},
        github_api_fetcher=_github_fetcher(sha=sha),
        github_artifact_fetcher=_artifact_fetcher(),
        local_head=sha,
        git_status="",
        dry_run=True,
        evidence_json_path=evidence_json,
    )

    assert roadmap.read_text(encoding="utf-8") == original_text
    assert not evidence_json.exists()


def test_finalize_crg3_rejects_future_verified_date_without_evidence_json(
    tmp_path: Path,
) -> None:
    roadmap = tmp_path / "roadmap.md"
    original_text = (
        "| ID | Pass | Priority | Status | Work item | Files | Proof required |\n"
        "| --- | --- | --- | --- | --- | --- | --- |\n"
        "| CRG3 | CI, Regression, and Production Readiness Gate | P0 | In progress | Add workflow | `.github/workflows/production-readiness.yml` | Awaiting workflow evidence. |\n"
    )
    roadmap.write_text(original_text, encoding="utf-8")
    sha = "0123456789abcdef0123456789abcdef01234567"

    def fail_github_api_fetcher(url: str):
        raise AssertionError(f"future-date validation should run before GitHub API fetch: {url}")

    def fail_artifact_fetcher(url: str) -> bytes:
        raise AssertionError(f"future-date validation should run before artifact fetch: {url}")

    with pytest.raises(ValueError, match="verified date must not be in the future"):
        finalize(
            run_url="https://github.com/chainsyncstore/hypothesis-research-engine/actions/runs/123456789",
            commit_sha=sha,
            sbom_artifact="production-sbom",
            verified_date=date(9999, 12, 31),
            roadmap_path=roadmap,
            allowed_repos={"chainsyncstore/hypothesis-research-engine"},
            github_api_fetcher=fail_github_api_fetcher,
            github_artifact_fetcher=fail_artifact_fetcher,
            local_head=sha,
            git_status="",
        )

    assert roadmap.read_text(encoding="utf-8") == original_text


def test_finalize_crg3_rejects_newline_tainted_url_and_sha_before_fetch(
    tmp_path: Path,
) -> None:
    roadmap = tmp_path / "roadmap.md"
    original_text = (
        "| ID | Pass | Priority | Status | Work item | Files | Proof required |\n"
        "| --- | --- | --- | --- | --- | --- | --- |\n"
        "| CRG3 | CI, Regression, and Production Readiness Gate | P0 | In progress | Add workflow | `.github/workflows/production-readiness.yml` | Awaiting workflow evidence. |\n"
    )
    roadmap.write_text(original_text, encoding="utf-8")
    sha = "0123456789abcdef0123456789abcdef01234567"

    def fail_github_api_fetcher(url: str):
        raise AssertionError(f"input validation should run before GitHub API fetch: {url}")

    def fail_artifact_fetcher(url: str) -> bytes:
        raise AssertionError(f"input validation should run before artifact fetch: {url}")

    common_kwargs = {
        "sbom_artifact": "production-sbom",
        "verified_date": date(2026, 6, 4),
        "roadmap_path": roadmap,
        "allowed_repos": {"chainsyncstore/hypothesis-research-engine"},
        "github_api_fetcher": fail_github_api_fetcher,
        "github_artifact_fetcher": fail_artifact_fetcher,
        "local_head": sha,
        "git_status": "",
    }

    with pytest.raises(ValueError, match="run URL must look like"):
        finalize(
            run_url=(
                "https://github.com/chainsyncstore/"
                "hypothesis-research-engine/actions/runs/123456789\n"
            ),
            commit_sha=sha,
            **common_kwargs,
        )

    with pytest.raises(ValueError, match="run URL must look like"):
        finalize(
            run_url=(
                "https://github.com/chainsyncstore/"
                "hypothesis research engine/actions/runs/123456789"
            ),
            commit_sha=sha,
            **common_kwargs,
        )

    with pytest.raises(ValueError, match="run URL must look like"):
        finalize(
            run_url=(
                "https://github.com/chainsyncstore-/"
                "hypothesis-research-engine/actions/runs/123456789"
            ),
            commit_sha=sha,
            **common_kwargs,
        )

    with pytest.raises(ValueError, match="40-character hex SHA"):
        finalize(
            run_url="https://github.com/chainsyncstore/hypothesis-research-engine/actions/runs/123456789",
            commit_sha=f"{sha}\n",
            **common_kwargs,
        )

    assert roadmap.read_text(encoding="utf-8") == original_text


def test_finalize_crg3_validates_evidence_json_schema_before_writing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    roadmap = tmp_path / "roadmap.md"
    original_text = (
        "| ID | Pass | Priority | Status | Work item | Files | Proof required |\n"
        "| --- | --- | --- | --- | --- | --- | --- |\n"
        "| CRG3 | CI, Regression, and Production Readiness Gate | P0 | In progress | Add workflow | `.github/workflows/production-readiness.yml` | Awaiting workflow evidence. |\n"
    )
    roadmap.write_text(original_text, encoding="utf-8")
    evidence_json = tmp_path / "crg3-evidence.json"
    schema_path = tmp_path / "strict-schema.json"
    schema_path.write_text(
        json.dumps(
            {
                "type": "object",
                "additionalProperties": True,
                "properties": {},
                "required": ["missing_required_key"],
            }
        ),
        encoding="utf-8",
    )
    sha = "0123456789abcdef0123456789abcdef01234567"

    monkeypatch.setattr(
        "tools.security.finalize_crg3.CRG3_EVIDENCE_SCHEMA",
        str(schema_path),
    )

    with pytest.raises(ValueError, match="does not match schema"):
        finalize(
            run_url="https://github.com/chainsyncstore/hypothesis-research-engine/actions/runs/123456789",
            commit_sha=sha,
            sbom_artifact="production-sbom",
            verified_date=date(2026, 6, 4),
            roadmap_path=roadmap,
            allowed_repos={"chainsyncstore/hypothesis-research-engine"},
            github_api_fetcher=_github_fetcher(sha=sha),
            github_artifact_fetcher=_artifact_fetcher(),
            local_head=sha,
            git_status="",
            evidence_json_path=evidence_json,
        )

    assert roadmap.read_text(encoding="utf-8") == original_text
    assert not evidence_json.exists()


def test_finalize_crg3_rejects_evidence_json_path_aliasing_roadmap(
    tmp_path: Path,
) -> None:
    roadmap = tmp_path / "roadmap.json"
    original_text = (
        "| ID | Pass | Priority | Status | Work item | Files | Proof required |\n"
        "| --- | --- | --- | --- | --- | --- | --- |\n"
        "| CRG3 | CI, Regression, and Production Readiness Gate | P0 | In progress | Add workflow | `.github/workflows/production-readiness.yml` | Awaiting workflow evidence. |\n"
    )
    roadmap.write_text(original_text, encoding="utf-8")

    with pytest.raises(ValueError, match="must not be the roadmap path"):
        finalize(
            run_url="https://github.com/chainsyncstore/hypothesis-research-engine/actions/runs/123456789",
            commit_sha="0123456789abcdef0123456789abcdef01234567",
            sbom_artifact="production-sbom",
            verified_date=date(2026, 6, 4),
            roadmap_path=roadmap,
            allowed_repos={"chainsyncstore/hypothesis-research-engine"},
            local_head="0123456789abcdef0123456789abcdef01234567",
            git_status="",
            evidence_json_path=tmp_path / "." / "roadmap.json",
        )

    assert roadmap.read_text(encoding="utf-8") == original_text


def test_finalize_crg3_rejects_evidence_json_path_aliasing_schema() -> None:
    roadmap = REPO_ROOT / "PRODUCTION_REFACTOR_ROADMAP.md"
    schema_path = REPO_ROOT / "docs" / "CRG3_EVIDENCE_SCHEMA.json"
    schema_text = schema_path.read_text(encoding="utf-8")

    with pytest.raises(ValueError, match="must not be the evidence schema path"):
        finalize(
            run_url="https://github.com/chainsyncstore/hypothesis-research-engine/actions/runs/123456789",
            commit_sha="0123456789abcdef0123456789abcdef01234567",
            sbom_artifact="production-sbom",
            verified_date=date(2026, 6, 4),
            roadmap_path=roadmap,
            allowed_repos={"chainsyncstore/hypothesis-research-engine"},
            local_head="0123456789abcdef0123456789abcdef01234567",
            git_status="",
            evidence_json_path=schema_path,
        )

    assert schema_path.read_text(encoding="utf-8") == schema_text


def test_finalize_crg3_rejects_evidence_json_path_aliasing_sbom_artifact(
    tmp_path: Path,
) -> None:
    repo_dir = tmp_path / "repo"
    security_dir = repo_dir / "build" / "security"
    security_dir.mkdir(parents=True)
    roadmap = repo_dir / "roadmap.md"
    original_text = (
        "| ID | Pass | Priority | Status | Work item | Files | Proof required |\n"
        "| --- | --- | --- | --- | --- | --- | --- |\n"
        "| CRG3 | CI, Regression, and Production Readiness Gate | P0 | In progress | Add workflow | `.github/workflows/production-readiness.yml` | Awaiting workflow evidence. |\n"
    )
    roadmap.write_text(original_text, encoding="utf-8")
    sbom_path = security_dir / "sbom.cdx.json"

    with pytest.raises(ValueError, match="must not overwrite sbom.cdx.json"):
        finalize(
            run_url="https://github.com/chainsyncstore/hypothesis-research-engine/actions/runs/123456789",
            commit_sha="0123456789abcdef0123456789abcdef01234567",
            sbom_artifact="production-sbom",
            verified_date=date(2026, 6, 4),
            roadmap_path=roadmap,
            allowed_repos={"chainsyncstore/hypothesis-research-engine"},
            local_head="0123456789abcdef0123456789abcdef01234567",
            git_status="",
            evidence_json_path=sbom_path,
        )

    assert roadmap.read_text(encoding="utf-8") == original_text
    assert not sbom_path.exists()


def test_finalize_crg3_requires_canonical_evidence_json_filename(
    tmp_path: Path,
) -> None:
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    roadmap = repo_dir / "roadmap.md"
    original_text = (
        "| ID | Pass | Priority | Status | Work item | Files | Proof required |\n"
        "| --- | --- | --- | --- | --- | --- | --- |\n"
        "| CRG3 | CI, Regression, and Production Readiness Gate | P0 | In progress | Add workflow | `.github/workflows/production-readiness.yml` | Awaiting workflow evidence. |\n"
    )
    roadmap.write_text(original_text, encoding="utf-8")
    arbitrary_json = repo_dir / "random-proof.json"

    with pytest.raises(ValueError, match="filename must be crg3-evidence.json"):
        finalize(
            run_url="https://github.com/chainsyncstore/hypothesis-research-engine/actions/runs/123456789",
            commit_sha="0123456789abcdef0123456789abcdef01234567",
            sbom_artifact="production-sbom",
            verified_date=date(2026, 6, 4),
            roadmap_path=roadmap,
            allowed_repos={"chainsyncstore/hypothesis-research-engine"},
            local_head="0123456789abcdef0123456789abcdef01234567",
            git_status="",
            evidence_json_path=arbitrary_json,
        )

    assert roadmap.read_text(encoding="utf-8") == original_text
    assert not arbitrary_json.exists()


def test_finalize_crg3_rejects_evidence_json_path_outside_roadmap_directory(
    tmp_path: Path,
) -> None:
    repo_dir = tmp_path / "repo"
    outside_dir = tmp_path / "outside"
    repo_dir.mkdir()
    outside_dir.mkdir()
    roadmap = repo_dir / "roadmap.md"
    original_text = (
        "| ID | Pass | Priority | Status | Work item | Files | Proof required |\n"
        "| --- | --- | --- | --- | --- | --- | --- |\n"
        "| CRG3 | CI, Regression, and Production Readiness Gate | P0 | In progress | Add workflow | `.github/workflows/production-readiness.yml` | Awaiting workflow evidence. |\n"
    )
    roadmap.write_text(original_text, encoding="utf-8")

    with pytest.raises(ValueError, match="must stay under the roadmap directory"):
        finalize(
            run_url="https://github.com/chainsyncstore/hypothesis-research-engine/actions/runs/123456789",
            commit_sha="0123456789abcdef0123456789abcdef01234567",
            sbom_artifact="production-sbom",
            verified_date=date(2026, 6, 4),
            roadmap_path=roadmap,
            allowed_repos={"chainsyncstore/hypothesis-research-engine"},
            local_head="0123456789abcdef0123456789abcdef01234567",
            git_status="",
            evidence_json_path=outside_dir / "crg3-evidence.json",
        )

    assert roadmap.read_text(encoding="utf-8") == original_text
    assert not (outside_dir / "crg3-evidence.json").exists()


def test_finalize_crg3_rejects_evidence_json_path_inside_git_directory(
    tmp_path: Path,
) -> None:
    repo_dir = tmp_path / "repo"
    git_dir = repo_dir / ".git"
    git_dir.mkdir(parents=True)
    roadmap = repo_dir / "roadmap.md"
    original_text = (
        "| ID | Pass | Priority | Status | Work item | Files | Proof required |\n"
        "| --- | --- | --- | --- | --- | --- | --- |\n"
        "| CRG3 | CI, Regression, and Production Readiness Gate | P0 | In progress | Add workflow | `.github/workflows/production-readiness.yml` | Awaiting workflow evidence. |\n"
    )
    roadmap.write_text(original_text, encoding="utf-8")
    evidence_json = git_dir / "crg3-evidence.json"

    with pytest.raises(ValueError, match=r"must not be inside \.git"):
        finalize(
            run_url="https://github.com/chainsyncstore/hypothesis-research-engine/actions/runs/123456789",
            commit_sha="0123456789abcdef0123456789abcdef01234567",
            sbom_artifact="production-sbom",
            verified_date=date(2026, 6, 4),
            roadmap_path=roadmap,
            allowed_repos={"chainsyncstore/hypothesis-research-engine"},
            local_head="0123456789abcdef0123456789abcdef01234567",
            git_status="",
            evidence_json_path=evidence_json,
        )

    assert roadmap.read_text(encoding="utf-8") == original_text
    assert not evidence_json.exists()


def test_finalize_crg3_rejects_evidence_json_path_inside_hidden_directory(
    tmp_path: Path,
) -> None:
    repo_dir = tmp_path / "repo"
    hidden_dir = repo_dir / ".github"
    hidden_dir.mkdir(parents=True)
    roadmap = repo_dir / "roadmap.md"
    original_text = (
        "| ID | Pass | Priority | Status | Work item | Files | Proof required |\n"
        "| --- | --- | --- | --- | --- | --- | --- |\n"
        "| CRG3 | CI, Regression, and Production Readiness Gate | P0 | In progress | Add workflow | `.github/workflows/production-readiness.yml` | Awaiting workflow evidence. |\n"
    )
    roadmap.write_text(original_text, encoding="utf-8")
    evidence_json = hidden_dir / "crg3-evidence.json"

    with pytest.raises(ValueError, match="must not be inside hidden directories"):
        finalize(
            run_url="https://github.com/chainsyncstore/hypothesis-research-engine/actions/runs/123456789",
            commit_sha="0123456789abcdef0123456789abcdef01234567",
            sbom_artifact="production-sbom",
            verified_date=date(2026, 6, 4),
            roadmap_path=roadmap,
            allowed_repos={"chainsyncstore/hypothesis-research-engine"},
            local_head="0123456789abcdef0123456789abcdef01234567",
            git_status="",
            evidence_json_path=evidence_json,
        )

    assert roadmap.read_text(encoding="utf-8") == original_text
    assert not evidence_json.exists()


def test_finalize_crg3_rejects_evidence_json_path_without_json_suffix(
    tmp_path: Path,
) -> None:
    roadmap = tmp_path / "roadmap.md"
    original_text = (
        "| ID | Pass | Priority | Status | Work item | Files | Proof required |\n"
        "| --- | --- | --- | --- | --- | --- | --- |\n"
        "| CRG3 | CI, Regression, and Production Readiness Gate | P0 | In progress | Add workflow | `.github/workflows/production-readiness.yml` | Awaiting workflow evidence. |\n"
    )
    roadmap.write_text(original_text, encoding="utf-8")

    with pytest.raises(ValueError, match=r"\.json suffix"):
        finalize(
            run_url="https://github.com/chainsyncstore/hypothesis-research-engine/actions/runs/123456789",
            commit_sha="0123456789abcdef0123456789abcdef01234567",
            sbom_artifact="production-sbom",
            verified_date=date(2026, 6, 4),
            roadmap_path=roadmap,
            allowed_repos={"chainsyncstore/hypothesis-research-engine"},
            local_head="0123456789abcdef0123456789abcdef01234567",
            git_status="",
            evidence_json_path=tmp_path / "crg3-evidence.txt",
        )

    assert roadmap.read_text(encoding="utf-8") == original_text


def test_finalize_crg3_does_not_write_evidence_json_when_roadmap_write_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from tools.security import finalize_crg3

    roadmap = tmp_path / "roadmap.md"
    original_text = (
        "| ID | Pass | Priority | Status | Work item | Files | Proof required |\n"
        "| --- | --- | --- | --- | --- | --- | --- |\n"
        "| CRG3 | CI, Regression, and Production Readiness Gate | P0 | In progress | Add workflow | `.github/workflows/production-readiness.yml` | Awaiting workflow evidence. |\n"
    )
    roadmap.write_text(original_text, encoding="utf-8")
    evidence_json = tmp_path / "crg3-evidence.json"
    sha = "0123456789abcdef0123456789abcdef01234567"
    original_atomic_write = finalize_crg3._atomic_write_text

    def fail_only_roadmap_write(path: Path, text: str) -> None:
        if path == roadmap:
            raise OSError("simulated roadmap write failure")
        original_atomic_write(path, text)

    monkeypatch.setattr(finalize_crg3, "_atomic_write_text", fail_only_roadmap_write)

    with pytest.raises(OSError, match="simulated roadmap write failure"):
        finalize(
            run_url="https://github.com/chainsyncstore/hypothesis-research-engine/actions/runs/123456789",
            commit_sha=sha,
            sbom_artifact="production-sbom",
            verified_date=date(2026, 6, 4),
            roadmap_path=roadmap,
            allowed_repos={"chainsyncstore/hypothesis-research-engine"},
            github_api_fetcher=_github_fetcher(sha=sha),
            github_artifact_fetcher=_artifact_fetcher(),
            local_head=sha,
            git_status="",
            evidence_json_path=evidence_json,
        )

    assert roadmap.read_text(encoding="utf-8") == original_text
    assert not evidence_json.exists()


def test_finalize_crg3_cli_dry_run_reports_no_write(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    roadmap = tmp_path / "roadmap.md"
    roadmap.write_text(
        "| ID | Pass | Priority | Status | Work item | Files | Proof required |\n"
        "| --- | --- | --- | --- | --- | --- | --- |\n"
        "| CRG3 | CI, Regression, and Production Readiness Gate | P0 | In progress | Add workflow | `.github/workflows/production-readiness.yml` | Awaiting workflow evidence. |\n",
        encoding="utf-8",
    )
    sha = "0123456789abcdef0123456789abcdef01234567"
    evidence_json = tmp_path / "crg3-evidence.json"
    calls = []

    def fake_finalize(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr("tools.security.finalize_crg3.finalize", fake_finalize)

    from tools.security.finalize_crg3 import main as finalize_main

    assert finalize_main(
        [
            "--run-url",
            "https://github.com/chainsyncstore/hypothesis-research-engine/actions/runs/123456789",
            "--commit-sha",
            sha,
            "--sbom-artifact",
            "production-sbom",
            "--roadmap",
            str(roadmap),
            "--evidence-json",
            str(evidence_json),
            "--dry-run",
        ]
    ) == 0

    assert calls
    assert calls[0]["dry_run"] is True
    assert calls[0]["roadmap_path"] == roadmap
    assert calls[0]["evidence_json_path"] == evidence_json
    assert "CRG3 finalizer dry run passed; roadmap not modified." in capsys.readouterr().out


def test_finalize_crg3_can_reverify_with_explicit_override(tmp_path: Path) -> None:
    roadmap = tmp_path / "roadmap.md"
    roadmap.write_text(
        "| ID | Pass | Priority | Status | Work item | Files | Proof required |\n"
        "| --- | --- | --- | --- | --- | --- | --- |\n"
        "| CRG3 | CI, Regression, and Production Readiness Gate | P0 | Verified 2026-06-03 | Add workflow | `.github/workflows/production-readiness.yml` | Existing workflow evidence. |\n",
        encoding="utf-8",
    )
    sha = "0123456789abcdef0123456789abcdef01234567"

    finalize(
        run_url="https://github.com/chainsyncstore/hypothesis-research-engine/actions/runs/123456789",
        commit_sha=sha,
        sbom_artifact="production-sbom",
        verified_date=date(2026, 6, 4),
        roadmap_path=roadmap,
        allowed_repos={"chainsyncstore/hypothesis-research-engine"},
        github_api_fetcher=_github_fetcher(sha=sha),
        github_artifact_fetcher=_artifact_fetcher(),
        local_head=sha,
        git_status="",
        allow_reverify=True,
    )

    text = roadmap.read_text(encoding="utf-8")
    assert "Verified 2026-06-04" in text
    assert "Existing workflow evidence" not in text
    assert "CRG3 Workflow Evidence Finalization" in text


def test_finalize_crg3_updates_roadmap_and_allows_p0_validation(tmp_path: Path) -> None:
    roadmap = tmp_path / "roadmap.md"
    roadmap.write_text(
        "| ID | Pass | Priority | Status | Work item | Files | Proof required |\n"
        "| --- | --- | --- | --- | --- | --- | --- |\n"
        "| CRG3 | CI, Regression, and Production Readiness Gate | P0 | In progress | Add workflow | `.github/workflows/production-readiness.yml` | Awaiting workflow evidence. |\n",
        encoding="utf-8",
    )
    sha = "0123456789abcdef0123456789abcdef01234567"

    finalize(
        run_url="https://github.com/chainsyncstore/hypothesis-research-engine/actions/runs/123456789",
        commit_sha=sha,
        sbom_artifact="production-sbom",
        verified_date=date(2026, 6, 4),
        roadmap_path=roadmap,
        allowed_repos={"chainsyncstore/hypothesis-research-engine"},
        github_api_fetcher=_github_fetcher(sha=sha),
        github_artifact_fetcher=_artifact_fetcher(),
        local_head=sha,
        git_status="",
    )

    text = roadmap.read_text(encoding="utf-8")
    assert "Verified 2026-06-04" in text
    assert "repo: `chainsyncstore/hypothesis-research-engine`" in text
    assert "repo binding: `explicit allowed repository set`" in text
    assert "run id: `123456789`" in text
    assert "run attempt: `1`" in text
    assert "Repository: `chainsyncstore/hypothesis-research-engine`" in text
    assert "Repository binding source: `explicit allowed repository set`" in text
    assert "GitHub API repository.full_name: `chainsyncstore/hypothesis-research-engine`" in text
    assert "Workflow run ID: `123456789`" in text
    assert "Workflow run attempt: `1`" in text
    assert "SBOM artifact ID: `42`" in text
    assert "SBOM artifact size: `2048`" in text
    assert f"SBOM artifact downloaded size: `{_default_sbom_artifact_metadata()['download_size_in_bytes']}`" in text
    assert "SBOM artifact download URL" in text
    assert f"SBOM artifact SHA-256: `{_default_sbom_artifact_metadata()['artifact_sha256']}`" in text
    assert f"SBOM JSON SHA-256: `{_default_sbom_artifact_metadata()['sbom_sha256']}`" in text
    assert f"SBOM component count: `{_default_sbom_artifact_metadata()['component_count']}`" in text
    assert "actions/artifacts/42/zip" in text
    assert "manual dispatch, pull-request probes, release-branch push triggers" in text
    assert "same-ref concurrency without cancellation" in text
    assert "Local production readiness workflow pins job-level `BOT_V2_ALLOW_LIVE_EXECUTION=0`" in text
    assert "uses pinned checkout/setup actions" in text
    assert "installs hash-locked CI tools before hash-locked runtime dependencies" in text
    assert "Local production readiness workflow uploads `build/security/sbom.cdx.json`" in text
    assert "Local checkout HEAD" in text
    assert "Local working tree: clean" in text
    assert "actions/runs/123456789" in text
    assert sha in text
    assert "production-sbom" in text
    assert "exact completed checks recorded in CRG3 evidence JSON" in text
    assert "Completed checks:" in text
    for check_name in _expected_crg3_evidence_checks():
        assert f"`{check_name}`" in text
    assert "CycloneDX" in text
    assert "sbom.cdx.json" in text
    assert "components match `requirements.lock` exactly" in text
    assert "Runtime lockfile: `requirements.lock`" in text
    assert f"Runtime lockfile SHA-256: `{_runtime_lock_metadata()['sha256']}`" in text
    assert "artifact ID/download URL binding" in text
    assert "canonical artifact URL validation" in text
    assert "cross-host redirect authorization stripping" in text
    assert "explicit `expired: false`" in text
    assert "bounded `size_in_bytes`" in text
    assert "artifact `id` matching `archive_download_url`" in text
    assert "canonical with no query string or fragment" in text
    assert "cross-host artifact download redirects strip `Authorization`" in text
    assert "CRG3 Workflow Evidence Finalization" in text
    assert validate_p0_rows(parse_ledger(roadmap)) == []


def test_roadmap_validator_blocks_unverified_p0_rows(tmp_path: Path) -> None:
    roadmap = tmp_path / "roadmap.md"
    roadmap.write_text(
        "| ID | Pass | Priority | Status | Work item | Files | Proof required |\n"
        "| --- | --- | --- | --- | --- | --- | --- |\n"
        "| CRG3 | CI | P0 | In progress | Add workflow | `.github/workflows/x.yml` | Workflow added but not run. |\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [sys.executable, "tools/security/check_roadmap.py", str(roadmap)],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "CRG3" in result.stdout
    assert "P0 status must be" in result.stdout


def test_roadmap_validator_accepts_verified_p0_rows(tmp_path: Path) -> None:
    roadmap = tmp_path / "roadmap.md"
    roadmap.write_text(
        "| ID | Pass | Priority | Status | Work item | Files | Proof required |\n"
        "| --- | --- | --- | --- | --- | --- | --- |\n"
        "| L1 | Live | P0 | Verified 2026-06-04 | Gate live | `tests/x.py` | Verified by pytest focused test. |\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [sys.executable, "tools/security/check_roadmap.py", str(roadmap)],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Roadmap P0 ledger validation passed." in result.stdout


def test_roadmap_validator_can_bootstrap_only_named_open_row(tmp_path: Path) -> None:
    roadmap = tmp_path / "roadmap.md"
    roadmap.write_text(
        "| ID | Pass | Priority | Status | Work item | Files | Proof required |\n"
        "| --- | --- | --- | --- | --- | --- | --- |\n"
        "| CRG3 | CI | P0 | In progress | Add workflow | `.github/workflows/x.yml` | Workflow added but not run. |\n"
        "| CRG6 | CI | P0 | In progress | Add checklist | `docs/x.md` | Docs checklist pending. |\n",
        encoding="utf-8",
    )

    crg3_allowed = subprocess.run(
        [
            sys.executable,
            "tools/security/check_roadmap.py",
            str(roadmap),
            "--allow-open-id",
            "CRG3",
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert crg3_allowed.returncode == 1
    assert "CRG3" not in crg3_allowed.stdout
    assert "CRG6" in crg3_allowed.stdout

    both_allowed = subprocess.run(
        [
            sys.executable,
            "tools/security/check_roadmap.py",
            str(roadmap),
            "--allow-open-id",
            "CRG3",
            "--allow-open-id",
            "CRG6",
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "Roadmap P0 ledger validation passed." in both_allowed.stdout


def test_focused_p0_suite_covers_deployment_blocking_gate_files() -> None:
    required_paths = {
        "tests/quant_v2/test_binance_adapter.py",
        "tests/quant_v2/test_execution_service.py",
        "tests/quant_v2/test_bounded_liquidation.py",
        "tests/quant_v2/test_chase_logic.py",
        "tests/quant_v2/test_live_readiness.py",
        "tests/quant_v2/test_execution_infra.py",
        "tests/quant_v2/test_reconciliation.py",
        "tests/quant/test_telebot_main_v2_handlers.py",
        "tests/quant/test_model_selection.py",
        "tests/quant_v2/test_model_registry.py",
        "tests/quant_v2/test_signal_manager.py",
        "tests/infra/test_docker_compose_services.py",
        "tests/infra/test_release_artifact_scan.py",
        "tests/infra/test_deployment_hardening_docs.py",
        "tests/infra/test_production_readiness.py",
    }

    assert required_paths <= set(FOCUSED_P0_TESTS)


def test_unsafe_historical_expectations_are_inverted_in_tests() -> None:
    binance_tests = (REPO_ROOT / "tests/quant_v2/test_binance_adapter.py").read_text(
        encoding="utf-8"
    )
    model_selection_tests = (REPO_ROOT / "tests/quant/test_model_selection.py").read_text(
        encoding="utf-8"
    )
    registry_tests = (REPO_ROOT / "tests/quant_v2/test_model_registry.py").read_text(
        encoding="utf-8"
    )
    signal_tests = (REPO_ROOT / "tests/quant_v2/test_signal_manager.py").read_text(
        encoding="utf-8"
    )

    assert "does_not_fallback_to_market" in binance_tests
    assert 'assert not any(call[3] == "MARKET"' in binance_tests
    assert "refusing latest model discovery fallback" in model_selection_tests
    assert "uses_latest_only_with_explicit_fallback_flag" in model_selection_tests
    assert "requires_explicit_eligibility_and_manifest" in registry_tests
    assert "rejects_corrupt_manifest_checksum" in registry_tests
    assert "rejects_partial_horizon_fallback_by_default" in signal_tests

    forbidden_positive_expectations = [
        r"assert\s+.*fallback_to_market",
        r"assert\s+.*placeholder.*promot",
        r"assert\s+.*latest.*fallback.*default",
    ]
    combined = "\n".join([binance_tests, model_selection_tests, registry_tests, signal_tests])
    for pattern in forbidden_positive_expectations:
        assert not re.search(pattern, combined, flags=re.IGNORECASE)
