from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest
import yaml

from scripts.verify_immutable_image import PATTERN
from tools.build_manifest import BUILD_INPUTS, generate, reviewed_context, source_digest
from tools.verify_external_artifacts import PINNED_IMAGES


ROOT = Path(__file__).resolve().parents[2]
CRITICAL = {
    "numpy": "2.2.6",
    "pandas": "2.3.3",
    "scikit-learn": "1.8.0",
    "lightgbm": "4.6.0",
    "joblib": "1.5.3",
    "torch": "2.9.1+cpu",
    "chronos-forecasting": "2.3.0",
}


def _locked_versions(path: Path) -> dict[str, str]:
    versions = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = re.match(r"^([a-zA-Z0-9_.-]+)==([^ \\]+)", line)
        if match:
            versions[match.group(1).lower()] = match.group(2)
    return versions


def test_runtime_lock_is_hashed_linux_cpu_and_critical_versions_are_exact() -> None:
    text = (ROOT / "requirements/runtime.lock").read_text(encoding="utf-8")
    assert "uv pip compile requirements/runtime.in" in text
    assert "--python-platform x86_64-manylinux_2_28" in text
    assert "--torch-backend cpu" in text
    assert "--unsafe-best-match" not in text
    assert "--extra-index-url" not in (ROOT / "requirements/runtime.in").read_text()
    locked = _locked_versions(ROOT / "requirements/runtime.lock")
    assert locked | CRITICAL == locked
    assert {"uv", "pytest", "ruff"}.isdisjoint(locked)
    for block in re.split(r"\n(?=[a-zA-Z0-9_.-]+==)", text)[1:]:
        assert "--hash=sha256:" in block


def test_test_lock_contains_exact_ruff_and_hashes() -> None:
    text = (ROOT / "requirements/test.lock").read_text(encoding="utf-8")
    assert _locked_versions(ROOT / "requirements/test.lock")["ruff"] == "0.14.10"
    assert "--hash=sha256:" in text
    assert "uv 0.11.23" in (ROOT / "docs/WP02_REPRODUCIBLE_BUILD.md").read_text()


def test_build_lock_hashes_exact_uv_bootstrap() -> None:
    text = (ROOT / "requirements/build.lock").read_text(encoding="utf-8")
    assert _locked_versions(ROOT / "requirements/build.lock") == {"uv": "0.11.23"}
    assert "--hash=sha256:" in text


def test_manifest_context_is_bounded_and_deterministic(tmp_path: Path) -> None:
    (tmp_path / "quant").mkdir()
    (tmp_path / "quant" / "app.py").write_text("VALUE = 1\n")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_app.py").write_text("def test_ok(): pass\n")
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "audit.md").write_text("excluded")
    (tmp_path / "deploy_optimized.tar.gz").write_bytes(b"excluded")
    paths = reviewed_context(tmp_path)
    assert paths == ["quant/app.py", "tests/test_app.py"]
    assert source_digest(tmp_path) == source_digest(tmp_path)


def test_compose_changes_source_digest(tmp_path: Path) -> None:
    compose = tmp_path / "docker-compose.yml"
    compose.write_text("services: {}\n", encoding="utf-8")
    first, records = source_digest(tmp_path)
    assert "docker-compose.yml" in {record["path"] for record in records}
    compose.write_text("services: {app: {}}\n", encoding="utf-8")
    second, _ = source_digest(tmp_path)
    assert first != second


def test_manifest_build_inputs_exist() -> None:
    assert all((ROOT / path).is_file() for path in BUILD_INPUTS)


def test_release_manifest_fails_on_dirty_tree(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "wp02@example.invalid"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "WP02 Test"], cwd=tmp_path, check=True)
    source = tmp_path / "quant"
    source.mkdir()
    app = source / "app.py"
    app.write_text("VALUE = 1\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "baseline"], cwd=tmp_path, check=True)
    app.write_text("VALUE = 2\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="clean Git tree"):
        generate(tmp_path, release=True)


def test_docker_release_depends_on_test_and_runtime_is_minimal() -> None:
    text = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "python:3.11.9-slim-bookworm@sha256:" in text
    assert "FROM runtime AS test" in text
    assert "RUN pytest -q /app/tests && ruff check /app/tools /app/tests/infra" in text
    assert "FROM runtime AS release" in text
    assert "COPY --from=test" in text
    assert "USER 10001:10001" in text
    runtime = text.split("FROM os-runtime AS runtime", 1)[1].split("FROM runtime AS test", 1)[0]
    assert "COPY tests/" not in runtime
    assert "build-essential" not in text
    assert "uv pip install" in text and "--torch-backend cpu" in text
    assert "python -m venv /opt/build-tools" in text
    assert "python -m venv /opt/runtime-venv" in text
    assert "COPY --from=dependencies /opt/runtime-venv /opt/runtime-venv" in runtime
    assert "/opt/build-tools" not in runtime
    assert "requirements/build.lock" not in runtime
    assert "pytest" not in runtime and "ruff" not in runtime


def test_apt_uses_declared_snapshot_and_exact_libgomp() -> None:
    text = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert 'DEBIAN_SNAPSHOT="20250201T000000Z"' in text
    assert "snapshot.debian.org/archive/debian/${DEBIAN_SNAPSHOT}/" in text
    assert "Acquire::Check-Valid-Until=false" in text
    assert "libgomp1=12.2.0-14+deb12u1" in text
    assert "apt-get update" not in text
    assert "apt-get -o Acquire::Check-Valid-Until=false update" in text


def test_dockerignore_fails_closed_and_allows_only_required_manifest() -> None:
    entries = set((ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines())
    required = {
        "state/",
        "**/state/",
        "logs/",
        "**/logs/",
        "secrets/",
        "**/secrets/",
        "signal_log*.json",
        ".build/*",
        "!.build/wp02-manifest.json",
        "*.db",
        "*.sqlite",
        "*.sqlite3",
        "*.log",
        "ubuntu_audit_*",
        "deploy_backups/",
        "scratch/",
        "docs/**/*audit*",
        "docs/**/*AUDIT*",
    }
    assert required <= entries
    assert "tests" not in entries and "tests/" not in entries
    assert not any(line.startswith("!.build/") and line != "!.build/wp02-manifest.json" for line in entries)


def test_compose_requires_digest_and_hardens_application_services() -> None:
    compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
    for name in ("telegram_bot", "retrain_scheduler", "model_evaluator"):
        service = compose["services"][name]
        assert "build" not in service
        assert "@sha256" in service["image"]
        assert service["user"] == "10001:10001"
        assert service["read_only"] is True
        assert service["cap_drop"] == ["ALL"]
        assert service["security_opt"] == ["no-new-privileges:true"]
        assert service["pids_limit"] > 0
        assert service["mem_limit"]
    redis = compose["services"]["redis"]
    assert "@sha256:" in redis["image"]
    assert "ports" not in redis


@pytest.mark.parametrize(
    "value,valid",
    [
        ("registry.example/quant:release@sha256:" + "a" * 64, True),
        ("quant_bot:latest", False),
        ("registry.example/quant@sha256:short", False),
    ],
)
def test_immutable_image_reference_contract(value: str, valid: bool) -> None:
    assert bool(PATTERN.fullmatch(value)) is valid


def test_ci_builds_release_once_and_retains_evidence() -> None:
    text = (ROOT / ".github/workflows/wp02-image.yml").read_text(encoding="utf-8")
    assert text.count("docker buildx build") == 1
    assert "--target release" in text
    assert text.index("python tools/build_manifest.py --release") < text.index("mkdir -p evidence")
    assert text.index("python tools/build_manifest.py --release") < text.index("python tools/verify_external_artifacts.py")
    assert "image_smoke.py" in text
    assert "sbom.cdx.json" in text
    assert "vulnerabilities.json" in text
    assert "secrets.json" in text
    assert "attestation.json" in text
    assert "runtime-exclusions.txt" in text
    assert "test \"$(id -u)\" = 10001" in text
    assert "! command -v uv" in text
    assert "! command -v pytest" in text
    assert "! command -v ruff" in text
    assert "SYFT_SHA256=1d3cc98b13ce3dfb6083ef42f64f1033e40d7dea292e8ea85ed1cf88efb2f542" in text
    assert "TRIVY_SHA256=0510e71e2fd39bf863856d499c8dc19feb4e7336546394c502a8f5cc7ab27460" in text
    assert "_checksums.txt" not in text
    action_refs = re.findall(r"uses:\s*[^@\s]+@([^\s#]+)", text)
    assert action_refs and all(re.fullmatch(r"[0-9a-f]{40}", ref) for ref in action_refs)


def test_runbook_and_workflow_pin_same_scanner_versions() -> None:
    workflow = (ROOT / ".github/workflows/wp02-image.yml").read_text(encoding="utf-8")
    runbook = (ROOT / "docs/WP02_REPRODUCIBLE_BUILD.md").read_text(encoding="utf-8")
    assert "SYFT_VERSION=1.42.2" in workflow
    assert "TRIVY_VERSION=0.71.2" in workflow
    assert "Syft `1.42.2`" in runbook
    assert "Trivy `0.71.2`" in runbook
    assert "0.69.1" not in runbook


def test_local_release_script_matches_runtime_acceptance_gates() -> None:
    script = (ROOT / "scripts/build_release.ps1").read_text(encoding="utf-8")
    assert "--require-release-marker" in script
    assert 'test "$(id -u)" = 10001' in script
    assert "test -f /app/.wp02-tests-passed" in script
    assert "test ! -d /app/tests" in script
    assert "test ! -d /opt/build-tools" in script
    assert "test ! -e /app/requirements/build.lock" in script
    assert "test ! -e /build/requirements/build.lock" in script
    assert "! command -v uv" in script
    assert "! command -v pytest" in script
    assert "! command -v ruff" in script
    assert "no-new-privileges:true" in script


def test_external_image_digest_contract_matches_docker_inputs() -> None:
    assert PINNED_IMAGES == {
        "library/python:3.11.9-slim-bookworm": "sha256:8fb099199b9f2d70342674bd9dbccd3ed03a258f26bbd1d556822c6dfc60c317",
        "library/redis:7-alpine": "sha256:6ab0b6e7381779332f97b8ca76193e45b0756f38d4c0dcda72dbb3c32061ab99",
    }
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    assert PINNED_IMAGES["library/python:3.11.9-slim-bookworm"] in dockerfile
    assert PINNED_IMAGES["library/redis:7-alpine"] in compose
