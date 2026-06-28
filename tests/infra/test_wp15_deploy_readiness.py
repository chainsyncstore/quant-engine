from __future__ import annotations

import json
import sys
import subprocess
from pathlib import Path
from types import SimpleNamespace

from sqlalchemy import create_engine

from quant.telebot.models import Base
import tools.deploy_readiness as deploy_readiness


def _write_sqlite_schema(path: Path) -> None:
    engine = create_engine(f"sqlite:///{path}")
    Base.metadata.create_all(engine)
    engine.dispose()


_COMPOSE_CONFIG_OUTPUT = """services:
  redis:
    read_only: true
  model_evaluator:
    read_only: true
  retrain_scheduler:
    read_only: true
  telegram_bot:
    read_only: true
"""


class _FakeRegistry:
    def __init__(self, root, image_reference: str):
        self.root = Path(root)
        self.image_reference = image_reference

    def validate_activation_ready(self, version_id: str):
        return SimpleNamespace(version_id=version_id, artifact_dir=str(self.root / version_id))

    def get_artifact_manifest(self, version_id: str):
        return {
            "schema_version": "wp10-model-artifact-v1",
            "checksums": {
                "artifact_manifest_sha256": "abc123",
            },
            "runtime": {
                "image_reference": self.image_reference,
            },
            "model": {
                "feature_schema_sha256": "schema123",
            },
            "training": {
                "dataset_digest": "dataset123",
            },
        }

    def _artifact_contract_payload(self, record):
        manifest = self.get_artifact_manifest(record.version_id)
        return {
            "artifact_manifest_sha256": manifest["checksums"]["artifact_manifest_sha256"],
            "artifact_image_reference": manifest["runtime"]["image_reference"],
            "artifact_feature_schema_sha256": manifest["model"]["feature_schema_sha256"],
            "artifact_dataset_digest": manifest["training"]["dataset_digest"],
        }


def _make_manifest(
    tmp_path: Path,
    *,
    image_reference: str,
    rollback_image_reference: str | None = None,
    stage_plan: list[dict[str, object]] | None = None,
) -> Path:
    compose_file = tmp_path / "docker-compose.yml"
    compose_file.write_text(
        "services:\n  redis:\n    image: redis:7-alpine\n",
        encoding="utf-8",
    )
    attestation_path = tmp_path / "attestation.json"
    attestation_path.write_text(
        json.dumps(
            {
                "exact_image_certified": True,
                "immutable_registry_digest": image_reference,
                "build_manifest_sha256": "build123",
            }
        ),
        encoding="utf-8",
    )
    redis_secret = tmp_path / "redis_password.txt"
    redis_secret.write_text("super-secret", encoding="utf-8")
    models_dir = tmp_path / "models"
    state_dir = tmp_path / "state"
    models_dir.mkdir()
    state_dir.mkdir()
    db_path = tmp_path / "quant.db"
    backup_path = tmp_path / "quant-backup.db"
    _write_sqlite_schema(db_path)
    _write_sqlite_schema(backup_path)
    positions_path = tmp_path / "positions.json"
    positions_path.write_text(json.dumps({"open_positions": {}}), encoding="utf-8")

    manifest = {
        "schema_version": "wp15-deployment-manifest-v1",
        "project_root": str(tmp_path),
        "image_reference": image_reference,
        "rollback_image_reference": rollback_image_reference or image_reference,
        "rollback_model_version_id": "model-rollback" if rollback_image_reference else "model-live",
        "compose_files": [compose_file.name],
        "compose_environment": {},
        "attestation_path": attestation_path.name,
        "model_registry_root": "registry",
        "active_model_version_id": "model-live",
        "writable_paths": [models_dir.name, state_dir.name],
        "redis_password_file": redis_secret.name,
        "redis_acl_username": "quant_app",
        "dns_probe_host": "localhost",
        "database_path": db_path.name,
        "backup_path": backup_path.name,
        "open_positions_path": positions_path.name,
        "migration_version": "2026-06-24T00:00:00Z",
        "service_order": ["redis", "model_evaluator", "retrain_scheduler", "telegram_bot"],
        "stage_plan": stage_plan
        or [
            {"name": "dependencies", "services": ["redis"]},
            {"name": "core", "services": ["model_evaluator", "retrain_scheduler"]},
            {"name": "interface", "services": ["telegram_bot"]},
        ],
        "smoke_command": [sys.executable, "-c", "print('smoke-ok')"],
        "smoke_expected_stdout": "smoke-ok\n",
        "smoke_expected_stdout_sha256": deploy_readiness._sha256_bytes(b"smoke-ok\n"),
        "operator_approvals": [
            {"approved_by": "ops-a", "evidence_digest": "ev-1"},
            {"approved_by": "ops-b", "evidence_digest": "ev-1"},
        ],
    }
    manifest_path = tmp_path / "deployment-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    (tmp_path / "registry" / "model-live").mkdir(parents=True)
    (tmp_path / "registry" / "model-rollback").mkdir(parents=True)
    return manifest_path


def test_deployment_readiness_preflight_and_idempotency(tmp_path, monkeypatch) -> None:
    manifest_path = _make_manifest(
        tmp_path,
        image_reference="registry.example/quant-bot@sha256:" + "0" * 64,
    )
    commands: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        commands.append(list(cmd))
        if cmd[-1] == "config":
            return SimpleNamespace(returncode=0, stdout=_COMPOSE_CONFIG_OUTPUT, stderr="")
        raise AssertionError(f"unexpected command: {cmd}")

    monkeypatch.setattr(
        deploy_readiness,
        "ModelRegistry",
        lambda root: _FakeRegistry(root, "registry.example/quant-bot@sha256:" + "0" * 64),
    )
    monkeypatch.setattr(deploy_readiness, "probe_dns_latency", lambda target=None: {"status": "healthy", "latency_ms": 3.2})
    monkeypatch.setattr(deploy_readiness, "probe_sqlite_lock_latency", lambda db_path=None: {"status": "healthy", "lock_latency_ms": 1.4})

    record = deploy_readiness.run(
        manifest_path,
        mode="deploy",
        execute=False,
        output_dir=tmp_path / "records",
        runner=fake_run,
    )
    rerun = deploy_readiness.run(
        manifest_path,
        mode="deploy",
        execute=False,
        output_dir=tmp_path / "records",
        runner=fake_run,
    )

    assert record == rerun
    assert record["schema_version"] == "wp15-deployment-record-v1"
    assert record["preflight"]["attestation"]["status"] == "healthy"
    assert record["preflight"]["compose"]["status"] == "healthy"
    assert record["preflight"]["active_model"]["status"] == "healthy"
    assert record["preflight"]["redis"]["status"] == "healthy"
    assert record["preflight"]["migrations"]["status"] == "healthy"
    assert record["preflight"]["migrations"]["table_name_drift"] == []
    assert record["preflight"]["migrations"]["forward_table_names"] == record["preflight"]["migrations"]["rollback_table_names"]
    assert record["preflight"]["migrations"]["schema_signature_sha256"]
    assert record["compose_configuration_sha256"]
    assert record["model"]["artifact_manifest_sha256"] == "abc123"
    assert record["stages"] == []
    assert record["smoke"]["status"] == "not_run"
    assert len(commands) == 2
    assert (tmp_path / "records" / f"deploy-{record['idempotency_key']}.json").exists()


def test_deployment_readiness_execute_runs_services_and_smoke(tmp_path, monkeypatch) -> None:
    manifest_path = _make_manifest(
        tmp_path,
        image_reference="registry.example/quant-bot@sha256:" + "2" * 64,
    )
    commands: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        commands.append(list(cmd))
        if cmd[-1] == "config":
            return SimpleNamespace(returncode=0, stdout=_COMPOSE_CONFIG_OUTPUT, stderr="")
        if cmd[:2] == ["docker", "compose"] and "up" in cmd:
            return SimpleNamespace(returncode=0, stdout=f"started {cmd[-1]}", stderr="")
        if cmd and cmd[0] == sys.executable:
            return SimpleNamespace(returncode=0, stdout="smoke-ok\n", stderr="")
        raise AssertionError(f"unexpected command: {cmd}")

    monkeypatch.setattr(
        deploy_readiness,
        "ModelRegistry",
        lambda root: _FakeRegistry(root, "registry.example/quant-bot@sha256:" + "2" * 64),
    )
    monkeypatch.setattr(deploy_readiness, "probe_dns_latency", lambda target=None: {"status": "healthy", "latency_ms": 3.2})
    monkeypatch.setattr(deploy_readiness, "probe_sqlite_lock_latency", lambda db_path=None: {"status": "healthy", "lock_latency_ms": 1.4})

    record = deploy_readiness.run(
        manifest_path,
        mode="deploy",
        execute=True,
        output_dir=tmp_path / "records",
        runner=fake_run,
    )

    assert record["smoke"]["status"] == "healthy"
    assert record["smoke"]["stdout_matches_expected"] is True
    assert record["smoke"]["expected_stdout"] == "smoke-ok\n"
    assert [stage["name"] for stage in record["stages"]] == ["dependencies", "core", "interface"]
    assert record["deploy_services"] and [entry["service"] for entry in record["deploy_services"]] == [
        "redis",
        "model_evaluator",
        "retrain_scheduler",
        "telegram_bot",
    ]
    assert commands[0][-1] == "config"
    assert any(cmd[-1] == "telegram_bot" for cmd in commands if "up" in cmd)


def test_deploy_smoke_mismatch_marks_rollout_degraded(tmp_path, monkeypatch) -> None:
    manifest_path = _make_manifest(
        tmp_path,
        image_reference="registry.example/quant-bot@sha256:" + "7" * 64,
    )
    commands: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        commands.append(list(cmd))
        if cmd[-1] == "config":
            return SimpleNamespace(returncode=0, stdout=_COMPOSE_CONFIG_OUTPUT, stderr="")
        if cmd[:2] == ["docker", "compose"] and "up" in cmd:
            return SimpleNamespace(returncode=0, stdout=f"started {cmd[-1]}", stderr="")
        if cmd and cmd[0] == sys.executable:
            return SimpleNamespace(returncode=0, stdout="unexpected\n", stderr="")
        raise AssertionError(f"unexpected command: {cmd}")

    monkeypatch.setattr(
        deploy_readiness,
        "ModelRegistry",
        lambda root: _FakeRegistry(root, "registry.example/quant-bot@sha256:" + "7" * 64),
    )
    monkeypatch.setattr(deploy_readiness, "probe_dns_latency", lambda target=None: {"status": "healthy", "latency_ms": 3.2})
    monkeypatch.setattr(deploy_readiness, "probe_sqlite_lock_latency", lambda db_path=None: {"status": "healthy", "lock_latency_ms": 1.4})

    record = deploy_readiness.run(
        manifest_path,
        mode="deploy",
        execute=True,
        output_dir=tmp_path / "records",
        runner=fake_run,
    )

    assert record["smoke"]["status"] == "degraded"
    assert record["smoke"]["stdout_matches_expected"] is False
    assert record["rollout"]["status"] == "degraded"
    assert any(cmd[-1] == "telegram_bot" for cmd in commands if "up" in cmd)
    assert commands[-1][0] == sys.executable


def test_deployment_readiness_flags_writable_compose_service(tmp_path, monkeypatch) -> None:
    manifest_path = _make_manifest(
        tmp_path,
        image_reference="registry.example/quant-bot@sha256:" + "8" * 64,
    )
    writable_compose_output = """services:
  redis:
    read_only: true
  model_evaluator:
    read_only: false
  retrain_scheduler:
    read_only: true
  telegram_bot:
    read_only: true
"""

    def fake_run(cmd, **kwargs):
        if cmd[-1] == "config":
            return SimpleNamespace(returncode=0, stdout=writable_compose_output, stderr="")
        raise AssertionError(f"unexpected command: {cmd}")

    monkeypatch.setattr(
        deploy_readiness,
        "ModelRegistry",
        lambda root: _FakeRegistry(root, "registry.example/quant-bot@sha256:" + "8" * 64),
    )
    monkeypatch.setattr(deploy_readiness, "probe_dns_latency", lambda target=None: {"status": "healthy", "latency_ms": 3.2})
    monkeypatch.setattr(deploy_readiness, "probe_sqlite_lock_latency", lambda db_path=None: {"status": "healthy", "lock_latency_ms": 1.4})

    record = deploy_readiness.run(
        manifest_path,
        mode="deploy",
        execute=False,
        output_dir=tmp_path / "records",
        runner=fake_run,
    )

    assert record["preflight"]["compose"]["status"] == "warning"
    assert record["preflight"]["compose"]["read_only_policy"]["status"] == "warning"
    assert record["preflight"]["compose"]["read_only_policy"]["writable_services"] == ["model_evaluator"]


def test_rollback_mode_uses_explicit_target_image_and_model(tmp_path, monkeypatch) -> None:
    manifest_path = _make_manifest(
        tmp_path,
        image_reference="registry.example/quant-bot@sha256:" + "3" * 64,
        rollback_image_reference="registry.example/quant-bot@sha256:" + "4" * 64,
    )

    commands: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        commands.append(list(cmd))
        if cmd[-1] == "config":
            return SimpleNamespace(returncode=0, stdout=_COMPOSE_CONFIG_OUTPUT, stderr="")
        if cmd[-1] == "--remove-orphans":
            return SimpleNamespace(returncode=0, stdout="stack-down\n", stderr="")
        if cmd[:2] == ["docker", "compose"] and "up" in cmd:
            return SimpleNamespace(returncode=0, stdout=f"started {cmd[-1]}", stderr="")
        if cmd and cmd[0] == sys.executable:
            return SimpleNamespace(returncode=0, stdout="smoke-ok\n", stderr="")
        raise AssertionError(f"unexpected command: {cmd}")

    monkeypatch.setattr(
        deploy_readiness,
        "ModelRegistry",
        lambda root: _FakeRegistry(root, "registry.example/quant-bot@sha256:" + "4" * 64),
    )
    monkeypatch.setattr(deploy_readiness, "probe_dns_latency", lambda target=None: {"status": "healthy", "latency_ms": 3.2})
    monkeypatch.setattr(deploy_readiness, "probe_sqlite_lock_latency", lambda db_path=None: {"status": "healthy", "lock_latency_ms": 1.4})

    record = deploy_readiness.run(
        manifest_path,
        mode="rollback",
        execute=True,
        output_dir=tmp_path / "records",
        runner=fake_run,
    )

    assert record["mode"] == "rollback"
    assert record["image_reference"] == "registry.example/quant-bot@sha256:" + "4" * 64
    assert record["rollback_target_image_reference"] == "registry.example/quant-bot@sha256:" + "4" * 64
    assert record["rollback_target_model_version_id"] == "model-rollback"
    assert record["compose_down"]["status"] == "healthy"
    record_copy = dict(record)
    record_sha = record_copy.pop("record_sha256")
    assert record_sha == deploy_readiness._sha256_json(record_copy)
    assert any(cmd[-1] == "--remove-orphans" for cmd in commands)
    assert any("up" in cmd and cmd[-1] == "telegram_bot" for cmd in commands)


def test_deploy_uses_custom_stage_plan(tmp_path, monkeypatch) -> None:
    manifest_path = _make_manifest(
        tmp_path,
        image_reference="registry.example/quant-bot@sha256:" + "6" * 64,
        stage_plan=[
            {"name": "dependencies", "services": ["redis"]},
            {"name": "supervisor", "services": ["model_evaluator"]},
            {"name": "signal", "services": ["telegram_bot"]},
        ],
    )
    commands: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        commands.append(list(cmd))
        if cmd[-1] == "config":
            return SimpleNamespace(returncode=0, stdout=_COMPOSE_CONFIG_OUTPUT, stderr="")
        if cmd[:2] == ["docker", "compose"] and "up" in cmd:
            return SimpleNamespace(returncode=0, stdout=f"started {cmd[-1]}", stderr="")
        if cmd and cmd[0] == sys.executable:
            return SimpleNamespace(returncode=0, stdout="smoke-ok\n", stderr="")
        raise AssertionError(f"unexpected command: {cmd}")

    monkeypatch.setattr(
        deploy_readiness,
        "ModelRegistry",
        lambda root: _FakeRegistry(root, "registry.example/quant-bot@sha256:" + "6" * 64),
    )
    monkeypatch.setattr(deploy_readiness, "probe_dns_latency", lambda target=None: {"status": "healthy", "latency_ms": 3.2})
    monkeypatch.setattr(deploy_readiness, "probe_sqlite_lock_latency", lambda db_path=None: {"status": "healthy", "lock_latency_ms": 1.4})

    record = deploy_readiness.run(
        manifest_path,
        mode="deploy",
        execute=True,
        output_dir=tmp_path / "records",
        runner=fake_run,
    )

    assert [stage["name"] for stage in record["stages"]] == ["dependencies", "supervisor", "signal"]
    assert [svc["service"] for svc in record["deploy_services"]] == ["redis", "model_evaluator", "telegram_bot"]
    assert any(cmd[-1] == "telegram_bot" for cmd in commands if "up" in cmd)


def test_deploy_rolls_back_on_smoke_failure(tmp_path, monkeypatch) -> None:
    manifest_path = _make_manifest(
        tmp_path,
        image_reference="registry.example/quant-bot@sha256:" + "5" * 64,
    )
    commands: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        commands.append(list(cmd))
        if cmd[-1] == "config":
            return SimpleNamespace(returncode=0, stdout=_COMPOSE_CONFIG_OUTPUT, stderr="")
        if cmd[:2] == ["docker", "compose"] and "up" in cmd:
            return SimpleNamespace(returncode=0, stdout=f"started {cmd[-1]}", stderr="")
        if cmd and cmd[0] == sys.executable:
            raise subprocess.CalledProcessError(returncode=1, cmd=cmd, output="smoke-fail", stderr="boom")
        if cmd[-1] == "--remove-orphans":
            return SimpleNamespace(returncode=0, stdout="stack-down\n", stderr="")
        raise AssertionError(f"unexpected command: {cmd}")

    monkeypatch.setattr(
        deploy_readiness,
        "ModelRegistry",
        lambda root: _FakeRegistry(root, "registry.example/quant-bot@sha256:" + "5" * 64),
    )
    monkeypatch.setattr(deploy_readiness, "probe_dns_latency", lambda target=None: {"status": "healthy", "latency_ms": 3.2})
    monkeypatch.setattr(deploy_readiness, "probe_sqlite_lock_latency", lambda db_path=None: {"status": "healthy", "lock_latency_ms": 1.4})

    record = deploy_readiness.run(
        manifest_path,
        mode="deploy",
        execute=True,
        output_dir=tmp_path / "records",
        runner=fake_run,
    )

    assert record["rollout"]["status"] == "degraded"
    assert record["rollout"]["error"] == "CalledProcessError"
    assert record["compose_down"]["status"] == "healthy"
    assert any(cmd[-1] == "--remove-orphans" for cmd in commands)
