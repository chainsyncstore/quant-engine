#!/usr/bin/env python3
"""Deployment and readiness harness for WP-15."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Callable

from quant_v2.model_registry import ModelRegistry
from quant_v2.monitoring.runtime_probes import probe_dns_latency, probe_sqlite_lock_latency

_IMAGE_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._/-]*(?::[a-zA-Z0-9._-]+)?@sha256:[0-9a-f]{64}$")
_MANIFEST_SCHEMA = "wp15-deployment-manifest-v1"
_RECORD_SCHEMA = "wp15-deployment-record-v1"


@dataclass(frozen=True)
class DeploymentPreflightResult:
    status: str
    checks: dict[str, Any]


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_json(payload: Any) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return _sha256_bytes(canonical)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_path(value: object, *, base: Path) -> Path:
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path
    return (base / path).resolve()


def _validate_image_reference(image_reference: str) -> str:
    image = str(image_reference).strip()
    if not _IMAGE_PATTERN.fullmatch(image):
        raise ValueError("image_reference must be an immutable registry digest ending in @sha256:<64 hex>")
    return image


def _normalise_service_order(value: object) -> tuple[str, ...]:
    if isinstance(value, (list, tuple)):
        order = [str(item).strip() for item in value if str(item).strip()]
        if order:
            return tuple(order)
    return ()


def _normalise_stage_plan(value: object, *, fallback_services: tuple[str, ...]) -> tuple[dict[str, Any], ...]:
    if isinstance(value, (list, tuple)) and value:
        stages: list[dict[str, Any]] = []
        for index, raw_stage in enumerate(value, start=1):
            if not isinstance(raw_stage, dict):
                continue
            stage_name = str(raw_stage.get("name") or raw_stage.get("stage") or f"stage-{index}").strip()
            raw_services = raw_stage.get("services")
            if isinstance(raw_services, (list, tuple)):
                services = [str(item).strip() for item in raw_services if str(item).strip()]
            else:
                single_service = str(raw_stage.get("service") or "").strip()
                services = [single_service] if single_service else []
            if not services:
                continue
            stages.append({"name": stage_name, "services": tuple(services)})
        if stages:
            return tuple(stages)
    return tuple(
        {"name": f"stage-{index}", "services": (service,)}
        for index, service in enumerate(fallback_services, start=1)
        if str(service).strip()
    )


def _default_compose_files(manifest: dict[str, Any], *, base: Path) -> tuple[Path, ...]:
    compose_files = manifest.get("compose_files") or []
    files = []
    for value in compose_files if isinstance(compose_files, Iterable) else []:
        text = str(value).strip()
        if text:
            files.append(_resolve_path(text, base=base))
    return tuple(files)


def _render_compose_config(
    compose_files: tuple[Path, ...],
    *,
    project_dir: Path,
    env: dict[str, str],
    runner: Callable[..., subprocess.CompletedProcess[str]],
) -> tuple[str, str]:
    if not compose_files:
        return "", ""

    command: list[str] = ["docker", "compose"]
    for compose_file in compose_files:
        command.extend(["-f", str(compose_file)])
    command.append("config")

    completed = runner(
        command,
        cwd=str(project_dir),
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    stdout = completed.stdout or ""
    return stdout, _sha256_bytes(stdout.encode("utf-8"))


def _check_write_access(path: Path) -> dict[str, Any]:
    candidate = path if path.suffix == "" or path.is_dir() else path.parent
    candidate.mkdir(parents=True, exist_ok=True)
    probe = candidate / ".wp15-write-probe"
    try:
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
    except Exception as exc:
        return {"path": str(candidate), "status": "degraded", "error": exc.__class__.__name__}
    return {"path": str(candidate), "status": "healthy"}


def _check_open_positions(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {"status": "unknown", "path": None, "open_position_count": 0}

    payload = _load_json(path)
    positions: object
    if isinstance(payload, dict):
        if isinstance(payload.get("open_positions"), dict):
            positions = payload.get("open_positions")
        elif isinstance(payload.get("positions"), dict):
            positions = payload.get("positions")
        else:
            positions = payload
    else:
        positions = {}

    open_count = 0
    if isinstance(positions, dict):
        for value in positions.values():
            try:
                if abs(float(value)) > 1e-12:
                    open_count += 1
            except (TypeError, ValueError):
                open_count += 1
    status = "healthy" if open_count == 0 else "degraded"
    return {"status": status, "path": str(path), "open_position_count": open_count}


def _check_redis_secret(path: Path | None, username: str | None) -> dict[str, Any]:
    if path is None:
        return {"status": "unknown", "path": None, "username": username or ""}

    if not path.exists():
        return {"status": "degraded", "path": str(path), "username": username or "", "error": "missing_secret"}

    secret = path.read_text(encoding="utf-8").strip()
    if not secret:
        return {"status": "degraded", "path": str(path), "username": username or "", "error": "empty_secret"}

    if username is not None:
        clean_username = str(username).strip()
        if not clean_username or not re.fullmatch(r"[A-Za-z0-9_-]+", clean_username):
            return {
                "status": "degraded",
                "path": str(path),
                "username": clean_username,
                "error": "invalid_username",
            }

    return {"status": "healthy", "path": str(path), "username": username or ""}


def _schema_signature(table_names: Iterable[str]) -> str:
    cleaned = sorted(str(name).strip() for name in table_names if str(name).strip())
    return _sha256_json(cleaned)


def _inspect_compose_read_only_policy(compose_yaml: str) -> dict[str, Any]:
    try:
        import yaml  # type: ignore
    except Exception as exc:  # pragma: no cover - dependency should exist in test/runtime
        return {"status": "unknown", "error": exc.__class__.__name__}

    try:
        payload = yaml.safe_load(compose_yaml) or {}
    except Exception as exc:
        return {"status": "degraded", "error": exc.__class__.__name__}

    services = payload.get("services") if isinstance(payload, dict) else {}
    if not isinstance(services, dict) or not services:
        return {"status": "degraded", "error": "missing_services"}

    missing_read_only = sorted(
        str(name)
        for name, service in services.items()
        if not isinstance(service, dict) or service.get("read_only") is not True
    )
    writable_services = sorted(
        str(name)
        for name, service in services.items()
        if isinstance(service, dict) and service.get("read_only") is not True
    )
    status = "healthy" if not writable_services else "warning"
    return {
        "status": status,
        "service_count": len(services),
        "read_only_service_count": len(services) - len(writable_services),
        "writable_services": writable_services,
        "missing_read_only": missing_read_only,
    }


def _check_manifest_shape(manifest: dict[str, Any]) -> None:
    schema = str(manifest.get("schema_version") or "").strip()
    if schema != _MANIFEST_SCHEMA:
        raise ValueError(f"unsupported deployment manifest schema: {schema!r}")
    image_reference = _validate_image_reference(manifest.get("image_reference", ""))
    if not image_reference:
        raise ValueError("image_reference cannot be empty")


def build_preflight(
    manifest: dict[str, Any],
    *,
    manifest_path: Path,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> DeploymentPreflightResult:
    """Evaluate deployment gates without mutating runtime state."""

    _check_manifest_shape(manifest)
    base_dir = manifest_path.parent.resolve()
    project_dir = _resolve_path(manifest.get("project_root", Path.cwd()), base=base_dir)
    image_reference = _validate_image_reference(str(manifest["image_reference"]))

    env = os.environ.copy()
    env["QUANT_IMAGE"] = image_reference
    for key, value in (manifest.get("compose_environment") or {}).items():
        env[str(key)] = str(value)

    attestation_result: dict[str, Any] = {"status": "degraded"}
    attestation_path_value = manifest.get("attestation_path")
    if attestation_path_value:
        attestation_path = _resolve_path(attestation_path_value, base=base_dir)
        if not attestation_path.exists():
            attestation_result = {
                "status": "degraded",
                "path": str(attestation_path),
                "error": "missing_attestation",
            }
        else:
            attestation = _load_json(attestation_path)
            attestation_result = {
                "status": "healthy" if bool(attestation.get("exact_image_certified", False)) else "degraded",
                "path": str(attestation_path),
                "immutable_registry_digest": str(attestation.get("immutable_registry_digest") or ""),
                "exact_image_certified": bool(attestation.get("exact_image_certified", False)),
                "build_manifest_sha256": str(attestation.get("build_manifest_sha256") or ""),
            }
            if attestation_result["immutable_registry_digest"] and attestation_result["immutable_registry_digest"] != image_reference:
                attestation_result["status"] = "degraded"
                attestation_result["error"] = "attestation_image_mismatch"
    else:
        attestation_result["error"] = "missing_attestation_path"

    compose_files = _default_compose_files(manifest, base=base_dir)
    if compose_files:
        compose_output, compose_digest = _render_compose_config(
            compose_files,
            project_dir=project_dir,
            env=env,
            runner=runner,
        )
        compose_policy = _inspect_compose_read_only_policy(compose_output)
        compose_result = {
            "status": "healthy",
            "project_dir": str(project_dir),
            "compose_files": [str(path) for path in compose_files],
            "configuration_digest": compose_digest,
            "configuration_preview": compose_output[:512],
            "read_only_policy": compose_policy,
        }
        if compose_policy.get("status") == "degraded":
            compose_result["status"] = "degraded"
        elif compose_policy.get("status") == "warning" and compose_result["status"] == "healthy":
            compose_result["status"] = "warning"
    else:
        compose_result = {
            "status": "degraded",
            "project_dir": str(project_dir),
            "compose_files": [],
            "configuration_digest": "",
            "configuration_preview": "",
            "read_only_policy": {"status": "degraded", "error": "missing_compose_files"},
            "error": "missing_compose_files",
        }

    active_model_result: dict[str, Any] = {"status": "degraded"}
    model_registry_root = manifest.get("model_registry_root")
    active_model_version_id = str(manifest.get("active_model_version_id") or "").strip()
    if model_registry_root and active_model_version_id:
        registry_root = _resolve_path(model_registry_root, base=base_dir)
        registry = ModelRegistry(registry_root)
        record = registry.validate_activation_ready(active_model_version_id)
        artifact_manifest = registry.get_artifact_manifest(active_model_version_id)
        artifact_contract = registry._artifact_contract_payload(record)  # noqa: SLF001
        active_model_result = {
            "status": "healthy",
            "version_id": record.version_id,
            "artifact_dir": record.artifact_dir,
            "artifact_manifest_sha256": artifact_contract.get("artifact_manifest_sha256", ""),
            "artifact_image_reference": artifact_contract.get("artifact_image_reference", ""),
            "artifact_feature_schema_sha256": artifact_contract.get("artifact_feature_schema_sha256", ""),
            "artifact_dataset_digest": artifact_contract.get("artifact_dataset_digest", ""),
            "manifest_schema_version": str(artifact_manifest.get("schema_version") or ""),
        }
        if active_model_result["artifact_image_reference"] and active_model_result["artifact_image_reference"] != image_reference:
            active_model_result["status"] = "degraded"
            active_model_result["error"] = "model_image_mismatch"
    else:
        active_model_result["error"] = "missing_model_registry_or_version"

    writable_paths = [str(path).strip() for path in (manifest.get("writable_paths") or []) if str(path).strip()]
    writable_results = [
        _check_write_access(_resolve_path(path, base=base_dir))
        for path in writable_paths
    ]
    writable_status = "healthy"
    if not writable_paths:
        writable_status = "degraded"
        writable_results = [{"status": "degraded", "error": "missing_writable_paths"}]
    elif any(result.get("status") == "degraded" for result in writable_results):
        writable_status = "degraded"

    redis_result = _check_redis_secret(
        _resolve_path(manifest["redis_password_file"], base=base_dir) if manifest.get("redis_password_file") else None,
        str(manifest.get("redis_acl_username") or "").strip() or None,
    )

    dns_result = {"status": "unknown"}
    dns_probe_host = str(manifest.get("dns_probe_host") or "").strip()
    if dns_probe_host:
        dns_result = probe_dns_latency(dns_probe_host)
    else:
        dns_result = {"status": "degraded", "error": "missing_dns_probe_host"}

    database_lock_result = {"status": "unknown"}
    db_path_value = manifest.get("database_path")
    if db_path_value:
        db_path = _resolve_path(db_path_value, base=base_dir)
        database_lock_result = probe_sqlite_lock_latency(db_path)
    else:
        database_lock_result = {"status": "degraded", "error": "missing_database_path"}

    open_positions_result = _check_open_positions(
        _resolve_path(manifest["open_positions_path"], base=base_dir) if manifest.get("open_positions_path") else None
    )
    if open_positions_result.get("status") == "unknown":
        open_positions_result = {
            "status": "degraded",
            "error": "missing_open_positions_path",
            "open_position_count": 0,
        }

    auto_promote = str(
        manifest.get("model_eval_auto_promote", os.environ.get("MODEL_EVAL_AUTO_PROMOTE", "0"))
    ).strip().lower()
    auto_promote_disabled = auto_promote in {"0", "false", "no", "off", ""}
    auto_promote_result = {
        "status": "healthy" if auto_promote_disabled else "degraded",
        "value": auto_promote or "0",
    }

    backup_result: dict[str, Any] = {"status": "unknown"}
    migration_version = str(manifest.get("migration_version") or "").strip()
    backup_path_value = manifest.get("backup_path")
    if db_path_value:
        db_path = _resolve_path(db_path_value, base=base_dir)
        if not db_path.exists():
            backup_result = {"status": "degraded", "error": "missing_database", "path": str(db_path)}
        elif not backup_path_value:
            backup_result = {"status": "degraded", "error": "missing_backup_path", "database_path": str(db_path)}
        else:
            from quant.telebot.models import Base
            from sqlalchemy import create_engine, inspect

            with tempfile.TemporaryDirectory() as tmpdir:
                forward_path = Path(tmpdir) / "forward.db"
                shutil.copy2(db_path, forward_path)
                forward_engine = create_engine(f"sqlite:///{forward_path}")
                Base.metadata.create_all(forward_engine)
                forward_tables = sorted(inspect(forward_engine).get_table_names())
                forward_engine.dispose()

                rollback_tables: list[str] = []
                backup_path = _resolve_path(backup_path_value, base=base_dir)
                if backup_path.exists():
                    rollback_path = Path(tmpdir) / "rollback.db"
                    shutil.copy2(backup_path, rollback_path)
                    rollback_engine = create_engine(f"sqlite:///{rollback_path}")
                    rollback_tables = sorted(inspect(rollback_engine).get_table_names())
                    rollback_engine.dispose()
                    backup_result = {
                        "status": "healthy" if forward_tables else "degraded",
                        "forward_table_count": len(forward_tables),
                        "rollback_table_count": len(rollback_tables),
                        "forward_table_names": forward_tables,
                        "rollback_table_names": rollback_tables,
                        "table_name_drift": sorted(set(forward_tables).symmetric_difference(rollback_tables)),
                        "schema_signature_sha256": _schema_signature(forward_tables),
                        "migration_version": migration_version,
                        "database_path": str(db_path),
                        "backup_path": str(backup_path),
                    }
                else:
                    backup_result = {
                        "status": "degraded",
                        "error": "missing_backup",
                        "backup_path": str(backup_path),
                    }

    smoke_command = manifest.get("smoke_command") or [
        sys.executable,
        "tools/image_smoke.py",
        "--require-release-marker",
    ]
    smoke_result: dict[str, Any] = {"status": "not_run"}

    checks = {
        "image_reference": image_reference,
        "attestation": attestation_result,
        "compose": compose_result,
        "active_model": active_model_result,
        "writable_paths": writable_results,
        "writable_status": writable_status,
        "redis": redis_result,
        "dns": dns_result,
        "database_lock": database_lock_result,
        "open_positions": open_positions_result,
        "auto_promote": auto_promote_result,
        "migrations": backup_result,
        "smoke_command": smoke_command,
        "smoke": smoke_result,
    }

    status = "healthy"
    for key, value in checks.items():
        if isinstance(value, dict) and value.get("status") == "degraded":
            status = "degraded"
            break
    if status == "healthy" and any(
        isinstance(value, dict) and value.get("status") == "warning" for value in checks.values()
    ):
        status = "warning"

    return DeploymentPreflightResult(status=status, checks=checks)


def _compose_command(
    *,
    compose_files: tuple[Path, ...],
    project_dir: Path,
    env: dict[str, str],
) -> list[str]:
    command = ["docker", "compose"]
    for compose_file in compose_files:
        command.extend(["-f", str(compose_file)])
    return command


def _run_services(
    *,
    compose_files: tuple[Path, ...],
    project_dir: Path,
    env: dict[str, str],
    services: tuple[str, ...],
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> list[dict[str, Any]]:
    if not compose_files or not services:
        return []

    base_command = _compose_command(compose_files=compose_files, project_dir=project_dir, env=env)
    results: list[dict[str, Any]] = []
    for service in services:
        completed = runner(
            [*base_command, "up", "-d", service],
            cwd=str(project_dir),
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        results.append(
            {
                "service": service,
                "returncode": completed.returncode,
                "stdout": completed.stdout or "",
                "stderr": completed.stderr or "",
            }
        )
    return results


def _run_compose_down(
    *,
    compose_files: tuple[Path, ...],
    project_dir: Path,
    env: dict[str, str],
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[str, Any]:
    if not compose_files:
        return {"status": "not_run"}

    base_command = _compose_command(compose_files=compose_files, project_dir=project_dir, env=env)
    completed = runner(
        [*base_command, "down", "--remove-orphans"],
        cwd=str(project_dir),
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    return {
        "status": "healthy",
        "returncode": completed.returncode,
        "stdout": completed.stdout or "",
        "stderr": completed.stderr or "",
    }


def _run_smoke(
    smoke_command: object,
    *,
    project_dir: Path,
    env: dict[str, str],
    expected_stdout: str | None = None,
    expected_stdout_sha256: str | None = None,
    expected_stderr_sha256: str | None = None,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[str, Any]:
    if not smoke_command:
        return {"status": "not_run"}

    command = [str(item) for item in smoke_command] if isinstance(smoke_command, (list, tuple)) else [str(smoke_command)]
    completed = runner(
        command,
        cwd=str(project_dir),
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    stdout = completed.stdout or ""
    stderr = completed.stderr or ""
    stdout_digest = _sha256_bytes(stdout.encode("utf-8"))
    stderr_digest = _sha256_bytes(stderr.encode("utf-8"))
    expected_stdout_digest = expected_stdout_sha256 or (_sha256_bytes(expected_stdout.encode("utf-8")) if expected_stdout is not None else "")
    stdout_matches_expected = True
    stderr_matches_expected = True
    if expected_stdout is not None:
        stdout_matches_expected = stdout == expected_stdout
    elif expected_stdout_digest:
        stdout_matches_expected = stdout_digest == expected_stdout_digest
    if expected_stderr_sha256:
        stderr_matches_expected = stderr_digest == expected_stderr_sha256

    status = "healthy"
    if not stdout_matches_expected or not stderr_matches_expected:
        status = "degraded"
    return {
        "status": status,
        "command": command,
        "returncode": completed.returncode,
        "stdout": stdout,
        "stderr": stderr,
        "stdout_digest": stdout_digest,
        "stderr_digest": stderr_digest,
        "expected_stdout": expected_stdout,
        "expected_stdout_sha256": expected_stdout_digest,
        "expected_stderr_sha256": expected_stderr_sha256,
        "stdout_matches_expected": stdout_matches_expected,
        "stderr_matches_expected": stderr_matches_expected,
    }


def _execute_staged_rollout(
    *,
    mode: str,
    compose_files: tuple[Path, ...],
    project_dir: Path,
    env: dict[str, str],
    stages: tuple[dict[str, Any], ...],
    smoke_command: object,
    smoke_expected_stdout: str | None = None,
    smoke_expected_stdout_sha256: str | None = None,
    smoke_expected_stderr_sha256: str | None = None,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[str, Any]:
    """Run a staged rollout and capture rollback attempts on failure."""

    compose_down_result: dict[str, Any] = {"status": "not_run"}
    stage_results: list[dict[str, Any]] = []
    service_results: list[dict[str, Any]] = []
    smoke_result: dict[str, Any] = {"status": "not_run"}
    rollout_status = "healthy"
    rollout_error: str | None = None

    try:
        if mode == "rollback":
            compose_down_result = _run_compose_down(
                compose_files=compose_files,
                project_dir=project_dir,
                env=env,
                runner=runner,
            )

        for stage in stages:
            stage_name = str(stage.get("name") or "stage").strip() or "stage"
            stage_services = tuple(str(service).strip() for service in (stage.get("services") or ()) if str(service).strip())
            stage_service_results = _run_services(
                compose_files=compose_files,
                project_dir=project_dir,
                env=env,
                services=stage_services,
                runner=runner,
            )
            stage_status = "healthy" if stage_service_results else "warning"
            service_results.extend(stage_service_results)
            stage_results.append(
                {
                    "name": stage_name,
                    "status": stage_status,
                    "services": stage_service_results,
                }
            )
        smoke_result = _run_smoke(
            smoke_command,
            project_dir=project_dir,
            env=env,
            expected_stdout=smoke_expected_stdout,
            expected_stdout_sha256=smoke_expected_stdout_sha256,
            expected_stderr_sha256=smoke_expected_stderr_sha256,
            runner=runner,
        )
    except Exception as exc:
        rollout_status = "degraded"
        rollout_error = exc.__class__.__name__
        if compose_files:
            try:
                compose_down_result = _run_compose_down(
                    compose_files=compose_files,
                    project_dir=project_dir,
                    env=env,
                    runner=runner,
                )
            except Exception as rollback_exc:
                compose_down_result = {
                    "status": "degraded",
                    "error": rollback_exc.__class__.__name__,
                }

    if rollout_status == "healthy":
        if compose_down_result.get("status") == "degraded":
            rollout_status = "warning"
        elif smoke_result.get("status") == "degraded":
            rollout_status = "degraded"

    return {
        "status": rollout_status,
        "error": rollout_error,
        "stages": stage_results,
        "service_results": service_results,
        "smoke_result": smoke_result,
        "compose_down_result": compose_down_result,
    }


def build_deployment_record(
    manifest: dict[str, Any],
    *,
    manifest_path: Path,
    mode: str,
    preflight: DeploymentPreflightResult,
    deploy_services: list[dict[str, Any]],
    smoke_result: dict[str, Any],
    smoke_expected_stdout: str | None,
    smoke_expected_stdout_sha256: str | None,
    smoke_expected_stderr_sha256: str | None,
    model_result: dict[str, Any],
    compose_result: dict[str, Any],
    attestation_result: dict[str, Any],
    rollout_result: dict[str, Any],
    stage_results: list[dict[str, Any]],
    rollback_target_image_reference: str = "",
    rollback_target_model_version_id: str = "",
) -> dict[str, Any]:
    """Create a stable deployment record."""

    manifest_digest = _sha256_bytes(manifest_path.read_bytes())
    record = {
        "schema_version": _RECORD_SCHEMA,
        "mode": mode,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "idempotency_key": _sha256_json(
            {
                "mode": mode,
                "manifest_digest": manifest_digest,
                "image_reference": manifest.get("image_reference"),
                "active_model_version_id": manifest.get("active_model_version_id"),
                "migration_version": manifest.get("migration_version"),
                "service_order": list(_normalise_service_order(manifest.get("service_order"))),
                "stage_plan": manifest.get("stage_plan"),
            }
        ),
        "image_reference": str(manifest.get("image_reference") or ""),
        "environment_manifest_path": str(manifest_path),
        "environment_manifest_sha256": manifest_digest,
        "compose_project_dir": compose_result.get("project_dir"),
        "compose_files": compose_result.get("compose_files", []),
        "compose_configuration_sha256": compose_result.get("configuration_digest"),
        "attestation": attestation_result,
        "model": model_result,
        "migrations": preflight.checks.get("migrations", {}),
        "preflight": preflight.checks,
        "smoke": smoke_result,
        "smoke_expected_stdout": smoke_expected_stdout,
        "smoke_expected_stdout_sha256": smoke_expected_stdout_sha256,
        "smoke_expected_stderr_sha256": smoke_expected_stderr_sha256,
        "deploy_services": deploy_services,
        "stages": stage_results,
        "compose_down": rollout_result.get("compose_down_result", preflight.checks.get("compose_down", {"status": "not_run"})),
        "rollout": rollout_result,
        "rollback_target_image_reference": rollback_target_image_reference,
        "rollback_target_model_version_id": rollback_target_model_version_id,
        "operator_approvals": list(manifest.get("operator_approvals") or []),
        "recorded_model_digest": model_result.get("artifact_manifest_sha256", ""),
        "recorded_migration_version": str(manifest.get("migration_version") or ""),
        "recorded_smoke_status": smoke_result.get("status", "not_run"),
    }
    record["record_sha256"] = _sha256_json(record)
    return record


def run(
    manifest_path: Path,
    *,
    mode: str = "preflight",
    execute: bool = False,
    output_dir: Path | None = None,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[str, Any]:
    """Run a deployment-readiness operation and persist a stable record."""

    manifest_path = manifest_path.expanduser().resolve()
    manifest = _load_json(manifest_path)
    effective_manifest = dict(manifest)
    if mode == "rollback":
        rollback_image = str(manifest.get("rollback_image_reference") or "").strip()
        if rollback_image:
            effective_manifest["image_reference"] = rollback_image
        rollback_model = str(manifest.get("rollback_model_version_id") or "").strip()
        if rollback_model:
            effective_manifest["active_model_version_id"] = rollback_model

    preflight = build_preflight(effective_manifest, manifest_path=manifest_path, runner=runner)

    base_dir = manifest_path.parent.resolve()
    project_dir = _resolve_path(effective_manifest.get("project_root", Path.cwd()), base=base_dir)
    image_reference = _validate_image_reference(str(effective_manifest["image_reference"]))
    env = os.environ.copy()
    env["QUANT_IMAGE"] = image_reference
    for key, value in (effective_manifest.get("compose_environment") or {}).items():
        env[str(key)] = str(value)

    compose_files = _default_compose_files(effective_manifest, base=base_dir)
    deploy_services = _normalise_service_order(effective_manifest.get("service_order"))
    if not deploy_services:
        deploy_services = ("redis", "model_evaluator", "retrain_scheduler", "telegram_bot")
    stage_plan = _normalise_stage_plan(effective_manifest.get("stage_plan"), fallback_services=deploy_services)
    service_results: list[dict[str, Any]] = []
    smoke_result: dict[str, Any] = {"status": "not_run"}
    smoke_expected_stdout = (
        str(effective_manifest["smoke_expected_stdout"])
        if effective_manifest.get("smoke_expected_stdout") is not None
        else None
    )
    smoke_expected_stdout_sha256 = (
        str(effective_manifest["smoke_expected_stdout_sha256"])
        if effective_manifest.get("smoke_expected_stdout_sha256") is not None
        else None
    )
    smoke_expected_stderr_sha256 = (
        str(effective_manifest["smoke_expected_stderr_sha256"])
        if effective_manifest.get("smoke_expected_stderr_sha256") is not None
        else None
    )
    rollout_result: dict[str, Any] = {
        "status": "not_run",
        "error": None,
        "service_results": [],
        "smoke_result": smoke_result,
        "compose_down_result": {"status": "not_run"},
    }

    if execute:
        rollout_result = _execute_staged_rollout(
            mode=mode,
            compose_files=compose_files,
            project_dir=project_dir,
            env=env,
            stages=stage_plan,
            smoke_command=effective_manifest.get("smoke_command"),
            smoke_expected_stdout=smoke_expected_stdout,
            smoke_expected_stdout_sha256=smoke_expected_stdout_sha256,
            smoke_expected_stderr_sha256=smoke_expected_stderr_sha256,
            runner=runner,
        )
        service_results = list(rollout_result.get("service_results", []))
        smoke_result = dict(rollout_result.get("smoke_result", smoke_result))
        preflight.checks["compose_down"] = dict(rollout_result.get("compose_down_result", {"status": "not_run"}))
        preflight.checks["rollout"] = rollout_result

    attestation_result = preflight.checks.get("attestation", {})
    compose_result = preflight.checks.get("compose", {})
    model_result = preflight.checks.get("active_model", {})
    record = build_deployment_record(
        effective_manifest,
        manifest_path=manifest_path,
        mode=mode,
        preflight=preflight,
        deploy_services=service_results,
        smoke_result=smoke_result,
        smoke_expected_stdout=smoke_expected_stdout,
        smoke_expected_stdout_sha256=smoke_expected_stdout_sha256,
        smoke_expected_stderr_sha256=smoke_expected_stderr_sha256,
        model_result=model_result,
        compose_result=compose_result,
        attestation_result=attestation_result,
        rollout_result=rollout_result,
        stage_results=list(rollout_result.get("stages", [])),
        rollback_target_image_reference=str(manifest.get("rollback_image_reference") or "") if mode == "rollback" else "",
        rollback_target_model_version_id=str(manifest.get("rollback_model_version_id") or "") if mode == "rollback" else "",
    )

    record_dir = output_dir or _resolve_path(manifest.get("deployment_record_dir", ".build/deployments"), base=base_dir)
    record_dir.mkdir(parents=True, exist_ok=True)
    record_path = record_dir / f"{record['mode']}-{record['idempotency_key']}.json"
    if record_path.exists():
        existing = _load_json(record_path)
        if existing.get("idempotency_key") == record["idempotency_key"]:
            return existing

    record_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate and record WP-15 deployment readiness")
    parser.add_argument("--manifest", type=Path, required=True, help="Deployment environment manifest JSON")
    parser.add_argument(
        "--mode",
        choices=("preflight", "deploy", "rollback"),
        default="preflight",
        help="Preflight or idempotent rollout mode",
    )
    parser.add_argument("--execute", action="store_true", help="Actually run compose service start and smoke checks")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory for the deployment record")
    args = parser.parse_args()

    record = run(
        args.manifest,
        mode=args.mode,
        execute=args.execute,
        output_dir=args.output_dir,
    )
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
