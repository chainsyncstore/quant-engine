"""Optional runtime probes for DNS, database lock, and restart telemetry."""

from __future__ import annotations

import json
import os
import socket
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any
from urllib.parse import urlparse

from quant_v2.config import get_runtime_profile


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _safe_int(value: object, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _resolve_target_host(target: str | None) -> tuple[str, int | None]:
    raw = str(target or "").strip()
    if not raw:
        return "", None

    if "://" in raw:
        parsed = urlparse(raw)
        host = parsed.hostname or ""
        port = parsed.port
        if not host and parsed.path:
            host = parsed.path.split(":", 1)[0]
        return host, port

    if raw.count(":") == 1 and not raw.startswith("["):
        host, port_raw = raw.rsplit(":", 1)
        try:
            return host, int(port_raw)
        except ValueError:
            return host, None

    return raw, None


def resolve_runtime_boot_marker_path(path: Path | str | None = None) -> Path:
    if path is not None:
        return Path(path).expanduser()
    runtime_root = get_runtime_profile().project_root / ".runtime"
    return runtime_root / "execution_runtime.json"


def read_runtime_boot_marker(path: Path | str | None = None) -> dict[str, Any]:
    marker_path = resolve_runtime_boot_marker_path(path)
    if not marker_path.exists():
        return {}

    try:
        raw = json.loads(marker_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(raw, dict):
        return {}
    return raw


def record_runtime_boot_marker(path: Path | str | None = None) -> dict[str, Any]:
    marker_path = resolve_runtime_boot_marker_path(path)
    current = read_runtime_boot_marker(marker_path)
    boot_count = _safe_int(current.get("boot_count", 0)) + 1
    payload = {
        "boot_count": boot_count,
        "restart_count": max(boot_count - 1, 0),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "host": socket.gethostname(),
        "process_id": os.getpid(),
    }
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = marker_path.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp_path.replace(marker_path)
    return payload


def probe_dns_latency(target: str | None = None) -> dict[str, Any]:
    host, port = _resolve_target_host(target)
    if not host:
        return {
            "status": "unknown",
            "target_host": "",
            "target_port": None,
            "latency_ms": 0.0,
            "address_count": 0,
        }

    started = perf_counter()
    try:
        addresses = socket.getaddrinfo(host, port or 0, type=socket.SOCK_STREAM)
    except Exception as exc:
        return {
            "status": "degraded",
            "target_host": host,
            "target_port": port,
            "latency_ms": max((perf_counter() - started) * 1000.0, 0.0),
            "address_count": 0,
            "error": str(exc),
        }

    latency_ms = max((perf_counter() - started) * 1000.0, 0.0)
    status = "healthy"
    if latency_ms >= 100.0:
        status = "warning"
    if latency_ms >= 500.0:
        status = "degraded"

    return {
        "status": status,
        "target_host": host,
        "target_port": port,
        "latency_ms": latency_ms,
        "address_count": len(addresses),
    }


def probe_sqlite_lock_latency(
    db_path: Path | str | None = None,
    *,
    timeout_seconds: float = 0.25,
) -> dict[str, Any]:
    raw_path = str(db_path or "").strip()
    if not raw_path:
        raw_path = os.getenv("BOT_DB_PATH", "").strip()
    if not raw_path:
        return {
            "status": "unknown",
            "db_path": "",
            "lock_latency_ms": 0.0,
            "lock_acquired": False,
        }

    path = Path(raw_path).expanduser()
    if not path.exists():
        return {
            "status": "unknown",
            "db_path": str(path),
            "lock_latency_ms": 0.0,
            "lock_acquired": False,
        }

    started = perf_counter()
    try:
        with sqlite3.connect(str(path), timeout=max(timeout_seconds, 0.05)) as conn:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute("ROLLBACK")
    except sqlite3.OperationalError as exc:
        return {
            "status": "degraded",
            "db_path": str(path),
            "lock_latency_ms": max((perf_counter() - started) * 1000.0, 0.0),
            "lock_acquired": False,
            "error": str(exc),
        }
    except Exception as exc:
        return {
            "status": "degraded",
            "db_path": str(path),
            "lock_latency_ms": max((perf_counter() - started) * 1000.0, 0.0),
            "lock_acquired": False,
            "error": str(exc),
        }

    lock_latency_ms = max((perf_counter() - started) * 1000.0, 0.0)
    status = "healthy"
    if lock_latency_ms >= 50.0:
        status = "warning"
    if lock_latency_ms >= 200.0:
        status = "degraded"

    return {
        "status": status,
        "db_path": str(path),
        "lock_latency_ms": lock_latency_ms,
        "lock_acquired": True,
    }
