"""Health dashboard artifacts for research runs and live sessions."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from quant_v2.config import get_runtime_profile
from quant_v2.monitoring.audit_sink import JsonlAuditSink
from quant_v2.monitoring.kill_switch import KillSwitchEvaluation
from quant_v2.monitoring.runtime_probes import (
    probe_dns_latency,
    probe_sqlite_lock_latency,
    read_runtime_boot_marker,
)

try:  # pragma: no cover - optional runtime dependency
    import psutil as _psutil
except ImportError:  # pragma: no cover - exercised when psutil is unavailable
    _psutil = None


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _extract_dataset_manifest(report: dict[str, Any]) -> dict[str, Any]:
    dataset = report.get("dataset", {})
    if isinstance(dataset, dict):
        manifest = dataset.get("manifest")
        if isinstance(manifest, dict):
            return manifest
    manifest = report.get("dataset_manifest")
    if isinstance(manifest, dict):
        return manifest
    return {}


def _extract_provider_probe(report: dict[str, Any]) -> dict[str, Any]:
    probes = report.get("provider_probes", {})
    if isinstance(probes, dict):
        market_data = probes.get("market_data")
        if isinstance(market_data, dict):
            return market_data
    market_data = report.get("market_data_probe")
    if isinstance(market_data, dict):
        return market_data
    return {}


def _extract_execution_health(report: dict[str, Any]) -> dict[str, Any]:
    execution_health = report.get("execution_health")
    if isinstance(execution_health, dict):
        return execution_health
    return {}


def _extract_feature_drift_alert(report: dict[str, Any]) -> bool:
    if bool(report.get("feature_drift_alert", False)):
        return True
    monitoring_snapshot = report.get("monitoring_snapshot")
    if isinstance(monitoring_snapshot, dict) and bool(monitoring_snapshot.get("feature_drift_alert", False)):
        return True
    execution_health = report.get("execution_health")
    if isinstance(execution_health, dict) and bool(execution_health.get("feature_drift_alert", False)):
        return True
    return False


def build_execution_health_dashboard(report: dict[str, Any]) -> dict[str, Any]:
    """Summarize execution queue, WAL, and reconciliation freshness."""

    execution_health = _extract_execution_health(report)
    feature_drift_alert = _extract_feature_drift_alert(report)
    if not execution_health:
        return {"feature_drift_alert": feature_drift_alert}

    stream_health = execution_health.get("stream_bus", {})
    wal_health = execution_health.get("wal", {})
    reconciliation_health = execution_health.get("reconciliation", {})

    if not isinstance(stream_health, dict):
        stream_health = {}
    if not isinstance(wal_health, dict):
        wal_health = {}
    if not isinstance(reconciliation_health, dict):
        reconciliation_health = {}
    redis_memory = execution_health.get("redis_memory", {})
    if not isinstance(redis_memory, dict):
        redis_memory = {}

    return {
        "status": str(execution_health.get("status", "unknown") or "unknown"),
        "captured_at": str(execution_health.get("captured_at") or ""),
        "live_session_count": int(execution_health.get("live_session_count", 0) or 0),
        "stream_status": str(stream_health.get("status", "unknown") or "unknown"),
        "stream_queue_lag_seconds": float(
            stream_health.get(
                "max_pending_age_seconds",
                stream_health.get("stream_queue_lag_seconds", 0.0),
            )
            or 0.0
        ),
        "stream_backlog_count": int(stream_health.get("stream_backlog_count", 0) or 0),
        "stream_pending_count": int(stream_health.get("pending_count", 0) or 0),
        "stream_lag_entries": int(stream_health.get("lag_entries", 0) or 0),
        "stream_latest_entry_age_seconds": float(stream_health.get("stream_age_seconds", 0.0) or 0.0),
        "wal_status": str(wal_health.get("status", "unknown") or "unknown"),
        "wal_entry_count": int(wal_health.get("entry_count", 0) or 0),
        "wal_latest_entry_age_seconds": float(
            wal_health.get("latest_entry_age_seconds", 0.0) or 0.0
        ),
        "wal_oldest_entry_age_seconds": float(
            wal_health.get("oldest_entry_age_seconds", 0.0) or 0.0
        ),
        "reconciliation_status": str(reconciliation_health.get("status", "unknown") or "unknown"),
        "reconciliation_lag_seconds": reconciliation_health.get("lag_seconds"),
        "reconciliation_error_at": str(reconciliation_health.get("last_error_at") or ""),
        "redis_memory_status": str(redis_memory.get("status", "unknown") or "unknown"),
        "redis_memory_used_mb": float(redis_memory.get("used_memory_mb", 0.0) or 0.0),
        "redis_memory_max_mb": float(redis_memory.get("maxmemory_mb", 0.0) or 0.0),
        "redis_memory_used_fraction": redis_memory.get("used_memory_fraction"),
        "feature_drift_alert": feature_drift_alert,
    }


def _objective_target(value: object, *, multiplier: float = 1.2) -> float:
    observed = _safe_float(value)
    if observed <= 0.0:
        return 0.0
    return round(observed * multiplier, 6)


def build_service_level_baseline(payload: dict[str, Any]) -> dict[str, Any]:
    """Derive initial SLI objectives from the measured runtime baseline."""

    data_health = payload.get("data_health", {}) or {}
    runtime_health = payload.get("runtime_health", {}) or {}
    execution_health = payload.get("execution_health", {}) or {}
    diagnostics = payload.get("performance_diagnostics", {}) or {}

    observed = {
        "cpu_percent": _safe_float(runtime_health.get("cpu_percent", 0.0)),
        "memory_percent": _safe_float(runtime_health.get("memory_percent", 0.0)),
        "disk_percent": _safe_float(runtime_health.get("disk_percent", 0.0)),
        "dns_latency_ms": _safe_float((runtime_health.get("dns_probe", {}) or {}).get("latency_ms", 0.0)),
        "database_lock_latency_ms": _safe_float((runtime_health.get("database_lock_probe", {}) or {}).get("lock_latency_ms", 0.0)),
        "stream_queue_lag_seconds": _safe_float(execution_health.get("stream_queue_lag_seconds", 0.0)),
        "wal_latest_entry_age_seconds": _safe_float(execution_health.get("wal_latest_entry_age_seconds", 0.0)),
        "reconciliation_lag_seconds": _safe_float((execution_health.get("reconciliation", {}) or {}).get("lag_seconds", 0.0)),
        "data_freshness_minutes": _safe_float(data_health.get("freshness_minutes", 0.0)),
        "provider_latency_ms_max": _safe_float(data_health.get("provider_latency_ms_max", 0.0)),
        "provider_rate_limit_used_weight_1m": _safe_float(data_health.get("provider_rate_limit_used_weight_1m", 0.0)),
        "provider_rate_limit_pressure_fraction": _safe_float(data_health.get("provider_rate_limit_pressure_fraction", 0.0)),
        "feature_missingness_max": _safe_float(data_health.get("feature_missingness_max", 0.0)),
        "quarantined_symbol_count": _safe_float(len(data_health.get("quarantined_symbols", []) or [])),
        "probability_calibration_mae": _safe_float(diagnostics.get("candidate_probability_calibration_mae", 0.0)),
        "probability_drift_mae": _safe_float(diagnostics.get("candidate_incumbent_probability_drift_mae", 0.0)),
        "turnover": _safe_float(diagnostics.get("candidate_turnover", 0.0)),
        "realized_cost_bps": _safe_float(diagnostics.get("candidate_realized_cost_bps", 0.0)),
        "feature_drift_alert": bool(payload.get("feature_drift_alert", False)),
    }

    targets = {key: _objective_target(value) for key, value in observed.items()}

    return {
        "source": "measured",
        "status": "healthy",
        "observed": observed,
        "targets": targets,
        "notes": "Initial service-level objectives derived from the measured current-state baseline.",
    }


def build_data_health_dashboard(report: dict[str, Any]) -> dict[str, Any]:
    """Summarize dataset freshness, coverage, and integrity."""

    manifest = _extract_dataset_manifest(report)
    provider_probe = _extract_provider_probe(report)
    coverage_by_symbol = manifest.get("coverage_by_symbol", {})
    gap_stats_per_symbol = manifest.get("gap_stats_per_symbol", {})
    duplicate_rows_per_symbol = manifest.get("duplicate_rows_per_symbol", {})
    failed_symbols = manifest.get("failed_symbols", {})
    null_ratio_by_column = manifest.get("null_ratio_by_column", {})

    coverage_values: list[float] = []
    min_coverage_by_symbol: dict[str, float] = {}
    for symbol, columns in coverage_by_symbol.items():
        if isinstance(columns, dict):
            values = [_safe_float(value) for value in columns.values()]
            coverage = min(values) if values else 0.0
            min_coverage_by_symbol[str(symbol)] = coverage
            coverage_values.append(coverage)

    duplicate_rows_total = sum(int(v or 0) for v in duplicate_rows_per_symbol.values()) if isinstance(duplicate_rows_per_symbol, dict) else 0
    gap_count_total = 0
    max_gap_count = 0
    for stats in gap_stats_per_symbol.values() if isinstance(gap_stats_per_symbol, dict) else []:
        if isinstance(stats, dict):
            gap_count = int(stats.get("gap_count", 0) or 0)
            gap_count_total += gap_count
            max_gap_count = max(max_gap_count, gap_count)

    source_retrieved_at = str(manifest.get("source_retrieved_at") or "")
    freshness_minutes = 0.0
    if source_retrieved_at:
        try:
            retrieved_at = datetime.fromisoformat(source_retrieved_at)
            if retrieved_at.tzinfo is None:
                retrieved_at = retrieved_at.replace(tzinfo=timezone.utc)
            freshness_minutes = max((datetime.now(timezone.utc) - retrieved_at).total_seconds() / 60.0, 0.0)
        except ValueError:
            freshness_minutes = 0.0

    status = "healthy"
    if failed_symbols or duplicate_rows_total > 0 or gap_count_total > 0:
        status = "degraded"
    elif coverage_values and min(coverage_values) < 0.98:
        status = "warning"

    feature_missingness_by_column = {
        str(column): _safe_float(value)
        for column, value in (null_ratio_by_column.items() if isinstance(null_ratio_by_column, dict) else [])
    }
    feature_missingness_max = max(feature_missingness_by_column.values()) if feature_missingness_by_column else 0.0
    feature_missingness_avg = (
        sum(feature_missingness_by_column.values()) / len(feature_missingness_by_column)
        if feature_missingness_by_column
        else 0.0
    )
    quarantined_symbols = sorted(str(symbol) for symbol in failed_symbols) if isinstance(failed_symbols, dict) else []

    provider_status = str(provider_probe.get("status", "unknown") or "unknown")
    if provider_status == "degraded":
        status = "degraded"
    elif provider_status == "warning" and status == "healthy":
        status = "warning"

    return {
        "dataset_name": str(manifest.get("dataset_name") or ""),
        "source_retrieved_at": source_retrieved_at,
        "freshness_minutes": freshness_minutes,
        "n_symbols": int(manifest.get("n_symbols", 0) or 0),
        "failed_symbol_count": len(failed_symbols) if isinstance(failed_symbols, dict) else 0,
        "failed_symbols": dict(failed_symbols) if isinstance(failed_symbols, dict) else {},
        "duplicate_rows_total": duplicate_rows_total,
        "duplicate_rows_per_symbol": dict(duplicate_rows_per_symbol) if isinstance(duplicate_rows_per_symbol, dict) else {},
        "gap_count_total": gap_count_total,
        "max_gap_count_per_symbol": max_gap_count,
        "coverage_min_by_symbol": min_coverage_by_symbol,
        "min_coverage_fraction": min(coverage_values) if coverage_values else 0.0,
        "avg_coverage_fraction": (sum(coverage_values) / len(coverage_values)) if coverage_values else 0.0,
        "feature_missingness_by_column": feature_missingness_by_column,
        "feature_missingness_max": feature_missingness_max,
        "feature_missingness_avg": feature_missingness_avg,
        "quarantined_symbols": quarantined_symbols,
        "provider_status": provider_status,
        "provider_probe_count": int(provider_probe.get("probe_count", 0) or 0),
        "provider_failure_count": int(provider_probe.get("failure_count", 0) or 0),
        "provider_stale_count": int(provider_probe.get("stale_count", 0) or 0),
        "provider_circuit_breaker_triggered": bool(provider_probe.get("circuit_breaker_triggered", False)),
        "provider_latency_ms_max": float(provider_probe.get("latency_ms_max", 0.0) or 0.0),
        "provider_latency_ms_avg": float(provider_probe.get("latency_ms_avg", 0.0) or 0.0),
        "provider_rate_limit_status": str(provider_probe.get("rate_limit_status", "unknown") or "unknown"),
        "provider_rate_limit_used_weight_1m": int(provider_probe.get("rate_limit_used_weight_1m", 0) or 0),
        "provider_rate_limit_weight_limit_1m": int(provider_probe.get("rate_limit_weight_limit_1m", 0) or 0),
        "provider_rate_limit_headroom_1m": int(provider_probe.get("rate_limit_headroom_1m", 0) or 0),
        "provider_rate_limit_pressure_fraction": provider_probe.get("rate_limit_pressure_fraction"),
        "provider_rate_limit_snapshot": dict(provider_probe.get("rate_limit_snapshot") or {}),
        "provider_freshness_minutes_max": provider_probe.get("freshness_minutes_max"),
        "provider_freshness_minutes_avg": provider_probe.get("freshness_minutes_avg"),
        "provider_stale_symbols": list(provider_probe.get("stale_symbols", [])),
        "provider_failed_symbols": dict(provider_probe.get("failed_symbols", {})),
        "status": status,
    }


def build_runtime_resource_health() -> dict[str, Any]:
    """Collect lightweight runtime resource indicators for the current process."""

    root_path = get_runtime_profile().project_root
    status = "unknown"
    cpu_percent = 0.0
    memory_percent = 0.0
    disk_percent = 0.0
    rss_mb = 0.0
    open_files = 0
    load_avg_1m = None
    process_uptime_seconds = None
    container_restart_count = None
    container_started_at = ""
    dns_probe: dict[str, Any] = {}
    database_lock_probe: dict[str, Any] = {}
    boot_marker = read_runtime_boot_marker()

    if _psutil is not None:
        try:
            process = _psutil.Process()
            cpu_percent = float(process.cpu_percent(interval=None) or 0.0)
            rss_mb = float(process.memory_info().rss or 0.0) / (1024.0 * 1024.0)
            open_files = len(process.open_files())
            process_uptime_seconds = max(datetime.now(timezone.utc).timestamp() - float(process.create_time()), 0.0)
            memory_percent = float(getattr(_psutil.virtual_memory(), "percent", 0.0) or 0.0)
            disk_percent = float(getattr(_psutil.disk_usage(root_path), "percent", 0.0) or 0.0)
            if hasattr(_psutil, "getloadavg"):
                try:
                    load_avg_1m = float(_psutil.getloadavg()[0])
                except Exception:
                    load_avg_1m = None
        except Exception:
            status = "unknown"
        else:
            status = "healthy"

    if isinstance(boot_marker, dict) and boot_marker:
        container_restart_count = int(boot_marker.get("restart_count", 0) or 0)
        container_started_at = str(boot_marker.get("started_at") or "")
        marker_status = str(boot_marker.get("status", "healthy") or "healthy")
        if marker_status == "degraded" and status == "healthy":
            status = "warning"

    dns_probe = probe_dns_latency(os.getenv("BOT_DNS_PROBE_HOST") or os.getenv("REDIS_URL"))
    database_lock_probe = probe_sqlite_lock_latency(os.getenv("BOT_DB_PATH"))

    if status == "healthy":
        if cpu_percent >= 90.0 or memory_percent >= 90.0 or disk_percent >= 90.0:
            status = "degraded"
        elif cpu_percent >= 75.0 or memory_percent >= 80.0 or disk_percent >= 85.0:
            status = "warning"
    if dns_probe.get("status") == "degraded" or database_lock_probe.get("status") == "degraded":
        status = "degraded"
    elif (
        dns_probe.get("status") == "warning"
        or database_lock_probe.get("status") == "warning"
    ) and status == "healthy":
        status = "warning"

    return {
        "status": status,
        "project_root": str(root_path),
        "cpu_percent": cpu_percent,
        "memory_percent": memory_percent,
        "disk_percent": disk_percent,
        "rss_mb": rss_mb,
        "open_file_count": open_files,
        "load_avg_1m": load_avg_1m,
        "process_uptime_seconds": process_uptime_seconds,
        "container_restart_count": container_restart_count,
        "container_started_at": container_started_at,
        "dns_probe": dns_probe,
        "database_lock_probe": database_lock_probe,
    }


def build_run_health_dashboard(report: dict[str, Any]) -> dict[str, Any]:
    """Build compact run-level health payload from validation report."""

    scorecard = report.get("scorecard", {})
    gates = scorecard.get("gates", {})
    forward_live = report.get("forward_live_simulation", {})
    replay = report.get("replay_regression", {})

    score = float(scorecard.get("score", 0.0) or 0.0)
    gates_passed = bool(gates.get("passed", False))

    replay_agg = replay.get("aggregate", {}) if isinstance(replay, dict) else {}
    mean_abs_ev_delta = float(replay_agg.get("mean_abs_ev_delta", 0.0) or 0.0)

    aggregate = forward_live.get("aggregate", {}) if isinstance(forward_live, dict) else {}
    stability = float(aggregate.get("stability_score", 0.0) or 0.0)
    data_health = build_data_health_dashboard(report)
    runtime_health = build_runtime_resource_health()
    execution_health = build_execution_health_dashboard(report)
    service_level_baseline = build_service_level_baseline(
        {
            "data_health": data_health,
            "runtime_health": runtime_health,
            "execution_health": execution_health,
            "performance_diagnostics": report.get("performance_diagnostics", {}),
            "feature_drift_alert": bool(execution_health.get("feature_drift_alert", False)),
        }
    )

    status = "healthy"
    if not gates_passed or score < 70.0:
        status = "warning"
    if mean_abs_ev_delta > 0.35:
        status = "degraded"
    if data_health["status"] == "degraded":
        status = "degraded"
    elif data_health["status"] == "warning" and status == "healthy":
        status = "warning"
    if runtime_health["status"] == "degraded":
        status = "degraded"
    elif runtime_health["status"] == "warning" and status == "healthy":
        status = "warning"
    if execution_health.get("status") == "degraded":
        status = "degraded"
    elif execution_health.get("status") == "warning" and status == "healthy":
        status = "warning"
    if execution_health.get("feature_drift_alert"):
        status = "degraded"

    return {
        "run_id": str(report.get("run_id") or ""),
        "timestamp": str(report.get("timestamp") or ""),
        "status": status,
        "score": score,
        "gates_passed": gates_passed,
        "stability_score": stability,
        "mean_abs_ev_delta": mean_abs_ev_delta,
        "data_health": data_health,
        "runtime_health": runtime_health,
        "execution_health": execution_health,
        "service_level_baseline": service_level_baseline,
        "feature_drift_alert": bool(execution_health.get("feature_drift_alert", False)),
    }


def render_run_health_summary(payload: dict[str, Any]) -> str:
    """Render bot-friendly single-run health summary."""

    data_health = payload.get("data_health", {}) or {}
    runtime_health = payload.get("runtime_health", {}) or {}
    execution_health = payload.get("execution_health", {}) or {}

    return "\n".join(
        [
            "Run Health Dashboard:",
            f"- Run ID: {payload.get('run_id', '')}",
            f"- Status: {payload.get('status', 'unknown')}",
            f"- Score: {float(payload.get('score', 0.0)):.2f}",
            f"- Gates passed: {bool(payload.get('gates_passed', False))}",
            f"- Stability: {float(payload.get('stability_score', 0.0)):.3f}",
            f"- Mean |EV delta|: {float(payload.get('mean_abs_ev_delta', 0.0)):.4f}",
            (
                "- Data health: "
                f"{data_health.get('status', 'unknown')} "
                f"(freshness={float(data_health.get('freshness_minutes', 0.0)):.1f}m, "
                f"coverage={float(data_health.get('min_coverage_fraction', 0.0)):.3f}, "
                f"missingness={float(data_health.get('feature_missingness_max', 0.0)):.3f}, "
                f"quarantined={len(data_health.get('quarantined_symbols', []) or [])})"
            ),
            (
                "- Runtime health: "
                f"{runtime_health.get('status', 'unknown')} "
                f"(cpu={float(runtime_health.get('cpu_percent', 0.0)):.1f}%, "
                f"mem={float(runtime_health.get('memory_percent', 0.0)):.1f}%, "
                f"disk={float(runtime_health.get('disk_percent', 0.0)):.1f}%)"
            ),
            (
                "- Runtime signals: "
                f"dns={runtime_health.get('dns_probe', {}).get('status', 'unknown')} "
                f"(latency={float(runtime_health.get('dns_probe', {}).get('latency_ms', 0.0)):.1f}ms), "
                f"db={runtime_health.get('database_lock_probe', {}).get('status', 'unknown')} "
                f"(latency={float(runtime_health.get('database_lock_probe', {}).get('lock_latency_ms', 0.0)):.1f}ms), "
                f"restarts={runtime_health.get('container_restart_count', 'unknown')}"
            ),
            (
                "- Provider health: "
                f"{data_health.get('provider_status', 'unknown')} "
                f"(probes={int(data_health.get('provider_probe_count', 0) or 0)}, "
                f"stale={int(data_health.get('provider_stale_count', 0) or 0)}, "
                f"failed={int(data_health.get('provider_failure_count', 0) or 0)}, "
                f"rate_limit={data_health.get('provider_rate_limit_status', 'unknown')} "
                f"(used={int(data_health.get('provider_rate_limit_used_weight_1m', 0) or 0)}/"
                f"{int(data_health.get('provider_rate_limit_weight_limit_1m', 0) or 0)}, "
                f"headroom={int(data_health.get('provider_rate_limit_headroom_1m', 0) or 0)}, "
                f"pressure={float(data_health.get('provider_rate_limit_pressure_fraction') or 0.0):.3f})"
            ),
            *(
                [
                    (
                        "- Execution health: "
                        f"{execution_health.get('status', 'unknown')} "
                        f"(queue={execution_health.get('stream_status', 'unknown')}, "
                        f"backlog={int(execution_health.get('stream_backlog_count', 0) or 0)}, "
                        f"wal={execution_health.get('wal_status', 'unknown')}, "
                        f"recon={execution_health.get('reconciliation_status', 'unknown')})"
                    )
                ]
                if execution_health
                else []
            ),
            *(
                [
                    (
                        "- Redis memory: "
                        f"{execution_health.get('redis_memory_status', 'unknown')} "
                        f"(used={float(execution_health.get('redis_memory_used_mb', 0.0)):.1f}MB, "
                        f"max={float(execution_health.get('redis_memory_max_mb', 0.0)):.1f}MB)"
                    )
                ]
                if execution_health
                and execution_health.get("redis_memory_status") not in ("", "unknown", None)
                else []
            ),
            (
                "- SLI baseline: "
                f"{payload.get('service_level_baseline', {}).get('source', 'unknown')} "
                f"objectives from measured baseline"
            ),
            (
                "- Feature drift: "
                f"{'alert' if bool(payload.get('feature_drift_alert', False)) else 'clear'}"
            ),
        ]
    )


def emit_run_health_artifacts(
    report: dict[str, Any],
    *,
    report_output_path: Path,
) -> tuple[dict[str, Any], Path, Path]:
    """Persist run-level health artifacts as JSON + text summary."""

    payload = build_run_health_dashboard(report)
    summary_text = render_run_health_summary(payload)

    health_json_path = report_output_path.with_suffix(".health.json")
    health_text_path = report_output_path.with_suffix(".health.txt")

    health_json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    health_text_path.write_text(summary_text, encoding="utf-8")

    return payload, health_json_path, health_text_path


def emit_run_audit_artifacts(
    report: dict[str, Any],
    *,
    report_output_path: Path,
) -> Path:
    """Persist append-only JSONL audit records for run and dataset health."""

    payload = build_run_health_dashboard(report)
    sink = JsonlAuditSink(report_output_path.with_suffix(".audit.jsonl"))
    sink.append("run_health", payload, created_at=str(report.get("timestamp") or ""))
    sink.append("data_health", payload.get("data_health", {}), created_at=str(report.get("timestamp") or ""))
    sink.append("runtime_health", payload.get("runtime_health", {}), created_at=str(report.get("timestamp") or ""))
    if payload.get("execution_health"):
        sink.append("execution_health", payload.get("execution_health", {}), created_at=str(report.get("timestamp") or ""))
    if payload.get("service_level_baseline"):
        sink.append("service_level_baseline", payload.get("service_level_baseline", {}), created_at=str(report.get("timestamp") or ""))
    sink.append("feature_drift", {"feature_drift_alert": bool(payload.get("feature_drift_alert", False))}, created_at=str(report.get("timestamp") or ""))
    sink.append(
        "provider_health",
        {
            "provider_status": payload.get("data_health", {}).get("provider_status", "unknown"),
            "provider_probe_count": payload.get("data_health", {}).get("provider_probe_count", 0),
            "provider_stale_count": payload.get("data_health", {}).get("provider_stale_count", 0),
            "provider_failure_count": payload.get("data_health", {}).get("provider_failure_count", 0),
            "provider_circuit_breaker_triggered": payload.get("data_health", {}).get("provider_circuit_breaker_triggered", False),
        },
        created_at=str(report.get("timestamp") or ""),
    )
    return sink.path


def build_session_health_summary(
    *,
    user_id: int,
    diagnostics: Any,
    kill_switch: KillSwitchEvaluation | None,
) -> str:
    """Render bot-readable runtime session health summary."""

    if diagnostics is None:
        return ""

    kill_state = "PAUSED" if kill_switch and kill_switch.pause_trading else "CLEAR"
    kill_reasons = ", ".join(kill_switch.reasons) if kill_switch and kill_switch.reasons else "none"

    return "\n".join(
        [
            "Session Health:",
            f"- User: {int(user_id)}",
            f"- Orders: {int(getattr(diagnostics, 'total_orders', 0) or 0)} (session cumulative)",
            (
                "- Kill-switch blocks: "
                f"cycles={int(getattr(diagnostics, 'paused_cycles', 0) or 0)}, "
                f"actionable_signals={int(getattr(diagnostics, 'blocked_actionable_signals', 0) or 0)}"
            ),
            f"- Reject rate: {float(getattr(diagnostics, 'reject_rate', 0.0) or 0.0)*100:.2f}%",
            f"- Go/No-Go: {'PASS' if bool(getattr(diagnostics, 'live_go_no_go_passed', True)) else 'FAIL'}",
            f"- Rollback required: {bool(getattr(diagnostics, 'rollback_required', False))}",
            f"- Kill switch: {kill_state} ({kill_reasons})",
        ]
    )
