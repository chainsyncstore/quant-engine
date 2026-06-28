from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

import quant_v2.monitoring.health_dashboard as health_dashboard
from quant_v2.monitoring.health_dashboard import (
    build_data_health_dashboard,
    build_run_health_dashboard,
    build_runtime_resource_health,
    build_session_health_summary,
    emit_run_audit_artifacts,
    emit_run_health_artifacts,
)
from quant_v2.monitoring.provider_probes import probe_market_data_provider
from quant_v2.monitoring.kill_switch import KillSwitchEvaluation


class _Diag:
    total_orders = 3
    reject_rate = 0.25
    live_go_no_go_passed = False
    rollback_required = True


class _FakeProcess:
    def cpu_percent(self, interval=None):
        return 92.5

    def memory_info(self):
        return SimpleNamespace(rss=768 * 1024 * 1024)

    def open_files(self):
        return [object(), object()]

    def create_time(self):
        return (datetime.now(timezone.utc) - timedelta(seconds=120)).timestamp()


class _FakePsutil:
    def Process(self):
        return _FakeProcess()

    def virtual_memory(self):
        return SimpleNamespace(percent=88.0)

    def disk_usage(self, path):
        return SimpleNamespace(percent=91.0)


class _FakeMarketClient:
    def fetch_historical(self, date_from, date_to, symbol=None, interval=None):
        if symbol == "STALE":
            ts = pd.date_range(datetime.now(timezone.utc) - timedelta(hours=4), periods=3, freq="1h", tz="UTC")
        else:
            ts = pd.date_range(datetime.now(timezone.utc) - timedelta(hours=2), periods=3, freq="1h", tz="UTC")
        return pd.DataFrame(
            {
                "open": [1.0, 2.0, 3.0],
                "high": [1.1, 2.1, 3.1],
                "low": [0.9, 1.9, 2.9],
                "close": [1.05, 2.05, 3.05],
                "volume": [100.0, 110.0, 120.0],
            },
            index=ts,
        )

    def fetch_funding_rates(self, date_from, date_to, symbol=None):
        ts = pd.date_range(datetime.now(timezone.utc) - timedelta(hours=2), periods=2, freq="2h", tz="UTC")
        return pd.DataFrame({"funding_rate_raw": [0.0001, 0.0002]}, index=ts)

    def fetch_open_interest(self, date_from, date_to, symbol=None, period="1h"):
        ts = pd.date_range(datetime.now(timezone.utc) - timedelta(hours=2), periods=3, freq="1h", tz="UTC")
        return pd.DataFrame(
            {"open_interest": [1000.0, 1001.0, 1002.0], "open_interest_value": [10_000.0, 10_010.0, 10_020.0]},
            index=ts,
        )

    def get_rate_limit_snapshot(self):
        return {
            "provider_name": "binance_futures_rest",
            "status": "warning",
            "used_weight_1m": 2200,
            "weight_limit_1m": 2400,
            "throttle_threshold_1m": 1800,
            "headroom_1m": 200,
            "pressure_fraction": 2200 / 2400,
            "throttle_interval_seconds": 0.5,
            "last_request_age_seconds": 3.5,
        }


def test_build_run_health_dashboard_and_emit_artifacts(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(health_dashboard, "probe_dns_latency", lambda target=None: {
        "status": "healthy",
        "target_host": "localhost",
        "target_port": 6379,
        "latency_ms": 4.2,
        "address_count": 1,
    })
    monkeypatch.setattr(health_dashboard, "probe_sqlite_lock_latency", lambda db_path=None: {
        "status": "healthy",
        "db_path": str(db_path or ""),
        "lock_latency_ms": 1.8,
        "lock_acquired": True,
    })
    monkeypatch.setattr(health_dashboard, "read_runtime_boot_marker", lambda path=None: {
        "boot_count": 3,
        "restart_count": 2,
        "started_at": "2026-01-01T00:00:00+00:00",
        "host": "unit-host",
        "process_id": 1234,
    })

    report = {
        "run_id": "abc123",
        "timestamp": "2026-01-01T00:00:00+00:00",
        "scorecard": {"score": 72.0, "gates": {"passed": True}},
        "forward_live_simulation": {"aggregate": {"stability_score": 0.7}},
        "replay_regression": {"aggregate": {"mean_abs_ev_delta": 0.05}},
        "dataset": {
            "manifest": {
                "dataset_name": "unit_dataset",
                "source_retrieved_at": "2026-01-01T00:00:00+00:00",
                "coverage_by_symbol": {
                    "BTCUSDT": {"close": 1.0, "volume": 0.995},
                    "ETHUSDT": {"close": 0.99, "volume": 0.99},
                },
                "gap_stats_per_symbol": {
                    "BTCUSDT": {"gap_count": 0},
                    "ETHUSDT": {"gap_count": 1},
                },
                "duplicate_rows_per_symbol": {"BTCUSDT": 0, "ETHUSDT": 0},
                "failed_symbols": {"XRPUSDT": "timeout"},
                "null_ratio_by_column": {
                    "open": 0.0,
                    "high": 0.0,
                    "low": 0.0,
                    "close": 0.10,
                    "volume": 0.0,
                    "f1": 0.05,
                },
                "n_symbols": 2,
            }
        },
        "provider_probes": {
            "market_data": probe_market_data_provider(
                ("BTCUSDT", "STALE"),
                client=_FakeMarketClient(),
                stale_after_minutes=60.0,
            )
        },
        "feature_drift_alert": True,
        "execution_health": {
            "status": "warning",
            "captured_at": "2026-01-01T00:00:00+00:00",
            "live_session_count": 1,
            "stream_bus": {
                "status": "warning",
                "max_pending_age_seconds": 45.0,
                "stream_backlog_count": 3,
                "pending_count": 2,
                "lag_entries": 1,
                "stream_age_seconds": 12.0,
            },
            "wal": {
                "status": "healthy",
                "entry_count": 8,
                "latest_entry_age_seconds": 14.0,
                "oldest_entry_age_seconds": 120.0,
            },
            "reconciliation": {
                "status": "healthy",
                "lag_seconds": 18.0,
                "last_error_at": "",
            },
            "redis_memory": {
                "status": "healthy",
                "used_memory_mb": 512.0,
                "maxmemory_mb": 1024.0,
                "used_memory_fraction": 0.5,
            },
        },
    }

    data_health = build_data_health_dashboard(report)
    assert data_health["dataset_name"] == "unit_dataset"
    assert data_health["gap_count_total"] == 1
    assert data_health["min_coverage_fraction"] == 0.99
    assert data_health["feature_missingness_max"] == 0.10
    assert data_health["quarantined_symbols"] == ["XRPUSDT"]
    assert data_health["provider_status"] == "degraded"
    assert data_health["provider_circuit_breaker_triggered"] is True
    assert data_health["provider_rate_limit_status"] == "warning"
    assert data_health["provider_rate_limit_used_weight_1m"] == 2200
    assert data_health["provider_rate_limit_weight_limit_1m"] == 2400
    assert data_health["provider_rate_limit_headroom_1m"] == 200
    assert data_health["provider_rate_limit_pressure_fraction"] == pytest.approx(2200 / 2400, abs=1e-9)

    payload = build_run_health_dashboard(report)
    assert payload["run_id"] == "abc123"
    assert payload["status"] == "degraded"
    assert payload["data_health"]["dataset_name"] == "unit_dataset"
    assert payload["data_health"]["gap_count_total"] == 1
    assert payload["data_health"]["min_coverage_fraction"] == 0.99
    assert payload["execution_health"]["stream_backlog_count"] == 3
    assert payload["execution_health"]["status"] == "warning"
    assert payload["execution_health"]["redis_memory_used_mb"] == 512.0
    assert payload["feature_drift_alert"] is True
    assert payload["runtime_health"]["container_restart_count"] == 2
    assert payload["runtime_health"]["dns_probe"]["status"] == "healthy"
    assert payload["runtime_health"]["database_lock_probe"]["status"] == "healthy"
    assert payload["service_level_baseline"]["source"] == "measured"
    assert payload["service_level_baseline"]["observed"]["dns_latency_ms"] == 4.2
    assert payload["service_level_baseline"]["observed"]["feature_drift_alert"] is True
    assert payload["service_level_baseline"]["observed"]["feature_missingness_max"] == 0.10
    assert payload["service_level_baseline"]["observed"]["provider_rate_limit_used_weight_1m"] == 2200.0
    assert payload["service_level_baseline"]["observed"]["provider_rate_limit_pressure_fraction"] == pytest.approx(2200 / 2400, abs=1e-9)
    assert payload["service_level_baseline"]["targets"]["dns_latency_ms"] == pytest.approx(5.04, abs=1e-9)

    out = tmp_path / "run.json"
    out.write_text("{}", encoding="utf-8")
    emitted_payload, json_path, text_path = emit_run_health_artifacts(report, report_output_path=out)
    audit_path = emit_run_audit_artifacts(report, report_output_path=out)
    summary_text = text_path.read_text(encoding="utf-8")

    assert emitted_payload["run_id"] == "abc123"
    assert json_path.exists()
    assert text_path.exists()
    assert audit_path.exists()
    audit_lines = audit_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(audit_lines) == 7
    assert "Runtime signals:" in summary_text
    assert "SLI baseline:" in summary_text
    assert "Feature drift: alert" in summary_text
    assert "rate_limit=warning" in summary_text
    assert "missingness=0.100" in summary_text


def test_build_runtime_resource_health_reads_psutil_snapshot(monkeypatch) -> None:
    monkeypatch.setattr(health_dashboard, "_psutil", _FakePsutil())

    payload = build_runtime_resource_health()

    assert payload["status"] == "degraded"
    assert payload["cpu_percent"] == 92.5
    assert payload["memory_percent"] == 88.0
    assert payload["disk_percent"] == 91.0
    assert payload["rss_mb"] == 768.0
    assert payload["open_file_count"] == 2


def test_build_session_health_summary_formats_runtime_view() -> None:
    summary = build_session_health_summary(
        user_id=11,
        diagnostics=_Diag(),
        kill_switch=KillSwitchEvaluation(pause_trading=True, reasons=("rollback_required",)),
    )

    assert "Session Health:" in summary
    assert "User: 11" in summary
    assert "Go/No-Go: FAIL" in summary
    assert "Rollback required: True" in summary
    assert "Execution health:" not in summary
