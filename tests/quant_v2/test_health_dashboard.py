from __future__ import annotations

from pathlib import Path

from quant_v2.monitoring.health_dashboard import (
    build_run_health_dashboard,
    build_session_health_summary,
    emit_run_health_artifacts,
)
from quant_v2.monitoring.kill_switch import KillSwitchEvaluation


class _Diag:
    total_orders = 3
    reject_rate = 0.25
    live_go_no_go_passed = False
    rollback_required = True


def test_build_run_health_dashboard_and_emit_artifacts(tmp_path: Path) -> None:
    report = {
        "run_id": "abc123",
        "timestamp": "2026-01-01T00:00:00+00:00",
        "scorecard": {"score": 72.0, "gates": {"passed": True}},
        "forward_live_simulation": {"aggregate": {"stability_score": 0.7}},
        "replay_regression": {"aggregate": {"mean_abs_ev_delta": 0.05}},
    }

    payload = build_run_health_dashboard(report)
    assert payload["run_id"] == "abc123"
    assert payload["status"] in {"healthy", "warning", "degraded"}

    out = tmp_path / "run.json"
    out.write_text("{}", encoding="utf-8")
    emitted_payload, json_path, text_path = emit_run_health_artifacts(report, report_output_path=out)

    assert emitted_payload["run_id"] == "abc123"
    assert json_path.exists()
    assert text_path.exists()


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
