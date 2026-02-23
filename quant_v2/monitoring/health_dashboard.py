"""Health dashboard artifacts for research runs and live sessions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from quant_v2.monitoring.kill_switch import KillSwitchEvaluation


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

    status = "healthy"
    if not gates_passed or score < 70.0:
        status = "warning"
    if mean_abs_ev_delta > 0.35:
        status = "degraded"

    return {
        "run_id": str(report.get("run_id") or ""),
        "timestamp": str(report.get("timestamp") or ""),
        "status": status,
        "score": score,
        "gates_passed": gates_passed,
        "stability_score": stability,
        "mean_abs_ev_delta": mean_abs_ev_delta,
    }


def render_run_health_summary(payload: dict[str, Any]) -> str:
    """Render bot-friendly single-run health summary."""

    return "\n".join(
        [
            "Run Health Dashboard:",
            f"- Run ID: {payload.get('run_id', '')}",
            f"- Status: {payload.get('status', 'unknown')}",
            f"- Score: {float(payload.get('score', 0.0)):.2f}",
            f"- Gates passed: {bool(payload.get('gates_passed', False))}",
            f"- Stability: {float(payload.get('stability_score', 0.0)):.3f}",
            f"- Mean |EV delta|: {float(payload.get('mean_abs_ev_delta', 0.0)):.4f}",
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
            f"- Orders: {int(getattr(diagnostics, 'total_orders', 0) or 0)}",
            f"- Reject rate: {float(getattr(diagnostics, 'reject_rate', 0.0) or 0.0)*100:.2f}%",
            f"- Go/No-Go: {'PASS' if bool(getattr(diagnostics, 'live_go_no_go_passed', True)) else 'FAIL'}",
            f"- Rollback required: {bool(getattr(diagnostics, 'rollback_required', False))}",
            f"- Kill switch: {kill_state} ({kill_reasons})",
        ]
    )
