"""CLI and orchestration for v2 multi-symbol group-purged validation runs."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from quant_v2.config import default_universe_symbols, get_runtime_profile
from quant_v2.monitoring.health_dashboard import emit_run_health_artifacts
from quant_v2.research.experiment_score import build_report_from_experiment
from quant_v2.research.forward_live import build_forward_live_simulation
from quant_v2.research.group_validation import (
    GroupValidationResult,
    prepare_multi_symbol_dataset,
    run_group_purged_validation,
)
from quant_v2.research.replay_regression import build_replay_regression_report
from quant_v2.research.stage1_pipeline import build_stage1_result, load_or_build_dataset

logger = logging.getLogger(__name__)


def parse_csv_symbols(raw: str | None) -> tuple[str, ...]:
    """Parse comma-separated symbol input."""

    if raw is None or not raw.strip():
        return default_universe_symbols()

    symbols = tuple(item.strip().upper() for item in raw.split(",") if item.strip())
    if not symbols:
        raise ValueError("No symbols parsed from --symbols")
    return symbols


def parse_csv_ints(raw: str | None, default: tuple[int, ...]) -> tuple[int, ...]:
    """Parse comma-separated integer list."""

    if raw is None or not raw.strip():
        return default

    out: list[int] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        out.append(int(token))

    values = tuple(out)
    if not values:
        raise ValueError("No integer values parsed")
    return values


def run_validation_pipeline(
    *,
    snapshot_path: str | None = None,
    baseline_report_path: str | None = None,
    months: int = 18,
    symbols: tuple[str, ...] | None = None,
    interval: str | None = None,
    horizons: tuple[int, ...] = (1, 4, 12),
    n_time_splits: int = 5,
    symbol_cluster_size: int = 2,
    embargo_bars: int = 24,
    min_train_rows: int = 200,
    output_path: Path | None = None,
    fail_fast: bool = False,
    client=None,
) -> dict[str, Any]:
    """Execute end-to-end group-purged validation and persist report JSON."""

    run_id = uuid4().hex[:8]
    started_at = datetime.now(timezone.utc)

    raw_df, snapshot = load_or_build_dataset(
        snapshot_path=snapshot_path,
        months=months,
        interval=interval,
        symbols=symbols,
        fail_fast=fail_fast,
        client=client,
    )

    prepared = prepare_multi_symbol_dataset(raw_df, horizons=horizons)
    stage1 = build_stage1_result(
        prepared,
        snapshot,
        n_time_splits=n_time_splits,
        symbol_cluster_size=symbol_cluster_size,
        embargo_bars=embargo_bars,
        min_train_rows=min_train_rows,
    )

    horizon_results: dict[int, GroupValidationResult] = {}
    for horizon in horizons:
        result = run_group_purged_validation(
            stage1.dataset,
            horizon=horizon,
            min_train_rows=min_train_rows,
            precomputed_splits=stage1.splits,
            split_summary=stage1.split_summary,
        )
        horizon_results[horizon] = result

    experiment_like = {
        "config": {"validation_mode": "group_purged_cpcv"},
        "results": {
            str(h): {
                "overall": result.overall,
                "robustness": result.robustness,
                "per_fold": [asdict(f.metrics) for f in result.folds],
            }
            for h, result in horizon_results.items()
        },
        "monte_carlo": {},
    }
    score_report = build_report_from_experiment(experiment_like)
    forward_live = build_forward_live_simulation(horizon_results)

    baseline_forward_live: dict[str, Any] | None = None
    if baseline_report_path:
        baseline_path = Path(baseline_report_path).expanduser()
        baseline_payload = json.loads(baseline_path.read_text(encoding="utf-8"))
        loaded = baseline_payload.get("forward_live_simulation")
        if isinstance(loaded, dict):
            baseline_forward_live = loaded

    replay_regression = build_replay_regression_report(
        current_forward_live=forward_live,
        baseline_forward_live=baseline_forward_live,
    )

    report = {
        "run_id": run_id,
        "timestamp": started_at.isoformat(),
        "config": {
            "mode": "group_purged_cpcv",
            "snapshot_path": str(snapshot.parquet_path),
            "baseline_report_path": str(Path(baseline_report_path).expanduser()) if baseline_report_path else None,
            "months": months,
            "symbols": list(symbols or default_universe_symbols()),
            "interval": interval or get_runtime_profile().universe.anchor_interval,
            "horizons": list(horizons),
            "n_time_splits": n_time_splits,
            "symbol_cluster_size": symbol_cluster_size,
            "embargo_bars": embargo_bars,
            "min_train_rows": min_train_rows,
            "fail_fast": fail_fast,
        },
        "dataset": {
            "parquet_path": str(snapshot.parquet_path),
            "manifest_path": str(snapshot.manifest_path),
            "manifest": snapshot.manifest,
            "prepared_rows": int(len(prepared)),
            "prepared_symbols": sorted(set(prepared.index.get_level_values("symbol"))),
            "stage1_split_summary": stage1.split_summary,
        },
        "horizons": {
            str(h): {
                "validation_mode": result.validation_mode,
                "split_summary": result.split_summary,
                "overall": result.overall,
                "robustness": result.robustness,
                "n_trials_assumed": result.n_trials_assumed,
                "folds": [
                    {
                        "split_id": fold.split_id,
                        "n_train_rows": fold.n_train_rows,
                        "n_test_rows": fold.n_test_rows,
                        "n_valid_rows": fold.n_valid_rows,
                        "test_symbols": list(fold.test_symbols),
                        "metrics": asdict(fold.metrics),
                    }
                    for fold in result.folds
                ],
            }
            for h, result in horizon_results.items()
        },
        "scorecard": {
            "score": score_report.score,
            "inputs": asdict(score_report.score_inputs),
            "gates": {
                "passed": score_report.gates.passed,
                "checks": score_report.gates.checks,
                "inputs": asdict(score_report.gate_inputs),
            },
        },
        "forward_live_simulation": forward_live,
        "replay_regression": replay_regression,
    }

    if output_path is None:
        output_dir = get_runtime_profile().project_root / "experiments"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"v2_group_validation_{started_at.strftime('%Y%m%d_%H%M%S')}_{run_id}.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    health_payload, health_json_path, health_text_path = emit_run_health_artifacts(
        report,
        report_output_path=output_path,
    )
    report["health_dashboard"] = {
        "payload": health_payload,
        "json_path": str(health_json_path),
        "summary_path": str(health_text_path),
    }
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    logger.info("Saved v2 group validation report: %s", output_path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run v2 multi-symbol group-purged validation")
    parser.add_argument("--snapshot-path", type=str, default="", help="Existing multi-symbol snapshot parquet path")
    parser.add_argument(
        "--baseline-report",
        type=str,
        default="",
        help="Optional baseline run report path for replay regression deltas",
    )
    parser.add_argument("--months", type=int, default=18, help="History months when fetching")
    parser.add_argument("--symbols", type=str, default="", help="Comma-separated symbols")
    parser.add_argument("--interval", type=str, default="", help="Interval (default: profile anchor)")
    parser.add_argument("--horizons", type=str, default="1,4,12", help="Comma-separated horizon bars")
    parser.add_argument("--n-time-splits", type=int, default=5)
    parser.add_argument("--symbol-cluster-size", type=int, default=2)
    parser.add_argument("--embargo-bars", type=int, default=24)
    parser.add_argument("--min-train-rows", type=int, default=200)
    parser.add_argument("--output", type=str, default="", help="Report output path")
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    symbols = parse_csv_symbols(args.symbols)
    horizons = parse_csv_ints(args.horizons, default=(1, 4, 12))

    run_validation_pipeline(
        snapshot_path=args.snapshot_path or None,
        baseline_report_path=args.baseline_report or None,
        months=args.months,
        symbols=symbols,
        interval=args.interval or None,
        horizons=horizons,
        n_time_splits=args.n_time_splits,
        symbol_cluster_size=args.symbol_cluster_size,
        embargo_bars=args.embargo_bars,
        min_train_rows=args.min_train_rows,
        output_path=Path(args.output).expanduser() if args.output else None,
        fail_fast=args.fail_fast,
    )


if __name__ == "__main__":
    main()
