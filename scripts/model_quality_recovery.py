"""Generate model-quality recovery diagnostics and benchmark evidence."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quant_v2.config import default_universe_symbols, get_runtime_profile  # noqa: E402
from quant_v2.data.storage import load_multi_symbol_snapshot  # noqa: E402
from quant_v2.research.model_quality_recovery import (  # noqa: E402
    generate_quality_recovery_bundle,
    fetch_quality_recovery_dataset,
)
from quant_v2.research.model_recovery_experiments import run_phase4_research_input_repair  # noqa: E402


def _parse_csv(raw: str | None) -> tuple[str, ...]:
    if raw is None or not raw.strip():
        return default_universe_symbols()
    values = tuple(item.strip().upper() for item in raw.split(",") if item.strip())
    if not values:
        raise ValueError("No symbols parsed from input")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate model quality recovery reports")
    parser.add_argument("--model-root", type=str, default="", help="Model root directory")
    parser.add_argument("--registry-root", type=str, default="", help="Registry root directory")
    parser.add_argument("--snapshot-path", type=str, default="", help="Existing dataset snapshot parquet path")
    parser.add_argument("--output-dir", type=str, default="", help="Directory for generated reports")
    parser.add_argument("--months", type=int, default=6, help="History months when fetching")
    parser.add_argument("--symbols", type=str, default="", help="Comma-separated symbols")
    parser.add_argument("--interval", type=str, default="", help="Market interval, default runtime anchor interval")
    parser.add_argument("--failed-record-path", type=str, default="", help="Optional specific failed retrain record")
    parser.add_argument("--phase4-repair", action="store_true", help="Run the Phase 4 research-input repair suite")
    parser.add_argument("--label-mode", choices=("directional_return", "trade_outcome"), default="directional_return", help="Label mode for recovery experiments")
    parser.add_argument("--trade-profit-target-bps", type=float, default=20.0, help="Trade-outcome profit target in bps")
    parser.add_argument("--trade-stop-loss-bps", type=float, default=30.0, help="Trade-outcome stop loss in bps")
    parser.add_argument("--trade-round-trip-cost-bps", type=float, default=8.0, help="Trade-outcome round-trip cost in bps")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    profile = get_runtime_profile()
    model_root = Path(args.model_root).expanduser() if args.model_root else profile.project_root / "models" / "production"
    registry_root = Path(args.registry_root).expanduser() if args.registry_root else model_root / "registry"
    output_dir = Path(args.output_dir).expanduser() if args.output_dir else profile.project_root / "docs" / "model_quality"
    symbols = _parse_csv(args.symbols)
    interval = args.interval or profile.universe.anchor_interval
    failed_record_path = Path(args.failed_record_path).expanduser() if args.failed_record_path else None

    if args.snapshot_path:
        dataset, _manifest = load_multi_symbol_snapshot(args.snapshot_path)
    else:
        dataset = fetch_quality_recovery_dataset(
            months=args.months,
            symbols=symbols,
            interval=interval,
        )

    if args.phase4_repair:
        result = run_phase4_research_input_repair(
            dataset,
            snapshot_path=Path(args.snapshot_path).expanduser() if args.snapshot_path else output_dir / "snapshot.parquet",
            output_root=output_dir / "phase4",
            docs_output_dir=output_dir,
            label_mode=str(args.label_mode),
            trade_outcome_profit_target_bps=float(args.trade_profit_target_bps),
            trade_outcome_stop_loss_bps=float(args.trade_stop_loss_bps),
            trade_outcome_round_trip_cost_bps=float(args.trade_round_trip_cost_bps),
        )
        summary = {
            "model_root": str(model_root),
            "registry_root": str(registry_root),
            "output_dir": str(output_dir),
            "phase4_output_dir": str(result.output_dir),
            "recommendation": result.summary.get("recommendation"),
            "selected_variant_id": result.summary.get("selected_variant_id"),
            "selected_candidate_id": result.summary.get("selected_candidate_id"),
            "diagnostics_path": str(output_dir / "research_input_diagnostics.md"),
        }
    else:
        bundle, paths = generate_quality_recovery_bundle(
            model_root=model_root,
            registry_root=registry_root,
            dataset=dataset,
            output_dir=output_dir,
            failed_record_path=failed_record_path,
        )

        summary = {
            "model_root": str(model_root),
            "registry_root": str(registry_root),
            "output_dir": str(output_dir),
            "paths": {name: str(path) for name, path in paths.items()},
            "recommendation": bundle.candidate_selection.get("recommendation"),
            "best_actor": bundle.candidate_selection.get("benchmark_best_actor"),
        }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
