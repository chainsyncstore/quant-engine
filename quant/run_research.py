"""
Main research pipeline orchestrator.

Usage:
    # Dry run with synthetic data (no API needed):
    python -m quant.run_research --dry-run

    # Full run fetching real data:
    python -m quant.run_research --fetch --months 3
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime, timedelta, timezone
from typing import Dict

import numpy as np
import pandas as pd

from quant.config import get_path_config, get_research_config
from quant.data.session_filter import filter_sessions
from quant.data.storage import snapshot, validate_ohlcv, report_gaps
from quant.features.pipeline import build_features, get_feature_columns
from quant.labels.labeler import add_labels
from quant.risk.monte_carlo import MonteCarloResult, simulate
from quant.validation.walk_forward import WalkForwardResult, run_walk_forward
from quant.experiment.logger import save_experiment, determine_verdict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def generate_synthetic_eurusd(n_bars: int = 40_000, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic EURUSD-like 1-minute data for testing.

    Produces realistic-looking data with varying volatility regimes.
    """
    rng = np.random.default_rng(seed)

    # Generate only weekday session hours (08:00-21:00 UTC)
    timestamps = []
    current = datetime(2025, 11, 1, 8, 0, tzinfo=timezone.utc)
    while len(timestamps) < n_bars:
        if current.weekday() < 5 and 8 <= current.hour < 21:
            timestamps.append(current)
        current += timedelta(minutes=1)

    price = 1.0850  # Typical EURUSD
    records = []

    for i, ts in enumerate(timestamps):
        # Regime-varying volatility
        regime_cycle = np.sin(2 * np.pi * i / 5000) * 0.5 + 0.5
        vol = 0.0001 + regime_cycle * 0.0003  # 1-4 pips

        ret = rng.normal(0, vol)
        o = price
        c = o + ret

        # High/low with realistic wicks
        wick_factor = rng.uniform(0.3, 1.5)
        h = max(o, c) + abs(ret) * wick_factor * rng.uniform(0.1, 1.0)
        l = min(o, c) - abs(ret) * wick_factor * rng.uniform(0.1, 1.0)

        volume = max(0, rng.normal(100, 30))

        records.append({"timestamp": ts, "open": o, "high": h, "low": l, "close": c, "volume": volume})
        price = c

    df = pd.DataFrame(records).set_index("timestamp")
    df.index = pd.DatetimeIndex(df.index, tz="UTC")
    return df


def run_pipeline(
    df: pd.DataFrame,
    optimize: bool = False,
    prune_threshold: float = 0.0,
) -> None:
    """Execute the full research pipeline on prepared data."""
    cfg = get_research_config()
    start_time = time.time()

    # --- Step 1: Validate ---
    logger.info("=" * 60)
    logger.info("STEP 1: Data Validation")
    logger.info("=" * 60)
    validate_ohlcv(df)
    gaps = report_gaps(df)
    if gaps:
        logger.warning("Data has %d gaps — proceeding but results may be affected", len(gaps))

    # --- Step 2: Snapshot ---
    logger.info("=" * 60)
    logger.info("STEP 2: Snapshot")
    logger.info("=" * 60)
    snap_path = snapshot(df, tag="research_run")

    # --- Step 3: Session filter ---
    logger.info("=" * 60)
    logger.info("STEP 3: Session Filter")
    logger.info("=" * 60)
    df_filtered = filter_sessions(df)

    if len(df_filtered) < cfg.min_total_bars:
        logger.error(
            "Insufficient data: %d bars (need %d). Fetch more data.",
            len(df_filtered),
            cfg.min_total_bars,
        )
        sys.exit(1)

    # --- Step 4: Feature engineering ---
    logger.info("=" * 60)
    logger.info("STEP 4: Feature Engineering")
    logger.info("=" * 60)
    df_features = build_features(df_filtered)
    feature_cols = get_feature_columns(df_features)
    logger.info("Features: %s", feature_cols)

    # --- Step 5: Labels ---
    logger.info("=" * 60)
    logger.info("STEP 5: Labeling")
    logger.info("=" * 60)
    df_labeled = add_labels(df_features)

    # --- Step 5b: Optimization (if --optimize) ---
    best_params = {}
    pruned_features = None
    if optimize:
        logger.info("=" * 60)
        logger.info("STEP 5b: Hyperparameter Optimization (Optuna)")
        logger.info("=" * 60)

        from quant.models.optimizer import optimize_hyperparams
        from quant.features.feature_selector import prune_features

        best_params = optimize_hyperparams(
            df_labeled, horizon=cfg.horizons[0], n_trials=50,
        )
        logger.info("Best LightGBM params: %s", best_params)

        # Run a quick walk-forward to get feature importance for pruning
        # (This is the old pruning method, we now have trainer-internal pruning too)
        # We'll stick to trainer-internal pruning via prune_threshold for simplicity in this phase
    
    # Inject pruning threshold into params
    best_params["prune_threshold"] = prune_threshold
    
    # --- Step 6: Walk-forward validation ---
    logger.info("=" * 60)
    logger.info("STEP 6: Walk-Forward Validation%s", " (OPTIMIZED)" if optimize else "")
    logger.info("=" * 60)
    wf_result = run_walk_forward(
        df_labeled,
        params_override=best_params,
        feature_subset=None, # We rely on trainer-internal pruning
    )

    # --- Step 7: Monte Carlo ---
    logger.info("=" * 60)
    logger.info("STEP 7: Monte Carlo Simulation")
    logger.info("=" * 60)
    mc_results: Dict[int, MonteCarloResult] = {}
    for h in cfg.horizons:
        pnl = wf_result.all_pnl.get(h, np.array([]))
        mc_results[h] = simulate(pnl)

    # --- Step 8: Save experiment ---
    duration = time.time() - start_time
    logger.info("=" * 60)
    logger.info("STEP 8: Saving Experiment")
    logger.info("=" * 60)
    exp_path = save_experiment(
        result=wf_result,
        mc_results=mc_results,
        snapshot_path=str(snap_path),
        duration_seconds=duration,
    )

    # --- Summary ---
    verdict = determine_verdict(wf_result)
    logger.info("=" * 60)
    logger.info("RESEARCH COMPLETE")
    logger.info("=" * 60)
    logger.info("Duration: %.1f seconds", duration)
    logger.info("Verdict: %s", verdict)
    logger.info("Experiment log: %s", exp_path)

    for h, report in wf_result.reports.items():
        logger.info(
            "  %dm → EV=%.6f | WR=%.1f%% | Sharpe=%.2f | Trades=%d | MaxDD=%.4f",
            h,
            report.overall_ev,
            report.overall_win_rate * 100,
            report.overall_sharpe,
            report.overall_n_trades,
            report.overall_max_drawdown,
        )
        for rm in report.per_regime:
            logger.info(
                "    Regime %d: EV=%.6f | WR=%.1f%% | Trades=%d | Thresh=%.2f",
                rm.regime,
                rm.ev,
                rm.win_rate * 100,
                rm.n_trades,
                rm.optimal_threshold,
            )

    mc_summary = mc_results.get(cfg.horizons[0])
    if mc_summary:
        logger.info(
            "  Monte Carlo: Ruin=%.1f%% | EV_CI=[%.6f, %.6f]",
            mc_summary.ruin_probability * 100,
            mc_summary.ev_ci_95[0],
            mc_summary.ev_ci_95[1],
        )


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="EURUSD Quant Research Engine")
    parser.add_argument("--dry-run", action="store_true", help="Use synthetic data (no API)")
    parser.add_argument("--fetch", action="store_true", help="Fetch data from Capital.com")
    parser.add_argument("--months", type=int, default=3, help="Months of history to fetch")
    parser.add_argument("--optimize", action="store_true", help="Run Optuna HPO + feature pruning")
    parser.add_argument("--prune", type=float, default=0.0, help="Feature pruning threshold (e.g. 0.01)")
    parser.add_argument("--bars", type=int, default=40000, help="Synthetic bar count for dry-run")
    args = parser.parse_args()

    # Ensure output dirs exist
    get_path_config()

    if args.dry_run:
        logger.info("DRY RUN — using synthetic EURUSD data (%d bars)", args.bars)
        df = generate_synthetic_eurusd(n_bars=args.bars)
        run_pipeline(df, optimize=args.optimize)

    elif args.fetch:
        from quant.data.capital_client import CapitalClient

        client = CapitalClient()
        client.authenticate()

        date_to = datetime.now(timezone.utc)
        date_from = date_to - timedelta(days=args.months * 30)

        logger.info("Fetching EURUSD 1m data: %s → %s", date_from, date_to)
        
        # Use a consistent cache directory for resumable fetching
        cache_dir = get_path_config().datasets_raw / "fetch_cache"
        df = client.fetch_historical(date_from, date_to, cache_dir=cache_dir)

        if df.empty:
            logger.error("No data received from API")
            sys.exit(1)

        # Save raw
        from quant.data.storage import save_raw

        label = f"EURUSD_1m_{date_from.strftime('%Y%m%d')}_{date_to.strftime('%Y%m%d')}"
        save_raw(df, label)

        run_pipeline(df, optimize=args.optimize, prune_threshold=args.prune)

    else:
        # Try loading latest snapshot
        from quant.data.storage import load_latest_snapshot

        df = load_latest_snapshot()
        if df is None:
            logger.error("No data found. Use --dry-run or --fetch")
            sys.exit(1)

        run_pipeline(df, optimize=args.optimize, prune_threshold=args.prune)


if __name__ == "__main__":
    main()
