"""
Main research pipeline orchestrator.

Usage:
    # Dry run with synthetic BTC-like data (no API needed):
    python -m quant.run_research --dry-run

    # Full run fetching real Binance data:
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
from quant.data.binance_client import BinanceClient
from quant.data.session_filter import filter_sessions
from quant.data.storage import snapshot, validate_ohlcv, report_gaps
from quant.features.pipeline import build_features, get_feature_columns
from quant.labels.labeler import add_labels
from quant.risk.monte_carlo import MonteCarloResult, simulate
from quant.validation.metrics import probabilistic_sharpe_ratio, deflated_sharpe_ratio
from quant.validation.walk_forward import WalkForwardResult, run_walk_forward
from quant.experiment.logger import save_experiment, determine_verdict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def generate_synthetic_crypto(n_bars: int = 10_000, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic BTCUSDT-like 1-hour data for testing.

    Produces realistic-looking data with varying volatility regimes and
    required supplementary columns for crypto feature modules.
    """
    rng = np.random.default_rng(seed)

    timestamps = pd.date_range(
        end=datetime.now(timezone.utc),
        periods=n_bars,
        freq="h",
        tz="UTC",
    )

    price = 45000.0
    records = []

    for i, ts in enumerate(timestamps):
        # Regime-varying volatility
        regime_cycle = np.sin(2 * np.pi * i / 1500) * 0.5 + 0.5
        vol = 0.002 + regime_cycle * 0.01

        ret = rng.normal(0, vol)
        o = price
        c = o * (1 + ret)

        # High/low with realistic wicks
        wick_factor = rng.uniform(0.3, 1.5)
        h = max(o, c) * (1 + abs(ret) * wick_factor * rng.uniform(0.1, 1.0))
        l = min(o, c) * (1 - abs(ret) * wick_factor * rng.uniform(0.1, 1.0))

        volume = max(0.0, rng.normal(2500, 900))
        taker_buy = max(0.0, min(volume, volume * rng.uniform(0.35, 0.65)))

        records.append(
            {
                "timestamp": ts,
                "open": o,
                "high": h,
                "low": l,
                "close": c,
                "volume": volume,
                "taker_buy_volume": taker_buy,
                "taker_sell_volume": volume - taker_buy,
                "funding_rate_raw": float(rng.normal(0.00001, 0.00015)),
                "open_interest": float(rng.normal(2.5e8, 6e7)),
                "open_interest_value": float(rng.normal(1.2e10, 2.5e9)),
            }
        )
        price = c

    df = pd.DataFrame(records).set_index("timestamp")
    df.index = pd.DatetimeIndex(df.index, tz="UTC")
    return df


def run_pipeline(
    df: pd.DataFrame,
    optimize: bool = False,
    prune_threshold: float = 0.0,
    validation_mode: str = "walk_forward",
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
    logger.info(
        "STEP 6: %s Validation%s",
        validation_mode.upper(),
        " (OPTIMIZED)" if optimize else "",
    )
    logger.info("=" * 60)
    wf_result = run_walk_forward(
        df_labeled,
        params_override=best_params,
        feature_subset=None, # We rely on trainer-internal pruning
        validation_mode=validation_mode,
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
        validation_mode=validation_mode,
    )

    # --- Summary ---
    verdict = determine_verdict(wf_result)
    logger.info("=" * 60)
    logger.info("RESEARCH COMPLETE")
    logger.info("=" * 60)
    logger.info("Duration: %.1f seconds", duration)
    logger.info("Verdict: %s", verdict)
    logger.info("Experiment log: %s", exp_path)

    threshold_count = int(
        round((cfg.threshold_max - cfg.threshold_min) / cfg.threshold_step)
    ) + 1
    n_trials_assumed = max(1, threshold_count * cfg.n_regimes)

    for h, report in wf_result.reports.items():
        pnl = wf_result.all_pnl.get(h, np.array([]))
        psr = probabilistic_sharpe_ratio(pnl)
        dsr = deflated_sharpe_ratio(pnl, n_trials=n_trials_assumed)

        logger.info(
            "  %dm → EV=%.6f | WR=%.1f%% | Sharpe=%.2f | PSR=%.3f | DSR=%.3f | Trades=%d | MaxDD=%.4f",
            h,
            report.overall_ev,
            report.overall_win_rate * 100,
            report.overall_sharpe,
            psr,
            dsr,
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
    parser = argparse.ArgumentParser(description="Crypto Quant Research Engine")
    parser.add_argument("--dry-run", action="store_true", help="Use synthetic crypto data (no API)")
    parser.add_argument("--fetch", action="store_true", help="Fetch data from Binance")
    parser.add_argument("--months", type=int, default=3, help="Months of history to fetch")
    parser.add_argument("--optimize", action="store_true", help="Run Optuna HPO + feature pruning")
    parser.add_argument("--prune", type=float, default=0.0, help="Feature pruning threshold (e.g. 0.01)")
    parser.add_argument("--bars", type=int, default=40000, help="Synthetic bar count for dry-run")
    parser.add_argument(
        "--validation-mode",
        type=str,
        choices=["walk_forward", "purged_kfold"],
        default="walk_forward",
        help="Validation engine to run (default: walk_forward)",
    )
    args = parser.parse_args()

    # Ensure output dirs exist
    get_path_config()

    if args.dry_run:
        logger.info("DRY RUN — using synthetic BTCUSDT data (%d bars)", args.bars)
        df = generate_synthetic_crypto(n_bars=args.bars)
        run_pipeline(
            df,
            optimize=args.optimize,
            prune_threshold=args.prune,
            validation_mode=args.validation_mode,
        )

    elif args.fetch:
        client = BinanceClient()

        date_to = datetime.now(timezone.utc)
        date_from = date_to - timedelta(days=args.months * 30)

        logger.info("Fetching BTCUSDT 1H data: %s → %s", date_from, date_to)
        ohlcv = client.fetch_historical(date_from, date_to)
        funding = client.fetch_funding_rates(date_from, date_to)
        oi = client.fetch_open_interest(date_from, date_to)
        df = BinanceClient.merge_supplementary(ohlcv, funding, oi)

        if df.empty:
            logger.error("No data received from API")
            sys.exit(1)

        # Save raw
        from quant.data.storage import save_raw

        label = f"BTCUSDT_1h_{date_from.strftime('%Y%m%d')}_{date_to.strftime('%Y%m%d')}"
        save_raw(df, label)

        run_pipeline(
            df,
            optimize=args.optimize,
            prune_threshold=args.prune,
            validation_mode=args.validation_mode,
        )

    else:
        # Try loading latest snapshot
        from quant.data.storage import load_latest_snapshot

        df = load_latest_snapshot()
        if df is None:
            logger.error("No data found. Use --dry-run or --fetch")
            sys.exit(1)

        run_pipeline(
            df,
            optimize=args.optimize,
            prune_threshold=args.prune,
            validation_mode=args.validation_mode,
        )


if __name__ == "__main__":
    main()
