"""CLI entry point for single-symbol walk-forward backtest.

Usage:
    python scratch/run_backtest.py
    python scratch/run_backtest.py --symbol ETHUSDT --start 2024-01-01 --end 2024-06-30
    python scratch/run_backtest.py --compare-version model_20240601_120000

Set environment variables before running:
    BOT_MODEL_ROOT       Path to model artifacts (default: models/production)
    BOT_MODEL_REGISTRY_ROOT  Path to registry (default: models/production/registry)
    BINANCE_API_KEY / BINANCE_API_SECRET  (not required for market data)
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("backtest_cli")


def main() -> None:
    parser = argparse.ArgumentParser(description="Walk-forward backtester")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--start", default="2024-01-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default="2024-06-30", help="End date YYYY-MM-DD")
    parser.add_argument("--equity", type=float, default=300.0, help="Initial equity in USD")
    parser.add_argument("--min-confidence", type=float, default=0.65)
    parser.add_argument("--max-exposure", type=float, default=0.15)
    parser.add_argument("--maker-fee-bps", type=float, default=2.0)
    parser.add_argument("--taker-fee-bps", type=float, default=4.0)
    parser.add_argument("--model-version", default=None, help="Specific version ID to test")
    parser.add_argument("--compare-version", default=None,
                        help="Second version ID for comparison report")
    parser.add_argument("--output", default=None, help="Output HTML path")
    args = parser.parse_args()

    from quant_v2.research.backtester import BacktestConfig, run_backtest
    from quant_v2.research.backtest_report import generate_report

    config = BacktestConfig(
        symbol=args.symbol,
        start_date=args.start,
        end_date=args.end,
        initial_equity=args.equity,
        min_confidence=args.min_confidence,
        max_symbol_exposure_frac=args.max_exposure,
        maker_fee_bps=args.maker_fee_bps,
        taker_fee_bps=args.taker_fee_bps,
        model_version=args.model_version,
    )

    logger.info("Running backtest: %s %s→%s", config.symbol, config.start_date, config.end_date)
    result = run_backtest(config)

    comparison = None
    if args.compare_version:
        logger.info("Running comparison backtest with version: %s", args.compare_version)
        comp_config = BacktestConfig(
            symbol=args.symbol,
            start_date=args.start,
            end_date=args.end,
            initial_equity=args.equity,
            min_confidence=args.min_confidence,
            max_symbol_exposure_frac=args.max_exposure,
            maker_fee_bps=args.maker_fee_bps,
            taker_fee_bps=args.taker_fee_bps,
            model_version=args.compare_version,
        )
        comparison = run_backtest(comp_config)

    output_path = Path(args.output) if args.output else None
    report_path = generate_report(result, output_path=output_path, comparison=comparison)
    logger.info("Report written to: %s", report_path)

    print("\n" + "=" * 60)
    print(f"  Symbol:       {result.config.symbol}")
    print(f"  Period:       {result.config.start_date} → {result.config.end_date}")
    print(f"  Net PnL:      ${result.net_pnl:,.2f}")
    print(f"  Sharpe:       {result.sharpe:.3f}")
    print(f"  Max DD:       {result.max_drawdown*100:.2f}%")
    print(f"  Win rate:     {result.win_rate*100:.1f}%")
    print(f"  Total trades: {result.total_trades}")
    print(f"  Total fees:   ${result.total_fees:.4f}")
    print(f"  Report:       {report_path}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
