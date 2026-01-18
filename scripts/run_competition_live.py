"""
Competition Live Runner - Single command to launch the hail mary trading system.

Usage:
    python scripts/run_competition_live.py --symbol GBPJPY --telegram --beep

This script:
1. Validates the FX data CSV exists
2. Promotes the CompetitionHailMary hypothesis
3. Starts the meta engine with aggressive settings
4. Optional: Starts signal watcher with Telegram alerts
"""

import argparse
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# MT5 data directory (where FXSymbolExporter writes)
MT5_FILES_DIR = Path(r"C:\Users\HP\AppData\Roaming\MetaQuotes\Terminal\10CE948A1DFC9A8C27E56E827008EBD4\MQL5\Files")
FX_DATA_PATH = MT5_FILES_DIR / "results" / "live_fx.csv"

# Competition settings
COMPETITION_CAPITAL = 10000.0
COMPETITION_POLICY = "COMPETITION_EVAL"
COMPETITION_EXECUTION_POLICY = "COMPETITION_5PERCENTERS"


def check_market_open() -> bool:
    """Check if FX market is likely open (crude check based on UTC time)."""
    now = datetime.now(timezone.utc)
    # FX market opens Sunday ~22:00 UTC, closes Friday ~22:00 UTC
    # This is a simplified check
    weekday = now.weekday()  # Monday=0, Sunday=6
    hour = now.hour
    
    if weekday == 6 and hour < 22:  # Sunday before open
        return False
    if weekday == 4 and hour >= 22:  # Friday after close
        return False
    if weekday == 5:  # Saturday
        return False
    return True


def wait_for_market_open():
    """Wait until FX market opens."""
    while not check_market_open():
        now = datetime.now(timezone.utc)
        print(f"[{now.strftime('%H:%M:%S')} UTC] FX market closed. Waiting...")
        time.sleep(60)  # Check every minute
    print("FX market is OPEN!")


def promote_hail_mary():
    """Ensure CompetitionHailMary hypothesis is promoted."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    from config.settings import get_settings
    from storage.repositories import EvaluationRepository
    from promotion.models import HypothesisStatus
    
    settings = get_settings()
    repo = EvaluationRepository(settings.database_path)
    
    hypothesis_id = "competition_hail_mary"
    
    # Check if already promoted
    promoted = repo.get_hypotheses_by_status(
        HypothesisStatus.PROMOTED.value,
        policy_id=COMPETITION_POLICY
    )
    
    if hypothesis_id in promoted:
        print(f"✓ {hypothesis_id} already promoted")
        return
    
    # Promote directly (bypass normal evaluation for competition)
    repo.store_hypothesis_status(
        hypothesis_id=hypothesis_id,
        status=HypothesisStatus.PROMOTED.value,
        policy_id=COMPETITION_POLICY,
        rationale=["Competition hail mary - manual promotion for final push"]
    )
    print(f"✓ Promoted {hypothesis_id}")


def validate_data_file(symbol: str) -> bool:
    """Validate the FX data CSV exists and has recent data."""
    if not FX_DATA_PATH.exists():
        print(f"✗ Data file not found: {FX_DATA_PATH}")
        print("  → Make sure FXSymbolExporter.mq5 is running in MT5")
        return False
    
    # Check file has content
    content = FX_DATA_PATH.read_text()
    lines = content.strip().split('\n')
    if len(lines) < 2:  # Header + at least 1 bar
        print(f"✗ Data file empty: {FX_DATA_PATH}")
        return False
    
    # Check symbol exists in data
    if symbol not in content:
        print(f"⚠ Symbol {symbol} not found in data file")
        print(f"  Available symbols: {set(l.split(',')[0] for l in lines[1:] if ',' in l)}")
        return False
    
    print(f"✓ Data file valid: {len(lines)-1} bars, {symbol} present")
    return True


def main():
    parser = argparse.ArgumentParser(description="Competition Live Runner")
    parser.add_argument("--symbol", default="GBPJPY", help="Primary trading symbol")
    parser.add_argument("--telegram", action="store_true", help="Enable Telegram alerts")
    parser.add_argument("--beep", action="store_true", help="Enable beep alerts")
    parser.add_argument("--wait-for-market", action="store_true", help="Wait for FX market to open")
    parser.add_argument("--dry-run", action="store_true", help="Validate setup without running")
    parser.add_argument("--skip-validation", action="store_true", help="Skip data file validation")
    args = parser.parse_args()
    
    print("=" * 60)
    print("COMPETITION HAIL MARY - LIVE RUNNER")
    print("=" * 60)
    print(f"Symbol: {args.symbol}")
    print(f"Capital: ${COMPETITION_CAPITAL:,.0f}")
    print(f"Leverage: 1:30 (max)")
    print(f"Policy: {COMPETITION_POLICY}")
    print()
    
    # Step 1: Promote hypothesis
    print("[1/4] Promoting CompetitionHailMary hypothesis...")
    try:
        promote_hail_mary()
    except Exception as e:
        print(f"✗ Failed to promote: {e}")
        return 1
    
    # Step 2: Wait for market if requested
    if args.wait_for_market:
        print("[2/4] Checking market status...")
        wait_for_market_open()
    else:
        print("[2/4] Skipping market check (--wait-for-market not set)")
    
    # Step 3: Validate data file
    if not args.skip_validation:
        print(f"[3/4] Validating data file...")
        if not validate_data_file(args.symbol):
            print()
            print("Data file not ready. Options:")
            print("  1. Attach FXSymbolExporter.mq5 to a chart in MT5")
            print("  2. Wait for first bar to be exported")
            print("  3. Run with --skip-validation to proceed anyway")
            return 1
    else:
        print("[3/4] Skipping data validation")
    
    if args.dry_run:
        print("[4/4] Dry run complete - not starting engine")
        return 0
    
    # Step 4: Start the engines
    print("[4/4] Starting trading engine...")
    print()
    
    # Build run_meta command
    meta_cmd = [
        sys.executable, "-m", "orchestrator.run_meta",
        "--policy", COMPETITION_POLICY,
        "--symbol", args.symbol,
        "--data-path", str(FX_DATA_PATH),
        "--capital", str(COMPETITION_CAPITAL),
        "--execution-policy", COMPETITION_EXECUTION_POLICY,
        "--paper",
        "--paper-log", "results/competition_live_log.jsonl",
        "--watch",
        "--poll-interval", "30",
        "--tag", "COMPETITION_HAILMARY",
    ]
    
    print(f"Command: {' '.join(meta_cmd)}")
    print()
    print("=" * 60)
    print("ENGINE STARTING - Monitor output below")
    print("=" * 60)
    print()
    
    # Run in foreground
    try:
        subprocess.run(meta_cmd, check=True)
    except KeyboardInterrupt:
        print("\nStopped by user")
    except subprocess.CalledProcessError as e:
        print(f"Engine exited with error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
