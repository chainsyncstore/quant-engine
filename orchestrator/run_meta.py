import argparse
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import pandas as pd

from config.competition_flags import COMPETITION_MODE
from config.competition_crypto_symbols import CRYPTO_SYMBOLS
from config.settings import get_settings
from config.policies import get_policy
from config.execution_policies import get_execution_policy
from storage.repositories import EvaluationRepository
from portfolio.meta_engine import MetaPortfolioEngine
from portfolio.ensemble import Ensemble
from portfolio.weighting import EqualWeighting, RobustnessWeighting
from portfolio.risk import (
    ExecutionPolicyRule,
    LossStreakGuard,
    MaxDrawdownRule,
    TradeThrottle,
)
from data.market_loader import MarketDataLoader
from hypotheses.registry import get_hypothesis
from promotion.models import HypothesisStatus
from execution.cost_model import CostModel
from execution_live import ExecutionEventLogger, PaperExecutionAdapter
from execution_live.intent_sink import FileIntentSink
from execution_live.intent_schema import ExecutionIntent as MT5Intent, Side, Mode
from execution_live.events import COMPETITION_PROFILE_LOADED

# Competition mode symbol rotation pool - use crypto symbols
COMPETITION_SYMBOL_POOL = CRYPTO_SYMBOLS
from execution_live.risk_checks import CashAvailabilityCheck, NotionalLimitCheck, ExecutionPolicyCheck
from execution_live.service import PaperExecutionService


def force_close_symbol(adapter: "PaperExecutionAdapter", symbol: str) -> None:
    """
    Force close any open position on the specified symbol.
    Used at startup in competition mode to exit stale positions.
    """
    positions = adapter.get_positions()
    for p in positions:
        if p.symbol == symbol:
            from execution_live.order_models import ExecutionIntent, IntentAction
            from datetime import datetime, timezone
            close_intent = ExecutionIntent(
                symbol=symbol,
                action=IntentAction.CLOSE,
                quantity=p.quantity,
                timestamp=datetime.now(timezone.utc),
                reference_price=p.average_price,
            )
            report = adapter.place_order(close_intent)
            logger.info(
                "[COMPETITION] Forced close on %s | status=%s qty=%.2f",
                symbol,
                report.status.value,
                p.quantity,
            )


def choose_rotation_symbol(adapter: "PaperExecutionAdapter", current_symbol: str) -> str:
    """
    Choose next symbol from rotation pool if no position is open.
    Rotates to the next symbol in the pool after closing a position.
    Only 1 position at a time - must close before trading a different pair.
    """
    positions = adapter.get_positions()
    if positions:
        # Already have a position, stay on current symbol
        return current_symbol

    # No position - rotate to NEXT symbol in pool (round-robin)
    if current_symbol in COMPETITION_SYMBOL_POOL:
        current_idx = COMPETITION_SYMBOL_POOL.index(current_symbol)
        next_idx = (current_idx + 1) % len(COMPETITION_SYMBOL_POOL)
        next_symbol = COMPETITION_SYMBOL_POOL[next_idx]
        if next_symbol != current_symbol:
            logger.info("[COMPETITION] Rotating from %s to %s", current_symbol, next_symbol)
        return next_symbol
    
    # Current symbol not in pool, start with first
    logger.info("[COMPETITION] Rotating to %s", COMPETITION_SYMBOL_POOL[0])
    return COMPETITION_SYMBOL_POOL[0]


def explore_best_symbol(
    hypotheses: list,
    csv_df: pd.DataFrame,
    symbol_pool: list,
    adapter: "PaperExecutionAdapter" = None,
) -> tuple[str, float]:
    """
    Exploratory symbol selection: evaluate ALL symbols and pick the one
    with the strongest actionable signal.
    
    Returns:
        (best_symbol, signal_strength) where signal_strength > 0 means actionable
    """
    from state.market_state import MarketState
    from hypotheses.base import IntentType
    
    # If we have an open position, stay on that symbol
    if adapter:
        positions = adapter.get_positions()
        if positions:
            return positions[0].symbol, 1.0
    
    best_symbol = symbol_pool[0] if symbol_pool else "BTCUSD"
    best_score = 0.0
    
    for symbol in symbol_pool:
        # Filter bars for this symbol
        symbol_df = csv_df[csv_df["symbol"] == symbol] if "symbol" in csv_df.columns else csv_df
        if symbol_df.empty:
            logger.debug("[EXPLORE] %s: no bars", symbol)
            continue
        
        try:
            bars = MarketDataLoader.load_from_dataframe(symbol_df, symbol=symbol)
            if not bars or len(bars) < 15:
                logger.debug("[EXPLORE] %s: insufficient bars (%d)", symbol, len(bars) if bars else 0)
                continue
            
            # Build market state for this symbol
            market_state = MarketState(lookback_window=100)
            for bar in bars:
                market_state.update(bar)
            
            # Evaluate each hypothesis
            symbol_score = 0.0
            for h in hypotheses:
                try:
                    from state.position_state import PositionState
                    from clock.clock import Clock
                    intent = h.on_bar(market_state, PositionState(), Clock())
                    if intent and intent.type in (IntentType.BUY, IntentType.SELL):
                        # Actionable signal found
                        symbol_score += intent.size
                        logger.info(
                            "[EXPLORE] %s: %s signal from %s (size=%.2f)",
                            symbol,
                            intent.type.value,
                            getattr(h, 'hypothesis_id', h.__class__.__name__),
                            intent.size,
                        )
                except Exception as e:
                    logger.debug("[EXPLORE] %s hypothesis error: %s", symbol, e)
                    continue
            
            if symbol_score > best_score:
                best_score = symbol_score
                best_symbol = symbol
                
        except Exception as e:
            logger.debug("[EXPLORE] %s load error: %s", symbol, e)
            continue
    
    if best_score > 0:
        logger.info("[EXPLORE] Best symbol: %s (score=%.2f)", best_symbol, best_score)
    else:
        logger.info("[EXPLORE] No actionable signals on any symbol, defaulting to %s", best_symbol)
    
    return best_symbol, best_score


def filter_new_bars(df: pd.DataFrame, last_seen_ts: datetime | None, lookback_bars: int = 50):
    """
    Idempotent guard to drop bars that have already been processed.
    Includes lookback_bars of historical context so hypotheses can calculate indicators.
    
    Returns: (filtered_df, first_new_bar_index) - index within the filtered df where new bars start
    """
    if df.empty:
        return df, 0

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    if last_seen_ts is None:
        return df, 0
    
    # Find new bars
    new_mask = df["timestamp"] > last_seen_ts
    if not new_mask.any():
        return df[new_mask], 0  # Empty dataframe
    
    # Include lookback_bars before the first new bar for hypothesis context
    first_new_idx = new_mask.idxmax()
    lookback_start = max(0, first_new_idx - lookback_bars)
    
    # Calculate where new bars start in the filtered result
    first_new_in_result = first_new_idx - lookback_start
    
    return df.iloc[lookback_start:].copy(), first_new_in_result


def _load_last_seen_timestamp(state_path: Path) -> datetime | None:
    if not state_path.exists():
        return None

    try:
        raw = state_path.read_text(encoding="utf-8").strip()
        if not raw:
            return None
        return datetime.fromisoformat(raw)
    except Exception as exc:
        logger.warning("Failed to parse last_seen state file %s: %s", state_path, exc)
        return None


def _persist_last_seen_timestamp(state_path: Path, timestamp: datetime) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(timestamp.isoformat(), encoding="utf-8")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
DEBUG_TRACE_PATH = Path("results/run_meta_debug.log")


def _write_mt5_intent(sink: "FileIntentSink", intent, policy_id: str) -> None:
    """Convert internal ExecutionIntent to MT5 format and write to file sink."""
    import uuid
    import json
    from execution_live.order_models import IntentAction
    
    side = "BUY" if intent.action == IntentAction.BUY else "SELL"
    if intent.action == IntentAction.CLOSE:
        # For CLOSE, we need to determine the opposite side or handle separately
        # For now, skip CLOSE intents as MT5 handles position closing differently
        logger.info("Skipping CLOSE intent for MT5 - position management handled by EA")
        return
    
    # Convert units to MT5 lots (1 lot = 100,000 units for forex)
    # Round to 2 decimal places (0.01 lot minimum for most brokers)
    LOT_SIZE = 100_000
    MIN_LOT = 0.01
    lots = round(intent.quantity / LOT_SIZE, 2)
    
    # If system wants to trade but lot size is below minimum, use minimum lot
    # This ensures we participate in the market when signals fire
    if lots < MIN_LOT:
        logger.info("MT5 lot size %.4f below min, using min lot %.2f", lots, MIN_LOT)
        lots = MIN_LOT
    
    mt5_intent = {
        "intent_id": str(uuid.uuid4()),
        "timestamp": intent.timestamp.isoformat() if hasattr(intent.timestamp, 'isoformat') else str(intent.timestamp),
        "symbol": intent.symbol,
        "side": side,
        "order_type": "MARKET",
        "quantity": lots,  # Now in lots, not units
        "stop_loss": None,
        "take_profit": None,
        "time_in_force": "GTC",
        "policy_hash": policy_id,
        "mode": "LIVE"
    }
    
    sink.emit(json.dumps(mt5_intent))
    logger.info("MT5 intent written | id=%s side=%s lots=%.2f", mt5_intent["intent_id"], side, lots)


def main():
    parser = argparse.ArgumentParser(description="Run meta-strategy simulation.")
    parser.add_argument("--policy", required=True, help="Research Policy ID")
    parser.add_argument("--symbol", default="SYNTHETIC", help="Market Symbol")
    parser.add_argument("--data-path", required=True, help="Path to market data CSV")
    parser.add_argument("--capital", type=float, default=100000.0, help="Initial Capital")
    parser.add_argument("--max-drawdown", type=float, default=0.20, help="Max Drawdown Limit")
    parser.add_argument("--weighting", default="equal", choices=["equal", "robustness"], help="Weighting strategy")
    parser.add_argument("--tag", default="META_RUN", help="Meta Run Tag")
    parser.add_argument("--paper", action="store_true", help="Enable paper execution boundary")
    parser.add_argument("--paper-log", default=None, help="Optional path to append execution events as JSON lines")
    parser.add_argument("--paper-max-notional", type=float, default=0.0, help="Max notional per order for paper execution (0 disables check)")
    parser.add_argument("--execution-policy", default="RESEARCH", help="Execution policy ID to enforce")
    parser.add_argument(
        "--state-path",
        default=None,
        help="Optional path to persist last processed bar timestamp (defaults to <data-path>.state)",
    )
    parser.add_argument(
        "--explain-decisions",
        action="store_true",
        help="Emit structured reasons when trade decisions are blocked"
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Enable live MT5 execution via file-based intent sink"
    )
    parser.add_argument(
        "--intent-dir",
        default=None,
        help="Directory for MT5 intent files (default: MQL5/Files/execution_intents)"
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Continuous mode: poll for new bars instead of running once"
    )
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=30,
        help="Seconds between polls in watch mode (default: 30)"
    )
    
    args = parser.parse_args()
    
    DEBUG_TRACE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with DEBUG_TRACE_PATH.open("a", encoding="utf-8") as handle:
        handle.write(f"{datetime.now().isoformat()} | ARGS: {args}\n")
    
    settings = get_settings()
    repo = EvaluationRepository(settings.database_path)
    policy = get_policy(args.policy)

    execution_policy_id = args.execution_policy
    if COMPETITION_MODE:
        logger.info(
            "COMPETITION_MODE enabled. Overriding requested execution policy %s with COMPETITION_5PERCENTERS",
            execution_policy_id,
        )
        execution_policy_id = "COMPETITION_5PERCENTERS"
    execution_policy = get_execution_policy(execution_policy_id)

    telemetry_hook: Optional[Callable[[str, dict], None]] = None
    if COMPETITION_MODE:
        def _emit_competition_event(event_type: str, payload: dict) -> None:
            logger.info(
                "competition_telemetry | event=%s payload=%s",
                event_type,
                payload,
            )

        telemetry_hook = _emit_competition_event
    
    logger.info(
        "Starting Meta-Strategy Simulation for policy %s on %s with execution policy %s",
        policy.policy_id,
        args.symbol,
        execution_policy.label,
    )
    
    # 1. Fetch Hypotheses
    promoted_ids = repo.get_hypotheses_by_status(
        HypothesisStatus.PROMOTED.value,
        policy_id=policy.policy_id
    )
    with DEBUG_TRACE_PATH.open("a", encoding="utf-8") as handle:
        handle.write(f"{datetime.now().isoformat()} | PROMOTED_COUNT: {len(promoted_ids)}\n")
    
    if not promoted_ids:
        logger.warning("No PROMOTED hypotheses found.")
        return

    # Competition mode: filter to aggressive hypotheses only
    AGGRESSIVE_HYPOTHESES = [
        "crypto_momentum_breakout",
        "rsi_extreme_reversal", 
        "volatility_expansion_assault",
    ]
    if COMPETITION_MODE:
        aggressive_ids = [h for h in promoted_ids if h in AGGRESSIVE_HYPOTHESES]
        if aggressive_ids:
            logger.info(
                "[COMPETITION] Filtering to aggressive hypotheses: %s (dropped: %s)",
                aggressive_ids,
                [h for h in promoted_ids if h not in AGGRESSIVE_HYPOTHESES],
            )
            promoted_ids = aggressive_ids

    logger.info(f"Found {len(promoted_ids)} promoted hypotheses: {promoted_ids}")
    
    import json
    hypotheses = []
    for hid in promoted_ids:
        details = repo.get_hypothesis_details(hid)
        params = {}
        if details and 'parameters_json' in details:
            try:
                params = json.loads(details['parameters_json'])
            except json.JSONDecodeError:
                logger.error(f"Failed to load params for {hid}")
        
        h_cls = get_hypothesis(hid)
        hypothesis = h_cls(**params)
        setattr(hypothesis, "explain_decisions", args.explain_decisions)
        logger.info(
            "hypothesis_init | name=%s | explain_decisions=%s",
            getattr(hypothesis, "name", getattr(hypothesis, "hypothesis_id", h_cls.__name__)),
            getattr(hypothesis, "explain_decisions", None),
        )
        hypotheses.append(hypothesis)
        
    # 2. Configure Weighting
    weighting_strategy = EqualWeighting()
    if args.weighting == "robustness":
        weighting_strategy = RobustnessWeighting()
        
    ensemble = Ensemble(
        hypotheses=hypotheses,
        weighting_strategy=weighting_strategy,
        repo=repo,
        policy_id=policy.policy_id
    )
    
    logger.info(f"Initial Weights: {ensemble.weights}")
    
    # 3. Load Data (include lookback for context, track where new bars start)
    state_path = Path(args.state_path) if args.state_path else Path(args.data_path).with_suffix(".state")
    
    def process_new_bars(current_symbol: str = None):
        """Process any new bars in the CSV. Returns True if bars were processed."""
        csv_df = pd.read_csv(args.data_path)
        last_seen_ts = _load_last_seen_timestamp(state_path)
        new_rows, first_actionable_idx = filter_new_bars(csv_df, last_seen_ts)

        if new_rows.empty:
            return False, None, None, 0

        symbol_to_load = current_symbol or args.symbol
        
        # In competition mode, check for bars for ANY symbol in the pool
        # The symbol selection happens later via explore_best_symbol
        if COMPETITION_MODE:
            # Check if any symbol in the pool has new bars
            has_any_bars = False
            for sym in COMPETITION_SYMBOL_POOL:
                sym_bars = MarketDataLoader.load_from_dataframe(new_rows, symbol=sym)
                if sym_bars:
                    has_any_bars = True
                    break
            if not has_any_bars:
                return False, None, None, 0
            # Load bars for current symbol (will be updated after exploration)
            bars = MarketDataLoader.load_from_dataframe(new_rows, symbol=symbol_to_load)
        else:
            bars = MarketDataLoader.load_from_dataframe(new_rows, symbol=symbol_to_load)
            if not bars:
                return False, None, None, 0
        
        return True, new_rows, bars, first_actionable_idx
    
    # Initial check for bars
    has_bars, new_rows, bars, first_actionable_idx = process_new_bars()
    
    if not has_bars:
        if args.watch:
            logger.info("Watch mode: No new bars yet, will poll every %ds...", args.poll_interval)
        else:
            logger.info("No new bars detected in %s", args.data_path)
            return
        
    # 4. Init Engine
    cost_model = CostModel(
        transaction_cost_bps=policy.transaction_cost_bps,
        slippage_bps=policy.slippage_bps
    )
    
    paper_adapter = None
    execution_sink = None
    event_logger: Optional[ExecutionEventLogger] = None
    
    # Initialize live sink early so it's available in paper execution block
    live_sink = None
    if args.live:
        intent_dir = args.intent_dir or r"C:\Users\HP\AppData\Roaming\MetaQuotes\Terminal\10CE948A1DFC9A8C27E56E827008EBD4\MQL5\Files\execution_intents"
        live_sink = FileIntentSink(intent_dir)
        logger.info("Live MT5 execution enabled. Intent dir: %s", intent_dir)
    
    if args.paper:
        if args.paper_log:
            Path(args.paper_log).parent.mkdir(parents=True, exist_ok=True)
        logger.info("Paper execution enabled.")
        event_logger = ExecutionEventLogger(persist_path=args.paper_log)
        event_logger.log(
            "execution_policy_loaded",
            {
                "policy_id": execution_policy.policy_id,
                "label": execution_policy.label,
                "config": execution_policy.serialize(),
            },
        )
        if COMPETITION_MODE:
            telemetry_hook = event_logger.log
            event_logger.log(
                COMPETITION_PROFILE_LOADED,
                {
                    "policy_id": execution_policy.policy_id,
                    "label": execution_policy.label,
                    "tag": args.tag,
                },
            )

        adapter_risk_checks = [CashAvailabilityCheck(leverage=30.0), ExecutionPolicyCheck(execution_policy)]
        if args.paper_max_notional > 0:
            adapter_risk_checks.append(NotionalLimitCheck(args.paper_max_notional))
        
        paper_adapter = PaperExecutionAdapter(
            cost_model=cost_model,
            initial_equity=args.capital,
            risk_checks=adapter_risk_checks,
            event_logger=event_logger,
            leverage=30.0,  # 1:30 leverage for crypto trading
        )
        paper_service = PaperExecutionService(paper_adapter)
        
        # Competition mode: exploratory multi-symbol evaluation
        if COMPETITION_MODE:
            force_close_symbol(paper_adapter, "EURUSD")
            # Explore ALL symbols and pick the one with best signal
            csv_df_init = pd.read_csv(args.data_path)
            best_symbol, signal_score = explore_best_symbol(
                hypotheses=hypotheses,
                csv_df=csv_df_init,
                symbol_pool=COMPETITION_SYMBOL_POOL,
                adapter=paper_adapter,
            )
            args.symbol = best_symbol
            logger.info("[COMPETITION] Active symbol: %s (signal_score=%.2f)", args.symbol, signal_score)
        
        def execution_sink(intent):
            # Skip execution for lookback context bars (already processed)
            bar_idx = intent.metadata.get("bar_index", 0)
            if bar_idx < first_actionable_idx:
                logger.debug(
                    "Skipping lookback bar %d (first actionable: %d)",
                    bar_idx, first_actionable_idx
                )
                return
            
            logger.info(
                "ExecutionIntent -> symbol=%s action=%s qty=%.6f bar=%s",
                intent.symbol,
                intent.action.value,
                intent.quantity,
                bar_idx,
            )
            report = paper_service.handle_intent(intent)
            logger.info(
                "ExecutionReport <- order=%s status=%s filled=%.2f msg=%s",
                report.order_id,
                report.status.value,
                report.filled_quantity,
                report.message or "",
            )
            
            # Competition logging: trade_executed event
            if report.status.value == "FILLED":
                logger.info(
                    "trade_executed | symbol=%s action=%s qty=%.2f price=%.5f order_id=%s",
                    intent.symbol,
                    intent.action.value,
                    report.filled_quantity,
                    report.avg_fill_price or 0.0,
                    report.order_id,
                )
            
            # If --live is also enabled, write intent file for MT5
            if args.live and live_sink is not None:
                _write_mt5_intent(live_sink, intent, execution_policy.policy_id)
    
    # If only --live (no --paper), create a standalone execution sink
    if args.live and not args.paper:
        def execution_sink(intent):
            # Skip execution for lookback context bars
            bar_idx = intent.metadata.get("bar_index", 0)
            if bar_idx < first_actionable_idx:
                return
            
            logger.info(
                "ExecutionIntent -> MT5 | symbol=%s action=%s qty=%.6f",
                intent.symbol,
                intent.action.value,
                intent.quantity,
            )
            _write_mt5_intent(live_sink, intent, execution_policy.policy_id)
    
    elif telemetry_hook and COMPETITION_MODE and not args.paper and not args.live:
        telemetry_hook(
            COMPETITION_PROFILE_LOADED,
            {
                "policy_id": execution_policy.policy_id,
                "label": execution_policy.label,
                "tag": args.tag,
            },
        )

    risk_rules = [
        MaxDrawdownRule(max_drawdown_pct=args.max_drawdown),
        TradeThrottle(telemetry_hook=telemetry_hook),
        LossStreakGuard(max_losses=5, telemetry_hook=telemetry_hook),  # Allow 5 losses/day for aggressive competition
        ExecutionPolicyRule(execution_policy),
    ]
    
    # Pass crypto rotation symbols in competition mode
    rotation_symbols = CRYPTO_SYMBOLS if COMPETITION_MODE else []
    
    engine = MetaPortfolioEngine(
        ensemble=ensemble,
        initial_capital=args.capital,
        cost_model=cost_model,
        risk_rules=risk_rules,
        symbol=args.symbol,
        execution_intent_sink=execution_sink,
        telemetry=telemetry_hook,
        explain_decisions=args.explain_decisions,
        rotation_symbols=rotation_symbols,
    )
    
    # Pre-populate all rotation symbols' market states with historical data
    if COMPETITION_MODE and rotation_symbols:
        csv_df_full = pd.read_csv(args.data_path)
        for sym in rotation_symbols:
            sym_bars = MarketDataLoader.load_from_dataframe(csv_df_full, symbol=sym)
            if sym_bars and sym in engine._symbol_market_states:
                for bar in sym_bars:
                    engine._symbol_market_states[sym].update(bar)
                logger.info("[INIT] Populated %s market state with %d bars", sym, len(sym_bars))
    
    # 5. Run (with watch loop support)
    def run_iteration():
        """Run one iteration of bar processing."""
        nonlocal has_bars, new_rows, bars, first_actionable_idx
        
        if not has_bars:
            return False
        
        history = engine.run(bars)
        
        # Store + persist progress
        if history:
            logger.info(f"Processed {len(history)} bars...")
            for state in history:
                repo.store_portfolio_evaluation(state, args.tag, policy.policy_id)

        latest_bar_ts = pd.to_datetime(new_rows["timestamp"]).max()
        if isinstance(latest_bar_ts, pd.Timestamp):
            latest_bar_ts = latest_bar_ts.to_pydatetime()
        if latest_bar_ts:
            _persist_last_seen_timestamp(state_path, latest_bar_ts)
            
        if history:
            final = history[-1]
            if paper_adapter:
                account = paper_adapter.get_account_state()
                logger.info(
                    "Equity: %.2f | Cash: %.2f | Positions: %d",
                    account.equity,
                    account.cash,
                    len(account.positions),
                )
        return True
    
    # Watch mode: continuous polling loop
    if args.watch:
        logger.info("=" * 50)
        logger.info("WATCH MODE: Polling every %ds (Ctrl+C to stop)", args.poll_interval)
        logger.info("=" * 50)
        
        try:
            while True:
                # Process any available bars
                if has_bars:
                    run_iteration()
                
                # Wait and poll for new bars
                time.sleep(args.poll_interval)
                logger.info("[WATCH] Polling for new bars...")
                has_bars, new_rows, bars, first_actionable_idx = process_new_bars(engine.symbol)
                
                if not has_bars:
                    logger.info("[WATCH] No new bars yet, waiting...")
                
                if has_bars:
                    logger.info("[WATCH] New bars detected, processing...")
                    # In watch mode, all bars are truly new (engine already has historical context)
                    # Reset first_actionable_idx to 0 so execution_sink doesn't skip them
                    first_actionable_idx = 0
                    
                    # Update engine symbol if needed for rotation
                    if COMPETITION_MODE and paper_adapter:
                        csv_df_fresh = pd.read_csv(args.data_path)
                        best_symbol, signal_score = explore_best_symbol(
                            hypotheses=hypotheses,
                            csv_df=csv_df_fresh,
                            symbol_pool=COMPETITION_SYMBOL_POOL,
                            adapter=paper_adapter,
                        )
                        rotating = best_symbol != engine.symbol
                        if rotating:
                            logger.info("[WATCH] Rotating to %s (score=%.2f)", best_symbol, signal_score)
                        engine.symbol = best_symbol
                        args.symbol = best_symbol
                        
                        # Check if new symbol's market state needs historical data
                        symbol_state = engine._symbol_market_states.get(best_symbol)
                        needs_history = symbol_state is None or symbol_state.bar_count() < 25
                        
                        if needs_history:
                            # Load full historical bars for new symbol to populate market state
                            all_hist_bars = MarketDataLoader.load_from_dataframe(csv_df_fresh, symbol=best_symbol)
                            if all_hist_bars:
                                logger.info("[WATCH] Populating %s market state with %d historical bars", best_symbol, len(all_hist_bars))
                                # Feed historical bars to engine (they won't trigger trades due to clock filter)
                                for hist_bar in all_hist_bars[:-1]:  # All but the last (new) bar
                                    if best_symbol in engine._symbol_market_states:
                                        engine._symbol_market_states[best_symbol].update(hist_bar)
                        
                        # Now load only the new bar(s) for actual processing
                        all_symbol_bars = MarketDataLoader.load_from_dataframe(new_rows, symbol=best_symbol)
                        current_clock = engine.clock.now() if engine.clock.is_initialized() else None
                        if current_clock:
                            bars = [b for b in all_symbol_bars if b.timestamp > current_clock]
                            logger.info("[WATCH] Loaded %d bars for %s after clock %s", len(bars), best_symbol, current_clock)
                            if not bars:
                                logger.info("[WATCH] No forward bars for %s, skipping iteration", best_symbol)
                                has_bars = False
                        else:
                            bars = all_symbol_bars
                            logger.info("[WATCH] Loaded %d bars for %s (no clock filter)", len(bars), best_symbol)
                            
        except KeyboardInterrupt:
            logger.info("\n[WATCH] Stopped by user")
            if paper_adapter:
                account = paper_adapter.get_account_state()
                logger.info("Final: Equity=%.2f | Positions=%d", account.equity, len(account.positions))
    else:
        # Single run mode
        run_iteration()
        
        if not has_bars:
            logger.warning("No portfolio snapshots generated - no bars matched current symbol")
            return
            
        logger.info("--- Meta Portfolio Result ---")
        if paper_adapter:
            account = paper_adapter.get_account_state()
            logger.info(
                "Equity: %.2f | Cash: %.2f | Positions: %d",
                account.equity,
                account.cash,
                len(account.positions),
            )


if __name__ == "__main__":
    main()
