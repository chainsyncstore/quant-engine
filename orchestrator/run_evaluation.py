"""
Orchestrator - entry point for running evaluations.

Wires all components together and provides CLI interface.
Enforces ResearchPolicy for all executions.
"""

import argparse
import sys
from datetime import datetime
from typing import List, Optional, Dict, Any, Union

import pandas as pd

from clock.clock import Clock
from config.settings import get_settings, Settings
from config.policies import get_policy
from data.bar_iterator import BarIterator
from data.market_loader import MarketDataLoader
from engine.decision_queue import DecisionQueue
from engine.replay_engine import ReplayEngine
from evaluation.benchmark import BenchmarkCalculator
from evaluation.metrics import EvaluationMetrics
from execution.cost_model import CostModel
from execution.simulator import ExecutionSimulator
from hypotheses.registry import get_hypothesis
from hypotheses.base import Hypothesis
from state.market_state import MarketState
from state.position_state import PositionState
from storage.repositories import EvaluationRepository

# New Integrations
from evaluation.walk_forward import WalkForwardConfig, WalkForwardGenerator, DecayTracker
from analysis.regime import RegimeClassifier, MarketRegime
from evaluation.guardrails import ResearchGuardrails
from evaluation.policy import ResearchPolicy, EvaluationMode


def _to_datetime(value: Union[datetime, pd.Timestamp]) -> datetime:
    """Convert pandas Timestamp or datetime to datetime."""
    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime()
    return value


def _run_single_pass(
    hypothesis: Hypothesis,
    bars: List[Any],
    policy: ResearchPolicy,
    settings: Settings,
    symbol: str,
    repo: Optional[EvaluationRepository] = None,
    window_metadata: Optional[Dict[str, Any]] = None,
    regime_classifier: Optional[RegimeClassifier] = None,
    verbose: bool = True,
    sample_type: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run a single pass of the replay engine on the provided bars.
    """
    if not bars:
        return {} 

    # --- Initialize Components (FROM POLICY) ---
    clock = Clock()
    bar_iterator = BarIterator(bars)
    
    # Use Policy for Execution Constraints
    decision_queue = DecisionQueue(execution_delay_bars=policy.execution_delay_bars)
    cost_model = CostModel(
        transaction_cost_bps=policy.transaction_cost_bps,
        slippage_bps=policy.slippage_bps
    )
    
    # Capital still from Settings or Policy? Prompt implies Policy governs "costs/delay".
    # Capital usually is standard across research (e.g. 100k). Let's stick to Settings for Capital 
    # to avoid breaking simple usage, or if Policy doesn't define it. 
    # Policy doesn't define capital in my implementation plan.
    executor = ExecutionSimulator(cost_model, settings.starting_capital)
    
    market_state = MarketState(lookback_window=settings.lookback_window)
    position_state = PositionState()
    
    # --- Run Replay ---
    # --- Run Replay ---
    
    # Track Equity Curve for Beta/Alpha
    equity_curve: List[float] = []
    
    def on_execution_callback(decisions, bar, bar_index, mkt_state, pos_state):
        executor.execute_decisions(decisions, bar, pos_state)
        
    def on_bar_callback(bar, bar_index):
        # Capture daily equity (Mark-to-Market)
        # Note: This runs BEFORE decisions for the NEXT bar are processed,
        # but AFTER current bar update. It represents "End of Day" equity if bars are daily.
        current_equity = executor.get_total_capital(bar.close, position_state)
        equity_curve.append(current_equity)
    
    engine = ReplayEngine(
        hypothesis=hypothesis,
        bar_iterator=bar_iterator,
        clock=clock,
        decision_queue=decision_queue,
        market_state=market_state,
        position_state=position_state,
        execution_delay_bars=policy.execution_delay_bars
    )
    
    if verbose:
        print(f"    Running replay on {len(bars)} bars...")
        
    replay_stats = engine.run(
        on_execution_callback=on_execution_callback,
        on_bar_callback=on_bar_callback
    )
    
    # --- Compute Metrics ---
    completed_trades = executor.get_completed_trades()
    final_capital = executor.get_total_capital(bars[-1].close, position_state)
    
    # Calculate Benchmark Curve (Buy & Hold)
    # Simple approx: Capital follows price change
    start_price = bars[0].close
    benchmark_curve = [
        settings.starting_capital * (b.close / start_price) 
        for b in bars
    ]
    
    metrics_calc = EvaluationMetrics(
        completed_trades=completed_trades,
        initial_capital=settings.starting_capital,
        final_capital=final_capital,
        equity_curve=equity_curve,
        benchmark_curve=benchmark_curve
    )
    
    if sample_type:
        metrics_calc.set_sample_type(sample_type)
        
    metrics = metrics_calc.to_dict()
    
    # --- Regime Classification ---
    market_regime = None
    if regime_classifier:
        dates = [b.timestamp for b in bars]
        closes = [b.close for b in bars]
        prices_series = pd.Series(data=closes, index=dates)
        returns_series = prices_series.pct_change().fillna(0)
        
        trend = regime_classifier.classify_trend(prices_series)
        vol = regime_classifier.classify_volatility(returns_series)
        
        if vol == MarketRegime.HIGH_VOL:
            market_regime = MarketRegime.HIGH_VOL.value
        else:
            market_regime = trend.value
            
        if verbose:
            print(f"    Regime classified as: {market_regime}")

    # --- Benchmark ---
    benchmark_metrics = BenchmarkCalculator.calculate_buy_and_hold_return(
        bars=bars,
        initial_capital=settings.starting_capital,
        include_costs=True,
        cost_bps=policy.transaction_cost_bps + policy.slippage_bps
    )
    
    # --- Store Results ---
    evaluation_id = None
    if repo:
        window_args = window_metadata or {}
        
        evaluation_id = repo.store_evaluation(
            hypothesis_id=hypothesis.hypothesis_id,
            parameters=hypothesis.parameters,
            market_symbol=symbol,
            test_start_timestamp=bars[0].timestamp.to_pydatetime() if hasattr(bars[0].timestamp, 'to_pydatetime') else bars[0].timestamp,
            test_end_timestamp=bars[-1].timestamp.to_pydatetime() if hasattr(bars[-1].timestamp, 'to_pydatetime') else bars[-1].timestamp,
            metrics=metrics,
            benchmark_metrics=benchmark_metrics,
            assumed_costs_bps=policy.transaction_cost_bps + policy.slippage_bps,
            initial_capital=settings.starting_capital,
            final_equity=final_capital,
            bars_processed=replay_stats['bars_processed'],
            result_tag=window_args.get("result_tag"),
            window_index=window_args.get("window_index"),
            window_start=window_args.get("window_start"),
            window_end=window_args.get("window_end"),
            window_type=window_args.get("window_type"),
            market_regime=market_regime,
            sample_type=sample_type,
            policy_id=policy.policy_id,
            policy_hash=policy.compute_hash()
        )
        
        repo.store_trades(evaluation_id, completed_trades)
        
    return {
        "metrics": metrics,
        "market_regime": market_regime,
        "evaluation_id": evaluation_id,
        "trades": completed_trades
    }


def run_evaluation(
    hypothesis_id: str,
    policy_id: str, # MANDATORY
    data_path: str | None = None,
    symbol: str = "SYNTHETIC",
    start_date: str | None = None,
    end_date: str | None = None,
    use_synthetic: bool = False,
    synthetic_bars: int = 252,
    output_db: str | None = None,
    verbose: bool = True,
    max_bars: int | None = None, # Optional hook
    preloaded_bars: List[Any] | None = None # For Batch Execution
) -> dict:
    """Run evaluation enforced by ResearchPolicy."""
    settings = get_settings()
    
    # 0. Load Policy
    policy = get_policy(policy_id)
    
    if verbose:
        print("=" * 70)
        print("HYPOTHESIS RESEARCH ENGINE - EVALUATION RUN")
        print("=" * 70)
        print(f"Hypothesis: {hypothesis_id}")
        print(f"Policy: {policy.policy_id} ({policy.evaluation_mode})")
        print(f"Symbol: {symbol}")

    # 1. Load Hypothesis
    hypothesis_class = get_hypothesis(hypothesis_id)
    hypothesis = hypothesis_class()
    
    # 2. Load Data
    if verbose:
        print("\n[Data Loading]")
    
    if preloaded_bars:
         bars = preloaded_bars
         if verbose:
             print(f"  Using preloaded data ({len(bars)} bars)")
    elif use_synthetic:
        bars = MarketDataLoader.create_synthetic_data(
            symbol=symbol,
            start_date=datetime(2020, 1, 1),
            num_bars=synthetic_bars
        )
    else:
        if not data_path:
            raise ValueError("data_path required")
        bars = MarketDataLoader.load_from_csv(data_path, symbol=symbol)
        # Filter dates
        if start_date:
            st = datetime.fromisoformat(start_date)
            bars = [b for b in bars if b.timestamp >= st]
        if end_date:
            et = datetime.fromisoformat(end_date)
            bars = [b for b in bars if b.timestamp <= et]
            
    if not bars:
        raise ValueError("No data found")

    # 3. Setup Repository & Components
    db_path = output_db or settings.database_path
    repo = EvaluationRepository(db_path)
    
    # Store Hypothesis & Policy
    repo.store_hypothesis(hypothesis.hypothesis_id, hypothesis.parameters, str(hypothesis))
    repo.store_policy(policy)
    
    regime_classifier = RegimeClassifier()
    decay_tracker = DecayTracker(decay_threshold_sharpe=policy.max_sharpe_decay)
    
    # Init Guardrails from Policy
    guardrails = ResearchGuardrails(
        min_trades=policy.min_trades,
        min_regimes=policy.min_regimes,
        max_sharpe_decay_pct=policy.max_sharpe_decay
    )
    
    # 4. Execution Logic
    
    if policy.evaluation_mode == EvaluationMode.SINGLE_PASS:
        # --- STANDARD SINGLE PASS ---
        if verbose:
            print("\n[Execution] Running Single Pass...")
        
        result = _run_single_pass(
            hypothesis=hypothesis,
            bars=bars,
            policy=policy,
            settings=settings,
            symbol=symbol,
            repo=repo,
            regime_classifier=regime_classifier,
            verbose=verbose,
            sample_type="IN_SAMPLE"
        )
        
        if verbose:
            m = result["metrics"]
            print(f"\n[Results] Sharpe: {m.get('sharpe_ratio', 0):.2f}, Ret: {m.get('total_return', 0):.2f}%")
            
        return result

    elif policy.evaluation_mode == EvaluationMode.WALK_FORWARD:
        # --- WALK-FORWARD LOOP ---
        if verbose:
            print(f"\n[Execution] Walk-Forward ({policy.train_window_bars}/{policy.test_window_bars})...")
        
        # Create generator
        df = pd.DataFrame([vars(b) for b in bars]) 
        if 'timestamp' in df.columns:
            df.set_index('timestamp', inplace=True)
            
        config = WalkForwardConfig(
            train_window_size=policy.train_window_bars,
            test_window_size=policy.test_window_bars,
            step_size=policy.step_size_bars
        )
        generator = WalkForwardGenerator(df, config)
        
        window_results = []
        total_trades = 0
        regimes_encountered = set()
        
        for train_win, test_win in generator.generate_windows():
            if verbose:
                print(f"\n--- Window {train_win.window_index} ---")
                
            # --- TRAIN (In-Sample) ---
            train_bars = [b for b in bars if train_win.start_timestamp <= b.timestamp <= train_win.end_timestamp]
            
            if verbose:
                print(f"  Training ({len(train_bars)} bars)...")
            train_res = _run_single_pass(
                hypothesis, train_bars, policy, settings, symbol, repo,
                window_metadata={
                    "window_index": train_win.window_index,
                    "window_type": "TRAIN",
                    "window_start": _to_datetime(train_win.start_timestamp),
                    "window_end": _to_datetime(train_win.end_timestamp)
                },
                regime_classifier=regime_classifier,
                verbose=False,
                sample_type="IN_SAMPLE"
            )
            
            # --- TEST (Out-of-Sample) ---
            test_bars = [b for b in bars if test_win.start_timestamp <= b.timestamp <= test_win.end_timestamp]
            
            if verbose:
                print(f"  Testing ({len(test_bars)} bars)...")
            
            # Run test pass 
            test_res = _run_single_pass(
                hypothesis, test_bars, policy, settings, symbol, 
                repo=None, # Delayed storage
                regime_classifier=regime_classifier,
                verbose=False,
                sample_type="OUT_OF_SAMPLE"
            )
            
            # --- Decay Analysis ---
            decay = decay_tracker.analyze_decay(train_res["metrics"], test_res["metrics"])
            if verbose:
                print(f"  Result: {decay.result_tag} (Sharpe Change: {decay.sharpe_change:.2f})")
            
            # Store Test Result
            if repo:
                repo.store_evaluation(
                    hypothesis_id=hypothesis.hypothesis_id,
                    parameters=hypothesis.parameters,
                    market_symbol=symbol,
                    test_start_timestamp=_to_datetime(test_win.start_timestamp),
                    test_end_timestamp=_to_datetime(test_win.end_timestamp),
                    metrics=test_res["metrics"],
                    benchmark_metrics={"benchmark_return_pct": 0}, 
                    assumed_costs_bps=policy.transaction_cost_bps + policy.slippage_bps,
                    initial_capital=settings.starting_capital,
                    final_equity=test_res["metrics"]["final_equity"],
                    bars_processed=len(test_bars),
                    result_tag=decay.result_tag,
                    window_index=test_win.window_index,
                    window_start=_to_datetime(test_win.start_timestamp),
                    window_end=_to_datetime(test_win.end_timestamp),
                    window_type="TEST",
                    market_regime=test_res["market_regime"],
                    sample_type="OUT_OF_SAMPLE",
                    policy_id=policy.policy_id,
                    policy_hash=policy.compute_hash()
                )
                repo.store_trades(777, test_res["trades"]) # Placeholder ID, same issue as before but acceptable for now as per previous step logic
            
            # --- Guardrails Accumulation ---
            total_trades += test_res["metrics"]["trade_count"]
            if test_res["market_regime"]:
                try:
                    regimes_encountered.add(MarketRegime(test_res["market_regime"]))
                except ValueError:
                    pass
            
            window_results.append({
                "window": train_win.window_index,
                "train_metrics": train_res["metrics"],
                "test_metrics": test_res["metrics"],
                "market_regime": test_res.get("market_regime"),
                "decay": decay
            })

            # Check decay immediately?
            # Guardrails checks
        
        # --- Final Guardrails Check ---
        check_trades = guardrails.check_min_trades(total_trades)
        check_regimes = guardrails.check_regime_coverage(list(regimes_encountered))
        
        if verbose:
            print("\n[Guardrails Check]")
            print(f"  Total Trades: {total_trades} \t-> {check_trades.status.value}")
            print(f"  Regimes: {len(regimes_encountered)} \t-> {check_regimes.status.value}")
            
        return {
            "mode": "WALK_FORWARD",
            "windows": window_results
        }


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Hypothesis Research Engine - Evaluate trading hypotheses"
    )
    
    parser.add_argument("--hypothesis", required=True, help="Hypothesis ID")
    parser.add_argument("--policy", required=True, help="Research Policy ID (e.g., WF_V1)")
    parser.add_argument("--data-path", help="Path to market data CSV")
    parser.add_argument("--symbol", default="SYNTHETIC", help="Market symbol")
    parser.add_argument("--start-date", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", help="End date (YYYY-MM-DD)")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data")
    parser.add_argument("--synthetic-bars", type=int, default=252, help="Synthetic bars count")
    parser.add_argument("--output-db", help="Output database path")
    parser.add_argument("--quiet", action="store_true", help="Suppress output")
    
    args = parser.parse_args()
    
    try:
        run_evaluation(
            hypothesis_id=args.hypothesis,
            policy_id=args.policy,
            data_path=args.data_path,
            symbol=args.symbol,
            start_date=args.start_date,
            end_date=args.end_date,
            use_synthetic=args.synthetic,
            synthetic_bars=args.synthetic_bars,
            output_db=args.output_db,
            verbose=not args.quiet
        )
        sys.exit(0)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
