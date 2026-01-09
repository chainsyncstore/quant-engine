"""
Replay engine - main event loop.

Owns control flow and orchestrates the bar-by-bar replay process.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from clock.clock import Clock
from data.bar_iterator import BarIterator
from engine.decision_queue import DecisionQueue
from hypotheses.base import Hypothesis

if TYPE_CHECKING:
    from state.market_state import MarketState
    from state.position_state import PositionState


class ReplayEngine:
    """
    Main replay engine that drives the simulation.
    
    The engine:
    1. Iterates through market data bar-by-bar
    2. Updates the clock
    3. Updates market state
    4. Invokes the hypothesis
    5. Queues trade intents
    6. Triggers execution on delayed intents
    7. Terminates cleanly at end of data
    
    This is the only component that controls time progression.
    """
    
    def __init__(
        self,
        hypothesis: Hypothesis,
        bar_iterator: BarIterator,
        clock: Clock,
        decision_queue: DecisionQueue,
        market_state: Optional["MarketState"] = None,
        position_state: Optional["PositionState"] = None,
        execution_delay_bars: int = 1
    ):
        """
        Initialize replay engine.
        
        Args:
            hypothesis: Hypothesis to evaluate
            bar_iterator: Iterator over market bars
            clock: Clock instance (will be updated by engine)
            decision_queue: Decision queue for intent buffering
            market_state: Optional existing market state
            position_state: Optional existing position state
            execution_delay_bars: Bars to delay execution
        """
        from state.market_state import MarketState
        from state.position_state import PositionState

        self._hypothesis = hypothesis
        self._bar_iterator = bar_iterator
        self._clock = clock
        self._decision_queue = decision_queue
        self._execution_delay_bars = execution_delay_bars
        self._current_bar_index = 0
        
        # Use provided state or create new
        self._market_state = market_state if market_state is not None else MarketState()
        self._position_state = position_state if position_state is not None else PositionState()
    
    def run(
        self,
        on_bar_callback=None,
        on_decision_callback=None,
        on_execution_callback=None
    ) -> dict:
        """
        Run the replay simulation.
        
        Args:
            on_bar_callback: Optional callback(bar, bar_index) called on each bar
            on_decision_callback: Optional callback(intent, bar_index) for decisions
            on_execution_callback: Optional callback(decisions) for executions
            
        Returns:
            Dictionary with replay statistics
        """
        bars_processed = 0
        decisions_made = 0
        executions_triggered = 0
        
        # Main replay loop
        for bar in self._bar_iterator:
            # 1. Update clock
            self._clock.set_time(bar.timestamp)
            
            # 2. Update market state
            self._market_state.update(bar)
            
            # 3. Check for executable decisions (no internal advance needed)
            # SAFETY CRITICAL: Execution must happen BEFORE hypothesis invocation
            # to ensure the hypothesis sees the correct post-execution state
            # and to prevent look-ahead bias or state contamination.
            executable_decisions = self._decision_queue.get_executable_decisions(self._current_bar_index)
            
            if executable_decisions and on_execution_callback:
                on_execution_callback(
                    executable_decisions,
                    bar,
                    self._current_bar_index,
                    self._market_state,
                    self._position_state
                )
                executions_triggered += len(executable_decisions)
            
            # 5. Invoke hypothesis (only if we have enough history)
            # Give the hypothesis a chance to observe state and make decisions
            intent = self._hypothesis.on_bar(
                market_state=self._market_state,
                position_state=self._position_state,
                clock=self._clock
            )
            
            # 6. Queue any new decisions
            if intent is not None and not intent.is_hold():
                self._decision_queue.enqueue(
                    intent=intent,
                    decision_timestamp=bar.timestamp,
                    decision_bar_index=self._current_bar_index
                )
                decisions_made += 1
                
                if on_decision_callback:
                    on_decision_callback(intent, self._current_bar_index)
            
            # 7. Call bar callback if provided
            if on_bar_callback:
                on_bar_callback(bar, self._current_bar_index)
            
            bars_processed += 1
            self._current_bar_index += 1
        
        # Process any remaining queued decisions at the end
        # (These won't actually execute since there are no more bars,
        # but we account for them)
        pending_decisions = self._decision_queue.has_pending_decisions()
        
        return {
            "bars_processed": bars_processed,
            "decisions_made": decisions_made,
            "executions_triggered": executions_triggered,
            "pending_decisions_at_end": pending_decisions,
            "hypothesis_id": self._hypothesis.hypothesis_id,
            "start_time": self._bar_iterator._bars[0].timestamp if bars_processed > 0 else None,
            "end_time": self._clock.now() if bars_processed > 0 else None
        }
    
    def get_current_bar_index(self) -> int:
        """
        Get the current bar index.
        
        Returns:
            Current bar index (0-based)
        """
        return self._current_bar_index
