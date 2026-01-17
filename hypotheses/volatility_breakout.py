"""
Volatility Expansion Breakout Hypothesis (M5)

Purpose: Trade when momentum starts, not after it's obvious.

Logic:
- Compute rolling ATR(14)
- Detect candle range > 1.5× ATR
- Direction = candle close vs open
- Confidence proportional to range / ATR
"""

from typing import Dict, Any, Optional

from hypotheses.base import Hypothesis, TradeIntent, IntentType
from state.market_state import MarketState
from state.position_state import PositionState
from clock.clock import Clock


class VolatilityExpansionBreakout(Hypothesis):
    """Trade volatility expansion with directional bias."""
    
    def __init__(
        self,
        atr_period: int = 14,
        breakout_mult: float = 1.5,
        hold_bars: int = 6,
        **kwargs
    ):
        self.atr_period = atr_period
        self.breakout_mult = breakout_mult
        self.hold_bars = hold_bars
        self._bars_held = 0
        self._position_direction: Optional[str] = None
    
    @property
    def hypothesis_id(self) -> str:
        return "volatility_expansion_breakout"
    
    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "atr_period": self.atr_period,
            "breakout_mult": self.breakout_mult,
            "hold_bars": self.hold_bars
        }
    
    def _compute_atr(self, market_state: MarketState) -> Optional[float]:
        """Compute ATR over the configured period."""
        history = market_state.get_bars(self.atr_period)
        if len(history) < self.atr_period:
            return None
        
        true_ranges = []
        for i, bar in enumerate(history):
            if i == 0:
                tr = bar.high - bar.low
            else:
                prev_close = history[i - 1].close
                tr = max(
                    bar.high - bar.low,
                    abs(bar.high - prev_close),
                    abs(bar.low - prev_close)
                )
            true_ranges.append(tr)
        
        return sum(true_ranges) / len(true_ranges)
    
    def on_bar(
        self,
        market_state: MarketState,
        position_state: PositionState,
        clock: Clock
    ) -> Optional[TradeIntent]:
        
        if market_state.bar_count() < self.atr_period:
            return None
        
        bar = market_state.current_bar()
        
        if position_state.has_position:
            self._bars_held += 1
            
            if self._bars_held >= self.hold_bars:
                self._bars_held = 0
                self._position_direction = None
                return TradeIntent(type=IntentType.CLOSE, size=1.0)
            
            return None
        
        self._bars_held = 0
        self._position_direction = None
        
        atr = self._compute_atr(market_state)
        if atr is None or atr <= 0:
            return None
        
        candle_range = bar.high - bar.low
        
        if candle_range > atr * self.breakout_mult:
            ratio = candle_range / atr
            confidence = min(1.0, (ratio - self.breakout_mult) / self.breakout_mult + 0.5)
            
            if bar.close > bar.open:
                self._position_direction = "long"
                return TradeIntent(type=IntentType.BUY, size=confidence)
            elif bar.close < bar.open:
                self._position_direction = "short"
                return TradeIntent(type=IntentType.SELL, size=confidence)
        
        return None
