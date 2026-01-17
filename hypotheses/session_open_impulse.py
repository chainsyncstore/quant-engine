"""
Session Open Impulse Hypothesis (London / NY)

Purpose: Exploit time-based structural volatility.

Logic:
- Time filter: first 30 minutes of London or NY session
- If first M5 candle closes outside prior session range:
  - Trade in breakout direction
- Confidence higher if aligned with higher-TF trend
"""

from typing import Dict, Any, Optional, List
from datetime import datetime, time

from hypotheses.base import Hypothesis, TradeIntent, IntentType
from state.market_state import MarketState
from state.position_state import PositionState
from clock.clock import Clock
from data.schemas import Bar


class SessionOpenImpulse(Hypothesis):
    """Trade session open breakouts for London and NY."""
    
    LONDON_OPEN = time(8, 0)
    LONDON_WINDOW_END = time(8, 30)
    NY_OPEN = time(13, 0)
    NY_WINDOW_END = time(13, 30)
    
    def __init__(
        self,
        lookback_bars: int = 24,
        trend_lookback: int = 48,
        hold_bars: int = 12,
        **kwargs
    ):
        self.lookback_bars = lookback_bars
        self.trend_lookback = trend_lookback
        self.hold_bars = hold_bars
        self._bars_held = 0
        self._traded_today_london = False
        self._traded_today_ny = False
        self._last_date: Optional[datetime] = None
    
    @property
    def hypothesis_id(self) -> str:
        return "session_open_impulse"
    
    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "lookback_bars": self.lookback_bars,
            "trend_lookback": self.trend_lookback,
            "hold_bars": self.hold_bars
        }
    
    def _is_in_london_window(self, ts: datetime) -> bool:
        """Check if timestamp is in London open window (08:00-08:30 UTC)."""
        t = ts.time()
        return self.LONDON_OPEN <= t < self.LONDON_WINDOW_END
    
    def _is_in_ny_window(self, ts: datetime) -> bool:
        """Check if timestamp is in NY open window (13:00-13:30 UTC)."""
        t = ts.time()
        return self.NY_OPEN <= t < self.NY_WINDOW_END
    
    def _get_prior_session_range(self, bars: List[Bar]) -> tuple[float, float]:
        """Get high/low of prior session bars."""
        if not bars:
            return 0.0, float('inf')
        
        high = max(b.high for b in bars)
        low = min(b.low for b in bars)
        return high, low
    
    def _get_higher_tf_trend(self, bars: List[Bar]) -> int:
        """Determine higher timeframe trend: 1=up, -1=down, 0=neutral."""
        if len(bars) < 10:
            return 0
        
        first_half = bars[:len(bars)//2]
        second_half = bars[len(bars)//2:]
        
        first_avg = sum(b.close for b in first_half) / len(first_half)
        second_avg = sum(b.close for b in second_half) / len(second_half)
        
        if second_avg > first_avg * 1.001:
            return 1
        elif second_avg < first_avg * 0.999:
            return -1
        return 0
    
    def on_bar(
        self,
        market_state: MarketState,
        position_state: PositionState,
        clock: Clock
    ) -> Optional[TradeIntent]:
        
        if market_state.bar_count() < self.lookback_bars:
            return None
        
        bar = market_state.current_bar()
        ts = bar.timestamp
        
        current_date = ts.date()
        if self._last_date != current_date:
            self._traded_today_london = False
            self._traded_today_ny = False
            self._last_date = current_date
        
        if position_state.has_position:
            self._bars_held += 1
            
            if self._bars_held >= self.hold_bars:
                self._bars_held = 0
                return TradeIntent(type=IntentType.CLOSE, size=1.0)
            
            return None
        
        self._bars_held = 0
        
        in_london = self._is_in_london_window(ts)
        in_ny = self._is_in_ny_window(ts)
        
        if in_london and self._traded_today_london:
            return None
        if in_ny and self._traded_today_ny:
            return None
        
        if not in_london and not in_ny:
            return None
        
        history = market_state.get_bars(self.lookback_bars)
        prior_high, prior_low = self._get_prior_session_range(history)
        
        trend_history = market_state.get_bars(self.trend_lookback)
        trend = self._get_higher_tf_trend(trend_history)
        
        base_confidence = 0.6
        
        if bar.close > prior_high:
            if in_london:
                self._traded_today_london = True
            else:
                self._traded_today_ny = True
            
            confidence = base_confidence + (0.2 if trend == 1 else 0.0)
            return TradeIntent(type=IntentType.BUY, size=min(1.0, confidence))
        
        if bar.close < prior_low:
            if in_london:
                self._traded_today_london = True
            else:
                self._traded_today_ny = True
            
            confidence = base_confidence + (0.2 if trend == -1 else 0.0)
            return TradeIntent(type=IntentType.SELL, size=min(1.0, confidence))
        
        return None
