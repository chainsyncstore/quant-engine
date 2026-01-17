"""
Volatility Compression Break Hypothesis

Purpose: Trade after long inactivity (squeeze breakouts).

Logic:
- Bollinger Band width (20) < rolling 10-day 20th percentile
- Wait for close outside bands
- Direction = breakout direction
- Confidence increases if volume spike present
"""

from typing import Dict, Any, Optional, List
import math

from hypotheses.base import Hypothesis, TradeIntent, IntentType
from state.market_state import MarketState
from state.position_state import PositionState
from clock.clock import Clock
from data.schemas import Bar


class VolatilityCompression(Hypothesis):
    """Trade breakouts from volatility compression (squeeze)."""
    
    def __init__(
        self,
        bb_period: int = 20,
        bb_std: float = 2.0,
        width_lookback: int = 50,
        width_percentile: float = 20.0,
        volume_spike_mult: float = 1.5,
        hold_bars: int = 8,
        **kwargs
    ):
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.width_lookback = width_lookback
        self.width_percentile = width_percentile
        self.volume_spike_mult = volume_spike_mult
        self.hold_bars = hold_bars
        self._bars_held = 0
        self._width_history: List[float] = []
    
    @property
    def hypothesis_id(self) -> str:
        return "volatility_compression"
    
    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "bb_period": self.bb_period,
            "bb_std": self.bb_std,
            "width_lookback": self.width_lookback,
            "width_percentile": self.width_percentile,
            "volume_spike_mult": self.volume_spike_mult,
            "hold_bars": self.hold_bars
        }
    
    def _compute_bollinger(self, closes: List[float]) -> tuple[float, float, float]:
        """Compute Bollinger Bands: (middle, upper, lower)."""
        if len(closes) < self.bb_period:
            return 0.0, 0.0, 0.0
        
        recent = closes[-self.bb_period:]
        sma = sum(recent) / len(recent)
        
        variance = sum((c - sma) ** 2 for c in recent) / len(recent)
        std = math.sqrt(variance)
        
        upper = sma + (self.bb_std * std)
        lower = sma - (self.bb_std * std)
        
        return sma, upper, lower
    
    def _compute_bb_width(self, upper: float, lower: float, middle: float) -> float:
        """Compute normalized Bollinger Band width."""
        if middle == 0:
            return 0.0
        return (upper - lower) / middle
    
    def _get_width_percentile_threshold(self) -> Optional[float]:
        """Get the threshold width at the configured percentile."""
        if len(self._width_history) < self.width_lookback:
            return None
        
        recent = sorted(self._width_history[-self.width_lookback:])
        idx = int(len(recent) * (self.width_percentile / 100.0))
        idx = max(0, min(idx, len(recent) - 1))
        
        return recent[idx]
    
    def _check_volume_spike(self, bars: List[Bar], current_volume: float) -> bool:
        """Check if current volume is a spike compared to recent average."""
        if not bars or current_volume <= 0:
            return False
        
        volumes = [b.volume for b in bars if b.volume > 0]
        if not volumes:
            return False
        
        avg_volume = sum(volumes) / len(volumes)
        if avg_volume == 0:
            return False
        
        return current_volume > avg_volume * self.volume_spike_mult
    
    def on_bar(
        self,
        market_state: MarketState,
        position_state: PositionState,
        clock: Clock
    ) -> Optional[TradeIntent]:
        
        min_history = max(self.bb_period, self.width_lookback)
        if market_state.bar_count() < min_history:
            return None
        
        bar = market_state.current_bar()
        history = market_state.get_bars(min_history)
        
        closes = [b.close for b in history]
        closes.append(bar.close)
        
        middle, upper, lower = self._compute_bollinger(closes)
        if middle == 0:
            return None
        
        current_width = self._compute_bb_width(upper, lower, middle)
        self._width_history.append(current_width)
        
        if len(self._width_history) > self.width_lookback * 2:
            self._width_history = self._width_history[-self.width_lookback:]
        
        if position_state.has_position:
            self._bars_held += 1
            
            if self._bars_held >= self.hold_bars:
                self._bars_held = 0
                return TradeIntent(type=IntentType.CLOSE, size=1.0)
            
            return None
        
        self._bars_held = 0
        
        width_threshold = self._get_width_percentile_threshold()
        if width_threshold is None:
            return None
        
        is_compressed = current_width < width_threshold
        if not is_compressed:
            return None
        
        has_volume_spike = self._check_volume_spike(history[-self.bb_period:], bar.volume)
        
        base_confidence = 0.55
        if has_volume_spike:
            base_confidence += 0.25
        
        if bar.close > upper:
            return TradeIntent(type=IntentType.BUY, size=min(1.0, base_confidence))
        
        if bar.close < lower:
            return TradeIntent(type=IntentType.SELL, size=min(1.0, base_confidence))
        
        return None
