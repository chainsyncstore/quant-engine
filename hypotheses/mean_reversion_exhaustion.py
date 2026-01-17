"""
Mean Reversion After Exhaustion Hypothesis

Purpose: Catch snapbacks that momentum systems ignore.

Logic:
- Price deviates > 2× ATR from VWAP(20)
- RSI(14) > 75 → short bias
- RSI(14) < 25 → long bias
- Confidence decays with time since deviation
"""

from typing import Dict, Any, Optional, List

from hypotheses.base import Hypothesis, TradeIntent, IntentType
from state.market_state import MarketState
from state.position_state import PositionState
from clock.clock import Clock
from data.schemas import Bar


class MeanReversionExhaustion(Hypothesis):
    """Trade mean reversion after extreme exhaustion moves."""
    
    def __init__(
        self,
        vwap_period: int = 20,
        atr_period: int = 14,
        rsi_period: int = 14,
        atr_deviation_mult: float = 2.0,
        rsi_overbought: float = 75.0,
        rsi_oversold: float = 25.0,
        max_hold_bars: int = 10,
        **kwargs
    ):
        self.vwap_period = vwap_period
        self.atr_period = atr_period
        self.rsi_period = rsi_period
        self.atr_deviation_mult = atr_deviation_mult
        self.rsi_overbought = rsi_overbought
        self.rsi_oversold = rsi_oversold
        self.max_hold_bars = max_hold_bars
        self._bars_held = 0
        self._bars_since_signal = 0
    
    @property
    def hypothesis_id(self) -> str:
        return "mean_reversion_exhaustion"
    
    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "vwap_period": self.vwap_period,
            "atr_period": self.atr_period,
            "rsi_period": self.rsi_period,
            "atr_deviation_mult": self.atr_deviation_mult,
            "rsi_overbought": self.rsi_overbought,
            "rsi_oversold": self.rsi_oversold,
            "max_hold_bars": self.max_hold_bars
        }
    
    def _compute_vwap(self, bars: List[Bar]) -> Optional[float]:
        """Compute VWAP over given bars."""
        if not bars:
            return None
        
        total_tp_vol = 0.0
        total_vol = 0.0
        
        for bar in bars:
            typical_price = (bar.high + bar.low + bar.close) / 3
            vol = bar.volume if bar.volume > 0 else 1.0
            total_tp_vol += typical_price * vol
            total_vol += vol
        
        if total_vol == 0:
            return None
        
        return total_tp_vol / total_vol
    
    def _compute_atr(self, bars: List[Bar]) -> Optional[float]:
        """Compute ATR over given bars."""
        if len(bars) < 2:
            return None
        
        true_ranges = []
        for i in range(1, len(bars)):
            bar = bars[i]
            prev_close = bars[i - 1].close
            tr = max(
                bar.high - bar.low,
                abs(bar.high - prev_close),
                abs(bar.low - prev_close)
            )
            true_ranges.append(tr)
        
        if not true_ranges:
            return None
        
        return sum(true_ranges) / len(true_ranges)
    
    def _compute_rsi(self, closes: List[float]) -> Optional[float]:
        """Compute RSI over given closes."""
        if len(closes) < self.rsi_period + 1:
            return None
        
        gains = []
        losses = []
        
        for i in range(1, len(closes)):
            change = closes[i] - closes[i - 1]
            if change > 0:
                gains.append(change)
                losses.append(0.0)
            else:
                gains.append(0.0)
                losses.append(abs(change))
        
        if len(gains) < self.rsi_period:
            return None
        
        avg_gain = sum(gains[-self.rsi_period:]) / self.rsi_period
        avg_loss = sum(losses[-self.rsi_period:]) / self.rsi_period
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def on_bar(
        self,
        market_state: MarketState,
        position_state: PositionState,
        clock: Clock
    ) -> Optional[TradeIntent]:
        
        min_history = max(self.vwap_period, self.atr_period, self.rsi_period + 1)
        if market_state.bar_count() < min_history:
            return None
        
        bar = market_state.current_bar()
        
        if position_state.has_position:
            self._bars_held += 1
            
            if self._bars_held >= self.max_hold_bars:
                self._bars_held = 0
                self._bars_since_signal = 0
                return TradeIntent(type=IntentType.CLOSE, size=1.0)
            
            return None
        
        self._bars_held = 0
        
        history = market_state.get_bars(min_history)
        
        vwap = self._compute_vwap(history[-self.vwap_period:])
        if vwap is None:
            return None
        
        atr = self._compute_atr(history[-self.atr_period:])
        if atr is None or atr <= 0:
            return None
        
        closes = [b.close for b in history]
        closes.append(bar.close)
        rsi = self._compute_rsi(closes)
        if rsi is None:
            return None
        
        current_price = bar.close
        deviation = abs(current_price - vwap)
        deviation_in_atr = deviation / atr
        
        if deviation_in_atr < self.atr_deviation_mult:
            self._bars_since_signal = 0
            return None
        
        base_confidence = min(1.0, (deviation_in_atr - self.atr_deviation_mult) / self.atr_deviation_mult + 0.5)
        decay_factor = max(0.3, 1.0 - (self._bars_since_signal * 0.1))
        confidence = base_confidence * decay_factor
        
        self._bars_since_signal += 1
        
        if rsi > self.rsi_overbought and current_price > vwap:
            return TradeIntent(type=IntentType.SELL, size=confidence)
        
        if rsi < self.rsi_oversold and current_price < vwap:
            return TradeIntent(type=IntentType.BUY, size=confidence)
        
        return None
