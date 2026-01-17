"""
RSI extreme reversal strategy for competition mode.

Catches oversold bounces and overbought reversals in crypto markets.
Aggressive thresholds tuned for volatile crypto pairs.
"""

from typing import Any, Dict, List, Optional

import numpy as np

from clock.clock import Clock
from hypotheses.base import Hypothesis, TradeIntent, IntentType
from market.regime import MarketRegime
from state.market_state import MarketState
from state.position_state import PositionState


class RSIExtremeReversal(Hypothesis):
    """
    RSI-based reversal strategy for crypto extremes.

    Entry Conditions:
    - RSI reaches extreme oversold/overbought levels
    - Price shows reversal candle pattern (engulfing or hammer-like)
    
    Aggressive thresholds for crypto volatility.
    """

    def __init__(
        self,
        rsi_period: int = 7,
        oversold: float = 25.0,
        overbought: float = 75.0,
        min_reversal_ratio: float = 0.4,
    ):
        self.rsi_period = rsi_period
        self.oversold = oversold
        self.overbought = overbought
        self.min_reversal_ratio = min_reversal_ratio

    @property
    def hypothesis_id(self) -> str:
        return "rsi_extreme_reversal"

    @property
    def allowed_regimes(self) -> List[MarketRegime]:
        return []

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "rsi_period": self.rsi_period,
            "oversold": self.oversold,
            "overbought": self.overbought,
            "min_reversal_ratio": self.min_reversal_ratio,
        }

    def _calculate_rsi(self, closes: np.ndarray) -> float:
        """Calculate RSI from close prices."""
        deltas = np.diff(closes)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        avg_gain = gains[-self.rsi_period:].mean()
        avg_loss = losses[-self.rsi_period:].mean()

        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def on_bar(
        self,
        market_state: MarketState,
        position_state: PositionState,
        clock: Clock,
    ) -> Optional[TradeIntent]:
        required_bars = self.rsi_period + 5
        if market_state.bar_count() < required_bars:
            return TradeIntent(type=IntentType.HOLD)

        bars = market_state.recent_bars(required_bars)
        if bars is None or len(bars) < required_bars:
            return TradeIntent(type=IntentType.HOLD)

        closes = np.array([b.close for b in bars])
        rsi = self._calculate_rsi(closes)

        last = bars[-1]
        prev = bars[-2]

        candle_range = last.high - last.low
        if candle_range == 0:
            return TradeIntent(type=IntentType.HOLD)

        body = abs(last.close - last.open)
        body_ratio = body / candle_range

        lower_wick = min(last.open, last.close) - last.low
        upper_wick = last.high - max(last.open, last.close)

        bullish_reversal = (
            last.close > last.open and
            lower_wick > body * 0.5 and
            body_ratio > self.min_reversal_ratio
        )

        bearish_reversal = (
            last.close < last.open and
            upper_wick > body * 0.5 and
            body_ratio > self.min_reversal_ratio
        )

        bullish_engulfing = (
            prev.close < prev.open and
            last.close > last.open and
            last.close > prev.open and
            last.open < prev.close
        )

        bearish_engulfing = (
            prev.close > prev.open and
            last.close < last.open and
            last.close < prev.open and
            last.open > prev.close
        )

        if rsi < self.oversold and (bullish_reversal or bullish_engulfing):
            return TradeIntent(type=IntentType.BUY, size=1.0)

        if rsi > self.overbought and (bearish_reversal or bearish_engulfing):
            return TradeIntent(type=IntentType.SELL, size=1.0)

        if rsi < self.oversold - 5:
            return TradeIntent(type=IntentType.BUY, size=0.6)

        if rsi > self.overbought + 5:
            return TradeIntent(type=IntentType.SELL, size=0.6)

        return TradeIntent(type=IntentType.HOLD)
