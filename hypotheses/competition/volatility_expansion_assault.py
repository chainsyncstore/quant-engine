"""
Aggressive volatility expansion strategy for competition mode.

Detects impulsive breakout candles and enters in the direction of expansion.
"""

from typing import Any, Dict, List, Optional

import numpy as np

from clock.clock import Clock
from hypotheses.base import Hypothesis, TradeIntent, IntentType
from market.regime import MarketRegime
from state.market_state import MarketState
from state.position_state import PositionState


class VolatilityExpansionAssault(Hypothesis):
    """
    Aggressive breakout strategy that triggers on volatility expansion.

    Entry Conditions:
    - Current candle range exceeds ATR * atr_mult (expansion)
    - Candle body ratio > min_body_ratio (impulsive, not indecisive)
    - Close breaks above/below recent lookback highs/lows
    """

    def __init__(
        self,
        lookback: int = 20,
        atr_mult: float = 1.4,
        min_body_ratio: float = 0.6,
    ):
        self.lookback = lookback
        self.atr_mult = atr_mult
        self.min_body_ratio = min_body_ratio

    @property
    def hypothesis_id(self) -> str:
        return "volatility_expansion_assault"

    @property
    def allowed_regimes(self) -> List[MarketRegime]:
        return []

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "lookback": self.lookback,
            "atr_mult": self.atr_mult,
            "min_body_ratio": self.min_body_ratio,
        }

    def on_bar(
        self,
        market_state: MarketState,
        position_state: PositionState,
        clock: Clock,
    ) -> Optional[TradeIntent]:
        required_bars = self.lookback + 5
        if market_state.bar_count() < required_bars:
            return TradeIntent(type=IntentType.HOLD)

        bars = market_state.recent_bars(required_bars)
        if bars is None or len(bars) < self.lookback + 2:
            return TradeIntent(type=IntentType.HOLD)

        closes = np.array([b.close for b in bars])
        highs = np.array([b.high for b in bars])
        lows = np.array([b.low for b in bars])
        opens = np.array([b.open for b in bars])

        trs = np.maximum(highs[1:], closes[:-1]) - np.minimum(lows[1:], closes[:-1])
        atr = trs[-self.lookback :].mean()

        last = bars[-1]
        body = abs(last.close - last.open)
        candle_range = last.high - last.low

        if candle_range == 0:
            return TradeIntent(type=IntentType.HOLD)

        body_ratio = body / candle_range

        expanding = candle_range > atr * self.atr_mult
        impulsive = body_ratio > self.min_body_ratio

        if not (expanding and impulsive):
            return TradeIntent(type=IntentType.HOLD)

        lookback_highs = highs[-self.lookback - 1 : -1]
        lookback_lows = lows[-self.lookback - 1 : -1]

        if last.close > lookback_highs.max():
            return TradeIntent(type=IntentType.BUY, size=1.0)

        if last.close < lookback_lows.min():
            return TradeIntent(type=IntentType.SELL, size=1.0)

        return TradeIntent(type=IntentType.HOLD)
