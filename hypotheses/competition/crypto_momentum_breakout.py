"""
Aggressive crypto momentum breakout strategy for competition mode.

Detects strong directional moves using EMA crossover and momentum confirmation.
Designed for high-volatility crypto pairs with frequent signals.
"""

from typing import Any, Dict, List, Optional

import numpy as np

from clock.clock import Clock
from hypotheses.base import Hypothesis, TradeIntent, IntentType
from market.regime import MarketRegime
from state.market_state import MarketState
from state.position_state import PositionState


class CryptoMomentumBreakout(Hypothesis):
    """
    Aggressive momentum strategy optimized for crypto volatility.

    Entry Conditions:
    - Fast EMA crosses above/below slow EMA
    - Price momentum (rate of change) confirms direction
    - Current bar closes in direction of signal
    
    Designed to catch strong directional moves in BTC/ETH.
    """

    def __init__(
        self,
        fast_period: int = 5,
        slow_period: int = 13,
        roc_period: int = 3,
        roc_threshold: float = 0.002,
    ):
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.roc_period = roc_period
        self.roc_threshold = roc_threshold

    @property
    def hypothesis_id(self) -> str:
        return "crypto_momentum_breakout"

    @property
    def allowed_regimes(self) -> List[MarketRegime]:
        return []

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "fast_period": self.fast_period,
            "slow_period": self.slow_period,
            "roc_period": self.roc_period,
            "roc_threshold": self.roc_threshold,
        }

    def _ema(self, data: np.ndarray, period: int) -> np.ndarray:
        """Calculate EMA."""
        alpha = 2 / (period + 1)
        ema = np.zeros_like(data)
        ema[0] = data[0]
        for i in range(1, len(data)):
            ema[i] = alpha * data[i] + (1 - alpha) * ema[i - 1]
        return ema

    def on_bar(
        self,
        market_state: MarketState,
        position_state: PositionState,
        clock: Clock,
    ) -> Optional[TradeIntent]:
        required_bars = self.slow_period + 5
        if market_state.bar_count() < required_bars:
            return TradeIntent(type=IntentType.HOLD)

        bars = market_state.recent_bars(required_bars)
        if bars is None or len(bars) < required_bars:
            return TradeIntent(type=IntentType.HOLD)

        closes = np.array([b.close for b in bars])
        
        fast_ema = self._ema(closes, self.fast_period)
        slow_ema = self._ema(closes, self.slow_period)

        prev_fast = fast_ema[-2]
        prev_slow = slow_ema[-2]
        curr_fast = fast_ema[-1]
        curr_slow = slow_ema[-1]

        bullish_cross = prev_fast <= prev_slow and curr_fast > curr_slow
        bearish_cross = prev_fast >= prev_slow and curr_fast < curr_slow

        roc = (closes[-1] - closes[-1 - self.roc_period]) / closes[-1 - self.roc_period]

        last = bars[-1]
        bullish_bar = last.close > last.open
        bearish_bar = last.close < last.open

        if bullish_cross and roc > self.roc_threshold and bullish_bar:
            return TradeIntent(type=IntentType.BUY, size=1.0)

        if bearish_cross and roc < -self.roc_threshold and bearish_bar:
            return TradeIntent(type=IntentType.SELL, size=1.0)

        if curr_fast > curr_slow and roc > self.roc_threshold * 2:
            return TradeIntent(type=IntentType.BUY, size=0.7)

        if curr_fast < curr_slow and roc < -self.roc_threshold * 2:
            return TradeIntent(type=IntentType.SELL, size=0.7)

        return TradeIntent(type=IntentType.HOLD)
