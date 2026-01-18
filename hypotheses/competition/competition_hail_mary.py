"""
Ultra-aggressive competition hypothesis for maximum profit potential.

Combines volatility breakout + momentum + RSI extremes with lowered thresholds.
Uses full position sizing on every signal. No regime gating.

WARNING: This is a HIGH RISK strategy for competition use only.
"""

from typing import Any, Dict, List, Optional

import numpy as np

from clock.clock import Clock
from hypotheses.base import Hypothesis, TradeIntent, IntentType
from market.regime import MarketRegime
from state.market_state import MarketState
from state.position_state import PositionState


class CompetitionHailMary(Hypothesis):
    """
    Ultra-aggressive multi-signal hypothesis for competition mode.

    This combines three signal types:
    1. Volatility expansion (impulsive breakout candles)
    2. Momentum confirmation (EMA cross + ROC)
    3. RSI extremes (oversold/overbought reversals)

    All thresholds are lowered for maximum signal generation.
    Position size is ALWAYS 1.0 (full allocation).
    """

    def __init__(
        self,
        # Volatility parameters (aggressive)
        lookback: int = 14,
        atr_mult: float = 1.0,  # Lowered from 1.4
        min_body_ratio: float = 0.35,  # Lowered from 0.6
        # Momentum parameters (aggressive)
        fast_period: int = 5,
        slow_period: int = 13,
        roc_period: int = 3,
        roc_threshold: float = 0.001,  # Lowered from 0.002
        # RSI parameters (aggressive)
        rsi_period: int = 7,
        rsi_oversold: float = 20.0,  # Lowered from 25
        rsi_overbought: float = 80.0,  # Raised from 75
    ):
        self.lookback = lookback
        self.atr_mult = atr_mult
        self.min_body_ratio = min_body_ratio
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.roc_period = roc_period
        self.roc_threshold = roc_threshold
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought

    @property
    def hypothesis_id(self) -> str:
        return "competition_hail_mary"

    @property
    def allowed_regimes(self) -> List[MarketRegime]:
        # No regime gating - trade ALL conditions
        return []

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "lookback": self.lookback,
            "atr_mult": self.atr_mult,
            "min_body_ratio": self.min_body_ratio,
            "fast_period": self.fast_period,
            "slow_period": self.slow_period,
            "roc_period": self.roc_period,
            "roc_threshold": self.roc_threshold,
            "rsi_period": self.rsi_period,
            "rsi_oversold": self.rsi_oversold,
            "rsi_overbought": self.rsi_overbought,
        }

    def _ema(self, data: np.ndarray, period: int) -> np.ndarray:
        """Calculate EMA."""
        alpha = 2 / (period + 1)
        ema = np.zeros_like(data)
        ema[0] = data[0]
        for i in range(1, len(data)):
            ema[i] = alpha * data[i] + (1 - alpha) * ema[i - 1]
        return ema

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
        required_bars = max(self.lookback, self.slow_period) + 10
        if market_state.bar_count() < required_bars:
            return TradeIntent(type=IntentType.HOLD)

        bars = market_state.recent_bars(required_bars)
        if bars is None or len(bars) < required_bars:
            return TradeIntent(type=IntentType.HOLD)

        closes = np.array([b.close for b in bars])
        highs = np.array([b.high for b in bars])
        lows = np.array([b.low for b in bars])
        opens = np.array([b.open for b in bars])

        # Current bar
        last = bars[-1]
        prev = bars[-2]
        body = abs(last.close - last.open)
        candle_range = last.high - last.low

        if candle_range == 0:
            return TradeIntent(type=IntentType.HOLD)

        body_ratio = body / candle_range

        # === SIGNAL 1: Volatility Expansion ===
        trs = np.maximum(highs[1:], closes[:-1]) - np.minimum(lows[1:], closes[:-1])
        atr = trs[-self.lookback:].mean()

        expanding = candle_range > atr * self.atr_mult
        impulsive = body_ratio > self.min_body_ratio

        lookback_highs = highs[-self.lookback - 1: -1]
        lookback_lows = lows[-self.lookback - 1: -1]

        vol_bullish = expanding and impulsive and last.close > lookback_highs.max()
        vol_bearish = expanding and impulsive and last.close < lookback_lows.min()

        # === SIGNAL 2: Momentum (EMA Cross + ROC) ===
        fast_ema = self._ema(closes, self.fast_period)
        slow_ema = self._ema(closes, self.slow_period)

        curr_fast = fast_ema[-1]
        curr_slow = slow_ema[-1]
        prev_fast = fast_ema[-2]
        prev_slow = slow_ema[-2]

        bullish_cross = prev_fast <= prev_slow and curr_fast > curr_slow
        bearish_cross = prev_fast >= prev_slow and curr_fast < curr_slow

        roc = (closes[-1] - closes[-1 - self.roc_period]) / closes[-1 - self.roc_period]

        mom_bullish = (bullish_cross or curr_fast > curr_slow) and roc > self.roc_threshold
        mom_bearish = (bearish_cross or curr_fast < curr_slow) and roc < -self.roc_threshold

        # === SIGNAL 3: RSI Extremes ===
        rsi = self._calculate_rsi(closes)

        rsi_bullish = rsi < self.rsi_oversold
        rsi_bearish = rsi > self.rsi_overbought

        # === COMBINED SIGNAL LOGIC ===
        # Count bullish and bearish confirmations
        bullish_count = sum([vol_bullish, mom_bullish, rsi_bullish])
        bearish_count = sum([vol_bearish, mom_bearish, rsi_bearish])

        # Strong signal: 2+ confirmations OR volatility breakout alone
        if bullish_count >= 2 or vol_bullish:
            return TradeIntent(type=IntentType.BUY, size=1.0)

        if bearish_count >= 2 or vol_bearish:
            return TradeIntent(type=IntentType.SELL, size=1.0)

        # Moderate signal: momentum + direction confirmation
        if mom_bullish and last.close > last.open:
            return TradeIntent(type=IntentType.BUY, size=1.0)

        if mom_bearish and last.close < last.open:
            return TradeIntent(type=IntentType.SELL, size=1.0)

        # RSI extreme with reversal candle
        if rsi_bullish and last.close > last.open and last.close > prev.high:
            return TradeIntent(type=IntentType.BUY, size=1.0)

        if rsi_bearish and last.close < last.open and last.close < prev.low:
            return TradeIntent(type=IntentType.SELL, size=1.0)

        return TradeIntent(type=IntentType.HOLD)
