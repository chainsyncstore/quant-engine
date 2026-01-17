"""
Market Regime Detection

Classifies market state into regimes (BULL, BEAR, CHOPPY, NEUTRAL).
Used for gating strategies (e.g., preventing Counter-Trend from trading in strong trends).
"""

from enum import Enum
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from state.market_state import MarketState


class MarketRegime(str, Enum):
    BULL = "BULL"
    BEAR = "BEAR"
    CHOPPY = "CHOPPY"
    NEUTRAL = "NEUTRAL"
    UNKNOWN = "UNKNOWN"


class RegimeConfidence(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    UNKNOWN = "UNKNOWN"

class RegimeClassifier:
    """
    Classifies market regime based on technical indicators.
    
    Logic:
    - BULL: Price > SMA200 AND SMA50 > SMA200 AND Adx > 25
    - BEAR: Price < SMA200 AND SMA50 < SMA200 AND Adx > 25
    - CHOPPY: ADX < 20
    - NEUTRAL: Anything else
    """
    
    def __init__(self, high_vol_threshold: float = 25.0, low_vol_threshold: float = 20.0):
        self.high_vol_threshold = high_vol_threshold
        self.low_vol_threshold = low_vol_threshold
        
    def classify(self, market_state: MarketState) -> MarketRegime:
        """
        Classify the current market regime.
        """
        regime, _ = self.classify_with_confidence(market_state)
        return regime

    def classify_with_confidence(
        self, market_state: MarketState
    ) -> Tuple[MarketRegime, RegimeConfidence]:
        """
        Classify the current market regime and return a coarse confidence label.
        """
        try:
            features = self._prepare_features(market_state)
        except Exception:
            return MarketRegime.UNKNOWN, RegimeConfidence.LOW

        if features is None:
            return MarketRegime.UNKNOWN, RegimeConfidence.LOW

        regime = self._determine_regime(features)
        confidence = self._determine_confidence(features, regime)
        return regime, confidence

    def _prepare_features(self, market_state: MarketState) -> Optional[dict]:
        if market_state.bar_count() < 200:
            return None

        df = market_state.to_dataframe(n=300)
        if df.empty:
            return None

        df["sma50"] = df["close"].rolling(window=50).mean()
        df["sma200"] = df["close"].rolling(window=200).mean()
        adx = self._calculate_adx(df)
        if adx.empty:
            return None

        return {
            "adx": float(adx.iloc[-1]),
            "price": float(df["close"].iloc[-1]),
            "sma50": float(df["sma50"].iloc[-1]),
            "sma200": float(df["sma200"].iloc[-1]),
        }

    def _determine_regime(self, features: dict) -> MarketRegime:
        current_adx = features["adx"]
        current_price = features["price"]
        sma50 = features["sma50"]
        sma200 = features["sma200"]

        if np.isnan(current_adx):
            return MarketRegime.UNKNOWN

        if current_adx < self.low_vol_threshold:
            return MarketRegime.CHOPPY

        if current_adx > self.high_vol_threshold:
            if current_price > sma200 and sma50 > sma200:
                return MarketRegime.BULL
            if current_price < sma200 and sma50 < sma200:
                return MarketRegime.BEAR

        return MarketRegime.NEUTRAL

    def _determine_confidence(
        self, features: dict, regime: MarketRegime
    ) -> RegimeConfidence:
        adx = features["adx"]

        if np.isnan(adx):
            return RegimeConfidence.LOW

        if regime == MarketRegime.CHOPPY:
            return RegimeConfidence.LOW

        if adx >= self.high_vol_threshold:
            return RegimeConfidence.HIGH

        if adx >= self.low_vol_threshold:
            return RegimeConfidence.MEDIUM

        return RegimeConfidence.LOW

    def _calculate_adx(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate ADX manually."""
        # True Range
        df['h-l'] = df['high'] - df['low']
        df['h-pc'] = abs(df['high'] - df['close'].shift(1))
        df['l-pc'] = abs(df['low'] - df['close'].shift(1))
        df['tr'] = df[['h-l', 'h-pc', 'l-pc']].max(axis=1)
        
        # DM
        df['dm_plus'] = np.where((df['high'] - df['high'].shift(1)) > (df['low'].shift(1) - df['low']), 
                                np.maximum(df['high'] - df['high'].shift(1), 0), 0)
        df['dm_minus'] = np.where((df['low'].shift(1) - df['low']) > (df['high'] - df['high'].shift(1)), 
                                 np.maximum(df['low'].shift(1) - df['low'], 0), 0)
        
        # Smoothed
        # Wilder's Smoothing (alpha = 1/n)
        # Using EMA as approximation for robustness/speed? 
        # Or rolling mean (SMA) for simplicity? 
        # Standard use is Wilder which is approx EMA(2*n-1). 
        # I'll use simple rolling mean for stability in research unless precise ADX needed.
        # Actually, let's use rolling sum for TR and DM to compute DI.
        
        tr_smooth = df['tr'].rolling(window=period).sum()
        dm_plus_smooth = df['dm_plus'].rolling(window=period).sum()
        dm_minus_smooth = df['dm_minus'].rolling(window=period).sum()
        
        di_plus = 100 * (dm_plus_smooth / tr_smooth)
        di_minus = 100 * (dm_minus_smooth / tr_smooth)
        
        dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus)
        adx = dx.rolling(window=period).mean()
        
        return adx
