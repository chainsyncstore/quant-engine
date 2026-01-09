"""
Market Regime Detection

Classifies market state into regimes (BULL, BEAR, CHOPPY, NEUTRAL).
Used for gating strategies (e.g., preventing Counter-Trend from trading in strong trends).
"""

from enum import Enum
import pandas as pd
import numpy as np
from state.market_state import MarketState

class MarketRegime(str, Enum):
    BULL = "BULL"
    BEAR = "BEAR"
    CHOPPY = "CHOPPY"
    NEUTRAL = "NEUTRAL"
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
        # Need at least 200 bars for SMA200
        if market_state.bar_count() < 200:
            return MarketRegime.UNKNOWN
            
        df = market_state.to_dataframe(n=300) # Get ample history
        if df.empty:
            return MarketRegime.UNKNOWN
            
        # Calculate Indicators
        try:
            # 1. SMAs
            df['sma50'] = df['close'].rolling(window=50).mean()
            df['sma200'] = df['close'].rolling(window=200).mean()
            
            # 2. ADX (Simplified TR/DX/ADX)
            # Need High, Low, Close
            # TR = Max(H-L, Abs(H-Cp), Abs(L-Cp))
            # ... Implementing full ADX is verbose. 
            # I'll use a simplified Trend Strength Estimator:
            # Ratio of (Abs(Close - Open)) to (High - Low)? No.
            # Efficiency Ratio? 
            # ADX is standard. I'll implement a helper.
            
            # Use 'ta' library if available? 
            # "The user's OS version is windows." - Check dependencies.
            # I will implement manually to avoid dependency issues.
            
            adx = self._calculate_adx(df)
            current_adx = adx.iloc[-1]
            
            current_price = df['close'].iloc[-1]
            sma50 = df['sma50'].iloc[-1]
            sma200 = df['sma200'].iloc[-1]
            
            # Classification Logic
            
            # CHOPPY: Low Trend Strength
            if current_adx < self.low_vol_threshold:
                return MarketRegime.CHOPPY
                
            # TRENDING: High Trend Strength
            if current_adx > self.high_vol_threshold:
                if current_price > sma200 and sma50 > sma200:
                    return MarketRegime.BULL
                if current_price < sma200 and sma50 < sma200:
                    return MarketRegime.BEAR
                    
            return MarketRegime.NEUTRAL
            
        except Exception:
            # Fallback
            return MarketRegime.UNKNOWN

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
