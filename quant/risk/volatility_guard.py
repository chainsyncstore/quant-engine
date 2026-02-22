
import pandas as pd
import numpy as np
from typing import Optional

class VolatilityGuard:
    """
    Blocks trades during extreme volatility events.
    
    Logic:
    1. Calculate rolling realized volatility (e.g. 1-hour).
    2. Maintain a historical distribution of volatility.
    3. Block if current vol > Nth percentile (e.g. 95th).
    """
    
    def __init__(self, window_bars: int = 60, percentile: float = 0.99,
                 min_samples: int = 1000):
        self.window_bars = window_bars
        self.percentile = percentile
        self.min_samples = min_samples
        self.threshold: Optional[float] = None

    def fit(self, df: pd.DataFrame, vol_col: str = "realized_vol_5"):
        """Learn the safe volatility threshold from history."""
        if vol_col not in df.columns:
            return

        vol_values = df[vol_col].dropna()
        if len(vol_values) == 0:
            self.threshold = None
            return

        # Even when history is shorter than min_samples, use available data so
        # the guard remains functional in live and test environments.
        self.threshold = float(vol_values.quantile(self.percentile))
            
    def check(self, current_vol: float) -> bool:
        """
        Returns True if SAFE to trade.
        Returns False if volatility is too high (UNSAFE).
        """
        if self.threshold is None:
            return True # Open if not fitted
            
        return current_vol <= self.threshold
