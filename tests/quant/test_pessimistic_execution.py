
import pytest
import pandas as pd
import numpy as np
from quant.config import ResearchConfig

def test_pessimistic_execution_logic():
    # Setup data
    # 5 bars. Horizon=2.
    # Bar 0: Enter. Exits at Bar 2.
    # Bar 1: Enter. Exits at Bar 3.
    
    # Prices
    closes = [1.0000, 1.0000, 1.0020, 1.0050, 1.0000]
    lows =   [0.9995, 0.9995, 0.9980, 0.9990, 0.9990]
    #         t=0     t=1     t=2     t=3     t=4
    
    df = pd.DataFrame({'close': closes, 'low': lows})
    h = 2
    sl_price = 0.0010 # 10 pips
    
    # Logic replication from walk_forward.py
    valid_len = len(df) - h # 5 - 2 = 3 (Indices 0, 1, 2)
    
    # 1. Price Moves Raw (Close[t+h] - Close[t])
    # t=0: Close[2] - Close[0] = 1.0020 - 1.0000 = +0.0020
    # t=1: Close[3] - Close[1] = 1.0050 - 1.0000 = +0.0050
    # t=2: Close[4] - Close[2] = 1.0000 - 1.0020 = -0.0020
    price_moves_raw = df["close"].shift(-h).values - df["close"].values
    price_moves = price_moves_raw[:valid_len].copy()
    
    assert np.isclose(price_moves[0], 0.0020)
    assert np.isclose(price_moves[1], 0.0050)
    
    # 2. Future Lows
    # rolling(2).min() at t=2 includes [t=1, t=2].
    # shift(-2) at t=0 takes min(1,2).
    future_lows = df["low"].rolling(window=h).min().shift(-h).values
    future_lows = future_lows[:valid_len]
    
    # t=0: min(Low[1], Low[2]) = min(0.9995, 0.9980) = 0.9980
    # t=1: min(Low[2], Low[3]) = min(0.9980, 0.9990) = 0.9980
    assert np.isclose(future_lows[0], 0.9980)
    
    # 3. Check SL
    entry_prices = df["close"].values[:valid_len]
    # t=0 Entry=1.0000. Stop=0.9990.
    # Low=0.9980 < 0.9990 -> HIT
    
    # t=1 Entry=1.0000. Stop=0.9990.
    # Low=0.9980 < 0.9990 -> HIT
    
    sl_hit_mask = future_lows < (entry_prices - sl_price)
    
    assert sl_hit_mask[0] == True
    assert sl_hit_mask[1] == True
    
    # 4. Apply PnL Cap
    price_moves[sl_hit_mask] = -sl_price
    
    assert np.isclose(price_moves[0], -0.0010) # Was +0.0020
    assert np.isclose(price_moves[1], -0.0010) # Was +0.0050
    
    print("Pessimistic Execution Test Passed!")

def test_pessimistic_execution_safe():
    # Safe scenario
    # Prices trend up, Lows must follow to avoid SL hit (SL relative to Entry)
    closes = [1.0000, 1.0000, 1.0020, 1.0050, 1.0000]
    # t=2 Entry=1.0020, SL=1.0010. Lows need to be > 1.0010.
    lows =   [0.9995, 0.9995, 0.9995, 1.0015, 1.0015] 
    df = pd.DataFrame({'close': closes, 'low': lows})
    h = 2
    sl_price = 0.0010
    
    valid_len = len(df) - h
    price_moves_raw = df["close"].shift(-h).values - df["close"].values
    price_moves = price_moves_raw[:valid_len].copy()
    
    future_lows = df["low"].rolling(window=h).min().shift(-h).values[:valid_len]
    entry_prices = df["close"].values[:valid_len]
    sl_hit_mask = future_lows < (entry_prices - sl_price)
    
    assert not sl_hit_mask.any()
    assert np.isclose(price_moves[0], 0.0020) # Unchanged
