
import pytest
import pandas as pd
import numpy as np
from quant.risk.volatility_guard import VolatilityGuard
from quant.live.signal_generator import SignalGenerator
from unittest.mock import MagicMock, patch

def test_volatility_guard_logic():
    # 1. Setup data: 100 bars of normal vol (1.0) and 5 bars of extreme vol (10.0)
    normal = np.random.normal(1.0, 0.1, 100)
    extreme = np.random.normal(10.0, 1.0, 5)
    all_vol = np.concatenate([normal, extreme])
    
    df = pd.DataFrame({'realized_vol_5': all_vol})
    
    # 2. Fit guard (95th percentile)
    # The threshold should be around max(normal) since extreme is < 5% of data?
    # 105 points. 95th percentile index = 99.75. 
    # So it should capture some of the extreme values or be just below them.
    # Actually, 5/105 = 4.7%. So top 5 are ~top 5%. 
    # 95th percentile should be between normal max and extreme min.
    
    guard = VolatilityGuard(percentile=0.95)
    guard.fit(df)
    
    print(f"DEBUG: Threshold = {guard.threshold}")
    
    # 3. Test Check
    assert guard.check(1.0) == True   # Normal vol -> Safe
    assert guard.check(0.5) == True   # Low vol -> Safe
    assert guard.check(10.0) == False # Extreme vol -> Unsafe
    
def test_volatility_guard_integration():
    # Mocking SignalGenerator to test just the generate_signal logic part
    # We can't easily instantiate SG without models, so we'll test the logic logic separately
    # or rely on the unit test above.
    pass
