"""Tests for regime classification."""
import pandas as pd
import numpy as np

from analysis.regime import RegimeClassifier, MarketRegime


def test_classify_trend():
    """Test trend classification."""
    classifier = RegimeClassifier()
    dates = pd.date_range(start="2023-01-01", periods=100)
    
    # Uptrend: Price consistently above SMA (linear growth)
    prices_up = pd.Series(data=np.linspace(100, 200, 100), index=dates)
    trend_up = classifier.classify_trend(prices_up, window=50)
    # Check last point. SMA50 of linspace(150..200) is approx 175. Price is 200. 
    # 200 > 175 * 1.02 (178.5) -> TRENDING_UP
    assert trend_up == MarketRegime.TRENDING_UP
    
    # Downtrend
    prices_down = pd.Series(data=np.linspace(200, 100, 100), index=dates)
    trend_down = classifier.classify_trend(prices_down, window=50)
    assert trend_down == MarketRegime.TRENDING_DOWN
    
    # Sideways
    prices_flat = pd.Series(data=[100] * 100, index=dates)
    trend_flat = classifier.classify_trend(prices_flat, window=50)
    assert trend_flat == MarketRegime.SIDEWAYS


def test_classify_volatility():
    """Test volatility classification."""
    classifier = RegimeClassifier(high_vol_threshold_annualized=0.20)
    dates = pd.date_range(start="2023-01-01", periods=100)
    
    # Low Vol: 1% daily moves? No, that's high. 
    # 0.20 annualized is ~1.2% daily vol? No, 20% / sqrt(252) ~= 1.26%.
    
    # Create distinct low vol series
    # 0.001 daily std dev -> ~1.5% annualized
    returns_low = pd.Series(np.random.normal(0, 0.001, 100), index=dates)
    vol_low = classifier.classify_volatility(returns_low, window=20)
    assert vol_low == MarketRegime.LOW_VOL
    
    # Create high vol series
    # 0.02 daily std dev -> ~31% annualized
    returns_high = pd.Series(np.random.normal(0, 0.02, 100), index=dates)
    vol_high = classifier.classify_volatility(returns_high, window=20)
    assert vol_high == MarketRegime.HIGH_VOL

