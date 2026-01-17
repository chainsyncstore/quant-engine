"""
Market state management.

Maintains a rolling window of historical bars for hypothesis access.
Provides read-only access to past and present market data.
"""

from collections import deque
from datetime import datetime
from typing import List, Optional

import pandas as pd

from data.schemas import Bar


class MarketState:
    """
    Maintains a rolling window of historical market bars.
    
    This provides hypotheses with access to past market data while preventing
    look-ahead bias (no access to future bars).
    """
    
    def __init__(self, lookback_window: int = 100):
        """
        Initialize market state.
        
        Args:
            lookback_window: Maximum number of historical bars to retain
        """
        if lookback_window < 1:
            raise ValueError("Lookback window must be at least 1")
        
        self._lookback_window = lookback_window
        self._bars: deque[Bar] = deque(maxlen=lookback_window)
        self._current_bar: Optional[Bar] = None
    
    def update(self, bar: Bar) -> None:
        """
        Update state with a new bar.
        
        This should only be called by the replay engine.
        
        Args:
            bar: New bar to add to history
        """
        # If we have a current bar, add it to history
        if self._current_bar is not None:
            self._bars.append(self._current_bar)
        
        # Set new current bar
        self._current_bar = bar
    
    def current_bar(self) -> Bar:
        """
        Get the current (most recent) bar.
        
        Returns:
            Current bar
            
        Raises:
            RuntimeError: If no bar has been set yet
        """
        if self._current_bar is None:
            raise RuntimeError("Market state has not been initialized with any bars")
        
        return self._current_bar
    
    def get_bars(self, n: int | None = None) -> List[Bar]:
        """
        Get the last n historical bars (not including current bar).
        
        Args:
            n: Number of bars to retrieve. If None, returns all available bars.
            
        Returns:
            List of bars in chronological order (oldest first)
        """
        if n is None:
            return list(self._bars)
        
        if n < 0:
            raise ValueError("Number of bars must be non-negative")
        
        # Return last n bars
        return list(self._bars)[-n:] if n > 0 else []
        
    def get_history(self) -> List[Bar]:
        """
        Get all available historical bars (alias for get_bars()).
        
        Returns:
            List of bars in chronological order
        """
        return self.get_bars()
    
    def recent_bars(self, n: int) -> List[Bar]:
        """
        Get the last n bars including the current bar.
        
        Args:
            n: Number of bars to retrieve
            
        Returns:
            List of bars in chronological order (oldest first), including current bar
        """
        if n <= 0:
            return []
        
        # Get historical bars + current bar
        all_bars = list(self._bars)
        if self._current_bar is not None:
            all_bars.append(self._current_bar)
        
        # Return last n bars
        return all_bars[-n:] if len(all_bars) >= n else all_bars
    
    def get_bar(self, index: int) -> Bar | None:
        """
        Get a specific historical bar by index.
        
        Args:
            index: Index into history (0 = oldest, -1 = most recent historical bar)
            
        Returns:
            Bar at the specified index, or None if out of range
        """
        try:
            return self._bars[index]
        except IndexError:
            return None
    
    def bar_count(self) -> int:
        """
        Get the number of historical bars available (not including current bar).
        
        Returns:
            Number of bars in history
        """
        return len(self._bars)
    
    def has_minimum_history(self, min_bars: int) -> bool:
        """
        Check if we have at least min_bars of history.
        
        Args:
            min_bars: Minimum required bars
            
        Returns:
            True if we have enough history, False otherwise
        """
        return len(self._bars) >= min_bars
    
    def get_close_prices(self, n: int | None = None) -> List[float]:
        """
        Get closing prices for the last n bars.
        
        Args:
            n: Number of prices to retrieve. If None, returns all available.
            
        Returns:
            List of closing prices in chronological order
        """
        bars = self.get_bars(n)
        return [bar.close for bar in bars]
    
    def get_current_price(self) -> float:
        """
        Get the current close price.
        
        Returns:
            Current bar's close price
            
        Raises:
            RuntimeError: If no bar has been set yet
        """
        return self.current_bar().close
    
    def get_current_timestamp(self) -> datetime:
        """
        Get the current timestamp.
        
        Returns:
            Current bar's timestamp
            
        Raises:
            RuntimeError: If no bar has been set yet
        """
        return self.current_bar().timestamp
    
    def to_dataframe(self, n: int | None = None) -> pd.DataFrame:
        """
        Convert history to pandas DataFrame.
        
        Args:
            n: Number of bars to include. If None, includes all history + current bar.
            
        Returns:
            DataFrame with columns [open, high, low, close, volume] and datetime index.
        """
        # Get history
        bars = self.get_bars(n)
        
        # Add current bar if initialized
        if self._current_bar and (n is None or n > len(bars)):
            bars.append(self._current_bar)
            
        if not bars:
            return pd.DataFrame()
            
        data = [
            {
                "timestamp": b.timestamp,
                "open": b.open,
                "high": b.high,
                "low": b.low,
                "close": b.close,
                "volume": b.volume
            }
            for b in bars
        ]
        
        df = pd.DataFrame(data)
        df.set_index("timestamp", inplace=True)
        return df

    def reset(self) -> None:
        """
        Reset the market state.
        
        Should only be used for testing or starting a new evaluation run.
        """
        self._bars.clear()
        self._current_bar = None
