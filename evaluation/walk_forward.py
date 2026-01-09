"""
Walk-Forward Evaluation Module.

Prevents overfitting by enforcing strict temporal separation between learning and evaluation.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Iterator, Tuple, Literal, cast

import pandas as pd


@dataclass
class WalkForwardConfig:
    """Configuration for walk-forward analysis."""
    train_window_size: int  # Number of bars for training
    test_window_size: int   # Number of bars for testing
    step_size: int          # Number of bars to step forward (usually same as test_window)


@dataclass
class EvaluationWindow:
    """A single evaluation window (train or test)."""
    window_index: int
    window_type: Literal["TRAIN", "TEST"]
    start_timestamp: datetime
    end_timestamp: datetime


class WalkForwardGenerator:
    """Generates walk-forward windows from a time series."""
    
    def __init__(self, data: pd.DataFrame, config: WalkForwardConfig):
        """
        Initialize the generator.
        
        Args:
            data: DataFrame with a DatetimeIndex
            config: WalkForward configuration
        """
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("Data must have a DatetimeIndex")
            
        self._data = data
        self._config = config
        
    def generate_windows(self) -> Iterator[Tuple[EvaluationWindow, EvaluationWindow]]:
        """
        Yield pairs of (train_window, test_window).
        
        Yields:
            Tuple of (train_window, test_window) metadata
        """
        total_bars = len(self._data)
        train_size = self._config.train_window_size
        test_size = self._config.test_window_size
        step = self._config.step_size
        
        # We need at least one train + test sequence
        if total_bars < train_size + test_size:
            return
            
        current_start_idx = 0
        window_idx = 0
        
        while current_start_idx + train_size + test_size <= total_bars:
            # Define indices
            train_end_idx = current_start_idx + train_size
            test_end_idx = train_end_idx + test_size
            
            # Get timestamps
            train_start = self._data.index[current_start_idx]
            train_end = self._data.index[train_end_idx - 1] # Inclusive
            
            test_start = self._data.index[train_end_idx]
            test_end = self._data.index[test_end_idx - 1]   # Inclusive
            
            # Create window objects
            train_window = EvaluationWindow(
                window_index=window_idx,
                window_type="TRAIN",
                start_timestamp=train_start,
                end_timestamp=train_end
            )
            
            test_window = EvaluationWindow(
                window_index=window_idx,
                window_type="TEST",
                start_timestamp=test_start,
                end_timestamp=test_end
            )
            
            yield train_window, test_window
            
            # Step forward
            current_start_idx += step
            window_idx += 1


@dataclass
class DecayMetrics:
    """Metrics for decay analysis."""
    sharpe_change: float
    win_rate_change: float
    drawdown_change: float
    result_tag: Literal["PASS", "FAIL", "DECAY"]


class DecayTracker:
    """Tracks performance decay between windows."""
    
    def __init__(self, decay_threshold_sharpe: float = 0.50):
        """
        Initialize decay tracker.
        
        Args:
            decay_threshold_sharpe: Percentage drop in Sharpe to consider DECAY (e.g. 0.50 = 50% drop)
        """
        self._threshold = decay_threshold_sharpe
        
    def analyze_decay(self, in_sample_metrics: dict, out_sample_metrics: dict) -> DecayMetrics:
        """
        Compare in-sample vs out-of-sample metrics.
        
        Args:
            in_sample_metrics: Metrics from TRAIN window
            out_sample_metrics: Metrics from TEST window
            
        Returns:
            DecayMetrics object
        """
        # Extract metrics (default to 0.0 if missing)
        is_sharpe = in_sample_metrics.get("sharpe_ratio", 0.0) or 0.0
        os_sharpe = out_sample_metrics.get("sharpe_ratio", 0.0) or 0.0
        
        is_win_rate = in_sample_metrics.get("win_rate", 0.0) or 0.0
        os_win_rate = out_sample_metrics.get("win_rate", 0.0) or 0.0
        
        is_dd = in_sample_metrics.get("max_drawdown", 0.0) or 0.0
        os_dd = out_sample_metrics.get("max_drawdown", 0.0) or 0.0
        
        # Calculate changes
        sharpe_change = os_sharpe - is_sharpe
        win_rate_change = os_win_rate - is_win_rate
        drawdown_change = os_dd - is_dd # Positive means worse drawdown usually (if DD is positive number)
        
        # Determine tag
        # DECAY logic: Significant drop in Sharpe
        tag: Literal["PASS", "FAIL", "DECAY"] = "PASS"
        
        # If IS Sharpe was positive and OS Sharpe drops by more than X%
        if is_sharpe > 0:
            drop_pct = (is_sharpe - os_sharpe) / is_sharpe
            if drop_pct > self._threshold:
                tag = "DECAY"
        
        # If OS Sharpe is negative, it's a FAIL (or DECAY if it was positive before)
        if os_sharpe < 0:
            tag = "FAIL"
            
        return DecayMetrics(
            sharpe_change=sharpe_change,
            win_rate_change=win_rate_change,
            drawdown_change=drawdown_change,
            result_tag=cast(Literal["PASS", "FAIL", "DECAY"], tag)
        )
