"""Tests for walk-forward evaluation module."""
import pandas as pd
from datetime import datetime, timedelta

from evaluation.walk_forward import WalkForwardConfig, WalkForwardGenerator


def test_walk_forward_generation():
    """Test standard window generation."""
    # Create 200 days of data
    start_date = datetime(2023, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(200)]
    data = pd.DataFrame(index=dates, data={"close": range(200)})
    
    config = WalkForwardConfig(
        train_window_size=100,
        test_window_size=20,
        step_size=20
    )
    
    generator = WalkForwardGenerator(data, config)
    windows = list(generator.generate_windows())
    
    # We expect:
    # 0: Train[0:100], Test[100:120]
    # 1: Train[20:120], Test[120:140]
    # 2: Train[40:140], Test[140:160]
    # 3: Train[60:160], Test[160:180]
    # 4: Train[80:180], Test[180:200]
    
    assert len(windows) == 5
    
    # Check first window
    w0_train, w0_test = windows[0]
    assert w0_train.window_index == 0
    assert w0_train.window_type == "TRAIN"
    assert w0_train.start_timestamp == dates[0]
    assert w0_train.end_timestamp == dates[99]
    
    assert w0_test.window_index == 0
    assert w0_test.window_type == "TEST"
    assert w0_test.start_timestamp == dates[100]
    assert w0_test.end_timestamp == dates[119]

    # Check overlaps (step size = test size, so no test overlap)
    w1_train, w1_test = windows[1]
    assert w1_train.start_timestamp == dates[20]
    assert w1_test.start_timestamp == dates[120]


def test_insufficient_data():
    """Test behavior when data is shorter than one window set."""
    dates = [datetime(2023, 1, 1) + timedelta(days=i) for i in range(50)]
    data = pd.DataFrame(index=dates, data={"close": range(50)})
    
    config = WalkForwardConfig(
        train_window_size=100,
        test_window_size=20,
        step_size=20
    )
    
    generator = WalkForwardGenerator(data, config)
    windows = list(generator.generate_windows())
    
    assert len(windows) == 0

def test_custom_step_size():
    """Test step size different from test size."""
    # 150 days
    dates = [datetime(2023, 1, 1) + timedelta(days=i) for i in range(150)]
    data = pd.DataFrame(index=dates, data={"close": range(150)})
    
    config = WalkForwardConfig(
        train_window_size=100,
        test_window_size=10,
        step_size=5  # Small step
    )
    
    generator = WalkForwardGenerator(data, config)
    windows = list(generator.generate_windows())
    
    # 0: Train[0:100], Test[100:110]
    # 1: Train[5:105], Test[105:115]
    # ...
    
    w0_train, w0_test = windows[0]
    w1_train, w1_test = windows[1]
    
    assert w1_train.start_timestamp == dates[5]
    assert w1_test.start_timestamp == dates[105]

