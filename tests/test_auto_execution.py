from unittest.mock import MagicMock, patch

import pandas as pd

# Mock dependencies before import if needed
with patch("quant.live.signal_generator.BinanceClient"), \
     patch("quant.live.signal_generator.load_model"), \
     patch("quant.live.signal_generator.gmm_regime.load_model"):
    from quant.live.signal_generator import SignalGenerator

def setup_generator():
    """Setup a SignalGenerator with mocked internals."""
    # Mock model_dir to pass .exists() checks
    mock_dir = MagicMock()
    mock_dir.exists.return_value = True
    # Handle the / operator (Path joining)
    mock_path = MagicMock()
    mock_path.exists.return_value = True
    mock_dir.__truediv__.return_value = mock_path
    
    # We need to patch the __init__ file reading parts
    with patch("builtins.open"), \
         patch("json.load", return_value={
             "mode": "crypto",
             "feature_cols": ["f1", "f2"], 
             "spread": 1.0,
             "horizons": [4],
             "regime_config": {},
             "regime_thresholds": {}
         }), \
         patch("quant.live.signal_generator.load_model"), \
         patch("quant.live.signal_generator.gmm_regime.load_model"):
         
        gen = SignalGenerator(model_dir=mock_dir, capital=10000, horizon=4, live=True)
        gen.binance_client = MagicMock()
        gen.binance_client._cfg.symbol = "BTCUSDT"
        gen._authenticated = True
        return gen

def test_execution_buy_no_position():
    gen = setup_generator()
    gen.binance_client.get_positions.return_value = [] # No current positions
    
    signal = {
        "signal": "BUY",
        "close_price": 50000.0,
        "position": {"lot_size": 1.5},
    }
    
    gen.execute_trade(signal)
    
    gen.binance_client.place_order.assert_called_once_with(
        symbol="BTCUSDT",
        side="BUY",
        quantity=1.5
    )

def test_execution_hold_does_nothing():
    gen = setup_generator()
    gen.binance_client.get_positions.return_value = []
    
    signal = {
        "signal": "HOLD",
        "position": {},
    }
    
    gen.execute_trade(signal)
    
    gen.binance_client.place_order.assert_not_called()
    gen.binance_client.close_position.assert_not_called()

def test_execution_flip_sell_to_buy():
    gen = setup_generator()
    # Current Short Position
    gen.binance_client.get_positions.return_value = [{
        "symbol": "BTCUSDT",
        "positionAmt": "-1.0",
    }]
    
    signal = {
        "signal": "BUY",
        "close_price": 50000.0,
        "position": {"lot_size": 2.0},
    }
    
    gen.execute_trade(signal)
    
    # Should close existing short
    gen.binance_client.close_position.assert_called_once_with("BTCUSDT")
    # And open new long
    gen.binance_client.place_order.assert_called_once_with(
        symbol="BTCUSDT",
        side="BUY",
        quantity=2.0
    )

def test_execution_same_direction_holds():
    gen = setup_generator()
    # Current Long Position
    gen.binance_client.get_positions.return_value = [{
        "symbol": "BTCUSDT",
        "positionAmt": "1.0",
    }]
    
    # Signal is also BUY
    signal = {
        "signal": "BUY",
        "close_price": 50000.0,
        "position": {"lot_size": 2.0},
    }
    
    gen.execute_trade(signal)
    
    # Should NOT close or open anything (Hold)
    gen.binance_client.close_position.assert_not_called()
    gen.binance_client.place_order.assert_not_called()


def test_execution_drift_alert_does_nothing():
    gen = setup_generator()

    signal = {
        "signal": "DRIFT_ALERT",
        "position": {},
    }

    gen.execute_trade(signal)

    gen.binance_client.place_order.assert_not_called()
    gen.binance_client.close_position.assert_not_called()


def test_feature_drift_detector_triggers_on_large_shift():
    gen = setup_generator()
    gen._feature_baseline_mean = pd.Series({"f1": 0.0, "f2": 0.0})
    gen._feature_baseline_std = pd.Series({"f1": 1.0, "f2": 1.0})

    msg = gen._check_feature_drift(pd.DataFrame({"f1": [9.0], "f2": [7.0]}))

    assert msg is not None
    assert "Feature drift detected" in msg


def test_confidence_drift_detector_triggers_on_neutral_collapse():
    gen = setup_generator()

    msg = None
    for _ in range(24):
        msg = gen._check_confidence_drift(0.5)

    assert msg is not None
    assert "Confidence drift detected" in msg
