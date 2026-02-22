from unittest.mock import MagicMock, patch

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
