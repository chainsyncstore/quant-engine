"""
FX symbols for competition mode.

High-volatility pairs suitable for aggressive trading with 1:30 leverage.
"""

# Primary FX pairs for competition (ranked by typical volatility)
FX_SYMBOLS = [
    "GBPJPY",   # Highest volatility major
    "USDJPY",   # Strong moves on BoJ/Fed
    "GBPUSD",   # Brexit-sensitive
]

# Focus symbol for single-symbol mode
FX_FOCUS_SYMBOL = "GBPJPY"
