"""
Validation script: Check if the system is just waiting for entry conditions.
Run this to diagnose why no trades are being generated.
"""
import pandas as pd
from pathlib import Path
from data.market_loader import MarketDataLoader
from state.market_state import MarketState
from state.position_state import PositionState
from clock.clock import Clock
from hypotheses.base import IntentType

# Config
CSV_PATH = r"C:\Users\HP\AppData\Roaming\MetaQuotes\Terminal\10CE948A1DFC9A8C27E56E827008EBD4\MQL5\Files\results\live_crypto.csv"
SYMBOLS = ["BTCUSD", "ETHUSD"]

def main():
    print("=" * 60)
    print("SYSTEM VALIDATION - Entry Condition Check")
    print("=" * 60)
    
    # 1. Load CSV data
    try:
        csv_df = pd.read_csv(CSV_PATH)
        print(f"\n[OK] CSV loaded: {len(csv_df)} total rows")
    except Exception as e:
        print(f"\n[FAIL] Cannot load CSV: {e}")
        return
    
    # 2. Check data per symbol
    for symbol in SYMBOLS:
        print(f"\n--- {symbol} ---")
        if "symbol" in csv_df.columns:
            symbol_df = csv_df[csv_df["symbol"] == symbol]
        else:
            symbol_df = csv_df
            
        bar_count = len(symbol_df)
        print(f"  Bars available: {bar_count}")
        
        if bar_count < 15:
            print(f"  [WAIT] Need at least 15 bars, have {bar_count}")
            continue
        
        # Load bars
        try:
            bars = MarketDataLoader.load_from_dataframe(symbol_df, symbol=symbol)
            print(f"  Bars loaded: {len(bars)}")
        except Exception as e:
            print(f"  [ERROR] Loading bars: {e}")
            continue
        
        # Build market state
        market_state = MarketState(lookback_window=100)
        for bar in bars:
            market_state.update(bar)
        
        current_bar = market_state.current_bar()
        print(f"  Latest bar: {current_bar.timestamp}")
        print(f"  Price: O={current_bar.open:.2f} H={current_bar.high:.2f} L={current_bar.low:.2f} C={current_bar.close:.2f}")
        
        # 3. Test each hypothesis
        from hypotheses.competition.crypto_momentum_breakout import CryptoMomentumBreakout
        from hypotheses.competition.rsi_extreme_reversal import RSIExtremeReversal
        from hypotheses.competition.volatility_expansion_assault import VolatilityExpansionAssault
        
        hypotheses = [
            CryptoMomentumBreakout(),
            RSIExtremeReversal(),
            VolatilityExpansionAssault(),
        ]
        
        print(f"\n  Hypothesis Evaluation:")
        for h in hypotheses:
            try:
                intent = h.on_bar(market_state, PositionState(), Clock())
                if intent and intent.type in (IntentType.BUY, IntentType.SELL):
                    print(f"    [SIGNAL] {h.hypothesis_id}: {intent.type.value} (size={intent.size:.2f})")
                else:
                    # Diagnose why no signal
                    diag = diagnose_hypothesis(h, market_state, bars)
                    print(f"    [WAIT] {h.hypothesis_id}: {diag}")
            except Exception as e:
                print(f"    [ERROR] {h.hypothesis_id}: {e}")
    
    print("\n" + "=" * 60)
    print("SUMMARY: System is waiting for entry conditions to be met.")
    print("This is NORMAL - aggressive hypotheses require specific setups.")
    print("=" * 60)


def diagnose_hypothesis(h, market_state, bars):
    """Diagnose why a hypothesis isn't generating a signal."""
    hid = h.hypothesis_id
    
    if hid == "crypto_momentum_breakout":
        # Check EMA crossover and ROC
        closes = [b.close for b in bars[-30:]]
        if len(closes) < 26:
            return f"Need 26 bars for EMA, have {len(closes)}"
        
        fast_ema = ema(closes, 12)
        slow_ema = ema(closes, 26)
        roc = (closes[-1] - closes[-5]) / closes[-5] * 100 if len(closes) >= 5 else 0
        
        if abs(fast_ema - slow_ema) / slow_ema < 0.001:
            return f"EMAs too close (fast={fast_ema:.2f}, slow={slow_ema:.2f})"
        if abs(roc) < 0.2:
            return f"ROC too weak ({roc:.3f}%, need >0.2%)"
        return f"No EMA cross detected (fast={fast_ema:.2f}, slow={slow_ema:.2f}, roc={roc:.3f}%)"
    
    elif hid == "rsi_extreme_reversal":
        closes = [b.close for b in bars[-20:]]
        if len(closes) < 14:
            return f"Need 14 bars for RSI, have {len(closes)}"
        rsi_val = rsi(closes, 14)
        if 25 <= rsi_val <= 75:
            return f"RSI not extreme ({rsi_val:.1f}, need <25 or >75)"
        return f"Candle pattern not confirming (RSI={rsi_val:.1f})"
    
    elif hid == "volatility_expansion_assault":
        if len(bars) < 19:
            return f"Need 19 bars for ATR, have {len(bars)}"
        atr_val = atr(bars[-19:], 14)
        current = bars[-1]
        candle_range = current.high - current.low
        ratio = candle_range / atr_val if atr_val > 0 else 0
        if ratio < 1.5:
            return f"Candle range not expanded ({ratio:.2f}x ATR, need >1.5x)"
        return f"Direction unclear (range={ratio:.2f}x ATR)"
    
    return "Conditions not met"


def ema(data, period):
    """Calculate EMA."""
    if len(data) < period:
        return data[-1]
    multiplier = 2 / (period + 1)
    ema_val = sum(data[:period]) / period
    for price in data[period:]:
        ema_val = (price - ema_val) * multiplier + ema_val
    return ema_val


def rsi(closes, period=14):
    """Calculate RSI."""
    if len(closes) < period + 1:
        return 50.0
    deltas = [closes[i] - closes[i-1] for i in range(1, len(closes))]
    gains = [d if d > 0 else 0 for d in deltas]
    losses = [-d if d < 0 else 0 for d in deltas]
    avg_gain = sum(gains[-period:]) / period
    avg_loss = sum(losses[-period:]) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def atr(bars, period=14):
    """Calculate ATR."""
    if len(bars) < 2:
        return 0
    trs = []
    for i in range(1, len(bars)):
        tr = max(
            bars[i].high - bars[i].low,
            abs(bars[i].high - bars[i-1].close),
            abs(bars[i].low - bars[i-1].close)
        )
        trs.append(tr)
    return sum(trs[-period:]) / min(period, len(trs)) if trs else 0


if __name__ == "__main__":
    main()
