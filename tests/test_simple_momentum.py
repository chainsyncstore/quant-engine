"""Quick test to verify simple_momentum generates trades."""
from datetime import datetime, timedelta
from hypotheses.examples.simple_momentum import SimpleMomentumHypothesis
from state.market_state import MarketState
from state.position_state import PositionState
from clock.clock import Clock
from data.schemas import Bar


def test_simple_momentum_generates_trades():
    """Verify simple_momentum generates entry and exit trades."""
    h = SimpleMomentumHypothesis(hold_bars=2)
    ms = MarketState(lookback_window=20)
    ps = PositionState()
    clock = Clock()
    
    # Create bars with up-close pattern
    start = datetime(2023, 1, 1)
    bars = []
    for i in range(10):
        # Alternating up and down bars
        if i % 2 == 0:
            # Up bar: close > open
            bars.append(Bar(
                timestamp=start + timedelta(days=i),
                open=100.0, high=105.0, low=99.0, close=104.0, volume=1000
            ))
        else:
            # Down bar: close < open
            bars.append(Bar(
                timestamp=start + timedelta(days=i),
                open=104.0, high=106.0, low=98.0, close=99.0, volume=1000
            ))
    
    intents = []
    for bar in bars:
        clock.set_time(bar.timestamp)
        ms.update(bar)
        intent = h.on_bar(ms, ps, clock)
        if intent:
            intents.append(intent)
            # Simulate execution
            from hypotheses.base import IntentType
            if intent.type == IntentType.BUY:
                ps.open_position(
                    side='LONG',
                    entry_price=bar.close,
                    size=100,
                    entry_timestamp=bar.timestamp,
                    entry_capital=bar.close * 100
                )
            elif intent.type == IntentType.CLOSE:
                ps.close_position()
    
    print(f"Intents generated: {len(intents)}")
    for i, intent in enumerate(intents):
        print(f"  {i}: {intent.type.value}")
    
    assert len(intents) >= 2, f"Expected at least 2 intents (entry + exit), got {len(intents)}"


if __name__ == "__main__":
    test_simple_momentum_generates_trades()
    print("SUCCESS!")
