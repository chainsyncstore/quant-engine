# MT5 Leveraged Trading Guide - Competition Mode

**Time-sensitive**: 3 days remaining in competition. Current rank: 3000+. Target: Top 10.

## Why You're Behind

Your system was sizing positions at **0.15x equity** (de-leveraged). Competitors using **1:30 leverage** are controlling positions worth **30x their margin**.

| Your Old Position | Competitor Position | Difference |
|-------------------|---------------------|------------|
| $1,500 notional   | $300,000 notional   | **200x smaller** |
| 0.5% move = $7.50 | 0.5% move = $1,500  | **200x less profit** |

---

## Quick Reference: Lot Sizing for 1:30 Leverage

### Forex Pairs (EURUSD, GBPUSD, etc.)

| Your Equity | Conservative (10x) | Moderate (20x) | Aggressive (25x) |
|-------------|-------------------|----------------|------------------|
| $5,000      | 0.50 lots         | 1.00 lots      | 1.25 lots        |
| $10,000     | 1.00 lots         | 2.00 lots      | 2.50 lots        |
| $25,000     | 2.50 lots         | 5.00 lots      | 6.25 lots        |

**Formula**: `Lots = (Equity × Target_Leverage) / 100,000`

### Crypto (BTCUSD/ETHUSD)

Contract sizes vary by broker. Check your broker's specifications.

**General rule**: If 1 lot = 1 BTC (~$100k), and you want 10x leverage on $10k:
- Target notional = $100,000
- Lots = 1.0

---

## Step-by-Step MT5 Execution

### When Telegram Signal Says: **BUY BTCUSD**

1. **Open MT5** → Press **F9** or click "New Order"

2. **Set Symbol**: Select the exact symbol from your signal

3. **Calculate Lot Size**:
   ```
   Target Leverage: 15-20x (aggressive but survivable)
   Lots = (Your_Equity × 15) / Contract_Value_Per_Lot
   ```

4. **Set Stop Loss** (CRITICAL):
   - **1.0-1.5%** from entry for leverage trades
   - This limits account loss to ~15-22% if stopped
   
5. **Set Take Profit**:
   - **2-3%** from entry (2:1 or 3:1 reward:risk)

6. **Execute**: Click BUY or SELL

### Example Trade

- **Equity**: $10,000
- **Signal**: BUY EURUSD @ 1.0850
- **Target leverage**: 15x
- **Calculation**: ($10,000 × 15) / $100,000 = **1.5 lots**
- **Stop Loss**: 1.0850 - 0.0108 (1%) = 1.0742
- **Take Profit**: 1.0850 + 0.0217 (2%) = 1.1067

---

## Risk Management Rules

### Position Limits
- **Max 1-2 positions** at a time with high leverage
- **Never use full 30x** unless signal confidence is maximum

### Daily Loss Limit
- **Stop trading** if down 4-5% for the day
- You need to survive to trade tomorrow

### The Math of Survival

| Leverage | 1% Adverse Move | 2% Adverse Move |
|----------|-----------------|-----------------|
| 10x      | -10% equity     | -20% equity     |
| 20x      | -20% equity     | -40% equity     |
| 30x      | -30% equity     | -60% equity     |

**Translation**: At 20x leverage, a 2% move against you wipes 40% of your account.

---

## Competition Strategy (3 Days Left)

### Day 1-2: Aggressive Catch-Up
- Use **15-20x effective leverage**
- Target **2-3 high-conviction trades** per day
- Focus on **volatile sessions** (London/NY overlap for Forex, any time for Crypto)

### Day 3: Protect Gains or Hail Mary
- If you've climbed significantly: reduce leverage, protect rank
- If still far behind: maintain aggression, nothing to lose

### Signal Priority
When your system sends a signal with **HIGH confidence**:
- Act immediately
- Size aggressively (20x+)
- Tight stop (1%)

When signal is **LOW/MEDIUM confidence**:
- Smaller size (10-15x)
- Or skip entirely if already have open position

---

## Common Mistakes to Avoid

1. **No Stop Loss**: One bad trade erases weeks of gains
2. **Full 30x on every trade**: Guaranteed blowup
3. **Averaging down**: Adding to losers multiplies losses
4. **Overtrading**: Quality > Quantity
5. **Moving stop loss further**: Accept the loss, move on

---

## System Integration

Your updated system now calculates leveraged position sizes in the Telegram signals.

The `quantity` field in signals now reflects:
```
quantity = equity × risk_fraction × effective_leverage / price
```

With the new competition configuration (`config/competition_leverage.py`), HIGH confidence signals will output ~10x larger position sizes than before.

**To activate**: Import and use `COMPETITION_RISK_RESOLVER` in your meta engine setup.
