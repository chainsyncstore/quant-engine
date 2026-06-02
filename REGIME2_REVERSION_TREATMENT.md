# Regime 2 Reversion Risk Treatment

Phase 3 makes regime 2 safer than regime 1. Regime 2 represents a strong downtrend with non-crowded funding, so the bot treats it as reversion/downtrend caution rather than a fully low-risk state.

## Defaults

- `quant_v2.strategy.regime` maps regime 2 to `regime_risk=0.35`.
- Signal thresholds are widened in `V2SignalManager` when `regime == 2`.
- `BOT_V2_REGIME2_BUY_THRESHOLD` defaults to `0.57`.
- `BOT_V2_REGIME2_SELL_THRESHOLD` defaults to `0.35`.
- Because regime 2 has `regime_risk=0.35`, the effective default BUY threshold is `max(0.55 + 0.08 * 0.35, 0.57)`, or about `0.578`.
- The effective default SELL threshold is `0.35`, meaning a SELL needs `probability_up <= 0.35` or sell confidence of at least `0.65`.

## Allocation Treatment

The allocator also dampens regime 2 SELL targets:

- `BOT_V2_REGIME2_SELL_ALLOCATION_MULT` defaults to `0.50`.
- Fresh or additive regime 2 SELL exposure is multiplied by this value after normal confidence, session, regime-bias, accuracy, event, agreement, and cost gates.
- If a regime 2 SELL arrives while the bot is long that symbol, allocation treats it as flatten-only. This lets the bot close or reduce the long without flipping directly into fresh short exposure under regime 2.

## Tuning

Use stricter settings during drawdown-sensitive operation:

- Lower `BOT_V2_REGIME2_SELL_THRESHOLD` to require stronger SELL confidence.
- Lower `BOT_V2_REGIME2_SELL_ALLOCATION_MULT` to reduce regime 2 short size.
- Raise `BOT_V2_REGIME2_BUY_THRESHOLD` if long entries should also be rarer in regime 2.

Set `BOT_V2_REGIME2_SELL_ALLOCATION_MULT=1.0` to disable allocation dampening while keeping the stricter signal threshold.
