# Market-Wide Short Guard

Phase 2 adds a portfolio-level guard against building fresh net-short exposure
after the traded crypto universe has already sold off together.

## Default Trigger

Each v2 signal cycle computes a market risk snapshot from recent close history
for the traded universe:

- Lookback: `30` hourly bars.
- Broad selloff down ratio: at least `70%` of evaluated symbols are down.
- Selloff severity: median universe return `<= -1.5%` or BTC return `<= -2.0%`.

When active, ordinary new `SELL` entries are blocked. A `SELL` is still allowed
when sell confidence is unusually strong, default `>= 0.75` (`probability_up <=
0.25`). Existing exposure reduction remains allowed because blocked weak `SELL`s
produce no fresh short target, allowing execution reconciliation to flatten or
reduce current positions rather than open a new short.

Strong short portfolios are still capped by default to `15%` net short exposure
under the guard.

## Tuning

Set these environment variables before starting the bot:

```text
BOT_V2_MARKET_SHORT_GUARD_LOOKBACK_HOURS=30
BOT_V2_MARKET_SHORT_GUARD_DOWN_RATIO=0.70
BOT_V2_MARKET_SHORT_GUARD_MEDIAN_RETURN=-0.015
BOT_V2_MARKET_SHORT_GUARD_BTC_RETURN=-0.020
BOT_V2_MARKET_SHORT_GUARD_STRONG_CONFIDENCE=0.75
BOT_V2_MARKET_SHORT_GUARD_NET_CAP_FRAC=0.15
```

Lowering the down-ratio or return thresholds makes the guard activate more
often. Raising `BOT_V2_MARKET_SHORT_GUARD_STRONG_CONFIDENCE` makes it harder for
new shorts to pass during broad selloffs. Lowering
`BOT_V2_MARKET_SHORT_GUARD_NET_CAP_FRAC` further reduces allowed net-short
exposure when the guard is active.
