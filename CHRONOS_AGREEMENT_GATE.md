# Chronos Agreement Gate

Phase 4 adds a source-level agreement gate for entries produced by the LightGBM + Chronos `FullEnsemble`.

## Source Metadata

`FullEnsemble.predict()` remains backward compatible and still returns:

```python
(probability, uncertainty, model_agreement)
```

`FullEnsemble.predict_with_details()` also returns `ModelSourceDetails`, which includes:

- `lgbm_probability`
- `chronos_probability`
- `final_probability`
- `lgbm_direction`
- `chronos_direction`
- `agreement`
- `chronos_enabled`

The signal manager attaches these details to `StrategySignal.model_sources` and includes them in signal logs and signal reasons when available.

## Default Gate

The allocator uses the metadata to prevent ordinary fresh directional entries when Chronos disagrees with LightGBM.

Default environment variables:

- `BOT_V2_REQUIRE_CHRONOS_AGREEMENT_FOR_ENTRY=1`
- `BOT_V2_CHRONOS_DISAGREEMENT_MULT=0.25`
- `BOT_V2_CHRONOS_EXTREME_CONFIDENCE=0.80`

With the default requirement enabled:

- Fresh BUY is blocked when LightGBM is bullish and Chronos is bearish, unless LightGBM BUY confidence is at least `0.80`.
- Fresh SELL is blocked when LightGBM is bearish and Chronos is bullish, unless LightGBM SELL confidence is at least `0.80`.
- If the signal would reduce or flatten an opposing position, it remains allowed as flatten-only.

If `BOT_V2_REQUIRE_CHRONOS_AGREEMENT_FOR_ENTRY=0`, disagreement does not block the entry, but the target exposure is multiplied by `BOT_V2_CHRONOS_DISAGREEMENT_MULT`.

## Fallback Behavior

If Chronos is disabled, the gate does not apply and the bot uses the existing LightGBM path.

If Chronos is enabled but unavailable during prediction, `FullEnsemble` falls back to the LightGBM probability and marks Chronos unavailable in `ModelSourceDetails`. When agreement is required, the allocator blocks fresh entries from that signal and still allows flatten-only behavior.

## Tuning

- Raise `BOT_V2_CHRONOS_EXTREME_CONFIDENCE` to make overrides rarer.
- Lower `BOT_V2_CHRONOS_DISAGREEMENT_MULT` for stronger dampening when agreement is not strictly required.
- Set `BOT_V2_REQUIRE_CHRONOS_AGREEMENT_FOR_ENTRY=0` to use dampening instead of blocking.
