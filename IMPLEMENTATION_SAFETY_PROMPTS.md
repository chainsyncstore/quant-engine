# Quant Engine Safety Implementation Prompts

Status key: pending, accepted, denied, implemented, revisited.

## Phase 1 - Persistent Hard Risk Pause

Status: implemented

Goal: Make hard risk breaches persistent across bot restarts and prevent automatic session resume while paused.

Acceptance notes:
- A hard risk breach stores a durable pause reason/state in the database.
- Bot restart does not clear the pause.
- Active sessions do not auto-resume while a hard risk pause is present.
- Telegram stats/help surface the pause state clearly.
- Tests cover restart/resume behavior.

Implementation notes:
- Durable pause fields added to `quant/telebot/models.py`.
- SQLite backfill added in `quant/telebot/main.py`.
- `HardRiskPauseEvent` and callback hook added in `quant_v2/execution/service.py`.
- Startup and demo/live start/continue flows now block on persisted hard-risk pause.
- `/status`, `/stats`, and lifetime stats show pause reason/timestamp.
- Manual DB/admin clearing documented in `HARD_RISK_PAUSE.md`.
- Verification reported: `576 passed, 23 warnings`.

## Phase 2 - Market-Wide Short Guard

Status: implemented

Goal: Prevent the bot from building a heavily net-short book during broad market selloffs unless evidence is unusually strong.

Acceptance notes:
- Compute a broad-market direction/risk score from the traded universe.
- If most symbols are down over a configurable window, new short entries are gated or heavily reduced.
- Flatten/reduce-only behavior still works.
- Tests cover broad-market selloff and normal market cases.

Implementation notes:
- `MarketRiskSnapshot` added and attached to `StrategySignal`.
- Signal cycles compute broad-market selloff metrics from fetched close histories.
- Weak fresh sells are blocked during broad selloffs, with very strong sells still allowed.
- Current long exposure can still be flattened.
- Allocation enforces the portfolio-aware guard and caps guarded net-short exposure.
- Planner passes current positions into allocation.
- Tuning documented in `MARKET_SHORT_GUARD.md`.
- Defaults: 30h lookback, 70% down ratio, median return <= -1.5% or BTC <= -2.0%, strong SELL confidence >= 0.75, guarded net-short cap 15%.
- Verification reported: `586 passed, 23 warnings`.

## Phase 3 - Regime 2 Reversion Risk Treatment

Status: implemented

Goal: Stop treating regime 2 as fully low-risk for directional entries.

Acceptance notes:
- Regime 2 thresholds and/or allocation multipliers are more conservative.
- Shorts in regime 2 require stronger confidence or reduced size.
- Tests assert threshold and sizing changes.

Implementation notes:
- Regime 2 risk changed to `0.35`; regime 1 remains `0.0`.
- Configurable regime 2 thresholds added:
  `BOT_V2_REGIME2_BUY_THRESHOLD=0.57`,
  `BOT_V2_REGIME2_SELL_THRESHOLD=0.35`.
- Fresh regime 2 shorts dampened with `BOT_V2_REGIME2_SELL_ALLOCATION_MULT=0.50`.
- Regime 2 SELL against an existing long is flatten-only and cannot flip into a fresh short.
- Operational note added in `REGIME2_REVERSION_TREATMENT.md`.
- Verification reported: targeted suite `77 passed`; full suite `597 passed, 23 warnings`.

## Phase 4 - Chronos Agreement Gate

Status: implemented

Goal: Use Chronos as a meaningful veto/dampener when LightGBM and Chronos disagree on direction.

Acceptance notes:
- FullEnsemble exposes source-level probabilities or agreement details.
- Strong directional entries require agreement, or disagreement reduces exposure to a safe fraction/HOLD.
- Logs show LGBM probability, Chronos probability, final probability, and agreement.
- Tests cover agreement/disagreement behavior.

Implementation notes:
- `ModelSourceDetails` and `StrategySignal.model_sources` added.
- `FullEnsemble.predict()` remains backward compatible; `predict_with_details()` added.
- Signal manager attaches/logs LGBM probability, Chronos probability, final probability, directions, and agreement state.
- Allocator blocks fresh Chronos-disagreed entries by default, allows extreme LGBM confidence override, and preserves flatten-only behavior.
- Operational note added in `CHRONOS_AGREEMENT_GATE.md`.
- Defaults:
  `BOT_V2_REQUIRE_CHRONOS_AGREEMENT_FOR_ENTRY=1`,
  `BOT_V2_CHRONOS_DISAGREEMENT_MULT=0.25`,
  `BOT_V2_CHRONOS_EXTREME_CONFIDENCE=0.80`.
- Verification reported: targeted suite `96 passed`; full suite `611 passed, 23 warnings`.

## Phase 5 - Manual Model Promotion / Paper Quarantine

Status: implemented

Goal: Retrains may train and register models, but production activation requires manual approval or a paper quarantine pass.

Acceptance notes:
- Scheduled retrain does not auto-promote by default.
- New model records include candidate status and metrics.
- A manual command or env flag is required to activate.
- Optional quarantine stats are recorded before activation.

Implementation notes:
- Scheduled retrain registers candidate models by default and only updates active pointer with explicit `BOT_RETRAIN_AUTO_PROMOTE=1`.
- Registry records now track status, promotion audit fields, candidate listing, paper quarantine marking, and validated manual promotion.
- Telegram admin commands added:
  `/model_candidates`,
  `/model_promote <version_id>`,
  `/model_quarantine <version_id>`.
- Existing `/model_active`, `/model_versions`, and `/model_rollback` now align with candidate workflow.
- Operational runbook added in `MODEL_PROMOTION_RUNBOOK.md`.
- Retrain timestamp weighting fragility fixed with positional/Series-safe timedelta handling.
- Verification reported: focused suite `16 passed`; full suite `618 passed, 23 warnings`.
