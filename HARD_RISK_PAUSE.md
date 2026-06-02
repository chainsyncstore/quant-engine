# Persistent Hard Risk Pause

Phase 1 stores confirmed hard-risk pauses on `user_context` so a Telegram bot or
container restart cannot clear them.

## Stored Fields

The pause is persisted per Telegram user/session in `user_context`:

- `hard_risk_paused`: active pause flag.
- `hard_risk_pause_reason`: currently `hard_risk_breach`.
- `hard_risk_pause_triggered_at`: UTC timestamp of the confirmed breach.
- `hard_risk_pause_breach_type`: `external_monitoring` or `portfolio_risk_policy`.
- `hard_risk_pause_details`: compact JSON payload with mode, live flag, equity,
  positions, symbol notionals, risk snapshot, and active risk-policy caps.

When a pause is persisted, `is_active` is also set to false and maintenance
resume metadata is cleared. Startup restore, `/start_demo`, `/start_live`,
`/continue_demo`, and `/continue_live` all refuse to resume while
`hard_risk_paused` is true.

## Operational Clear

There is intentionally no Telegram clear command in Phase 1. After an operator
has inspected the breach and decided trading may resume, clear the persisted
pause directly in the DB:

```sql
UPDATE user_context
SET hard_risk_paused = 0,
    hard_risk_pause_reason = NULL,
    hard_risk_pause_triggered_at = NULL,
    hard_risk_pause_breach_type = NULL,
    hard_risk_pause_details = NULL
WHERE telegram_id = :telegram_id;
```

Leave `is_active` false. The user or operator should restart trading explicitly
with `/start_demo`, `/start_live`, or an appropriate maintenance continue command
after the pause has been cleared.
