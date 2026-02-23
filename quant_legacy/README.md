---
description: Legacy runtime fallback notes
---

# quant_legacy fallback policy

Legacy runtime modules are retained only as an explicit fallback path.

## Primary runtime
- Default execution backend is v2 (`v2_memory` unless overridden).
- `quant_v2/*` is the primary runtime path for start/route/stats flows.

## Legacy fallback activation (explicit)
To enable legacy runtime intentionally:

1. Set `BOT_ALLOW_LEGACY_RUNTIME=1`
2. Set `BOT_EXECUTION_BACKEND=v1` (or `v1_legacy`)

Without `BOT_ALLOW_LEGACY_RUNTIME=1`, legacy backend requests are downgraded to `v2_memory`.

## Decommission checklist
- [x] Primary runtime defaults to v2.
- [x] Legacy runtime requires explicit opt-in.
- [x] Legacy fallback behavior is documented.
- [ ] Physical module archive/move can be performed after operational sign-off.
