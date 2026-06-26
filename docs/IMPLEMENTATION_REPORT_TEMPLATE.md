# Implementation Report Template

Use this template after every production-refactor slice. Do not mark a ledger row verified unless the report has concrete evidence.

### YYYY-MM-DD Pass N Slice Name

Ledger IDs changed:
- `ID` status change and reason.

Files changed:
- `path/to/file`

Safety invariants proven:
- Invariant stated as observable behavior.

Tests added or inverted:
- New or changed test names.
- Unsafe historical expectations that were inverted, if any.

Verification commands:
- `command`

Results:
- Exact pass/fail summary from the command output.

Rollback plan:
- How to disable, revert, or fail closed if this slice misbehaves.

Residual risk:
- Remaining risk or explicit "none beyond listed live blockers".

Live-block status:
- Whether live deployment remains blocked.
- If blocked, name the exact remaining gate IDs.

Minimum standard:

- Reports must name ledger IDs, files, invariants, tests or checks, commands, results, rollback plan, residual risk, and live-block status.
- P0 rows require dated `Verified YYYY-MM-DD` status plus proof before strict live readiness can pass.
- External evidence such as GitHub Actions must include run URL or run ID, commit SHA, and artifact names when relevant.
