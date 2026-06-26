# Native Torch Confirmation Model Roadmap

Created: 2026-06-06

Parent roadmap: `PRODUCTION_REFACTOR_ROADMAP.md`

## Objective

Replace the Chronos-specific second-opinion path with a production-grade native Torch confirmation model workflow. The confirmation model must start in shadow mode, produce auditable source-level predictions, and only become an entry gate after backtest, shadow, dependency, artifact, and deployment evidence prove it improves the current baseline.

This is not a request to turn on a new live gate immediately. Until the gates below are complete, production must continue to run the LightGBM/horizon ensemble path without confirmation-model blocking.

## Current Production Decision

- `torch` is locked in the default runtime as a CPU-only wheel.
- `chronos-forecasting` is not part of the default production image because its current compatible dependency path requires `transformers<5`, which is blocked by dependency-audit advisories.
- Production should keep `BOT_ENABLE_CHRONOS=0`.
- The current allocator intentionally bypasses Chronos constraints when `ModelSourceDetails.chronos_enabled` is false.
- The next production-grade direction is a small native Torch confirmation classifier, not a forced Chronos dependency override.

## Production-Grade Definition

A native Torch confirmation model is production-grade only when these invariants hold:

- It is packaged from locked, audit-clean dependencies and does not require unsafe runtime downloads.
- It loads only from trusted artifact roots with signed or hashed manifests.
- It uses safe Torch serialization formats where practical, preferring `state_dict`, TorchScript, or safetensors-style formats over arbitrary object deserialization.
- It can run on CPU within the deployed instance memory and latency budget.
- It emits source-level probability, direction, uncertainty, version, and freshness metadata.
- It starts in shadow mode and cannot block, dampen, or route live orders until explicitly promoted.
- Agreement logic is generic, not Chronos-named, so future confirmation models can be swapped without changing allocator semantics.
- Entry gating fails closed only when confirmation gating is explicitly enabled and the configured confirmation model is healthy and fresh.
- Tests prove disabled, unavailable, shadow-only, disagreement, and agreement paths.
- Deployment docs explain how to disable the confirmation model quickly without stopping the baseline bot.

## Lead Review Workflow

The lead agent owns final acceptance. Implementation agents produce endnotes; the lead validates each note against the roadmap and either accepts it, rejects it, or asks for correction.

Agent endnotes must use this shape:

```text
END NOTES
Scope reviewed:
- ...

Findings:
- [Severity] Title
  Evidence: file:line
  Why it matters:
  Recommended implementation:
  Tests/checks:
  Risks/tradeoffs:

Implementation notes:
- Files changed or proposed:
- Behavior changed:
- Validation run:

Open questions:
- ...
```

Lead decisions:

- `Accepted`: note is correct and can guide implementation.
- `Accepted with correction`: note is directionally correct but the roadmap or implementation must narrow it.
- `Rejected`: note is inaccurate, unsafe, or outside scope.
- `Deferred`: useful but not needed for this pass.

## Implementation Phases

### Phase 0: Design and Risk Audit

Scope:
- `quant_v2/contracts.py`
- `quant_v2/models/ensemble.py`
- `quant_v2/models/chronos_wrapper.py`
- `quant_v2/telebot/signal_manager.py`
- `quant_v2/portfolio/allocation.py`
- `tests/quant_v2/test_full_ensemble.py`
- `tests/quant_v2/test_portfolio.py`
- model registry and retrain modules

Targets:
- Map every Chronos-specific name, env var, constraint, and log path.
- Decide the generic confirmation contract before code changes.
- Identify data, training, artifact, and deployment requirements for a small Torch model.

Acceptance:
- Lead-approved endnotes identify all files and tests required for the refactor.
- Roadmap is updated before implementation starts.

### Phase 1: Generic Confirmation Contracts

Scope:
- `quant_v2/contracts.py`
- `quant_v2/portfolio/allocation.py`
- tests under `tests/quant_v2`

Targets:
- Replace Chronos-specific source fields with generic confirmation fields while preserving backwards compatibility where needed.
- Introduce generic env names such as:
  - `BOT_V2_CONFIRMATION_REQUIRE_AGREEMENT_FOR_ENTRY`
  - `BOT_V2_CONFIRMATION_DISAGREEMENT_MULT`
  - `BOT_V2_CONFIRMATION_EXTREME_CONFIDENCE`
- Keep old Chronos env names as temporary compatibility aliases only if tests prove deterministic precedence.
- Rename allocator constraints from `chronos_*` to `confirmation_*`.
- Preserve the current safe behavior: when confirmation is disabled, confirmation constraints do not apply.

Acceptance:
- Tests cover confirmation disabled, unavailable, disagreement block, disagreement dampen, flatten-only, and extreme-confidence override.
- Existing Chronos-specific tests are either migrated or retained as compatibility tests.
- No live order behavior changes when confirmation is disabled.

### Phase 2: Native Torch Model Interface

Scope:
- new module under `quant_v2/models/`
- `quant_v2/models/ensemble.py`
- `quant_v2/telebot/signal_manager.py`
- tests under `tests/quant_v2`

Targets:
- Add a small Torch confirmation model interface with:
  - deterministic CPU inference
  - `predict_proba` or equivalent source probability
  - bounded uncertainty
  - explicit model version
  - no runtime network downloads
- Keep the model optional and off by default.
- Support shadow-only prediction collection independent of gating.

Acceptance:
- A fake or fixture Torch model can produce confirmation details without importing Chronos.
- Missing/corrupt confirmation artifact fails safe and does not block baseline signals unless explicit gating is enabled.
- Inference tests run without external model downloads.

### Phase 3: Training and Artifact Pipeline

Scope:
- research/retrain modules under `quant_v2/research/`
- model registry and artifact validation
- new tests for artifact manifest and loadability

Targets:
- Train a compact CPU-friendly classifier on existing feature windows and directional labels.
- Store artifacts with manifest fields:
  - model type
  - input feature schema
  - horizon
  - Torch version
  - artifact hashes
  - training dataset snapshot metadata
  - validation metrics
  - promotion eligibility
- Keep artifacts out of git and release archives unless explicitly allowlisted.

Acceptance:
- Training produces a loadable fixture artifact in tests.
- Registry validation rejects missing manifest, schema mismatch, bad hash, unsupported model type, and unsafe paths.
- Promotion cannot activate a confirmation model without loadability and minimum metric evidence.

### Phase 4: Shadow Mode and Metrics

Scope:
- signal manager
- monitoring/drift modules
- tests and docs

Targets:
- Emit confirmation predictions in shadow mode without changing route decisions.
- Record agreement rate, disagreement rate, latency, unavailable rate, and directional hit-rate where labels become available.
- Add operator-visible logs or summaries that do not expose secrets.

Acceptance:
- Shadow mode never changes `StrategySignal.signal`, confidence, or allocation.
- Metrics are bounded, serializable, and test-covered.
- A missing confirmation model produces an explicit unavailable metric, not noisy repeated stack traces.

### Phase 5: Controlled Gating Enablement

Scope:
- allocation/routing tests
- deployment docs
- production readiness checks

Targets:
- Allow confirmation gating only through explicit env/config.
- Require confirmation model health/freshness before using it as a gate.
- Define emergency disable: one env flip returns to baseline LGBM path.

Acceptance:
- Production readiness blocks confirmation gating unless shadow evidence and model artifact checks pass.
- Disabling confirmation returns to baseline behavior without code changes.
- Deployment docs include rollback and disable procedure.

## Initial Implementation Agent Assignments

| Agent | Status | Scope | Expected endnotes |
| --- | --- | --- | --- |
| Explorer A | Complete | Signal generation, contracts, allocation/routing, tests, env/config behavior | Chronos path enters through `FullEnsemble`, `ModelSourceDetails`, signal-manager env construction, and allocator gating. |
| Explorer B | Complete | Training, artifact format, registry, dependency/SBOM, docs/tests | Torch runtime lock is ready; registry/training are horizon-model centric and need typed confirmation artifacts. |
| Worker 1 | Complete | Generic contracts, allocator env aliases, allocator tests | Canonical `confirmation_*` contract and constraints implemented with Chronos aliases retained for compatibility. |
| Worker 2 | Complete | Standalone native Torch confirmation interface and tests | `state_dict` + config loader implemented with strict feature schema and CPU deterministic inference. |
| Worker 3 | Complete | Registry manifest validation for optional confirmation artifacts | Typed `confirmation/config.json` and `confirmation/state_dict.pt` manifest entries validated with hashes, path checks, and activation smoke-load. |
| Explorer C | Complete | Training/export and shadow-metrics implementation map | Smallest safe path is a single 4-bar engineered-feature confirmation trainer, env-gated in scheduled retrain. |

## Lead Decision Ledger

| Date | Agent | Decision | Notes |
| --- | --- | --- | --- |
| 2026-06-06 | Lead | Accepted | Keep Chronos out of default production; implement native Torch confirmation model in shadow mode first. |
| 2026-06-06 | Explorer A | Accepted with correction | Generic contract/allocation migration is required, but first implementation must preserve disabled-confirmation behavior and avoid enabling live gating. Chronos env names may remain as temporary aliases for one release. |
| 2026-06-06 | Explorer B | Accepted with correction | Native Torch model and artifact validation are accepted. Training/registry expansion follows the generic contract and shadow interface; it must not create auto-gating or promotion shortcuts. |
| 2026-06-06 | Worker 1 | Accepted with correction | Generic confirmation contract and allocator migration accepted. Lead added `confirmation_shadow_only` so observational metadata cannot trigger confirmation gates. |
| 2026-06-06 | Worker 2 | Accepted | Native Torch confirmation interface accepted. Full pytest validation requires a Torch-enabled environment; production-image smoke validation passed on `torch 2.12.0+cpu`. |
| 2026-06-06 | Lead | Accepted | Signal manager now loads optional native Torch confirmation artifacts from `confirmation/` only when `BOT_ENABLE_CONFIRMATION=1`, and hard-codes that path as shadow-only. |
| 2026-06-06 | Worker 3 | Accepted with correction | Registry confirmation artifact validation accepted. Lead corrected smoke-load behavior so hash/path scans can run without Torch, while activation with `smoke_load=True` still proves loadability. |
| 2026-06-06 | Explorer C | Accepted with correction | First trainer target is a single 4-bar engineered-feature model, no scaler, `RETRAIN_ENABLE_CONFIRMATION=1`, and failed confirmation export must not break baseline candidate registration. |
| 2026-06-06 | Lead | Accepted | Added native Torch confirmation trainer/export and scheduled retrain integration. Production-image smoke validated export, manifest inclusion, and registry smoke-load on `torch 2.12.0+cpu`. |
| 2026-06-06 | Explorer D | Accepted with correction | Offline agreement/disagreement profitability evidence is required before any active gate. Lead corrected retrain so baseline probability capture runs only when confirmation export is enabled. |
| 2026-06-06 | Lead | Accepted | Added offline holdout profitability evidence for confirmation artifacts: agreement/disagreement PnL, win rate, drawdown, coverage, net after round-trip costs, and `gate_eligible` metadata. |

## Current Implementation Status

Completed in this pass:

- `ModelSourceDetails` now uses canonical `confirmation_*` fields with temporary `chronos_*` aliases.
- Allocator confirmation constraints use `confirmation_*` names and ignore confirmation metadata when disabled or shadow-only.
- `NativeTorchConfirmationModel` loads CPU Torch `state_dict.pt` plus `config.json` with strict feature-schema validation.
- `ConfirmationEnsemble` can emit source-level confirmation metadata without changing primary probability or uncertainty in shadow mode.
- `V2SignalManager` can load an optional `confirmation/` artifact under the active model version when `BOT_ENABLE_CONFIRMATION=1`.
- Model registry manifests can include typed native Torch confirmation entries and validate them during activation.
- Scheduled retrain can optionally export a single 4-bar native Torch confirmation artifact when `RETRAIN_ENABLE_CONFIRMATION=1`.
- Scheduled retrain now records offline holdout profitability evidence for the confirmation artifact when the matching baseline horizon model is available.
- `python -m quant_v2.research.confirmation_shadow_export` can attach a native Torch confirmation artifact to the current active model version for runtime shadow observation without promoting a new baseline model.
- Runtime confirmation remains shadow-only: `BOT_ENABLE_CONFIRMATION=1` loads and reports the source metadata, but the signal manager still constructs `ConfirmationEnsemble(..., shadow_only=True)`.

Still open:

- Runtime shadow metrics persistence after labeled outcomes become available.
- Route-level shadow replay that compares actual routed entries, allocation thresholds, position sizing, and market-cost regimes.
- Production readiness checks that block non-shadow confirmation gating. Current code has no accepted active-gating path.

## Lead Defaults For Implementation

- First model input: engineered feature rows from the existing feature pipeline. Close-price windows are deferred until the feature-row classifier is stable.
- First runtime mode: shadow-only. Confirmation output may be logged and attached to metadata, but it must not alter signal confidence, direction, allocation, or execution.
- First production shadow export command: `python -m quant_v2.research.confirmation_shadow_export`. It writes `confirmation/config.json`, `confirmation/state_dict.pt`, manifest checksums, and registry evidence for the currently active version only.
- Artifact location: optional `confirmation/` subdirectory inside the active model version directory, represented by typed manifest entries.
- Serialization preference: `state_dict` plus explicit architecture/config and feature schema for the first pass; TorchScript can be evaluated later if it simplifies deployment.
- Gating scope: future gating may block or dampen fresh entries only. Exits, flattening, and risk-reducing actions must not be blocked by confirmation disagreement.
- Compatibility: old Chronos env names can be read as aliases for one release, but new code and docs must use `CONFIRMATION` names.
- Shadow evidence before gating: gating remains blocked until a separate lead-reviewed evidence pass defines and satisfies minimum sample count, duration, latency, unavailable-rate, and hit-rate requirements.
- Offline evidence before shadow: the first confirmation artifact must include `profitability.mode=offline_holdout_shadow_evidence`, agreement/disagreement cohorts, round-trip cost bps, coverage, win rate, max drawdown, and `gate_eligible`.
- `gate_eligible=True` is not sufficient by itself to enable real gating. It only permits moving to runtime shadow observation; active gating still requires runtime shadow evidence and a lead-reviewed readiness check.

## Open Questions

- What minimum shadow window is required before confirmation can gate entries: fixed days, cycles, labeled samples, or all three?
- What is the target CPU inference latency budget per symbol?
