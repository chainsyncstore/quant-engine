# Model Recovery Data Connectivity Spec

## Purpose

Fix the current blocker preventing model-quality recovery: 4arm cannot reliably resolve or reach Binance Futures market-data endpoints, so it cannot build a fresh multi-symbol training snapshot. Do not trigger another retrain until this spec is complete.

## Background

On 2026-06-26 the live `quant_retrain` container already attempted retraining and failed all quality gates:

- `2h` development accuracy: `0.5242`, threshold `0.6000`
- `4h` development accuracy: `0.5216`, threshold `0.6000`
- `8h` development accuracy: `0.5437`, threshold `0.6000`
- final result: no model passed validation, no candidate registered

Attempts to build a fresh 4arm recovery snapshot failed because Python requests to:

```text
https://fapi.binance.com/fapi/v1/time
```

raised:

```text
Temporary failure in name resolution
```

The correct next step is to restore reliable market-data connectivity, build a fresh dataset snapshot, and run the recovery experiment. Do not force a retrain against stale or unavailable data.

## Non-Goals

Do not:

- enable paper trading
- enable live trading
- clear hard-risk breach state
- flatten or open positions
- promote any model
- lower model quality thresholds
- write to production active pointers
- append production promotion events
- edit secrets, API keys, bot tokens, chat IDs, or private credentials
- commit hostnames, private IPs, account IDs, or Tailscale details into docs

## Required Reading

Before editing code, inspect:

- `quant/data/binance_client.py`
- `quant_v2/data/multi_symbol_dataset.py`
- `quant_v2/research/build_universe_snapshot.py`
- `quant_v2/research/model_quality_recovery.py`
- `quant_v2/research/model_recovery_experiments.py`
- `scripts/run_model_recovery_experiments.py`
- `docs/model_quality/README.md`

If `quant_v2/research/model_recovery_experiments.py` or `scripts/run_model_recovery_experiments.py` is not present on 4arm, do not patch production in place. Use the canonical repo implementation and deploy through the normal reviewed path, or run the experiment locally against a copied snapshot.

## Phase 1: Connectivity Diagnosis

Run these checks on 4arm host and inside `quant_retrain`:

```bash
getent hosts fapi.binance.com
python3 - <<'PY'
import socket, urllib.request
for host in ["fapi.binance.com", "api.binance.com", "google.com"]:
    try:
        print(host, socket.getaddrinfo(host, 443)[:1])
    except Exception as exc:
        print(host, type(exc).__name__, exc)
try:
    print(urllib.request.urlopen("https://fapi.binance.com/fapi/v1/time", timeout=10).read().decode())
except Exception as exc:
    print("urlopen", type(exc).__name__, exc)
PY
docker exec quant_retrain python - <<'PY'
import socket, urllib.request
for host in ["fapi.binance.com", "api.binance.com", "google.com"]:
    try:
        print(host, socket.getaddrinfo(host, 443)[:1])
    except Exception as exc:
        print(host, type(exc).__name__, exc)
try:
    print(urllib.request.urlopen("https://fapi.binance.com/fapi/v1/time", timeout=10).read().decode())
except Exception as exc:
    print("urlopen", type(exc).__name__, exc)
PY
```

Acceptance:

- host resolves `fapi.binance.com`
- `quant_retrain` resolves `fapi.binance.com`
- host Python can fetch `/fapi/v1/time`
- container Python can fetch `/fapi/v1/time`
- failures are recorded with exact command output

## Phase 2: DNS/Network Fix

If host or container DNS fails, apply the least invasive fix first.

Preferred order:

1. Restart system DNS resolver:

```bash
sudo systemctl restart systemd-resolved
resolvectl flush-caches || true
```

2. Verify `/etc/resolv.conf` points to the systemd stub or a valid resolver.

3. Restart only affected containers after host DNS works:

```bash
docker compose -f docker-compose.yml -f docker-compose.prod.yml restart quant_retrain
```

4. If Docker DNS remains broken while host DNS works, inspect Docker daemon DNS configuration and container `/etc/resolv.conf`. Do not change it blindly; record the observed resolver values first.

Acceptance:

- host and `quant_retrain` connectivity checks from Phase 1 pass twice, at least 60 seconds apart
- `docker logs --tail 100 quant_retrain` shows no new Binance DNS failure after the fix

## Phase 3: Fresh Snapshot Build

After connectivity passes, build a fresh recovery snapshot. First use the production universe and 6 months of history:

```bash
cd /home/admin-4arm/hypothesis-research-engine
PYTHONPATH=. .venv/bin/python -m quant_v2.research.build_universe_snapshot \
  --months 6 \
  --dataset-name model_recovery_real_1h_$(date -u +%Y%m%d)
```

If the full universe is too slow, build a bounded core-universe snapshot:

```bash
PYTHONPATH=. .venv/bin/python -m quant_v2.research.build_universe_snapshot \
  --months 6 \
  --symbols BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT \
  --dataset-name model_recovery_core_1h_$(date -u +%Y%m%d)
```

Do not use `--no-funding` or `--no-open-interest` unless the full fetch fails specifically on those side-channel endpoints. If those flags are used, record that the run is OHLCV-only and cannot be treated as final production evidence.

Acceptance:

- snapshot parquet exists under the expected dataset snapshot directory
- manifest exists next to it
- manifest records requested symbols, fetched symbols, interval, start, end, and row count
- dataset has at least two symbols
- dataset has enough monthly periods for temporal validation and holdout

## Phase 4: Recovery Experiment

Run the validated model recovery experiment against the fresh snapshot. Required safety flag:

```bash
python scripts/run_model_recovery_experiments.py \
  --snapshot-path <fresh_snapshot_parquet> \
  --output-root models/experiments/model_recovery \
  --docs-output-dir docs/model_quality \
  --max-candidates 12 \
  --no-production-registry
```

If running on 4arm is not possible because the runner has not been deployed, copy the snapshot and manifest to the Windows repo and run the same command locally. Do not patch ad hoc code into production just to run the experiment.

Acceptance:

- command exits `0`
- `experiment_summary.json` exists
- `experiment_summary.md` exists
- `latest_experiment_summary.md` exists
- candidate directories exist for evaluated candidates
- no production active pointer changed
- no production registry event was appended

## Decision Rules

If the experiment says:

```text
recommendation=remain_no_trade
```

then:

- do not retrain manually
- keep no-trade posture
- inspect label audit, feature coverage, and benchmark replay
- open a follow-up spec for label/feature research

If the experiment says:

```text
recommendation=paper_quarantine_candidate
```

then:

- do not activate production trading
- register or stage the candidate only as paper quarantine if a later spec explicitly allows it
- run paper soak before live resumption

## Validation Commands

Run:

```bash
python -m py_compile quant_v2/research/model_recovery_experiments.py scripts/run_model_recovery_experiments.py
python -m pytest tests/quant_v2/test_model_recovery_experiments.py tests/quant_v2/test_model_quality_recovery.py tests/quant_v2/test_scheduled_retrain_candidates.py -q
```

If tests cannot run on 4arm, run them locally against the canonical repo and record the exact reason they were skipped remotely.

## End Note Format

Finish with:

```text
Implementation complete: yes/no

Connectivity:
- host DNS/API: pass/fail
- quant_retrain DNS/API: pass/fail

Snapshot:
- path:
- symbols:
- rows:
- start:
- end:

Recovery experiment:
- run_id:
- recommendation:
- evaluated_candidates:
- passed_candidates:
- selected_candidate_id:

Production safety:
- active pointer changed: yes/no
- registry event appended: yes/no
- paper/live trading enabled: yes/no

Next action:
- remain_no_trade / paper_quarantine_candidate / blocked

Notes:
- <risks, failures, or follow-up>
```
