"""
Diagnostic script: measures how often the live ML model would have fired BUY/SELL
over the last N hours, showing regime distribution and threshold analysis.

Run inside the Docker container:
    docker exec -it quant_execution python /app/scratch/test_inference.py

Or locally (needs model files):
    python scratch/test_inference.py
"""

import os
import sys
import pathlib
import pandas as pd
from datetime import datetime, timezone, timedelta

# ── resolve model dir ────────────────────────────────────────────────────────
# BOT_MODEL_REGISTRY_ROOT is what the live container uses (points to the registry subdir).
# Fall back to BOT_MODEL_ROOT/registry, then bare MODEL_ROOT.
_model_root = os.getenv("BOT_MODEL_ROOT", "/app/models/production")
MODEL_DIR = pathlib.Path(
    os.getenv("BOT_MODEL_REGISTRY_ROOT", os.path.join(_model_root, "registry"))
)
SYMBOLS = os.getenv("BOT_DIAG_SYMBOLS", "BTCUSDT,ETHUSDT").split(",")
HISTORY_HOURS = int(os.getenv("BOT_DIAG_HOURS", "168"))   # 7 days default
HORIZON_BARS  = int(os.getenv("BOT_DIAG_HORIZON", "4"))
INTERVAL      = "1h"


def banner(msg):
    print("\n" + "=" * 60)
    print(f"  {msg}")
    print("=" * 60)


def main():
    banner("INFERENCE DIAGNOSTIC")
    print(f"Model dir  : {MODEL_DIR}")
    print(f"Symbols    : {SYMBOLS}")
    print(f"History    : {HISTORY_HOURS}h")
    print(f"Horizon    : {HORIZON_BARS} bars")

    # ── 1. Model registry check ──────────────────────────────────────────────
    banner("1. Model Registry")
    try:
        from quant_v2.model_registry import ModelRegistry
        registry = ModelRegistry(MODEL_DIR)
        pointer  = registry.get_active_version()
        if pointer is None:
            print("ERROR: No active model version registered.")
            print("       Run model training or set the active pointer.")
            sys.exit(1)
        artifact_dir = pathlib.Path(pointer.artifact_dir)
        print(f"Active version  : {pointer.version_id}")
        print(f"Artifact dir    : {artifact_dir}")
        print(f"Dir exists      : {artifact_dir.exists()}")
        if artifact_dir.exists():
            files = list(artifact_dir.iterdir())
            print(f"Artifact files  : {[f.name for f in files]}")
    except Exception as exc:
        print(f"Registry error  : {exc}")
        sys.exit(1)

    # ── 2. Load ensemble ─────────────────────────────────────────────────────
    banner("2. Model Load")
    try:
        from quant_v2.models.ensemble import HorizonEnsemble
        ensemble = HorizonEnsemble.from_directory(artifact_dir)
        if ensemble is None or ensemble.horizon_count == 0:
            print("WARNING: HorizonEnsemble not found or empty.")
            ensemble = None
        else:
            print(f"HorizonEnsemble : {ensemble.horizon_count} horizon(s) loaded")
    except Exception as exc:
        print(f"Ensemble load error: {exc}")
        ensemble = None

    # Single-model fallback
    active_model = None
    if ensemble is None:
        try:
            from quant_v2.models.trainer import load_model
            for fname in (f"model_{HORIZON_BARS}m.pkl", f"model_{HORIZON_BARS}m.joblib",
                          "lgbm_model.joblib"):
                candidate = artifact_dir / fname
                if candidate.exists():
                    active_model = load_model(candidate)
                    print(f"Single model    : loaded {candidate.name}")
                    break
            if active_model is None:
                print("ERROR: No usable model file found in artifact directory.")
                sys.exit(1)
        except Exception as exc:
            print(f"Model load error: {exc}")
            sys.exit(1)

    # ── 3. Fetch data ────────────────────────────────────────────────────────
    banner("3. Data Fetch")
    from quant_v2.telebot.signal_manager import V2SignalManager
    manager = V2SignalManager.__new__(V2SignalManager)
    manager.horizon_bars = HORIZON_BARS
    manager.anchor_interval = INTERVAL
    manager.symbols = tuple(SYMBOLS)
    manager._oi_cache = {}
    manager._cached_events = []
    manager._events_fetched_at = None
    manager.active_model = active_model
    manager.horizon_ensemble = ensemble
    manager.full_ensemble = None
    manager._last_model_agreement = None
    from quant_v2.telebot.symbol_scorecard import SymbolScorecard
    manager.scorecard = SymbolScorecard(lookback_hours=72, min_samples=8)

    client = V2SignalManager._default_client_factory({}, False, SYMBOLS[0], INTERVAL)

    date_to   = datetime.now(timezone.utc)
    date_from = date_to - timedelta(hours=HISTORY_HOURS + 10)  # +10 warm-up

    results_per_symbol = {}

    for symbol in SYMBOLS:
        print(f"\nFetching {HISTORY_HOURS}h of {symbol} bars...", flush=True)
        try:
            bars = V2SignalManager._default_fetch_bars(
                V2SignalManager._default_client_factory({}, False, symbol, INTERVAL),
                date_from, date_to, symbol, INTERVAL,
            )
        except Exception as exc:
            print(f"  Fetch failed: {exc}")
            continue

        if bars is None or bars.empty:
            print(f"  No bars returned for {symbol}")
            continue

        print(f"  {len(bars)} bars fetched ({bars.index[0]} → {bars.index[-1]})")

        # ── 4. Per-hour back-simulation ──────────────────────────────────────
        rows = []
        warmup = 50   # minimum bars needed before evaluating
        eval_start = len(bars) - HISTORY_HOURS  # evaluate last N hours only
        eval_start = max(eval_start, warmup)

        for i in range(eval_start, len(bars)):
            subset = bars.iloc[:i+1]
            try:
                payload = manager._build_signal_payload(symbol, subset)
            except Exception as exc:
                rows.append({
                    "ts": subset.index[-1], "signal": "ERROR", "proba": float("nan"),
                    "threshold": float("nan"), "regime": -1, "regime_risk": -1, "reason": str(exc),
                })
                continue

            rows.append({
                "ts":           subset.index[-1],
                "signal":       payload.get("signal", "HOLD"),
                "proba":        payload.get("probability", 0.5),
                "threshold":    payload.get("threshold", 0.65),
                "regime":       payload.get("regime", 3),
                "regime_risk":  1 if payload.get("regime", 3) in (3, 4) else 0,
                "reason":       payload.get("reason", ""),
            })

        results_per_symbol[symbol] = pd.DataFrame(rows)

    # ── 5. Summary report ────────────────────────────────────────────────────
    banner("4. Results Summary")

    for symbol, df in results_per_symbol.items():
        print(f"\n{'─'*50}")
        print(f"Symbol: {symbol}  ({len(df)} evaluated bars)")

        if df.empty:
            print("  No results.")
            continue

        counts = df["signal"].value_counts()
        print(f"  Signal counts   : {dict(counts)}")

        buy_rate  = (df["signal"] == "BUY").mean() * 100
        sell_rate = (df["signal"] == "SELL").mean() * 100
        print(f"  BUY rate        : {buy_rate:.1f}%")
        print(f"  SELL rate       : {sell_rate:.1f}%")

        proba_valid = df["proba"].dropna()
        if not proba_valid.empty:
            print(f"  Proba (mean)    : {proba_valid.mean():.3f}")
            print(f"  Proba (median)  : {proba_valid.median():.3f}")
            print(f"  Proba > 0.65    : {(proba_valid > 0.65).sum()} bars  ({(proba_valid > 0.65).mean()*100:.1f}%)")
            print(f"  Proba > 0.70    : {(proba_valid > 0.70).sum()} bars  ({(proba_valid > 0.70).mean()*100:.1f}%)")
            print(f"  Proba < 0.35    : {(proba_valid < 0.35).sum()} bars  ({(proba_valid < 0.35).mean()*100:.1f}%)")

        regime_counts = df["regime"].value_counts().sort_index()
        print(f"  Regime dist     : {dict(regime_counts)}")
        regime_risk_1 = (df["regime_risk"] == 1).mean() * 100
        print(f"  Regime risk=1   : {regime_risk_1:.1f}%  (threshold inflated to ≥0.75)")

        thresh_valid = df["threshold"].dropna()
        if not thresh_valid.empty:
            print(f"  Threshold (mean): {thresh_valid.mean():.3f}")

        print(f"\n  Recent 24 bars:")
        recent = df.tail(24)[["ts", "signal", "proba", "threshold", "regime", "reason"]]
        for _, row in recent.iterrows():
            proba_str = f"{row['proba']:.3f}" if pd.notna(row['proba']) else "n/a"
            thresh_str = f"{row['threshold']:.2f}" if pd.notna(row['threshold']) else "n/a"
            marker = " <<<" if row["signal"] in ("BUY", "SELL") else ""
            print(f"    [{row['ts']}] {row['signal']:12s} P={proba_str} T={thresh_str} R={row['regime']}  {marker}")

    # ── 6. Threshold diagnosis ────────────────────────────────────────────────
    banner("5. Threshold Diagnosis")
    for symbol, df in results_per_symbol.items():
        if df.empty:
            continue
        proba_valid = df["proba"].dropna()
        if proba_valid.empty:
            continue

        mean_p = proba_valid.mean()
        regime_risk_pct = (df["regime_risk"] == 1).mean() * 100

        print(f"\n{symbol}:")
        if mean_p < 0.55 and mean_p > 0.45:
            print("  DIAGNOSIS: Model probabilities are clustered near 0.50 — likely")
            print("             constant-output or feature pipeline producing flat signals.")
            print("  ACTION: Check model training data, feature coverage, and pipeline.")
        elif regime_risk_pct > 70:
            print(f"  DIAGNOSIS: Regime risk=1 for {regime_risk_pct:.0f}% of bars.")
            print("             Effective threshold inflated to 0.75+ most of the time.")
            print("             Model needs proba > 0.75 to fire — very rare condition.")
            print("  ACTION: Consider lowering BOT_V2_REGIME_RISK_THRESHOLD_BOOST env var,")
            print("          or re-train with regime labels as features so model calibrates.")
        elif buy_rate < 2.0 and sell_rate < 2.0:
            print("  DIAGNOSIS: Model fires BUY/SELL < 2% of bars even with normal thresholds.")
            print("             Signals are extremely conservative.")
            print("  ACTION: Re-evaluate signal generation thresholds or model calibration.")
        else:
            print(f"  OK: Buy rate={buy_rate:.1f}%, Sell rate={sell_rate:.1f}%, mean proba={mean_p:.3f}")

    banner("DONE")


if __name__ == "__main__":
    main()
