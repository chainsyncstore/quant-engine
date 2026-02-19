"""
Live signal generator for paper trading.

Loads production models and periodically fetches new bars from Capital.com.
For each new bar, it computes features, detects the current regime,
and generates a BUY/SELL/HOLD signal based on regime-gated thresholds.

Includes:
- Position sizing via Kelly criterion (quarter-Kelly, 2% cap)
- Risk guardrails (daily loss limit, consecutive loss breaker, max trades)

Usage:
    python -m quant.live.signal_generator --model-dir models/production/model_XXXXXX
    python -m quant.live.signal_generator --model-dir models/production/model_XXXXXX --once
"""

from __future__ import annotations

import argparse
import json
import logging
import signal
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from quant.config import get_research_config, CapitalAPIConfig
from quant.risk.volatility_guard import VolatilityGuard
from quant.risk.volatility_guard import VolatilityGuard
from quant.data.capital_client import CapitalClient
from quant.data.session_filter import filter_sessions
from quant.features.pipeline import build_features, get_feature_columns
from quant.models.trainer import TrainedModel, load_model
from quant.models.predictor import predict_proba
from quant.regime import gmm_regime
from quant.regime.gmm_regime import RegimeModel
from quant.risk.position_sizing import compute_position_size
from quant.risk.guardrails import RiskGuardrails

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# How many recent bars we need for feature computation (warmup)
WARMUP_BARS = 200
# Lookback window for regime model context
REGIME_LOOKBACK = 500


class SignalGenerator:
    """Live signal generator with regime-gated trading, position sizing, and risk guardrails."""

    def __init__(self, model_dir: Path, capital: float = 10000.0, horizon: int = 10, api_config: Optional[CapitalAPIConfig] = None):
        self.model_dir = model_dir
        self.capital = capital
        self.horizon = horizon
        self.api_config = api_config

        # Load config
        with open(model_dir / "config.json") as f:
            self.config = json.load(f)

        # Validate horizon
        available_horizons = self.config.get("horizons", [])
        if available_horizons and self.horizon not in available_horizons:
            logger.warning(
                "Horizon %dm not found in config horizons %s. Proceeding anyway...",
                self.horizon, available_horizons
            )

        self.feature_cols: list = self.config["feature_cols"]
        self.spread: float = self.config["spread"]

        # Parse regime config for specific horizon
        # Config structure: "regime_config": {"10": {"0": {...}, ...}}
        h_str = str(self.horizon)
        
        # Handle both old (single) and new (multi) config structures
        if "regime_config" in self.config and h_str in self.config["regime_config"]:
             # New structure
            self.regime_config = {
                int(k): v for k, v in self.config["regime_config"][h_str].items()
            }
            self.regime_thresholds = {
                int(k): v for k, v in self.config["regime_thresholds"][h_str].items()
            }
        else:
            # Fallback to old structure or simple dict
            self.regime_config = {
                int(k): v for k, v in self.config.get("regime_config", {}).items()
            }
            self.regime_thresholds = {
                int(k): v for k, v in self.config.get("regime_thresholds", {}).items()
            }

        self.tradeable_regimes = {
            r for r, cfg in self.regime_config.items() if cfg.get("tradeable", False)
        }

        # Load models
        model_path = model_dir / f"model_{self.horizon}m.joblib"
        if not model_path.exists():
             # Fallback for old single-model name
             fallback = model_dir / "lgbm_model.joblib"
             if fallback.exists():
                 model_path = fallback
             else:
                 raise FileNotFoundError(f"Model not found: {model_path}")

        logger.info("Loading LightGBM model from %s", model_path)
        self.model: TrainedModel = load_model(model_path)

        logger.info("Loading regime model from %s", model_dir)
        self.regime_model: RegimeModel = gmm_regime.load_model(
            model_dir / "regime_model.joblib"
        )

        # API client
        self.client = CapitalClient(config=self.api_config)
        self._authenticated = False

        # Signal log
        self.signal_log: list = []
        self.last_processed_ts = None

        # Risk guardrails
        self.guardrails = RiskGuardrails(
            max_daily_loss=0.02,          # 2% max daily loss
            max_consecutive_losses=3,     # Circuit breaker after 3 losses
            max_daily_trades=10,          # Max 10 trades per day
            cooldown_minutes=30,          # 30 min cooldown after breaker
        )
        self.guardrails.initialize(capital)

        logger.info(
            "Signal generator initialized: horizon=%dm, capital=$%.0f, tradeable regimes=%s",
            self.horizon,
            self.capital,
            self.tradeable_regimes,
        )
        for r, cfg in sorted(self.regime_config.items()):
            status = "✅ TRADE" if cfg.get("tradeable") else "❌ SKIP"
            logger.info(
                "  Regime %d: %s | thresh=%.2f | EV=%.6f | WR=%.1f%%",
                r, status, cfg.get("threshold", 0.0), cfg.get("ev", 0.0), cfg.get("win_rate", 0.0) * 100,
            )

    def _ensure_authenticated(self) -> None:
        if not self._authenticated:
            self.client.authenticate()
            self._authenticated = True

    def fetch_recent_bars(self, n_bars: int = 600) -> pd.DataFrame:
        """Fetch recent bars for feature computation."""
        self._ensure_authenticated()
        date_to = datetime.now(timezone.utc)
        # Fetch fixed 5 days history (plenty for warmup/features, avoids API pagination lag)
        date_from = date_to - timedelta(days=5)

        df = self.client.fetch_historical(date_from, date_to)
        if df.empty:
            raise RuntimeError("No data received from API")

        return df

    def _compute_position_size(self, regime: int) -> dict:
        """Compute position size using Kelly criterion for the given regime."""
        cfg = self.regime_config.get(regime, {})
        win_rate = cfg.get("win_rate", 0.5)
        ev = cfg.get("ev", 0.0)

        # Estimate avg win/loss from EV and win rate
        # EV = WR * avg_win - (1-WR) * avg_loss
        # Using spread as proxy for avg_loss
        avg_loss = self.spread * 2  # conservative estimate
        if win_rate > 0:
            avg_win = (ev + (1 - win_rate) * avg_loss) / win_rate
        else:
            avg_win = 0.0

        pos = compute_position_size(
            capital=self.capital,
            win_rate=win_rate,
            avg_win=max(avg_win, 0),
            avg_loss=avg_loss,
            kelly_divisor=4.0,        # Quarter-Kelly for safety
            max_risk_fraction=0.02,   # 2% max risk per trade
        )

        return {
            "lot_size": pos.units,
            "risk_fraction": round(pos.fraction, 4),
            "kelly_raw": round(pos.kelly_raw, 4),
            "kelly_capped": round(pos.kelly_capped, 4),
            "reason": pos.reason,
        }

    def generate_signal(self, df: pd.DataFrame) -> dict:
        """
        Generate a trading signal from recent bar data.

        Returns:
            Dict with signal, probability, regime, threshold, position sizing,
            and risk guardrail status.
        """
        # Feature engineering
        df_filtered = filter_sessions(df)
        df_features = build_features(df_filtered)
        feature_cols = get_feature_columns(df_features)

        # Verify feature alignment
        missing = set(self.feature_cols) - set(feature_cols)
        if missing:
            logger.warning("Missing features: %s — using available subset", missing)

        # Get the latest bar
        latest = df_features.iloc[[-1]]
        latest_ts = latest.index[0]
        close_price = float(latest["close"].iloc[0])

        # --- Volatility Guard ---
        # Initialize and fit on the full history available in df_features
        if "realized_vol_5" in df_features.columns:
            vol_guard = VolatilityGuard(percentile=0.95)
            # Use all available data to estimate the percentile
            vol_guard.fit(df_features, vol_col="realized_vol_5")
            
            # Check most recent bar
            last_vol = df_features["realized_vol_5"].iloc[-1]
            if not vol_guard.check(last_vol):
                msg = f"⛔ VOLATILITY LOCKOUT: Current={last_vol:.5f} > Thresh={vol_guard.threshold:.5f}"
                logger.warning(msg)
                
                # Return HOLD signal immediately
                risk_status = self.guardrails.get_status()
                result = {
                    "timestamp": str(latest_ts),
                    "close_price": close_price,
                    "signal": "HOLD",
                    "probability": 0.0,
                    "regime": -1, # Unknown/Skipped
                    "regime_probability": 0.0,
                    "regime_tradeable": False,
                    "threshold": 0.0,
                    "reason": msg,
                    "horizon": self.horizon,
                    "position": {},
                    "risk_status": risk_status,
                }
                self.signal_log.append(result)
                return result

        # Detect regime
        labels, probas = gmm_regime.predict(self.regime_model, latest)
        current_regime = int(labels[0])
        regime_prob = float(probas[0].max())

        # Check if regime is tradeable
        regime_tradeable = current_regime in self.tradeable_regimes
        threshold = self.regime_thresholds.get(current_regime, 0.5)

        # Generate prediction
        X = latest[self.feature_cols]
        proba = float(predict_proba(self.model, X)[0])

        # Check risk guardrails
        can_trade, risk_reason = self.guardrails.can_trade()

        # Determine signal
        if not can_trade:
            signal_type = "HOLD"
            reason = f"Risk guardrail: {risk_reason}"
        elif not regime_tradeable:
            signal_type = "HOLD"
            reason = f"Regime {current_regime} has negative historical EV"
        elif proba >= threshold:
            signal_type = "BUY"
            reason = f"P(up)={proba:.3f} >= thresh={threshold:.2f}"
            
            # Additional check: Signal must match logic (prob > 0.5 for BUY)
            # Threshold > 0.5 implied.
            
        elif proba <= (1 - threshold):
            # Only enable SHORT if 10m strategy supports it. 
            # Our current strategy is primarily directional prediction.
            # If 1-threshold < 0.5, it means selling on low prob of UP.
            # Assumption: Model P(UP) < 0.25 -> P(DOWN) > 0.75
            signal_type = "SELL"
            reason = f"P(up)={proba:.3f} <= {1-threshold:.2f}"
        else:
            signal_type = "HOLD"
            reason = f"P(up)={proba:.3f} below threshold {threshold:.2f}"

        # Compute position size (only for actionable signals)
        position = {}
        if signal_type in ("BUY", "SELL"):
            position = self._compute_position_size(current_regime)

        risk_status = self.guardrails.get_status()

        result = {
            "timestamp": str(latest_ts),
            "close_price": close_price,
            "signal": signal_type,
            "probability": round(proba, 4),
            "regime": current_regime,
            "regime_probability": round(regime_prob, 4),
            "regime_tradeable": regime_tradeable,
            "threshold": threshold,
            "reason": reason,
            "horizon": self.horizon,
            "position": position,
            "risk_status": risk_status,
        }

        self.signal_log.append(result)
        return result

    def record_trade_result(self, pnl: float, signal_type: str, regime: int, proba: float) -> None:
        """Record a completed trade for risk tracking."""
        from quant.risk.guardrails import TradeRecord
        trade = TradeRecord(
            timestamp=datetime.now(timezone.utc),
            pnl=pnl,
            signal=signal_type,
            regime=regime,
            probability=proba,
        )
        self.guardrails.record_trade(trade)

    def run_once(self) -> Optional[dict]:
        """Fetch data and generate a single signal."""
        logger.info("Fetching recent bars...")
        df = self.fetch_recent_bars()
        
        latest_ts = df.index[-1]
        if self.last_processed_ts == latest_ts:
            logger.info("Data stale (ts=%s). Skipping signal generation.", latest_ts)
            return None

        self.last_processed_ts = latest_ts
        logger.info("Received %d bars, latest: %s", len(df), latest_ts)

        sig = self.generate_signal(df)

        # Pretty print
        emoji = {"BUY": "🟢", "SELL": "🔴", "HOLD": "⚪"}.get(sig["signal"], "❓")
        logger.info(
            "%s SIGNAL: %s @ %.5f | P(up)=%.3f | Regime=%d (%s) | Thresh=%.2f",
            emoji,
            sig["signal"],
            sig["close_price"],
            sig["probability"],
            sig["regime"],
            "tradeable" if sig["regime_tradeable"] else "SKIP",
            sig["threshold"],
        )
        logger.info("  Reason: %s", sig["reason"])

        if sig["position"]:
            logger.info(
                "  Position: %.2f lots | Kelly=%.3f | Risk=%.1f%%",
                sig["position"]["lot_size"],
                sig["position"]["kelly_raw"],
                sig["position"]["risk_fraction"] * 100,
            )

        risk = sig["risk_status"]
        logger.info(
            "  Risk: %d trades today | consec_losses=%d | can_trade=%s",
            risk["trades_today"],
            risk["consecutive_losses"],
            risk["can_trade"],
        )

        return sig

    def run_loop(self, interval_seconds: int = 60) -> None:
        """
        Run continuous signal generation loop.

        Args:
            interval_seconds: Seconds between signal checks (default: 60 = 1 bar).
        """
        logger.info(
            "Starting signal loop (interval=%ds). Press Ctrl+C to stop.",
            interval_seconds,
        )

        running = True

        def stop_handler(signum, frame):
            nonlocal running
            running = False
            logger.info("Stopping signal loop...")

        signal.signal(signal.SIGINT, stop_handler)
        signal.signal(signal.SIGTERM, stop_handler)

        while running:
            try:
                result = self.run_once()

                # Save signal log periodically
                if len(self.signal_log) % 10 == 0:
                    self._save_log()

            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error("Signal generation error: %s", e)
                # Don't force re-auth on every error, maybe network
                # self._authenticated = False  
                # But if connection dropped, maybe re-auth helps? 
                # Keeping it for robustness manually if needed.

            if running:
                time.sleep(interval_seconds)

        self._save_log()
        logger.info("Signal loop stopped. %d signals generated.", len(self.signal_log))

    def _save_log(self) -> None:
        """Save signal log to disk."""
        log_path = self.model_dir / "signal_log.json"
        with open(log_path, "w") as f:
            json.dump(self.signal_log, f, indent=2)
        logger.info("Signal log saved: %s (%d entries)", log_path, len(self.signal_log))


def main() -> None:
    parser = argparse.ArgumentParser(description="Live Signal Generator")
    parser.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help="Path to production model directory",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Generate a single signal and exit",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=60,
        help="Seconds between signals (default: 60)",
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=10000.0,
        help="Account capital in USD (default: 10000)",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=10,
        help="Prediction horizon in minutes (default: 10)",
    )
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        logger.error("Model directory not found: %s", model_dir)
        sys.exit(1)

    gen = SignalGenerator(model_dir, capital=args.capital, horizon=args.horizon)

    if args.once:
        gen.run_once()
    else:
        gen.run_loop(interval_seconds=args.interval)


if __name__ == "__main__":
    main()
