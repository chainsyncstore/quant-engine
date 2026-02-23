"""
Live signal generator for Binance-backed crypto paper/live trading.

Loads production models and periodically fetches new bars from Binance.
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
from collections import deque
import json
import logging
import signal
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Deque, Dict, Optional

import numpy as np
import pandas as pd

from quant.config import BinanceAPIConfig
from quant.risk.volatility_guard import VolatilityGuard
from quant.data.binance_client import BinanceClient
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
# Drift monitoring defaults
DRIFT_BASELINE_BARS = 500
DRIFT_Z_SCORE_THRESHOLD = 4.0
DRIFT_MIN_FEATURE_FRACTION = 0.15
CONFIDENCE_WINDOW = 24
CONFIDENCE_NEUTRAL_BAND = 0.04
CONFIDENCE_STD_FLOOR = 0.03
DRIFT_ALERT_COOLDOWN_SECONDS = 6 * 3600


class SignalGenerator:
    """Live signal generator with regime-gated trading, position sizing, and risk guardrails."""

    def __init__(
        self,
        model_dir: Path,
        capital: float = 10000.0,
        horizon: int = 10,
        binance_config: Optional[BinanceAPIConfig] = None,
        live: bool = False,
        auto_execute: bool = True,
    ):
        self.model_dir = model_dir
        self.capital = capital
        self.horizon = horizon
        self.binance_config = binance_config
        self.live = live
        self.auto_execute = auto_execute

        # Load config
        with open(model_dir / "config.json") as f:
            self.config = json.load(f)

        # Determine mode from model config
        self.mode = self.config.get("mode", "crypto")
        if self.mode != "crypto":
            raise RuntimeError(
                f"Model at {model_dir} is mode={self.mode!r}. Only crypto models are supported."
            )
        self.taker_fee_rate = self.config.get("taker_fee_rate", 0.0004)

        # Validate horizon
        available_horizons = self.config.get("horizons", [])
        if available_horizons and self.horizon not in available_horizons:
            logger.warning(
                "Horizon %dh not found in config horizons %s. Proceeding anyway...",
                self.horizon, available_horizons
            )

        self.feature_cols: list = self.config["feature_cols"]
        self.spread: float = self.config["spread"]

        # Parse regime config for specific horizon
        h_str = str(self.horizon)

        if "regime_config" in self.config and h_str in self.config["regime_config"]:
            self.regime_config = {
                int(k): v for k, v in self.config["regime_config"][h_str].items()
            }
            self.regime_thresholds = {
                int(k): v for k, v in self.config["regime_thresholds"][h_str].items()
            }
        else:
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

        # API client — Binance only
        self.binance_client = BinanceClient(config=self.binance_config)
        self._authenticated = False

        # Signal log
        self.signal_log: list = []
        self.last_processed_ts = None
        self._last_signal_time: float = 0.0  # epoch time of last signal generation

        # Paper balance tracking
        self.initial_capital = capital
        self.paper_balance = capital

        # Position management
        self._open_position: Optional[Dict] = None  # tracks live/paper open position
        self.stop_loss_pct = 0.02   # 2% hard stop loss

        # Win rate tracking
        self.evaluated_count = 0
        self.win_count = 0
        self.loss_count = 0

        # Risk guardrails
        self.guardrails = RiskGuardrails(
            max_daily_loss=0.02,          # 2% max daily loss
            max_consecutive_losses=3,     # Circuit breaker after 3 losses
            max_daily_trades=10,          # Max 10 trades per day
            cooldown_minutes=30,          # 30 min cooldown after breaker
        )
        self.guardrails.initialize(capital)

        # Drift monitoring state
        self._feature_baseline_mean: Optional[pd.Series] = None
        self._feature_baseline_std: Optional[pd.Series] = None
        self._proba_history: Deque[float] = deque(maxlen=CONFIDENCE_WINDOW)
        self._last_drift_alert_epoch: float = 0.0

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
        if self.live and not self._authenticated:
            # Live crypto mode: verify Binance API credentials
            self.binance_client.authenticate()
            self._authenticated = True
        else:
            # Paper mode: read-only endpoints don't need auth
            self._authenticated = True

    def fetch_recent_bars(self, n_bars: int = 800) -> pd.DataFrame:
        """Fetch recent bars for feature computation."""
        self._ensure_authenticated()
        date_to = datetime.now(timezone.utc)

        # Fetch n bars of 1H data from Binance
        date_from = date_to - timedelta(hours=n_bars)

        ohlcv = self.binance_client.fetch_historical(date_from, date_to)
        if ohlcv.empty:
            raise RuntimeError("No OHLCV data from Binance")

        # Fetch supplementary data for feature computation
        funding = self.binance_client.fetch_funding_rates(date_from, date_to)
        oi = self.binance_client.fetch_open_interest(date_from, date_to)
        df = BinanceClient.merge_supplementary(ohlcv, funding, oi)

        if df.empty:
            raise RuntimeError("No data received from API")

        return df

    def _compute_position_size(self, regime: int, close_price: float = 0.0) -> dict:
        """Compute position size using Kelly criterion for the given regime."""
        cfg = self.regime_config.get(regime, {})
        win_rate = cfg.get("win_rate", 0.5)
        ev = cfg.get("ev", 0.0)

        # For crypto: avg_loss based on fee rate + typical adverse move
        avg_loss = close_price * self.taker_fee_rate * 2 + close_price * 0.005
        if win_rate > 0 and avg_loss > 0:
            avg_win = (ev + (1 - win_rate) * avg_loss) / win_rate
        else:
            avg_win = 0.0

        pos = compute_position_size(
            capital=self.capital,
            win_rate=win_rate,
            avg_win=max(avg_win, 0),
            avg_loss=max(avg_loss, 1e-8),
            kelly_divisor=4.0,
            max_risk_fraction=0.02,
        )

        return {
            "lot_size": pos.units,
            "risk_fraction": round(pos.fraction, 4),
            "kelly_raw": round(pos.kelly_raw, 4),
            "kelly_capped": round(pos.kelly_capped, 4),
            "reason": pos.reason,
        }

    def _init_drift_baseline(self, df_features: pd.DataFrame) -> None:
        """Initialize feature distribution baseline from recent history."""
        if self._feature_baseline_mean is not None and self._feature_baseline_std is not None:
            return

        available = [c for c in self.feature_cols if c in df_features.columns]
        if not available:
            return

        baseline = df_features[available].tail(min(len(df_features), DRIFT_BASELINE_BARS))
        if len(baseline) < 100:
            return

        baseline_mean = baseline.mean()
        baseline_std = baseline.std().replace(0.0, np.nan)
        if int(baseline_std.notna().sum()) == 0:
            return

        self._feature_baseline_mean = baseline_mean
        self._feature_baseline_std = baseline_std

    def _check_feature_drift(self, latest: pd.DataFrame) -> str | None:
        """Return feature-drift alert message when latest bar is out-of-distribution."""
        if self._feature_baseline_mean is None or self._feature_baseline_std is None:
            return None

        latest_row = latest.reindex(columns=self._feature_baseline_mean.index).iloc[0]
        z_scores = ((latest_row - self._feature_baseline_mean).abs() / self._feature_baseline_std)
        z_scores = z_scores.replace([np.inf, -np.inf], np.nan).dropna()
        if z_scores.empty:
            return None

        drifted = z_scores[z_scores >= DRIFT_Z_SCORE_THRESHOLD]
        drift_fraction = float(len(drifted) / len(z_scores))
        if drift_fraction < DRIFT_MIN_FEATURE_FRACTION:
            return None

        top_outliers = ", ".join(
            f"{col}:{float(val):.1f}σ"
            for col, val in drifted.sort_values(ascending=False).head(3).items()
        )
        return (
            "Feature drift detected "
            f"({drift_fraction * 100:.1f}% features > {DRIFT_Z_SCORE_THRESHOLD:.1f}σ). "
            f"Top outliers: {top_outliers}"
        )

    def _check_confidence_drift(self, proba: float) -> str | None:
        """Return confidence-collapse alert when probabilities cluster near 0.5."""
        self._proba_history.append(float(proba))
        if len(self._proba_history) < CONFIDENCE_WINDOW:
            return None

        history = np.array(self._proba_history, dtype=float)
        mean_proba = float(np.mean(history))
        std_proba = float(np.std(history))
        if abs(mean_proba - 0.5) <= CONFIDENCE_NEUTRAL_BAND and std_proba <= CONFIDENCE_STD_FLOOR:
            return (
                "Confidence drift detected "
                f"(mean P(up)={mean_proba:.3f}, std={std_proba:.3f}, window={len(history)})."
            )
        return None

    def _check_drift_alert(self, latest: pd.DataFrame, proba: float) -> str | None:
        """Combine drift signals and rate-limit alerts."""
        feature_alert = self._check_feature_drift(latest)
        confidence_alert = self._check_confidence_drift(proba)

        alerts = [a for a in (feature_alert, confidence_alert) if a]
        if not alerts:
            return None

        now = time.time()
        if (now - self._last_drift_alert_epoch) < DRIFT_ALERT_COOLDOWN_SECONDS:
            return None

        self._last_drift_alert_epoch = now
        return " | ".join(alerts)

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
        self._init_drift_baseline(df_features)

        # Verify feature alignment — fill missing columns with 0
        missing = set(self.feature_cols) - set(feature_cols)
        if missing:
            logger.warning("Missing features: %s — filling with 0", missing)
            for col in missing:
                df_features[col] = 0.0

        # Get the latest bar
        latest = df_features.iloc[[-1]]
        latest_ts = latest.index[0]
        close_price = float(latest["close"].iloc[0])

        # --- Volatility Guard ---
        # Initialize and fit on the full history available in df_features
        if "realized_vol_5" in df_features.columns:
            vol_guard = VolatilityGuard(percentile=0.99, min_samples=1000)
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
                    "drift_alert": False,
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

        drift_reason = self._check_drift_alert(latest, proba)
        if drift_reason:
            logger.warning("%s", drift_reason)
            risk_status = self.guardrails.get_status()
            result = {
                "timestamp": str(latest_ts),
                "close_price": close_price,
                "signal": "DRIFT_ALERT",
                "probability": round(proba, 4),
                "regime": current_regime,
                "regime_probability": round(regime_prob, 4),
                "regime_tradeable": False,
                "threshold": threshold,
                "reason": drift_reason,
                "horizon": self.horizon,
                "position": {},
                "risk_status": risk_status,
                "drift_alert": True,
            }
            self.signal_log.append(result)
            return result

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
            position = self._compute_position_size(current_regime, close_price=close_price)

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
            "drift_alert": False,
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

    def _open_new_position(self, signal: dict) -> None:
        """Track a newly opened position for auto-close and stop loss."""
        self._open_position = {
            "signal": signal["signal"],
            "entry_price": signal["close_price"],
            "entry_time": datetime.now(timezone.utc),
            "size": signal.get("position", {}).get("lot_size", 0),
            "risk_fraction": signal.get("position", {}).get("risk_fraction", 0.02),
        }
        logger.info(
            "Position opened: %s @ $%.2f | SL=%.1f%% | Auto-close in %dH",
            signal["signal"], signal["close_price"],
            self.stop_loss_pct * 100, self.horizon,
        )

    def _close_open_position(self, reason: str, current_price: float) -> None:
        """Close the tracked position (live: API call, paper: log only)."""
        if not self._open_position:
            return

        pos = self._open_position
        entry = pos["entry_price"]
        direction = pos["signal"]

        if direction == "BUY":
            pnl_pct = (current_price - entry) / entry * 100
        else:
            pnl_pct = (entry - current_price) / entry * 100

        logger.info(
            "Position closed [%s]: %s @ $%.2f -> $%.2f (%.3f%%)",
            reason, direction, entry, current_price, pnl_pct,
        )

        # Live mode: close on Binance
        if self.live:
            symbol = self.binance_client._cfg.symbol
            try:
                self.binance_client.close_position(symbol)
            except Exception as e:
                logger.error(f"Failed to close Binance position: {e}")

        self._open_position = None

    def check_position_management(self, current_price: float) -> Optional[str]:
        """
        Check if open position should be closed due to horizon expiry or stop loss.

        Returns close reason string if closed, None otherwise.
        """
        if not self._open_position:
            return None

        pos = self._open_position
        entry = pos["entry_price"]
        direction = pos["signal"]
        age = datetime.now(timezone.utc) - pos["entry_time"]

        horizon_delta = timedelta(hours=self.horizon)

        # Check stop loss
        if direction == "BUY":
            stop_level = entry * (1 - self.stop_loss_pct)
            if current_price <= stop_level:
                self._close_open_position("STOP_LOSS", stop_level)
                return "stop_loss"
        else:  # SELL
            stop_level = entry * (1 + self.stop_loss_pct)
            if current_price >= stop_level:
                self._close_open_position("STOP_LOSS", stop_level)
                return "stop_loss"

        # Check horizon expiry
        if age >= horizon_delta:
            self._close_open_position("HORIZON_EXPIRY", current_price)
            return "horizon_expiry"

        return None

    def execute_trade(self, signal: dict) -> None:
        """
        Execute trade based on signal.

        In paper mode, logs synthetic fills.
        In live mode, executes via Binance Futures.
        """
        sig_type = signal["signal"]
        if sig_type not in ("BUY", "SELL"):
            return

        size = signal.get("position", {}).get("lot_size", 0)

        if not self.live:
            # Paper trading: close opposing position if one exists
            if self._open_position and self._open_position["signal"] != sig_type:
                self._close_open_position("OPPOSING_SIGNAL", signal["close_price"])

            logger.info(
                "PAPER TRADE: %s %.4f BTC @ $%.2f (risk=%.1f%%)",
                sig_type, size, signal["close_price"],
                signal.get("position", {}).get("risk_fraction", 0) * 100,
            )
            self._open_new_position(signal)
            return

        # LIVE execution via Binance Futures
        if not self._authenticated:
            self._ensure_authenticated()

        symbol = self.binance_client._cfg.symbol
        try:
            positions = self.binance_client.get_positions(symbol)
        except Exception as e:
            logger.error(f"Failed to fetch Binance positions: {e}")
            return

        current_pos = positions[0] if positions else None
        if current_pos:
            pos_amt = float(current_pos["positionAmt"])
            # Close if opposite direction: long (>0) vs SELL, short (<0) vs BUY
            is_long = pos_amt > 0
            if (is_long and sig_type == "SELL") or (not is_long and sig_type == "BUY"):
                logger.info("Closing opposite %s position before %s", "LONG" if is_long else "SHORT", sig_type)
                try:
                    self.binance_client.close_position(symbol)
                    self._open_position = None  # Clear tracked position
                except Exception as e:
                    logger.error(f"Failed to close Binance position: {e}")
                    return
                current_pos = None
            else:
                # Already positioned in same direction
                logger.info("Already %s %s. Skipping duplicate order.", "LONG" if is_long else "SHORT", symbol)
                return

        if size <= 0:
            logger.warning("Signal %s but lot_size=0. Skipping.", sig_type)
            return

        logger.info("LIVE TRADE: %s %.4f %s @ $%.2f", sig_type, size, symbol, signal["close_price"])
        try:
            self.binance_client.place_order(symbol=symbol, side=sig_type, quantity=size)
            self._open_new_position(signal)
        except Exception as e:
            logger.error(f"Failed to place Binance order: {e}")

    def _evaluate_past_signals(self, df: pd.DataFrame) -> None:
        """
        Check past BUY/SELL signals against actual price movement.

        Simulates stop loss by checking intra-bar highs/lows during the
        horizon window. A stop is hit if price moves against the trade
        by more than stop_loss_pct before the horizon expires.
        """
        if df.empty:
            return

        latest_price = float(df["close"].iloc[-1])
        latest_ts = df.index[-1]

        horizon_delta = timedelta(hours=self.horizon)

        for sig in self.signal_log:
            if sig.get("outcome") is not None:
                continue
            if sig["signal"] not in ("BUY", "SELL"):
                sig["outcome"] = "skip"
                continue

            sig_ts = pd.Timestamp(sig["timestamp"])
            if (latest_ts - sig_ts) < horizon_delta:
                continue

            entry_price = sig["close_price"]
            target_ts = sig_ts + horizon_delta

            # Get bars between signal and horizon expiry
            window = df.loc[(df.index > sig_ts) & (df.index <= target_ts)]

            # Check if stop loss was hit during the window
            stop_hit = False
            stop_price = 0.0
            if not window.empty and self.stop_loss_pct > 0:
                if sig["signal"] == "BUY":
                    stop_level = entry_price * (1 - self.stop_loss_pct)
                    worst = window["low"].min()
                    if worst <= stop_level:
                        stop_hit = True
                        stop_price = stop_level
                else:  # SELL
                    stop_level = entry_price * (1 + self.stop_loss_pct)
                    worst = window["high"].max()
                    if worst >= stop_level:
                        stop_hit = True
                        stop_price = stop_level

            if stop_hit:
                exit_price = stop_price
                sig["exit_reason"] = "stop_loss"
            else:
                # Normal horizon expiry
                future_bars = df.loc[df.index >= target_ts]
                if future_bars.empty:
                    exit_price = latest_price
                else:
                    exit_price = float(future_bars["close"].iloc[0])
                sig["exit_reason"] = "horizon_expiry"

            if sig["signal"] == "BUY":
                won = exit_price > entry_price
            else:
                won = exit_price < entry_price

            sig["outcome"] = "win" if won else "loss"
            sig["exit_price"] = exit_price

            pnl_pct = (exit_price - entry_price) / entry_price * 100
            if sig["signal"] == "SELL":
                pnl_pct = -pnl_pct
            sig["pnl_pct"] = round(pnl_pct, 3)

            # Update paper balance using the risk fraction from the trade
            risk_frac = sig.get("position", {}).get("risk_fraction", 0.02)
            trade_pnl_usd = self.paper_balance * risk_frac * (pnl_pct / 100) / self.stop_loss_pct
            sig["pnl_usd"] = round(trade_pnl_usd, 2)
            self.paper_balance += trade_pnl_usd

            pnl_label = f"{pnl_pct:+.3f}% (${trade_pnl_usd:+.2f})"

            self.evaluated_count += 1
            if won:
                self.win_count += 1
            else:
                self.loss_count += 1

            exit_reason = "STOP" if stop_hit else "4H"
            logger.info(
                "Signal evaluated: %s @ %.2f -> %.2f [%s] = %s (%s) | Balance: $%.2f",
                sig["signal"], entry_price, exit_price, exit_reason,
                sig["outcome"], pnl_label, self.paper_balance,
            )

    def reset_paper_balance(self) -> None:
        """Reset paper balance, signal log, and all stats to initial state."""
        self.paper_balance = self.initial_capital
        self.signal_log.clear()
        self.evaluated_count = 0
        self.win_count = 0
        self.loss_count = 0
        self.last_processed_ts = None
        self._last_signal_time = 0.0
        self.guardrails.initialize(self.initial_capital)
        logger.info("Paper balance reset to $%.2f", self.initial_capital)

    def get_win_rate_stats(self) -> dict:
        """Return current win rate statistics."""
        total_signals = len(self.signal_log)
        actionable = sum(1 for s in self.signal_log if s["signal"] in ("BUY", "SELL"))
        buys = sum(1 for s in self.signal_log if s["signal"] == "BUY")
        sells = sum(1 for s in self.signal_log if s["signal"] == "SELL")
        holds = total_signals - actionable

        win_rate = (self.win_count / self.evaluated_count * 100) if self.evaluated_count > 0 else 0.0

        total_pnl_usd = sum(s.get("pnl_usd", 0) for s in self.signal_log if s.get("outcome") in ("win", "loss"))
        total_pnl_pct = (self.paper_balance - self.initial_capital) / self.initial_capital * 100
        pnl_label = f"${total_pnl_usd:+,.2f} ({total_pnl_pct:+.2f}%)"

        stop_losses = sum(1 for s in self.signal_log if s.get("exit_reason") == "stop_loss")

        return {
            "total_signals": total_signals,
            "buys": buys,
            "sells": sells,
            "holds": holds,
            "evaluated": self.evaluated_count,
            "wins": self.win_count,
            "losses": self.loss_count,
            "win_rate": round(win_rate, 1),
            "total_pnl": pnl_label,
            "pending": actionable - self.evaluated_count,
            "paper_balance": round(self.paper_balance, 2),
            "stop_losses": stop_losses,
        }

    def run_once(self) -> Optional[dict]:
        """Fetch data, generate signal, and EXECUTE."""
        logger.info("Fetching recent bars...")
        try:
            df = self.fetch_recent_bars()
        except Exception as e:
            logger.error(f"Data fetch failed: {e}")
            self._authenticated = False  # Force re-auth on next cycle
            return None

        current_price = float(df["close"].iloc[-1])

        # Check position management (stop loss / horizon expiry) before new signal
        close_reason = self.check_position_management(current_price)
        if close_reason:
            logger.info("Position auto-closed: %s @ $%.2f", close_reason, current_price)

        # Evaluate past signals before generating new one
        self._evaluate_past_signals(df)

        latest_ts = df.index[-1]
        elapsed = time.time() - self._last_signal_time
        if self.last_processed_ts == latest_ts and elapsed < 300:
            logger.info("Data stale (ts=%s, %ds ago). Skipping.", latest_ts, int(elapsed))
            return None

        self.last_processed_ts = latest_ts
        self._last_signal_time = time.time()
        logger.info("Received %d bars, latest: %s", len(df), latest_ts)

        try:
            sig = self.generate_signal(df)
        except Exception as e:
            logger.error("Signal generation failed: %s", e, exc_info=True)
            return None

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

        # AUTO EXECUTION
        if self.auto_execute:
            try:
                if risk["can_trade"]:
                    self.execute_trade(sig)
                else:
                    logger.warning("Auto-execution skipped due to risk guardrails.")
            except Exception as e:
                logger.error("Trade execution failed: %s", e, exc_info=True)
        else:
            logger.info("Auto-execution disabled; signal emitted without order placement.")

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
        default=None,
        help="Seconds between signals (default: 3600)",
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
        default=4,
        help="Prediction horizon in bars (default: 4)",
    )
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        logger.error("Model directory not found: %s", model_dir)
        sys.exit(1)

    gen = SignalGenerator(model_dir, capital=args.capital, horizon=args.horizon)

    # Default interval: 1H
    interval = args.interval or 3600

    if args.once:
        gen.run_once()
    else:
        gen.run_loop(interval_seconds=interval)


if __name__ == "__main__":
    main()
