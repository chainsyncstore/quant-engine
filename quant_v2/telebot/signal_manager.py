"""Native v2 signal-source manager for Telegram execution loops."""

from __future__ import annotations

import asyncio
import inspect
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from functools import partial
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from quant.config import BinanceAPIConfig
from quant.data.binance_client import BinanceClient
from quant_v2.config import default_universe_symbols
from quant_v2.contracts import StrategySignal
from quant_v2.monitoring.kill_switch import MonitoringSnapshot

logger = logging.getLogger(__name__)

SignalCallback = Callable[[dict[str, Any]], Any]
FetchBarsFn = Callable[[object, datetime, datetime, str, str], pd.DataFrame]
ClientFactory = Callable[[dict[str, str], bool, str, str], object]


@dataclass
class _SignalSession:
    """Runtime state for one Telegram user's native v2 signal loop."""

    user_id: int
    live: bool
    client: object
    on_signal: SignalCallback
    running: bool = False
    task: asyncio.Task | None = None
    last_bar_timestamp: dict[str, pd.Timestamp] = field(default_factory=dict)
    signal_log: list[dict[str, Any]] = field(default_factory=list)


class V2SignalManager:
    """Manage native v2 signal loops without depending on legacy SignalGenerator."""

    def __init__(
        self,
        model_dir: Path,
        *,
        registry_root: Path | str | None = None,
        symbols: tuple[str, ...] | None = None,
        anchor_interval: str = "1h",
        horizon_bars: int = 4,
        history_bars: int = 192,
        loop_interval_seconds: int | None = None,
        max_consecutive_errors: int = 20,
        max_signal_log: int = 300,
        client_factory: ClientFactory | None = None,
        fetch_bars_fn: FetchBarsFn | None = None,
    ) -> None:
        from quant_v2.model_registry import ModelRegistry
        from quant_v2.models.trainer import load_model, TrainedModel
        
        self.model_dir = Path(model_dir).expanduser()
        self.registry_root = (
            Path(registry_root).expanduser() if registry_root is not None else self.model_dir
        )
        self.registry = ModelRegistry(self.registry_root)
        self.active_model: TrainedModel | None = None
        
        self.symbols = tuple(symbols or default_universe_symbols())
        self.anchor_interval = anchor_interval
        self.horizon_bars = int(horizon_bars)
        self.history_bars = max(int(history_bars), 48)
        self.loop_interval_seconds = self._resolve_loop_interval(loop_interval_seconds)
        self.max_consecutive_errors = max(int(max_consecutive_errors), 1)
        self.max_signal_log = max(int(max_signal_log), 20)
        self._client_factory = client_factory or self._default_client_factory
        self._fetch_bars_fn = fetch_bars_fn or self._default_fetch_bars
        self.sessions: dict[int, _SignalSession] = {}

    @staticmethod
    def _resolve_loop_interval(loop_interval_seconds: int | None) -> int:
        if loop_interval_seconds is not None:
            return max(int(loop_interval_seconds), 1)

        raw = os.getenv("BOT_V2_SIGNAL_LOOP_SECONDS", "3600").strip() or "3600"
        try:
            parsed = int(raw)
        except ValueError:
            parsed = 3600
        return max(parsed, 1)

    async def start_session(
        self,
        user_id: int,
        creds: dict,
        on_signal: SignalCallback,
        *,
        execute_orders: bool = True,
    ) -> bool:
        """Start a native v2 signal loop for a Telegram user."""

        _ = execute_orders  # kept for interface parity with legacy BotManager

        if user_id in self.sessions:
            logger.info("User %s native v2 session already active.", user_id)
            return False

        live = bool(creds.get("live", False))
        if live and (not creds.get("binance_api_key") or not creds.get("binance_api_secret")):
            raise RuntimeError(
                "Binance API credentials required for live trading. "
                "Run /setup BINANCE_API_KEY BINANCE_API_SECRET first."
            )

        default_symbol = self.symbols[0] if self.symbols else "BTCUSDT"
        client = self._client_factory(creds, live, default_symbol, self.anchor_interval)

        if live and hasattr(client, "authenticate"):
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, client.authenticate)

        session = _SignalSession(
            user_id=user_id,
            live=live,
            client=client,
            on_signal=on_signal,
            running=True,
        )
        session.task = asyncio.create_task(self._loop(session))
        self.sessions[user_id] = session
        logger.info(
            "Native v2 signal session started for user %s (mode=%s, symbols=%s).",
            user_id,
            "LIVE" if live else "PAPER",
            ",".join(self.symbols),
        )
        return True

    async def stop_session(self, user_id: int) -> bool:
        """Stop a native v2 signal loop."""

        session = self.sessions.pop(user_id, None)
        if session is None:
            return False

        session.running = False
        if session.task is not None:
            session.task.cancel()
            try:
                await session.task
            except asyncio.CancelledError:
                pass
        logger.info("Native v2 signal session stopped for user %s.", user_id)
        return True

    def reset_session_state(self, user_id: int) -> bool:
        """Reset in-session paper state while keeping the loop alive."""

        session = self.sessions.get(user_id)
        if session is None:
            return False
        if session.live:
            return False

        session.last_bar_timestamp.clear()
        session.signal_log.clear()
        return True

    def is_running(self, user_id: int) -> bool:
        session = self.sessions.get(user_id)
        if session is None:
            return False

        if not session.running:
            self.sessions.pop(user_id, None)
            return False

        if session.task is not None and session.task.done():
            self.sessions.pop(user_id, None)
            return False

        return True

    def get_active_count(self) -> int:
        active = [user_id for user_id in tuple(self.sessions.keys()) if self.is_running(user_id)]
        return len(active)

    def get_session_mode(self, user_id: int) -> str | None:
        session = self.sessions.get(user_id)
        if session is None:
            return None
        return "live" if session.live else "paper"

    def get_signal_stats(self, user_id: int) -> dict[str, int]:
        """Return aggregate signal counts for the active user session."""

        session = self.sessions.get(user_id)
        if session is None:
            return {
                "total_signals": 0,
                "buys": 0,
                "sells": 0,
                "holds": 0,
                "drift_alerts": 0,
                "symbols": 0,
            }

        log = session.signal_log
        buys = 0
        sells = 0
        holds = 0
        drift_alerts = 0
        symbols: set[str] = set()

        for entry in log:
            signal_type = str(entry.get("signal", "")).upper()
            if signal_type == "BUY":
                buys += 1
            elif signal_type == "SELL":
                sells += 1
            elif signal_type == "DRIFT_ALERT":
                drift_alerts += 1
            else:
                holds += 1

            symbol = str(entry.get("symbol", "")).strip().upper()
            if symbol:
                symbols.add(symbol)

        return {
            "total_signals": len(log),
            "buys": buys,
            "sells": sells,
            "holds": holds,
            "drift_alerts": drift_alerts,
            "symbols": len(symbols),
        }

    def get_recent_signals(self, user_id: int, *, limit: int = 5) -> tuple[dict[str, Any], ...]:
        """Return recent emitted signals for diagnostics display."""

        session = self.sessions.get(user_id)
        if session is None or limit <= 0:
            return ()

        recent = session.signal_log[-int(limit) :]
        return tuple(dict(item) for item in recent)

    async def _loop(self, session: _SignalSession) -> None:
        consecutive_errors = 0

        while session.running:
            try:
                await self._run_cycle(session)
                consecutive_errors = 0
            except asyncio.CancelledError:
                break
            except Exception as err:
                consecutive_errors += 1
                logger.error(
                    "Native v2 loop error for user %s (%s/%s): %s",
                    session.user_id,
                    consecutive_errors,
                    self.max_consecutive_errors,
                    err,
                    exc_info=True,
                )

                if consecutive_errors >= self.max_consecutive_errors:
                    session.running = False
                    await self._emit(
                        session,
                        {
                            "signal": "ENGINE_CRASH",
                            "reason": (
                                "Native v2 signal loop stopped after "
                                f"{consecutive_errors} consecutive errors. Last: {err}"
                            ),
                            "close_price": 0.0,
                            "probability": 0.0,
                            "regime": -1,
                            "position": {},
                            "symbol": self.symbols[0] if self.symbols else "BTCUSDT",
                        },
                    )
                    break

                backoff_seconds = min(60 * (2 ** (consecutive_errors - 1)), 300)
                await asyncio.sleep(backoff_seconds)
                continue

            try:
                await asyncio.sleep(self.loop_interval_seconds)
            except asyncio.CancelledError:
                break

        session.running = False

    async def _run_cycle(self, session: _SignalSession) -> None:
        loop = asyncio.get_running_loop()
        date_to = datetime.now(timezone.utc)
        date_from = date_to - timedelta(hours=self.history_bars)

        try:
            from quant_v2.models.trainer import load_model
            active_pointer = self.registry.get_active_version()
            if active_pointer and (self.active_model is None or getattr(self.active_model, "_version_id", None) != active_pointer.version_id):
                model_path = self._resolve_active_model_path(Path(active_pointer.artifact_dir))
                if model_path is not None:
                    self.active_model = load_model(model_path)
                    setattr(self.active_model, "_version_id", active_pointer.version_id)
                    logger.info("Loaded active ML model version %s for horizon %sm", active_pointer.version_id, self.horizon_bars)
        except Exception as e:
            logger.warning("Failed to refresh active model from registry: %s", e)

        for symbol in self.symbols:
            fetch_call = partial(
                self._fetch_bars_fn,
                session.client,
                date_from,
                date_to,
                symbol,
                self.anchor_interval,
            )
            bars = await loop.run_in_executor(None, fetch_call)
            if bars is None or bars.empty or "close" not in bars.columns:
                continue

            latest_ts = pd.Timestamp(bars.index[-1])
            prev_ts = session.last_bar_timestamp.get(symbol)
            if prev_ts is not None and latest_ts == prev_ts:
                continue

            session.last_bar_timestamp[symbol] = latest_ts
            payload = self._build_signal_payload(symbol, bars)
            session.signal_log.append(payload)
            if len(session.signal_log) > self.max_signal_log:
                del session.signal_log[:-self.max_signal_log]

            await self._emit(session, payload)

    async def _emit(self, session: _SignalSession, payload: dict[str, Any]) -> None:
        callback_result = session.on_signal(payload)
        if inspect.isawaitable(callback_result):
            await callback_result

    def _build_signal_payload(self, symbol: str, bars: pd.DataFrame) -> dict[str, Any]:
        close_series = pd.to_numeric(bars["close"], errors="coerce").dropna()
        if close_series.empty:
            raise RuntimeError(f"No valid close series for symbol={symbol}")

        close_price = float(close_series.iloc[-1])
        timestamp = str(pd.Timestamp(close_series.index[-1]))

        if len(close_series) < 30:
            return self._attach_native_v2_fields(
                {
                "timestamp": timestamp,
                "symbol": symbol,
                "close_price": close_price,
                "signal": "HOLD",
                "probability": 0.5,
                "regime": 0,
                "regime_probability": 0.0,
                "regime_tradeable": False,
                "threshold": 0.56,
                "reason": "insufficient_history",
                "horizon": self.horizon_bars,
                "position": {},
                "risk_status": {"can_trade": True},
                "drift_alert": False,
                "execution_anomaly_rate": 0.0,
                "connectivity_error_rate": 0.0,
                }
            )

        signal = "HOLD"
        threshold = 0.56
        drift_alert = False
        reason = "no_active_model"
        proba_up = 0.5
        regime = 0
        regime_probability = 0.0
        
        # Calculate some basic volatility for drift alerts
        returns = close_series.pct_change().dropna()
        if not returns.empty:
            recent_vol = float(returns.tail(24).std() or 0.0)
            baseline_vol = float(returns.tail(120).std() or 0.0)
            if baseline_vol <= 0.0:
                baseline_vol = max(recent_vol, 1e-6)
            vol_ratio = recent_vol / baseline_vol if baseline_vol > 0 else 1.0
            
            if vol_ratio >= 3.5:
                signal = "DRIFT_ALERT"
                drift_alert = True
                reason = (
                    f"volatility_spike (recent_vol={recent_vol:.4f}, baseline_vol={baseline_vol:.4f}, ratio={vol_ratio:.2f})"
                )

        if self.active_model is not None and not drift_alert:
            try:
                feature_row = self._prepare_model_features(symbol, bars)
                proba_up, uncertainty = self._predict_with_uncertainty(feature_row)
                
                # Signal logic based on model probability
                if proba_up >= threshold:
                    signal = "BUY"
                    reason = f"ML_Proba: {proba_up:.3f} >= {threshold:.2f} (unc: {uncertainty:.2f})"
                elif proba_up <= (1.0 - threshold):
                    signal = "SELL"
                    reason = f"ML_Proba: {proba_up:.3f} <= {1.0 - threshold:.2f} (unc: {uncertainty:.2f})"
                else:
                    signal = "HOLD"
                    reason = f"ML_Proba: {proba_up:.3f} inside deadband"
                    
                regime = 1 if proba_up > 0.55 else -1 if proba_up < 0.45 else 0
                regime_probability = max(proba_up, 1.0 - proba_up)
            except Exception as e:
                logger.error("Error generating ML prediction for %s: %s", symbol, e)
                signal = "HOLD"
                reason = f"ml_inference_error: {e.__class__.__name__}"
                proba_up = 0.5

        return self._attach_native_v2_fields(
            {
            "timestamp": timestamp,
            "symbol": symbol,
            "close_price": close_price,
            "signal": signal,
            "probability": round(proba_up, 4),
            "regime": regime,
            "regime_probability": round(regime_probability, 4),
            "regime_tradeable": signal in {"BUY", "SELL"},
            "threshold": threshold,
            "reason": reason,
            "horizon": self.horizon_bars,
            "position": {},
            "risk_status": {"can_trade": True},
            "drift_alert": drift_alert,
            "execution_anomaly_rate": 0.0,
            "connectivity_error_rate": 0.0,
            }
        )

    def _predict_with_uncertainty(self, feature_row: pd.DataFrame) -> tuple[float, float]:
        """Run model inference for either v2 or legacy trained model objects."""

        model = self.active_model
        if model is None:
            raise RuntimeError("No active model loaded")

        if hasattr(model, "primary_model"):
            from quant_v2.models.predictor import predict_proba_with_uncertainty

            proba_arr, uncertainty_arr = predict_proba_with_uncertainty(model, feature_row)
            return float(proba_arr[0]), float(uncertainty_arr[0])

        if hasattr(model, "raw_model"):
            from quant.models.predictor import predict_proba as legacy_predict_proba

            proba_arr = legacy_predict_proba(model, feature_row)
            proba_up = float(proba_arr[0])
            uncertainty = float(1.0 - abs(2.0 * proba_up - 1.0))
            return proba_up, uncertainty

        raise TypeError(f"Unsupported model type for inference: {type(model)!r}")

    def _prepare_model_features(self, symbol: str, bars: pd.DataFrame) -> pd.DataFrame:
        """Build and align one-row model features for inference."""

        from quant.features.pipeline import build_features

        frame = bars.copy()
        if not isinstance(frame.index, pd.DatetimeIndex):
            frame.index = pd.to_datetime(frame.index, utc=True, errors="coerce")
        else:
            if frame.index.tz is None:
                frame.index = frame.index.tz_localize("UTC")
            else:
                frame.index = frame.index.tz_convert("UTC")
        frame = frame[~frame.index.isna()].sort_index()
        frame.index.name = "timestamp"

        featured = build_features(frame)
        if featured.empty:
            raise ValueError(f"No feature rows available for symbol={symbol}")

        last_row = featured.iloc[[-1]].copy()
        feature_names = [str(name) for name in getattr(self.active_model, "feature_names", [])]
        if not feature_names:
            return last_row.reset_index(drop=True)

        missing = [name for name in feature_names if name not in last_row.columns]
        for name in missing:
            last_row[name] = 0.0

        return last_row[feature_names].reset_index(drop=True)

    def _attach_native_v2_fields(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Attach direct v2 routing artifacts to emitted signal payloads."""

        signal_type = str(payload.get("signal", "HOLD")).strip().upper()
        if signal_type not in {"BUY", "SELL", "HOLD", "DRIFT_ALERT"}:
            signal_type = "HOLD"

        symbol = str(payload.get("symbol") or (self.symbols[0] if self.symbols else "BTCUSDT")).strip().upper()
        if not symbol:
            symbol = "BTCUSDT"

        close_price = float(payload.get("close_price", 0.0) or 0.0)
        if close_price <= 0.0:
            close_price = 0.0

        proba_up = self._bounded_rate(payload.get("probability", 0.5))
        if signal_type == "BUY":
            confidence = proba_up
        elif signal_type == "SELL":
            confidence = 1.0 - proba_up
        else:
            confidence = max(proba_up, 1.0 - proba_up)
        confidence = self._bounded_rate(confidence)
        uncertainty = self._bounded_rate(1.0 - confidence)

        reason = str(payload.get("reason", ""))
        native_signal = StrategySignal(
            symbol=symbol,
            timeframe=self.anchor_interval,
            horizon_bars=self.horizon_bars,
            signal=signal_type,
            confidence=confidence,
            uncertainty=uncertainty,
            reason=reason,
        )

        monitoring_snapshot = MonitoringSnapshot(
            feature_drift_alert=bool(payload.get("drift_alert", False)) or signal_type == "DRIFT_ALERT",
            confidence_collapse_alert=(
                "confidence drift" in reason.lower() or "confidence collapse" in reason.lower()
            ),
            execution_anomaly_rate=self._bounded_rate(payload.get("execution_anomaly_rate", 0.0)),
            connectivity_error_rate=self._bounded_rate(payload.get("connectivity_error_rate", 0.0)),
            hard_risk_breach=not bool((payload.get("risk_status") or {}).get("can_trade", True)),
        )

        payload["v2_signal"] = native_signal
        payload["v2_prices"] = {symbol: close_price}
        payload["v2_monitoring_snapshot"] = monitoring_snapshot
        return payload

    def _resolve_active_model_path(self, artifact_dir: Path) -> Path | None:
        candidates = (
            artifact_dir / f"model_{self.horizon_bars}m.pkl",
            artifact_dir / f"model_{self.horizon_bars}m.joblib",
            artifact_dir / "lgbm_model.joblib",
        )
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None

    @staticmethod
    def _bounded_rate(value: object) -> float:
        try:
            rate = float(value)
        except (TypeError, ValueError):
            return 0.0
        if rate < 0.0:
            return 0.0
        if rate > 1.0:
            return 1.0
        return rate

    @staticmethod
    def _default_client_factory(creds: dict[str, str], live: bool, symbol: str, interval: str) -> object:
        base_url = "https://fapi.binance.com" if live else "https://testnet.binancefuture.com"
        cfg = BinanceAPIConfig(
            api_key=str(creds.get("binance_api_key", "") or ""),
            api_secret=str(creds.get("binance_api_secret", "") or ""),
            base_url=base_url,
            symbol=symbol,
            interval=interval,
        )
        return BinanceClient(config=cfg)

    @staticmethod
    def _default_fetch_bars(
        client: object,
        date_from: datetime,
        date_to: datetime,
        symbol: str,
        interval: str,
    ) -> pd.DataFrame:
        fetch_historical = getattr(client, "fetch_historical", None)
        if not callable(fetch_historical):
            raise RuntimeError("Client does not expose fetch_historical")

        bars = fetch_historical(
            date_from,
            date_to,
            symbol=symbol,
            interval=interval,
        )
        if bars is None or bars.empty:
            return pd.DataFrame() if bars is None else bars

        funding = pd.DataFrame(columns=["funding_rate_raw"])
        fetch_funding_rates = getattr(client, "fetch_funding_rates", None)
        if callable(fetch_funding_rates):
            try:
                funding = fetch_funding_rates(date_from, date_to, symbol=symbol)
            except Exception as exc:
                logger.warning("Funding fetch failed for %s: %s", symbol, exc)

        open_interest = pd.DataFrame(columns=["open_interest", "open_interest_value"])
        fetch_open_interest = getattr(client, "fetch_open_interest", None)
        if callable(fetch_open_interest):
            try:
                open_interest = fetch_open_interest(date_from, date_to, symbol=symbol, period=interval)
            except TypeError:
                open_interest = fetch_open_interest(date_from, date_to, symbol=symbol)
            except Exception as exc:
                logger.warning("Open-interest fetch failed for %s: %s", symbol, exc)

        from quant.data.binance_client import BinanceClient

        try:
            merged = BinanceClient.merge_supplementary(ohlcv=bars, funding=funding, oi=open_interest)
        except Exception as exc:
            logger.warning("Supplementary merge failed for %s: %s", symbol, exc)
            return bars

        merged.index.name = "timestamp"
        return merged
