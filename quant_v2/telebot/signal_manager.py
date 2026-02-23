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
        self.model_dir = Path(model_dir).expanduser()
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

        ema_fast = float(close_series.ewm(span=8, adjust=False).mean().iloc[-1])
        ema_slow = float(close_series.ewm(span=21, adjust=False).mean().iloc[-1])
        momentum = (ema_fast / ema_slow - 1.0) if ema_slow > 0 else 0.0

        returns = close_series.pct_change().dropna()
        if returns.empty:
            returns = pd.Series([0.0], dtype=float)

        short_return = float(close_series.iloc[-1] / close_series.iloc[-4] - 1.0) if len(close_series) >= 4 else 0.0
        recent_vol = float(returns.tail(24).std() or 0.0)
        baseline_vol = float(returns.tail(120).std() or 0.0)
        if baseline_vol <= 0.0:
            baseline_vol = max(recent_vol, 1e-6)
        vol_ratio = recent_vol / baseline_vol if baseline_vol > 0 else 1.0

        score = (momentum * 450.0) + (short_return * 120.0)
        proba_up = min(max(0.5 + score, 0.01), 0.99)

        signal = "HOLD"
        threshold = 0.56
        reason = f"momentum={momentum:+.4f}, vol_ratio={vol_ratio:.2f}"
        drift_alert = False

        if vol_ratio >= 3.5:
            signal = "DRIFT_ALERT"
            drift_alert = True
            reason = (
                f"volatility_spike (recent_vol={recent_vol:.4f}, baseline_vol={baseline_vol:.4f}, ratio={vol_ratio:.2f})"
            )
        elif proba_up >= threshold:
            signal = "BUY"
            reason = f"proba_up={proba_up:.3f} >= {threshold:.2f}, momentum={momentum:+.4f}"
        elif proba_up <= (1.0 - threshold):
            signal = "SELL"
            reason = f"proba_up={proba_up:.3f} <= {1.0 - threshold:.2f}, momentum={momentum:+.4f}"

        regime = 1 if momentum > 0.0007 else -1 if momentum < -0.0007 else 0
        regime_probability = min(max(abs(momentum) * 400.0, 0.0), 1.0)

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

        return fetch_historical(
            date_from,
            date_to,
            symbol=symbol,
            interval=interval,
        )
