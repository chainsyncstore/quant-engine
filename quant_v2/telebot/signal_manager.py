"""Native v2 signal-source manager for Telegram execution loops."""

from __future__ import annotations

import asyncio
import inspect
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from functools import partial
from pathlib import Path
from typing import Any, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from quant_v2.models.ensemble import HorizonEnsemble, FullEnsemble

import pandas as pd

from quant.config import BinanceAPIConfig
from quant.data.binance_client import BinanceClient
from quant_v2.config import default_universe_symbols
from quant_v2.contracts import StrategySignal
from quant_v2.monitoring.kill_switch import MonitoringSnapshot
from quant_v2.data.news_client import CryptoCompareNewsClient, FearGreedClient, symbol_to_base_ticker
from quant_v2.strategy.event_gate import evaluate_event_gate
from quant_v2.portfolio.cost_model import confidence_to_edge_bps, get_default_cost_model
from quant_v2.telebot.symbol_scorecard import SymbolScorecard

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
        self.horizon_ensemble: "HorizonEnsemble | None" = None  # Phase 1: multi-horizon ensemble
        self.full_ensemble: "FullEnsemble | None" = None  # Phase 3: LightGBM + Chronos
        self._last_model_agreement: float | None = None  # Phase 3: cached per-symbol

        # Ensure BTCUSDT is processed first (needed for cross-pair features)
        symbols_list = list(symbols or default_universe_symbols())
        if "BTCUSDT" in symbols_list:
            symbols_list.remove("BTCUSDT")
            symbols_list.insert(0, "BTCUSDT")
        self.symbols = tuple(symbols_list)
        self.anchor_interval = anchor_interval
        self.horizon_bars = int(horizon_bars)
        self.history_bars = max(int(history_bars), 48)
        self.loop_interval_seconds = self._resolve_loop_interval(loop_interval_seconds)
        self.max_consecutive_errors = max(int(max_consecutive_errors), 1)
        self.max_signal_log = max(int(max_signal_log), 20)
        self._client_factory = client_factory or self._default_client_factory
        self._fetch_bars_fn = fetch_bars_fn or self._default_fetch_bars
        self.sessions: dict[int, _SignalSession] = {}
        # OI cache for API 202 fallback: symbol -> (timestamp, oi_value)
        self._oi_cache: dict[str, tuple[datetime, float]] = {}
        # Per-symbol prediction accuracy scorecard (shared across sessions)
        self.scorecard = SymbolScorecard(lookback_hours=72, min_samples=8)

        # --- Event gate: news awareness layer (Phase 2) ---
        _news_api_key = os.getenv("CRYPTOCOMPARE_API_KEY", "")
        self.news_client: CryptoCompareNewsClient | None = (
            CryptoCompareNewsClient(_news_api_key) if _news_api_key else None
        )
        self.fear_greed_client = FearGreedClient()
        self._cached_events: list = []
        self._events_fetched_at: datetime | None = None
        # Rolling close price histories for optimizer (symbol -> Series of close prices)
        self._price_history_cache: dict[str, pd.Series] = {}
        # Shared per-cycle prediction cache (key=(symbol, anchor_interval, bar_ts_iso))
        # value=(monotonic_inserted_at, payload_dict).  De-duplicates `_build_signal_payload`
        # work across concurrent user sessions running on the same bar timestamp.
        # Refs: audit_20260423 P2-2.
        self._shared_inference_cache: dict[tuple[str, str, str], tuple[float, dict[str, Any]]] = {}
        self._shared_inference_cache_max: int = max(len(self.symbols) * 4, 32)

    @staticmethod
    def _resolve_loop_interval(loop_interval_seconds: int | None) -> int:
        if loop_interval_seconds is not None:
            return max(int(loop_interval_seconds), 1)

        raw = os.getenv("BOT_V2_SIGNAL_LOOP_SECONDS", "900").strip() or "900"
        try:
            parsed = int(raw)
        except ValueError:
            parsed = 900
        return max(parsed, 1)

    def _shared_cache_ttl(self) -> float:
        # Allow modest stagger between concurrent user loops; anything older than
        # 1.5x the loop interval is definitely from a previous bar and must be
        # refreshed.  Cap at 30 minutes so paper/live divergence on a stuck loop
        # is still bounded.
        return min(float(self.loop_interval_seconds) * 1.5, 1800.0)

    def _shared_cache_get(self, key: tuple[str, str, str]) -> dict[str, Any] | None:
        entry = self._shared_inference_cache.get(key)
        if entry is None:
            return None
        inserted_at, payload = entry
        if (time.monotonic() - inserted_at) > self._shared_cache_ttl():
            self._shared_inference_cache.pop(key, None)
            return None
        return payload

    def _shared_cache_put(self, key: tuple[str, str, str], payload: dict[str, Any]) -> None:
        self._shared_inference_cache[key] = (time.monotonic(), dict(payload))
        # Bounded eviction: drop oldest insertions when over the cap.
        if len(self._shared_inference_cache) > self._shared_inference_cache_max:
            overflow = len(self._shared_inference_cache) - self._shared_inference_cache_max
            for stale_key in list(self._shared_inference_cache.keys())[:overflow]:
                self._shared_inference_cache.pop(stale_key, None)

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
        self.scorecard.reset()
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

    def get_traded_signal_stats(self, user_id: int) -> dict[str, Any]:
        """Return stats for only actionable (BUY/SELL) model picks."""

        session = self.sessions.get(user_id)
        if session is None:
            return {"total_trades": 0, "buys": 0, "sells": 0, "symbols": 0, "per_symbol": {}}

        buys = 0
        sells = 0
        per_symbol: dict[str, dict[str, int]] = {}

        for entry in session.signal_log:
            signal_type = str(entry.get("signal", "")).upper()
            if signal_type not in ("BUY", "SELL"):
                continue
            symbol = str(entry.get("symbol", "")).strip().upper()
            if signal_type == "BUY":
                buys += 1
            else:
                sells += 1
            if symbol:
                sym_stats = per_symbol.setdefault(symbol, {"buys": 0, "sells": 0})
                sym_stats["buys" if signal_type == "BUY" else "sells"] += 1

        return {
            "total_trades": buys + sells,
            "buys": buys,
            "sells": sells,
            "symbols": len(per_symbol),
            "per_symbol": per_symbol,
        }

    def get_recent_traded_signals(self, user_id: int, *, limit: int = 8) -> tuple[dict[str, Any], ...]:
        """Return recent BUY/SELL signals only (actual model trade picks)."""

        session = self.sessions.get(user_id)
        if session is None or limit <= 0:
            return ()

        traded = [
            entry for entry in session.signal_log
            if str(entry.get("signal", "")).upper() in ("BUY", "SELL")
        ]
        recent = traded[-int(limit):]
        return tuple(dict(item) for item in recent)

    def get_scorecard_summary(self) -> dict[str, dict[str, Any]]:
        """Expose per-symbol scorecard accuracy for diagnostics."""
        return self.scorecard.get_summary()

    def get_recent_signals(self, user_id: int, *, limit: int = 5) -> tuple[dict[str, Any], ...]:
        """Return recent emitted signals for diagnostics display."""

        session = self.sessions.get(user_id)
        if session is None or limit <= 0:
            return ()

        recent = session.signal_log[-int(limit) :]
        return tuple(dict(item) for item in recent)

    async def get_realtime_prices(
        self,
        user_id: int,
        *,
        symbols: tuple[str, ...] | None = None,
    ) -> dict[str, float]:
        """Fetch latest per-symbol market prices on demand for diagnostics paths.

        Falls back to any available session client or an ephemeral client so
        that /stats always returns live prices even when the requesting user's
        own signal session is not active.
        """

        session = self.sessions.get(user_id)
        client = getattr(session, "client", None) if session is not None else None

        # Fallback: borrow a client from any running session
        if client is None:
            for other_session in self.sessions.values():
                if getattr(other_session, "client", None) is not None:
                    client = other_session.client
                    break

        # Fallback: create an ephemeral read-only client for market data
        if client is None:
            try:
                client = self._client_factory(
                    {}, False,
                    self.symbols[0] if self.symbols else "BTCUSDT",
                    self.anchor_interval,
                )
            except Exception as exc:
                logger.warning("Ephemeral client creation failed for price refresh: %s", exc)
                return {}

        target_symbols = tuple(
            dict.fromkeys(
                str(symbol).strip().upper()
                for symbol in (symbols or self.symbols)
                if str(symbol).strip()
            )
        )
        if not target_symbols:
            return {}

        loop = asyncio.get_running_loop()

        async def _fetch_symbol(symbol: str) -> tuple[str, float]:
            fetch_call = partial(
                self._fetch_realtime_symbol_price,
                client,
                symbol,
                self.anchor_interval,
            )
            try:
                price = await loop.run_in_executor(None, fetch_call)
            except Exception as exc:
                logger.warning(
                    "Realtime price refresh failed for user %s symbol=%s: %s",
                    user_id,
                    symbol,
                    exc,
                )
                return symbol, 0.0
            return symbol, float(price)

        price_pairs = await asyncio.gather(*(_fetch_symbol(symbol) for symbol in target_symbols))
        return {
            symbol: price
            for symbol, price in price_pairs
            if float(price) > 0.0
        }

    @staticmethod
    def _first_orderbook_price(levels: object) -> float:
        if not isinstance(levels, list) or not levels:
            return 0.0

        first_level = levels[0]
        raw_price: object
        if isinstance(first_level, (list, tuple)) and first_level:
            raw_price = first_level[0]
        elif isinstance(first_level, dict):
            raw_price = first_level.get("price", 0.0)
        else:
            raw_price = 0.0

        try:
            return float(raw_price)
        except (TypeError, ValueError):
            return 0.0

    @classmethod
    def _fetch_realtime_symbol_price(
        cls,
        client: object,
        symbol: str,
        interval: str,
    ) -> float:
        """Fetch latest symbol mark proxy from orderbook midpoint, fallback to klines close."""

        get_orderbook = getattr(client, "get_orderbook", None)
        if callable(get_orderbook):
            try:
                book = get_orderbook(symbol, limit=5)
                if isinstance(book, dict):
                    bid = cls._first_orderbook_price(book.get("bids", []))
                    ask = cls._first_orderbook_price(book.get("asks", []))
                    if bid > 0.0 and ask > 0.0:
                        return float((bid + ask) / 2.0)
                    if bid > 0.0:
                        return float(bid)
                    if ask > 0.0:
                        return float(ask)
            except Exception as exc:
                logger.debug("Orderbook refresh failed for %s: %s", symbol, exc)

        fetch_historical = getattr(client, "fetch_historical", None)
        if callable(fetch_historical):
            date_to = datetime.now(timezone.utc)
            date_from = date_to - timedelta(hours=4)
            bars = fetch_historical(
                date_from,
                date_to,
                symbol=symbol,
                interval=interval,
            )
            if isinstance(bars, pd.DataFrame) and not bars.empty and "close" in bars.columns:
                close_series = pd.to_numeric(bars["close"], errors="coerce").dropna()
                if not close_series.empty:
                    return float(close_series.iloc[-1])

        return 0.0

    async def _loop(self, session: _SignalSession) -> None:
        consecutive_errors = 0

        while session.running:
            try:
                cycle_cache: dict[tuple[str, str, str], dict[str, Any]] = {}
                await self._run_cycle(session, cycle_cache=cycle_cache)
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
                now_ts = datetime.now(timezone.utc).timestamp()
                next_run_ts = ((int(now_ts) // self.loop_interval_seconds) + 1) * self.loop_interval_seconds
                sleep_sec = next_run_ts - now_ts + 3.0  # +3s safety buffer to ensure Binance bars are closed
                await asyncio.sleep(max(0.1, sleep_sec))
            except asyncio.CancelledError:
                break

        session.running = False

    async def _run_cycle(
        self,
        session: _SignalSession,
        *,
        cycle_cache: dict[tuple[str, str, str], dict[str, Any]] | None = None,
    ) -> None:
        loop = asyncio.get_running_loop()
        date_to = datetime.now(timezone.utc)
        date_from = date_to - timedelta(hours=self.history_bars)

        try:
            from quant_v2.models.trainer import load_model
            from quant_v2.models.ensemble import HorizonEnsemble
            active_pointer = self.registry.get_active_version()
            if active_pointer and (self.active_model is None or getattr(self.active_model, "_version_id", None) != active_pointer.version_id):
                model_path = self._resolve_active_model_path(Path(active_pointer.artifact_dir))
                if model_path is not None:
                    self.active_model = load_model(model_path)
                    setattr(self.active_model, "_version_id", active_pointer.version_id)
                    logger.info("Loaded active ML model version %s for horizon %sm", active_pointer.version_id, self.horizon_bars)

                # Attempt to load multi-horizon ensemble from artifact directory
                artifact_dir = Path(active_pointer.artifact_dir)
                ensemble = HorizonEnsemble.from_directory(artifact_dir)
                if ensemble is not None and ensemble.horizon_count > 0:
                    self.horizon_ensemble = ensemble
                    logger.info("Loaded %d-horizon ensemble", ensemble.horizon_count)
                else:
                    self.horizon_ensemble = None

                # Phase 3: Build full ensemble (LightGBM + Chronos)
                # Chronos requires PyTorch (~2-3GB RAM). Disable on small instances.
                _enable_chronos = os.getenv("BOT_ENABLE_CHRONOS", "0").strip().lower() in {"1", "true", "yes"}
                if _enable_chronos:
                    from quant_v2.models.ensemble import FullEnsemble
                    try:
                        self.full_ensemble = FullEnsemble(
                            lgbm_ensemble=self.horizon_ensemble,
                            enable_chronos=True,
                        )
                        logger.info("FullEnsemble initialized (chronos=True, lgbm=%s)",
                                    self.horizon_ensemble is not None)
                    except Exception as fe_err:
                        logger.warning("Failed to initialize FullEnsemble: %s", fe_err)
                        self.full_ensemble = None
                else:
                    self.full_ensemble = None
        except Exception as e:
            logger.warning("Failed to refresh active model from registry: %s", e)

        # --- Fetch news events (once per cycle, cached for 15 min) ---
        now = datetime.now(timezone.utc)
        if (
            self._events_fetched_at is None
            or (now - self._events_fetched_at).total_seconds() > 900
        ):
            merged_events: list = []
            # CryptoCompare per-coin news (requires API key)
            if self.news_client is not None:
                base_tickers = [symbol_to_base_ticker(s) for s in self.symbols]
                merged_events.extend(self.news_client.fetch_recent(symbols=base_tickers))
            # Alternative.me Fear & Greed (no key needed, global macro)
            try:
                fng_event = self.fear_greed_client.fetch_current()
                if fng_event is not None:
                    merged_events.append(fng_event)
            except Exception as fng_err:
                logger.debug("Fear & Greed fetch skipped: %s", fng_err)
            self._cached_events = merged_events
            self._events_fetched_at = now
            if self._cached_events:
                logger.info("Event gate: fetched %d events (news + F&G)", len(self._cached_events))

        cycle_prices: dict[str, float] = {}
        btc_returns: pd.Series | None = None

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

            # Cache BTC returns for cross-pair feature injection
            if symbol == "BTCUSDT":
                btc_close = pd.to_numeric(bars["close"], errors="coerce").dropna()
                btc_returns = btc_close.pct_change()

            latest_ts = pd.Timestamp(bars.index[-1])
            prev_ts = session.last_bar_timestamp.get(symbol)
            if prev_ts is not None and latest_ts == prev_ts:
                continue

            # --- OI resilience: detect missing data and apply fallback ---
            data_quality_flag = False
            if "open_interest" in bars.columns:
                oi_col = pd.to_numeric(bars["open_interest"], errors="coerce")
                if oi_col.dropna().empty:
                    # OI fetch failed — try cache fallback
                    cached = self._oi_cache.get(symbol)
                    if cached is not None:
                        cache_ts, cache_val = cached
                        staleness = (date_to - cache_ts).total_seconds()
                        if staleness < 180:  # < 3 min stale: use cached
                            bars["open_interest"] = cache_val
                            bars["open_interest_value"] = 0.0
                        else:
                            bars["open_interest"] = 0.0
                            bars["open_interest_value"] = 0.0
                            data_quality_flag = True
                    else:
                        bars["open_interest"] = 0.0
                        bars["open_interest_value"] = 0.0
                        data_quality_flag = True
                else:
                    # Cache the latest valid OI value
                    last_valid_oi = float(oi_col.dropna().iloc[-1])
                    self._oi_cache[symbol] = (date_to, last_valid_oi)

            # Fetch L2 order book snapshot for this symbol
            ob_snapshot: dict | None = None
            try:
                ob_snapshot = await loop.run_in_executor(
                    None, lambda s=symbol: session.client.get_orderbook(s, limit=20)
                )
            except Exception as ob_err:
                logger.debug("Order book fetch failed for %s: %s", symbol, ob_err)

            # Cache close price history for optimizer
            close_hist = pd.to_numeric(bars["close"], errors="coerce").dropna()
            if not close_hist.empty:
                self._price_history_cache[symbol] = close_hist

            session.last_bar_timestamp[symbol] = latest_ts

            # --- Per-cycle cache: compute once per (symbol, bar_timestamp) ---
            # Lookup order: per-cycle (in-memory) cache → manager-level shared cache
            # → fresh compute.  The shared cache de-duplicates work across concurrent
            # user sessions; the per-cycle cache is the authoritative copy within one
            # session's loop iteration.  Refs: audit_20260423 P2-2.
            bar_ts_iso = latest_ts.isoformat()
            cache_key = (symbol, self.anchor_interval, bar_ts_iso)
            cached_payload: dict[str, Any] | None = None
            if cycle_cache is not None:
                cached_payload = cycle_cache.get(cache_key)
            if cached_payload is None:
                cached_payload = self._shared_cache_get(cache_key)
            if cached_payload is not None:
                payload = dict(cached_payload)
            else:
                payload = self._build_signal_payload(
                    symbol, bars, btc_returns=btc_returns, data_quality_flag=data_quality_flag,
                    ob_snapshot=ob_snapshot,
                )
                self._shared_cache_put(cache_key, payload)
            if cycle_cache is not None:
                cycle_cache[cache_key] = dict(payload)

            # --- Collect price for scorecard evaluation ---
            close_price = float(payload.get("close_price", 0.0) or 0.0)
            if close_price > 0.0:
                cycle_prices[symbol] = close_price

            # --- Record actionable prediction in scorecard ---
            signal_type = str(payload.get("signal", "HOLD")).upper()
            if signal_type in ("BUY", "SELL") and close_price > 0.0:
                self.scorecard.record_prediction(
                    symbol=symbol,
                    direction=signal_type,
                    confidence=float(payload.get("probability", 0.5)),
                    entry_price=close_price,
                    horizon_bars=self.horizon_bars,
                )

            session.signal_log.append(payload)
            if len(session.signal_log) > self.max_signal_log:
                del session.signal_log[:-self.max_signal_log]

            await self._emit(session, payload)

        # --- Evaluate pending scorecard predictions after all symbols processed ---
        if cycle_prices:
            resolved = self.scorecard.evaluate_pending(cycle_prices)
            if resolved > 0:
                logger.info("Scorecard: resolved %d pending predictions", resolved)

    async def _emit(self, session: _SignalSession, payload: dict[str, Any]) -> None:
        callback_result = session.on_signal(payload)
        if inspect.isawaitable(callback_result):
            await callback_result

    def _build_signal_payload(
        self,
        symbol: str,
        bars: pd.DataFrame,
        *,
        btc_returns: pd.Series | None = None,
        data_quality_flag: bool = False,
        ob_snapshot: dict | None = None,
    ) -> dict[str, Any]:
        close_series = pd.to_numeric(bars["close"], errors="coerce").dropna()
        if close_series.empty:
            raise RuntimeError(f"No valid close series for symbol={symbol}")

        close_price = float(close_series.iloc[-1])
        timestamp = str(pd.Timestamp(close_series.index[-1]))

        high_series = pd.to_numeric(bars["high"], errors="coerce").dropna() if "high" in bars.columns else pd.Series(dtype=float)
        low_series = pd.to_numeric(bars["low"], errors="coerce").dropna() if "low" in bars.columns else pd.Series(dtype=float)

        if len(close_series) < 30:
            return self._attach_native_v2_fields(
                {
                "timestamp": timestamp,
                "symbol": symbol,
                "close_price": close_price,
                "signal": "HOLD",
                "probability": 0.5,
                "regime": 3,
                "regime_probability": 0.0,
                "regime_tradeable": False,
                "threshold": 0.65,
                "reason": "insufficient_history",
                "horizon": self.horizon_bars,
                "position": {},
                "risk_status": {"can_trade": True},
                "drift_alert": False,
                "execution_anomaly_rate": 0.0,
                "connectivity_error_rate": 0.0,
                "_close_series": close_series,
                "_high_series": high_series,
                "_low_series": low_series,
                }
            )

        signal = "HOLD"
        drift_alert = False
        reason = "no_active_model"
        proba_up = 0.5
        regime = 3
        regime_risk = 0.5  # float; 0=calm, 1=adverse/unknown
        regime_probability = 0.0

        # --- Drift detection ---
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

        # --- Feature pipeline + regime classification ---
        featured = None
        if not drift_alert:
            try:
                featured = self._build_featured_frame(bars, btc_returns=btc_returns, ob_snapshot=ob_snapshot)
                if featured is not None and not featured.empty:
                    from quant_v2.strategy.regime import classify_latest

                    fz = (
                        featured["funding_rate_zscore"]
                        if "funding_rate_zscore" in featured.columns
                        else pd.Series(0.0, index=featured.index)
                    )
                    regime_state = classify_latest(
                        pd.to_numeric(featured["close"], errors="coerce").dropna(),
                        fz,
                    )
                    regime = regime_state.regime
                    regime_risk = regime_state.regime_risk
                    regime_probability = max(0.0, 1.0 - regime_risk * 0.5)
            except Exception as exc:
                logger.warning("Regime classification failed for %s: %s", symbol, exc)

        # --- Model inference with regime-scaled thresholds ---
        # Base 0.55 is reachable by a ~53% accuracy model; regime_risk
        # widens the deadband proportionally (0 = calm → 0.55, 1 = adverse → 0.63).
        buy_threshold = min(0.90, 0.55 + 0.08 * regime_risk)
        sell_threshold = max(0.10, 1.0 - buy_threshold)

        self._last_model_agreement = None  # Reset before each symbol prediction
        if (self.active_model is not None or self.full_ensemble is not None) and not drift_alert:
            try:
                if featured is not None and not featured.empty:
                    feature_row = self._extract_model_row(featured)
                else:
                    feature_row = self._prepare_model_features(symbol, bars)
                proba_up, uncertainty = self._predict_with_uncertainty(
                    feature_row, close_series=close_series,
                )

                # Data quality penalty: shrink probability toward 0.5 by 25%
                if data_quality_flag:
                    proba_up = 0.5 + (proba_up - 0.5) * 0.75

                if proba_up >= buy_threshold:
                    signal = "BUY"
                    reason = (
                        f"ML_Proba: {proba_up:.3f} >= {buy_threshold:.2f} "
                        f"(regime={regime}, risk={regime_risk}, unc: {uncertainty:.2f})"
                    )
                elif proba_up <= sell_threshold:
                    signal = "SELL"
                    reason = (
                        f"ML_Proba: {proba_up:.3f} <= {sell_threshold:.2f} "
                        f"(regime={regime}, risk={regime_risk}, unc: {uncertainty:.2f})"
                    )
                else:
                    signal = "HOLD"
                    reason = (
                        f"ML_Proba: {proba_up:.3f} inside deadband "
                        f"[{sell_threshold:.2f},{buy_threshold:.2f}] (regime={regime})"
                    )

                regime_probability = max(proba_up, 1.0 - proba_up)
                logger.info(
                    "Signal decision: %s %s proba=%.4f buy_th=%.2f sell_th=%.2f regime=%d risk=%.1f",
                    symbol, signal, proba_up, buy_threshold, sell_threshold, regime, regime_risk,
                )
            except Exception as e:
                logger.error("Error generating ML prediction for %s: %s", symbol, e)
                signal = "HOLD"
                reason = f"ml_inference_error: {e.__class__.__name__}"
                proba_up = 0.5

        effective_threshold = buy_threshold  # for payload reporting

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
            "threshold": effective_threshold,
            "reason": reason,
            "horizon": self.horizon_bars,
            "position": {},
            "risk_status": {"can_trade": True},
            "drift_alert": drift_alert,
            "execution_anomaly_rate": 0.0,
            "connectivity_error_rate": 0.0,
            "_close_series": close_series,
            "_high_series": high_series,
            "_low_series": low_series,
            }
        )

    def _predict_with_uncertainty(
        self, feature_row: pd.DataFrame, close_series: pd.Series | None = None,
    ) -> tuple[float, float]:
        """Run model inference — full ensemble / horizon ensemble / single model fallback."""

        # Phase 3: Try full ensemble (LightGBM + Chronos) first
        if self.full_ensemble is not None and close_series is not None:
            try:
                p, u, agreement = self.full_ensemble.predict(
                    feature_row, close_series, prediction_length=self.horizon_bars,
                )
                # Store agreement for later attachment to StrategySignal
                self._last_model_agreement = agreement
                return p, u
            except Exception as e:
                logger.warning("FullEnsemble prediction failed, falling back: %s", e)

        # Try horizon ensemble
        if self.horizon_ensemble is not None:
            return self.horizon_ensemble.predict(feature_row)

        # Single model fallback (existing code)
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

    def _build_featured_frame(
        self,
        bars: pd.DataFrame,
        btc_returns: pd.Series | None = None,
        ob_snapshot: dict | None = None,
    ) -> pd.DataFrame | None:
        """Build the full feature DataFrame for regime classification and model inference."""

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

        # Inject BTC returns for cross-pair features
        if btc_returns is not None:
            frame["_btc_returns"] = btc_returns.reindex(frame.index, method="ffill").fillna(0.0)

        # Inject order book snapshot onto the latest bar for order_book.compute()
        if ob_snapshot is not None:
            snapshots = [None] * len(frame)
            snapshots[-1] = ob_snapshot
            frame["_ob_snapshot"] = snapshots

        featured = build_features(frame)
        return featured if not featured.empty else None

    def _extract_model_row(self, featured: pd.DataFrame) -> pd.DataFrame:
        """Extract and align the last row of a featured DataFrame for model inference."""

        last_row = featured.iloc[[-1]].copy()
        feature_names = [str(name) for name in getattr(self.active_model, "feature_names", [])]
        if not feature_names:
            return last_row.reset_index(drop=True)

        missing = [name for name in feature_names if name not in last_row.columns]
        for name in missing:
            last_row[name] = 0.0

        return last_row[feature_names].reset_index(drop=True)

    def _prepare_model_features(self, symbol: str, bars: pd.DataFrame) -> pd.DataFrame:
        """Build and align one-row model features for inference (legacy path)."""

        featured = self._build_featured_frame(bars)
        if featured is None or featured.empty:
            raise ValueError(f"No feature rows available for symbol={symbol}")

        return self._extract_model_row(featured)

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

        # --- Session hour (UTC) from signal timestamp ---
        session_hour_utc: int | None = None
        try:
            ts_raw = payload.get("timestamp")
            if ts_raw is not None:
                ts = pd.Timestamp(ts_raw)
                if ts.tzinfo is None:
                    ts = ts.tz_localize("UTC")
                session_hour_utc = int(ts.hour)
        except Exception:
            session_hour_utc = None

        # --- Momentum bias from EMA crossover [-1, 1] ---
        momentum_bias: float | None = None
        try:
            close_series_raw = payload.get("_close_series")
            if isinstance(close_series_raw, pd.Series) and len(close_series_raw) >= 50:
                ema_fast = close_series_raw.ewm(span=12, adjust=False).mean()
                ema_slow = close_series_raw.ewm(span=50, adjust=False).mean()
                last_slow = float(ema_slow.iloc[-1])
                if last_slow > 0:
                    momentum_bias = max(-1.0, min(1.0, (float(ema_fast.iloc[-1]) - last_slow) / last_slow * 100.0))
        except Exception:
            momentum_bias = None

        # --- ATR% for take-profit scaling ---
        atr_pct: float | None = None
        try:
            close_series_raw = payload.get("_close_series")
            high_series = payload.get("_high_series")
            low_series = payload.get("_low_series")
            if (
                isinstance(close_series_raw, pd.Series)
                and isinstance(high_series, pd.Series)
                and isinstance(low_series, pd.Series)
                and len(close_series_raw) >= 15
            ):
                tr = pd.concat([
                    high_series - low_series,
                    (high_series - close_series_raw.shift(1)).abs(),
                    (low_series - close_series_raw.shift(1)).abs(),
                ], axis=1).max(axis=1)
                atr_14 = float(tr.tail(14).mean())
                last_close = float(close_series_raw.iloc[-1])
                if last_close > 0:
                    atr_pct = atr_14 / last_close
        except Exception:
            atr_pct = None

        # --- Symbol accuracy from scorecard ---
        symbol_hit_rate = self.scorecard.get_hit_rate(symbol)
        accuracy_mult = self.scorecard.get_accuracy_multiplier(symbol)
        if accuracy_mult < 1.0:
            logger.info(
                "Scorecard dampening %s: hit_rate=%.2f, mult=%.2f",
                symbol,
                symbol_hit_rate if symbol_hit_rate is not None else -1.0,
                accuracy_mult,
            )

        # --- Event gate evaluation (Phase 2) ---
        event_gate_mult: float | None = None
        if self._cached_events and signal_type in ("BUY", "SELL"):
            gate_result = evaluate_event_gate(
                symbol=symbol,
                signal_direction=signal_type,
                events=self._cached_events,
            )
            if gate_result.has_event:
                event_gate_mult = gate_result.multiplier

        # --- Model agreement (Phase 3) ---
        model_agreement: float | None = self._last_model_agreement

        # --- Estimated transaction cost (Phase A) ---
        estimated_cost_bps: float | None = None
        if signal_type in ("BUY", "SELL"):
            try:
                _cost_model = get_default_cost_model()
                _edge_bps = confidence_to_edge_bps(confidence, uncertainty)
                _est = _cost_model.estimate(symbol, notional_usd=0.0)  # zero-notional for bps only
                estimated_cost_bps = _est.round_trip_cost_bps
            except Exception:
                estimated_cost_bps = None

        native_signal = StrategySignal(
            symbol=symbol,
            timeframe=self.anchor_interval,
            horizon_bars=self.horizon_bars,
            signal=signal_type,
            confidence=confidence,
            uncertainty=uncertainty,
            reason=reason,
            session_hour_utc=session_hour_utc,
            momentum_bias=momentum_bias,
            atr_pct=atr_pct,
            symbol_hit_rate=symbol_hit_rate,
            event_gate_mult=event_gate_mult,
            model_agreement=model_agreement,
            estimated_cost_bps=estimated_cost_bps,
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
        # Prefer the configured horizon
        candidates = (
            artifact_dir / f"model_{self.horizon_bars}m.pkl",
            artifact_dir / f"model_{self.horizon_bars}m.joblib",
            artifact_dir / "lgbm_model.joblib",
        )
        for candidate in candidates:
            if candidate.exists():
                return candidate
        # Fallback: use any available horizon model from a partial ensemble
        for horizon in (2, 4, 8):
            for suffix in ("pkl", "joblib"):
                fallback = artifact_dir / f"model_{horizon}m.{suffix}"
                if fallback.exists():
                    logger.info("Configured horizon=%dm not found; falling back to %s", self.horizon_bars, fallback.name)
                    return fallback
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
        # Always use production Binance for market data (klines, OI, funding).
        # The testnet does not support /futures/data/openInterestHist (returns 202).
        # Testnet is only needed for authenticated order execution endpoints.
        base_url = "https://fapi.binance.com"
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
            # Offset OI end time back by 5 min to avoid querying data
            # Binance hasn't aggregated yet (causes persistent HTTP 202).
            oi_date_to = date_to - timedelta(minutes=5)
            try:
                open_interest = fetch_open_interest(date_from, oi_date_to, symbol=symbol, period=interval)
            except TypeError:
                open_interest = fetch_open_interest(date_from, oi_date_to, symbol=symbol)
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
