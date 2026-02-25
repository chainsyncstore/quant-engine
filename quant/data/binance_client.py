"""
Binance Futures REST API client.

Read-only: Fetches OHLCV (with taker buy/sell volumes), funding rates, and OI.
Authenticated: Order placement, position management, account info.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import time
from datetime import datetime, timezone
from typing import Any, Optional
from urllib.parse import urlencode

import pandas as pd
import requests

from quant.config import get_binance_config, BinanceAPIConfig

logger = logging.getLogger(__name__)


class BinanceClient:
    """Client for Binance Futures REST API (read-only + authenticated trading)."""

    # Binance rate limit: 2400 weight/min. Klines = 2 weight each.
    _MIN_REQUEST_INTERVAL = 0.1  # 100ms between requests
    _BACKOFF_MAX_RETRIES = 5
    _BACKOFF_BASE_DELAY = 1.0
    _WEIGHT_THROTTLE_THRESHOLD = 1800  # 75% of 2400
    _WEIGHT_ELEVATED_INTERVAL = 0.5  # 500ms when weight is elevated

    def __init__(self, config: Optional[BinanceAPIConfig] = None) -> None:
        self._cfg = config if config else get_binance_config()
        self._last_request_time: float = 0.0
        self._exchange_info_cache: tuple[float, dict[str, Any]] | None = None
        self._used_weight_1m: int = 0
        self._time_offset_ms: int = 0  # Clock drift vs Binance server

    def _throttle(self) -> None:
        interval = (
            self._WEIGHT_ELEVATED_INTERVAL
            if self._used_weight_1m >= self._WEIGHT_THROTTLE_THRESHOLD
            else self._MIN_REQUEST_INTERVAL
        )
        elapsed = time.time() - self._last_request_time
        if elapsed < interval:
            time.sleep(interval - elapsed)
        self._last_request_time = time.time()

    def _update_rate_limit_weight(self, resp: requests.Response) -> None:
        """Track Binance request weight from response headers."""
        weight_str = resp.headers.get("X-MBX-USED-WEIGHT-1M", "")
        if weight_str:
            try:
                self._used_weight_1m = int(weight_str)
            except ValueError:
                pass

    def _request_with_backoff(
        self,
        method: str,
        url: str,
        params: dict,
        headers: dict | None = None,
    ) -> requests.Response:
        """Execute an HTTP request with exponential backoff on 429/5xx.

        - HTTP 429: honour Retry-After header, else exponential backoff.
        - HTTP 418: Binance IP ban — raise immediately, do NOT retry.
        - HTTP 5xx: retry with backoff up to 3 times.
        """
        delay = self._BACKOFF_BASE_DELAY
        request_fn = getattr(requests, method.lower())
        last_resp: requests.Response | None = None

        for attempt in range(self._BACKOFF_MAX_RETRIES):
            self._throttle()

            # Guard against socket-level panics (DNS fail, TCP RST, TLS
            # handshake failure, read timeout).  Without this, a transient
            # network blip raises an uncaught exception that propagates
            # through run_in_executor and kills the asyncio event loop.
            try:
                resp = request_fn(url, params=params, headers=headers, timeout=30)
            except requests.exceptions.RequestException as net_err:
                if attempt < self._BACKOFF_MAX_RETRIES - 1:
                    logger.warning(
                        "Network error (attempt %d/%d): %s — retrying in %.1fs",
                        attempt + 1,
                        self._BACKOFF_MAX_RETRIES,
                        net_err.__class__.__name__,
                        delay,
                    )
                    time.sleep(delay)
                    delay = min(delay * 2, 60.0)
                    continue
                logger.error(
                    "Network error exhausted retries (%d): %s",
                    self._BACKOFF_MAX_RETRIES,
                    net_err,
                )
                raise

            self._update_rate_limit_weight(resp)

            if resp.status_code == 200:
                return resp

            last_resp = resp

            # IP ban — abort immediately
            if resp.status_code == 418:
                logger.critical(
                    "BINANCE IP BAN (HTTP 418) — DO NOT RETRY. URL=%s", url
                )
                resp.raise_for_status()

            # Rate limit — back off
            if resp.status_code == 429:
                retry_after = int(resp.headers.get("Retry-After", delay))
                logger.warning(
                    "HTTP 429 Rate Limit (attempt %d/%d). Backing off for %ds. weight=%d",
                    attempt + 1,
                    self._BACKOFF_MAX_RETRIES,
                    retry_after,
                    self._used_weight_1m,
                )
                time.sleep(retry_after)
                delay = min(delay * 2, 60.0)
                continue

            # Server errors — retry with backoff (max 3)
            if resp.status_code >= 500 and attempt < 3:
                logger.warning(
                    "HTTP %d server error (attempt %d/3). Retrying in %.1fs.",
                    resp.status_code,
                    attempt + 1,
                    delay,
                )
                time.sleep(delay)
                delay = min(delay * 2, 30.0)
                continue

            # Any other non-200 — raise immediately
            break

        if last_resp is not None:
            self._handle_binance_error(last_resp)
            last_resp.raise_for_status()
        raise RuntimeError("_request_with_backoff exhausted retries with no response")

    def _get(self, url: str, params: dict) -> dict | list:
        resp = self._request_with_backoff("get", url, params)
        return resp.json()

    def get_exchange_info(self, *, cache_ttl_seconds: float = 300.0) -> dict[str, Any]:
        """Fetch and cache futures exchange metadata (filters, precision, etc.)."""

        now = time.time()
        cached = self._exchange_info_cache
        if cached is not None:
            cached_at, payload = cached
            if (now - cached_at) <= max(float(cache_ttl_seconds), 0.0):
                return payload

        url = f"{self._cfg.base_url}/fapi/v1/exchangeInfo"
        payload = self._get(url, {})
        if not isinstance(payload, dict):
            raise RuntimeError("Unexpected exchangeInfo payload type")

        self._exchange_info_cache = (now, payload)
        return payload

    def get_symbol_filters(self, symbol: str) -> dict[str, float]:
        """Return normalized quantity/notional filters for a futures symbol."""

        symbol_upper = str(symbol).strip().upper()
        if not symbol_upper:
            return {}

        info = self.get_exchange_info()
        symbols = info.get("symbols", []) if isinstance(info, dict) else []
        if not isinstance(symbols, list):
            return {}

        payload: dict[str, Any] | None = None
        for item in symbols:
            if not isinstance(item, dict):
                continue
            if str(item.get("symbol", "")).upper() == symbol_upper:
                payload = item
                break
        if payload is None:
            return {}

        filters = payload.get("filters", [])
        if not isinstance(filters, list):
            return {}

        parsed: dict[str, float] = {
            "step_size": 0.0,
            "min_qty": 0.0,
            "min_notional": 0.0,
        }
        for item in filters:
            if not isinstance(item, dict):
                continue
            filter_type = str(item.get("filterType", "")).upper()
            if filter_type == "LOT_SIZE":
                parsed["step_size"] = float(item.get("stepSize", 0.0) or 0.0)
                parsed["min_qty"] = float(item.get("minQty", 0.0) or 0.0)
            elif filter_type in {"MIN_NOTIONAL", "NOTIONAL"}:
                parsed["min_notional"] = float(item.get("notional", item.get("minNotional", 0.0)) or 0.0)
        return parsed

    # ------------------------------------------------------------------
    # OHLCV with taker buy/sell volumes
    # ------------------------------------------------------------------
    def fetch_historical(
        self,
        date_from: datetime,
        date_to: datetime,
        symbol: Optional[str] = None,
        interval: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Fetch historical klines (OHLCV + taker volumes).

        Binance kline response fields:
        [open_time, open, high, low, close, volume, close_time,
         quote_volume, trades, taker_buy_base_vol, taker_buy_quote_vol, ignore]

        Returns:
            DataFrame with columns: open, high, low, close, volume,
            taker_buy_volume, taker_sell_volume
            and a UTC DatetimeIndex named 'timestamp'.
        """
        symbol = symbol or self._cfg.symbol
        interval = interval or self._cfg.interval
        url = f"{self._cfg.base_url}/fapi/v1/klines"

        start_ms = int(date_from.timestamp() * 1000)
        end_ms = int(date_to.timestamp() * 1000)
        all_frames: list[pd.DataFrame] = []

        while start_ms < end_ms:
            params = {
                "symbol": symbol,
                "interval": interval,
                "startTime": start_ms,
                "endTime": end_ms,
                "limit": self._cfg.max_bars_per_request,
            }

            data = self._get(url, params)
            if not data:
                break

            chunk = self._parse_klines(data)
            all_frames.append(chunk)

            # Move cursor past last bar
            last_close_time = int(data[-1][6])  # close_time in ms
            start_ms = last_close_time + 1

            logger.info(
                "Fetched %d bars up to %s (%d total)",
                len(chunk),
                chunk.index[-1],
                sum(len(f) for f in all_frames),
            )

        if not all_frames:
            return pd.DataFrame(
                columns=["open", "high", "low", "close", "volume",
                         "taker_buy_volume", "taker_sell_volume"]
            )

        df = pd.concat(all_frames)
        df = df[~df.index.duplicated(keep="first")]
        df = df.sort_index()

        # Trim to exact requested range
        ts_from = pd.Timestamp(date_from).tz_localize("UTC") if date_from.tzinfo is None else pd.Timestamp(date_from)
        ts_to = pd.Timestamp(date_to).tz_localize("UTC") if date_to.tzinfo is None else pd.Timestamp(date_to)
        df = df[df.index >= ts_from]
        df = df[df.index <= ts_to]

        return df

    @staticmethod
    def _parse_klines(data: list) -> pd.DataFrame:
        records = []
        for k in data:
            ts = pd.Timestamp(int(k[0]), unit="ms", tz="UTC")
            vol = float(k[5])
            taker_buy = float(k[9])
            records.append({
                "timestamp": ts,
                "open": float(k[1]),
                "high": float(k[2]),
                "low": float(k[3]),
                "close": float(k[4]),
                "volume": vol,
                "taker_buy_volume": taker_buy,
                "taker_sell_volume": vol - taker_buy,
            })
        df = pd.DataFrame(records)
        return df.set_index("timestamp")

    # ------------------------------------------------------------------
    # Funding rates (8H intervals, forward-filled to match kline freq)
    # ------------------------------------------------------------------
    def fetch_funding_rates(
        self,
        date_from: datetime,
        date_to: datetime,
        symbol: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Fetch historical funding rates.

        Returns:
            DataFrame with 'funding_rate' column and UTC DatetimeIndex.
            Rates are at 8H intervals (00:00, 08:00, 16:00 UTC).
        """
        symbol = symbol or self._cfg.symbol
        url = f"{self._cfg.base_url}/fapi/v1/fundingRate"

        start_ms = int(date_from.timestamp() * 1000)
        end_ms = int(date_to.timestamp() * 1000)
        all_records: list[dict] = []

        while start_ms < end_ms:
            params = {
                "symbol": symbol,
                "startTime": start_ms,
                "endTime": end_ms,
                "limit": 1000,
            }
            data = self._get(url, params)
            if not data:
                break

            for item in data:
                all_records.append({
                    "timestamp": pd.Timestamp(int(item["fundingTime"]), unit="ms", tz="UTC"),
                    "funding_rate_raw": float(item["fundingRate"]),
                })

            # Move past last entry
            start_ms = int(data[-1]["fundingTime"]) + 1

            logger.info("Fetched %d funding rate entries", len(all_records))

        if not all_records:
            return pd.DataFrame(columns=["funding_rate_raw"])

        df = pd.DataFrame(all_records).set_index("timestamp")
        df = df[~df.index.duplicated(keep="first")]
        df = df.sort_index()
        return df

    # ------------------------------------------------------------------
    # Open interest (5-min intervals from Binance, resampled to kline freq)
    # ------------------------------------------------------------------
    def fetch_open_interest(
        self,
        date_from: datetime,
        date_to: datetime,
        symbol: Optional[str] = None,
        period: str = "1h",
    ) -> pd.DataFrame:
        """
        Fetch historical open interest.

        Uses /futures/data/openInterestHist (data available from ~2020).

        Returns:
            DataFrame with 'open_interest' column and UTC DatetimeIndex.
        """
        symbol = symbol or self._cfg.symbol
        url = f"{self._cfg.base_url}/futures/data/openInterestHist"

        start_ms = int(date_from.timestamp() * 1000)
        end_ms = int(date_to.timestamp() * 1000)
        all_records: list[dict] = []

        # Binance OI endpoint limits date range to ~30 days per request
        CHUNK_MS = 29 * 24 * 3600 * 1000  # 29 days in ms

        cursor = start_ms
        while cursor < end_ms:
            # We must cap the endTime of the query to +29 days to satisfy API constraints
            chunk_end = min(cursor + CHUNK_MS, end_ms)

            params = {
                "symbol": symbol,
                "period": period,
                "startTime": cursor,
                "endTime": chunk_end,
                "limit": 500,
            }
            try:
                data = self._get(url, params)
            except Exception as e:
                logger.warning("OI fetch failed at %s: %s", cursor, e)
                data = []

            if not data:
                # If no data is returned in this slice, we must still advance 
                # the cursor to avoid an infinite loop or skipping the entire rest of the period
                cursor = chunk_end + 1
                continue

            for item in data:
                all_records.append({
                    "timestamp": pd.Timestamp(int(item["timestamp"]), unit="ms", tz="UTC"),
                    "open_interest": float(item["sumOpenInterest"]),
                    "open_interest_value": float(item["sumOpenInterestValue"]),
                })

            new_cursor = int(data[-1]["timestamp"]) + 1
            if new_cursor <= cursor:
                # Fallback safeguard
                new_cursor = chunk_end + 1
            cursor = new_cursor

            logger.info("Fetched %d open interest entries so far", len(all_records))

        if not all_records:
            return pd.DataFrame(columns=["open_interest", "open_interest_value"])

        df = pd.DataFrame(all_records).set_index("timestamp")
        df = df[~df.index.duplicated(keep="first")]
        df = df.sort_index()
        return df

    # ------------------------------------------------------------------
    # Merge supplementary data into OHLCV
    # ------------------------------------------------------------------
    @staticmethod
    def merge_supplementary(
        ohlcv: pd.DataFrame,
        funding: pd.DataFrame,
        oi: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Left-join funding rates and open interest onto OHLCV by timestamp.

        Funding rates (8H) are forward-filled to 1H resolution.
        Open interest is joined directly (already 1H).
        """
        result = ohlcv.copy()

        # Merge funding rates (forward-fill since 8H -> 1H)
        if not funding.empty:
            result = result.join(funding, how="left")
            result["funding_rate_raw"] = result["funding_rate_raw"].ffill()

        # Merge open interest
        if not oi.empty:
            result = result.join(oi, how="left")
            result["open_interest"] = result["open_interest"].ffill()
            result["open_interest_value"] = result["open_interest_value"].ffill()

        return result

    # ==================================================================
    # Authenticated methods (order placement, positions, account)
    # ==================================================================

    def sync_time(self) -> None:
        """Calibrate local clock drift relative to Binance server time.

        Measures round-trip time to ``/fapi/v1/time`` and computes an
        offset so that ``_sign_params`` timestamps stay within the
        Binance ``recvWindow`` even if the host NTP drifts.
        """
        url = f"{self._cfg.base_url}/fapi/v1/time"
        t0 = int(time.time() * 1000)
        resp = self._get(url, {})
        t1 = int(time.time() * 1000)

        server_time = int(resp["serverTime"])
        rtt = t1 - t0
        self._time_offset_ms = server_time - (t0 + rtt // 2)
        logger.info(
            "Clock synced with Binance. Drift offset: %d ms (rtt=%d ms)",
            self._time_offset_ms,
            rtt,
        )

    def _sign_params(self, params: dict) -> dict:
        """Add timestamp and HMAC-SHA256 signature to request params."""
        params["timestamp"] = int(time.time() * 1000) + self._time_offset_ms
        params["recvWindow"] = self._cfg.recv_window
        query_string = urlencode(params)
        signature = hmac.new(
            self._cfg.api_secret.encode("utf-8"),
            query_string.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        params["signature"] = signature
        return params

    def _auth_headers(self) -> dict:
        """Return headers with API key for authenticated requests."""
        return {"X-MBX-APIKEY": self._cfg.api_key}

    def _signed_get(self, endpoint: str, params: dict | None = None) -> dict | list:
        """Authenticated GET request with backoff."""
        params = self._sign_params(params or {})
        url = f"{self._cfg.base_url}{endpoint}"
        resp = self._request_with_backoff("get", url, params, headers=self._auth_headers())
        return resp.json()

    def _signed_post(self, endpoint: str, params: dict | None = None) -> dict | list:
        """Authenticated POST request with backoff."""
        params = self._sign_params(params or {})
        url = f"{self._cfg.base_url}{endpoint}"
        resp = self._request_with_backoff("post", url, params, headers=self._auth_headers())
        return resp.json()

    def _signed_delete(self, endpoint: str, params: dict | None = None) -> dict | list:
        """Authenticated DELETE request with backoff."""
        params = self._sign_params(params or {})
        url = f"{self._cfg.base_url}{endpoint}"
        resp = self._request_with_backoff("delete", url, params, headers=self._auth_headers())
        return resp.json()

    @staticmethod
    def _handle_binance_error(resp: requests.Response) -> None:
        """Check response for Binance API errors and raise with context."""
        if resp.status_code == 200:
            return
        try:
            body = resp.json()
            code = body.get("code", "?")
            msg = body.get("msg", resp.text)
        except Exception:
            code, msg = resp.status_code, resp.text

        error_msg = f"Binance API error {code}: {msg}"
        logger.error(error_msg)
        resp.raise_for_status()

    # --- Account ---

    def authenticate(self) -> dict:
        """
        Verify API key validity by fetching account info.

        Raises on invalid credentials or network error.
        Returns account info dict.
        """
        if not self._cfg.api_key or not self._cfg.api_secret:
            raise RuntimeError(
                "Binance API key and secret are required for authenticated operations. "
                "Set them via /setup or BINANCE_API_KEY/BINANCE_API_SECRET env vars."
            )
        # Calibrate clock before any authenticated request
        self.sync_time()
        account = self.get_account_info()
        logger.info(
            "Binance auth OK: USDT balance=%.2f, positions=%d",
            float(account.get("totalWalletBalance", 0)),
            sum(1 for p in account.get("positions", []) if float(p.get("positionAmt", 0)) != 0),
        )
        return account

    def get_account_info(self) -> dict:
        """Fetch account balance and position info. GET /fapi/v2/account."""
        return self._signed_get("/fapi/v2/account")

    def get_positions(self, symbol: Optional[str] = None) -> list[dict]:
        """
        Get open positions (non-zero quantity).

        Returns list of dicts with keys: symbol, positionAmt, entryPrice,
        unrealizedProfit, positionSide, leverage, etc.
        """
        account = self.get_account_info()
        positions = [
            p for p in account.get("positions", [])
            if float(p.get("positionAmt", 0)) != 0
        ]
        if symbol:
            positions = [p for p in positions if p["symbol"] == symbol]
        return positions

    # --- Orderbook ---

    def get_orderbook(self, symbol: str, limit: int = 5) -> dict:
        """Fetch depth-of-book (bids/asks). GET /fapi/v1/depth."""
        url = f"{self._cfg.base_url}/fapi/v1/depth"
        params = {"symbol": symbol, "limit": limit}
        return self._get(url, params)

    # --- Orders ---

    def place_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        order_type: str = "MARKET",
    ) -> dict:
        """
        Place a futures order. POST /fapi/v1/order.

        Args:
            symbol: e.g. "BTCUSDT"
            side: "BUY" or "SELL"
            quantity: Order size in base asset (BTC). Rounded to 3 decimals.
            order_type: "MARKET" or "LIMIT"

        Returns:
            Order response dict with orderId, status, etc.
        """
        quantity = round(quantity, 3)
        if quantity <= 0:
            raise ValueError(f"Invalid order quantity: {quantity}")

        params = {
            "symbol": symbol,
            "side": side,
            "type": order_type,
            "quantity": quantity,
        }

        logger.info("Placing %s %s order: %s %.3f", order_type, side, symbol, quantity)
        result = self._signed_post("/fapi/v1/order", params)
        logger.info(
            "Order placed: orderId=%s, status=%s, avgPrice=%s",
            result.get("orderId"), result.get("status"), result.get("avgPrice"),
        )
        return result

    def place_limit_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        time_in_force: str = "GTC",
        post_only: bool = True,
    ) -> dict:
        """
        Place a LIMIT order. POST /fapi/v1/order.
        """
        quantity = round(quantity, 3)
        if quantity <= 0:
            raise ValueError(f"Invalid order quantity: {quantity}")
        if price <= 0:
            raise ValueError(f"Invalid order price: {price}")
            
        params = {
            "symbol": symbol,
            "side": side,
            "type": "LIMIT",
            "quantity": quantity,
            "price": price,
            "timeInForce": "GTX" if post_only else time_in_force,
        }
        
        logger.info(
            "Placing %s %s order: %s %.3f at price %s", 
            "POST_ONLY LIMIT" if post_only else "LIMIT", 
            side, symbol, quantity, price
        )
        result = self._signed_post("/fapi/v1/order", params)
        logger.info(
            "Order placed: orderId=%s, status=%s, price=%s",
            result.get("orderId"), result.get("status"), result.get("price")
        )
        return result
        
    def cancel_order(self, symbol: str, order_id: str | int) -> dict:
        """Cancel an active order by ID. DELETE /fapi/v1/order."""
        params = {"symbol": symbol, "orderId": order_id}
        logger.info("Canceling order %s for %s", order_id, symbol)
        result = self._signed_delete("/fapi/v1/order", params)
        return result
        
    def get_open_orders(self, symbol: Optional[str] = None) -> list[dict]:
        """Get all open orders for a symbol. GET /fapi/v1/openOrders."""
        params = {}
        if symbol:
            params["symbol"] = symbol
        return self._signed_get("/fapi/v1/openOrders", params)

    def close_position(self, symbol: str, limit_price: float | None = None) -> Optional[dict]:
        """
        Close any open position for the symbol by placing an opposing market order.

        Returns order response or None if no position to close.
        """
        positions = self.get_positions(symbol)
        if not positions:
            logger.info("No open position for %s to close", symbol)
            return None

        pos = positions[0]
        pos_amt = float(pos["positionAmt"])

        # Opposing side: if long (posAmt > 0) → SELL, if short (posAmt < 0) → BUY
        if pos_amt > 0:
            side = "SELL"
            qty = pos_amt
        elif pos_amt < 0:
            side = "BUY"
            qty = abs(pos_amt)
        else:
            return None

        if limit_price is not None:
            logger.info("Closing %s position: LIMIT %s %.3f at %s", symbol, side, qty, limit_price)
            # Typically when closing, we don't strictly require post-only to guarantee execution
            return self.place_limit_order(symbol, side, qty, limit_price, "GTC", post_only=False)
        else:
            logger.info("Closing %s position: MARKET %s %.3f", symbol, side, qty)
            return self.place_order(symbol, side, qty, "MARKET")

    # --- Account setup ---

    def set_leverage(self, symbol: str, leverage: int) -> dict:
        """Set position leverage. POST /fapi/v1/leverage."""
        params = {"symbol": symbol, "leverage": leverage}
        logger.info("Setting leverage: %s %dx", symbol, leverage)
        return self._signed_post("/fapi/v1/leverage", params)

    def set_margin_type(self, symbol: str, margin_type: str = "ISOLATED") -> dict:
        """
        Set margin type (ISOLATED or CROSSED). POST /fapi/v1/marginType.

        Note: Raises if margin type is already set to the requested value.
        This is a Binance API quirk — we catch and ignore that specific error.
        """
        params = {"symbol": symbol, "marginType": margin_type}
        logger.info("Setting margin type: %s %s", symbol, margin_type)
        try:
            return self._signed_post("/fapi/v1/marginType", params)
        except requests.HTTPError as e:
            # Binance returns -4046 "No need to change margin type" if already set
            if "-4046" in str(e) or "No need to change" in str(e):
                logger.info("Margin type already %s for %s", margin_type, symbol)
                return {"msg": "already set"}
            raise
