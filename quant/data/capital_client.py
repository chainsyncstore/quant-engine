"""
Capital.com REST API client for historical OHLCV data.

Handles authentication, pagination, and rate limiting.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Optional

import pandas as pd
import requests

from quant.config import get_api_config

logger = logging.getLogger(__name__)


class CapitalClient:
    """Client for Capital.com REST API v1."""

    def __init__(self) -> None:
        self._cfg = get_api_config()
        self._session_token: Optional[str] = None
        self._cst: Optional[str] = None
        self._last_request_time: float = 0.0

    # ------------------------------------------------------------------
    # Authentication
    # ------------------------------------------------------------------
    def authenticate(self) -> None:
        """Create a new API session."""
        url = f"{self._cfg.base_url}/api/v1/session"
        headers = {"X-CAP-API-KEY": self._cfg.api_key, "Content-Type": "application/json"}
        payload = {
            "identifier": self._cfg.identifier,
            "password": self._cfg.password,
        }

        resp = requests.post(url, json=payload, headers=headers, timeout=30)
        resp.raise_for_status()

        self._cst = resp.headers.get("CST")
        self._session_token = resp.headers.get("X-SECURITY-TOKEN")
        logger.info("Authenticated with Capital.com API")

    # ------------------------------------------------------------------
    # Rate limiting
    # ------------------------------------------------------------------
    def _throttle(self) -> None:
        """Enforce rate limit (max requests per second)."""
        elapsed = time.time() - self._last_request_time
        min_interval = 1.0 / self._cfg.rate_limit_per_sec
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        self._last_request_time = time.time()

    # ------------------------------------------------------------------
    # Auth headers
    # ------------------------------------------------------------------
    def _headers(self) -> dict:
        if not self._cst or not self._session_token:
            raise RuntimeError("Not authenticated. Call authenticate() first.")
        return {
            "CST": self._cst,
            "X-SECURITY-TOKEN": self._session_token,
            "Content-Type": "application/json",
        }

    # ------------------------------------------------------------------
    # Historical prices
    # ------------------------------------------------------------------
    def fetch_historical(
        self,
        date_from: datetime,
        date_to: datetime,
        epic: Optional[str] = None,
        resolution: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Fetch historical OHLCV data with automatic pagination.

        Args:
            date_from: Start datetime (UTC).
            date_to: End datetime (UTC).
            epic: Instrument epic (default: EURUSD from config).
            resolution: Bar resolution (default: MINUTE from config).

        Returns:
            DataFrame with columns [open, high, low, close, volume]
            and a UTC DatetimeIndex named 'timestamp'.
        """
        epic = epic or self._cfg.epic
        resolution = resolution or self._cfg.resolution

        all_frames: list[pd.DataFrame] = []
        current_from = date_from

        while current_from < date_to:
            self._throttle()

            url = f"{self._cfg.base_url}/api/v1/prices/{epic}"
            params = {
                "resolution": resolution,
                "max": self._cfg.max_bars_per_request,
                "from": current_from.strftime("%Y-%m-%dT%H:%M:%S"),
                "to": date_to.strftime("%Y-%m-%dT%H:%M:%S"),
            }

            resp = requests.get(url, params=params, headers=self._headers(), timeout=30)
            resp.raise_for_status()
            data = resp.json()

            prices = data.get("prices", [])
            if not prices:
                break

            chunk = self._parse_prices(prices)
            all_frames.append(chunk)

            # Move cursor forward past the last bar received
            last_ts = chunk.index[-1]
            current_from = last_ts.to_pydatetime() + pd.Timedelta(minutes=1)

            logger.info(
                "Fetched %d bars up to %s (%d total so far)",
                len(chunk),
                last_ts,
                sum(len(f) for f in all_frames),
            )

        if not all_frames:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        df = pd.concat(all_frames)
        df = df[~df.index.duplicated(keep="first")]
        df = df.sort_index()
        return df

    # ------------------------------------------------------------------
    # Parse API response
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_prices(prices: list[dict]) -> pd.DataFrame:
        """Convert API price list to DataFrame.

        Capital.com returns bid/ask OHLC — we use the mid-price.
        """
        records = []
        for p in prices:
            ts = pd.Timestamp(p["snapshotTime"], tz="UTC")
            bid = p.get("closePrice", p.get("bidPrice", {}))
            ask = p.get("highPrice", p.get("askPrice", {}))

            # Extract OHLC from bid/ask mid-price
            bid_o = float(p.get("openPrice", {}).get("bid", 0))
            ask_o = float(p.get("openPrice", {}).get("ask", 0))
            bid_h = float(p.get("highPrice", {}).get("bid", 0))
            ask_h = float(p.get("highPrice", {}).get("ask", 0))
            bid_l = float(p.get("lowPrice", {}).get("bid", 0))
            ask_l = float(p.get("lowPrice", {}).get("ask", 0))
            bid_c = float(p.get("closePrice", {}).get("bid", 0))
            ask_c = float(p.get("closePrice", {}).get("ask", 0))

            records.append(
                {
                    "timestamp": ts,
                    "open": (bid_o + ask_o) / 2,
                    "high": (bid_h + ask_h) / 2,
                    "low": (bid_l + ask_l) / 2,
                    "close": (bid_c + ask_c) / 2,
                    "volume": float(p.get("lastTradedVolume", 0)),
                }
            )

        df = pd.DataFrame(records).set_index("timestamp")
        return df
