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

from quant.config import get_api_config, CapitalAPIConfig

logger = logging.getLogger(__name__)


class CapitalClient:
    """Client for Capital.com REST API v1."""

    def __init__(self, config: Optional[CapitalAPIConfig] = None) -> None:
        self._cfg = config if config else get_api_config()
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
        cache_dir: Optional[Path] = None,
    ) -> pd.DataFrame:
        """
        Fetch historical OHLCV data with automatic pagination and caching.

        Args:
            date_from: Start datetime (UTC).
            date_to: End datetime (UTC).
            epic: Instrument epic (default: EURUSD from config).
            resolution: Bar resolution (default: MINUTE from config).
            cache_dir: Directory to save/load partial chunks.

        Returns:
            DataFrame with columns [open, high, low, close, volume]
            and a UTC DatetimeIndex named 'timestamp'.
        """
        epic = epic or self._cfg.epic
        resolution = resolution or self._cfg.resolution

        all_frames: list[pd.DataFrame] = []
        current_from = date_from

        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)

        while current_from < date_to:
            # Check cache first
            chunk_loaded = False
            if cache_dir:
                # Filename based on start time (safe for filesystem)
                ts_str = current_from.strftime("%Y%m%d_%H%M%S")
                cache_file = cache_dir / f"chunk_{ts_str}.parquet"
                
                if cache_file.exists():
                    try:
                        chunk = pd.read_parquet(cache_file)
                        all_frames.append(chunk)
                        last_ts = chunk.index[-1]
                        current_from = last_ts.to_pydatetime() + pd.Timedelta(minutes=1)
                        logger.info(f"Loaded cached chunk: {ts_str} ({len(chunk)} bars)")
                        chunk_loaded = True
                    except Exception as e:
                        logger.warning(f"Failed to load cache {cache_file}: {e}")

            if not chunk_loaded:
                self._throttle()

                url = f"{self._cfg.base_url}/api/v1/prices/{epic}"
                params = {
                    "resolution": resolution,
                    "max": self._cfg.max_bars_per_request,
                    "from": current_from.strftime("%Y-%m-%dT%H:%M:%S"),
                }

                try:
                    resp = requests.get(url, params=params, headers=self._headers(), timeout=30)
                    resp.raise_for_status()
                    data = resp.json()
                except Exception as e:
                    logger.error(f"Request failed at {current_from}: {e}")
                    # If we have gathered some data, return it partial? 
                    # Or raise? Better to raise, but save what we have? 
                    # We are saving chunks, so "what we have" is safe on disk.
                    raise e

                prices = data.get("prices", [])
                if not prices:
                    break

                chunk = self._parse_prices(prices)
                
                # Save to cache
                if cache_dir:
                    ts_str = current_from.strftime("%Y%m%d_%H%M%S")
                    cache_file = cache_dir / f"chunk_{ts_str}.parquet"
                    chunk.to_parquet(cache_file)

                all_frames.append(chunk)

                # Move cursor forward
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
                    "bid_close": bid_c,
                    "ask_close": ask_c,
                    "spread": ask_c - bid_c,
                }
            )

        df = pd.DataFrame(records)
        if df.empty:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume", "bid_close", "ask_close", "spread"])
        return df.set_index("timestamp")

    # ------------------------------------------------------------------
    # Execution & Account Management
    # ------------------------------------------------------------------
    def get_accounts(self) -> dict:
        """Fetch account details (balance, equity, P&L)."""
        url = f"{self._cfg.base_url}/api/v1/accounts"
        resp = requests.get(url, headers=self._headers(), timeout=10)
        resp.raise_for_status()
        # Returns: {"accounts": [...]}
        data = resp.json()
        if "accounts" in data and len(data["accounts"]) > 0:
            return data["accounts"][0]  # Return primary account
        return {}

    def get_positions(self) -> list[dict]:
        """Fetch open positions."""
        url = f"{self._cfg.base_url}/api/v1/positions"
        resp = requests.get(url, headers=self._headers(), timeout=10)
        resp.raise_for_status()
        # Returns: {"positions": [...]}
        return resp.json().get("positions", [])

    def place_order(
        self,
        epic: str,
        direction: str,
        size: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
    ) -> dict:
        """
        Place a market order.
        
        Args:
            epic: Instrument epic (e.g., 'BTCUSD').
            direction: 'BUY' or 'SELL'.
            size: Lot size.
            stop_loss: Price level for SL.
            take_profit: Price level for TP.
        """
        url = f"{self._cfg.base_url}/api/v1/positions"
        payload = {
            "epic": epic,
            "direction": direction,
            "size": size,
            "guaranteedStop": False,
        }
        
        if stop_loss:
            payload["stopLevel"] = stop_loss
        if take_profit:
            payload["profitLevel"] = take_profit
            
        logger.info(f"Placing Order: {direction} {size} {epic}")
        resp = requests.post(url, json=payload, headers=self._headers(), timeout=10)
        
        if resp.status_code != 200:
            logger.error(f"Order failed: {resp.text}")
            
        resp.raise_for_status()
        return resp.json()

    def close_position(self, deal_id: str) -> dict:
        """Close a specific position by deal ID."""
        # Note: Capital.com /positions endpoint DELETE usually closes the position.
        url = f"{self._cfg.base_url}/api/v1/positions/{deal_id}"
        logger.info(f"Closing Position: {deal_id}")
        resp = requests.delete(url, headers=self._headers(), timeout=10)
        resp.raise_for_status()
        return resp.json()
