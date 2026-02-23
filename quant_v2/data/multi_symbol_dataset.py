"""Multi-symbol dataset fetch helpers for the v2 pipeline."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Iterable, Protocol

import pandas as pd

from quant.data.binance_client import BinanceClient

logger = logging.getLogger(__name__)


class MarketDataClient(Protocol):
    """Minimal client contract used by dataset fetch helpers."""

    def fetch_historical(
        self,
        date_from: datetime,
        date_to: datetime,
        symbol: str | None = None,
        interval: str | None = None,
    ) -> pd.DataFrame:
        ...

    def fetch_funding_rates(
        self,
        date_from: datetime,
        date_to: datetime,
        symbol: str | None = None,
    ) -> pd.DataFrame:
        ...

    def fetch_open_interest(
        self,
        date_from: datetime,
        date_to: datetime,
        symbol: str | None = None,
        period: str = "1h",
    ) -> pd.DataFrame:
        ...


def fetch_symbol_dataset(
    symbol: str,
    *,
    date_from: datetime,
    date_to: datetime,
    interval: str = "1h",
    include_funding: bool = True,
    include_open_interest: bool = True,
    client: MarketDataClient | None = None,
) -> pd.DataFrame:
    """Fetch OHLCV + supplementary market data for a single symbol."""

    if not symbol:
        raise ValueError("symbol cannot be empty")

    market_client = client or BinanceClient()

    ohlcv = market_client.fetch_historical(
        date_from=date_from,
        date_to=date_to,
        symbol=symbol,
        interval=interval,
    )
    if ohlcv.empty:
        logger.warning("No OHLCV data for symbol %s", symbol)
        return ohlcv

    funding = (
        market_client.fetch_funding_rates(date_from=date_from, date_to=date_to, symbol=symbol)
        if include_funding
        else pd.DataFrame(columns=["funding_rate_raw"])
    )
    open_interest = (
        market_client.fetch_open_interest(
            date_from=date_from,
            date_to=date_to,
            symbol=symbol,
            period=interval,
        )
        if include_open_interest
        else pd.DataFrame(columns=["open_interest", "open_interest_value"])
    )

    merged = BinanceClient.merge_supplementary(ohlcv=ohlcv, funding=funding, oi=open_interest)
    merged.index.name = "timestamp"
    return merged


def fetch_universe_dataset(
    symbols: Iterable[str],
    *,
    date_from: datetime,
    date_to: datetime,
    interval: str = "1h",
    include_funding: bool = True,
    include_open_interest: bool = True,
    fail_fast: bool = False,
    client: MarketDataClient | None = None,
) -> pd.DataFrame:
    """Fetch and merge market data for multiple symbols into one MultiIndex frame."""

    symbol_list = [s.strip().upper() for s in symbols if s and s.strip()]
    if not symbol_list:
        raise ValueError("symbols cannot be empty")

    frames: list[pd.DataFrame] = []
    market_client = client or BinanceClient()

    for symbol in symbol_list:
        try:
            sym_df = fetch_symbol_dataset(
                symbol,
                date_from=date_from,
                date_to=date_to,
                interval=interval,
                include_funding=include_funding,
                include_open_interest=include_open_interest,
                client=market_client,
            )
            if sym_df.empty:
                logger.warning("Skipping %s due to empty dataset", symbol)
                continue

            with_symbol = sym_df.copy()
            with_symbol["symbol"] = symbol
            with_symbol = with_symbol.reset_index().set_index(["timestamp", "symbol"]).sort_index()
            frames.append(with_symbol)
        except Exception as exc:
            logger.exception("Failed fetching symbol %s", symbol)
            if fail_fast:
                raise
            logger.warning("Continuing after symbol error for %s: %s", symbol, exc)

    if not frames:
        empty_index = pd.MultiIndex(
            levels=[[], []],
            codes=[[], []],
            names=["timestamp", "symbol"],
        )
        return pd.DataFrame(index=empty_index)

    combined = pd.concat(frames).sort_index()
    combined.index = combined.index.set_names(["timestamp", "symbol"])
    return combined
