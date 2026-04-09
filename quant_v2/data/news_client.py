"""Lightweight news/event fetcher for traded symbols.

Data sources:
  1. CryptoCompare News API — free tier, 100k calls/month.
     Docs: https://min-api.cryptocompare.com/documentation
  2. Alternative.me Fear & Greed Index — free, no API key.
     Docs: https://alternative.me/crypto/fear-and-greed-index/
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CryptoCompare News API
# ---------------------------------------------------------------------------
_CRYPTOCOMPARE_NEWS_URL = "https://min-api.cryptocompare.com/data/v2/news/"

# ---------------------------------------------------------------------------
# Alternative.me Fear & Greed Index
# ---------------------------------------------------------------------------
_FEAR_GREED_URL = "https://api.alternative.me/fng/"

# Weighted keyword lists for headline sentiment classification.
# High-impact keywords (weight 3) represent clear market-moving events.
# Moderate keywords (weight 1) represent softer directional signals.
_BEARISH_KEYWORDS: dict[str, int] = {
    # High-impact (weight 3)
    "hack": 3, "hacked": 3, "exploit": 3, "breach": 3, "crash": 3,
    "ban": 3, "banned": 3, "fraud": 3, "bankrupt": 3, "rug pull": 3,
    "rugpull": 3, "scam": 3, "indictment": 3, "arrest": 3,
    # Moderate (weight 1)
    "sec": 1, "lawsuit": 1, "liquidat": 1, "delist": 1, "sanction": 1,
    "investigation": 1, "vulnerability": 1, "attack": 1,
    "insolvency": 1, "default": 1,
}
_BULLISH_KEYWORDS: dict[str, int] = {
    # High-impact (weight 3)
    "etf": 3, "approval": 3, "approved": 3, "record high": 3, "ath": 3,
    "institutional": 3, "inflow": 3,
    # Moderate (weight 1)
    "partnership": 1, "upgrade": 1, "launch": 1, "adoption": 1,
    "bullish": 1, "integration": 1, "listing": 1, "listed": 1,
    "rally": 1, "fund": 1, "accumulation": 1, "acquisition": 1,
}


@dataclass(frozen=True)
class NewsEvent:
    """A single news event relevant to a symbol."""

    symbol: str
    title: str
    source: str
    published_at: datetime
    sentiment: str          # "bullish", "bearish", "neutral"
    severity: str           # "low", "medium", "high"
    url: str = ""


class CryptoCompareNewsClient:
    """Fetch recent crypto news from CryptoCompare News API.

    Requires a free API key from https://www.cryptocompare.com/cryptopian/api-keys
    Set via environment variable CRYPTOCOMPARE_API_KEY.
    """

    def __init__(self, api_key: str, timeout: int = 10) -> None:
        self._api_key = api_key
        self._timeout = timeout

    def fetch_recent(
        self,
        symbols: list[str] | None = None,
        max_results: int = 20,
    ) -> list[NewsEvent]:
        """Fetch recent news, optionally filtered by symbol base tickers.

        Parameters
        ----------
        symbols : list[str] | None
            Base tickers, e.g. ["BTC", "ETH"]. Pass None for all crypto news.
        max_results : int
            Maximum events to return.

        Returns
        -------
        list[NewsEvent]
            Parsed events sorted by recency.
        """
        params: dict[str, Any] = {
            "lang": "EN",
            "sortOrder": "latest",
        }
        if symbols:
            params["categories"] = ",".join(symbols)

        headers = {"authorization": f"Apikey {self._api_key}"}

        try:
            resp = requests.get(
                _CRYPTOCOMPARE_NEWS_URL,
                params=params,
                headers=headers,
                timeout=self._timeout,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.warning("CryptoCompare news fetch failed: %s", e)
            return []

        events: list[NewsEvent] = []
        for item in (data.get("Data") or [])[:max_results]:
            title = item.get("title", "")
            body = item.get("body", "")
            headline = f"{title} {body}".lower()

            # Keyword-based sentiment classification
            sentiment = _classify_headline_sentiment(headline)

            # Severity heuristic: source reputation + category tags
            categories = item.get("categories", "").lower()
            source_name = item.get("source_info", {}).get("name", "unknown")
            severity = _classify_severity(categories, source_name)

            # Map categories to USDT pairs
            category_list = [c.strip().upper() for c in item.get("categories", "").split("|") if c.strip()]
            target_symbols = _categories_to_usdt_symbols(category_list, symbols)

            published_ts = item.get("published_on", 0)
            published_at = (
                datetime.fromtimestamp(published_ts, tz=timezone.utc)
                if published_ts
                else datetime.now(timezone.utc)
            )

            for sym in target_symbols:
                events.append(NewsEvent(
                    symbol=sym,
                    title=title,
                    source=source_name,
                    published_at=published_at,
                    sentiment=sentiment,
                    severity=severity,
                    url=item.get("url", ""),
                ))

        return events


class FearGreedClient:
    """Fetch the global crypto Fear & Greed Index from Alternative.me.

    No API key required. Returns a synthetic NewsEvent representing the
    current macro sentiment, applicable to all symbols.
    """

    def __init__(self, timeout: int = 10) -> None:
        self._timeout = timeout

    def fetch_current(self) -> NewsEvent | None:
        """Fetch the latest Fear & Greed Index value.

        Returns a single NewsEvent with:
        - symbol = "MARKET" (global, applies to all)
        - sentiment derived from the index value
        - severity = "high" for extreme readings, "medium" for moderate
        """
        try:
            resp = requests.get(
                _FEAR_GREED_URL,
                params={"limit": "1"},
                timeout=self._timeout,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.warning("Fear & Greed fetch failed: %s", e)
            return None

        entries = data.get("data") or []
        if not entries:
            return None

        entry = entries[0]
        value = int(entry.get("value", 50))
        classification = entry.get("value_classification", "Neutral")
        ts = int(entry.get("timestamp", 0))

        # Derive sentiment and severity from index value
        if value <= 20:
            sentiment, severity = "bearish", "high"      # Extreme Fear
        elif value <= 35:
            sentiment, severity = "bearish", "medium"    # Fear
        elif value >= 80:
            sentiment, severity = "bullish", "high"      # Extreme Greed
        elif value >= 65:
            sentiment, severity = "bullish", "medium"    # Greed
        else:
            sentiment, severity = "neutral", "low"       # Neutral zone

        return NewsEvent(
            symbol="MARKET",
            title=f"Fear & Greed Index: {value} ({classification})",
            source="alternative.me",
            published_at=datetime.fromtimestamp(ts, tz=timezone.utc) if ts else datetime.now(timezone.utc),
            sentiment=sentiment,
            severity=severity,
            url="https://alternative.me/crypto/fear-and-greed-index/",
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _classify_headline_sentiment(headline: str) -> str:
    """Classify a headline as bullish, bearish, or neutral via weighted keyword scoring.

    High-impact keywords (hack, crash, ETF approval) contribute weight 3;
    moderate keywords contribute weight 1. This ensures a single high-impact
    keyword outweighs multiple soft signals.
    """
    bearish_score = sum(w for kw, w in _BEARISH_KEYWORDS.items() if kw in headline)
    bullish_score = sum(w for kw, w in _BULLISH_KEYWORDS.items() if kw in headline)
    if bearish_score > bullish_score:
        return "bearish"
    if bullish_score > bearish_score:
        return "bullish"
    return "neutral"


def _classify_severity(categories: str, source_name: str) -> str:
    """Heuristic severity from category tags and source reputation."""
    high_categories = {"regulation", "security", "hack", "etf", "legal"}
    if any(cat in categories for cat in high_categories):
        return "high"
    reputable_sources = {"coindesk", "cointelegraph", "the block", "bloomberg", "reuters"}
    if source_name.lower() in reputable_sources:
        return "medium"
    return "low"


def _categories_to_usdt_symbols(
    category_list: list[str],
    requested_tickers: list[str] | None,
) -> list[str]:
    """Map CryptoCompare category tags to USDT trading pair symbols.

    CryptoCompare tags articles with pipe-separated categories like "BTC|ETH|Trading".
    We match these against known base tickers.
    """
    known_bases = {"BTC", "ETH", "BNB", "XRP", "SOL", "ADA", "DOGE", "AVAX", "LINK", "LTC"}
    if requested_tickers:
        known_bases = known_bases | {t.upper() for t in requested_tickers}

    symbols: list[str] = []
    for cat in category_list:
        if cat in known_bases:
            symbols.append(f"{cat}USDT")
    return symbols if symbols else []


def symbol_to_base_ticker(usdt_symbol: str) -> str:
    """Convert 'BTCUSDT' to 'BTC'."""
    return usdt_symbol.replace("USDT", "").replace("BUSD", "")
