"""Deprecated legacy client stub (disabled in crypto-only runtime)."""

from __future__ import annotations

from typing import Optional


class CapitalClient:
    """Deprecated Capital.com client placeholder (disabled)."""

    def __init__(self, config: Optional[object] = None) -> None:
        _ = config
        raise RuntimeError(
            "CapitalClient is deprecated and disabled. "
            "This codebase is crypto-only (Binance)."
        )

    def __getattr__(self, name: str) -> object:
        raise RuntimeError(
            f"CapitalClient.{name} is unavailable. "
            "This codebase is crypto-only (Binance)."
        )
