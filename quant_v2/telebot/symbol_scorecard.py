"""Per-symbol prediction accuracy scorecard for adaptive allocation dampening.

Records directional predictions and evaluates them against actual price
movement after the signal horizon expires.  Provides a rolling hit-rate
per symbol that the allocator uses as a soft multiplier.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class PredictionRecord:
    """One recorded directional prediction."""

    symbol: str
    direction: str  # "BUY" or "SELL"
    confidence: float
    entry_price: float
    predicted_at: datetime
    horizon_expires: datetime
    resolved: bool = False
    hit: bool = False


class SymbolScorecard:
    """Track rolling prediction accuracy per symbol.

    Parameters
    ----------
    lookback_hours:
        Only consider predictions from the last *lookback_hours* when
        computing hit rates.  Older resolved records are pruned.
    min_samples:
        Minimum resolved predictions required before a symbol gets a
        non-neutral accuracy multiplier.  Prevents overreacting to tiny
        sample sizes.
    """

    # --- Accuracy multiplier bands ---
    STRONG_HIT_RATE: float = 0.55
    WEAK_HIT_RATE: float = 0.45
    MULT_STRONG: float = 1.0
    MULT_NEUTRAL: float = 0.60
    MULT_WEAK: float = 0.30

    def __init__(
        self,
        *,
        lookback_hours: int = 72,
        min_samples: int = 8,
    ) -> None:
        self._lookback_hours = max(int(lookback_hours), 1)
        self._min_samples = max(int(min_samples), 1)
        self._predictions: dict[str, list[PredictionRecord]] = {}

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record_prediction(
        self,
        symbol: str,
        direction: str,
        confidence: float,
        entry_price: float,
        horizon_bars: int = 4,
        bar_interval_hours: float = 1.0,
    ) -> None:
        """Record a new actionable prediction."""
        if direction not in ("BUY", "SELL"):
            return
        if entry_price <= 0.0:
            return

        now = datetime.now(timezone.utc)
        expires = now + timedelta(hours=horizon_bars * bar_interval_hours)
        rec = PredictionRecord(
            symbol=symbol,
            direction=direction,
            confidence=confidence,
            entry_price=entry_price,
            predicted_at=now,
            horizon_expires=expires,
        )
        self._predictions.setdefault(symbol, []).append(rec)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate_pending(self, prices: dict[str, float]) -> int:
        """Resolve predictions whose horizon has expired.

        Returns the number of newly resolved predictions.
        """
        now = datetime.now(timezone.utc)
        resolved_count = 0

        for symbol, records in self._predictions.items():
            price = prices.get(symbol)
            if price is None or price <= 0.0:
                continue

            for rec in records:
                if rec.resolved:
                    continue
                if now < rec.horizon_expires:
                    continue

                if rec.direction == "BUY":
                    rec.hit = price > rec.entry_price
                else:
                    rec.hit = price < rec.entry_price
                rec.resolved = True
                resolved_count += 1

        if resolved_count > 0:
            self._prune_old()

        return resolved_count

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_hit_rate(self, symbol: str) -> float | None:
        """Return rolling hit rate for *symbol*, or None if insufficient data."""
        resolved = [r for r in self._predictions.get(symbol, []) if r.resolved]
        if len(resolved) < self._min_samples:
            return None
        hits = sum(1 for r in resolved if r.hit)
        return hits / len(resolved)

    def get_accuracy_multiplier(self, symbol: str) -> float:
        """Convert rolling hit rate into a soft allocation multiplier.

        Returns 1.0 (neutral) when there is insufficient data.
        """
        rate = self.get_hit_rate(symbol)
        if rate is None:
            return 1.0
        if rate >= self.STRONG_HIT_RATE:
            return self.MULT_STRONG
        if rate >= self.WEAK_HIT_RATE:
            return self.MULT_NEUTRAL
        return self.MULT_WEAK

    def get_summary(self) -> dict[str, dict[str, Any]]:
        """Return a per-symbol summary for diagnostics / Telegram display."""
        summary: dict[str, dict[str, Any]] = {}
        for symbol in sorted(self._predictions):
            records = self._predictions[symbol]
            resolved = [r for r in records if r.resolved]
            pending = [r for r in records if not r.resolved]
            hit_rate = self.get_hit_rate(symbol)
            summary[symbol] = {
                "resolved": len(resolved),
                "pending": len(pending),
                "hits": sum(1 for r in resolved if r.hit),
                "hit_rate": round(hit_rate, 3) if hit_rate is not None else None,
                "accuracy_mult": self.get_accuracy_multiplier(symbol),
            }
        return summary

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _prune_old(self) -> None:
        """Remove resolved predictions older than the lookback window."""
        cutoff = datetime.now(timezone.utc) - timedelta(hours=self._lookback_hours)
        for symbol in list(self._predictions):
            self._predictions[symbol] = [
                r
                for r in self._predictions[symbol]
                if not r.resolved or r.predicted_at >= cutoff
            ]
            if not self._predictions[symbol]:
                del self._predictions[symbol]

    def reset(self) -> None:
        """Clear all prediction history."""
        self._predictions.clear()
