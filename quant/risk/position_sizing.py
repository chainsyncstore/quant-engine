"""
Position sizing using fractional Kelly criterion.

Computes optimal bet size based on estimated edge and win rate,
with configurable caps to limit downside risk.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class PositionSize:
    """Result of position sizing calculation."""

    fraction: float       # Fraction of capital to risk (0-1)
    kelly_raw: float      # Raw Kelly fraction (unbounded)
    kelly_capped: float   # Kelly after cap
    units: float          # Position size in units (lots)
    reason: str           # Human-readable explanation


def kelly_fraction(
    win_rate: float,
    avg_win: float,
    avg_loss: float,
) -> float:
    """
    Compute the Kelly criterion fraction.

    Kelly = W - (1 - W) / R
    where W = win probability, R = win/loss ratio

    Args:
        win_rate: Historical win probability (0-1).
        avg_win: Average winning trade return (positive).
        avg_loss: Average losing trade return (positive, absolute value).

    Returns:
        Kelly fraction (can be negative if no edge).
    """
    if avg_loss <= 0 or avg_win <= 0:
        return 0.0

    r = avg_win / avg_loss  # Win/loss ratio
    kelly = win_rate - (1 - win_rate) / r

    return kelly


def compute_position_size(
    capital: float,
    win_rate: float,
    avg_win: float,
    avg_loss: float,
    pip_value: float = 10.0,
    kelly_divisor: float = 4.0,
    max_risk_fraction: float = 0.02,
    min_lots: float = 0.01,
    max_lots: float = 1.0,
) -> PositionSize:
    """
    Compute position size using fractional Kelly criterion.

    Uses Kelly / kelly_divisor (default: quarter-Kelly) for safety,
    capped at max_risk_fraction of capital per trade.

    Args:
        capital: Account equity in base currency.
        win_rate: Historical win probability from regime stats.
        avg_win: Average winning PnL per trade (in price units).
        avg_loss: Average losing PnL per trade (positive, in price units).
        pip_value: Value of 1 pip per standard lot (default: $10).
        kelly_divisor: Fraction of Kelly to use (4 = quarter-Kelly).
        max_risk_fraction: Maximum capital risk per trade (default: 2%).
        min_lots: Minimum position size.
        max_lots: Maximum position size.

    Returns:
        PositionSize with computed fraction and lot size.
    """
    # Compute raw Kelly
    raw_kelly = kelly_fraction(win_rate, avg_win, abs(avg_loss))

    if raw_kelly <= 0:
        return PositionSize(
            fraction=0.0,
            kelly_raw=raw_kelly,
            kelly_capped=0.0,
            units=0.0,
            reason=f"No edge: Kelly={raw_kelly:.4f} (WR={win_rate:.1%}, R={avg_win/max(abs(avg_loss), 1e-10):.2f})",
        )

    # Fractional Kelly (e.g., quarter-Kelly for safety)
    fractional_kelly = raw_kelly / kelly_divisor

    # Cap at max risk per trade
    risk_fraction = min(fractional_kelly, max_risk_fraction)

    # Convert to lots
    risk_amount = capital * risk_fraction
    # lot_size = risk_amount / (stop_loss_pips * pip_value)
    # For simplicity, use avg_loss as implied stop
    stop_pips = abs(avg_loss) / 0.0001  # Convert price to pips
    if stop_pips > 0:
        lots = risk_amount / (stop_pips * pip_value)
    else:
        lots = min_lots

    lots = max(min_lots, min(lots, max_lots))

    reason = (
        f"Kelly={raw_kelly:.4f}, {1/kelly_divisor:.0%}-Kelly={fractional_kelly:.4f}, "
        f"capped={risk_fraction:.4f}, lots={lots:.2f}"
    )

    logger.info(
        "Position size: %.2f lots (%.1f%% risk, Kelly=%.3f, WR=%.1f%%)",
        lots, risk_fraction * 100, raw_kelly, win_rate * 100,
    )

    return PositionSize(
        fraction=risk_fraction,
        kelly_raw=raw_kelly,
        kelly_capped=fractional_kelly,
        units=lots,
        reason=reason,
    )
