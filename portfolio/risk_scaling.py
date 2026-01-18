"""
Regime-aware risk tiering for meta execution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from market.regime import RegimeConfidence


@dataclass(frozen=True)
class RiskTier:
    """
    Simple immutable container describing a risk tier.
    """

    label: str
    risk_fraction: float  # expressed as 0.xx of account equity
    leverage_multiplier: float = 1.0  # how much of max_leverage to use (0.0 to 1.0)


class RiskTierResolver:
    """
    Resolves regime confidences into fixed risk fractions under strict caps.
    """

    def __init__(
        self,
        tiers: Dict[RegimeConfidence, RiskTier] | None = None,
        max_leverage: float = 30.0,
    ) -> None:
        self._tiers = tiers or {
            # ALL-IN COMPETITION MODE: 10x leverage across all confidence levels
            # $10k account × 10x = $100k notional → 2% move = $2,000 profit
            RegimeConfidence.UNKNOWN: RiskTier(label="ALLIN", risk_fraction=1.0, leverage_multiplier=0.33),  # 1.0 × 10x = 10x notional
            RegimeConfidence.LOW: RiskTier(label="ALLIN", risk_fraction=1.0, leverage_multiplier=0.33),      # 1.0 × 10x = 10x notional
            RegimeConfidence.MEDIUM: RiskTier(label="ALLIN", risk_fraction=1.0, leverage_multiplier=0.33),   # 1.0 × 10x = 10x notional
            RegimeConfidence.HIGH: RiskTier(label="ALLIN", risk_fraction=1.0, leverage_multiplier=0.40),     # 1.0 × 12x = 12x notional (max confidence)
        }
        self.max_leverage = max(1.0, max_leverage)

    def resolve(self, confidence: RegimeConfidence) -> RiskTier:
        """
        Return the configured risk tier for the supplied confidence label.
        """
        return self._tiers.get(confidence, self._tiers[RegimeConfidence.LOW])
