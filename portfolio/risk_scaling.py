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
            RegimeConfidence.LOW: RiskTier(label="LOW", risk_fraction=0.03),       # 3%
            RegimeConfidence.MEDIUM: RiskTier(label="MEDIUM", risk_fraction=0.05), # 5%
            RegimeConfidence.HIGH: RiskTier(label="HIGH", risk_fraction=0.08),     # 8%
        }
        self.max_leverage = max(1.0, max_leverage)

    def resolve(self, confidence: RegimeConfidence) -> RiskTier:
        """
        Return the configured risk tier for the supplied confidence label.
        """
        return self._tiers.get(confidence, self._tiers[RegimeConfidence.LOW])
