"""
Competition-aggressive leverage configuration for prop firm ranking events.

WARNING: These settings are designed for SHORT-TERM COMPETITION USE ONLY.
High leverage = high risk. Do NOT use for funded accounts or real capital.
"""

from market.regime import RegimeConfidence
from portfolio.risk_scaling import RiskTier, RiskTierResolver


def get_competition_risk_resolver(max_leverage: float = 30.0) -> RiskTierResolver:
    """
    Returns a RiskTierResolver configured for aggressive competition trading.
    
    Position sizing examples with $10,000 equity and max_leverage=30:
    
    | Confidence | risk_frac | lev_mult | effective_lev | notional   |
    |------------|-----------|----------|---------------|------------|
    | UNKNOWN    | 0.25      | 0.6      | 18x           | $45,000    |
    | LOW        | 0.20      | 0.5      | 15x           | $30,000    |
    | MEDIUM     | 0.30      | 0.7      | 21x           | $63,000    |
    | HIGH       | 0.40      | 0.85     | 25.5x         | $102,000   |
    
    With HIGH confidence signals, you'd control ~10x your equity in notional.
    A 1% move = 10% account gain/loss.
    """
    # COMPETITION HAIL MARY: MAX LEVERAGE ON ALL SIGNALS
    competition_tiers = {
        RegimeConfidence.UNKNOWN: RiskTier(
            label="COMP_UNKNOWN",
            risk_fraction=0.40,
            leverage_multiplier=1.0,  # 30x effective - FULL
        ),
        RegimeConfidence.LOW: RiskTier(
            label="COMP_LOW",
            risk_fraction=0.35,
            leverage_multiplier=0.95,  # 28.5x effective
        ),
        RegimeConfidence.MEDIUM: RiskTier(
            label="COMP_MEDIUM",
            risk_fraction=0.45,
            leverage_multiplier=1.0,  # 30x effective - FULL
        ),
        RegimeConfidence.HIGH: RiskTier(
            label="COMP_HIGH",
            risk_fraction=0.50,
            leverage_multiplier=1.0,  # 30x effective - FULL
        ),
    }
    return RiskTierResolver(tiers=competition_tiers, max_leverage=max_leverage)


# Pre-built resolver for quick import
COMPETITION_RISK_RESOLVER = get_competition_risk_resolver(max_leverage=30.0)
