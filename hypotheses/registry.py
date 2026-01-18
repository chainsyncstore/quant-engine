"""
Hypothesis registry.

Centralized registration and lookup for available hypotheses.
Prevents dynamic imports and ensures all hypotheses are explicitly registered.
"""

from typing import Dict, Type

from hypotheses.base import Hypothesis
from hypotheses.examples.always_long import AlwaysLongHypothesis
from hypotheses.examples.mean_reversion import MeanReversionHypothesis
from hypotheses.examples.simple_momentum import SimpleMomentumHypothesis
from hypotheses.examples.volatility_breakout import VolatilityBreakoutHypothesis
from hypotheses.examples.time_exit import TimeExitHypothesis
from hypotheses.examples.counter_trend import CounterTrendHypothesis
from hypotheses.volatility_breakout import VolatilityExpansionBreakout
from hypotheses.mean_reversion_exhaustion import MeanReversionExhaustion
from hypotheses.session_open_impulse import SessionOpenImpulse
from hypotheses.volatility_compression import VolatilityCompression
from hypotheses.competition.volatility_expansion_assault import VolatilityExpansionAssault
from hypotheses.competition.crypto_momentum_breakout import CryptoMomentumBreakout
from hypotheses.competition.rsi_extreme_reversal import RSIExtremeReversal
from hypotheses.competition.competition_hail_mary import CompetitionHailMary


class HypothesisRegistry:
    """
    Registry for available hypotheses.
    
    All hypotheses must be explicitly registered here to be usable
    by the orchestrator.
    """
    
    def __init__(self):
        """Initialize empty registry."""
        self._hypotheses: Dict[str, Type[Hypothesis]] = {}
    
    def register(self, hypothesis_class: Type[Hypothesis]) -> None:
        """
        Register a hypothesis class.
        
        Args:
            hypothesis_class: Hypothesis class to register
            
        Raises:
            ValueError: If a hypothesis with this ID is already registered
        """
        # Instantiate temporarily to get the ID
        instance = hypothesis_class()
        hypothesis_id = instance.hypothesis_id
        
        if hypothesis_id in self._hypotheses:
            raise ValueError(
                f"Hypothesis '{hypothesis_id}' is already registered"
            )
        
        self._hypotheses[hypothesis_id] = hypothesis_class
    
    def get(self, hypothesis_id: str) -> Type[Hypothesis]:
        """
        Get a hypothesis class by ID.
        
        Args:
            hypothesis_id: Hypothesis identifier
            
        Returns:
            Hypothesis class
            
        Raises:
            KeyError: If hypothesis is not registered
        """
        if hypothesis_id not in self._hypotheses:
            available = list(self._hypotheses.keys())
            raise KeyError(
                f"Hypothesis '{hypothesis_id}' not found. "
                f"Available hypotheses: {available}"
            )
        
        return self._hypotheses[hypothesis_id]
    
    def list_hypotheses(self) -> list[str]:
        """
        List all registered hypothesis IDs.
        
        Returns:
            List of hypothesis IDs
        """
        return list(self._hypotheses.keys())
    
    def is_registered(self, hypothesis_id: str) -> bool:
        """
        Check if a hypothesis is registered.
        
        Args:
            hypothesis_id: Hypothesis identifier
            
        Returns:
            True if registered, False otherwise
        """
        return hypothesis_id in self._hypotheses


# Global registry instance
registry = HypothesisRegistry()


registry.register(AlwaysLongHypothesis)
registry.register(MeanReversionHypothesis)
registry.register(SimpleMomentumHypothesis)
registry.register(VolatilityBreakoutHypothesis)
registry.register(TimeExitHypothesis)
registry.register(CounterTrendHypothesis)
registry.register(VolatilityExpansionBreakout)
registry.register(MeanReversionExhaustion)
registry.register(SessionOpenImpulse)
registry.register(VolatilityCompression)
registry.register(VolatilityExpansionAssault)
registry.register(CryptoMomentumBreakout)
registry.register(RSIExtremeReversal)
registry.register(CompetitionHailMary)


def get_hypothesis(hypothesis_id: str) -> Type[Hypothesis]:
    """
    Get a hypothesis class by ID.
    
    Args:
        hypothesis_id: Hypothesis identifier
        
    Returns:
        Hypothesis class
    """
    return registry.get(hypothesis_id)


def list_hypotheses() -> list[str]:
    """
    List all available hypothesis IDs.
    
    Returns:
        List of hypothesis IDs
    """
    return registry.list_hypotheses()
