"""
Data schemas and validation.

Defines the structure of market data passing through the system.
"""

from datetime import datetime
from pydantic import BaseModel, ConfigDict, Field, model_validator


class Bar(BaseModel):
    """
    Market data bar (OHLCV).
    
    Immutable and strictly validated.
    """
    model_config = ConfigDict(frozen=True)
    
    timestamp: datetime
    open: float = Field(gt=0.0)
    high: float = Field(gt=0.0)
    low: float = Field(gt=0.0)
    close: float = Field(gt=0.0)
    volume: float = Field(ge=0.0)
    symbol: str | None = Field(default=None, description="Optional market symbol")
    
    @model_validator(mode='after')
    def validate_prices(self) -> 'Bar':
        """
        Validate price consistency.
        
        Evaluates:
        - High >= Low
        - High >= Open
        - High >= Close
        - Low <= Open
        - Low <= Close
        """
        if self.high < self.low:
            raise ValueError(f"High ({self.high}) cannot be less than Low ({self.low})")
        
        if self.high < self.open:
            raise ValueError(f"High ({self.high}) cannot be less than Open ({self.open})")
        
        if self.high < self.close:
            raise ValueError(f"High ({self.high}) cannot be less than Close ({self.close})")
        
        if self.low > self.open:
            raise ValueError(f"Low ({self.low}) cannot be greater than Open ({self.open})")
        
        if self.low > self.close:
            raise ValueError(f"Low ({self.low}) cannot be greater than Close ({self.close})")
            
        return self
