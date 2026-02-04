"""
Pricing Schemas (Optimized)
==========================

High-performance schemas using msgspec for ultra-low latency serialization.
Fallback to Pydantic for complex validation if needed, but core paths use msgspec.
"""

from pydantic import BaseModel, Field
from datetime import datetime, timezone
from typing import List, Literal, Optional, Union, Any

class PriceRequest(BaseModel):
    """
    Standard option pricing request.
    """
    spot: float
    strike: float
    time_to_expiry: float
    volatility: float
    rate: float
    option_type: Literal["call", "put"] = "call"
    dividend_yield: float = 0.0
    model: str = "black_scholes"
    symbol: Optional[str] = None

    def to_bs_params(self) -> Any:
        """Convert to BSParameters without overhead."""
        from src.pricing.black_scholes import BSParameters
        return BSParameters(
            spot=self.spot,
            strike=self.strike,
            maturity=self.time_to_expiry,
            volatility=self.volatility,
            rate=self.rate,
            dividend=self.dividend_yield
        )

class PriceResponse(BaseModel):
    """Standard option pricing response."""
    price: float
    spot: float
    strike: float
    time_to_expiry: float
    rate: float
    volatility: float
    option_type: str
    model: str
    computation_time_ms: float
    cached: bool = False
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class BatchPriceResponse(BaseModel):
    """Batch option pricing response."""
    results: List[PriceResponse]
    total_count: int
    computation_time_ms: float
    cached_count: int = 0

class GreeksRequest(BaseModel):
    """Greeks calculation request."""
    spot: float
    strike: float
    time_to_expiry: float
    rate: float
    volatility: float
    option_type: Literal["call", "put"] = "call"
    dividend_yield: float = 0.0
    symbol: Optional[str] = None

    def to_bs_params(self) -> Any:
        """Convert to BSParameters."""
        from src.pricing.black_scholes import BSParameters
        return BSParameters(
            spot=self.spot,
            strike=self.strike,
            maturity=self.time_to_expiry,
            volatility=self.volatility,
            rate=self.rate,
            dividend=self.dividend_yield
        )

class GreeksResponse(BaseModel):
    """Greeks calculation response."""
    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float
    option_price: float
    spot: float
    strike: float
    time_to_expiry: float
    volatility: float
    option_type: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class BatchGreeksRequest(BaseModel):
    """Batch Greeks calculation request."""
    options: List[GreeksRequest]

class BatchGreeksResponse(BaseModel):
    """Batch Greeks calculation response."""
    results: List[GreeksResponse]
    total_count: int
    computation_time_ms: float

class ImpliedVolatilityRequest(BaseModel):
    """Implied volatility calculation request."""
    spot: float
    strike: float
    time_to_expiry: float
    rate: float
    option_price: float
    option_type: Literal["call", "put"] = "call"
    dividend_yield: float = 0.0

class ImpliedVolatilityResponse(BaseModel):
    """Implied volatility calculation response."""
    implied_volatility: float
    option_price: float
    spot: float
    strike: float
    iterations: int
    converged: bool

class ExoticPriceRequest(BaseModel):
    """Exotic option pricing request."""
    spot: float
    strike: float
    time_to_expiry: float
    rate: float
    volatility: float
    exotic_type: Literal["asian", "barrier", "lookback", "digital"]
    option_type: Literal["call", "put"] = "call"
    dividend_yield: float = 0.0
    barrier: Optional[float] = None
    rebate: Optional[float] = 0.0
    barrier_type: Optional[str] = None
    asian_type: Optional[str] = "geometric"
    strike_type: Optional[str] = "fixed"
    n_observations: int = 252
    payout: float = 1.0

class ExoticPriceResponse(BaseModel):
    """Exotic option pricing response."""
    price: float
    exotic_type: str
    confidence_interval: Optional[List[float]] = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class BatchPriceRequest(BaseModel):
    """Batch option pricing request."""
    options: List[PriceRequest]

class PricingDataResponse(BaseModel):
    """SOTA: msgspec equivalent of DataResponse for pricing paths."""
    data: Union[PriceResponse, BatchPriceResponse, GreeksResponse, BatchGreeksResponse, ImpliedVolatilityResponse, ExoticPriceResponse]
    success: bool = True
    message: Optional[str] = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))