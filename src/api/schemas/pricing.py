"""
Pricing Schemas (Optimized)

High-performance schemas using msgspec for ultra-low latency serialization.
Hybrid approach: Pydantic for strict input validation, msgspec for near-instant responses.
"""

from datetime import UTC, datetime
from typing import Any, Literal

import msgspec
from pydantic import BaseModel


class OptionGreeksStruct(msgspec.Struct):
    """Zero-copy greeks structure."""

    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float


class PriceResult(msgspec.Struct):
    """Ultra-fast response for single price result."""

    price: float
    spot: float
    strike: float
    time_to_expiry: float
    rate: float
    volatility: float
    option_type: str
    model: str
    computation_time_ms: float
    greeks: OptionGreeksStruct | None = None
    cached: bool = False
    timestamp: datetime = msgspec.field(default_factory=lambda: datetime.now(UTC))


class GreeksResult(msgspec.Struct):
    """Ultra-fast response for single greeks result."""

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
    timestamp: datetime = msgspec.field(default_factory=lambda: datetime.now(UTC))


class BatchPriceResult(msgspec.Struct):
    """Ultra-fast response for batch results."""

    results: list[PriceResult]
    total_count: int
    computation_time_ms: float
    cached_count: int = 0


class BatchGreeksResult(msgspec.Struct):
    """Ultra-fast response for batch greeks results."""

    results: list[GreeksResult]
    total_count: int
    computation_time_ms: float


class PriceRequest(BaseModel):
    """
    Standard option pricing request (Pydantic for Validation).
    """

    spot: float
    strike: float
    time_to_expiry: float
    volatility: float
    rate: float
    option_type: Literal["call", "put"] = "call"
    dividend_yield: float = 0.0
    model: str = "black_scholes"
    symbol: str | None = None

    def to_bs_params(self) -> Any:
        """Convert to BSParameters without overhead."""
        from src.pricing.black_scholes import BSParameters

        return BSParameters(
            spot=self.spot,
            strike=self.strike,
            maturity=self.time_to_expiry,
            volatility=self.volatility,
            rate=self.rate,
            dividend=self.dividend_yield,
        )


# Aliases for backward compatibility
PriceResponse = PriceResult
BatchPriceResponse = BatchPriceResult
GreeksResponse = GreeksResult
BatchGreeksResponse = BatchGreeksResult


class GreeksRequest(BaseModel):
    """Greeks calculation request (Pydantic for Validation)."""

    spot: float
    strike: float
    time_to_expiry: float
    rate: float
    volatility: float
    option_type: Literal["call", "put"] = "call"
    dividend_yield: float = 0.0
    symbol: str | None = None

    def to_bs_params(self) -> Any:
        """Convert to BSParameters."""
        from src.pricing.black_scholes import BSParameters

        return BSParameters(
            spot=self.spot,
            strike=self.strike,
            maturity=self.time_to_expiry,
            volatility=self.volatility,
            rate=self.rate,
            dividend=self.dividend_yield,
        )


class BatchGreeksRequest(BaseModel):
    """Batch Greeks calculation request."""

    options: list[GreeksRequest]


class ImpliedVolatilityRequest(BaseModel):
    """Implied volatility calculation request."""

    spot: float
    strike: float
    time_to_expiry: float
    rate: float
    option_price: float
    option_type: Literal["call", "put"] = "call"
    dividend_yield: float = 0.0


class ImpliedVolatilityResponse(msgspec.Struct):
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
    barrier: float | None = None
    rebate: float | None = 0.0
    barrier_type: str | None = None
    asian_type: str | None = "geometric"
    strike_type: str | None = "fixed"
    n_observations: int = 252
    payout: float = 1.0


class ExoticPriceResponse(msgspec.Struct):
    """Exotic option pricing response."""

    price: float
    exotic_type: str
    confidence_interval: list[float] | None = None
    timestamp: datetime = msgspec.field(default_factory=lambda: datetime.now(UTC))


class BatchPriceRequest(BaseModel):
    """Batch option pricing request."""

    options: list[PriceRequest]


class PricingDataResponse(msgspec.Struct):
    """OPTIMIZED: msgspec equivalent of DataResponse for pricing paths."""

    data: Any
    success: bool = True
    message: str | None = None
    timestamp: datetime = msgspec.field(default_factory=lambda: datetime.now(UTC))
