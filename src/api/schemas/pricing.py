"""
Pricing Schemas
===============

Pydantic models for options pricing endpoints.
"""

from datetime import datetime, timezone
from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class PriceRequest(BaseModel):
    """Option pricing request."""

    spot: float = Field(..., gt=0, description="Current spot price of underlying")
    strike: float = Field(..., gt=0, description="Option strike price")
    time_to_expiry: float = Field(..., gt=0, le=10, description="Time to expiry in years")
    rate: float = Field(..., ge=0, le=1, description="Risk-free interest rate (decimal)")
    volatility: float = Field(
        ..., gt=0, le=5, description="Volatility (decimal, e.g., 0.2 for 20%)"
    )
    option_type: Literal["call", "put"] = Field("call", description="Option type")
    dividend_yield: float = Field(0, ge=0, le=1, description="Dividend yield (decimal)")
    model: Literal["black_scholes", "monte_carlo", "binomial", "heston"] = Field(
        "black_scholes", description="Pricing model to use"
    )
    symbol: Optional[str] = Field(None, description="Underlying symbol (required for Heston model)")

    @field_validator("time_to_expiry")
    @classmethod
    def validate_time(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("Time to expiry must be positive")
        return v

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "spot": 100.0,
                "strike": 105.0,
                "time_to_expiry": 0.5,
                "rate": 0.05,
                "volatility": 0.2,
                "option_type": "call",
                "dividend_yield": 0.02,
                "model": "black_scholes",
            }
        }
    )


class PriceResponse(BaseModel):
    """Option pricing response."""

    price: float = Field(..., description="Calculated option price")
    spot: float = Field(..., description="Spot price used")
    strike: float = Field(..., description="Strike price")
    time_to_expiry: float = Field(..., description="Time to expiry in years")
    rate: float = Field(..., description="Risk-free rate used")
    volatility: float = Field(..., description="Volatility used")
    option_type: str = Field(..., description="Option type")
    model: str = Field(..., description="Pricing model used")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    cached: bool = Field(False, description="Whether result was cached")
    computation_time_ms: Optional[float] = Field(
        None, description="Computation time in milliseconds"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "price": 5.67,
                "spot": 100.0,
                "strike": 105.0,
                "time_to_expiry": 0.5,
                "rate": 0.05,
                "volatility": 0.2,
                "option_type": "call",
                "model": "black_scholes",
                "timestamp": "2024-01-15T10:30:00Z",
                "cached": False,
                "computation_time_ms": 1.5,
            }
        }
    )


class BatchPriceRequest(BaseModel):
    """Batch option pricing request."""

    options: List[PriceRequest] = Field(
        ..., min_length=1, max_length=1000, description="List of options to price"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "options": [
                    {
                        "spot": 100.0,
                        "strike": 105.0,
                        "time_to_expiry": 0.5,
                        "rate": 0.05,
                        "volatility": 0.2,
                        "option_type": "call",
                    },
                    {
                        "spot": 100.0,
                        "strike": 95.0,
                        "time_to_expiry": 0.5,
                        "rate": 0.05,
                        "volatility": 0.2,
                        "option_type": "put",
                    },
                ]
            }
        }
    )


class BatchPriceResponse(BaseModel):
    """Batch option pricing response."""

    results: List[PriceResponse] = Field(..., description="Pricing results")
    total_count: int = Field(..., description="Total options priced")
    computation_time_ms: float = Field(..., description="Total computation time")
    cached_count: int = Field(0, description="Number of cached results")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "results": [
                    {"price": 5.67, "spot": 100.0, "strike": 105.0, "option_type": "call"},
                    {"price": 3.21, "spot": 100.0, "strike": 95.0, "option_type": "put"},
                ],
                "total_count": 2,
                "computation_time_ms": 2.5,
                "cached_count": 0,
            }
        }
    )


class GreeksRequest(BaseModel):
    """Greeks calculation request."""

    spot: float = Field(..., gt=0, description="Current spot price")
    strike: float = Field(..., gt=0, description="Strike price")
    time_to_expiry: float = Field(..., gt=0, le=10, description="Time to expiry in years")
    rate: float = Field(..., ge=0, le=1, description="Risk-free rate")
    volatility: float = Field(..., gt=0, le=5, description="Volatility")
    option_type: Literal["call", "put"] = Field("call", description="Option type")
    dividend_yield: float = Field(0, ge=0, le=1, description="Dividend yield")
    symbol: Optional[str] = Field(None, description="Underlying symbol")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "spot": 100.0,
                "strike": 105.0,
                "time_to_expiry": 0.5,
                "rate": 0.05,
                "volatility": 0.2,
                "option_type": "call",
                "dividend_yield": 0.02,
            }
        }
    )


class GreeksResponse(BaseModel):
    """Greeks calculation response."""

    delta: float = Field(..., description="Delta - price sensitivity to spot")
    gamma: float = Field(..., description="Gamma - delta sensitivity to spot")
    theta: float = Field(..., description="Theta - time decay (per day)")
    vega: float = Field(..., description="Vega - sensitivity to volatility (per 1%)")
    rho: float = Field(..., description="Rho - sensitivity to interest rate (per 1%)")
    option_price: float = Field(..., description="Option price")
    spot: float = Field(..., description="Spot price used")
    strike: float = Field(..., description="Strike price")
    time_to_expiry: float = Field(..., description="Time to expiry")
    volatility: float = Field(..., description="Volatility used")
    option_type: str = Field(..., description="Option type")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "delta": 0.5234,
                "gamma": 0.0421,
                "theta": -0.0156,
                "vega": 0.1823,
                "rho": 0.0234,
                "option_price": 5.67,
                "spot": 100.0,
                "strike": 105.0,
                "time_to_expiry": 0.5,
                "volatility": 0.2,
                "option_type": "call",
                "timestamp": "2024-01-15T10:30:00Z",
            }
        }
    )


class ImpliedVolatilityRequest(BaseModel):
    """Implied volatility calculation request."""

    spot: float = Field(..., gt=0, description="Current spot price")
    strike: float = Field(..., gt=0, description="Strike price")
    time_to_expiry: float = Field(..., gt=0, description="Time to expiry in years")
    rate: float = Field(..., ge=0, le=1, description="Risk-free rate")
    option_price: float = Field(..., gt=0, description="Market option price")
    option_type: Literal["call", "put"] = Field("call", description="Option type")
    dividend_yield: float = Field(0, ge=0, description="Dividend yield")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "spot": 100.0,
                "strike": 105.0,
                "time_to_expiry": 0.5,
                "rate": 0.05,
                "option_price": 5.67,
                "option_type": "call",
            }
        }
    )


class ImpliedVolatilityResponse(BaseModel):
    """Implied volatility calculation response."""

    implied_volatility: float = Field(..., description="Calculated implied volatility")
    option_price: float = Field(..., description="Option price used")
    spot: float = Field(..., description="Spot price")
    strike: float = Field(..., description="Strike price")
    iterations: int = Field(..., description="Newton-Raphson iterations used")
    converged: bool = Field(..., description="Whether calculation converged")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "implied_volatility": 0.2034,
                "option_price": 5.67,
                "spot": 100.0,
                "strike": 105.0,
                "iterations": 5,
                "converged": True,
            }
        }
    )


class ExoticPriceRequest(BaseModel):
    """Exotic option pricing request."""

    spot: float = Field(..., gt=0)
    strike: float = Field(..., gt=0)
    time_to_expiry: float = Field(..., gt=0)
    rate: float = Field(..., ge=0)
    volatility: float = Field(..., gt=0)
    option_type: Literal["call", "put"] = "call"
    dividend_yield: float = 0
    exotic_type: Literal["asian", "barrier", "lookback", "digital"]
    
    # Specific params for different exotics
    barrier: Optional[float] = None
    rebate: Optional[float] = 0.0
    barrier_type: Optional[str] = None # down-and-out, etc.
    asian_type: Optional[str] = "geometric"
    strike_type: Optional[str] = "fixed" # fixed or floating for lookback/asian
    n_observations: int = 252
    payout: float = 1.0 # for digital

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "spot": 100.0,
                "strike": 100.0,
                "time_to_expiry": 1.0,
                "rate": 0.05,
                "volatility": 0.2,
                "exotic_type": "barrier",
                "barrier": 90.0,
                "barrier_type": "down-and-out"
            }
        }
    )


class ExoticPriceResponse(BaseModel):
    """Exotic option pricing response."""

    price: float
    confidence_interval: Optional[List[float]] = None
    exotic_type: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))