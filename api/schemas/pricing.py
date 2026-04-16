from datetime import UTC, datetime
from typing import Any, Literal

import msgspec
from pydantic import BaseModel


class OptionGreeksStruct(msgspec.Struct):
    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float


class PriceResult(msgspec.Struct):
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
    results: list[PriceResult]
    total_count: int
    computation_time_ms: float
    cached_count: int = 0


class BatchGreeksResult(msgspec.Struct):
    results: list[GreeksResult]
    total_count: int
    computation_time_ms: float


class PriceRequest(BaseModel):
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
        from src.math_kernel.black_scholes import BSParameters

        return BSParameters(
            spot=self.spot,
            strike=self.strike,
            maturity=self.time_to_expiry,
            volatility=self.volatility,
            rate=self.rate,
            dividend=self.dividend_yield,
        )


PriceResponse = PriceResult
BatchPriceResponse = BatchPriceResult
GreeksResponse = GreeksResult
BatchGreeksResponse = BatchGreeksResult


class GreeksRequest(BaseModel):
    spot: float
    strike: float
    time_to_expiry: float
    rate: float
    volatility: float
    option_type: Literal["call", "put"] = "call"
    dividend_yield: float = 0.0
    symbol: str | None = None

    def to_bs_params(self) -> Any:
        from src.math_kernel.black_scholes import BSParameters

        return BSParameters(
            spot=self.spot,
            strike=self.strike,
            maturity=self.time_to_expiry,
            volatility=self.volatility,
            rate=self.rate,
            dividend=self.dividend_yield,
        )


class BatchGreeksRequest(BaseModel):
    options: list[GreeksRequest]


class ImpliedVolatilityRequest(BaseModel):
    spot: float
    strike: float
    time_to_expiry: float
    rate: float
    option_price: float
    option_type: Literal["call", "put"] = "call"
    dividend_yield: float = 0.0


class ImpliedVolatilityResponse(msgspec.Struct):
    implied_volatility: float
    option_price: float
    spot: float
    strike: float
    iterations: int
    converged: bool


class ExoticPriceRequest(BaseModel):
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
    price: float
    exotic_type: str
    confidence_interval: list[float] | None = None
    timestamp: datetime = msgspec.field(default_factory=lambda: datetime.now(UTC))


class BatchPriceRequest(BaseModel):
    options: list[PriceRequest]


class PricingDataResponse(msgspec.Struct):
    data: Any
    success: bool = True
    message: str | None = None
    timestamp: datetime = msgspec.field(default_factory=lambda: datetime.now(UTC))


class HeatmapRequest(BaseModel):
    spot: float
    strike: float
    time_to_expiry: float
    volatility: float
    rate: float
    option_type: Literal["call", "put"] = "call"
    dividend_yield: float = 0.0
    model: str = "black_scholes"
    
    price_shifts: list[float] = [-10, -5, -2, 0, 2, 5, 10]
    vol_shifts: list[float] = [-5, -2, 0, 2, 5]


class HeatmapCell(msgspec.Struct):
    price_shift: float
    vol_shift: float
    pnl: float
    theoretical_price: float


class HeatmapResponse(msgspec.Struct):
    grid: list[list[HeatmapCell]]
    price_steps: list[float]
    vol_steps: list[float]
    computation_time_ms: float
    timestamp: datetime = msgspec.field(default_factory=lambda: datetime.now(UTC))
