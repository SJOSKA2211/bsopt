from collections.abc import Iterable as _Iterable
from collections.abc import Mapping as _Mapping
from typing import ClassVar as _ClassVar

from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf.internal import containers as _containers

DESCRIPTOR: _descriptor.FileDescriptor

class PriceRequest(_message.Message):
    __slots__ = (
        "dividend_yield",
        "model",
        "option_type",
        "rate",
        "spot",
        "strike",
        "symbol",
        "time_to_expiry",
        "volatility",
    )
    SPOT_FIELD_NUMBER: _ClassVar[int]
    STRIKE_FIELD_NUMBER: _ClassVar[int]
    TIME_TO_EXPIRY_FIELD_NUMBER: _ClassVar[int]
    VOLATILITY_FIELD_NUMBER: _ClassVar[int]
    RATE_FIELD_NUMBER: _ClassVar[int]
    OPTION_TYPE_FIELD_NUMBER: _ClassVar[int]
    DIVIDEND_YIELD_FIELD_NUMBER: _ClassVar[int]
    MODEL_FIELD_NUMBER: _ClassVar[int]
    SYMBOL_FIELD_NUMBER: _ClassVar[int]
    spot: float
    strike: float
    time_to_expiry: float
    volatility: float
    rate: float
    option_type: str
    dividend_yield: float
    model: str
    symbol: str
    def __init__(
        self,
        spot: float | None = ...,
        strike: float | None = ...,
        time_to_expiry: float | None = ...,
        volatility: float | None = ...,
        rate: float | None = ...,
        option_type: str | None = ...,
        dividend_yield: float | None = ...,
        model: str | None = ...,
        symbol: str | None = ...,
    ) -> None: ...

class PriceResponse(_message.Message):
    __slots__ = (
        "computation_time_ms",
        "model",
        "option_type",
        "price",
        "rate",
        "spot",
        "strike",
        "time_to_expiry",
        "volatility",
    )
    PRICE_FIELD_NUMBER: _ClassVar[int]
    SPOT_FIELD_NUMBER: _ClassVar[int]
    STRIKE_FIELD_NUMBER: _ClassVar[int]
    TIME_TO_EXPIRY_FIELD_NUMBER: _ClassVar[int]
    RATE_FIELD_NUMBER: _ClassVar[int]
    VOLATILITY_FIELD_NUMBER: _ClassVar[int]
    OPTION_TYPE_FIELD_NUMBER: _ClassVar[int]
    MODEL_FIELD_NUMBER: _ClassVar[int]
    COMPUTATION_TIME_MS_FIELD_NUMBER: _ClassVar[int]
    price: float
    spot: float
    strike: float
    time_to_expiry: float
    rate: float
    volatility: float
    option_type: str
    model: str
    computation_time_ms: float
    def __init__(
        self,
        price: float | None = ...,
        spot: float | None = ...,
        strike: float | None = ...,
        time_to_expiry: float | None = ...,
        rate: float | None = ...,
        volatility: float | None = ...,
        option_type: str | None = ...,
        model: str | None = ...,
        computation_time_ms: float | None = ...,
    ) -> None: ...

class BatchPriceRequest(_message.Message):
    __slots__ = ("options",)
    OPTIONS_FIELD_NUMBER: _ClassVar[int]
    options: _containers.RepeatedCompositeFieldContainer[PriceRequest]
    def __init__(self, options: _Iterable[PriceRequest | _Mapping] | None = ...) -> None: ...

class BatchPriceResponse(_message.Message):
    __slots__ = ("computation_time_ms", "results", "total_count")
    RESULTS_FIELD_NUMBER: _ClassVar[int]
    TOTAL_COUNT_FIELD_NUMBER: _ClassVar[int]
    COMPUTATION_TIME_MS_FIELD_NUMBER: _ClassVar[int]
    results: _containers.RepeatedCompositeFieldContainer[PriceResponse]
    total_count: int
    computation_time_ms: float
    def __init__(
        self,
        results: _Iterable[PriceResponse | _Mapping] | None = ...,
        total_count: int | None = ...,
        computation_time_ms: float | None = ...,
    ) -> None: ...

class GreeksRequest(_message.Message):
    __slots__ = (
        "dividend_yield",
        "option_type",
        "rate",
        "spot",
        "strike",
        "time_to_expiry",
        "volatility",
    )
    SPOT_FIELD_NUMBER: _ClassVar[int]
    STRIKE_FIELD_NUMBER: _ClassVar[int]
    TIME_TO_EXPIRY_FIELD_NUMBER: _ClassVar[int]
    RATE_FIELD_NUMBER: _ClassVar[int]
    VOLATILITY_FIELD_NUMBER: _ClassVar[int]
    OPTION_TYPE_FIELD_NUMBER: _ClassVar[int]
    DIVIDEND_YIELD_FIELD_NUMBER: _ClassVar[int]
    spot: float
    strike: float
    time_to_expiry: float
    rate: float
    volatility: float
    option_type: str
    dividend_yield: float
    def __init__(
        self,
        spot: float | None = ...,
        strike: float | None = ...,
        time_to_expiry: float | None = ...,
        rate: float | None = ...,
        volatility: float | None = ...,
        option_type: str | None = ...,
        dividend_yield: float | None = ...,
    ) -> None: ...

class GreeksResponse(_message.Message):
    __slots__ = ("delta", "gamma", "rho", "theta", "vega")
    DELTA_FIELD_NUMBER: _ClassVar[int]
    GAMMA_FIELD_NUMBER: _ClassVar[int]
    THETA_FIELD_NUMBER: _ClassVar[int]
    VEGA_FIELD_NUMBER: _ClassVar[int]
    RHO_FIELD_NUMBER: _ClassVar[int]
    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float
    def __init__(
        self,
        delta: float | None = ...,
        gamma: float | None = ...,
        theta: float | None = ...,
        vega: float | None = ...,
        rho: float | None = ...,
    ) -> None: ...
