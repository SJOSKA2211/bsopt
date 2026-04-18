import common_pb2 as _common_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class PriceRequest(_message.Message):
    __slots__ = ("spot", "strike", "time_to_expiry", "volatility", "rate", "option_type", "dividend_yield", "model", "symbol")
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
    def __init__(self, spot: _Optional[float] = ..., strike: _Optional[float] = ..., time_to_expiry: _Optional[float] = ..., volatility: _Optional[float] = ..., rate: _Optional[float] = ..., option_type: _Optional[str] = ..., dividend_yield: _Optional[float] = ..., model: _Optional[str] = ..., symbol: _Optional[str] = ...) -> None: ...

class PriceResponse(_message.Message):
    __slots__ = ("price", "spot", "strike", "time_to_expiry", "rate", "volatility", "option_type", "model", "computation_time_ms")
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
    def __init__(self, price: _Optional[float] = ..., spot: _Optional[float] = ..., strike: _Optional[float] = ..., time_to_expiry: _Optional[float] = ..., rate: _Optional[float] = ..., volatility: _Optional[float] = ..., option_type: _Optional[str] = ..., model: _Optional[str] = ..., computation_time_ms: _Optional[float] = ...) -> None: ...

class BatchPriceRequest(_message.Message):
    __slots__ = ("options",)
    OPTIONS_FIELD_NUMBER: _ClassVar[int]
    options: _containers.RepeatedCompositeFieldContainer[PriceRequest]
    def __init__(self, options: _Optional[_Iterable[_Union[PriceRequest, _Mapping]]] = ...) -> None: ...

class BatchPriceResponse(_message.Message):
    __slots__ = ("results", "total_count", "computation_time_ms")
    RESULTS_FIELD_NUMBER: _ClassVar[int]
    TOTAL_COUNT_FIELD_NUMBER: _ClassVar[int]
    COMPUTATION_TIME_MS_FIELD_NUMBER: _ClassVar[int]
    results: _containers.RepeatedCompositeFieldContainer[PriceResponse]
    total_count: int
    computation_time_ms: float
    def __init__(self, results: _Optional[_Iterable[_Union[PriceResponse, _Mapping]]] = ..., total_count: _Optional[int] = ..., computation_time_ms: _Optional[float] = ...) -> None: ...

class GreeksRequest(_message.Message):
    __slots__ = ("spot", "strike", "time_to_expiry", "rate", "volatility", "option_type", "dividend_yield")
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
    def __init__(self, spot: _Optional[float] = ..., strike: _Optional[float] = ..., time_to_expiry: _Optional[float] = ..., rate: _Optional[float] = ..., volatility: _Optional[float] = ..., option_type: _Optional[str] = ..., dividend_yield: _Optional[float] = ...) -> None: ...

class GreeksResponse(_message.Message):
    __slots__ = ("delta", "gamma", "theta", "vega", "rho")
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
    def __init__(self, delta: _Optional[float] = ..., gamma: _Optional[float] = ..., theta: _Optional[float] = ..., vega: _Optional[float] = ..., rho: _Optional[float] = ...) -> None: ...
