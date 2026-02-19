from typing import ClassVar as _ClassVar

from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message

DESCRIPTOR: _descriptor.FileDescriptor

class InferenceRequest(_message.Message):
    __slots__ = (
        "underlying_price",
        "strike",
        "time_to_expiry",
        "is_call",
        "moneyness",
        "log_moneyness",
        "sqrt_time_to_expiry",
        "days_to_expiry",
        "implied_volatility",
        "model_type",
    )
    UNDERLYING_PRICE_FIELD_NUMBER: _ClassVar[int]
    STRIKE_FIELD_NUMBER: _ClassVar[int]
    TIME_TO_EXPIRY_FIELD_NUMBER: _ClassVar[int]
    IS_CALL_FIELD_NUMBER: _ClassVar[int]
    MONEYNESS_FIELD_NUMBER: _ClassVar[int]
    LOG_MONEYNESS_FIELD_NUMBER: _ClassVar[int]
    SQRT_TIME_TO_EXPIRY_FIELD_NUMBER: _ClassVar[int]
    DAYS_TO_EXPIRY_FIELD_NUMBER: _ClassVar[int]
    IMPLIED_VOLATILITY_FIELD_NUMBER: _ClassVar[int]
    MODEL_TYPE_FIELD_NUMBER: _ClassVar[int]
    underlying_price: float
    strike: float
    time_to_expiry: float
    is_call: int
    moneyness: float
    log_moneyness: float
    sqrt_time_to_expiry: float
    days_to_expiry: float
    implied_volatility: float
    model_type: str
    def __init__(
        self,
        underlying_price: float | None = ...,
        strike: float | None = ...,
        time_to_expiry: float | None = ...,
        is_call: int | None = ...,
        moneyness: float | None = ...,
        log_moneyness: float | None = ...,
        sqrt_time_to_expiry: float | None = ...,
        days_to_expiry: float | None = ...,
        implied_volatility: float | None = ...,
        model_type: str | None = ...,
    ) -> None: ...

class InferenceResponse(_message.Message):
    __slots__ = ("price", "model_type", "latency_ms")
    PRICE_FIELD_NUMBER: _ClassVar[int]
    MODEL_TYPE_FIELD_NUMBER: _ClassVar[int]
    LATENCY_MS_FIELD_NUMBER: _ClassVar[int]
    price: float
    model_type: str
    latency_ms: float
    def __init__(
        self,
        price: float | None = ...,
        model_type: str | None = ...,
        latency_ms: float | None = ...,
    ) -> None: ...
