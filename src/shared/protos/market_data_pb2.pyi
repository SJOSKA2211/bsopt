from collections.abc import Iterable as _Iterable
from collections.abc import Mapping as _Mapping
from typing import ClassVar as _ClassVar

from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf.internal import containers as _containers

DESCRIPTOR: _descriptor.FileDescriptor

class TickerUpdate(_message.Message):
    __slots__ = ("price", "symbol")
    SYMBOL_FIELD_NUMBER: _ClassVar[int]
    PRICE_FIELD_NUMBER: _ClassVar[int]
    symbol: str
    price: float
    def __init__(self, symbol: str | None = ..., price: float | None = ...) -> None: ...

class Tick(_message.Message):
    __slots__ = ("price", "source", "ticker", "timestamp")
    TICKER_FIELD_NUMBER: _ClassVar[int]
    PRICE_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    SOURCE_FIELD_NUMBER: _ClassVar[int]
    ticker: str
    price: float
    timestamp: int
    source: str
    def __init__(self, ticker: str | None = ..., price: float | None = ..., timestamp: int | None = ..., source: str | None = ...) -> None: ...

class TickBatch(_message.Message):
    __slots__ = ("ticks",)
    TICKS_FIELD_NUMBER: _ClassVar[int]
    ticks: _containers.RepeatedCompositeFieldContainer[Tick]
    def __init__(self, ticks: _Iterable[Tick | _Mapping] | None = ...) -> None: ...

class IngestRequest(_message.Message):
    __slots__ = ("ticks",)
    TICKS_FIELD_NUMBER: _ClassVar[int]
    ticks: _containers.RepeatedCompositeFieldContainer[Tick]
    def __init__(self, ticks: _Iterable[Tick | _Mapping] | None = ...) -> None: ...

class IngestResponse(_message.Message):
    __slots__ = ("processed_count",)
    PROCESSED_COUNT_FIELD_NUMBER: _ClassVar[int]
    processed_count: int
    def __init__(self, processed_count: int | None = ...) -> None: ...

class HistoryRequest(_message.Message):
    __slots__ = ("end_time", "start_time", "ticker")
    TICKER_FIELD_NUMBER: _ClassVar[int]
    START_TIME_FIELD_NUMBER: _ClassVar[int]
    END_TIME_FIELD_NUMBER: _ClassVar[int]
    ticker: str
    start_time: int
    end_time: int
    def __init__(self, ticker: str | None = ..., start_time: int | None = ..., end_time: int | None = ...) -> None: ...

class HistoryResponse(_message.Message):
    __slots__ = ("data",)
    DATA_FIELD_NUMBER: _ClassVar[int]
    data: _containers.RepeatedCompositeFieldContainer[Tick]
    def __init__(self, data: _Iterable[Tick | _Mapping] | None = ...) -> None: ...
