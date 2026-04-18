from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class TickerUpdate(_message.Message):
    __slots__ = ("symbol", "price")
    SYMBOL_FIELD_NUMBER: _ClassVar[int]
    PRICE_FIELD_NUMBER: _ClassVar[int]
    symbol: str
    price: float
    def __init__(self, symbol: _Optional[str] = ..., price: _Optional[float] = ...) -> None: ...

class Tick(_message.Message):
    __slots__ = ("ticker", "price", "timestamp", "source")
    TICKER_FIELD_NUMBER: _ClassVar[int]
    PRICE_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    SOURCE_FIELD_NUMBER: _ClassVar[int]
    ticker: str
    price: float
    timestamp: int
    source: str
    def __init__(self, ticker: _Optional[str] = ..., price: _Optional[float] = ..., timestamp: _Optional[int] = ..., source: _Optional[str] = ...) -> None: ...

class TickBatch(_message.Message):
    __slots__ = ("ticks",)
    TICKS_FIELD_NUMBER: _ClassVar[int]
    ticks: _containers.RepeatedCompositeFieldContainer[Tick]
    def __init__(self, ticks: _Optional[_Iterable[_Union[Tick, _Mapping]]] = ...) -> None: ...

class IngestRequest(_message.Message):
    __slots__ = ("ticks",)
    TICKS_FIELD_NUMBER: _ClassVar[int]
    ticks: _containers.RepeatedCompositeFieldContainer[Tick]
    def __init__(self, ticks: _Optional[_Iterable[_Union[Tick, _Mapping]]] = ...) -> None: ...

class IngestResponse(_message.Message):
    __slots__ = ("processed_count",)
    PROCESSED_COUNT_FIELD_NUMBER: _ClassVar[int]
    processed_count: int
    def __init__(self, processed_count: _Optional[int] = ...) -> None: ...

class HistoryRequest(_message.Message):
    __slots__ = ("ticker", "start_time", "end_time")
    TICKER_FIELD_NUMBER: _ClassVar[int]
    START_TIME_FIELD_NUMBER: _ClassVar[int]
    END_TIME_FIELD_NUMBER: _ClassVar[int]
    ticker: str
    start_time: int
    end_time: int
    def __init__(self, ticker: _Optional[str] = ..., start_time: _Optional[int] = ..., end_time: _Optional[int] = ...) -> None: ...

class HistoryResponse(_message.Message):
    __slots__ = ("data",)
    DATA_FIELD_NUMBER: _ClassVar[int]
    data: _containers.RepeatedCompositeFieldContainer[Tick]
    def __init__(self, data: _Optional[_Iterable[_Union[Tick, _Mapping]]] = ...) -> None: ...
