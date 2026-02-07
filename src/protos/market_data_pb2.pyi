from typing import ClassVar as _ClassVar

from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message

DESCRIPTOR: _descriptor.FileDescriptor

class TickerUpdate(_message.Message):
    __slots__ = ("symbol", "price")
    SYMBOL_FIELD_NUMBER: _ClassVar[int]
    PRICE_FIELD_NUMBER: _ClassVar[int]
    symbol: str
    price: float
    def __init__(self, symbol: str | None = ..., price: float | None = ...) -> None: ...
